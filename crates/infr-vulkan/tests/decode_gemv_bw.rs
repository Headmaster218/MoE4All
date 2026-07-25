//! Cold-weight decode GEMV bandwidth at the real Qwen3-8B decode shapes. Rotates through DISTINCT
//! weight buffers so the aggregate working set exceeds the 96 MiB Infinity Cache — the reported
//! GB/s is TRUE DRAM bandwidth (the in-model INFR_PROF2 numbers for small tensors are cache-
//! contaminated). A/Bs the RM=1 grid vs the multi-output-row (RM=2/4) grid, and asserts the RM
//! path is BIT-IDENTICAL to RM=1 (per-row math is unchanged). Run:
//!   cargo test -p infr-vulkan --test decode_gemv_bw -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

/// Bytes per block, from the shared decode spec (`infr_core::decode_spec` via
/// `infr_gguf::block_layout`) — not a hand-kept local table.
fn blk_bytes(dt: DType) -> usize {
    infr_gguf::block_layout(dt).1
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn decode_gemv_bw() {
    let be = VulkanBackend::new().unwrap();
    // (label, in_f, out_f, dtype) — the Qwen3-8B decode GEMV shapes.
    let shapes = [
        ("q      ", 4096usize, 4096usize, DType::Q4K),
        ("k/v    ", 4096, 1024, DType::Q4K),
        ("qkv-fus", 4096, 6144, DType::Q4K),
        ("o      ", 4096, 4096, DType::Q4K),
        ("gate+up", 4096, 24576, DType::Q4K),
        ("down-q4", 12288, 4096, DType::Q4K),
        ("down-q6", 12288, 4096, DType::Q6K),
        ("v-q6   ", 4096, 1024, DType::Q6K),
        ("v2k-q6 ", 2816, 2048, DType::Q6K),
        ("o2k-q6 ", 4096, 2048, DType::Q6K),
        ("o-q6   ", 4096, 4096, DType::Q6K),
        ("lm_head", 4096, 151936, DType::Q6K),
        // IQ4_XS: same shapes as the Q4_K rows above — codebook-decode ALU-bound-ness (fewer
        // bytes/weight than Q4_K, yet slower) must hold at TRUE (>96 MiB, cache-busting) DRAM
        // bandwidth too, not just in the small-model INFR_PROF2 numbers (which are cache-warm).
        ("q -iq4x", 4096, 4096, DType::Iq4Xs),
        ("k/v-iq4", 4096, 1024, DType::Iq4Xs),
        ("qkv-iq4", 4096, 6144, DType::Iq4Xs),
        ("o  -iq4", 4096, 4096, DType::Iq4Xs),
        ("g+u-iq4", 4096, 24576, DType::Iq4Xs),
        ("dn -iq4", 12288, 4096, DType::Iq4Xs),
    ];
    let xmax = 12288;
    let xs: Vec<f32> = (0..xmax).map(|i| ((i % 61) as f32 - 30.0) * 0.01).collect();
    let x = be.alloc(xmax * 4, BufferUsage::Activations).unwrap();
    be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();

    let read = |b: &dyn infr_core::backend::Buffer, n: usize| -> Vec<f32> {
        let mut out = vec![0u8; n * 4];
        be.download(b, &mut out).unwrap();
        bytemuck::cast_slice::<u8, f32>(&out).to_vec()
    };

    for (label, in_f, out_f, dt) in shapes {
        let wbytes = in_f * out_f / 256 * blk_bytes(dt);
        let n_w = (200usize << 20).div_ceil(wbytes).clamp(3, 24);
        let ws: Vec<_> = (0..n_w)
            .map(|s| {
                let mut src: Vec<u8> = (0..wbytes)
                    .map(|i| {
                        ((i as u32).wrapping_mul(2654435761).wrapping_add(s as u32) >> 24) as u8
                    })
                    .collect();
                for blk in src.chunks_exact_mut(blk_bytes(dt)) {
                    match dt {
                        DType::Q4K => {
                            blk[0..2].copy_from_slice(&[0x00, 0x1C]);
                            blk[2..4].copy_from_slice(&[0x00, 0x1C]);
                        }
                        DType::Q6K => blk[208..210].copy_from_slice(&[0x00, 0x1C]),
                        DType::Iq4Xs => blk[0..2].copy_from_slice(&[0x00, 0x1C]),
                        _ => unreachable!(),
                    }
                }
                let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
                be.upload(w.as_ref(), &src).unwrap();
                w
            })
            .collect();
        let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        let reps = (n_w * 4).max(40);

        // Mode selects tree(RM1) / RM=2/4 (bit-identical) / SG NR=2/4/8 (reassociated). Each mode
        // fully re-sets the env so precedence is explicit (SG > RM > tree in the recorder).
        let cfg = |mode: &str| {
            std::env::set_var("INFR_GEMV_RM_MAXOUT", "999999");
            std::env::set_var("INFR_GEMV_SG_MINOUT", "0");
            std::env::set_var("INFR_GEMV_SG_MAXOUT", "999999");
            match mode {
                "1" => {
                    std::env::set_var("INFR_NO_GEMV_RM", "1");
                    std::env::set_var("INFR_NO_GEMV_SG", "1");
                }
                "R2" | "R4" => {
                    std::env::remove_var("INFR_NO_GEMV_RM");
                    std::env::set_var("INFR_NO_GEMV_SG", "1");
                    std::env::set_var("INFR_GEMV_RM", &mode[1..]);
                }
                _ => {
                    // SG NR = mode[1..]
                    std::env::set_var("INFR_NO_GEMV_RM", "1");
                    std::env::remove_var("INFR_NO_GEMV_SG");
                    std::env::set_var("INFR_GEMV_SG_NR", &mode[1..]);
                }
            }
        };
        let run_once = |mode: &str| -> Vec<f32> {
            cfg(mode);
            let rec = be.recorder().unwrap();
            rec.linear_native(
                dt,
                ws[0].as_ref(),
                0,
                x.as_ref(),
                y.as_ref(),
                1,
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            read(y.as_ref(), out_f)
        };
        let base = run_once("1");
        // RM stays BIT-identical to the tree kernel.
        for m in ["R2", "R4"] {
            let got = run_once(m);
            let mism = base
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(mism, 0, "{label} {m}: {mism} bits differ from tree");
        }
        // SG is reassociated (not bit-identical) — measure how close it stays to the tree ref.
        let mut sg_maxrel = 0f32;
        for m in ["S2", "S4", "S8"] {
            let got = run_once(m);
            let mut mr = 0f32;
            for (a, b) in base.iter().zip(&got) {
                let d = (a - b).abs() / (a.abs().max(b.abs()) + 1e-6);
                mr = mr.max(d);
            }
            sg_maxrel = sg_maxrel.max(mr);
        }

        // Bandwidth A/B (cold, rotated).
        let bench = |mode: &str| -> f64 {
            cfg(mode);
            let run = || {
                let rec = be.recorder().unwrap();
                for r in 0..reps {
                    rec.linear_native(
                        dt,
                        ws[r % n_w].as_ref(),
                        0,
                        x.as_ref(),
                        y.as_ref(),
                        1,
                        in_f,
                        out_f,
                    );
                }
                rec.finish().unwrap();
            };
            run();
            let t = std::time::Instant::now();
            run();
            let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            wbytes as f64 / (us * 1e-6) / 1e9
        };
        // Best-of-3, interleaved, to fight the ~25% thermal swing.
        let modes = ["1", "R2", "R4", "S2", "S4", "S8"];
        let mut best = [0f64; 6];
        for _ in 0..3 {
            for (i, m) in modes.iter().enumerate() {
                best[i] = best[i].max(bench(m));
            }
        }
        let [g1, r2, r4, s2, s4, s8] = best;
        let rm_best = r2.max(r4);
        let sg_best = s2.max(s4).max(s8);
        println!(
            "  {label} in{in_f} out{out_f} {dt:?} [{} MiB]:  tree {g1:5.0}  RM2 {r2:5.0} RM4 {r4:5.0}  |  SG2 {s2:5.0} SG4 {s4:5.0} SG8 {s8:5.0} GB/s  (SGbest vs tree {:+.0}%, vs RMbest {:+.0}%, maxrel {sg_maxrel:.1e})",
            wbytes >> 20,
            (sg_best / g1 - 1.0) * 100.0,
            (sg_best / rm_best - 1.0) * 100.0,
        );
    }
    for v in [
        "INFR_NO_GEMV_RM",
        "INFR_GEMV_RM",
        "INFR_GEMV_RM_MAXOUT",
        "INFR_NO_GEMV_SG",
        "INFR_GEMV_SG_NR",
        "INFR_GEMV_SG_MINOUT",
        "INFR_GEMV_SG_MAXOUT",
    ] {
        std::env::remove_var(v);
    }
}
