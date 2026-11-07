//! Cold-weight decode GEMV bandwidth at the real Qwen3-8B decode shapes. Rotates through DISTINCT
//! weight buffers so the aggregate working set exceeds the 96 MiB Infinity Cache — the reported
//! GiB/s is TRUE DRAM bandwidth (the in-model INFR_PROF_OPS numbers for small tensors are cache-
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
    // `kernels.vulkan.gemv.*` (`INFR_GEMV_*` / `INFR_NO_GEMV_*`) is resolved at backend
    // construction since S5b — the routing knobs are a VALUE, not process env — so each mode is its
    // own backend and the six stay live together (the interleaved best-of-3 below needs that).
    // Each mode fully specifies the group so precedence is explicit (SG > RM > tree in the
    // recorder), exactly as the old `cfg()` env re-set did.
    //
    // The six devices are built INSIDE the shape loop and dropped with it. Six live backends each
    // holding a shape's ~200 MiB cache-busting weight set is 6x the pre-S5b footprint, and holding
    // that across all 18 shapes trips the VRAM guard on a 24 GiB part; per-shape devices bound the
    // peak to one shape (worst case: lm_head, 6 x 3 x 487 MiB).
    let mode_cfg = |mode: &str| -> infr_core::config::GemvCfg {
        let mut g = infr_core::config::GemvCfg {
            rm_maxout: 999_999,
            sg_minout: 0,
            sg_maxout: 999_999,
            ..Default::default()
        };
        match mode {
            "1" => {
                g.no_rm = true;
                g.no_sg = true;
            }
            "R2" | "R4" => {
                g.no_sg = true;
                g.rm = mode[1..].parse().unwrap();
            }
            _ => {
                // SG NR = mode[1..]
                g.no_rm = true;
                g.sg_nr = mode[1..].parse().unwrap();
            }
        }
        g
    };
    let be_for = |mode: &str| -> VulkanBackend {
        let mut cfg = infr_core::config::Config::default();
        cfg.kernels.vulkan.gemv = mode_cfg(mode);
        VulkanBackend::new_with(std::sync::Arc::new(cfg)).unwrap()
    };
    let modes = ["1", "R2", "R4", "S2", "S4", "S8"];
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
        // bandwidth too, not just in the small-model INFR_PROF_OPS numbers (which are cache-warm).
        ("q -iq4x", 4096, 4096, DType::Iq4Xs),
        ("k/v-iq4", 4096, 1024, DType::Iq4Xs),
        ("qkv-iq4", 4096, 6144, DType::Iq4Xs),
        ("o  -iq4", 4096, 4096, DType::Iq4Xs),
        ("g+u-iq4", 4096, 24576, DType::Iq4Xs),
        ("dn -iq4", 12288, 4096, DType::Iq4Xs),
    ];
    let xmax = 12288;
    let xs: Vec<f32> = (0..xmax).map(|i| ((i % 61) as f32 - 30.0) * 0.01).collect();

    for (label, in_f, out_f, dt) in shapes {
        let bes: Vec<VulkanBackend> = modes.iter().map(|m| be_for(m)).collect();
        let x: Vec<_> = bes
            .iter()
            .map(|be| {
                let b = be.alloc(xmax * 4, BufferUsage::Activations).unwrap();
                be.upload(b.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
                b
            })
            .collect();
        let wbytes = in_f * out_f / 256 * blk_bytes(dt);
        let n_w = (200usize << 20).div_ceil(wbytes).clamp(3, 24);
        // The same synthetic weight bytes, uploaded once per mode-backend (a buffer belongs to the
        // device that allocated it) — so the reported bit-parity compare is still against the SAME
        // weights, and the working set per mode is unchanged from the pre-S5b single-backend run.
        let src: Vec<Vec<u8>> = (0..n_w)
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
                src
            })
            .collect();
        let per_mode: Vec<(Vec<_>, _)> = bes
            .iter()
            .map(|be| {
                let ws: Vec<_> = src
                    .iter()
                    .map(|s| {
                        let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
                        be.upload(w.as_ref(), s).unwrap();
                        w
                    })
                    .collect();
                let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
                (ws, y)
            })
            .collect();
        let reps = (n_w * 4).max(40);

        // Mode selects tree(RM1) / RM=2/4 (bit-identical) / SG NR=2/4/8 (reassociated).
        let run_once = |i: usize| -> Vec<f32> {
            let (be, (ws, y)) = (&bes[i], &per_mode[i]);
            let rec = be.recorder().unwrap();
            rec.linear_native(
                dt,
                ws[0].as_ref(),
                0,
                x[i].as_ref(),
                y.as_ref(),
                1,
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            let mut out = vec![0u8; out_f * 4];
            be.download(y.as_ref(), &mut out).unwrap();
            bytemuck::cast_slice::<u8, f32>(&out).to_vec()
        };
        let base = run_once(0);
        // RM stays BIT-identical to the tree kernel.
        for i in [1usize, 2] {
            let got = run_once(i);
            let mism = base
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                mism, 0,
                "{label} {}: {mism} bits differ from tree",
                modes[i]
            );
        }
        // SG is reassociated (not bit-identical) — measure how close it stays to the tree ref.
        let mut sg_maxrel = 0f32;
        for i in [3usize, 4, 5] {
            let got = run_once(i);
            let mut mr = 0f32;
            for (a, b) in base.iter().zip(&got) {
                let d = (a - b).abs() / (a.abs().max(b.abs()) + 1e-6);
                mr = mr.max(d);
            }
            sg_maxrel = sg_maxrel.max(mr);
        }

        // Bandwidth A/B (cold, rotated).
        let bench = |i: usize| -> f64 {
            let (be, (ws, y)) = (&bes[i], &per_mode[i]);
            let run = || {
                let rec = be.recorder().unwrap();
                for r in 0..reps {
                    rec.linear_native(
                        dt,
                        ws[r % n_w].as_ref(),
                        0,
                        x[i].as_ref(),
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
            wbytes as f64 / (us * 1e-6) / (1u64 << 30) as f64
        };
        // Best-of-3, interleaved, to fight the ~25% thermal swing.
        let mut best = [0f64; 6];
        for _ in 0..3 {
            for (i, b) in best.iter_mut().enumerate() {
                *b = b.max(bench(i));
            }
        }
        let [g1, r2, r4, s2, s4, s8] = best;
        let rm_best = r2.max(r4);
        let sg_best = s2.max(s4).max(s8);
        println!(
            "  {label} in{in_f} out{out_f} {dt:?} [{} MiB]:  tree {g1:5.0}  RM2 {r2:5.0} RM4 {r4:5.0}  |  SG2 {s2:5.0} SG4 {s4:5.0} SG8 {s8:5.0} GiB/s  (SGbest vs tree {:+.0}%, vs RMbest {:+.0}%, maxrel {sg_maxrel:.1e})",
            wbytes >> 20,
            (sg_best / g1 - 1.0) * 100.0,
            (sg_best / rm_best - 1.0) * 100.0,
        );
    }
}
