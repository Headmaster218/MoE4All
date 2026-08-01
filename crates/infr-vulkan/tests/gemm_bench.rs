//! Serialized micro-bench of the Q4_K prefill GEMM variants at qwen35-in-proj shape
//! ([512,1024]×[1024,6144]): mmq (dp4a int8), native 64×64 coopmat, native 8-warp warptile.
//! WAW hazards on the shared output buffer serialize the repeated dispatches, so wall/REPS is the
//! per-GEMM cost. Weights are zeros (perf only). Run: cargo test -p infr-vulkan --test gemm_bench -- --ignored --nocapture

use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_vulkan::VulkanBackend;

/// A backend on an EXPLICIT Vulkan kernel-tier configuration (`docs/config-plan.md` S5b).
///
/// The GEMM tier knobs (`INFR_NO_GEMM_WARP`, `INFR_GEMM_WIDE_TILE`, `INFR_NO_SMALL_BM`,
/// `INFR_NO_BM16`) are resolved at BACKEND CONSTRUCTION now, so a bench that used to flip an env
/// var between two runs on one device gets one device per tier instead. That is also what makes
/// these benches honest: an env flip after `VulkanBackend::new()` would silently be a no-op and the
/// two arms would measure the same kernel.
fn be_with(f: impl FnOnce(&mut infr_core::config::VulkanCfg)) -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    f(&mut cfg.kernels.vulkan);
    VulkanBackend::new_with(std::sync::Arc::new(cfg)).unwrap()
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn q4k_gemm_variants_bench() {
    let (m, k, n) = (512usize, 1024usize, 6144usize);
    let reps = 20usize;
    // `mmq` = the dp4a int8 arm; `native` = `matmul_native`, whose tile pick reads
    // `kernels.vulkan.gemm_warp` off the backend it is recorded on.
    let run = |be: &VulkanBackend, name: &str, mmq: bool| {
        // Q4_K: 144 bytes / 256 elems
        let wbytes = n * k / 256 * 144;
        let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
        let a = be.alloc(m * k * 4, BufferUsage::Activations).unwrap();
        let c = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();
        // mmq activation quant buffers
        let nblk = k / 32;
        let qa = be.alloc(m * k, BufferUsage::Activations).unwrap();
        let dact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
        let f = |rec: &infr_vulkan::Recorder| {
            if mmq {
                rec.quant_q8(a.as_ref(), qa.as_ref(), dact.as_ref(), sact.as_ref(), m, k);
                rec.matmul_mmq_q4k(
                    qa.as_ref(),
                    dact.as_ref(),
                    sact.as_ref(),
                    w.as_ref(),
                    0,
                    c.as_ref(),
                    m,
                    k,
                    n,
                );
            } else {
                rec.matmul_native(
                    infr_core::DType::Q4K,
                    a.as_ref(),
                    w.as_ref(),
                    c.as_ref(),
                    m,
                    k,
                    n,
                );
            }
        };
        // warmup (pipeline compile)
        let rec = be.recorder().unwrap();
        f(&rec);
        rec.finish().unwrap();
        let t0 = std::time::Instant::now();
        let rec = be.recorder().unwrap();
        for _ in 0..reps {
            f(&rec);
        }
        rec.finish().unwrap();
        let us = t0.elapsed().as_micros() as f64 / reps as f64;
        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / (us * 1e3);
        println!("{name:>10}: {us:8.1} us/GEMM  ({gflops:.0} GFLOP/s)");
    };

    let be = be_with(|_| {});
    run(&be, "mmq", true);
    let be64 = be_with(|v| v.gemm_warp = false);
    run(&be64, "native64", false);
    run(&be, "warp", false);
}

/// Sum the real qwen35 prefill GEMM inventory (one 512-row chunk) on the warp kernel — ground
/// truth for how much of the chunk's execute time is genuinely GEMM.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn qwen35_gemm_inventory_bench() {
    let be = VulkanBackend::new().unwrap();
    let m = 512usize;
    // (k, n, count): DeltaNet-layer in-proj/out/FFN ×18, attention-layer qkv/out/FFN ×6
    let shapes = [
        (1024usize, 6144usize, 18usize), // qkvz in-proj
        (2048, 1024, 18),                // deltanet out-proj
        (1024, 4096, 24),                // FFN gate+up (both layer kinds)
        (2048, 1024, 24),                // FFN down
        (1024, 3072, 6),                 // attn qkv(+gate)
        (2048, 1024, 6),                 // attn out
    ];
    let a = be.alloc(m * 2048 * 4, BufferUsage::Activations).unwrap();
    let c = be.alloc(m * 6144 * 4, BufferUsage::Activations).unwrap();
    let mut total = 0f64;
    for (k, n, cnt) in shapes {
        let w = be.alloc(n * k / 256 * 144, BufferUsage::Weights).unwrap();
        // warmup
        let rec = be.recorder().unwrap();
        rec.matmul_native(
            infr_core::DType::Q4K,
            a.as_ref(),
            w.as_ref(),
            c.as_ref(),
            m,
            k,
            n,
        );
        rec.finish().unwrap();
        let reps = 10usize;
        let t0 = std::time::Instant::now();
        let rec = be.recorder().unwrap();
        for _ in 0..reps {
            rec.matmul_native(
                infr_core::DType::Q4K,
                a.as_ref(),
                w.as_ref(),
                c.as_ref(),
                m,
                k,
                n,
            );
        }
        rec.finish().unwrap();
        let us = t0.elapsed().as_micros() as f64 / reps as f64;
        println!(
            "[{k}x{n}] {us:8.1} us  ×{cnt} = {:.1} ms",
            us * cnt as f64 / 1e3
        );
        total += us * cnt as f64 / 1e3;
    }
    println!("qwen35 512-row chunk GEMM total: {total:.1} ms");
}

/// Sum the real qwen3-0.6B Q8_0 prefill GEMM inventory (m=512) per kernel variant — the pp512
/// sweep gap is 84% GEMM time, and the suspect is narrow-n occupancy on the warp tile (n=1024 →
/// 32 workgroups on a 96-CU part). Prints per-shape µs + effective TFLOPS for warp vs native64.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn qwen3_gemm_inventory_bench() {
    let m = 512usize;
    // (k, n, count/layer-set): q, k+v, o, gate+up (fused), down — 28 layers.
    let shapes = [
        (1024usize, 2048usize, 28usize), // q
        (1024, 1024, 56),                // k, v
        (2048, 1024, 28),                // o
        (1024, 6144, 28),                // gate+up (combined)
        (3072, 1024, 28),                // down
    ];
    // One backend per tile tier (`kernels.vulkan.gemm_warp`), each with its own buffers.
    for (variant, be) in [
        ("warp", be_with(|_| {})),
        ("native64", be_with(|v| v.gemm_warp = false)),
    ] {
        let a = be.alloc(m * 3072 * 4, BufferUsage::Activations).unwrap();
        let c = be.alloc(m * 6144 * 4, BufferUsage::Activations).unwrap();
        let mut total = 0f64;
        for (k, n, cnt) in shapes {
            let w = be.alloc(n * k / 32 * 34, BufferUsage::Weights).unwrap();
            let rec = be.recorder().unwrap(); // warmup (pipeline compile)
            rec.matmul_native(
                infr_core::DType::Q8_0,
                a.as_ref(),
                w.as_ref(),
                c.as_ref(),
                m,
                k,
                n,
            );
            rec.finish().unwrap();
            let reps = 20usize;
            let t0 = std::time::Instant::now();
            let rec = be.recorder().unwrap();
            for _ in 0..reps {
                rec.matmul_native(
                    infr_core::DType::Q8_0,
                    a.as_ref(),
                    w.as_ref(),
                    c.as_ref(),
                    m,
                    k,
                    n,
                );
            }
            rec.finish().unwrap();
            let us = t0.elapsed().as_micros() as f64 / reps as f64;
            let tflops = (2.0 * m as f64 * k as f64 * n as f64) / us / 1e6;
            println!(
                "[{variant:>8}] [{k}x{n}] {us:8.1} us  {tflops:5.1} TF  ×{cnt} = {:.2} ms",
                us * cnt as f64 / 1e3
            );
            total += us * cnt as f64 / 1e3;
        }
        println!("[{variant:>8}] qwen3-0.6B m=512 GEMM total: {total:.1} ms\n");
    }
}

/// The kernel-tuning harness: the real 8B Q4_K prefill shapes (m=512) on the warp kernel —
/// these run at the kernel's ceiling (~33-36 TF vs llama.cpp's ~45-50 on the same shapes), so
/// any micro-arch change shows here directly. Serialized reps = per-dispatch latency.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn qwen3_8b_gemm_shapes_bench() {
    let be = VulkanBackend::new().unwrap();
    let m = 512usize;
    let shapes = [
        (4096usize, 6144usize, "qkv"),
        (4096, 4096, "o"),
        (4096, 24576, "gate+up"),
        (12288, 4096, "down"),
    ];
    let a = be.alloc(m * 12288 * 4, BufferUsage::Activations).unwrap();
    let a16 = be.alloc(m * 12288 * 2, BufferUsage::Activations).unwrap();
    let c = be.alloc(m * 24576 * 4, BufferUsage::Activations).unwrap();
    for (k, n, label) in shapes {
        let w = be.alloc(n * k / 256 * 144, BufferUsage::Weights).unwrap();
        for f16a in [false, true] {
            let run = |reps: usize| {
                let rec = be.recorder().unwrap();
                for _ in 0..reps {
                    if f16a {
                        rec.store_f16(a.as_ref(), a16.as_ref(), m * k, 0);
                        rec.matmul_native_f16a(
                            infr_core::DType::Q4K,
                            a16.as_ref(),
                            w.device_addr().unwrap(),
                            0,
                            c.as_ref(),
                            m,
                            k,
                            n,
                        );
                    } else {
                        rec.matmul_native(
                            infr_core::DType::Q4K,
                            a.as_ref(),
                            w.as_ref(),
                            c.as_ref(),
                            m,
                            k,
                            n,
                        );
                    }
                }
                rec.finish().unwrap();
            };
            run(1);
            let reps = 10usize;
            let t0 = std::time::Instant::now();
            run(reps);
            let us = t0.elapsed().as_micros() as f64 / reps as f64;
            let tflops = (2.0 * m as f64 * k as f64 * n as f64) / us / 1e6;
            let tag = if f16a { "f16a" } else { "f32 " };
            println!("[{label:>8}] [{k}x{n}] {tag} {us:8.1} us  {tflops:5.1} TF");
        }
    }
}

/// Wide-square occupancy sweep: the n=4096 Q4_K prefill shapes (o-proj 4096x4096, down
/// 12288x4096) land on the WIDE ag tile at exactly ceil(m/64)·(n/256) = 8·16 = 128 workgroups —
/// underfilling a 48-WGP part (144 slots at occ 3) → ~36 TF vs ~48-50 for the well-filled shapes.
/// Sweeps: wide ag (old default, via INFR_GEMM_WIDE_TILE), n128 ag (new default), and split-K
/// at splits 2/3/4 (2×/3×/4× workgroups + a reduce). Run:
/// cargo test -p infr-vulkan --test gemm_bench wide_square -- --ignored --nocapture
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn wide_square_occupancy_sweep() {
    let m = 512usize;
    let dt = infr_core::DType::Q4K;
    let shapes = [
        (4096usize, 4096usize, "o"),
        (12288, 4096, "down"),
        (4096, 6144, "qkv"),
        (4096, 24576, "gate+up"),
    ];
    // Two backends: the default (n128) tile and the restored BN=256 wide tile
    // (`kernels.vulkan.gemm_wide_tile`, `INFR_GEMM_WIDE_TILE`). Buffers belong to their backend.
    let be = be_with(|_| {});
    let bew = be_with(|v| v.gemm_wide_tile = true);
    let scratch = |b: &VulkanBackend| {
        (
            b.alloc(m * 12288 * 4, BufferUsage::Activations).unwrap(),
            b.alloc(m * 12288 * 2, BufferUsage::Activations).unwrap(),
            b.alloc(m * 24576 * 4, BufferUsage::Activations).unwrap(),
        )
    };
    let (a, a16, c) = scratch(&be);
    let (aw, a16w, cw) = scratch(&bew);
    let reps = 30usize;
    let tf = |us: f64, k: usize, n: usize| (2.0 * m as f64 * k as f64 * n as f64) / us / 1e6;

    for (k, n, label) in shapes {
        let w = be.alloc(n * k / 256 * 144, BufferUsage::Weights).unwrap();
        let ww = bew.alloc(n * k / 256 * 144, BufferUsage::Weights).unwrap();
        let mpad = m.div_ceil(64) * 64;
        let time = |b: &VulkanBackend, f: &dyn Fn(&infr_vulkan::Recorder)| -> f64 {
            let rec = b.recorder().unwrap();
            f(&rec);
            rec.finish().unwrap(); // warmup (pipeline compile)
            let t0 = std::time::Instant::now();
            let rec = b.recorder().unwrap();
            for _ in 0..reps {
                f(&rec);
            }
            rec.finish().unwrap();
            t0.elapsed().as_micros() as f64 / reps as f64
        };

        // wide ag (old BN=256 tile, restored via `kernels.vulkan.gemm_wide_tile`)
        let us = time(&bew, &|rec| {
            rec.store_f16(aw.as_ref(), a16w.as_ref(), m * k, 0);
            rec.matmul_native_f16a(
                dt,
                a16w.as_ref(),
                ww.device_addr().unwrap(),
                0,
                cw.as_ref(),
                m,
                k,
                n,
            );
        });
        println!(
            "[{label:>5}] [{k}x{n}] wide_ag      {us:7.1} us  {:5.1} TF",
            tf(us, k, n)
        );

        // n128 ag (BN=128 → 2× workgroups) — the new default
        let us = time(&be, &|rec| {
            rec.store_f16(a.as_ref(), a16.as_ref(), m * k, 0);
            rec.matmul_native_f16a(
                dt,
                a16.as_ref(),
                w.device_addr().unwrap(),
                0,
                c.as_ref(),
                m,
                k,
                n,
            );
        });
        println!(
            "[{label:>5}] [{k}x{n}] n128_ag      {us:7.1} us  {:5.1} TF",
            tf(us, k, n)
        );

        // narrow split-K, splits 2/3/4
        for splits in [2usize, 3, 4] {
            let pk = be
                .alloc(splits * mpad * n * 4, BufferUsage::Activations)
                .unwrap();
            let us = time(&be, &|rec| {
                rec.store_f16(a.as_ref(), a16.as_ref(), m * k, 0);
                rec.matmul_native_splitk(
                    dt,
                    a16.as_ref(),
                    w.device_addr().unwrap(),
                    0,
                    pk.as_ref(),
                    c.as_ref(),
                    m,
                    k,
                    n,
                    splits,
                    true,
                );
            });
            println!(
                "[{label:>5}] [{k}x{n}] sk_ag x{splits}     {us:7.1} us  {:5.1} TF",
                tf(us, k, n)
            );
        }
    }
}

/// Crossover characterization: wide (BN=256) vs n128 (BN=128) ag tile across a grid of (m, k, n)
/// that spans the small-model regime (shallow k 1024-1152, small n) and the 8B regime (deep k,
/// large n). Finds whether the wide tile EVER beats n128 (→ a shape-gated selection), or n128 wins
/// throughout (→ the flat flip is safe). Run:
/// cargo test -p infr-vulkan --test gemm_bench crossover -- --ignored --nocapture
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn wide_n128_crossover_sweep() {
    // One backend per tile (`kernels.vulkan.gemm_wide_tile`) — the arms interleave, so they must be
    // two live devices rather than one device and an env flip between runs.
    let be = be_with(|_| {});
    let bew = be_with(|v| v.gemm_wide_tile = true);
    let dt = infr_core::DType::Q4K;
    // (m, k, n): only n%256==0 (wide-eligible). Covers qwen3-0.6b (k=1024, n up to 6144),
    // gemma-3-1b-shaped (k=1152), and the 8B shapes; plus small-m (last prefill chunk) rows.
    let grid = [
        (512usize, 1024usize, 2048usize),
        (512, 1024, 6144),
        (512, 1152, 6912),
        (512, 1152, 13824),
        (512, 2048, 1024),
        (512, 3072, 1024),
        (512, 4096, 4096),
        (512, 4096, 6144),
        (512, 4096, 24576),
        (512, 12288, 4096),
        (256, 1024, 6144),
        (128, 1024, 6144),
        (64, 4096, 4096),
        (256, 4096, 4096),
    ];
    let amax = 12288usize;
    let nmax = 24576usize;
    let a16 = be.alloc(512 * amax * 2, BufferUsage::Activations).unwrap();
    let c = be.alloc(512 * nmax * 4, BufferUsage::Activations).unwrap();
    let a16w = bew.alloc(512 * amax * 2, BufferUsage::Activations).unwrap();
    let cw = bew.alloc(512 * nmax * 4, BufferUsage::Activations).unwrap();
    let reps = 40usize;
    println!(
        "{:>4} {:>6} {:>6} | {:>8} {:>8} | {:>8} {:>8} | winner",
        "m", "k", "n", "wide us", "wideTF", "n128 us", "n128TF"
    );
    for (m, k, n) in grid {
        let w = be.alloc(n * k / 256 * 144, BufferUsage::Weights).unwrap();
        let ww = bew.alloc(n * k / 256 * 144, BufferUsage::Weights).unwrap();
        let tf = |us: f64| (2.0 * m as f64 * k as f64 * n as f64) / us / 1e6;
        let time = |wide: bool| -> f64 {
            let (b, aa, cc, wt) = if wide {
                (&bew, &a16w, &cw, &ww)
            } else {
                (&be, &a16, &c, &w)
            };
            let f = |rec: &infr_vulkan::Recorder| {
                rec.matmul_native_f16a(
                    dt,
                    aa.as_ref(),
                    wt.device_addr().unwrap(),
                    0,
                    cc.as_ref(),
                    m,
                    k,
                    n,
                );
            };
            let rec = b.recorder().unwrap();
            f(&rec);
            rec.finish().unwrap(); // warmup
            let t0 = std::time::Instant::now();
            let rec = b.recorder().unwrap();
            for _ in 0..reps {
                f(&rec);
            }
            rec.finish().unwrap();
            t0.elapsed().as_micros() as f64 / reps as f64
        };
        // interleave wide/n128 twice, take the min of each (thermal-robust)
        let (mut uw, mut un) = (f64::MAX, f64::MAX);
        for _ in 0..2 {
            uw = uw.min(time(true));
            un = un.min(time(false));
        }
        let win = if un < uw { "n128" } else { "WIDE" };
        let wide_grid = m.div_ceil(64) * (n / 256).max(1);
        println!(
            "{m:>4} {k:>6} {n:>6} | {uw:8.1} {:8.1} | {un:8.1} {:8.1} | {win}  (wg={wide_grid})",
            tf(uw),
            tf(un)
        );
    }
}

/// Per-op serialization floor: a chain of small hazard-dependent dispatches (each reads the
/// previous one's output → global barrier each). wall/ops ≈ the fixed bubble every seam op pays
/// on top of its kernel time — the number that says how much op-count reduction / overlap is worth.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn chained_op_bubble_bench() {
    let be = VulkanBackend::new().unwrap();
    let n = 512usize * 1024; // one chunk's hidden activations
    let a = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    let b = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    let w = be.alloc(1024 * 4, BufferUsage::Activations).unwrap();
    for ops in [50usize, 400] {
        // warmup
        let rec = be.recorder().unwrap();
        rec.rmsnorm(a.as_ref(), w.as_ref(), b.as_ref(), 512, 1024, 1e-6);
        rec.finish().unwrap();
        let t0 = std::time::Instant::now();
        let rec = be.recorder().unwrap();
        for i in 0..ops {
            // ping-pong a→b→a…: every dispatch RAW-depends on the previous one
            let (x, y) = if i % 2 == 0 { (&a, &b) } else { (&b, &a) };
            rec.rmsnorm(x.as_ref(), w.as_ref(), y.as_ref(), 512, 1024, 1e-6);
        }
        rec.finish().unwrap();
        let us = t0.elapsed().as_micros() as f64;
        println!(
            "{ops} chained rmsnorm(512x1024): {:.1} us total, {:.1} us/op",
            us,
            us / ops as f64
        );
    }
}

/// Real isolated cost of the two remaining big prefill ops at qwen35 shapes.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn qwen35_dn_attn_bench() {
    let be = VulkanBackend::new().unwrap();
    let rows = 512usize;
    let (nv, nk, kd, vd) = (16usize, 16usize, 128usize, 128usize);
    let q = be
        .alloc(rows * nk * kd * 4, BufferUsage::Activations)
        .unwrap();
    let k = be
        .alloc(rows * nk * kd * 4, BufferUsage::Activations)
        .unwrap();
    let v = be
        .alloc(rows * nv * vd * 4, BufferUsage::Activations)
        .unwrap();
    let b = be.alloc(rows * nv * 4, BufferUsage::Activations).unwrap();
    let al = be.alloc(rows * nv * 4, BufferUsage::Activations).unwrap();
    let ac = be.alloc(nv * 4, BufferUsage::Weights).unwrap();
    let dt = be.alloc(nv * 4, BufferUsage::Weights).unwrap();
    let st = be
        .alloc(nv * kd * vd * 4, BufferUsage::Activations)
        .unwrap();
    let o = be
        .alloc(rows * nv * vd * 4, BufferUsage::Activations)
        .unwrap();
    let reps = 10usize;
    let rec = be.recorder().unwrap();
    rec.deltanet_chunked(
        q.as_ref(),
        k.as_ref(),
        v.as_ref(),
        b.as_ref(),
        al.as_ref(),
        ac.as_ref(),
        dt.as_ref(),
        st.as_ref(),
        o.as_ref(),
        rows,
        nv,
        nk,
        kd,
        vd,
        1e-6,
    );
    rec.finish().unwrap();
    let t0 = std::time::Instant::now();
    let rec = be.recorder().unwrap();
    for _ in 0..reps {
        rec.deltanet_chunked(
            q.as_ref(),
            k.as_ref(),
            v.as_ref(),
            b.as_ref(),
            al.as_ref(),
            ac.as_ref(),
            dt.as_ref(),
            st.as_ref(),
            o.as_ref(),
            rows,
            nv,
            nk,
            kd,
            vd,
            1e-6,
        );
    }
    rec.finish().unwrap();
    let us = t0.elapsed().as_micros() as f64 / reps as f64;
    println!(
        "deltanet_chunked rows=512: {us:.1} us/op  ×18 = {:.1} ms/chunk",
        us * 18.0 / 1e3
    );

    // split variant (prep + gates + scan)
    let nchunk = rows.div_ceil(32);
    let kn = be
        .alloc(rows * nk * kd * 4, BufferUsage::Activations)
        .unwrap();
    let qn = be
        .alloc(rows * nk * kd * 4, BufferUsage::Activations)
        .unwrap();
    let dkb = be
        .alloc(nchunk * nk * 1024 * 4, BufferUsage::Activations)
        .unwrap();
    let dqb = be
        .alloc(nchunk * nk * 1024 * 4, BufferUsage::Activations)
        .unwrap();
    let bg = be
        .alloc(nchunk * nv * 32 * 4, BufferUsage::Activations)
        .unwrap();
    let gg = be
        .alloc(nchunk * nv * 32 * 4, BufferUsage::Activations)
        .unwrap();
    let split = |rec: &infr_vulkan::Recorder| {
        rec.deltanet_chunked_split(
            q.as_ref(),
            k.as_ref(),
            v.as_ref(),
            b.as_ref(),
            al.as_ref(),
            ac.as_ref(),
            dt.as_ref(),
            st.as_ref(),
            o.as_ref(),
            kn.as_ref(),
            qn.as_ref(),
            dkb.as_ref(),
            dqb.as_ref(),
            bg.as_ref(),
            gg.as_ref(),
            rows,
            nv,
            nk,
            kd,
            vd,
            1e-6,
        );
    };
    let rec = be.recorder().unwrap();
    split(&rec);
    rec.finish().unwrap();
    let t0 = std::time::Instant::now();
    let rec = be.recorder().unwrap();
    for _ in 0..reps {
        split(&rec);
    }
    rec.finish().unwrap();
    let us = t0.elapsed().as_micros() as f64 / reps as f64;
    println!(
        "deltanet_split   rows=512: {us:.1} us/op  ×18 = {:.1} ms/chunk",
        us * 18.0 / 1e3
    );

    // nonfa attention at qwen35 attn shape: rows=512, kv=822, nh=16, nkv=2, hd=256
    let (nh, nkv, hd, kv_len) = (16usize, 2usize, 256usize, 822usize);
    let mpad = 512usize;
    let kv_pad = kv_len.div_ceil(256) * 256;
    let qb = be
        .alloc(mpad * nh * hd * 2, BufferUsage::Activations)
        .unwrap();
    let kc = be
        .alloc(kv_len * nkv * hd * 2, BufferUsage::Activations)
        .unwrap();
    let vc = be
        .alloc(kv_len * nkv * hd * 2, BufferUsage::Activations)
        .unwrap();
    let at = be
        .alloc(mpad * nh * hd * 4, BufferUsage::Activations)
        .unwrap();
    let s = be
        .alloc(nh * mpad * kv_pad * 2, BufferUsage::Activations)
        .unwrap();
    let pv = be
        .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
        .unwrap();
    let rec = be.recorder().unwrap();
    rec.attention_prefill_nonfa(
        qb.as_ref(),
        kc.as_ref(),
        vc.as_ref(),
        at.as_ref(),
        s.as_ref(),
        pv.as_ref(),
        mpad,
        kv_len,
        nh,
        nkv,
        hd,
        310,
        0,
        0.0,
    );
    rec.finish().unwrap();
    let t0 = std::time::Instant::now();
    let rec = be.recorder().unwrap();
    for _ in 0..reps {
        rec.attention_prefill_nonfa(
            qb.as_ref(),
            kc.as_ref(),
            vc.as_ref(),
            at.as_ref(),
            s.as_ref(),
            pv.as_ref(),
            mpad,
            kv_len,
            nh,
            nkv,
            hd,
            310,
            0,
            0.0,
        );
    }
    rec.finish().unwrap();
    let us = t0.elapsed().as_micros() as f64 / reps as f64;
    println!(
        "nonfa attn rows=512 kv=822 hd=256: {us:.1} us/op  ×6 = {:.1} ms/chunk",
        us * 6.0 / 1e3
    );
}

/// Isolated decode-attention kernel A/B at deep context (qwen3-0.6b dims, kv=8000): the push-const
/// split, the params-driven dyn split, and the self-chunking dynac variant. Hunts the seam-vs-
/// bespoke deep-decode gap.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn decode_attn_variants_bench() {
    let be = VulkanBackend::new().unwrap();
    let (nh, nkv, hd) = (16usize, 8usize, 128usize);
    let kv_len = 8000usize;
    let cap = 8065usize;
    let q = be.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
    let kc = be
        .alloc(cap * nkv * hd * 2, BufferUsage::Activations)
        .unwrap();
    let vc = be
        .alloc(cap * nkv * hd * 2, BufferUsage::Activations)
        .unwrap();
    let o = be.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
    let params = be.alloc(8, BufferUsage::Activations).unwrap();
    be.upload(
        params.as_ref(),
        bytemuck::cast_slice(&[kv_len as u32 - 1, kv_len as u32]),
    )
    .unwrap();
    let reps = 200usize;

    let run = |name: &str, f: &dyn Fn(&infr_vulkan::Recorder)| {
        let rec = be.recorder().unwrap();
        f(&rec);
        rec.finish().unwrap();
        let t0 = std::time::Instant::now();
        let rec = be.recorder().unwrap();
        for _ in 0..reps {
            f(&rec);
        }
        rec.finish().unwrap();
        println!(
            "{name:>22}: {:8.1} us/op",
            t0.elapsed().as_micros() as f64 / reps as f64
        );
    };

    // static split (bespoke-style): adaptive chunk for kv=8000
    let chunk = (kv_len / 32).clamp(64, 512);
    let n_chunks = kv_len.div_ceil(chunk);
    let pm = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc = be
        .alloc(nh * n_chunks * hd * 4, BufferUsage::Activations)
        .unwrap();
    run("static split c250", &|rec| {
        rec.attention_kv_split(
            q.as_ref(),
            kc.as_ref(),
            vc.as_ref(),
            o.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            1,          // rows (decode shape)
            kv_len - 1, // pos of the single query row
            kv_len,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            0.0,
            0,
            None,  // canvas_lo
            false, // k f16
            false, // v f16
            0,     // cap (unused for f16)
            false, // batched: decode shape stays on the per-row grid
        );
    });
    // dyn split, same chunks
    run("dyn split c250", &|rec| {
        rec.attention_kv_split_dyn(
            q.as_ref(),
            kc.as_ref(),
            vc.as_ref(),
            o.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            params.as_ref(),
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            0.0,
            0,
        );
    });
    // dynac: baked min chunk 64, capacity-sized scratch (the seam's config)
    let cap_chunks = cap.div_ceil(64);
    let pm2 = be
        .alloc(nh * cap_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl2 = be
        .alloc(nh * cap_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc2 = be
        .alloc(nh * cap_chunks * hd * 4, BufferUsage::Activations)
        .unwrap();
    let args = be.alloc(16, BufferUsage::Activations).unwrap();
    run("dynac cap126", &|rec| {
        rec.attn_live_prologue(params.as_ref(), args.as_ref(), nh, 64, 0);
        rec.attention_kv_split_dynac(
            q.as_ref(),
            kc.as_ref(),
            vc.as_ref(),
            o.as_ref(),
            pm2.as_ref(),
            pl2.as_ref(),
            pacc2.as_ref(),
            params.as_ref(),
            args.as_ref(),
            nh,
            nkv,
            hd,
            64,
            cap_chunks,
            0.0,
            0,
            false, // f16 KV cache
            0,     // cap (unused for f16)
        );
    });
    // dynac with a TIGHT bake (capacity == live): isolates the dead-workgroup/scan cost from the
    // SELF_CHUNK in-kernel logic cost.
    run("dynac tight c250", &|rec| {
        rec.attn_live_prologue(params.as_ref(), args.as_ref(), nh, chunk, 0);
        rec.attention_kv_split_dynac(
            q.as_ref(),
            kc.as_ref(),
            vc.as_ref(),
            o.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            params.as_ref(),
            args.as_ref(),
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            0.0,
            0,
            false, // f16 KV cache
            0,     // cap (unused for f16)
        );
    });
    // dyn split with chunk=64 all live (the earlier env-sweep shape)
    let n64 = kv_len.div_ceil(64);
    run("dyn split c64", &|rec| {
        rec.attention_kv_split_dyn(
            q.as_ref(),
            kc.as_ref(),
            vc.as_ref(),
            o.as_ref(),
            pm2.as_ref(),
            pl2.as_ref(),
            pacc2.as_ref(),
            params.as_ref(),
            nh,
            nkv,
            hd,
            64,
            n64,
            0.0,
            0,
        );
    });
}

/// DiffusionGemma slice 6: probe how much of `matmul_mmq_experts`'s cost at DG's shapes is
/// GENUINE compute vs the fixed worst-case grid (`rows` bound = canvas tokens = 256, so
/// `gx = ceil(rows/64)*(n/64)` is the SAME regardless of the real per-expert counts — early-exit
/// workgroups are supposed to be nearly free per docs/perf/playbook.md's class-4 precedent). Varies ONLY
/// the `rows` bound argument (grid-sizing only, never read by the shader for real row ranges)
/// against a FIXED, realistic packed layout (128 experts, 2048 pairs, ~16 rows/expert average —
/// the canvas-256/n_used-8 arithmetic) to isolate "cost of the bound" from "cost of the real
/// work". If shrinking the bound doesn't shrink wall time, early-exit workgroups really are free
/// and an adaptive (indirect-dispatch) bound isn't worth building. If it does, the delta between
/// `bound=256` (today's production grid) and `bound=<realistic max>` is the ceiling an adaptive
/// bound could recover.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn moe_expert_grid_bound_bench() {
    let be = VulkanBackend::new().unwrap();
    let (ne, nff, n_expert, n_used, tokens) = (2816usize, 704usize, 128usize, 8usize, 256usize);
    let n_pairs = tokens * n_used; // 2048
    let npad = n_pairs.div_ceil(64) * 64 + 64;
    let reps = 30usize;

    // Packed activation buffers: gate_up reads k=ne=2816 (qa/dact/sact), down reads its OWN
    // k=nff=704 packed activations (dqa/dda) — separate buffers, correctly strided, matching the
    // real two-quantization-layer pipeline (`quant_q8_gather` then `quant_q8` on the FFN act).
    let qa = be.alloc(npad * ne, BufferUsage::Activations).unwrap();
    let dact = be
        .alloc(npad * (ne / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let sact = be
        .alloc(npad * (ne / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let dqa = be.alloc(npad * nff, BufferUsage::Activations).unwrap();
    let dda = be
        .alloc(npad * (nff / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let gu_c = be
        .alloc(npad * (2 * nff) * 4, BufferUsage::Activations)
        .unwrap();
    let down_c = be.alloc(npad * ne * 4, BufferUsage::Activations).unwrap();

    // Weight banks: Q4_K gate_up [n_expert, 2*nff, ne] (144B/256-elem block), Q5_0 down
    // [n_expert, ne, nff] (22B/32-elem block). Contents are zeros — perf only, no golden check.
    let gu_w = be
        .alloc(
            n_expert * (2 * nff) * (ne / 256) * 144,
            BufferUsage::Weights,
        )
        .unwrap();
    let down_w = be
        .alloc(n_expert * ne * (nff / 32) * 22, BufferUsage::Weights)
        .unwrap();

    // counts/offsets uploader for a given per-expert distribution (must sum to n_pairs and every
    // entry must be <= every `bound` this distribution is tested against, or the run under-covers
    // its own segment — fine for a perf-only probe, but keep the invariant so timings stay honest).
    let upload_dist = |counts_v: &[u32]| -> (
        Box<dyn infr_core::backend::Buffer>,
        Box<dyn infr_core::backend::Buffer>,
    ) {
        assert_eq!(counts_v.len(), n_expert);
        assert_eq!(counts_v.iter().sum::<u32>() as usize, n_pairs);
        let mut offsets_v = vec![0u32; n_expert];
        let mut acc = 0u32;
        for e in 0..n_expert {
            offsets_v[e] = acc;
            acc += counts_v[e];
        }
        let counts = be.alloc(n_expert * 4, BufferUsage::Activations).unwrap();
        let offsets = be.alloc(n_expert * 4, BufferUsage::Activations).unwrap();
        be.upload(counts.as_ref(), bytemuck::cast_slice(counts_v))
            .unwrap();
        be.upload(offsets.as_ref(), bytemuck::cast_slice(&offsets_v))
            .unwrap();
        (counts, offsets)
    };

    // 16 each, sum 2048.
    let balanced: Vec<u32> = vec![(n_pairs / n_expert) as u32; n_expert];
    // Skewed: one "hot" expert takes 150 of the 256 tokens' assignments (near the hard ceiling —
    // a top-k router picks each expert at most once per token, so any expert's count is bounded by
    // `tokens`=256, never `n_pairs`=2048; see matmul_mmq_experts' doc), the rest share the
    // remainder — stresses the same bound question under real imbalance instead of the average.
    let hot = 150u32;
    let mut skewed = vec![(n_pairs as u32 - hot) / (n_expert as u32 - 1); n_expert];
    skewed[0] = hot;
    {
        let sum: u32 = skewed.iter().sum();
        skewed[n_expert - 1] += n_pairs as u32 - sum; // fix up rounding remainder
    }

    // `gemm` takes (rec, bound, counts, offsets) — the caller supplies the distribution buffers
    // so this closure only knows the GEMM shape (gate_up vs down), not any specific distribution.
    #[allow(clippy::type_complexity)]
    let run =
        |label: &str, gemm: &dyn Fn(&infr_vulkan::Recorder, usize, &dyn Buffer, &dyn Buffer)| {
            for (dist_name, counts_v) in [("balanced~16", &balanced), ("skewed", &skewed)] {
                let (counts, offsets) = upload_dist(counts_v);
                let max_real = *counts_v.iter().max().unwrap();
                for &bound in &[64usize, 128, 192, 256] {
                    if (bound as u32) < max_real {
                        continue; // would silently truncate this distribution's hottest expert
                    }
                    let rec = be.recorder().unwrap();
                    gemm(&rec, bound, counts.as_ref(), offsets.as_ref()); // warmup (pipeline compile)
                    rec.finish().unwrap();
                    let t0 = std::time::Instant::now();
                    let rec = be.recorder().unwrap();
                    for _ in 0..reps {
                        gemm(&rec, bound, counts.as_ref(), offsets.as_ref());
                    }
                    rec.finish().unwrap();
                    let us = t0.elapsed().as_micros() as f64 / reps as f64;
                    println!(
                    "[{label:>11}] dist={dist_name:>11} (max={max_real:3}) bound={bound:3}: {us:7.1} us"
                );
                }
            }
        };

    run("gate_up", &|rec, bound, counts, offsets| {
        rec.matmul_mmq_experts(
            infr_core::DType::Q4K,
            "expert_gateup",
            qa.as_ref(),
            dact.as_ref(),
            Some(sact.as_ref()),
            gu_w.as_ref(),
            0,
            ne, // stride = k (per-expert weight stride, elements)
            counts,
            offsets,
            gu_c.as_ref(),
            bound,
            ne,
            2 * nff,
            n_expert,
            n_used,
        );
    });
    run("down", &|rec, bound, counts, offsets| {
        rec.matmul_mmq_experts(
            infr_core::DType::Q5_0,
            "expert_down",
            dqa.as_ref(),
            dda.as_ref(),
            None, // Q5_0 is symmetric — no min term
            down_w.as_ref(),
            0,
            nff, // stride = k
            counts,
            offsets,
            down_c.as_ref(),
            bound,
            nff,
            ne,
            n_expert,
            n_used,
        );
    });
    // A BN=128 tile (halving down's 44 N-tiles to 22, matching gate_up's grid granularity) was
    // probed here at both TN=8 (doubles the per-thread accumulator: 1.3-1.6ms, a clear LOSS vs
    // baseline's ~1.25-1.37ms — register-pressure/occupancy cost, the exact class-3 risk
    // docs/perf/playbook.md warns "bigger tiles often lose" for) and TN=4/THREADS=512 (register-neutral:
    // ~1.3-1.35ms, a wash/marginal loss within noise). Neither improved on the baseline BN=64
    // tile, so down's lower TFLOPS-vs-gate_up efficiency (measured via INFR_PROF_OPS: ~8.1 vs
    // ~10.2 TFLOPS at production counts) is a K-depth ceiling (k=nff=704 → only 22 BLK=32
    // iterations, less loop depth to amortize fixed per-iteration cost), not a tile/occupancy
    // config bug — reverted both variants per docs/perf/playbook.md's "a measured wash gets reverted"
    // rule rather than landing a wash.
}

/// BM=64 vs BM=32 ROW tile at REAL small-rows-per-expert shapes (post routing-fix `be47c91`):
/// qwen3.6-MoE's 256-expert pool averages ~16 rows/expert at pp512 (`rows·n_used/n_expert`),
/// qwen3-30B-A3B's 128-expert pool averages ~32. `matmul_mmq_experts` picks BM=32
/// (`native_gemm_mmq_*_xp32`) below `MOE_EXPERT_SMALL_TILE_AVG_ROWS` and BM=64 (unchanged,
/// default) at/above it — this bench drives BOTH tiles against the SAME balanced counts
/// distribution per shape (the `n_used` argument only steers the Rust-side tile pick, never read
/// by the shader, so overriding it to force the "other" tile for comparison doesn't change
/// correctness, only which kernel variant is dispatched) to find the crossover and confirm the
/// picked threshold (24) sits on the right side of both production shapes.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn moe_expert_row_tile_bench() {
    let be = VulkanBackend::new().unwrap();
    let reps = 30usize;

    // (label, ne, nff, n_expert, real n_used, tokens)
    let shapes: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("qwen3.6-moe avg~16", 2048, 512, 256, 8, 512),
        ("qwen3-30B-a3b avg~32", 2048, 768, 128, 8, 512),
        ("avg~48", 2048, 512, 256, 8, 1536),
        ("deep-ctx avg~64", 2048, 512, 256, 8, 2048),
        ("avg~96", 2048, 512, 256, 8, 3072),
        ("avg~128", 2048, 512, 256, 8, 4096),
        ("avg~256(pp8000)", 2048, 512, 256, 8, 8192),
    ];

    for &(label, ne, nff, n_expert, n_used, tokens) in shapes {
        let n_pairs = tokens * n_used;
        let npad = n_pairs.div_ceil(64) * 64 + 64;

        let qa = be.alloc(npad * ne, BufferUsage::Activations).unwrap();
        let dact = be
            .alloc(npad * (ne / 32) * 2, BufferUsage::Activations)
            .unwrap();
        let sact = be
            .alloc(npad * (ne / 32) * 2, BufferUsage::Activations)
            .unwrap();
        let gu_c = be
            .alloc(npad * (2 * nff) * 4, BufferUsage::Activations)
            .unwrap();
        let gu_w = be
            .alloc(
                n_expert * (2 * nff) * (ne / 256) * 144,
                BufferUsage::Weights,
            )
            .unwrap();

        // Balanced: n_pairs spread as evenly as possible (a trained router with a load-balance
        // aux loss keeps aggregate assignment close to uniform over a few hundred tokens).
        let base = (n_pairs / n_expert) as u32;
        let rem = n_pairs - base as usize * n_expert;
        let mut counts_v = vec![base; n_expert];
        for c in counts_v.iter_mut().take(rem) {
            *c += 1;
        }
        let mut offsets_v = vec![0u32; n_expert];
        let mut acc = 0u32;
        for e in 0..n_expert {
            offsets_v[e] = acc;
            acc += counts_v[e];
        }
        let counts = be.alloc(n_expert * 4, BufferUsage::Activations).unwrap();
        let offsets = be.alloc(n_expert * 4, BufferUsage::Activations).unwrap();
        be.upload(counts.as_ref(), bytemuck::cast_slice(&counts_v))
            .unwrap();
        be.upload(offsets.as_ref(), bytemuck::cast_slice(&offsets_v))
            .unwrap();

        // n_used_probe = n_expert*100 forces avg_rows far past the threshold → BM=64, regardless
        // of the shape's real n_used — the counts/offsets/rows stay the SAME real distribution.
        for (tile_label, n_used_probe) in [("BM32", n_used), ("BM64", n_expert * 100)] {
            let rec = be.recorder().unwrap();
            rec.matmul_mmq_experts(
                infr_core::DType::Q4K,
                "bench_gateup",
                qa.as_ref(),
                dact.as_ref(),
                Some(sact.as_ref()),
                gu_w.as_ref(),
                0,
                ne,
                counts.as_ref(),
                offsets.as_ref(),
                gu_c.as_ref(),
                tokens,
                ne,
                2 * nff,
                n_expert,
                n_used_probe,
            );
            rec.finish().unwrap(); // warmup (pipeline compile)
            let t0 = std::time::Instant::now();
            let rec = be.recorder().unwrap();
            for _ in 0..reps {
                rec.matmul_mmq_experts(
                    infr_core::DType::Q4K,
                    "bench_gateup",
                    qa.as_ref(),
                    dact.as_ref(),
                    Some(sact.as_ref()),
                    gu_w.as_ref(),
                    0,
                    ne,
                    counts.as_ref(),
                    offsets.as_ref(),
                    gu_c.as_ref(),
                    tokens,
                    ne,
                    2 * nff,
                    n_expert,
                    n_used_probe,
                );
            }
            rec.finish().unwrap();
            let us = t0.elapsed().as_micros() as f64 / reps as f64;
            let flops = 2.0 * (n_pairs as f64) * (ne as f64) * (2.0 * nff as f64);
            let tflops = flops / (us * 1e-6) / 1e12;
            println!("[{label:>20}] tile={tile_label}: {us:8.1} us  ({tflops:5.2} TFLOP/s useful)");
        }
    }
}

/// Same crossover question as `moe_expert_row_tile_bench`, but for the DOWN projection (Q5_K,
/// qwen3.6-MoE's down bank format): k=nff=512 (16 BLK=32 iterations vs gate_up's 64) — fewer K
/// iterations means the per-workgroup fixed cost (barrier/staging) amortizes over LESS loop depth,
/// which is exactly the "K-depth ceiling" `moe_expert_grid_bound_bench` flagged for why down runs
/// at lower TFLOP/s than gate_up already; a smaller BM tile could plausibly make that fixed-cost
/// ratio worse instead of better, so down needs its own check rather than assuming gate_up's
/// verdict transfers.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn moe_expert_row_tile_bench_down() {
    let be = VulkanBackend::new().unwrap();
    let reps = 30usize;
    let (ne, nff, n_expert, n_used) = (2048usize, 512usize, 256usize, 8usize);

    let shapes: &[(&str, usize)] = &[
        ("avg~16", 512),
        ("avg~32", 1024),
        ("avg~64", 2048),
        ("avg~128", 4096),
    ];

    for &(label, tokens) in shapes {
        let n_pairs = tokens * n_used;
        let npad = n_pairs.div_ceil(64) * 64 + 64;

        let dqa = be.alloc(npad * nff, BufferUsage::Activations).unwrap();
        let dda = be
            .alloc(npad * (nff / 32) * 2, BufferUsage::Activations)
            .unwrap();
        let dsa = be
            .alloc(npad * (nff / 32) * 2, BufferUsage::Activations)
            .unwrap();
        let down_c = be.alloc(npad * ne * 4, BufferUsage::Activations).unwrap();
        let down_w = be
            .alloc(
                n_expert * ne * (nff / 256).max(1) * 176,
                BufferUsage::Weights,
            )
            .unwrap();

        let base = (n_pairs / n_expert) as u32;
        let rem = n_pairs - base as usize * n_expert;
        let mut counts_v = vec![base; n_expert];
        for c in counts_v.iter_mut().take(rem) {
            *c += 1;
        }
        let mut offsets_v = vec![0u32; n_expert];
        let mut acc = 0u32;
        for e in 0..n_expert {
            offsets_v[e] = acc;
            acc += counts_v[e];
        }
        let counts = be.alloc(n_expert * 4, BufferUsage::Activations).unwrap();
        let offsets = be.alloc(n_expert * 4, BufferUsage::Activations).unwrap();
        be.upload(counts.as_ref(), bytemuck::cast_slice(&counts_v))
            .unwrap();
        be.upload(offsets.as_ref(), bytemuck::cast_slice(&offsets_v))
            .unwrap();

        for (tile_label, n_used_probe) in [("BM32", n_used), ("BM64", n_expert * 100)] {
            let rec = be.recorder().unwrap();
            rec.matmul_mmq_experts(
                infr_core::DType::Q5K,
                "bench_down",
                dqa.as_ref(),
                dda.as_ref(),
                Some(dsa.as_ref()),
                down_w.as_ref(),
                0,
                nff,
                counts.as_ref(),
                offsets.as_ref(),
                down_c.as_ref(),
                tokens,
                nff,
                ne,
                n_expert,
                n_used_probe,
            );
            rec.finish().unwrap();
            let t0 = std::time::Instant::now();
            let rec = be.recorder().unwrap();
            for _ in 0..reps {
                rec.matmul_mmq_experts(
                    infr_core::DType::Q5K,
                    "bench_down",
                    dqa.as_ref(),
                    dda.as_ref(),
                    Some(dsa.as_ref()),
                    down_w.as_ref(),
                    0,
                    nff,
                    counts.as_ref(),
                    offsets.as_ref(),
                    down_c.as_ref(),
                    tokens,
                    nff,
                    ne,
                    n_expert,
                    n_used_probe,
                );
            }
            rec.finish().unwrap();
            let us = t0.elapsed().as_micros() as f64 / reps as f64;
            let flops = 2.0 * (n_pairs as f64) * (nff as f64) * (ne as f64);
            let tflops = flops / (us * 1e-6) / 1e12;
            println!(
                "[down {label:>10}] tile={tile_label}: {us:8.1} us  ({tflops:5.2} TFLOP/s useful)"
            );
        }
    }
}

/// BM=64 vs BM=32 dense A_GLOBAL warp-GEMM row tile at REAL small-m batched-prefill shapes: MTP
/// verify's draft window (m≈6-8 steady state, growing to ~m30-50 under the no-rewind fallback)
/// runs every dense projection GEMM through `matmul_native_f16a` (n128_ag family: wide-N shapes
/// like gate_up/vocab-head) or `matmul_native_splitk` (sk_ag family: narrow-N deep-k shapes like
/// down/o/kv-proj) on the qwen35-4B-UD-Q4_K_XL shapes seen under `INFR_MTP=1 INFR_PROF_OP_SHAPES=1`.
/// Sweeps m to find the BM=32/BM=64 crossover and confirm `DENSE_SMALL_TILE_MAX_M` (recorder.rs)
/// sits on the right side of both the steady-state (m≈7) and no-rewind-tail (m≈30-50) regimes.
/// `bm(dtype)`: probes the recorder's real gate through `kernels.vulkan.small_bm`
/// (`INFR_NO_SMALL_BM`, which forces BM=64), so this exercises the SAME code path production uses,
/// not a hand-rolled kernel pick — one backend per tier since S5b.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn dense_small_m_row_tile_bench() {
    let be32 = be_with(|_| {}); // small-tile default: BM=32 within the band
    let be64 = be_with(|v| v.small_bm = false); // `INFR_NO_SMALL_BM`: BM=64
    let reps = 30usize;
    let ms: &[usize] = &[4, 6, 8, 12, 16, 20, 24, 32, 48, 64];

    fn blk(dtype: infr_core::DType) -> (usize, usize) {
        use infr_core::DType::*;
        match dtype {
            Q4K => (256, 144),
            Q5K => (256, 176),
            Q6K => (256, 210),
            Q8_0 => (32, 34),
            _ => unreachable!("bench dtype not covered"),
        }
    }

    // n128_ag family (matmul_native_f16a): wide-N shapes — gate_up fused proj, vocab head.
    let n128_shapes: &[(&str, infr_core::DType, usize, usize)] = &[
        ("gate_up", infr_core::DType::Q4K, 2560, 18432),
        ("vocab_head", infr_core::DType::Q6K, 2560, 248320),
    ];
    for &(label, dtype, k, n) in n128_shapes {
        let (k_, n_) = (k, n);
        let mpad_max = 64usize;
        let (belem, bbytes) = blk(dtype);
        let scratch = |b: &VulkanBackend| {
            (
                b.alloc(mpad_max * k_ * 2, BufferUsage::Activations)
                    .unwrap(),
                b.alloc(n_ * (k_ / belem) * bbytes, BufferUsage::Weights)
                    .unwrap(),
                b.alloc(mpad_max * n_ * 4, BufferUsage::Activations)
                    .unwrap(),
            )
        };
        let s32 = scratch(&be32);
        let s64 = scratch(&be64);
        for &m in ms {
            for (tile_label, be_, (a16, w, c), bm) in
                [("BM32", &be32, &s32, 32.0), ("BM64", &be64, &s64, 64.0)]
            {
                let rec = be_.recorder().unwrap();
                rec.matmul_native_f16a(
                    dtype,
                    a16.as_ref(),
                    w.device_addr().unwrap(),
                    0,
                    c.as_ref(),
                    m,
                    k_,
                    n_,
                );
                rec.finish().unwrap(); // warmup (pipeline compile)
                let t0 = std::time::Instant::now();
                let rec = be_.recorder().unwrap();
                for _ in 0..reps {
                    rec.matmul_native_f16a(
                        dtype,
                        a16.as_ref(),
                        w.device_addr().unwrap(),
                        0,
                        c.as_ref(),
                        m,
                        k_,
                        n_,
                    );
                }
                rec.finish().unwrap();
                let us = t0.elapsed().as_micros() as f64 / reps as f64;
                let flops = 2.0 * m as f64 * k_ as f64 * n_ as f64;
                let tflops = flops / (us * 1e-6) / 1e12;
                let fill = m as f64 / bm;
                println!(
                    "[n128_ag {label:>10} k={k_} n={n_:>6}] m={m:3} tile={tile_label}: {us:7.1} us  ({tflops:5.2} TFLOP/s, fill={fill:4.0}%)",
                    fill = fill * 100.0,
                );
            }
        }
    }

    // sk_ag family (matmul_native_splitk, a_is_f16=true): narrow-N deep-k shapes — down/o/kv-proj.
    let sk_shapes: &[(&str, infr_core::DType, usize, usize)] = &[
        ("down", infr_core::DType::Q4K, 9216, 2560),
        ("attn_out", infr_core::DType::Q8_0, 4096, 2560),
        ("kv", infr_core::DType::Q5K, 2560, 4096),
        ("q_proj", infr_core::DType::Q4K, 2560, 8192),
        ("o_small", infr_core::DType::Q6K, 2560, 1024),
    ];
    let splits = 8usize;
    for &(label, dtype, k, n) in sk_shapes {
        let (k_, n_) = (k, n);
        let mpad_max = 64usize;
        let (belem, bbytes) = blk(dtype);
        let scratch = |b: &VulkanBackend| {
            (
                b.alloc(mpad_max * k_ * 2, BufferUsage::Activations)
                    .unwrap(),
                b.alloc(n_ * (k_ / belem) * bbytes, BufferUsage::Weights)
                    .unwrap(),
                b.alloc(mpad_max * n_ * 4, BufferUsage::Activations)
                    .unwrap(),
                b.alloc(splits * mpad_max * n_ * 4, BufferUsage::Activations)
                    .unwrap(),
            )
        };
        let s32 = scratch(&be32);
        let s64 = scratch(&be64);
        for &m in ms {
            for (tile_label, be_, (a16, w, c, partials), bm) in
                [("BM32", &be32, &s32, 32.0), ("BM64", &be64, &s64, 64.0)]
            {
                let rec = be_.recorder().unwrap();
                rec.matmul_native_splitk(
                    dtype,
                    a16.as_ref(),
                    w.device_addr().unwrap(),
                    0,
                    partials.as_ref(),
                    c.as_ref(),
                    m,
                    k_,
                    n_,
                    splits,
                    true,
                );
                rec.finish().unwrap(); // warmup (pipeline compile)
                let t0 = std::time::Instant::now();
                let rec = be_.recorder().unwrap();
                for _ in 0..reps {
                    rec.matmul_native_splitk(
                        dtype,
                        a16.as_ref(),
                        w.device_addr().unwrap(),
                        0,
                        partials.as_ref(),
                        c.as_ref(),
                        m,
                        k_,
                        n_,
                        splits,
                        true,
                    );
                }
                rec.finish().unwrap();
                let us = t0.elapsed().as_micros() as f64 / reps as f64;
                let flops = 2.0 * m as f64 * k_ as f64 * n_ as f64;
                let tflops = flops / (us * 1e-6) / 1e12;
                let fill = m as f64 / bm;
                println!(
                    "[sk_ag {label:>10} k={k_:>5} n={n_:>5}] m={m:3} tile={tile_label}: {us:7.1} us  ({tflops:5.2} TFLOP/s, fill={fill:4.0}%)",
                    fill = fill * 100.0,
                );
            }
        }
    }
}

/// BM=16 vs BM=32 vs BM=64 dense A_GLOBAL warp-GEMM row tile at the SAME real qwen35-4B verify
/// shapes as `dense_small_m_row_tile_bench` (n128_ag family only — BM16 has no sk_ag variant, see
/// that bench's doc). BM=16 is one coopmat M-frag (the tiling floor): at m<=16 it halves BM=32's
/// remaining masked waste again, but it also doubles WARPS_N (halves WN) to keep all 8 launched
/// warps mapped to a valid tile — fewer, thinner accumulator frags per warp. Sweeps the recorder's
/// REAL gate (`DENSE_SMALL_TILE_MAX_M16` in recorder.rs) via `INFR_NO_BM16` (forces BM=32 within
/// the small-tile band) / `INFR_NO_SMALL_BM` (forces BM=64), so — like `dense_small_m_row_tile_bench`
/// — this exercises the same code path production uses. Finds the m where BM16 stops beating BM32
/// so `DENSE_SMALL_TILE_MAX_M16` can be set to the measured crossover.
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn bm16_crossover_bench() {
    // One backend per tile tier: BM16 = the defaults, BM32 = `INFR_NO_BM16`
    // (`kernels.vulkan.bm16 = false`), BM64 = `INFR_NO_SMALL_BM` (`small_bm = false`).
    let be16 = be_with(|_| {});
    let be32 = be_with(|v| v.bm16 = false);
    let be64 = be_with(|v| v.small_bm = false);
    let reps = 30usize;
    let ms: &[usize] = &[4, 6, 8, 12, 16, 24, 32];

    fn blk(dtype: infr_core::DType) -> (usize, usize) {
        use infr_core::DType::*;
        match dtype {
            Q4K => (256, 144),
            Q5K => (256, 176),
            Q6K => (256, 210),
            Q8_0 => (32, 34),
            _ => unreachable!("bench dtype not covered"),
        }
    }

    // n128_ag family (matmul_native_f16a) at the qwen35-4B-UD-Q4_K_XL verify's dominant projection
    // shapes captured via INFR_MTP=1 INFR_PROF_OP_SHAPES=1 — attn qkv/o (deep-k narrow-n, routed
    // through sk_ag / matmul_native_splitk in production so NOT reachable by BM16, listed here only
    // via the wide gate_up/vocab_head n128_ag shapes that ARE reachable).
    let n128_shapes: &[(&str, infr_core::DType, usize, usize)] = &[
        ("gate_up", infr_core::DType::Q4K, 2560, 18432),
        ("vocab_head", infr_core::DType::Q6K, 2560, 248320),
    ];
    for &(label, dtype, k, n) in n128_shapes {
        let (k_, n_) = (k, n);
        let mpad_max = 64usize;
        let (belem, bbytes) = blk(dtype);
        let scratch = |b: &VulkanBackend| {
            (
                b.alloc(mpad_max * k_ * 2, BufferUsage::Activations)
                    .unwrap(),
                b.alloc(n_ * (k_ / belem) * bbytes, BufferUsage::Weights)
                    .unwrap(),
                b.alloc(mpad_max * n_ * 4, BufferUsage::Activations)
                    .unwrap(),
            )
        };
        let s16 = scratch(&be16);
        let s32 = scratch(&be32);
        let s64 = scratch(&be64);
        for &m in ms {
            for (tile_label, be_, (a16, w, c), bm) in [
                ("BM16", &be16, &s16, 16.0),
                ("BM32", &be32, &s32, 32.0),
                ("BM64", &be64, &s64, 64.0),
            ] {
                let rec = be_.recorder().unwrap();
                rec.matmul_native_f16a(
                    dtype,
                    a16.as_ref(),
                    w.device_addr().unwrap(),
                    0,
                    c.as_ref(),
                    m,
                    k_,
                    n_,
                );
                rec.finish().unwrap(); // warmup (pipeline compile)
                let t0 = std::time::Instant::now();
                let rec = be_.recorder().unwrap();
                for _ in 0..reps {
                    rec.matmul_native_f16a(
                        dtype,
                        a16.as_ref(),
                        w.device_addr().unwrap(),
                        0,
                        c.as_ref(),
                        m,
                        k_,
                        n_,
                    );
                }
                rec.finish().unwrap();
                let us = t0.elapsed().as_micros() as f64 / reps as f64;
                let flops = 2.0 * m as f64 * k_ as f64 * n_ as f64;
                let tflops = flops / (us * 1e-6) / 1e12;
                let fill = m as f64 / bm;
                println!(
                    "[n128_ag {label:>10} k={k_} n={n_:>6}] m={m:3} tile={tile_label}: {us:7.1} us  ({tflops:5.2} TFLOP/s, fill={fill:4.0}%)",
                    fill = fill * 100.0,
                );
            }
        }
    }
}

// REAL production routing distributions captured via `INFR_MOE_COUNTS_DEBUG=1
// INFR_MOE_COUNTS_DUMP=1 infr bench <model> -p 512 -n 0 -r 1 --ngl 0` (CPU reference path, whose
// top-k routing is bit-identical to the GPU's) at pp512 on the bench's synthetic `i%100` prompt.
// qwen3-30B-A3B: 128 experts, avg=32/expert. qwen3.6-MoE: 256 experts, avg=16/expert. Both are
// HEAVILY skewed (a couple of hot experts near the `rows`=511 ceiling, a long tail of near-empty
// ones) — nothing like the mean-balanced counts `moe_expert_row_tile_bench` sweeps.
#[rustfmt::skip]
const COUNTS_30B: [u32; 128] = [510, 0, 367, 0, 10, 0, 65, 12, 10, 0, 0, 290, 1, 413, 24, 0, 0, 16, 0, 3, 0, 1, 0, 1, 0, 0, 13, 1, 0, 0, 0, 167, 0, 0, 27, 0, 0, 0, 0, 42, 0, 0, 1, 0, 21, 0, 1, 0, 0, 23, 261, 1, 0, 2, 0, 100, 13, 1, 0, 0, 0, 1, 0, 0, 11, 0, 0, 0, 0, 21, 1, 0, 0, 0, 0, 0, 0, 0, 10, 85, 0, 4, 111, 0, 3, 472, 0, 0, 0, 0, 344, 3, 10, 0, 0, 28, 0, 0, 0, 1, 10, 0, 0, 0, 0, 0, 0, 61, 423, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0];
#[rustfmt::skip]
const COUNTS_36MOE: [u32; 256] = [0, 11, 0, 0, 0, 0, 46, 5, 2, 25, 0, 2, 6, 0, 0, 20, 0, 2, 8, 0, 0, 0, 0, 0, 1, 0, 68, 0, 0, 0, 0, 2, 0, 32, 0, 0, 32, 14, 1, 0, 39, 138, 0, 66, 42, 3, 16, 0, 26, 0, 0, 0, 1, 1, 0, 6, 17, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 21, 0, 26, 191, 0, 2, 0, 2, 0, 0, 15, 3, 0, 0, 13, 21, 0, 3, 1, 0, 0, 0, 28, 0, 0, 0, 173, 0, 0, 0, 11, 413, 44, 1, 5, 0, 0, 0, 139, 0, 0, 0, 0, 81, 1, 0, 0, 0, 13, 0, 4, 0, 249, 9, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 7, 0, 18, 7, 31, 0, 0, 0, 22, 258, 79, 0, 1, 28, 7, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 17, 28, 0, 2, 4, 0, 0, 0, 44, 0, 347, 0, 15, 0, 6, 6, 0, 0, 16, 0, 3, 37, 0, 9, 0, 10, 22, 0, 0, 0, 0, 0, 0, 0, 0, 4, 34, 0, 0, 0, 0, 10, 34, 0, 0, 0, 0, 0, 0, 1, 28, 5, 1, 0, 22, 1, 0, 33, 0, 204, 53, 33, 137, 0, 0, 4, 131, 8, 0, 2, 0, 0, 0, 0, 0, 3, 33, 0, 0, 52, 0, 0, 9, 6, 0, 5, 0, 43, 28, 3, 0, 0, 0, 0, 0, 0, 0];

/// BM=32 vs BM=64 (gate_up AND down) against REAL (heavily skewed) pp512 routing distributions
/// captured from `infr bench <model> -p 512 -n 0 --ngl 0` with `INFR_MOE_COUNTS_DEBUG=1
/// INFR_MOE_COUNTS_DUMP=1` (CPU reference, bit-identical top-k to the GPU router) — see
/// `COUNTS_30B`/`COUNTS_36MOE` above. `moe_expert_row_tile_bench`'s BM=32 vs BM=64 crossover
/// (`MOE_EXPERT_SMALL_TILE_AVG_ROWS` in recorder.rs) was picked against a MEAN-BALANCED synthetic
/// distribution; production routing is nowhere near balanced (a couple of hot experts absorb most
/// of a chunk's rows, most experts get single-digit or zero rows), so this bench re-runs that same
/// tile question against the REAL shape instead — CONFIRMS the shipped threshold is still right:
/// BM=32 wins both gate_up and down at qwen3.6-MoE's real avg~16, BM=64 wins down (clearly, ~26%)
/// at qwen3-30B-A3B's real avg~32.
///
/// Two more aggressive levers were tried against this same real-skew data and REJECTED:
/// - A BM=16 tile: lost to both 32 and 64 in every case (extra per-workgroup fixed cost from
///   twice the real tiles outweighs the smaller masked-waste bound) — never wired up as a shipped
///   kernel variant.
/// - A separate, higher BM=32 threshold for gate_up (deep K, more BLK=32 loop iterations to
///   amortize fixed cost over) vs down (shallow K): gate_up's BM32-vs-BM64 isolated-dispatch delta
///   at qwen3-30B-A3B's real avg~32 was NOT a stable win (ranged +4% to -25% across repeated runs
///   of this same bench — noise-dominated, no clear direction), and an end-to-end interleaved
///   `infr bench` pp512 A/B with the split threshold wired in showed no measurable difference from
///   baseline (~3040 t/s either way). Not shipped — the shared threshold is already
///   near-optimal for both stages at the shapes that matter here.
///
/// `n_used_probe` forces a specific tile (`matmul_mmq_experts`' avg-rows heuristic only steers
/// kernel selection, never touches counts/offsets/data) into the BM=32 vs BM=64 buckets set by
/// `MOE_EXPERT_SMALL_TILE_AVG_ROWS`.
#[allow(clippy::type_complexity)]
#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn moe_expert_row_tile_bench_real_skew() {
    let be = VulkanBackend::new().unwrap();
    let reps = 30usize;

    // (label, ne, gate/up nff, gate/up dtype, down dtype, n_expert, tokens, counts,
    //  probe32 = an n_used value that lands avg_rows comfortably inside BOTH thresholds' BM=32
    //  bucket for THIS n_expert/tokens combo, so the SAME probe exercises BM=32 for both stages)
    let cases: &[(
        &str,
        usize,
        usize,
        infr_core::DType,
        infr_core::DType,
        usize,
        usize,
        &[u32],
        usize,
    )] = &[
        (
            "qwen3-30B-A3B avg~32",
            2048,
            768,
            infr_core::DType::Q4K,
            infr_core::DType::Q6K,
            128,
            511,
            &COUNTS_30B,
            5,
        ),
        (
            "qwen3.6-moe avg~16",
            2048,
            512,
            infr_core::DType::Q4K,
            infr_core::DType::Q5K,
            256,
            511,
            &COUNTS_36MOE,
            8,
        ),
    ];

    for &(label, ne, nff, gdt, ddt, n_expert, tokens, counts_v, probe32) in cases {
        let n_pairs: usize = counts_v.iter().map(|&c| c as usize).sum();
        let npad = n_pairs.div_ceil(64) * 64 + 64;

        let qa = be.alloc(npad * ne, BufferUsage::Activations).unwrap();
        let dact = be
            .alloc(npad * (ne / 32) * 2, BufferUsage::Activations)
            .unwrap();
        let sact = be
            .alloc(npad * (ne / 32) * 2, BufferUsage::Activations)
            .unwrap();
        let gu_c = be
            .alloc(npad * (2 * nff) * 4, BufferUsage::Activations)
            .unwrap();
        let gu_w = be
            .alloc(
                n_expert * (2 * nff) * (ne / 256) * 144,
                BufferUsage::Weights,
            )
            .unwrap();

        let dqa = be.alloc(npad * nff, BufferUsage::Activations).unwrap();
        let dda = be
            .alloc(npad * (nff / 32) * 2, BufferUsage::Activations)
            .unwrap();
        let dsa = be
            .alloc(npad * (nff / 32) * 2, BufferUsage::Activations)
            .unwrap();
        let down_c = be.alloc(npad * ne * 4, BufferUsage::Activations).unwrap();
        let down_w = be
            .alloc(
                n_expert * ne * (nff / 256).max(1) * 176,
                BufferUsage::Weights,
            )
            .unwrap();

        let mut offsets_v = vec![0u32; n_expert];
        let mut acc = 0u32;
        for e in 0..n_expert {
            offsets_v[e] = acc;
            acc += counts_v[e];
        }
        let counts = be.alloc(n_expert * 4, BufferUsage::Activations).unwrap();
        let offsets = be.alloc(n_expert * 4, BufferUsage::Activations).unwrap();
        be.upload(counts.as_ref(), bytemuck::cast_slice(counts_v))
            .unwrap();
        be.upload(offsets.as_ref(), bytemuck::cast_slice(&offsets_v))
            .unwrap();

        // n_used_probe values chosen so avg_rows=tokens*probe/n_expert lands comfortably under
        // MOE_EXPERT_SMALL_TILE_AVG_ROWS (BM32 bucket) regardless of the real n_used (probe32 is
        // NOT the model's real n_used=8 for the 30B case — it's an artificial probe forcing BM32
        // on that distribution for comparison, since its real avg~32 already selects BM64).
        for (tile_label, n_used_probe) in [("BM32", probe32), ("BM64", n_expert * 100)] {
            let rec = be.recorder().unwrap();
            rec.matmul_mmq_experts(
                gdt,
                "bench_gateup",
                qa.as_ref(),
                dact.as_ref(),
                Some(sact.as_ref()),
                gu_w.as_ref(),
                0,
                ne,
                counts.as_ref(),
                offsets.as_ref(),
                gu_c.as_ref(),
                tokens,
                ne,
                2 * nff,
                n_expert,
                n_used_probe,
            );
            rec.finish().unwrap();
            let t0 = std::time::Instant::now();
            let rec = be.recorder().unwrap();
            for _ in 0..reps {
                rec.matmul_mmq_experts(
                    gdt,
                    "bench_gateup",
                    qa.as_ref(),
                    dact.as_ref(),
                    Some(sact.as_ref()),
                    gu_w.as_ref(),
                    0,
                    ne,
                    counts.as_ref(),
                    offsets.as_ref(),
                    gu_c.as_ref(),
                    tokens,
                    ne,
                    2 * nff,
                    n_expert,
                    n_used_probe,
                );
            }
            rec.finish().unwrap();
            let us = t0.elapsed().as_micros() as f64 / reps as f64;
            let flops = 2.0 * (n_pairs as f64) * (ne as f64) * (2.0 * nff as f64);
            let tflops = flops / (us * 1e-6) / 1e12;
            println!("[gate_up {label:>20}] tile={tile_label}: {us:8.1} us  ({tflops:5.2} TFLOP/s useful)");
        }

        for (tile_label, n_used_probe) in [("BM32", probe32), ("BM64", n_expert * 100)] {
            let sact_d: Option<&dyn Buffer> = if matches!(ddt, infr_core::DType::Q5K) {
                Some(dsa.as_ref())
            } else {
                None
            };
            let rec = be.recorder().unwrap();
            rec.matmul_mmq_experts(
                ddt,
                "bench_down",
                dqa.as_ref(),
                dda.as_ref(),
                sact_d,
                down_w.as_ref(),
                0,
                nff,
                counts.as_ref(),
                offsets.as_ref(),
                down_c.as_ref(),
                tokens,
                nff,
                ne,
                n_expert,
                n_used_probe,
            );
            rec.finish().unwrap();
            let t0 = std::time::Instant::now();
            let rec = be.recorder().unwrap();
            for _ in 0..reps {
                rec.matmul_mmq_experts(
                    ddt,
                    "bench_down",
                    dqa.as_ref(),
                    dda.as_ref(),
                    sact_d,
                    down_w.as_ref(),
                    0,
                    nff,
                    counts.as_ref(),
                    offsets.as_ref(),
                    down_c.as_ref(),
                    tokens,
                    nff,
                    ne,
                    n_expert,
                    n_used_probe,
                );
            }
            rec.finish().unwrap();
            let us = t0.elapsed().as_micros() as f64 / reps as f64;
            let flops = 2.0 * (n_pairs as f64) * (nff as f64) * (ne as f64);
            let tflops = flops / (us * 1e-6) / 1e12;
            println!("[down    {label:>20}] tile={tile_label}: {us:8.1} us  ({tflops:5.2} TFLOP/s useful)");
        }
    }
}
