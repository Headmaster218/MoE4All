//! Probe: achieved memory bandwidth of the quantized GEMV kernels at decode shapes (m=1).
//! Temporary evidence for kernel-efficiency work — run with
//! `cargo test -p infr-metal --release --test gemv_bw -- --ignored --nocapture`.
#![cfg(target_os = "macos")]

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc};
use infr_metal::MetalBackend;

fn lcg_bytes(mut seed: u32, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as u8
        })
        .collect()
}

fn synth_q4k(n_elem: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 144];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());
        blk[2..4].copy_from_slice(&half::f16::from_f32(0.10).to_le_bytes());
        blk[4..144].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 140));
        out.extend_from_slice(&blk);
    }
    out
}

fn synth_q6k(n_elem: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 210];
        blk[0..208].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 208));
        blk[208..210].copy_from_slice(&half::f16::from_f32(0.03).to_le_bytes());
        out.extend_from_slice(&blk);
    }
    out
}

fn synth_q5k(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0);
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 176];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());
        blk[2..4].copy_from_slice(&half::f16::from_f32(0.10).to_le_bytes());
        blk[4..176].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 172));
        out.extend_from_slice(&blk);
    }
    out
}

fn bench(dtype: DType, wbytes: Vec<u8>, in_f: usize, out_f: usize, bpw: f64, label: &str) {
    let be = MetalBackend::new().unwrap();
    let m = 1usize;
    let xs: Vec<f32> = (0..in_f).map(|i| (i % 7) as f32 * 0.01).collect();

    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], dtype));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });

    let xb = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
    let wb = be.alloc(wbytes.len(), BufferUsage::Weights).unwrap();
    be.upload(wb.as_ref(), &wbytes).unwrap();
    let ob = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();

    let mut binds = Bindings::new();
    binds.bind(x, xb.as_ref());
    binds.bind(w, wb.as_ref());
    binds.bind(dst, ob.as_ref());
    let plan = be.compile(&g).unwrap();

    // First execute builds + caches the factored weight; then timed reps.
    be.execute(plan.as_ref(), &binds).unwrap();
    let reps = 30;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        be.execute(plan.as_ref(), &binds).unwrap();
    }
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    let bytes = (out_f * in_f) as f64 * bpw / 8.0;
    println!(
        "{label}: {out_f}x{in_f} -> {:.3} ms/dispatch, {:.1} GB/s (stream {:.1} MB)",
        dt * 1e3,
        bytes / dt / 1e9,
        bytes / 1e6
    );
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn gemv_bandwidth() {
    // lm_head shape (Qwen3-0.6B): 151936x1024 Q6K -> quik6, 8.125 bpw
    bench(
        DType::Q6K,
        synth_q6k(151936 * 1024, 1),
        1024,
        151936,
        8.125,
        "quik6",
    );
    // Same shape as Q4K -> quik4, 6.125 bpw
    bench(
        DType::Q4K,
        synth_q4k(151936 * 1024, 2),
        1024,
        151936,
        6.125,
        "quik4",
    );
    // FFN shape: 3072x1024 Q4K (small stream: launch overhead visible)
    bench(
        DType::Q4K,
        synth_q4k(3072 * 1024, 3),
        1024,
        3072,
        6.125,
        "quik4-ffn",
    );
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn q5k_swar_probe() {
    let wq5k = synth_q5k(65536 * 1152, 4);
    bench_chained(DType::Q5K, &wq5k, 1152, 65536, 5.5, "q5k head");
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn f16_native_probe() {
    // One EnvGuard for the whole probe: serializes against every other test in this binary
    // that reads these knobs, and restores the caller's values on drop (see
    // infr_core::test_env — a bare set_var/remove_var pair races and leaks).
    #[allow(unused_mut)]
    let mut env = infr_core::test_env::EnvGuard::new();
    let (in_f, out_f) = (1152usize, 65536usize);
    let wf: Vec<f32> = (0..out_f * in_f).map(|i| (i % 13) as f32 * 0.01).collect();
    let w16: Vec<u8> = wf
        .into_iter()
        .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
        .collect();

    env.set("INFR_METAL_NO_F16_NATIVE", "1");
    bench_chained(DType::F16, &w16, in_f, out_f, 32.0, "f16 cached-f32");
    env.unset("INFR_METAL_NO_F16_NATIVE");
    bench_chained(DType::F16, &w16, in_f, out_f, 16.0, "f16 native");
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn bf16_native_probe() {
    // One EnvGuard for the whole probe: serializes against every other test in this binary
    // that reads these knobs, and restores the caller's values on drop (see
    // infr_core::test_env — a bare set_var/remove_var pair races and leaks).
    #[allow(unused_mut)]
    let mut env = infr_core::test_env::EnvGuard::new();
    let (in_f, out_f) = (1152usize, 65536usize);
    let w16: Vec<u8> = (0..out_f * in_f)
        .flat_map(|i| {
            let v = (i % 13) as f32 * 0.01;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();

    env.set("INFR_METAL_NO_BF16_NATIVE", "1");
    bench_chained(DType::Bf16, &w16, in_f, out_f, 32.0, "bf16 cached-f32");
    env.unset("INFR_METAL_NO_BF16_NATIVE");
    bench_chained(DType::Bf16, &w16, in_f, out_f, 16.0, "bf16 native");
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn bf16_rt_probe() {
    // One EnvGuard for the whole probe: serializes against every other test in this binary
    // that reads these knobs, and restores the caller's values on drop (see
    // infr_core::test_env — a bare set_var/remove_var pair races and leaks).
    #[allow(unused_mut)]
    let mut env = infr_core::test_env::EnvGuard::new();
    let (in_f, out_f) = (1152usize, 8192usize);
    let w16: Vec<u8> = (0..out_f * in_f)
        .flat_map(|i| {
            let v = (i % 13) as f32 * 0.01;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();

    env.set("INFR_METAL_NO_BF16_CMM", "1");
    for m in [2usize, 4, 8, 16, 32] {
        env.set("INFR_METAL_NO_BF16_RT", "1");
        bench_chained_m(
            DType::Bf16,
            &w16,
            m,
            in_f,
            out_f,
            16.0 * m as f64,
            "bf16 native-gemv",
        );
        env.unset("INFR_METAL_NO_BF16_RT");
        bench_chained_m(
            DType::Bf16,
            &w16,
            m,
            in_f,
            out_f,
            16.0 * m.div_ceil(8) as f64,
            "bf16 rt",
        );
    }
    env.unset("INFR_METAL_NO_BF16_CMM");
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn bf16_cmm_probe() {
    // One EnvGuard for the whole probe: serializes against every other test in this binary
    // that reads these knobs, and restores the caller's values on drop (see
    // infr_core::test_env — a bare set_var/remove_var pair races and leaks).
    #[allow(unused_mut)]
    let mut env = infr_core::test_env::EnvGuard::new();
    let (in_f, out_f) = (1152usize, 8192usize);
    let w16: Vec<u8> = (0..out_f * in_f)
        .flat_map(|i| {
            let v = (i % 13) as f32 * 0.01;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();

    for m in [6usize, 8, 12, 16, 32] {
        env.set("INFR_METAL_NO_BF16_CMM", "1");
        bench_chained_m(
            DType::Bf16,
            &w16,
            m,
            in_f,
            out_f,
            16.0 * m.div_ceil(8) as f64,
            "bf16 rt",
        );
        env.unset("INFR_METAL_NO_BF16_CMM");
        bench_chained_m(DType::Bf16, &w16, m, in_f, out_f, 16.0, "bf16 cmm");
    }
}

fn bench_f32_cold(wbytes: &[u8], in_f: usize, out_f: usize, force_cache: bool, label: &str) {
    // One EnvGuard for the whole probe: serializes against every other test in this binary
    // that reads these knobs, and restores the caller's values on drop (see
    // infr_core::test_env — a bare set_var/remove_var pair races and leaks).
    #[allow(unused_mut)]
    let mut env = infr_core::test_env::EnvGuard::new();
    if force_cache {
        env.set("INFR_METAL_NO_F32_NATIVE", "1");
    } else {
        env.unset("INFR_METAL_NO_F32_NATIVE");
    }
    let be = MetalBackend::new().unwrap();
    let xs = vec![0.01f32; in_f];
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![1, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], DType::F32));
    let dst = g.output(TensorDesc::new(vec![1, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: 1,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let xb = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    let wb = be.alloc(wbytes.len(), BufferUsage::Weights).unwrap();
    let ob = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
    be.upload(wb.as_ref(), wbytes).unwrap();
    let mut bindings = Bindings::new();
    bindings.bind(x, xb.as_ref());
    bindings.bind(w, wb.as_ref());
    bindings.bind(dst, ob.as_ref());
    let plan = be.compile(&g).unwrap();

    let t0 = std::time::Instant::now();
    be.execute(plan.as_ref(), &bindings).unwrap();
    println!(
        "{label}: {:.3} ms cold execute",
        t0.elapsed().as_secs_f64() * 1e3
    );
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn f32_native_cold_probe() {
    // One EnvGuard for the whole probe: serializes against every other test in this binary
    // that reads these knobs, and restores the caller's values on drop (see
    // infr_core::test_env — a bare set_var/remove_var pair races and leaks).
    #[allow(unused_mut)]
    let mut env = infr_core::test_env::EnvGuard::new();
    let (in_f, out_f) = (1024usize, 16384usize);
    let wf: Vec<f32> = (0..out_f * in_f).map(|i| (i % 13) as f32 * 0.01).collect();
    let wbytes = bytemuck::cast_slice(&wf);
    bench_f32_cold(wbytes, in_f, out_f, true, "f32 cached copy");
    bench_f32_cold(wbytes, in_f, out_f, false, "f32 direct");
    env.unset("INFR_METAL_NO_F32_NATIVE");
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn f32_rt_probe() {
    // One EnvGuard for the whole probe: serializes against every other test in this binary
    // that reads these knobs, and restores the caller's values on drop (see
    // infr_core::test_env — a bare set_var/remove_var pair races and leaks).
    #[allow(unused_mut)]
    let mut env = infr_core::test_env::EnvGuard::new();
    let (in_f, out_f) = (1152usize, 8192usize);
    let w32: Vec<u8> = (0..out_f * in_f)
        .flat_map(|i| ((i % 13) as f32 * 0.01).to_le_bytes())
        .collect();

    for m in [2usize, 4, 8] {
        env.set("INFR_METAL_NO_F32_RT", "1");
        bench_chained_m(
            DType::F32,
            &w32,
            m,
            in_f,
            out_f,
            32.0 * m as f64,
            "f32 native-gemv",
        );
        env.unset("INFR_METAL_NO_F32_RT");
        bench_chained_m(DType::F32, &w32, m, in_f, out_f, 32.0, "f32 rt");
    }
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn f32_cmm_probe() {
    // One EnvGuard for the whole probe: serializes against every other test in this binary
    // that reads these knobs, and restores the caller's values on drop (see
    // infr_core::test_env — a bare set_var/remove_var pair races and leaks).
    #[allow(unused_mut)]
    let mut env = infr_core::test_env::EnvGuard::new();
    let (in_f, out_f) = (1152usize, 8192usize);
    let w32: Vec<u8> = (0..out_f * in_f)
        .flat_map(|i| ((i % 13) as f32 * 0.01).to_le_bytes())
        .collect();

    for m in [8usize, 12, 16, 32] {
        env.set("INFR_METAL_NO_F32_CMM", "1");
        let (fallback_bpw, fallback_label) = if m < 16 {
            (32.0, "f32 rt")
        } else {
            (32.0 * m as f64, "f32 native-gemv")
        };
        bench_chained_m(
            DType::F32,
            &w32,
            m,
            in_f,
            out_f,
            fallback_bpw,
            fallback_label,
        );
        env.unset("INFR_METAL_NO_F32_CMM");
        bench_chained_m(DType::F32, &w32, m, in_f, out_f, 32.0, "f32 cmm");
    }
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn f16_cmm_probe() {
    // One EnvGuard for the whole probe: serializes against every other test in this binary
    // that reads these knobs, and restores the caller's values on drop (see
    // infr_core::test_env — a bare set_var/remove_var pair races and leaks).
    #[allow(unused_mut)]
    let mut env = infr_core::test_env::EnvGuard::new();
    let (in_f, out_f) = (1152usize, 8192usize);
    let w16: Vec<u8> = (0..out_f * in_f)
        .flat_map(|i| half::f16::from_f32((i % 13) as f32 * 0.01).to_le_bytes())
        .collect();

    for m in [8usize, 12, 16, 32] {
        env.set("INFR_METAL_NO_F16_CMM", "1");
        let (fallback_bpw, fallback_label) = if m < 16 {
            (16.0 * m.div_ceil(8) as f64, "f16 rt")
        } else {
            (16.0 * m as f64, "f16 native-gemv")
        };
        bench_chained_m(
            DType::F16,
            &w16,
            m,
            in_f,
            out_f,
            fallback_bpw,
            fallback_label,
        );
        env.unset("INFR_METAL_NO_F16_CMM");
        bench_chained_m(DType::F16, &w16, m, in_f, out_f, 16.0, "f16 cmm");
    }
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn f16_rt_probe() {
    // One EnvGuard for the whole probe: serializes against every other test in this binary
    // that reads these knobs, and restores the caller's values on drop (see
    // infr_core::test_env — a bare set_var/remove_var pair races and leaks).
    #[allow(unused_mut)]
    let mut env = infr_core::test_env::EnvGuard::new();
    let (in_f, out_f) = (1152usize, 8192usize);
    let w16: Vec<u8> = (0..out_f * in_f)
        .flat_map(|i| half::f16::from_f32((i % 13) as f32 * 0.01).to_le_bytes())
        .collect();

    for m in [2usize, 4, 8] {
        env.set("INFR_METAL_NO_F16_RT", "1");
        bench_chained_m(
            DType::F16,
            &w16,
            m,
            in_f,
            out_f,
            16.0 * m as f64,
            "f16 native-gemv",
        );
        env.unset("INFR_METAL_NO_F16_RT");
        bench_chained_m(DType::F16, &w16, m, in_f, out_f, 16.0, "f16 rt");
    }
}

fn synth_q5_0(n_elem: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 32) {
        let mut blk = vec![0u8; 22];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.04).to_le_bytes());
        blk[2..22].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 20));
        out.extend_from_slice(&blk);
    }
    out
}

fn quantize_q8_0(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in w.chunks(32) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for &v in blk {
            let q = if d > 0.0 {
                (v / d).round().clamp(-127.0, 127.0) as i8
            } else {
                0
            };
            out.push(q as u8);
        }
    }
    out
}

fn synth_iq4nl(n_elem: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 32) {
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.004).to_le_bytes());
        blk[2..18].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 16));
        out.extend_from_slice(&blk);
    }
    out
}

// Chained bench: K identical GEMVs over K DISTINCT weight copies in ONE graph (one command
// buffer) — the per-cb commit+wait overhead amortizes away and the number reflects the
// in-decode-chain cost. Distinct weights so the stream is not cache-resident.
fn bench_chained_m(
    dtype: DType,
    wbytes: &[u8],
    m: usize,
    in_f: usize,
    out_f: usize,
    bpw: f64,
    label: &str,
) {
    let be = MetalBackend::new().unwrap();
    let k = 8usize;
    let xs: Vec<f32> = (0..m * in_f).map(|i| (i % 7) as f32 * 0.01).collect();

    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let ws: Vec<_> = (0..k)
        .map(|_| g.weight(TensorDesc::new(vec![out_f, in_f], dtype)))
        .collect();
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    for w in &ws {
        g.push(Op::Linear {
            x,
            weight: *w,
            dst,
            m: m as u32,
            in_f: in_f as u32,
            out_f: out_f as u32,
            w_off: 0,
        });
    }

    let xb = be.alloc(m * in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
    let mut wbs = Vec::new();
    for _ in 0..k {
        let wb = be.alloc(wbytes.len(), BufferUsage::Weights).unwrap();
        be.upload(wb.as_ref(), wbytes).unwrap();
        wbs.push(wb);
    }
    let ob = be.alloc(m * out_f * 4, BufferUsage::Activations).unwrap();

    let mut binds = Bindings::new();
    binds.bind(x, xb.as_ref());
    for (w, wb) in ws.iter().zip(&wbs) {
        binds.bind(*w, wb.as_ref());
    }
    binds.bind(dst, ob.as_ref());
    let plan = be.compile(&g).unwrap();
    be.execute(plan.as_ref(), &binds).unwrap();
    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        be.execute(plan.as_ref(), &binds).unwrap();
    }
    let dt = t0.elapsed().as_secs_f64() / (reps * k) as f64;
    let bytes = (out_f * in_f) as f64 * bpw / 8.0;
    println!(
        "{label}: m={m} {out_f}x{in_f} chained -> {:.3} ms/gemv, {:.1} GB/s (stream {:.1} MB)",
        dt * 1e3,
        bytes / dt / 1e9,
        bytes / 1e6
    );
}

fn bench_chained(dtype: DType, wbytes: &[u8], in_f: usize, out_f: usize, bpw: f64, label: &str) {
    bench_chained_m(dtype, wbytes, 1, in_f, out_f, bpw, label);
}

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn gemv_bandwidth_gemma_shapes() {
    // gemma3-1b decode shapes, chained (in-decode-chain cost, cb overhead amortized)
    let wq5 = synth_q5_0(6912 * 1152, 11);
    bench_chained(DType::Q5_0, &wq5, 1152, 6912, 5.5, "q5_0 gate/up");
    let wq6 = synth_q6k(6912 * 1152, 12);
    bench_chained(DType::Q6K, &wq6, 1152, 6912, 6.5625, "q6k down");
    let wq4 = synth_q4k(1024 * 1152, 13);
    bench_chained(DType::Q4K, &wq4, 1024, 1152, 4.5, "q4k o");
    let wq5q = synth_q5_0(1152 * 1024, 14);
    bench_chained(DType::Q5_0, &wq5q, 1152, 1024, 5.5, "q5_0 q");
    let wf: Vec<f32> = (0..(65536usize * 1152))
        .map(|i| (i % 13) as f32 * 0.01)
        .collect();
    let w8 = quantize_q8_0(&wf);
    bench_chained(DType::Q8_0, &w8, 1152, 65536, 8.5, "q8_0 head/4");
    let wiq4 = synth_iq4nl(6912 * 1152, 15);
    bench_chained(DType::Iq4Nl, &wiq4, 1152, 6912, 4.5, "iq4_nl gate/up");
    for m in 2..=8 {
        bench_chained_m(
            DType::Iq4Nl,
            &wiq4,
            m,
            1152,
            6912,
            4.5,
            "iq4_nl gate/up multirow",
        );
    }
    let wiq4rt = synth_iq4nl(6911 * 1152, 18);
    for m in 2..=8 {
        bench_chained_m(
            DType::Iq4Nl,
            &wiq4rt,
            m,
            1152,
            6911,
            4.5,
            "iq4_nl gate/up multirow rt",
        );
    }
    let wiq4d = synth_iq4nl(1152 * 6912, 17);
    bench_chained(DType::Iq4Nl, &wiq4d, 6912, 1152, 4.5, "iq4_nl down");
    let wiq4h = synth_iq4nl(65536 * 1152, 16);
    bench_chained(DType::Iq4Nl, &wiq4h, 1152, 65536, 4.5, "iq4_nl head");
}
