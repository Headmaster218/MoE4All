//! Milestone 1 smoke test — runs on real hardware (not a unit test).
//!
//! Dispatches a WGSL f32 matmul on the GPU and compares against a CPU reference.
//! Asserts max relative error < 1e-3.
//!
//! Run with:
//!   cargo run -p infr-vulkan --example smoke

use std::time::Instant;

use infr_core::Backend;
use infr_vulkan::VulkanBackend;

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn main() -> infr_core::Result<()> {
    // ── init Vulkan backend ───────────────────────────────────────────────────
    let backend = VulkanBackend::new()?;
    let caps = backend.capabilities();
    println!("=== Vulkan device: {} ===", caps.name);
    println!("    f16={} coop_matrix={}", caps.f16, caps.f16_coopmat(),);
    println!();

    // ── problem size ──────────────────────────────────────────────────────────
    let (m, k, n) = (64usize, 64usize, 64usize);

    // Fill A and B with predictable values (row/col index scaled small).
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

    // ── CPU reference ─────────────────────────────────────────────────────────
    let t_cpu = Instant::now();
    let c_ref = cpu_matmul(&a, &b, m, k, n);
    let cpu_ms = t_cpu.elapsed().as_millis();
    println!("CPU matmul {m}×{k}×{n} in {cpu_ms}ms");

    // ── GPU dispatch ──────────────────────────────────────────────────────────
    // First call compiles WGSL→SPIR-V via naga (cached for subsequent calls).
    let t_gpu = Instant::now();
    let c_gpu = backend.matmul_f32(&a, &b, m, k, n)?;
    let gpu_ms = t_gpu.elapsed().as_millis();
    println!("GPU matmul {m}×{k}×{n} in {gpu_ms}ms (includes pipeline create + upload + dispatch + download)");

    // ── validate ──────────────────────────────────────────────────────────────
    assert_eq!(c_gpu.len(), m * n, "output length mismatch");

    let max_abs_err = c_gpu
        .iter()
        .zip(c_ref.iter())
        .map(|(g, r)| (*g - r).abs())
        .fold(0.0f32, f32::max);
    let max_ref = c_ref.iter().map(|r| r.abs()).fold(0.0f32, f32::max);
    let max_rel_err = if max_ref > 1e-9 {
        max_abs_err / max_ref
    } else {
        max_abs_err
    };

    println!();
    println!("max absolute error : {max_abs_err:.3e}");
    println!("max relative error : {max_rel_err:.3e}  (threshold 1e-3)");

    assert!(
        max_rel_err < 1e-3,
        "SMOKE FAIL — max relative error {max_rel_err:.3e} exceeds 1e-3"
    );

    println!();
    println!("SMOKE OK");

    // TODO(sonnet): coop-matrix f16 matmul variant gated on caps.f16_coopmat
    // Leave as stretch goal once the naive f32 path is verified.

    Ok(())
}
