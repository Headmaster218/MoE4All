//! Serialized micro-probe of the two kernels that dominate Qwen4's single-row decode:
//! `deltanet` (the gated-DeltaNet recurrence) and `rmsnorm_gate_sigmoid` (the fused per-head
//! RMSNorm + sigmoid gate). Both show ~500-700 us per dispatch in production per-op profiling at
//! ANY context depth — 100x a comparable tiny GEMV — so this measures the isolated kernel to
//! separate "the kernel really costs this" from "the production environment stalls it".
//!
//! Arms:
//!   * `rows` swept 1 -> 64: the recurrent state is 1 MiB and is read+written by every dispatch, so
//!     a state-bandwidth-bound kernel barely moves with rows.
//!   * `state` in VRAM (`Activations`) vs in system RAM (`Readback`, the tier `INFR_KV_OVERFLOW`
//!     spills KV/state into): quantifies the PCIe penalty directly.
//!
//! A WAW hazard on the output buffer serializes the repeated dispatches, so wall/REPS is the serial
//! per-dispatch cost. Run:
//!   cargo test -p infr-vulkan --test recurrent_kernel_probe -- --ignored --nocapture

use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

const REPS: usize = 200;

fn time(be: &VulkanBackend, f: impl Fn(&infr_vulkan::Recorder)) -> f64 {
    let rec = be.recorder().unwrap();
    f(&rec);
    rec.finish().unwrap();
    let t0 = std::time::Instant::now();
    let rec = be.recorder().unwrap();
    for _ in 0..REPS {
        f(&rec);
    }
    rec.finish().unwrap();
    t0.elapsed().as_micros() as f64 / REPS as f64
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-probe)"]
fn deltanet_single_row_probe() {
    let (nv, nk, kd, vd) = (16usize, 16usize, 128usize, 128usize);
    println!(
        "\n== deltanet (nv={nv} nk={nk} kd={kd} vd={vd}, state = {} KiB) ==",
        nv * kd * vd * 4 / 1024
    );
    for rows in [1usize, 8, 64] {
        for host_state in [false, true] {
            let be = VulkanBackend::new().unwrap();
            let q = be.alloc(rows * nk * kd * 4, BufferUsage::Activations).unwrap();
            let k = be.alloc(rows * nk * kd * 4, BufferUsage::Activations).unwrap();
            let v = be.alloc(rows * nv * vd * 4, BufferUsage::Activations).unwrap();
            let blog = be.alloc(rows * nv * 4, BufferUsage::Activations).unwrap();
            let alpha = be.alloc(rows * nv * 4, BufferUsage::Activations).unwrap();
            let acoef = be.alloc(nv * 4, BufferUsage::Activations).unwrap();
            let dtbias = be.alloc(nv * 4, BufferUsage::Activations).unwrap();
            let state = be
                .alloc(
                    nv * kd * vd * 4,
                    if host_state {
                        BufferUsage::Readback
                    } else {
                        BufferUsage::Activations
                    },
                )
                .unwrap();
            let out = be.alloc(rows * nv * vd * 4, BufferUsage::Activations).unwrap();
            let us = time(&be, |rec| {
                rec.deltanet(
                    q.as_ref(),
                    k.as_ref(),
                    v.as_ref(),
                    blog.as_ref(),
                    alpha.as_ref(),
                    acoef.as_ref(),
                    dtbias.as_ref(),
                    state.as_ref(),
                    out.as_ref(),
                    rows,
                    nv,
                    nk,
                    kd,
                    vd,
                    1e-6,
                )
            });
            println!(
                "  rows={rows:<3} state={:<12} {us:>8.1} us/dispatch",
                if host_state { "SYSTEM RAM" } else { "VRAM" }
            );
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-probe)"]
fn gated_rmsnorm_single_row_probe() {
    // Production shape: Op::GatedRmsNorm with n_head = nv = 16, head_dim = vd = 128 -> dim 2048.
    for dim in [2048usize, 2560] {
        for rows in [1usize, 64] {
            let be = VulkanBackend::new().unwrap();
            let x = be.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
            let w = be.alloc(dim * 4, BufferUsage::Activations).unwrap();
            let z = be.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
            let y = be.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
            let us = time(&be, |rec| {
                rec.rmsnorm_gate_sigmoid(
                    x.as_ref(),
                    w.as_ref(),
                    z.as_ref(),
                    y.as_ref(),
                    rows,
                    dim,
                    1e-6,
                )
            });
            println!("  rmsnorm_gate_sigmoid rows={rows:<3} dim={dim:<5} {us:>8.1} us/dispatch");
        }
    }
}
