//! Per-device memory + PCIe bandwidth probe.
//!
//! Answers two questions the scheduler needs and sysfs cannot settle on its own:
//!   1. How fast can this card stream its OWN VRAM (the ceiling every decode kernel is
//!      measured against)?
//!   2. What is the real host<->device bandwidth on this slot, which is what a
//!      tensor-parallel / all-reduce design would have to live inside?
//!
//! sysfs reports the GPU endpoint's negotiated width, but a card with an on-board PCIe
//! switch advertises the GPU<->switch link (Gen4 x16) while the real bottleneck is the
//! switch<->CPU link. Only a byte-moving measurement settles it.
//!
//! Run: `cargo test -p infr-vulkan --release --test pcie_link_probe -- --ignored --nocapture`

use std::time::Instant;

use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

const MIB: usize = 1024 * 1024;
const BLOCK: usize = 16 * MIB;

fn gib_per_sec(bytes: usize, secs: f64) -> f64 {
    (bytes as f64 / secs) / (1u64 << 30) as f64
}

fn bench(iters: usize, bytes: usize, mut f: impl FnMut()) -> f64 {
    f(); // warmup
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    gib_per_sec(bytes * iters, t0.elapsed().as_secs_f64())
}

#[test]
#[ignore = "requires Vulkan GPUs; perf probe, not a correctness test"]
fn pcie_link_probe() {
    let devs = match VulkanBackend::enumerate_devices_from_env() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };
    for d in &devs {
        eprintln!("  Vulkan{}: {} [{}]", d.index, d.name, d.device_type);
    }

    let iters = 12usize;
    for d in &devs {
        // The CPU (llvmpipe) device has no VRAM and no PCIe link; it is only in the
        // enumeration so the seam can fall back to it. Skip it here.
        if !d.name.contains("RADV") && !d.name.contains("radeon") && !d.name.contains("AMD") {
            eprintln!("Vulkan{}: {} — skipping (not a RADV device)", d.index, d.name);
            continue;
        }
        let be = match VulkanBackend::new_on(d.index) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Vulkan{}: build failed: {e}", d.index);
                continue;
            }
        };

        eprintln!("Vulkan{}: building backend...", d.index);
        let staging = be
            .alloc_uninit(BLOCK, BufferUsage::Staging)
            .expect("alloc staging");
        eprintln!("  staging ok");
        let dev_a = be
            .alloc_uninit(BLOCK, BufferUsage::Weights)
            .expect("alloc weight a");
        eprintln!("  weights ok");
        let dev_b = be
            .alloc_uninit(BLOCK, BufferUsage::Weights)
            .expect("alloc weight b");
        let host_src = vec![0x5Au8; BLOCK];
        let mut host_dst = vec![0u8; BLOCK];

        // H2D: host Vec -> host-visible staging (memcpy), then staging -> VRAM (DMA).
        eprintln!("  bench: h2d staging memcpy");
        let h2d_stage = bench(iters, BLOCK, || {
            be.upload(staging.as_ref(), &host_src).expect("upload");
        });
        let h2d_dma = bench(iters, BLOCK, || {
            be.copy_buffer(staging.as_ref(), dev_a.as_ref(), BLOCK)
                .expect("copy h2d");
        });
        let h2d_total = bench(iters, BLOCK, || {
            be.upload(staging.as_ref(), &host_src).expect("upload");
            be.copy_buffer(staging.as_ref(), dev_a.as_ref(), BLOCK)
                .expect("copy h2d");
        });

        // D2H: VRAM -> staging (DMA), then staging -> host Vec.
        let d2h_dma = bench(iters, BLOCK, || {
            be.copy_buffer(dev_a.as_ref(), staging.as_ref(), BLOCK)
                .expect("copy d2h");
        });
        let d2h_total = bench(iters, BLOCK, || {
            be.copy_buffer(dev_a.as_ref(), staging.as_ref(), BLOCK)
                .expect("copy d2h");
            be.download(staging.as_ref(), &mut host_dst).expect("download");
        });

        // VRAM -> VRAM copy: reads BLOCK and writes BLOCK, so the effective traffic is 2*BLOCK.
        let vram_copy = bench(iters, BLOCK, || {
            be.copy_buffer(dev_a.as_ref(), dev_b.as_ref(), BLOCK)
                .expect("copy vram");
        });

        eprintln!(
            "\nVulkan{}: {}",
            d.index, d.name
        );
        eprintln!("  H2D staging memcpy   : {:6.2} GiB/s", h2d_stage);
        eprintln!("  H2D staging->VRAM DMA: {:6.2} GiB/s", h2d_dma);
        eprintln!("  H2D combined         : {:6.2} GiB/s", h2d_total);
        eprintln!("  D2H VRAM->staging DMA: {:6.2} GiB/s", d2h_dma);
        eprintln!("  D2H combined         : {:6.2} GiB/s", d2h_total);
        eprintln!(
            "  VRAM->VRAM copy      : {:6.2} GiB/s (effective r+w {:6.2})",
            vram_copy,
            vram_copy * 2.0
        );
    }
}
