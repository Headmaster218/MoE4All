//! D2H readback probe: the direct mapped-CPU-read paths (default) vs the DMA-into-imported-RAM
//! escape hatch (`INFR_D2H_DMA=1` — see `VulkanBackend::download_dma`).
//!
//! The DMA route has the copy engine WRITE the bytes into an ordinary, page-table-backed (cached)
//! host allocation imported with `VK_EXT_external_memory_host` (the pager's expert-cache
//! transport, reversed), instead of the CPU reading a mapped readback buffer directly. That wins
//! ONLY where the mapped reads are uncached — BAR-placed Readback classes on drivers that don't
//! expose cached sysmem types (observed ~25 MB/s on RX 7700 XT during the MTP stochastic-verify
//! work). On drivers that DO expose cached sysmem (the same card, current driver, ReBAR on) the
//! direct paths measure 14.4 GB/s mapped / 5.5 GB/s staged at 8 MiB while the DMA route pays a
//! ~1.5 ms one-shot submit on top of the transfer — hence the channel is opt-in, and THIS probe
//! is the A/B that decides which mode a given box should run.
//!
//! Both directions below go through the SAME `Backend` trait surface every other test in this
//! crate uses. Correctness is asserted against the same uploaded pattern on both paths.
//!
//! Run: `cargo test -p infr-vulkan --release --test d2h_dma_bench -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;
use std::sync::Mutex;
use std::time::Instant;

/// Same order of magnitude as the stochastic verify's worst logits download
/// (m=6 × 151936 × 4B ≈ 3.5 MiB), comfortably above the 256 KiB DMA threshold.
const BYTES: usize = 8 * 1024 * 1024;
const ITERS: usize = 20;

/// The knob is process-global env read at backend construction; serialize the tests that touch
/// it (cargo runs a binary's tests on parallel threads).
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn gbs(bytes: usize, secs: f64) -> f64 {
    (bytes as f64 / secs) / 1e9
}

/// One backend with the D2H DMA knob pinned, a buffer of class `usage` holding `pattern`, and
/// the achieved download bandwidth over `ITERS` timed readbacks (first call = warmup). Returns
/// (GB/s, ms per call).
fn probe(dma: bool, usage: BufferUsage, pattern: &[u8]) -> Option<(f64, f64, VulkanBackend)> {
    let _env = ENV_LOCK.lock().unwrap();
    // The knob is read at construction, so this pin decides the route for this backend only.
    std::env::set_var("INFR_D2H_DMA", if dma { "1" } else { "" });
    let be = VulkanBackend::new().ok()?;
    let dev = be.alloc_uninit(BYTES, usage).expect("alloc source buffer");
    be.upload(dev.as_ref(), pattern).expect("upload pattern");

    let mut out = vec![0u8; BYTES];
    be.download(dev.as_ref(), &mut out)
        .expect("warmup download");
    assert_eq!(
        &out[..],
        &pattern[..],
        "downloaded bytes diverge (dma={dma})"
    );

    let t0 = Instant::now();
    for _ in 0..ITERS {
        be.download(dev.as_ref(), &mut out).expect("download");
    }
    let secs = t0.elapsed().as_secs_f64();
    let per_call_ms = secs * 1e3 / ITERS as f64;
    Some((gbs(BYTES * ITERS, secs), per_call_ms, be))
}

/// Perf probe on real hardware — prints the A/B, asserts both paths return the uploaded bytes.
/// Two source classes because they can behave OPPOSITELY:
///   - `Activations` (GpuOnly): the legacy path's fresh staging lands in cached sysmem → already
///     fast; both routes pay the same per-call submit+wait overhead.
///   - `Readback` (mapped GpuToCpu): with ReBAR the driver may place the class in BAR, whose CPU
///     reads are uncached — the suspected production pathology (m×vocab logits Output) the DMA
///     path exists to fix.
#[test]
#[ignore = "requires a Vulkan GPU; perf probe, not a correctness test"]
fn d2h_dma_probe() {
    let pattern: Vec<u8> = (0..BYTES)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect();

    for (label, usage) in [
        ("Readback (mapped)", BufferUsage::Readback),
        ("Activations (GpuOnly)", BufferUsage::Activations),
    ] {
        let Some((mapped_gbs, mapped_ms, _legacy)) = probe(false, usage, &pattern) else {
            eprintln!("skip: no Vulkan device");
            return;
        };
        let (dma_gbs, dma_ms, _dma) = probe(true, usage, &pattern).expect("no Vulkan device");
        println!("\nD2H readback of {BYTES} bytes x{ITERS} — source {label}:");
        println!("  mapped CPU read (legacy): {mapped_gbs:8.2} GB/s  ({mapped_ms:6.2} ms/call)");
        println!(
            "  DMA into imported RAM   : {dma_gbs:8.2} GB/s  ({dma_ms:6.2} ms/call)  ({:.2}x)",
            dma_gbs / mapped_gbs
        );
    }
}

/// Non-ignored correctness gate: with the DMA route pinned ON (`INFR_D2H_DMA=1`), an
/// above-threshold download exercises the DMA slab and must return the uploaded bytes exactly.
/// Skips silently on boxes without a Vulkan device (same convention as the rest of this crate's
/// GPU tests).
#[test]
fn d2h_dma_readback_matches_pattern() {
    let _env = ENV_LOCK.lock().unwrap();
    std::env::set_var("INFR_D2H_DMA", "1");
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let pattern: Vec<u8> = (0..BYTES)
        .map(|i| (i as u8).wrapping_mul(13).wrapping_add(11))
        .collect();
    let dev = be
        .alloc_uninit(BYTES, BufferUsage::Activations)
        .expect("alloc device-local source");
    be.upload(dev.as_ref(), &pattern).expect("upload pattern");

    let mut out = vec![0u8; BYTES];
    be.download(dev.as_ref(), &mut out).expect("download");
    assert_eq!(&out[..], &pattern[..], "DMA readback returned wrong bytes");

    // Also below-threshold: the mapped path must stay byte-exact (the threshold route is chosen
    // by size alone; correctness is size-independent).
    let small = &pattern[..4096];
    let mut out_small = vec![0u8; small.len()];
    be.download(dev.as_ref(), &mut out_small)
        .expect("small download");
    assert_eq!(
        &out_small[..],
        small,
        "mapped readback returned wrong bytes"
    );
}
