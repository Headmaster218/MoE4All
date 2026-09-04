//! Multi-GPU Slice 0 — foundation gates + the interconnect decision spike.
//!
//! Two tests, both hardware-gated (need ≥2 Vulkan devices; skip cleanly otherwise):
//!
//!   1. `two_backends_coexist` — proves the multi-backend refactor: build a `VulkanBackend` on the
//!      discrete GPU AND one on the integrated GPU AT THE SAME TIME (two live instances + logical
//!      devices), then allocate a buffer and run a trivial op (upload → download round-trip) on EACH
//!      independently, asserting the bytes survive. This is the prerequisite every later
//!      tensor/expert-parallel slice builds on.
//!
//!   2. `interconnect_probe` — the DECISION DATA. Measures cross-device transfer over the only path
//!      available today (host-mediated: device A → host RAM → device B): sustained BANDWIDTH (256
//!      MiB blocks) and round-trip LATENCY (a tiny transfer). Also reports whether RADV exposes the
//!      external-memory extensions a host-LESS GPU↔GPU (dma-buf / fd import) path would need. The
//!      numbers say whether tensor/expert parallelism can pay for its cross-device traffic on THIS
//!      box's interconnect, or whether the later slices should be correctness-only.
//!
//! Run: `cargo test -p infr-vulkan --test interconnect_probe -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::{DeviceInfo, P2pHandleType, VulkanBackend};
use std::time::Instant;

const MIB: usize = 1024 * 1024;

fn gib_per_sec(bytes: usize, secs: f64) -> f64 {
    (bytes as f64 / secs) / (1u64 << 30) as f64
}

/// (discrete index, integrated index) if the box has both — else `None` (test self-skips).
fn dgpu_igpu(devs: &[DeviceInfo]) -> Option<(usize, usize)> {
    let dgpu = devs.iter().find(|d| d.device_type == "discrete")?;
    let igpu = devs.iter().find(|d| d.integrated)?;
    Some((dgpu.index, igpu.index))
}

#[test]
#[ignore = "requires ≥2 Vulkan devices (a discrete + an integrated GPU); multi-device foundation gate"]
fn two_backends_coexist() {
    let Ok(devs) = VulkanBackend::enumerate_devices_from_env() else {
        eprintln!("skip: no Vulkan");
        return;
    };
    let Some((d_idx, i_idx)) = dgpu_igpu(&devs) else {
        eprintln!("skip: need both a discrete and an integrated GPU (have {devs:?})");
        return;
    };

    // Build BOTH backends and hold them live simultaneously.
    let dgpu = VulkanBackend::new_on(d_idx).expect("build dGPU backend");
    let igpu = VulkanBackend::new_on(i_idx).expect("build iGPU backend");
    eprintln!("[coexist] two backends live: Vulkan{d_idx} (dGPU) + Vulkan{i_idx} (iGPU)");

    // A trivial op on EACH, independently: alloc a device buffer, upload a known pattern, read it
    // back, assert the round-trip is exact. Distinct patterns per device catch a cross-wire.
    for (label, be, byte) in [("dGPU", &dgpu, 0xA5u8), ("iGPU", &igpu, 0x5Au8)] {
        let n = 4 * MIB;
        let src = vec![byte; n];
        let buf = be
            .alloc_uninit(n, BufferUsage::Weights)
            .expect("alloc device buffer");
        be.upload(buf.as_ref(), &src).expect("upload");
        be.sync().expect("sync");
        let mut back = vec![0u8; n];
        be.download(buf.as_ref(), &mut back).expect("download");
        assert_eq!(back, src, "{label}: round-trip mismatch");
        eprintln!("[coexist] {label}: {n}-byte upload→download round-trip OK");
    }
    eprintln!("[coexist] PASS — dGPU + iGPU backends coexist, trivial op on each succeeds");
}

#[test]
#[ignore = "requires ≥2 Vulkan devices; perf/decision spike, prints numbers, not a CI gate"]
fn interconnect_probe() {
    let Ok(devs) = VulkanBackend::enumerate_devices_from_env() else {
        eprintln!("skip: no Vulkan");
        return;
    };
    let Some((d_idx, i_idx)) = dgpu_igpu(&devs) else {
        eprintln!("skip: need both a discrete and an integrated GPU");
        return;
    };

    // ── P2P / external-memory feasibility ────────────────────────────────────────────────────
    eprintln!("\n── external-memory / P2P feasibility (RADV) ──");
    for d in &devs {
        eprintln!(
            "  Vulkan{}: {} [{}]  external_memory={} fd={} dma_buf={}",
            d.index,
            d.name,
            d.device_type,
            d.external_memory,
            d.external_memory_fd,
            d.external_memory_dma_buf,
        );
    }
    let both_fd = devs
        .iter()
        .filter(|d| d.index == d_idx || d.index == i_idx)
        .all(|d| d.external_memory_fd);
    let both_dma = devs
        .iter()
        .filter(|d| d.index == d_idx || d.index == i_idx)
        .all(|d| d.external_memory_dma_buf);
    eprintln!(
        "  host-LESS GPU↔GPU import feasible? external_memory_fd on both: {both_fd}; \
         dma_buf on both: {both_dma}  (extensions present ⇒ a P2P import path is BUILDABLE; this \
         probe measures only the host-bounce baseline below)"
    );

    let dgpu = VulkanBackend::new_on(d_idx).expect("build dGPU backend");
    let igpu = VulkanBackend::new_on(i_idx).expect("build iGPU backend");

    // ── sustained bandwidth: 256 MiB, device A → host RAM → device B ─────────────────────────
    let block = 256 * MIB;
    let iters = 5;
    let d_buf = dgpu
        .alloc_uninit(block, BufferUsage::Weights)
        .expect("dGPU buffer");
    let i_buf = igpu
        .alloc_uninit(block, BufferUsage::Weights)
        .expect("iGPU buffer");
    // Seed the source device buffer with a real pattern (not zeros — avoids any zero-page shortcut).
    let seed: Vec<u8> = (0..block).map(|k| (k as u8).wrapping_mul(31)).collect();
    dgpu.upload(d_buf.as_ref(), &seed).expect("seed dGPU");
    dgpu.sync().expect("sync");

    let mut host = vec![0u8; block];

    // warmup (first-touch pages, pipeline/allocator warm)
    dgpu.download(d_buf.as_ref(), &mut host).expect("d->host");
    igpu.upload(i_buf.as_ref(), &host).expect("host->i");
    igpu.sync().expect("sync");

    // Phase A: dGPU → host
    let t = Instant::now();
    for _ in 0..iters {
        dgpu.download(d_buf.as_ref(), &mut host).expect("d->host");
    }
    dgpu.sync().expect("sync");
    let d2h = gib_per_sec(block * iters, t.elapsed().as_secs_f64());

    // Phase B: host → iGPU
    let t = Instant::now();
    for _ in 0..iters {
        igpu.upload(i_buf.as_ref(), &host).expect("host->i");
    }
    igpu.sync().expect("sync");
    let h2i = gib_per_sec(block * iters, t.elapsed().as_secs_f64());

    // Combined: full dGPU → host → iGPU cross-device copy (the real cost of moving a tensor).
    let t = Instant::now();
    for _ in 0..iters {
        dgpu.download(d_buf.as_ref(), &mut host).expect("d->host");
        igpu.upload(i_buf.as_ref(), &host).expect("host->i");
    }
    igpu.sync().expect("sync");
    let combined = gib_per_sec(block * iters, t.elapsed().as_secs_f64());

    // Verify the bytes actually crossed (correctness of the host-bounce path).
    let mut check = vec![0u8; block];
    igpu.download(i_buf.as_ref(), &mut check).expect("i->host");
    assert_eq!(
        &check[..4096],
        &seed[..4096],
        "cross-device bytes corrupted"
    );

    eprintln!(
        "\n── host-mediated cross-device bandwidth ({} MiB blocks, {iters} iters) ──",
        block / MIB
    );
    eprintln!("  dGPU → host          : {d2h:.2} GiB/s");
    eprintln!("  host → iGPU          : {h2i:.2} GiB/s");
    eprintln!("  dGPU → host → iGPU   : {combined:.2} GiB/s  (effective cross-device throughput)");

    // ── round-trip latency: a tiny transfer, dGPU → host → iGPU → host → dGPU ────────────────
    let small = 4096usize;
    let sd = dgpu.alloc_uninit(small, BufferUsage::Weights).expect("sd");
    let si = igpu.alloc_uninit(small, BufferUsage::Weights).expect("si");
    let payload = vec![0x7Eu8; small];
    dgpu.upload(sd.as_ref(), &payload).expect("seed small");
    dgpu.sync().expect("sync");
    let mut hbuf = vec![0u8; small];
    // warmup
    dgpu.download(sd.as_ref(), &mut hbuf).expect("rt warm d");
    igpu.upload(si.as_ref(), &hbuf).expect("rt warm i");
    igpu.sync().expect("sync");

    let rt_iters = 200;
    let t = Instant::now();
    for _ in 0..rt_iters {
        dgpu.download(sd.as_ref(), &mut hbuf).expect("rt d->h");
        igpu.upload(si.as_ref(), &hbuf).expect("rt h->i");
        igpu.download(si.as_ref(), &mut hbuf).expect("rt i->h");
        dgpu.upload(sd.as_ref(), &hbuf).expect("rt h->d");
    }
    dgpu.sync().expect("sync");
    igpu.sync().expect("sync");
    let rt_us = t.elapsed().as_secs_f64() * 1e6 / rt_iters as f64;
    eprintln!("\n── round-trip latency (4 KiB, dGPU→host→iGPU→host→dGPU) ──");
    eprintln!("  {rt_us:.1} µs / full round-trip (4 host-mediated hops)");
    eprintln!(
        "  (~{:.1} µs per single host-mediated device hop)",
        rt_us / 4.0
    );
    eprintln!("\n[interconnect] done — see numbers above for the tensor/expert-parallel decision");
}

/// Slice-1 baseline numbers this P2P slice is measured against (Slice-0 `interconnect_probe`,
/// RX 7900 XTX ↔ 9950X3D iGPU): effective host-bounce cross-device throughput and per-hop latency.
const HOST_BOUNCE_GIB_PER_SEC: f64 = 3.56;
const HOST_BOUNCE_HOP_US: f64 = 33.0;

/// Move `block` bytes producer→consumer through SHARED external memory (no host bounce) and verify
/// they arrive byte-correct; return a one-line bandwidth+latency report or the exact rejection.
///
/// Mechanism: the producer allocates+exports a buffer (`handle_type`), uploads a known pattern into
/// it and `sync()`s (host-side ordering — the write is fully flushed before the consumer reads); the
/// consumer IMPORTS the fd (its buffer now aliases the producer's physical bytes) and pulls them with
/// a device-side `copy_buffer` (imported alias → consumer-local), which is the real P2P read over
/// PCIe. Correctness is asserted; bandwidth/latency are measured from the consumer's copies.
fn p2p_move(
    producer: &VulkanBackend,
    consumer: &VulkanBackend,
    handle_type: P2pHandleType,
    block: usize,
    label: &str,
) -> Result<String, String> {
    let export = producer
        .p2p_export(block, handle_type)
        .map_err(|e| format!("{label}: export failed: {e}"))?;

    // Seed the shared buffer on the PRODUCER with a real pattern (not zeros — a zero-page shortcut
    // would not prove the bytes actually crossed the device boundary).
    let seed: Vec<u8> = (0..block)
        .map(|k| (k as u8).wrapping_mul(37).wrapping_add(11))
        .collect();
    producer
        .upload(export.buffer(), &seed)
        .map_err(|e| format!("{label}: producer upload failed: {e}"))?;
    producer
        .sync()
        .map_err(|e| format!("{label}: producer sync failed: {e}"))?;

    // Import on the CONSUMER — its buffer now aliases the producer's memory over PCIe.
    let imported = consumer
        .p2p_import(&export)
        .map_err(|e| format!("{label}: import REJECTED: {e}"))?;

    // Consumer-local landing buffer; the consumer pulls the shared bytes into it.
    let local = consumer
        .alloc_uninit(block, BufferUsage::Weights)
        .map_err(|e| format!("{label}: consumer local alloc failed: {e}"))?;

    // Correctness FIRST: one pull, then read back on the consumer and compare every byte.
    consumer
        .copy_buffer(imported.as_ref(), local.as_ref(), block)
        .map_err(|e| format!("{label}: consumer copy failed: {e}"))?;
    consumer
        .sync()
        .map_err(|e| format!("{label}: consumer sync failed: {e}"))?;
    let mut check = vec![0u8; block];
    consumer
        .download(local.as_ref(), &mut check)
        .map_err(|e| format!("{label}: consumer download failed: {e}"))?;
    if check != seed {
        let first_bad = (0..block).find(|&i| check[i] != seed[i]).unwrap_or(0);
        return Err(format!(
            "{label}: CROSS-DEVICE BYTES CORRUPTED at offset {first_bad}: got {:#04x}, want {:#04x} \
             (import succeeded but the alias does not read the producer's memory correctly)",
            check[first_bad], seed[first_bad]
        ));
    }

    // ── sustained bandwidth: consumer pulls the whole block over PCIe, N times ────────────────
    let iters = 5;
    let t = Instant::now();
    for _ in 0..iters {
        consumer
            .copy_buffer(imported.as_ref(), local.as_ref(), block)
            .map_err(|e| format!("{label}: bw copy failed: {e}"))?;
    }
    consumer
        .sync()
        .map_err(|e| format!("{label}: bw sync failed: {e}"))?;
    let bw = gib_per_sec(block * iters, t.elapsed().as_secs_f64());

    // ── single-hop latency: a small shared buffer, one consumer pull ─────────────────────────
    let small = 4096usize;
    let small_export = producer
        .p2p_export(small, handle_type)
        .map_err(|e| format!("{label}: small export failed: {e}"))?;
    producer
        .upload(small_export.buffer(), &vec![0x7Eu8; small])
        .map_err(|e| format!("{label}: small upload failed: {e}"))?;
    producer.sync().ok();
    let small_import = consumer
        .p2p_import(&small_export)
        .map_err(|e| format!("{label}: small import failed: {e}"))?;
    let small_local = consumer
        .alloc_uninit(small, BufferUsage::Weights)
        .map_err(|e| format!("{label}: small local alloc failed: {e}"))?;
    // warmup
    consumer
        .copy_buffer(small_import.as_ref(), small_local.as_ref(), small)
        .ok();
    consumer.sync().ok();
    let rt_iters = 200;
    let t = Instant::now();
    for _ in 0..rt_iters {
        consumer
            .copy_buffer(small_import.as_ref(), small_local.as_ref(), small)
            .map_err(|e| format!("{label}: latency copy failed: {e}"))?;
    }
    consumer.sync().ok();
    let hop_us = t.elapsed().as_secs_f64() * 1e6 / rt_iters as f64;

    Ok(format!(
        "{label}: bytes OK ✓  {bw:.2} GiB/s ({:.2}x host-bounce)  {hop_us:.1} µs/hop ({:.2}x)",
        bw / HOST_BOUNCE_GIB_PER_SEC,
        HOST_BOUNCE_HOP_US / hop_us,
    ))
}

#[test]
#[ignore = "requires ≥2 Vulkan devices with external-memory support; P2P transport spike, prints numbers"]
fn p2p_external_memory_probe() {
    let Ok(devs) = VulkanBackend::enumerate_devices_from_env() else {
        eprintln!("skip: no Vulkan");
        return;
    };
    let Some((d_idx, i_idx)) = dgpu_igpu(&devs) else {
        eprintln!("skip: need both a discrete and an integrated GPU");
        return;
    };

    let dgpu = VulkanBackend::new_on(d_idx).expect("build dGPU backend");
    let igpu = VulkanBackend::new_on(i_idx).expect("build iGPU backend");

    eprintln!("\n── host-LESS cross-device P2P (external-memory export/import) ──");
    for (label, be) in [("dGPU", &dgpu), ("iGPU", &igpu)] {
        eprintln!(
            "  {label}: p2p dma-buf={}  opaque-fd={}",
            be.p2p_supported(P2pHandleType::DmaBuf),
            be.p2p_supported(P2pHandleType::OpaqueFd),
        );
    }
    eprintln!(
        "  baseline (Slice 0, host-bounce): {HOST_BOUNCE_GIB_PER_SEC:.2} GiB/s effective, \
         ~{HOST_BOUNCE_HOP_US:.0} µs/hop"
    );

    let block = 256 * MIB;
    // The direction the campaign cares about most first: export dGPU VRAM, import on the iGPU (the
    // iGPU reads the discrete card's VRAM directly over PCIe). Then the reverse, then opaque-fd.
    let runs = [
        (
            &dgpu,
            &igpu,
            P2pHandleType::DmaBuf,
            "dma-buf  dGPU-VRAM → iGPU  (iGPU reads dGPU VRAM)",
        ),
        (
            &igpu,
            &dgpu,
            P2pHandleType::DmaBuf,
            "dma-buf  iGPU-mem  → dGPU  (dGPU reads iGPU/system RAM)",
        ),
        (
            &dgpu,
            &igpu,
            P2pHandleType::OpaqueFd,
            "opaque-fd dGPU-VRAM → iGPU (cross-device opaque-fd probe)",
        ),
    ];

    let mut any_ok = false;
    eprintln!("\n── P2P transfer (256 MiB, verified byte-correct across the device boundary) ──");
    for (producer, consumer, ht, label) in runs {
        match p2p_move(producer, consumer, ht, block, label) {
            Ok(report) => {
                any_ok = true;
                eprintln!("  {report}");
            }
            Err(why) => eprintln!("  {label}: {why}"),
        }
    }

    eprintln!(
        "\n[p2p] done — a '✓' line beating {HOST_BOUNCE_GIB_PER_SEC:.2} GiB/s means host-less P2P is the \
         viable cross-device transport on this box; a rejection line is the equally-valid finding \
         that this handle type/direction is not usable on RADV here."
    );
    // At least ONE path must have moved bytes correctly for the capability to be real; a total
    // rejection is reported above (not silently swallowed) and is itself the deliverable finding.
    assert!(
        any_ok,
        "no P2P path moved bytes correctly — see the per-run rejection lines above"
    );
}
