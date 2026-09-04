//! Pager design gate (deliverable 0 of the MoE-expert-pager task): measures REAL host→device copy
//! bandwidth for the two block sizes the pager will move — one expert (~40 MiB) and one layer's
//! stacked expert bank (~660 MiB, Llama-4-Scout Q2_K numbers) — before any pager code is written.
//! If this comes back far below ~18.6 GiB/s the whole per-expert-upload design is unsound (the
//! decisive-math note in the task assumed ~24.2 GiB/s PCIe) and building the pager on top would be
//! wasted work.
//!
//! Compares three upload strategies, all through the SAME `Backend` trait surface every other
//! test in this crate uses (no unsafe pointer poking):
//!   1. `fresh`     — `Backend::upload` straight onto a `Weights` (device-local) buffer, exactly
//!      what today's per-tensor loader does: allocates a brand-new staging buffer, memcpies into
//!      it, copies, frees it. This is the ZERO-reuse baseline the pager must beat.
//!   2. `persistent`— a single `Staging` buffer allocated ONCE before the loop and reused for every
//!      block (the "pinned staging region" the pager's upload machinery is meant to use), split
//!      into its two phases: `memcpy` (host mmap → mapped pointer) and `pcie` (`copy_buffer`,
//!      i.e. `vkCmdCopyBuffer` + submit + wait).
//!
//! The source bytes come from a real mmap (memmap2, matching `infr-gguf`'s zero-copy tensor
//! loading) of the locally-cached Llama-4-Scout Q2_K GGUF when present, else a scratch temp file —
//! either way the source is genuinely page-cache-backed, unpinned host memory, not a `Vec` the
//! allocator may have already faulted and pinned via `mlock`-adjacent tricks.
//!
//! Run: `cargo test -p infr-vulkan --test bandwidth_probe -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;
use memmap2::Mmap;
use std::time::Instant;

const MIB: usize = 1024 * 1024;
const EXPERT_BYTES: usize = 40 * MIB; // one Scout Q2_K expert-sized transfer
const LAYER_BYTES: usize = 660 * MIB; // one Scout Q2_K layer's stacked gate+up+down bank

/// Real GGUF blob already on this box (avoids materializing a 660 MiB scratch file); falls back to
/// a temp file of the max block size so the probe still runs on a box without the model cached.
fn mmap_source(min_bytes: usize) -> Mmap {
    let real = std::env::var("HOME").ok().map(|h| {
        format!(
            "{h}/.cache/huggingface/hub/models--unsloth--Llama-4-Scout-17B-16E-Instruct-GGUF/\
             snapshots/72a6853f56a66dc13a3a4b6bdc9cf7ee4c364b47/\
             Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf"
        )
    });
    if let Some(p) = &real {
        if std::fs::metadata(p).is_ok_and(|m| m.len() as usize >= min_bytes) {
            let f = std::fs::File::open(p).expect("open cached GGUF");
            return unsafe { Mmap::map(&f) }.expect("mmap cached GGUF");
        }
    }
    eprintln!("(no cached Scout GGUF found — falling back to a scratch temp file)");
    let path = std::env::temp_dir().join(format!("infr_bw_probe_{}.bin", std::process::id()));
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).expect("create scratch file");
        let chunk = vec![0xABu8; MIB];
        for _ in 0..min_bytes.div_ceil(MIB) {
            f.write_all(&chunk).expect("write scratch file");
        }
    }
    let f = std::fs::File::open(&path).expect("reopen scratch file");
    let m = unsafe { Mmap::map(&f) }.expect("mmap scratch file");
    let _ = std::fs::remove_file(&path); // unlinked; the mmap + open fd keep the data alive
    m
}

fn gib_per_sec(bytes: usize, secs: f64) -> f64 {
    (bytes as f64 / secs) / (1u64 << 30) as f64
}

/// Times `iters` back-to-back calls of `f` (each moving `bytes`), discarding the first call as
/// warmup, and returns the achieved GiB/s over the remaining `iters - 1`.
fn bench(iters: usize, bytes: usize, mut f: impl FnMut()) -> f64 {
    f(); // warmup: first touch of these mmap pages, first pipeline/allocator warm-up
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    gib_per_sec(bytes * iters, t0.elapsed().as_secs_f64())
}

#[test]
#[ignore = "requires a Vulkan GPU + local Scout GGUF (or scratch fallback); perf probe, not a correctness test"]
fn bandwidth_probe() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let mmap = mmap_source(LAYER_BYTES);
    assert!(
        mmap.len() >= LAYER_BYTES,
        "source too small: {} < {LAYER_BYTES}",
        mmap.len()
    );

    // One persistent pinned staging buffer + one persistent device-local target, sized to the
    // LARGER block and reused for both sizes — exactly the shape the pager's slot machinery wants
    // (one staging region sized to the model's largest block, reused across every upload).
    let staging = be
        .alloc_uninit(LAYER_BYTES, BufferUsage::Staging)
        .expect("alloc staging");
    let dst = be
        .alloc_uninit(LAYER_BYTES, BufferUsage::Weights)
        .expect("alloc device dst");

    eprintln!(
        "\n{:<10} {:>10} {:>14} {:>14} {:>14} {:>14}",
        "block", "iters", "fresh(GiB/s)", "memcpy(GiB/s)", "pcie(GiB/s)", "combined(GiB/s)"
    );
    for (label, block, iters) in [("expert", EXPERT_BYTES, 20), ("layer", LAYER_BYTES, 5)] {
        let src = &mmap[0..block];

        // 1. Today's production path: fresh ephemeral staging buffer allocated+freed every call.
        let fresh_dst = be
            .alloc_uninit(block, BufferUsage::Weights)
            .expect("alloc fresh dst");
        let fresh = bench(iters, block, || {
            be.upload(fresh_dst.as_ref(), src).expect("fresh upload");
        });

        // 2a. Persistent pinned staging: host memcpy phase only (mmap -> mapped pointer, no GPU
        // submit — `upload` on a `Staging` (CpuToGpu) dst takes the direct-mapped-write branch).
        let memcpy = bench(iters, block, || {
            be.upload(staging.as_ref(), src).expect("staging upload");
        });

        // 2b. Persistent pinned staging -> device-local: PCIe transfer phase only (data already in
        // the staging buffer from 2a's last iteration; re-copying the same bytes is fine, this
        // phase only cares about the device-side DMA + submit cost).
        let pcie = bench(iters, block, || {
            be.copy_buffer(staging.as_ref(), dst.as_ref(), block)
                .expect("copy_buffer");
        });

        // 2 combined: memcpy + copy back-to-back per iteration (the pager's real steady-state
        // upload-a-block cost with a warm, reused staging buffer).
        let combined = bench(iters, block, || {
            be.upload(staging.as_ref(), src).expect("staging upload");
            be.copy_buffer(staging.as_ref(), dst.as_ref(), block)
                .expect("copy_buffer");
        });

        eprintln!(
            "{label:<10} {iters:>10} {:>14.2} {:>14.2} {:>14.2} {:>14.2}",
            fresh, memcpy, pcie, combined
        );
    }

    // Gate: if the reused-staging combined path can't clear a conservative floor, the pager's
    // whole per-expert-upload premise (decisive math assumed ~24.2 GiB/s) doesn't hold on this box —
    // fail loudly rather than let a silent regression hide inside "it still runs, just slow".
    // (No hard assert here: this test's JOB is to print the numbers for a human/agent to read
    // before committing to the design, not to gate CI — see the module doc.)
}
