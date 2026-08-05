//! Proves the dense streaming path's THIRD tier — VRAM over host DRAM over the model file
//! (`docs/disk-streaming-plan.md` §3.7) — stages the right bytes in all three of its cases, and
//! that it stages the SAME bytes the mmap source does.
//!
//! `DensePagerSession::stage` resolves one weight group per call and has three outcomes:
//!
//!   1. **VRAM hit** — the slot already holds the block; nothing is copied.
//!   2. **VRAM miss, DRAM hit** — the block is pinned in the host arena and memcpy'd to the ring.
//!   3. **VRAM miss, DRAM full** — the host tier reads the block off the model file STRAIGHT into
//!      the ring, admitting nothing (see `HostPager::fill`).
//!
//! A case that never runs looks identical to one that works, so the budgets here are picked to
//! force all three (VRAM: 3 slots for 8 blocks; DRAM: 5 slots for the same 8) and the test ASSERTS
//! each was taken, from the counters `pool_stats` reports. Correctness is checked the only way that
//! catches a wrong slot or a torn multi-extent read: every staged block is dispatched through the
//! streamed GEMV and its output compared against the same weight uploaded to a plain arena — a
//! block served from a neighbour's slot decodes to visibly different finite floats, not an error.
//!
//! The two sessions run the identical sweep over the identical blocks, differing only in
//! `DenseBytes`, so any divergence between them is the host tier's doing and nothing else.
//!
//! Weight bytes are drawn from `0x00..=0x3F` for the reason `weight_addr_parity.rs`'s header gives:
//! every f16 scale in that range is finite, so no output is NaN — a NaN would make a bitwise
//! compare pass vacuously, hiding exactly the mis-addressing this test exists to catch.
//!
//! Run: `cargo test -p infr-vulkan --test dense_tier_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::blockio::{BlockDesc, BlockExtent, FileBlockIo};
use infr_core::hostpager::HostPager;
use infr_core::DType;
use infr_vulkan::pager::{
    buffer_identity, DenseBytes, DensePagerLayout, DensePagerSession, DensePoolSpec, DenseSource,
};
use infr_vulkan::VulkanBackend;
use std::sync::Arc;

const DTYPE: DType = DType::Q4_0;
const N_BLOCKS: usize = 8;
const IN_F: usize = 256;
const OUT_F: usize = 32;
/// VRAM slots for `N_BLOCKS` blocks — fewer, so the sweep both hits and misses.
const VRAM_SLOTS: usize = 3;
/// Host slots for the same blocks — fewer again, so some blocks are admitted and the rest stream
/// past a full arena on every pass, which is what exercises case 3 beyond the first sweep.
const DRAM_SLOTS: usize = 5;
const PASSES: usize = 3;

/// Pseudo-random weight bytes in `0x00..=0x3F`; `seed` is the block index, so every block decodes
/// to a different output and a mis-served slot is visible.
fn synth_weight_bytes(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7;
            (h % 0x40) as u8
        })
        .collect()
}

fn synth_x(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect()
}

/// What `DensePagerSession::pool_stats` reports for one pool: VRAM residency, and the tier below
/// when there is one.
type PoolStats = (
    infr_core::pager::PagerStats,
    Option<infr_core::hostpager::HostPagerStats>,
);

fn bits(v: &[u8]) -> Vec<u32> {
    bytemuck::cast_slice::<u8, u32>(v).to_vec()
}

/// One block's byte size, and the pool stride that holds it (a whole number of quant blocks AND of
/// u32 words — what the seam's `stride_of` computes).
fn block_and_stride() -> (usize, usize) {
    let (blk_elems, blk_bytes) = infr_gguf::block_layout(DTYPE);
    let elems = IN_F * OUT_F;
    assert_eq!(elems % blk_elems, 0, "shape must be whole {DTYPE:?} blocks");
    let raw = elems / blk_elems * blk_bytes;
    let lcm = {
        let (mut x, mut y) = (blk_bytes, 4usize);
        while y != 0 {
            (x, y) = (y, x % y);
        }
        blk_bytes / x * 4
    };
    (raw, raw.next_multiple_of(lcm))
}

/// Dispatch the streamed GEMV against `addr` and return the output's bits.
fn gemv_at(be: &VulkanBackend, addr: u64, x: &dyn infr_core::backend::Buffer) -> Vec<u32> {
    let y = be.alloc(OUT_F * 4, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.linear_native_at(DTYPE, addr, 0, x, y.as_ref(), 1, IN_F, OUT_F);
    rec.finish().unwrap();
    let mut out = vec![0u8; OUT_F * 4];
    be.download(y.as_ref(), &mut out).unwrap();
    bits(&out)
}

/// What one sweep observed.
struct Sweep {
    /// Each block's GEMV output, from the last pass that staged it.
    outputs: Vec<Vec<u32>>,
    stats: Vec<PoolStats>,
    /// How many times the ring half filled and had to be swapped.
    rotations: usize,
}

/// Build a session over `sources`, sweep every block `PASSES` times dispatching each staged
/// address, and report what happened.
///
/// The ring cursor persists across blocks, so a half genuinely fills and `stage` returns `None` —
/// the rotation path, which is the caller's job and not the pager's. Each recording is submitted
/// before the half it wrote is reused, which is `stage`'s ring-lifetime contract; the adapter
/// pipelines the two halves against a fence instead, and that difference is deliberate — this test
/// is about which bytes land, not about overlapping them with compute.
fn sweep(
    be: &VulkanBackend,
    spec: DensePoolSpec,
    ring_bytes: usize,
    make_source: impl Fn(usize) -> DenseBytes,
    x: &dyn infr_core::backend::Buffer,
) -> Sweep {
    let mut sess = DensePagerSession::new(
        be,
        DensePagerLayout {
            pools: vec![spec],
            ring_bytes,
        },
    )
    .unwrap();
    // Real placeholder buffers, so the `buf_id` keys are the same kind of identity production uses.
    let placeholders: Vec<_> = (0..N_BLOCKS)
        .map(|_| be.alloc_uninit(4, BufferUsage::Weights).unwrap())
        .collect();
    let ids: Vec<usize> = placeholders
        .iter()
        .map(|p| buffer_identity(p.as_ref()))
        .collect();
    for (b, &buf_id) in ids.iter().enumerate() {
        sess.register(
            0,
            buf_id,
            DenseSource {
                bytes: make_source(b),
                block_id: b as u32,
            },
        )
        .unwrap();
    }

    let mut outputs = vec![Vec::new(); N_BLOCKS];
    let (mut half, mut cursor, mut rotations) = (0usize, 0usize, 0usize);
    for _ in 0..PASSES {
        for (b, &buf_id) in ids.iter().enumerate() {
            let addr = loop {
                let rec = be.recorder().unwrap();
                let half_base = half * sess.ring_half_bytes();
                let staged = sess.stage(&rec, half_base, &mut cursor, buf_id).unwrap();
                rec.finish().unwrap();
                match staged {
                    Some(a) => break a,
                    None => {
                        // The half cannot hold this block's copy: submit what is recorded (done
                        // above), swap halves and retry. Progress is guaranteed because a half
                        // always holds at least the largest slot.
                        half ^= 1;
                        cursor = 0;
                        rotations += 1;
                    }
                }
            };
            outputs[b] = gemv_at(be, addr, x);
        }
    }
    Sweep {
        outputs,
        stats: sess.pool_stats(),
        rotations,
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn the_host_tier_stages_the_same_bytes_the_mmap_source_does() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };

    let (raw, stride) = block_and_stride();
    let blocks: Vec<Vec<u8>> = (0..N_BLOCKS).map(|b| synth_weight_bytes(raw, b)).collect();

    // The "model file": the same blocks back to back, so a block's extent is `(b * raw, raw)`.
    let dir = std::env::temp_dir().join(format!("infr-dense-tier-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("weights.bin");
    std::fs::write(&path, blocks.concat()).unwrap();

    let x = synth_x(IN_F);
    let x_buf = be.alloc(IN_F * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

    // Reference: each weight in an arena of its own, no pager involved.
    let expect: Vec<Vec<u32>> = blocks
        .iter()
        .map(|w| {
            let (arena, addr) = be.alloc_arena_bda(w.len()).unwrap();
            be.upload(arena.as_ref(), w).unwrap();
            gemv_at(&be, addr, x_buf.as_ref())
        })
        .collect();
    for (b, e) in expect.iter().enumerate() {
        assert!(
            e.iter().any(|&v| v != 0),
            "block {b}'s reference output is all zeros — the case exercises nothing"
        );
    }
    for b in 1..N_BLOCKS {
        assert_ne!(
            expect[0], expect[b],
            "blocks 0 and {b} decode identically — a mis-served slot would be invisible"
        );
    }

    // ── Leg 1: the mmap source (what every streamed model took before the host tier) ───────────
    let mmap = sweep(
        &be,
        DensePoolSpec {
            slot_bytes: stride,
            n_slots: VRAM_SLOTS,
            n_blocks: N_BLOCKS,
            host: None,
        },
        4 * stride,
        |b| {
            DenseBytes::Mmap(vec![
                Arc::new(blocks[b].clone()) as Arc<dyn AsRef<[u8]> + Send + Sync>
            ])
        },
        x_buf.as_ref(),
    );

    // ── Leg 2: the host tier, reading the same bytes off the file ─────────────────────────────
    let io = Arc::new(FileBlockIo::open(&path).unwrap());
    let host = Arc::new(HostPager::new(DRAM_SLOTS, stride, io).unwrap());
    for b in 0..N_BLOCKS {
        host.register(BlockDesc {
            id: b as u32,
            extents: vec![BlockExtent {
                offset: (b * raw) as u64,
                len: raw,
            }],
        })
        .unwrap();
    }
    let host_leg = sweep(
        &be,
        DensePoolSpec {
            slot_bytes: stride,
            n_slots: VRAM_SLOTS,
            n_blocks: N_BLOCKS,
            host: Some(host.clone()),
        },
        4 * stride,
        |_| DenseBytes::Host,
        x_buf.as_ref(),
    );

    // ── Leg 3: the ARENA-LESS tier — what a unified-memory device gets ────────────────────────
    // There the arena above is already GPU-accessible RAM, so nothing is cached beneath it and
    // every miss is read straight into the ring. This machine is a discrete GPU, so the SELECTION
    // of this mode cannot be exercised here — but the mode itself can, and must stage the same
    // bytes as the other two.
    let io2 = Arc::new(FileBlockIo::open(&path).unwrap());
    let host_ro = Arc::new(HostPager::stream_only(stride, io2).unwrap());
    for b in 0..N_BLOCKS {
        host_ro
            .register(BlockDesc {
                id: b as u32,
                extents: vec![BlockExtent {
                    offset: (b * raw) as u64,
                    len: raw,
                }],
            })
            .unwrap();
    }
    let stream_leg = sweep(
        &be,
        DensePoolSpec {
            slot_bytes: stride,
            n_slots: VRAM_SLOTS,
            n_blocks: N_BLOCKS,
            host: Some(host_ro.clone()),
        },
        4 * stride,
        |_| DenseBytes::Host,
        x_buf.as_ref(),
    );

    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(&dir).unwrap();

    for (b, want) in expect.iter().enumerate() {
        assert_eq!(&mmap.outputs[b], want, "mmap source staged block {b} wrong");
        assert_eq!(
            &host_leg.outputs[b], want,
            "host tier staged block {b} wrong"
        );
        assert_eq!(
            &stream_leg.outputs[b], want,
            "arena-less tier staged block {b} wrong"
        );
    }

    // The arena-less tier caches NOTHING and reads through on every VRAM miss. If it ever admitted,
    // a unified device would be holding a second copy of a block in the one pool of RAM it has.
    let (vram_ro, ro) = &stream_leg.stats[0];
    let ro = ro.expect("leg 3 has a host tier");
    assert_eq!(ro.pager.hits, 0, "an arena-less tier cannot hit");
    assert_eq!(ro.pager.misses, 0, "an arena-less tier must admit nothing");
    assert_eq!(ro.pager.evictions, 0);
    assert_eq!(
        ro.streamed, vram_ro.misses,
        "every VRAM miss must be one streamed read"
    );
    assert_eq!(ro.reads, ro.streamed, "every read must be a streamed one");
    assert_eq!(
        (vram_ro.hits, vram_ro.misses),
        (mmap.stats[0].0.hits, mmap.stats[0].0.misses),
        "the arena-less tier changed the VRAM schedule"
    );

    // ── Every case was actually taken ─────────────────────────────────────────────────────────
    let (vram, host_s) = &host_leg.stats[0];
    let host_s = host_s.expect("leg 2 has a host tier");
    assert!(
        vram.hits > 0,
        "case 1 (VRAM hit) never ran: {VRAM_SLOTS} slots over {N_BLOCKS} blocks, {PASSES} passes"
    );
    assert!(vram.misses > 0, "no VRAM miss — cases 2 and 3 never ran");
    assert!(
        host_s.pager.hits > 0,
        "case 2 (VRAM miss, DRAM hit) never ran: host {} hits",
        host_s.pager.hits
    );
    assert!(
        host_s.streamed > 0,
        "case 3 (VRAM miss, DRAM full) never ran — nothing streamed past the arena"
    );
    // The PARTITION property, and the reason `fill` exists: once the arena is full the tier streams
    // rather than evicting. An eviction here would mean the arena is churning — paying a copy to
    // admit a block whose next use is a whole sweep away, which is the shape that measured slower
    // than the mmap it replaces.
    assert_eq!(
        host_s.pager.evictions, 0,
        "the host tier evicted; a full arena must stream past itself instead"
    );
    assert_eq!(
        host_s.pager.misses, DRAM_SLOTS as u64,
        "admissions must stop at the arena's slot count"
    );
    // The tier below is asked exactly on the VRAM misses, and no more: a probe that also fired on
    // hits would make the hit rate meaningless (the counters would report the sweep as warmer than
    // it is), which is the failure `HostPager::repin` exists to prevent.
    let consulted = host_s.pager.hits + host_s.pager.misses + host_s.streamed;
    assert_eq!(
        consulted, vram.misses,
        "the host tier was consulted {consulted} times for {} VRAM misses",
        vram.misses
    );
    assert_eq!(
        host_s.reads,
        host_s.pager.misses + host_s.streamed,
        "every admission and every streamed block is exactly one file read"
    );
    assert_eq!(
        host_s.bytes_read,
        host_s.reads * raw as u64,
        "reads moved a partial block"
    );

    // The mmap leg must reach the same VRAM residency decisions — same policy, same order — so any
    // difference in the numbers would mean the host arm changed the schedule, not just the source.
    assert_eq!(
        (mmap.stats[0].0.hits, mmap.stats[0].0.misses),
        (vram.hits, vram.misses),
        "the host tier changed the VRAM schedule"
    );
    assert!(mmap.stats[0].1.is_none(), "leg 1 must have no host tier");
    // The ring half filled and was swapped mid-sweep on both legs. Without a rotation the sweep
    // would only ever stage into a fresh half, which is the one case that cannot expose a stale
    // ring offset.
    assert!(
        host_leg.rotations > 0 && mmap.rotations == host_leg.rotations,
        "ring rotations differ or never happened: mmap {} vs host {}",
        mmap.rotations,
        host_leg.rotations
    );

    eprintln!(
        "vram: {} hits / {} misses; dram: {} hits / {} misses, {} reads ({} bytes); \
         {} ring rotations",
        vram.hits,
        vram.misses,
        host_s.pager.hits,
        host_s.pager.misses,
        host_s.reads,
        host_s.bytes_read,
        host_leg.rotations,
    );
}

/// A `Host` source whose block the tier below never registered must be refused at REGISTRATION —
/// the load — rather than at the first miss, mid-generation.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_host_block_missing_from_the_tier_below_is_refused_at_load() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (raw, stride) = block_and_stride();
    let dir = std::env::temp_dir().join(format!("infr-dense-tier-neg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("weights.bin");
    std::fs::write(&path, synth_weight_bytes(raw, 0)).unwrap();

    let io = Arc::new(FileBlockIo::open(&path).unwrap());
    let host = Arc::new(HostPager::new(1, stride, io).unwrap());
    let mut sess = DensePagerSession::new(
        &be,
        DensePagerLayout {
            pools: vec![DensePoolSpec {
                slot_bytes: stride,
                n_slots: 1,
                n_blocks: 2,
                host: Some(host),
            }],
            ring_bytes: 4 * stride,
        },
    )
    .unwrap();
    let ph = be.alloc_uninit(4, BufferUsage::Weights).unwrap();
    let err = sess
        .register(
            0,
            buffer_identity(ph.as_ref()),
            DenseSource {
                bytes: DenseBytes::Host,
                block_id: 0,
            },
        )
        .expect_err("an unregistered host block must be refused");
    assert!(
        err.to_string().contains("host tier"),
        "unexpected error: {err}"
    );

    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(&dir).unwrap();
}
