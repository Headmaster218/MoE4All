//! ROCm/HIP dense-weight prefetch ring (Slice 37) — hides the PCIe cost of the Slice-35
//! host-spilled dense weight banks behind compute.
//!
//! # The problem this solves
//! Slice 35 (`INFR_ROCM_WEIGHT_OVERFLOW`) spills the tail of a model's dense weight banks to
//! page-locked, device-mapped HOST RAM. Its `RocmBuffer::ptr` is the device alias of that host
//! allocation, so the native `Linear` GEMV reads the weight DIRECTLY over PCIe — synchronously,
//! every token, with the GEMV's scattered per-block access pattern. On Qwen3-14B Q4_K_M forced to
//! a 2 GiB weight budget that decodes at ~1.8 t/s: the transfer is neither coalesced nor overlapped.
//!
//! # The fix — a VRAM staging ring + prefetch (the HIP twin of Vulkan's `dense_paged` streamer)
//! The transformer runs layers SEQUENTIALLY, and decode replays the exact same op sequence every
//! token, so the spilled bank a Linear will read is known well ahead of its GEMV. This ring keeps a
//! small pool of VRAM slots and, on a dedicated **copy stream**, bulk-DMAs the NEXT spilled bank(s)
//! from pinned host memory into a free slot while the CURRENT layer computes. The GEMV then reads a
//! *resident VRAM slot* (a single coalesced copy, fully overlapped) instead of streaming the bank
//! over PCIe inside the kernel. The copy is still PCIe-bound in aggregate, but it is (a) a bulk
//! contiguous DMA at full link bandwidth rather than the kernel's scattered reads and (b) hidden
//! behind compute, so the per-token wall clock collapses toward the raw PCIe-bandwidth floor.
//!
//! Unlike the MoE pager ([`crate::pager`]) whose source is the PAGEABLE GGUF mmap (needing a
//! mmap→pinned staging hop), a spilled dense bank is ALREADY in `hipHostMalloc` pinned memory, so
//! the copy is a single `hipMemcpyAsync(slot, host_ptr, len, H2D, copy_stream)` — no staging ring on
//! the host side.
//!
//! # Ring schedule (double/N-buffered, per-slot events)
//! At the start of each `execute` the caller builds the ordered list of spilled-native Linear banks
//! (`SpilledBank`) and hands it to [`RocmWeightRing::begin_execute`]. Bank `k` is assigned slot
//! `k % n_slots`. Two events per slot order the two streams:
//! - `ready[s]`: recorded on the COPY stream after bank `k`'s H2D fill; the compute stream
//!   `hipStreamWaitEvent`s it before dispatching bank `k`'s GEMV.
//! - `free[s]`: recorded on the COMPUTE stream after bank `k`'s GEMV; the copy stream
//!   `hipStreamWaitEvent`s it before overwriting slot `s` with bank `k + n_slots` — so a prefetch
//!   never clobbers a slot an in-flight GEMV still reads (eviction safety).
//!
//! `begin_execute` PRIMES the ring (fills the first `n_slots` slots). Then, per spilled Linear the
//! executor dispatches: [`RocmWeightRing::stage`] waits `ready[s]` and returns the slot pointer;
//! after the GEMV, [`RocmWeightRing::consumed`] records `free[s]` and kicks off the copy of bank
//! `k + n_slots` into slot `s`. So bank `k`'s GEMV overlaps bank `k+1..`'s copies; the copy stream
//! stays PCIe-saturated while compute mostly waits on it (the decode path is copy-bound).
//!
//! # Oversized banks + fallback
//! A slot is sized to the LARGEST STAGED bank. Banks larger than a fixed cap
//! (`INFR_ROCM_WEIGHT_PREFETCH_MAX_BANK_MB`, default 256 MiB — the lm_head / token_embd banks that
//! dwarf a per-layer projection) are NOT staged: the executor keeps reading them via the Slice-35
//! host alias, and they are excluded from the schedule (so the cursor stays in lockstep). A ring
//! allocation / stream-create failure yields `None` — every bank then falls back to the Slice-35
//! direct read, correct just un-overlapped.
//!
//! # Interaction with the MoE pager
//! Independent: a dense model has no MoE pager; a paged-MoE model's experts live in the MoE arena
//! and its DENSE banks (attn/router/shared-expert projections) flow through THIS ring. Each owns its
//! own copy stream + VRAM; the only shared resource is total VRAM, covered by the weight-overflow
//! reserve. No dense Linear reads an expert bank, so there is no tensor overlap.

use std::ffi::c_void;

use infr_core::error::Result;

use crate::backend::RocmBuffer;
use crate::ffi::{self, HIP_MEMCPY_HOST_TO_DEVICE, HIP_SUCCESS};

/// Terse local shorthand for the shared backend-error constructor.
use infr_core::error::backend as be;

/// Default per-bank size cap (MiB): banks larger than this are NOT staged (they fall back to the
/// Slice-35 host-alias read) so one giant bank — the lm_head / token_embd, which is `vocab*hidden`
/// and dwarfs a per-layer projection — can't blow the ring's VRAM up to `n_slots × its size`.
const DEFAULT_MAX_BANK_MB: usize = 256;

/// Default ring depth. The copy stream is a single PCIe pipe, so more slots don't raise copy
/// throughput; the depth just needs to keep the next copy queued while compute drains its waits.
/// 4 comfortably covers the tiny decode compute-per-layer without wasting VRAM.
const DEFAULT_N_SLOTS: usize = 4;

/// Per-bank cap in bytes (`INFR_ROCM_WEIGHT_PREFETCH_MAX_BANK_MB`, default [`DEFAULT_MAX_BANK_MB`]).
/// A spilled native Linear bank is staged only when its byte length is `<= cap`; larger banks fall
/// back to the Slice-35 direct read. This is the SINGLE predicate the schedule build and the
/// executor's per-op staged/fallback decision must agree on (keeping the ring cursor in lockstep).
pub fn max_bank_bytes(paging: &infr_core::config::PagingCfg) -> usize {
    paging
        .rocm_prefetch_max_bank_mb
        .unwrap_or(DEFAULT_MAX_BANK_MB)
        * 1024
        * 1024
}

/// Ring depth (`INFR_ROCM_WEIGHT_PREFETCH_SLOTS`, default [`DEFAULT_N_SLOTS`]), floored at 2 so the
/// ring can always double-buffer (one slot filling while another computes). The floor is POLICY and
/// stays here, at the accessor, not in the env layer (`docs/config-plan.md` R5).
fn n_slots(paging: &infr_core::config::PagingCfg) -> usize {
    paging.rocm_prefetch_slots.unwrap_or(DEFAULT_N_SLOTS).max(2)
}

/// One spilled dense weight bank scheduled for prefetch: its pinned host source (the Slice-35
/// `hipHostMalloc` allocation), the device alias that identifies the bound buffer, and the byte
/// length (the whole bank is copied; the Linear applies its own `w_off` block offset within the
/// staged slot). `dev_alias` is the identity the executor matches against the bound weight buffer's
/// `ptr` so the ring's internal cursor stays in lockstep with the op walk.
#[derive(Clone, Copy)]
pub struct SpilledBank {
    pub host_src: *const c_void,
    pub dev_alias: *mut c_void,
    pub len: usize,
}

// Raw device/pinned-host pointers identify fixed allocations, not CPU-thread state — Send/Sync like
// `RocmBuffer` / the MoE pager.
unsafe impl Send for SpilledBank {}
unsafe impl Sync for SpilledBank {}

/// The dense-weight prefetch ring. Lives on the `RocmBackend` (`Some` only once a spilled-native
/// Linear has been seen and the ring built); `None` for every resident / small model — zero cost,
/// zero behavior change on the resident path.
pub struct RocmWeightRing {
    /// Compute stream (the GEMVs run here; cross-stream `ready`-waits are enqueued here).
    compute_stream: ffi::hipStream_t,
    /// Dedicated H2D copy stream carrying the bank fills, decoupled from compute.
    copy_stream: ffi::hipStream_t,
    /// `n_slots * slot_bytes` contiguous VRAM buffer of uniform slots.
    slots: RocmBuffer,
    /// Byte size of one slot (>= every staged bank).
    slot_bytes: usize,
    n_slots: usize,
    /// Per-slot "fill complete" events (copy stream → compute stream ordering).
    ready: Vec<ffi::hipEvent_t>,
    /// Per-slot "GEMV done consuming" events (compute stream → copy stream ordering).
    free: Vec<ffi::hipEvent_t>,
    /// This forward's ordered staged banks (set by [`Self::begin_execute`]).
    schedule: Vec<SpilledBank>,
    /// Index of the next spilled Linear the executor will [`Self::stage`] — advances in lockstep
    /// with the op walk (only staged Linears touch it), reset per forward.
    cursor: usize,
    /// `INFR_ROCM_WEIGHT_PREFETCH_STATS`: dump ring config + staged-bank tally once.
    print_stats: bool,
    stats_printed: bool,
    staged_banks: u64,
    staged_bytes: u64,
}

// Owns device buffers + opaque HIP stream/event handles — Send/Sync like `RocmBackend`.
unsafe impl Send for RocmWeightRing {}
unsafe impl Sync for RocmWeightRing {}

impl RocmWeightRing {
    /// Build the ring for a max staged-bank size of `slot_bytes`: a copy stream, an `n_slots`-slot
    /// VRAM arena, and `2 * n_slots` ordering events. Returns `None` (→ the Slice-35 direct-read
    /// fallback) if any HIP call fails — prefetch must never hard-error just because VRAM for the
    /// ring or a second stream can't be had. `INFR_ROCM_WEIGHT_PREFETCH_OFF` also forces `None`
    /// (A/B lever: measure the Slice-35 synchronous-PCIe baseline).
    pub fn try_new(
        slot_bytes: usize,
        compute_stream: ffi::hipStream_t,
        paging: &infr_core::config::PagingCfg,
    ) -> Option<Self> {
        if paging.rocm_prefetch_off {
            return None;
        }
        if slot_bytes == 0 {
            return None;
        }
        // Round the slot stride to 256 bytes so slot `k` inherits the arena's `hipMalloc`
        // alignment. The F4 decode GEMV loads Q4_K/Q5_K weights 128 bits at a time and proves that
        // load's 16-byte alignment from "allocation base + a whole number of blocks" — a staged
        // bank sits at `slot * slot_bytes`, and `slot_bytes` is the MAX bank length across formats,
        // which need not be a multiple of 16 (Q6_K's 210-byte blocks, say). Rounding costs at most
        // 255 bytes per slot and keeps the alignment argument true for every tier.
        let slot_bytes = slot_bytes.next_multiple_of(256);
        let n = n_slots(paging);
        let mut copy_stream: ffi::hipStream_t = std::ptr::null_mut();
        if unsafe { ffi::hipStreamCreate(&mut copy_stream) } != HIP_SUCCESS {
            return None;
        }
        // Zero-init (calloc contract): every read stays inside the copied bank region, but the
        // one-time memset is cheap belt-and-suspenders against a slot's untouched tail.
        let slots = match RocmBuffer::try_alloc(n * slot_bytes, compute_stream) {
            Ok(b) => b,
            Err(_) => {
                unsafe { ffi::hipStreamDestroy(copy_stream) };
                return None;
            }
        };
        let mut ready = Vec::with_capacity(n);
        let mut free = Vec::with_capacity(n);
        let mut ok = true;
        for _ in 0..(2 * n) {
            let mut ev: ffi::hipEvent_t = std::ptr::null_mut();
            if unsafe { ffi::hipEventCreateWithFlags(&mut ev, ffi::HIP_EVENT_DISABLE_TIMING) }
                != HIP_SUCCESS
            {
                ok = false;
                break;
            }
            if ready.len() < n {
                ready.push(ev);
            } else {
                free.push(ev);
            }
        }
        if !ok {
            for &e in ready.iter().chain(free.iter()) {
                unsafe { ffi::hipEventDestroy(e) };
            }
            unsafe { ffi::hipStreamDestroy(copy_stream) };
            return None;
        }
        Some(Self {
            compute_stream,
            copy_stream,
            slots,
            slot_bytes,
            n_slots: n,
            ready,
            free,
            schedule: Vec::new(),
            cursor: 0,
            print_stats: paging.rocm_prefetch_stats,
            stats_printed: false,
            staged_banks: 0,
            staged_bytes: 0,
        })
    }

    /// The ring's slot size — the executor recreates the ring if a later forward's max staged bank
    /// exceeds it (never happens for a fixed weight set, but cheap to guard).
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Device pointer to `slot`'s base within the arena.
    fn slot_ptr(&self, slot: usize) -> *mut c_void {
        unsafe { (self.slots.ptr as *mut u8).add(slot * self.slot_bytes) as *mut c_void }
    }

    /// Issue the H2D fill of `schedule[k]` into slot `k % n_slots` on the copy stream and record its
    /// `ready` event. `wait_free` gates the copy on the slot's previous consumer (the `free` event
    /// recorded after that GEMV) — skipped for the priming fills, which land in slots left free by
    /// the caller's pre-`begin_execute` compute drain.
    fn issue_fill(&self, k: usize, wait_free: bool) -> Result<()> {
        let bank = self.schedule[k];
        let s = k % self.n_slots;
        if wait_free
            && unsafe { ffi::hipStreamWaitEvent(self.copy_stream, self.free[s], 0) } != HIP_SUCCESS
        {
            return Err(be(
                "rocm weight ring: hipStreamWaitEvent(free) on copy stream failed",
            ));
        }
        let rc = unsafe {
            ffi::hipMemcpyAsync(
                self.slot_ptr(s),
                bank.host_src,
                bank.len,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.copy_stream,
            )
        };
        if rc != HIP_SUCCESS {
            return Err(be(format!(
                "rocm weight ring: hipMemcpyAsync H2D bank fill: rc={rc}"
            )));
        }
        if unsafe { ffi::hipEventRecord(self.ready[s], self.copy_stream) } != HIP_SUCCESS {
            return Err(be(
                "rocm weight ring: hipEventRecord(ready) on copy stream failed",
            ));
        }
        Ok(())
    }

    /// Install this forward's ordered staged-bank schedule, reset the cursor, and PRIME the ring:
    /// fill the first `min(n_slots, len)` slots. The caller guarantees the previous forward is fully
    /// drained (the executor's terminal `hipStreamSynchronize`) before this runs, so the prime fills
    /// need no `free`-wait. Draining the copy stream first is belt-and-suspenders (it is already
    /// idle by that same terminal sync).
    pub fn begin_execute(&mut self, schedule: Vec<SpilledBank>) -> Result<()> {
        unsafe { ffi::hipStreamSynchronize(self.copy_stream) };
        self.schedule = schedule;
        self.cursor = 0;
        let prime = self.n_slots.min(self.schedule.len());
        for k in 0..prime {
            debug_assert!(
                self.schedule[k].len <= self.slot_bytes,
                "staged bank exceeds slot"
            );
            self.issue_fill(k, false)?;
        }
        if self.print_stats && !self.stats_printed {
            self.stats_printed = true;
            eprintln!(
                "[rocm weight ring] slots={} x {:.1} MiB ({:.1} MiB VRAM); {} banks staged/forward",
                self.n_slots,
                self.slot_bytes as f64 / (1024.0 * 1024.0),
                (self.n_slots * self.slot_bytes) as f64 / (1024.0 * 1024.0),
                self.schedule.len(),
            );
        }
        Ok(())
    }

    /// Ensure the CURRENT spilled Linear's bank is resident and return its VRAM slot pointer: make
    /// the compute stream wait on the bank's `ready` event, then hand back the slot base (the caller
    /// applies its own `w_off` block offset). `dev_alias` MUST match the scheduled bank at the
    /// cursor — a mismatch means the op walk and the schedule desynced (a build bug), which would
    /// feed the GEMV the wrong weight, so it hard-errors rather than corrupting silently.
    pub fn stage(&mut self, dev_alias: *mut c_void) -> Result<*mut c_void> {
        let k = self.cursor;
        if k >= self.schedule.len() {
            return Err(be(
                "rocm weight ring: stage past the schedule end (cursor/op-walk desync)",
            ));
        }
        let bank = self.schedule[k];
        if bank.dev_alias != dev_alias {
            return Err(be(
                "rocm weight ring: staged bank identity mismatch (cursor/op-walk desync)",
            ));
        }
        let s = k % self.n_slots;
        if unsafe { ffi::hipStreamWaitEvent(self.compute_stream, self.ready[s], 0) } != HIP_SUCCESS
        {
            return Err(be(
                "rocm weight ring: hipStreamWaitEvent(ready) on compute stream failed",
            ));
        }
        Ok(self.slot_ptr(s))
    }

    /// Mark the current spilled Linear's GEMV dispatched: record the slot's `free` event on the
    /// compute stream (so the next fill into it waits on this GEMV), kick off the prefetch of bank
    /// `cursor + n_slots` (the next occupant of the freed slot), and advance the cursor. Call once,
    /// AFTER the GEMV of the bank the matching [`Self::stage`] returned.
    pub fn consumed(&mut self) -> Result<()> {
        let k = self.cursor;
        let s = k % self.n_slots;
        if unsafe { ffi::hipEventRecord(self.free[s], self.compute_stream) } != HIP_SUCCESS {
            return Err(be(
                "rocm weight ring: hipEventRecord(free) on compute stream failed",
            ));
        }
        let next = k + self.n_slots;
        if next < self.schedule.len() {
            debug_assert!(
                self.schedule[next].len <= self.slot_bytes,
                "staged bank exceeds slot"
            );
            self.issue_fill(next, true)?;
        }
        self.staged_banks += 1;
        self.staged_bytes += self.schedule[k].len as u64;
        self.cursor += 1;
        Ok(())
    }
}

impl Drop for RocmWeightRing {
    fn drop(&mut self) {
        // Drain outstanding copies before tearing down the stream / events / arena.
        unsafe { ffi::hipStreamSynchronize(self.copy_stream) };
        for &e in self.ready.iter().chain(self.free.iter()) {
            unsafe { ffi::hipEventDestroy(e) };
        }
        unsafe { ffi::hipStreamDestroy(self.copy_stream) };
        // `self.slots` (RocmBuffer) frees its VRAM arena in its own Drop.
    }
}

/// S6 (`docs/config-plan.md` §8): the prefetch sizing knobs come off `PagingCfg`, and the `max(2)`
/// slot floor stays HERE — policy in the accessor, not in the env layer (R5).
#[cfg(test)]
mod config_tests {
    use super::{max_bank_bytes, n_slots, DEFAULT_MAX_BANK_MB, DEFAULT_N_SLOTS};
    use infr_core::config::PagingCfg;

    fn paging(f: impl FnOnce(&mut PagingCfg)) -> PagingCfg {
        let mut p = PagingCfg::default();
        f(&mut p);
        p
    }

    #[test]
    fn prefetch_bank_cap_defaults_and_overrides() {
        let d = PagingCfg::default();
        assert_eq!(max_bank_bytes(&d), DEFAULT_MAX_BANK_MB * 1024 * 1024);
        let p = paging(|p| p.rocm_prefetch_max_bank_mb = Some(512));
        assert_eq!(max_bank_bytes(&p), 512 * 1024 * 1024);
    }

    #[test]
    fn prefetch_slots_default_and_floor_at_two() {
        let d = PagingCfg::default();
        assert_eq!(n_slots(&d), DEFAULT_N_SLOTS);
        assert_eq!(n_slots(&paging(|p| p.rocm_prefetch_slots = Some(6))), 6);
        // The double-buffer floor is the accessor's policy, so 0 and 1 both clamp up to 2.
        assert_eq!(n_slots(&paging(|p| p.rocm_prefetch_slots = Some(0))), 2);
        assert_eq!(n_slots(&paging(|p| p.rocm_prefetch_slots = Some(1))), 2);
    }
}
