//! Runtime pager profiling (`INFR_PAGER_PROFILE=1`).
//!
//! This is deliberately separate from `INFR_PROF_OPS`: the latter uses Vulkan timestamp queries and
//! changes the paged ring path by forcing `finish_nowait` to drain. The pager profile stays on host
//! wall-clock timings and aggregate counters so it can be left on for the exact workload shape being
//! investigated.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use std::io::Write as _;

static ENABLED: AtomicBool = AtomicBool::new(false);
static PRINTED: AtomicBool = AtomicBool::new(false);
static COUNTERS: Counters = Counters::new();

struct Counters {
    gpu_lookups: AtomicU64,
    gpu_lookup_ns: AtomicU64,
    gpu_hits: AtomicU64,
    gpu_misses: AtomicU64,
    gpu_evictions: AtomicU64,
    gpu_eviction_ns: AtomicU64,

    host_hits: AtomicU64,
    host_misses: AtomicU64,
    host_evictions: AtomicU64,
    host_waits: AtomicU64,
    host_wait_ns: AtomicU64,
    host_bytes: AtomicU64,
    host_reads: AtomicU64,
    host_read_bytes: AtomicU64,
    host_read_ns: AtomicU64,
    host_streamed: AtomicU64,
    mmap_fallbacks: AtomicU64,
    mmap_fallback_bytes: AtomicU64,
    mmap_fallback_ns: AtomicU64,

    memcpys: AtomicU64,
    memcpy_bytes: AtomicU64,
    memcpy_ns: AtomicU64,

    staging_acquires: AtomicU64,
    staging_acquire_ns: AtomicU64,
    staging_waits: AtomicU64,
    staging_wait_ns: AtomicU64,

    gpu_copies: AtomicU64,
    gpu_copy_bytes: AtomicU64,

    queue_submits: AtomicU64,
    queue_submit_ns: AtomicU64,
    submitted_dispatches: AtomicU64,

    sync_waits: AtomicU64,
    sync_wait_ns: AtomicU64,
    queue_idle_waits: AtomicU64,
    queue_idle_wait_ns: AtomicU64,
    fence_waits: AtomicU64,
    fence_wait_ns: AtomicU64,
    paging_sync_waits: AtomicU64,
    paging_sync_wait_ns: AtomicU64,

    splitter_forwards: AtomicU64,
    splitter_armed_forwards: AtomicU64,
    splitter_cap_start_last: AtomicU64,
    splitter_cap_end_last: AtomicU64,
    splitter_cap_min: AtomicU64,
    splitter_cap_max: AtomicU64,
    splitter_result_submits: AtomicU64,
    splitter_result_dispatches: AtomicU64,

    lru_mark_calls: AtomicU64,
    lru_mark_scan_steps: AtomicU64,
    lru_mark_max_scan_steps: AtomicU64,
    lru_mark_queue_len_sum: AtomicU64,
    lru_mark_max_queue_len: AtomicU64,
    lru_victim_calls: AtomicU64,
    lru_victim_scan_steps: AtomicU64,
    lru_victim_max_scan_steps: AtomicU64,
    lru_victim_queue_len_sum: AtomicU64,
    lru_victim_max_queue_len: AtomicU64,
}

impl Counters {
    const fn new() -> Self {
        Self {
            gpu_lookups: AtomicU64::new(0),
            gpu_lookup_ns: AtomicU64::new(0),
            gpu_hits: AtomicU64::new(0),
            gpu_misses: AtomicU64::new(0),
            gpu_evictions: AtomicU64::new(0),
            gpu_eviction_ns: AtomicU64::new(0),

            host_hits: AtomicU64::new(0),
            host_misses: AtomicU64::new(0),
            host_evictions: AtomicU64::new(0),
            host_waits: AtomicU64::new(0),
            host_wait_ns: AtomicU64::new(0),
            host_bytes: AtomicU64::new(0),
            host_reads: AtomicU64::new(0),
            host_read_bytes: AtomicU64::new(0),
            host_read_ns: AtomicU64::new(0),
            host_streamed: AtomicU64::new(0),
            mmap_fallbacks: AtomicU64::new(0),
            mmap_fallback_bytes: AtomicU64::new(0),
            mmap_fallback_ns: AtomicU64::new(0),

            memcpys: AtomicU64::new(0),
            memcpy_bytes: AtomicU64::new(0),
            memcpy_ns: AtomicU64::new(0),

            staging_acquires: AtomicU64::new(0),
            staging_acquire_ns: AtomicU64::new(0),
            staging_waits: AtomicU64::new(0),
            staging_wait_ns: AtomicU64::new(0),

            gpu_copies: AtomicU64::new(0),
            gpu_copy_bytes: AtomicU64::new(0),

            queue_submits: AtomicU64::new(0),
            queue_submit_ns: AtomicU64::new(0),
            submitted_dispatches: AtomicU64::new(0),

            sync_waits: AtomicU64::new(0),
            sync_wait_ns: AtomicU64::new(0),
            queue_idle_waits: AtomicU64::new(0),
            queue_idle_wait_ns: AtomicU64::new(0),
            fence_waits: AtomicU64::new(0),
            fence_wait_ns: AtomicU64::new(0),
            paging_sync_waits: AtomicU64::new(0),
            paging_sync_wait_ns: AtomicU64::new(0),

            splitter_forwards: AtomicU64::new(0),
            splitter_armed_forwards: AtomicU64::new(0),
            splitter_cap_start_last: AtomicU64::new(0),
            splitter_cap_end_last: AtomicU64::new(0),
            splitter_cap_min: AtomicU64::new(u64::MAX),
            splitter_cap_max: AtomicU64::new(0),
            splitter_result_submits: AtomicU64::new(0),
            splitter_result_dispatches: AtomicU64::new(0),

            lru_mark_calls: AtomicU64::new(0),
            lru_mark_scan_steps: AtomicU64::new(0),
            lru_mark_max_scan_steps: AtomicU64::new(0),
            lru_mark_queue_len_sum: AtomicU64::new(0),
            lru_mark_max_queue_len: AtomicU64::new(0),
            lru_victim_calls: AtomicU64::new(0),
            lru_victim_scan_steps: AtomicU64::new(0),
            lru_victim_max_scan_steps: AtomicU64::new(0),
            lru_victim_queue_len_sum: AtomicU64::new(0),
            lru_victim_max_queue_len: AtomicU64::new(0),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LruWorkStats {
    pub mark_calls: u64,
    pub mark_scan_steps: u64,
    pub mark_max_scan_steps: u64,
    pub mark_queue_len_sum: u64,
    pub mark_max_queue_len: u64,
    pub victim_calls: u64,
    pub victim_scan_steps: u64,
    pub victim_max_scan_steps: u64,
    pub victim_queue_len_sum: u64,
    pub victim_max_queue_len: u64,
}

impl LruWorkStats {
    #[inline]
    pub fn record_mark_mru(&mut self, queue_len: usize, scan_steps: usize) {
        self.mark_calls += 1;
        self.mark_scan_steps += scan_steps as u64;
        self.mark_max_scan_steps = self.mark_max_scan_steps.max(scan_steps as u64);
        self.mark_queue_len_sum += queue_len as u64;
        self.mark_max_queue_len = self.mark_max_queue_len.max(queue_len as u64);
    }

    #[inline]
    pub fn record_victim_select(&mut self, queue_len: usize, scan_steps: usize) {
        self.victim_calls += 1;
        self.victim_scan_steps += scan_steps as u64;
        self.victim_max_scan_steps = self.victim_max_scan_steps.max(scan_steps as u64);
        self.victim_queue_len_sum += queue_len as u64;
        self.victim_max_queue_len = self.victim_max_queue_len.max(queue_len as u64);
    }

    #[inline]
    pub fn has_activity(&self) -> bool {
        self.mark_calls != 0 || self.victim_calls != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SyncKind {
    QueueIdle,
    Fence,
}

#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub gpu_lookups: u64,
    pub gpu_lookup_ns: u64,
    pub gpu_hits: u64,
    pub gpu_misses: u64,
    pub gpu_evictions: u64,
    pub gpu_eviction_ns: u64,

    pub host_hits: u64,
    pub host_misses: u64,
    pub host_evictions: u64,
    pub host_waits: u64,
    pub host_wait_ns: u64,
    pub host_bytes: u64,
    pub host_reads: u64,
    pub host_read_bytes: u64,
    pub host_read_ns: u64,
    pub host_streamed: u64,
    pub mmap_fallbacks: u64,
    pub mmap_fallback_bytes: u64,
    pub mmap_fallback_ns: u64,

    pub memcpys: u64,
    pub memcpy_bytes: u64,
    pub memcpy_ns: u64,

    pub staging_acquires: u64,
    pub staging_acquire_ns: u64,
    pub staging_waits: u64,
    pub staging_wait_ns: u64,

    pub gpu_copies: u64,
    pub gpu_copy_bytes: u64,

    pub queue_submits: u64,
    pub queue_submit_ns: u64,
    pub submitted_dispatches: u64,

    pub sync_waits: u64,
    pub sync_wait_ns: u64,
    pub queue_idle_waits: u64,
    pub queue_idle_wait_ns: u64,
    pub fence_waits: u64,
    pub fence_wait_ns: u64,
    pub paging_sync_waits: u64,
    pub paging_sync_wait_ns: u64,

    pub splitter_forwards: u64,
    pub splitter_armed_forwards: u64,
    pub splitter_cap_start_last: u64,
    pub splitter_cap_end_last: u64,
    pub splitter_cap_min: u64,
    pub splitter_cap_max: u64,
    pub splitter_result_submits: u64,
    pub splitter_result_dispatches: u64,

    pub lru_mark_calls: u64,
    pub lru_mark_scan_steps: u64,
    pub lru_mark_max_scan_steps: u64,
    pub lru_mark_queue_len_sum: u64,
    pub lru_mark_max_queue_len: u64,
    pub lru_victim_calls: u64,
    pub lru_victim_scan_steps: u64,
    pub lru_victim_max_scan_steps: u64,
    pub lru_victim_queue_len_sum: u64,
    pub lru_victim_max_queue_len: u64,
}

impl Snapshot {
    pub fn gpu_hit_rate(&self) -> f64 {
        rate(self.gpu_hits, self.gpu_hits + self.gpu_misses)
    }

    pub fn host_hit_rate(&self) -> f64 {
        rate(self.host_hits, self.host_hits + self.host_misses)
    }

    pub fn avg_lru_mark_scan_steps(&self) -> f64 {
        avg(self.lru_mark_scan_steps, self.lru_mark_calls)
    }

    pub fn avg_lru_mark_queue_len(&self) -> f64 {
        avg(self.lru_mark_queue_len_sum, self.lru_mark_calls)
    }

    pub fn avg_lru_victim_scan_steps(&self) -> f64 {
        avg(self.lru_victim_scan_steps, self.lru_victim_calls)
    }

    pub fn avg_lru_victim_queue_len(&self) -> f64 {
        avg(self.lru_victim_queue_len_sum, self.lru_victim_calls)
    }
}

/// Process-lifetime guard used by CLI entry points to print the profile once after command teardown.
pub struct SummaryGuard;

impl SummaryGuard {
    pub fn new(enabled: bool) -> Self {
        set_enabled(enabled);
        Self
    }
}

impl Drop for SummaryGuard {
    fn drop(&mut self) {
        print_summary_if_enabled();
    }
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

#[inline]
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

#[inline]
pub fn active() -> bool {
    enabled() && !crate::prof::suppressed()
}

#[inline]
pub fn start() -> Option<Instant> {
    active().then(Instant::now)
}

#[inline]
pub fn elapsed(t0: Option<Instant>) -> Option<Duration> {
    t0.map(|t| t.elapsed())
}

#[inline]
pub fn queue_submit_count() -> u64 {
    COUNTERS.queue_submits.load(Ordering::Relaxed)
}

#[inline]
pub fn record_gpu_cache_lookup(hit: bool, evicted: bool, elapsed: Duration) {
    COUNTERS.gpu_lookups.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .gpu_lookup_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
    if hit {
        COUNTERS.gpu_hits.fetch_add(1, Ordering::Relaxed);
    } else {
        COUNTERS.gpu_misses.fetch_add(1, Ordering::Relaxed);
    }
    if evicted {
        COUNTERS.gpu_evictions.fetch_add(1, Ordering::Relaxed);
        COUNTERS
            .gpu_eviction_ns
            .fetch_add(ns(elapsed), Ordering::Relaxed);
    }
}

#[inline]
pub fn record_host_hit(bytes: usize) {
    COUNTERS.host_hits.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .host_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
}

#[inline]
pub fn record_host_miss(bytes: usize, evicted: bool) {
    COUNTERS.host_misses.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .host_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
    if evicted {
        COUNTERS.host_evictions.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub fn record_host_wait(elapsed: Duration) {
    COUNTERS.host_waits.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .host_wait_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_host_read(bytes: usize, elapsed: Duration, streamed: bool) {
    COUNTERS.host_reads.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .host_read_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
    COUNTERS
        .host_read_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
    if streamed {
        COUNTERS.host_streamed.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub fn record_mmap_fallback(bytes: usize, elapsed: Duration) {
    COUNTERS.mmap_fallbacks.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .mmap_fallback_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
    COUNTERS
        .mmap_fallback_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_memcpy(bytes: usize, elapsed: Duration) {
    COUNTERS.memcpys.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .memcpy_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
    COUNTERS.memcpy_ns.fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_staging_acquire(elapsed: Duration) {
    COUNTERS.staging_acquires.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .staging_acquire_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_staging_wait(elapsed: Duration) {
    COUNTERS.staging_waits.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .staging_wait_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_gpu_copy(bytes: usize) {
    COUNTERS.gpu_copies.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .gpu_copy_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
}

#[inline]
pub fn record_queue_submit(dispatches: usize, elapsed: Duration) {
    COUNTERS.queue_submits.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .queue_submit_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
    COUNTERS
        .submitted_dispatches
        .fetch_add(dispatches as u64, Ordering::Relaxed);
}

#[inline]
pub fn record_sync_wait(kind: SyncKind, elapsed: Duration) {
    let ns = ns(elapsed);
    COUNTERS.sync_waits.fetch_add(1, Ordering::Relaxed);
    COUNTERS.sync_wait_ns.fetch_add(ns, Ordering::Relaxed);
    match kind {
        SyncKind::QueueIdle => {
            COUNTERS.queue_idle_waits.fetch_add(1, Ordering::Relaxed);
            COUNTERS.queue_idle_wait_ns.fetch_add(ns, Ordering::Relaxed);
        }
        SyncKind::Fence => {
            COUNTERS.fence_waits.fetch_add(1, Ordering::Relaxed);
            COUNTERS.fence_wait_ns.fetch_add(ns, Ordering::Relaxed);
        }
    }
}

#[inline]
pub fn record_paging_sync_wait(elapsed: Duration) {
    COUNTERS.paging_sync_waits.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .paging_sync_wait_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_splitter_forward(
    cap_start: usize,
    cap_end: usize,
    dispatches: usize,
    resulting_submits: u64,
) {
    COUNTERS.splitter_forwards.fetch_add(1, Ordering::Relaxed);
    if cap_start > 0 || cap_end > 0 {
        COUNTERS
            .splitter_armed_forwards
            .fetch_add(1, Ordering::Relaxed);
    }
    record_cap(cap_start);
    record_cap(cap_end);
    COUNTERS
        .splitter_cap_start_last
        .store(cap_start as u64, Ordering::Relaxed);
    COUNTERS
        .splitter_cap_end_last
        .store(cap_end as u64, Ordering::Relaxed);
    COUNTERS
        .splitter_result_submits
        .fetch_add(resulting_submits, Ordering::Relaxed);
    COUNTERS
        .splitter_result_dispatches
        .fetch_add(dispatches as u64, Ordering::Relaxed);
}

pub fn record_lru_work(stats: LruWorkStats) {
    if !stats.has_activity() {
        return;
    }
    COUNTERS
        .lru_mark_calls
        .fetch_add(stats.mark_calls, Ordering::Relaxed);
    COUNTERS
        .lru_mark_scan_steps
        .fetch_add(stats.mark_scan_steps, Ordering::Relaxed);
    COUNTERS
        .lru_mark_max_scan_steps
        .fetch_max(stats.mark_max_scan_steps, Ordering::Relaxed);
    COUNTERS
        .lru_mark_queue_len_sum
        .fetch_add(stats.mark_queue_len_sum, Ordering::Relaxed);
    COUNTERS
        .lru_mark_max_queue_len
        .fetch_max(stats.mark_max_queue_len, Ordering::Relaxed);
    COUNTERS
        .lru_victim_calls
        .fetch_add(stats.victim_calls, Ordering::Relaxed);
    COUNTERS
        .lru_victim_scan_steps
        .fetch_add(stats.victim_scan_steps, Ordering::Relaxed);
    COUNTERS
        .lru_victim_max_scan_steps
        .fetch_max(stats.victim_max_scan_steps, Ordering::Relaxed);
    COUNTERS
        .lru_victim_queue_len_sum
        .fetch_add(stats.victim_queue_len_sum, Ordering::Relaxed);
    COUNTERS
        .lru_victim_max_queue_len
        .fetch_max(stats.victim_max_queue_len, Ordering::Relaxed);
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        gpu_lookups: load(&COUNTERS.gpu_lookups),
        gpu_lookup_ns: load(&COUNTERS.gpu_lookup_ns),
        gpu_hits: load(&COUNTERS.gpu_hits),
        gpu_misses: load(&COUNTERS.gpu_misses),
        gpu_evictions: load(&COUNTERS.gpu_evictions),
        gpu_eviction_ns: load(&COUNTERS.gpu_eviction_ns),

        host_hits: load(&COUNTERS.host_hits),
        host_misses: load(&COUNTERS.host_misses),
        host_evictions: load(&COUNTERS.host_evictions),
        host_waits: load(&COUNTERS.host_waits),
        host_wait_ns: load(&COUNTERS.host_wait_ns),
        host_bytes: load(&COUNTERS.host_bytes),
        host_reads: load(&COUNTERS.host_reads),
        host_read_bytes: load(&COUNTERS.host_read_bytes),
        host_read_ns: load(&COUNTERS.host_read_ns),
        host_streamed: load(&COUNTERS.host_streamed),
        mmap_fallbacks: load(&COUNTERS.mmap_fallbacks),
        mmap_fallback_bytes: load(&COUNTERS.mmap_fallback_bytes),
        mmap_fallback_ns: load(&COUNTERS.mmap_fallback_ns),

        memcpys: load(&COUNTERS.memcpys),
        memcpy_bytes: load(&COUNTERS.memcpy_bytes),
        memcpy_ns: load(&COUNTERS.memcpy_ns),

        staging_acquires: load(&COUNTERS.staging_acquires),
        staging_acquire_ns: load(&COUNTERS.staging_acquire_ns),
        staging_waits: load(&COUNTERS.staging_waits),
        staging_wait_ns: load(&COUNTERS.staging_wait_ns),

        gpu_copies: load(&COUNTERS.gpu_copies),
        gpu_copy_bytes: load(&COUNTERS.gpu_copy_bytes),

        queue_submits: load(&COUNTERS.queue_submits),
        queue_submit_ns: load(&COUNTERS.queue_submit_ns),
        submitted_dispatches: load(&COUNTERS.submitted_dispatches),

        sync_waits: load(&COUNTERS.sync_waits),
        sync_wait_ns: load(&COUNTERS.sync_wait_ns),
        queue_idle_waits: load(&COUNTERS.queue_idle_waits),
        queue_idle_wait_ns: load(&COUNTERS.queue_idle_wait_ns),
        fence_waits: load(&COUNTERS.fence_waits),
        fence_wait_ns: load(&COUNTERS.fence_wait_ns),
        paging_sync_waits: load(&COUNTERS.paging_sync_waits),
        paging_sync_wait_ns: load(&COUNTERS.paging_sync_wait_ns),

        splitter_forwards: load(&COUNTERS.splitter_forwards),
        splitter_armed_forwards: load(&COUNTERS.splitter_armed_forwards),
        splitter_cap_start_last: load(&COUNTERS.splitter_cap_start_last),
        splitter_cap_end_last: load(&COUNTERS.splitter_cap_end_last),
        splitter_cap_min: load(&COUNTERS.splitter_cap_min),
        splitter_cap_max: load(&COUNTERS.splitter_cap_max),
        splitter_result_submits: load(&COUNTERS.splitter_result_submits),
        splitter_result_dispatches: load(&COUNTERS.splitter_result_dispatches),

        lru_mark_calls: load(&COUNTERS.lru_mark_calls),
        lru_mark_scan_steps: load(&COUNTERS.lru_mark_scan_steps),
        lru_mark_max_scan_steps: load(&COUNTERS.lru_mark_max_scan_steps),
        lru_mark_queue_len_sum: load(&COUNTERS.lru_mark_queue_len_sum),
        lru_mark_max_queue_len: load(&COUNTERS.lru_mark_max_queue_len),
        lru_victim_calls: load(&COUNTERS.lru_victim_calls),
        lru_victim_scan_steps: load(&COUNTERS.lru_victim_scan_steps),
        lru_victim_max_scan_steps: load(&COUNTERS.lru_victim_max_scan_steps),
        lru_victim_queue_len_sum: load(&COUNTERS.lru_victim_queue_len_sum),
        lru_victim_max_queue_len: load(&COUNTERS.lru_victim_max_queue_len),
    }
}

pub fn print_summary_if_enabled() {
    if !enabled() || PRINTED.swap(true, Ordering::Relaxed) {
        return;
    }
    let s = snapshot();
    let mut out = std::io::stderr().lock();
    let _ = writeln!(out);
    let _ = writeln!(out, "== INFR_PAGER_PROFILE summary ==");
    let _ = writeln!(
        out,
        "gpu cache: lookups={} hits={} misses={} hit_rate={:.1}% lookup={} evictions={} eviction_time={}",
        s.gpu_lookups,
        s.gpu_hits,
        s.gpu_misses,
        s.gpu_hit_rate() * 100.0,
        fmt_ns(s.gpu_lookup_ns),
        s.gpu_evictions,
        fmt_ns(s.gpu_eviction_ns),
    );
    let _ = writeln!(
        out,
        "host tier: hits={} misses={} hit_rate={:.1}% evictions={} waits={} wait={} bytes={} reads={} read_bytes={} read_time={} streamed={}",
        s.host_hits,
        s.host_misses,
        s.host_hit_rate() * 100.0,
        s.host_evictions,
        s.host_waits,
        fmt_ns(s.host_wait_ns),
        fmt_bytes(s.host_bytes),
        s.host_reads,
        fmt_bytes(s.host_read_bytes),
        fmt_ns(s.host_read_ns),
        s.host_streamed,
    );
    let _ = writeln!(
        out,
        "mmap/page-cache fallback: count={} bytes={} time={}",
        s.mmap_fallbacks,
        fmt_bytes(s.mmap_fallback_bytes),
        fmt_ns(s.mmap_fallback_ns),
    );
    let _ = writeln!(
        out,
        "host->staging memcpy: count={} bytes={} time={} bw={:.2} GB/s",
        s.memcpys,
        fmt_bytes(s.memcpy_bytes),
        fmt_ns(s.memcpy_ns),
        bandwidth_gbs(s.memcpy_bytes, s.memcpy_ns),
    );
    let _ = writeln!(
        out,
        "staging ring: acquisitions={} acquire_time={} blocked_waits={} blocked_wait={}",
        s.staging_acquires,
        fmt_ns(s.staging_acquire_ns),
        s.staging_waits,
        fmt_ns(s.staging_wait_ns),
    );
    let _ = writeln!(
        out,
        "gpu upload: copies={} bytes={} gpu_copy_device_time=n/a (use INFR_PROF_OPS copy_buffer)",
        s.gpu_copies,
        fmt_bytes(s.gpu_copy_bytes),
    );
    let _ = writeln!(
        out,
        "vulkan submit: submits={} cpu_submit_time={} dispatches={} avg_dispatches_per_submit={:.1}",
        s.queue_submits,
        fmt_ns(s.queue_submit_ns),
        s.submitted_dispatches,
        if s.queue_submits == 0 {
            0.0
        } else {
            s.submitted_dispatches as f64 / s.queue_submits as f64
        },
    );
    let _ = writeln!(
        out,
        "sync: waits={} total={} queue_wait_idle={}({}) fence={}({}) paging_explicit={}({})",
        s.sync_waits,
        fmt_ns(s.sync_wait_ns),
        s.queue_idle_waits,
        fmt_ns(s.queue_idle_wait_ns),
        s.fence_waits,
        fmt_ns(s.fence_wait_ns),
        s.paging_sync_waits,
        fmt_ns(s.paging_sync_wait_ns),
    );
    let cap_range = if s.splitter_cap_min == u64::MAX {
        "unlimited".to_string()
    } else if s.splitter_cap_min == s.splitter_cap_max {
        format!("split/{}", s.splitter_cap_min)
    } else {
        format!("split/{}..{}", s.splitter_cap_min, s.splitter_cap_max)
    };
    let _ = writeln!(
        out,
        "submit splitter: forwards={} armed_forwards={} cap_start={} cap_end={} cap_range={} resulting_submits={} dispatches={}",
        s.splitter_forwards,
        s.splitter_armed_forwards,
        cap_label(s.splitter_cap_start_last),
        cap_label(s.splitter_cap_end_last),
        cap_range,
        s.splitter_result_submits,
        s.splitter_result_dispatches,
    );
    let _ = writeln!(
        out,
        "lru work: mark_mru_calls={} mark_scan_steps={} avg_mark_scan={:.1} max_mark_scan={} avg_mark_queue_len={:.1} max_mark_queue_len={} victim_selects={} victim_scan_steps={} avg_victim_scan={:.1} max_victim_scan={} avg_victim_queue_len={:.1} max_victim_queue_len={}",
        s.lru_mark_calls,
        s.lru_mark_scan_steps,
        s.avg_lru_mark_scan_steps(),
        s.lru_mark_max_scan_steps,
        s.avg_lru_mark_queue_len(),
        s.lru_mark_max_queue_len,
        s.lru_victim_calls,
        s.lru_victim_scan_steps,
        s.avg_lru_victim_scan_steps(),
        s.lru_victim_max_scan_steps,
        s.avg_lru_victim_queue_len(),
        s.lru_victim_max_queue_len,
    );
}

#[inline]
fn load(a: &AtomicU64) -> u64 {
    a.load(Ordering::Relaxed)
}

#[inline]
fn record_cap(cap: usize) {
    if cap == 0 {
        return;
    }
    let cap = cap as u64;
    COUNTERS.splitter_cap_min.fetch_min(cap, Ordering::Relaxed);
    COUNTERS.splitter_cap_max.fetch_max(cap, Ordering::Relaxed);
}

#[inline]
fn ns(d: Duration) -> u64 {
    d.as_nanos().min(u64::MAX as u128) as u64
}

fn rate(part: u64, total: u64) -> f64 {
    if total == 0 {
        1.0
    } else {
        part as f64 / total as f64
    }
}

fn avg(sum: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        sum as f64 / count as f64
    }
}

fn bandwidth_gbs(bytes: u64, ns: u64) -> f64 {
    if ns == 0 {
        0.0
    } else {
        bytes as f64 / ns as f64
    }
}

fn cap_label(cap: u64) -> String {
    if cap == 0 {
        "unlimited".to_string()
    } else {
        format!("split/{cap}")
    }
}

fn fmt_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

fn fmt_ns(ns: u64) -> String {
    if ns >= 10_000_000_000 {
        format!("{:.1}s", ns as f64 / 1e9)
    } else if ns >= 1_000_000_000 {
        format!("{:.2}s", ns as f64 / 1e9)
    } else if ns >= 1_000_000 {
        format!("{:.1}ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.1}us", ns as f64 / 1e3)
    } else {
        format!("{ns}ns")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_are_vacuously_full_before_activity() {
        let s = Snapshot::default();
        assert_eq!(s.gpu_hit_rate(), 1.0);
        assert_eq!(s.host_hit_rate(), 1.0);
    }

    #[test]
    fn bandwidth_uses_decimal_gb_per_second() {
        assert_eq!(bandwidth_gbs(2_000_000_000, 1_000_000_000), 2.0);
        assert_eq!(bandwidth_gbs(10, 0), 0.0);
    }

    #[test]
    fn lru_work_stats_accumulate_scan_and_queue_lengths() {
        let mut s = LruWorkStats::default();
        s.record_mark_mru(10, 3);
        s.record_mark_mru(8, 8);
        s.record_victim_select(10, 1);
        s.record_victim_select(10, 7);

        assert_eq!(s.mark_calls, 2);
        assert_eq!(s.mark_scan_steps, 11);
        assert_eq!(s.mark_max_scan_steps, 8);
        assert_eq!(s.mark_queue_len_sum, 18);
        assert_eq!(s.mark_max_queue_len, 10);
        assert_eq!(s.victim_calls, 2);
        assert_eq!(s.victim_scan_steps, 8);
        assert_eq!(s.victim_max_scan_steps, 7);
        assert_eq!(s.victim_queue_len_sum, 20);
        assert_eq!(s.victim_max_queue_len, 10);
        assert!(s.has_activity());
    }
}
