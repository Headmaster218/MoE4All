//! Runtime pager profiling (`INFR_PAGER_PROFILE=1`).
//!
//! This is deliberately separate from `INFR_PROF_OPS`: the latter uses Vulkan timestamp queries and
//! changes the paged ring path by forcing `finish_nowait` to drain. The pager profile stays on host
//! wall-clock timings and aggregate counters so it can be left on for the exact workload shape being
//! investigated.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use std::io::Write as _;

static ENABLED: AtomicBool = AtomicBool::new(false);
static PRINTED: AtomicBool = AtomicBool::new(false);
static COUNTERS: Counters = Counters::new();
static DEVICE_INTERVALS: Mutex<Vec<DeviceInterval>> = Mutex::new(Vec::new());
static PAGED_MOE_LAYERS: Mutex<Vec<PagedMoeLayerStats>> = Mutex::new(Vec::new());

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceIntervalKind {
    MainQueue,
    DedicatedTransfer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendSetupKind {
    Layout,
    PhaseScratch,
    Rope,
    Fusion,
    PagedMoeScan,
}

#[derive(Clone, Copy, Debug)]
struct DeviceInterval {
    kind: DeviceIntervalKind,
    start_tick: u64,
    end_tick: u64,
    valid_bits: u32,
    period_ns: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceTimelineStats {
    pub main_intervals: usize,
    pub dma_intervals: usize,
    pub span_ns: u64,
    pub main_busy_ns: u64,
    pub dma_busy_ns: u64,
    pub overlap_ns: u64,
    pub outside_intervals_ns: u64,
}

/// Lightweight counter sample around one paged-MoE boundary. Unlike [`Snapshot`], this contains
/// only counters needed to explain the decode critical path, so taking it once per layer stays
/// cheap enough for an opt-in production-shaped profile.
#[derive(Clone, Copy, Debug, Default)]
pub struct PagedMoeHotSnapshot {
    gpu_hits: u64,
    gpu_misses: u64,
    gpu_evictions: u64,
    memcpy_bytes: u64,
    memcpy_ns: u64,
    dedicated_transfer_submits: u64,
    dedicated_transfer_bytes: u64,
    dedicated_transfer_submit_cpu_ns: u64,
    dedicated_transfer_slot_wait_ns: u64,
    dedicated_transfer_timeline_wait_ns: u64,
    queue_submits: u64,
    queue_submit_ns: u64,
    command_record_segments: u64,
    command_record_ns: u64,
    command_recorder_acquire_ns: u64,
    paging_sync_waits: u64,
    paging_sync_wait_ns: u64,
    staging_wait_ns: u64,
    device_main_raw_ns: u64,
    device_dma_raw_ns: u64,
}

/// Aggregate for all profiled visits to one model layer. `main_done_ns` and `dma_done_ns` name
/// queue intervals whose timestamp queries became available while crossing this layer boundary;
/// pipelining means they describe work completed at the boundary, not necessarily work issued by
/// the same layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct PagedMoeLayerStats {
    pub calls: u64,
    pub wall_ns: u64,
    pub gpu_hits: u64,
    pub gpu_misses: u64,
    pub gpu_evictions: u64,
    pub memcpy_bytes: u64,
    pub memcpy_ns: u64,
    pub dedicated_transfer_submits: u64,
    pub dedicated_transfer_bytes: u64,
    pub dedicated_transfer_submit_cpu_ns: u64,
    pub dedicated_transfer_slot_wait_ns: u64,
    pub dedicated_transfer_timeline_wait_ns: u64,
    pub queue_submits: u64,
    pub queue_submit_ns: u64,
    pub command_record_segments: u64,
    pub command_record_ns: u64,
    pub command_recorder_acquire_ns: u64,
    pub paging_sync_waits: u64,
    pub paging_sync_wait_ns: u64,
    pub staging_wait_ns: u64,
    pub main_done_ns: u64,
    pub dma_done_ns: u64,
}

impl PagedMoeLayerStats {
    pub fn saturating_sub(self, before: Self) -> Self {
        Self {
            calls: self.calls.saturating_sub(before.calls),
            wall_ns: self.wall_ns.saturating_sub(before.wall_ns),
            gpu_hits: self.gpu_hits.saturating_sub(before.gpu_hits),
            gpu_misses: self.gpu_misses.saturating_sub(before.gpu_misses),
            gpu_evictions: self.gpu_evictions.saturating_sub(before.gpu_evictions),
            memcpy_bytes: self.memcpy_bytes.saturating_sub(before.memcpy_bytes),
            memcpy_ns: self.memcpy_ns.saturating_sub(before.memcpy_ns),
            dedicated_transfer_submits: self
                .dedicated_transfer_submits
                .saturating_sub(before.dedicated_transfer_submits),
            dedicated_transfer_bytes: self
                .dedicated_transfer_bytes
                .saturating_sub(before.dedicated_transfer_bytes),
            dedicated_transfer_submit_cpu_ns: self
                .dedicated_transfer_submit_cpu_ns
                .saturating_sub(before.dedicated_transfer_submit_cpu_ns),
            dedicated_transfer_slot_wait_ns: self
                .dedicated_transfer_slot_wait_ns
                .saturating_sub(before.dedicated_transfer_slot_wait_ns),
            dedicated_transfer_timeline_wait_ns: self
                .dedicated_transfer_timeline_wait_ns
                .saturating_sub(before.dedicated_transfer_timeline_wait_ns),
            queue_submits: self.queue_submits.saturating_sub(before.queue_submits),
            queue_submit_ns: self.queue_submit_ns.saturating_sub(before.queue_submit_ns),
            command_record_segments: self
                .command_record_segments
                .saturating_sub(before.command_record_segments),
            command_record_ns: self
                .command_record_ns
                .saturating_sub(before.command_record_ns),
            command_recorder_acquire_ns: self
                .command_recorder_acquire_ns
                .saturating_sub(before.command_recorder_acquire_ns),
            paging_sync_waits: self
                .paging_sync_waits
                .saturating_sub(before.paging_sync_waits),
            paging_sync_wait_ns: self
                .paging_sync_wait_ns
                .saturating_sub(before.paging_sync_wait_ns),
            staging_wait_ns: self.staging_wait_ns.saturating_sub(before.staging_wait_ns),
            main_done_ns: self.main_done_ns.saturating_sub(before.main_done_ns),
            dma_done_ns: self.dma_done_ns.saturating_sub(before.dma_done_ns),
        }
    }
}

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
    prefetch_windows: AtomicU64,
    prefetch_compute_live_at_start: AtomicU64,
    prefetch_compute_live_after_prepare: AtomicU64,
    prefetch_compute_live_after_enqueue: AtomicU64,
    prefetch_push_bytes: AtomicU64,
    prefetch_push_prepare_ns: AtomicU64,
    prefetch_push_enqueue_ns: AtomicU64,
    prefetch_confirmed_overlap_ns: AtomicU64,
    prefetch_bytes_submitted_while_compute_live: AtomicU64,
    decode_prefetch_calibrations: AtomicU64,
    decode_prefetch_candidates: AtomicU64,
    decode_prefetch_vram_resident: AtomicU64,
    decode_prefetch_host_loading: AtomicU64,
    decode_prefetch_ram_experts: AtomicU64,
    decode_prefetch_ram_bytes: AtomicU64,
    decode_prefetch_ram_ns: AtomicU64,
    decode_prefetch_ssd_jobs: AtomicU64,
    decode_prefetch_ssd_blocks: AtomicU64,
    decode_prefetch_deadline_stops: AtomicU64,
    decode_prefetch_overruns: AtomicU64,
    decode_prefetch_target_finishes: AtomicU64,
    decode_prefetch_target_finish_ns: AtomicU64,
    decode_prefetch_quiesce_lock_waits: AtomicU64,
    decode_prefetch_quiesce_lock_wait_ns: AtomicU64,

    gpu_copies: AtomicU64,
    gpu_copy_bytes: AtomicU64,
    dedicated_transfer_submits: AtomicU64,
    dedicated_transfer_regions: AtomicU64,
    dedicated_transfer_bytes: AtomicU64,
    dedicated_transfer_submit_cpu_ns: AtomicU64,
    dedicated_transfer_gpu_timed_submits: AtomicU64,
    dedicated_transfer_gpu_timed_bytes: AtomicU64,
    dedicated_transfer_gpu_ns: AtomicU64,
    dedicated_transfer_slot_waits: AtomicU64,
    dedicated_transfer_slot_wait_ns: AtomicU64,
    dedicated_transfer_timeline_waits: AtomicU64,
    dedicated_transfer_timeline_wait_ns: AtomicU64,
    device_main_raw_ns: AtomicU64,
    device_dma_raw_ns: AtomicU64,

    queue_submits: AtomicU64,
    queue_submit_ns: AtomicU64,
    submitted_dispatches: AtomicU64,
    command_record_segments: AtomicU64,
    command_record_dispatches: AtomicU64,
    command_record_ns: AtomicU64,
    command_recorder_acquires: AtomicU64,
    command_recorder_acquire_ns: AtomicU64,
    backend_executes: AtomicU64,
    backend_execute_ns: AtomicU64,
    backend_setups: AtomicU64,
    backend_setup_ns: AtomicU64,
    backend_setup_layout_ns: AtomicU64,
    backend_setup_phase_scratch_ns: AtomicU64,
    backend_setup_rope_ns: AtomicU64,
    backend_setup_fusion_ns: AtomicU64,
    backend_setup_paged_moe_scan_ns: AtomicU64,

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

    ple_gathers: AtomicU64,
    ple_tokens: AtomicU64,
    ple_row_requests: AtomicU64,
    ple_unique_rows: AtomicU64,
    ple_logical_pages: AtomicU64,
    ple_output_bytes: AtomicU64,
    ple_parallel_gathers: AtomicU64,
    ple_plan_ns: AtomicU64,
    ple_work_ns: AtomicU64,
    ple_waits: AtomicU64,
    ple_wait_ns: AtomicU64,
    ple_uploads: AtomicU64,
    ple_upload_bytes: AtomicU64,
    ple_upload_ns: AtomicU64,

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
            prefetch_windows: AtomicU64::new(0),
            prefetch_compute_live_at_start: AtomicU64::new(0),
            prefetch_compute_live_after_prepare: AtomicU64::new(0),
            prefetch_compute_live_after_enqueue: AtomicU64::new(0),
            prefetch_push_bytes: AtomicU64::new(0),
            prefetch_push_prepare_ns: AtomicU64::new(0),
            prefetch_push_enqueue_ns: AtomicU64::new(0),
            prefetch_confirmed_overlap_ns: AtomicU64::new(0),
            prefetch_bytes_submitted_while_compute_live: AtomicU64::new(0),
            decode_prefetch_calibrations: AtomicU64::new(0),
            decode_prefetch_candidates: AtomicU64::new(0),
            decode_prefetch_vram_resident: AtomicU64::new(0),
            decode_prefetch_host_loading: AtomicU64::new(0),
            decode_prefetch_ram_experts: AtomicU64::new(0),
            decode_prefetch_ram_bytes: AtomicU64::new(0),
            decode_prefetch_ram_ns: AtomicU64::new(0),
            decode_prefetch_ssd_jobs: AtomicU64::new(0),
            decode_prefetch_ssd_blocks: AtomicU64::new(0),
            decode_prefetch_deadline_stops: AtomicU64::new(0),
            decode_prefetch_overruns: AtomicU64::new(0),
            decode_prefetch_target_finishes: AtomicU64::new(0),
            decode_prefetch_target_finish_ns: AtomicU64::new(0),
            decode_prefetch_quiesce_lock_waits: AtomicU64::new(0),
            decode_prefetch_quiesce_lock_wait_ns: AtomicU64::new(0),

            gpu_copies: AtomicU64::new(0),
            gpu_copy_bytes: AtomicU64::new(0),
            dedicated_transfer_submits: AtomicU64::new(0),
            dedicated_transfer_regions: AtomicU64::new(0),
            dedicated_transfer_bytes: AtomicU64::new(0),
            dedicated_transfer_submit_cpu_ns: AtomicU64::new(0),
            dedicated_transfer_gpu_timed_submits: AtomicU64::new(0),
            dedicated_transfer_gpu_timed_bytes: AtomicU64::new(0),
            dedicated_transfer_gpu_ns: AtomicU64::new(0),
            dedicated_transfer_slot_waits: AtomicU64::new(0),
            dedicated_transfer_slot_wait_ns: AtomicU64::new(0),
            dedicated_transfer_timeline_waits: AtomicU64::new(0),
            dedicated_transfer_timeline_wait_ns: AtomicU64::new(0),
            device_main_raw_ns: AtomicU64::new(0),
            device_dma_raw_ns: AtomicU64::new(0),

            queue_submits: AtomicU64::new(0),
            queue_submit_ns: AtomicU64::new(0),
            submitted_dispatches: AtomicU64::new(0),
            command_record_segments: AtomicU64::new(0),
            command_record_dispatches: AtomicU64::new(0),
            command_record_ns: AtomicU64::new(0),
            command_recorder_acquires: AtomicU64::new(0),
            command_recorder_acquire_ns: AtomicU64::new(0),
            backend_executes: AtomicU64::new(0),
            backend_execute_ns: AtomicU64::new(0),
            backend_setups: AtomicU64::new(0),
            backend_setup_ns: AtomicU64::new(0),
            backend_setup_layout_ns: AtomicU64::new(0),
            backend_setup_phase_scratch_ns: AtomicU64::new(0),
            backend_setup_rope_ns: AtomicU64::new(0),
            backend_setup_fusion_ns: AtomicU64::new(0),
            backend_setup_paged_moe_scan_ns: AtomicU64::new(0),

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

            ple_gathers: AtomicU64::new(0),
            ple_tokens: AtomicU64::new(0),
            ple_row_requests: AtomicU64::new(0),
            ple_unique_rows: AtomicU64::new(0),
            ple_logical_pages: AtomicU64::new(0),
            ple_output_bytes: AtomicU64::new(0),
            ple_parallel_gathers: AtomicU64::new(0),
            ple_plan_ns: AtomicU64::new(0),
            ple_work_ns: AtomicU64::new(0),
            ple_waits: AtomicU64::new(0),
            ple_wait_ns: AtomicU64::new(0),
            ple_uploads: AtomicU64::new(0),
            ple_upload_bytes: AtomicU64::new(0),
            ple_upload_ns: AtomicU64::new(0),

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
    pub prefetch_windows: u64,
    pub prefetch_compute_live_at_start: u64,
    pub prefetch_compute_live_after_prepare: u64,
    pub prefetch_compute_live_after_enqueue: u64,
    pub prefetch_push_bytes: u64,
    pub prefetch_push_prepare_ns: u64,
    pub prefetch_push_enqueue_ns: u64,
    pub prefetch_confirmed_overlap_ns: u64,
    pub prefetch_bytes_submitted_while_compute_live: u64,
    pub decode_prefetch_calibrations: u64,
    pub decode_prefetch_candidates: u64,
    pub decode_prefetch_vram_resident: u64,
    pub decode_prefetch_host_loading: u64,
    pub decode_prefetch_ram_experts: u64,
    pub decode_prefetch_ram_bytes: u64,
    pub decode_prefetch_ram_ns: u64,
    pub decode_prefetch_ssd_jobs: u64,
    pub decode_prefetch_ssd_blocks: u64,
    pub decode_prefetch_deadline_stops: u64,
    pub decode_prefetch_overruns: u64,
    pub decode_prefetch_target_finishes: u64,
    pub decode_prefetch_target_finish_ns: u64,
    pub decode_prefetch_quiesce_lock_waits: u64,
    pub decode_prefetch_quiesce_lock_wait_ns: u64,

    pub gpu_copies: u64,
    pub gpu_copy_bytes: u64,
    pub dedicated_transfer_submits: u64,
    pub dedicated_transfer_regions: u64,
    pub dedicated_transfer_bytes: u64,
    pub dedicated_transfer_submit_cpu_ns: u64,
    pub dedicated_transfer_gpu_timed_submits: u64,
    pub dedicated_transfer_gpu_timed_bytes: u64,
    pub dedicated_transfer_gpu_ns: u64,
    pub dedicated_transfer_slot_waits: u64,
    pub dedicated_transfer_slot_wait_ns: u64,
    pub dedicated_transfer_timeline_waits: u64,
    pub dedicated_transfer_timeline_wait_ns: u64,

    pub queue_submits: u64,
    pub queue_submit_ns: u64,
    pub submitted_dispatches: u64,
    pub command_record_segments: u64,
    pub command_record_dispatches: u64,
    pub command_record_ns: u64,
    pub command_recorder_acquires: u64,
    pub command_recorder_acquire_ns: u64,
    pub backend_executes: u64,
    pub backend_execute_ns: u64,
    pub backend_setups: u64,
    pub backend_setup_ns: u64,
    pub backend_setup_layout_ns: u64,
    pub backend_setup_phase_scratch_ns: u64,
    pub backend_setup_rope_ns: u64,
    pub backend_setup_fusion_ns: u64,
    pub backend_setup_paged_moe_scan_ns: u64,

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

    pub ple_gathers: u64,
    pub ple_tokens: u64,
    pub ple_row_requests: u64,
    pub ple_unique_rows: u64,
    pub ple_logical_pages: u64,
    pub ple_output_bytes: u64,
    pub ple_parallel_gathers: u64,
    pub ple_plan_ns: u64,
    pub ple_work_ns: u64,
    pub ple_waits: u64,
    pub ple_wait_ns: u64,
    pub ple_uploads: u64,
    pub ple_upload_bytes: u64,
    pub ple_upload_ns: u64,

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
    record_memcpy_batch(1, bytes, elapsed);
}

#[inline]
pub fn record_memcpy_batch(copies: usize, bytes: usize, elapsed: Duration) {
    COUNTERS.memcpys.fetch_add(copies as u64, Ordering::Relaxed);
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

/// Record host-side work and non-blocking fence samples for one hit-first demand-copy window.
/// `confirmed_overlap` is deliberately a lower bound: only an interval whose fence was live at
/// both ends is counted. Dedicated-queue timestamps account for the copy's GPU duration separately.
#[inline]
pub fn record_prefetch_window(compute_live_at_start: bool, compute_live_after_enqueue: bool) {
    record_prefetch_window_detailed(
        compute_live_at_start,
        compute_live_after_enqueue,
        compute_live_after_enqueue,
        0,
        Duration::ZERO,
        Duration::ZERO,
    );
}

#[inline]
pub fn record_prefetch_window_detailed(
    compute_live_at_start: bool,
    compute_live_after_prepare: bool,
    compute_live_after_enqueue: bool,
    bytes: usize,
    prepare_elapsed: Duration,
    enqueue_elapsed: Duration,
) {
    COUNTERS.prefetch_windows.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .prefetch_push_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
    COUNTERS
        .prefetch_push_prepare_ns
        .fetch_add(ns(prepare_elapsed), Ordering::Relaxed);
    COUNTERS
        .prefetch_push_enqueue_ns
        .fetch_add(ns(enqueue_elapsed), Ordering::Relaxed);
    if compute_live_at_start {
        COUNTERS
            .prefetch_compute_live_at_start
            .fetch_add(1, Ordering::Relaxed);
    }
    if compute_live_after_prepare {
        COUNTERS
            .prefetch_compute_live_after_prepare
            .fetch_add(1, Ordering::Relaxed);
    }
    if compute_live_after_enqueue {
        COUNTERS
            .prefetch_compute_live_after_enqueue
            .fetch_add(1, Ordering::Relaxed);
        COUNTERS
            .prefetch_bytes_submitted_while_compute_live
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }
    let mut confirmed_overlap = Duration::ZERO;
    if compute_live_at_start && compute_live_after_prepare {
        confirmed_overlap = confirmed_overlap.saturating_add(prepare_elapsed);
    }
    if compute_live_after_prepare && compute_live_after_enqueue {
        confirmed_overlap = confirmed_overlap.saturating_add(enqueue_elapsed);
    }
    COUNTERS
        .prefetch_confirmed_overlap_ns
        .fetch_add(ns(confirmed_overlap), Ordering::Relaxed);
}

#[inline]
pub fn record_decode_prefetch_calibration() {
    COUNTERS
        .decode_prefetch_calibrations
        .fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn record_decode_prefetch_candidate() {
    COUNTERS
        .decode_prefetch_candidates
        .fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn record_decode_prefetch_vram_resident() {
    COUNTERS
        .decode_prefetch_vram_resident
        .fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn record_decode_prefetch_host_loading() {
    COUNTERS
        .decode_prefetch_host_loading
        .fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn record_decode_prefetch_ram(bytes: usize) {
    COUNTERS
        .decode_prefetch_ram_experts
        .fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .decode_prefetch_ram_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
}

#[inline]
pub fn record_decode_prefetch_ram_timed(bytes: usize, elapsed: Duration) {
    record_decode_prefetch_ram(bytes);
    COUNTERS
        .decode_prefetch_ram_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_decode_prefetch_ssd(blocks: usize) {
    COUNTERS
        .decode_prefetch_ssd_jobs
        .fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .decode_prefetch_ssd_blocks
        .fetch_add(blocks as u64, Ordering::Relaxed);
}

#[inline]
pub fn record_decode_prefetch_deadline_stop() {
    COUNTERS
        .decode_prefetch_deadline_stops
        .fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn record_decode_prefetch_overrun() {
    COUNTERS
        .decode_prefetch_overruns
        .fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn record_decode_prefetch_target_finish(elapsed: Duration) {
    COUNTERS
        .decode_prefetch_target_finishes
        .fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .decode_prefetch_target_finish_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_decode_prefetch_quiesce_lock_wait(elapsed: Duration) {
    COUNTERS
        .decode_prefetch_quiesce_lock_waits
        .fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .decode_prefetch_quiesce_lock_wait_ns
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
pub fn record_dedicated_transfer_submit(bytes: u64, regions: u64) {
    COUNTERS
        .dedicated_transfer_submits
        .fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .dedicated_transfer_regions
        .fetch_add(regions, Ordering::Relaxed);
    COUNTERS
        .dedicated_transfer_bytes
        .fetch_add(bytes, Ordering::Relaxed);
}

#[inline]
pub fn record_dedicated_transfer_submit_cpu(elapsed: Duration) {
    COUNTERS
        .dedicated_transfer_submit_cpu_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_dedicated_transfer_gpu_time(bytes: u64, elapsed: Duration) {
    COUNTERS
        .dedicated_transfer_gpu_timed_submits
        .fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .dedicated_transfer_gpu_timed_bytes
        .fetch_add(bytes, Ordering::Relaxed);
    COUNTERS
        .dedicated_transfer_gpu_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_dedicated_transfer_slot_wait(elapsed: Duration) {
    COUNTERS
        .dedicated_transfer_slot_waits
        .fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .dedicated_transfer_slot_wait_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_dedicated_transfer_timeline_wait(elapsed: Duration) {
    COUNTERS
        .dedicated_transfer_timeline_waits
        .fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .dedicated_transfer_timeline_wait_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

/// Record one queue interval in the physical device timestamp domain. This vector exists only for
/// opt-in profiling and is folded once at shutdown; normal execution never takes its mutex.
pub fn record_device_interval(
    kind: DeviceIntervalKind,
    start_tick: u64,
    end_tick: u64,
    valid_bits: u32,
    period_ns: f32,
) {
    if valid_bits == 0 || !period_ns.is_finite() || period_ns <= 0.0 {
        return;
    }
    let mask = if valid_bits >= 64 {
        u64::MAX as u128
    } else {
        (1u128 << valid_bits) - 1
    };
    let modulus = mask + 1;
    let start = start_tick as u128 & mask;
    let mut end = end_tick as u128 & mask;
    if end < start {
        end += modulus;
    }
    let elapsed_ns =
        ((end.saturating_sub(start) as f64 * period_ns as f64).min(u64::MAX as f64)) as u64;
    match kind {
        DeviceIntervalKind::MainQueue => &COUNTERS.device_main_raw_ns,
        DeviceIntervalKind::DedicatedTransfer => &COUNTERS.device_dma_raw_ns,
    }
    .fetch_add(elapsed_ns, Ordering::Relaxed);
    DEVICE_INTERVALS.lock().unwrap().push(DeviceInterval {
        kind,
        start_tick,
        end_tick,
        valid_bits,
        period_ns,
    });
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
pub fn record_command_recording(dispatches: usize, elapsed: Duration) {
    COUNTERS
        .command_record_segments
        .fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .command_record_dispatches
        .fetch_add(dispatches as u64, Ordering::Relaxed);
    COUNTERS
        .command_record_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_command_recorder_acquire(elapsed: Duration) {
    COUNTERS
        .command_recorder_acquires
        .fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .command_recorder_acquire_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_backend_execute(elapsed: Duration) {
    COUNTERS.backend_executes.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .backend_execute_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_backend_setup(elapsed: Duration) {
    COUNTERS.backend_setups.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .backend_setup_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_backend_setup_part(kind: BackendSetupKind, elapsed: Duration) {
    let counter = match kind {
        BackendSetupKind::Layout => &COUNTERS.backend_setup_layout_ns,
        BackendSetupKind::PhaseScratch => &COUNTERS.backend_setup_phase_scratch_ns,
        BackendSetupKind::Rope => &COUNTERS.backend_setup_rope_ns,
        BackendSetupKind::Fusion => &COUNTERS.backend_setup_fusion_ns,
        BackendSetupKind::PagedMoeScan => &COUNTERS.backend_setup_paged_moe_scan_ns,
    };
    counter.fetch_add(ns(elapsed), Ordering::Relaxed);
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

#[allow(clippy::too_many_arguments)]
pub fn record_ple_gather(
    tokens: usize,
    row_requests: usize,
    unique_rows: usize,
    logical_pages: usize,
    output_bytes: usize,
    parallel: bool,
    plan_elapsed: Duration,
    work_elapsed: Duration,
) {
    COUNTERS.ple_gathers.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .ple_tokens
        .fetch_add(tokens as u64, Ordering::Relaxed);
    COUNTERS
        .ple_row_requests
        .fetch_add(row_requests as u64, Ordering::Relaxed);
    COUNTERS
        .ple_unique_rows
        .fetch_add(unique_rows as u64, Ordering::Relaxed);
    COUNTERS
        .ple_logical_pages
        .fetch_add(logical_pages as u64, Ordering::Relaxed);
    COUNTERS
        .ple_output_bytes
        .fetch_add(output_bytes as u64, Ordering::Relaxed);
    if parallel {
        COUNTERS
            .ple_parallel_gathers
            .fetch_add(1, Ordering::Relaxed);
    }
    COUNTERS
        .ple_plan_ns
        .fetch_add(ns(plan_elapsed), Ordering::Relaxed);
    COUNTERS
        .ple_work_ns
        .fetch_add(ns(work_elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_ple_wait(elapsed: Duration) {
    COUNTERS.ple_waits.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .ple_wait_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
}

#[inline]
pub fn record_ple_upload(bytes: usize, elapsed: Duration) {
    COUNTERS.ple_uploads.fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .ple_upload_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
    COUNTERS
        .ple_upload_ns
        .fetch_add(ns(elapsed), Ordering::Relaxed);
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

/// Read the counters needed to attribute one paged-MoE boundary. Call only when [`active`] is
/// true; keeping that branch at the caller avoids dozens of relaxed loads in normal execution.
pub fn paged_moe_hot_snapshot() -> PagedMoeHotSnapshot {
    PagedMoeHotSnapshot {
        gpu_hits: load(&COUNTERS.gpu_hits),
        gpu_misses: load(&COUNTERS.gpu_misses),
        gpu_evictions: load(&COUNTERS.gpu_evictions),
        memcpy_bytes: load(&COUNTERS.memcpy_bytes),
        memcpy_ns: load(&COUNTERS.memcpy_ns),
        dedicated_transfer_submits: load(&COUNTERS.dedicated_transfer_submits),
        dedicated_transfer_bytes: load(&COUNTERS.dedicated_transfer_bytes),
        dedicated_transfer_submit_cpu_ns: load(&COUNTERS.dedicated_transfer_submit_cpu_ns),
        dedicated_transfer_slot_wait_ns: load(&COUNTERS.dedicated_transfer_slot_wait_ns),
        dedicated_transfer_timeline_wait_ns: load(&COUNTERS.dedicated_transfer_timeline_wait_ns),
        queue_submits: load(&COUNTERS.queue_submits),
        queue_submit_ns: load(&COUNTERS.queue_submit_ns),
        command_record_segments: load(&COUNTERS.command_record_segments),
        command_record_ns: load(&COUNTERS.command_record_ns),
        command_recorder_acquire_ns: load(&COUNTERS.command_recorder_acquire_ns),
        paging_sync_waits: load(&COUNTERS.paging_sync_waits),
        paging_sync_wait_ns: load(&COUNTERS.paging_sync_wait_ns),
        staging_wait_ns: load(&COUNTERS.staging_wait_ns),
        device_main_raw_ns: load(&COUNTERS.device_main_raw_ns),
        device_dma_raw_ns: load(&COUNTERS.device_dma_raw_ns),
    }
}

/// Attribute host work and queue intervals completed while crossing one paged-MoE layer boundary.
pub fn record_paged_moe_layer(
    layer: u32,
    wall: Duration,
    before: PagedMoeHotSnapshot,
    after: PagedMoeHotSnapshot,
) {
    let mut layers = PAGED_MOE_LAYERS.lock().unwrap();
    let layer = layer as usize;
    if layers.len() <= layer {
        layers.resize(layer + 1, PagedMoeLayerStats::default());
    }
    let stats = &mut layers[layer];
    let delta = |new: u64, old: u64| new.saturating_sub(old);
    stats.calls += 1;
    stats.wall_ns = stats.wall_ns.saturating_add(ns(wall));
    stats.gpu_hits += delta(after.gpu_hits, before.gpu_hits);
    stats.gpu_misses += delta(after.gpu_misses, before.gpu_misses);
    stats.gpu_evictions += delta(after.gpu_evictions, before.gpu_evictions);
    stats.memcpy_bytes += delta(after.memcpy_bytes, before.memcpy_bytes);
    stats.memcpy_ns += delta(after.memcpy_ns, before.memcpy_ns);
    stats.dedicated_transfer_submits += delta(
        after.dedicated_transfer_submits,
        before.dedicated_transfer_submits,
    );
    stats.dedicated_transfer_bytes += delta(
        after.dedicated_transfer_bytes,
        before.dedicated_transfer_bytes,
    );
    stats.dedicated_transfer_submit_cpu_ns += delta(
        after.dedicated_transfer_submit_cpu_ns,
        before.dedicated_transfer_submit_cpu_ns,
    );
    stats.dedicated_transfer_slot_wait_ns += delta(
        after.dedicated_transfer_slot_wait_ns,
        before.dedicated_transfer_slot_wait_ns,
    );
    stats.dedicated_transfer_timeline_wait_ns += delta(
        after.dedicated_transfer_timeline_wait_ns,
        before.dedicated_transfer_timeline_wait_ns,
    );
    stats.queue_submits += delta(after.queue_submits, before.queue_submits);
    stats.queue_submit_ns += delta(after.queue_submit_ns, before.queue_submit_ns);
    stats.command_record_segments += delta(
        after.command_record_segments,
        before.command_record_segments,
    );
    stats.command_record_ns += delta(after.command_record_ns, before.command_record_ns);
    stats.command_recorder_acquire_ns += delta(
        after.command_recorder_acquire_ns,
        before.command_recorder_acquire_ns,
    );
    stats.paging_sync_waits += delta(after.paging_sync_waits, before.paging_sync_waits);
    stats.paging_sync_wait_ns += delta(after.paging_sync_wait_ns, before.paging_sync_wait_ns);
    stats.staging_wait_ns += delta(after.staging_wait_ns, before.staging_wait_ns);
    stats.main_done_ns += delta(after.device_main_raw_ns, before.device_main_raw_ns);
    stats.dma_done_ns += delta(after.device_dma_raw_ns, before.device_dma_raw_ns);
}

pub fn paged_moe_layer_snapshot() -> Vec<PagedMoeLayerStats> {
    PAGED_MOE_LAYERS.lock().unwrap().clone()
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
        prefetch_windows: load(&COUNTERS.prefetch_windows),
        prefetch_compute_live_at_start: load(&COUNTERS.prefetch_compute_live_at_start),
        prefetch_compute_live_after_prepare: load(&COUNTERS.prefetch_compute_live_after_prepare),
        prefetch_compute_live_after_enqueue: load(&COUNTERS.prefetch_compute_live_after_enqueue),
        prefetch_push_bytes: load(&COUNTERS.prefetch_push_bytes),
        prefetch_push_prepare_ns: load(&COUNTERS.prefetch_push_prepare_ns),
        prefetch_push_enqueue_ns: load(&COUNTERS.prefetch_push_enqueue_ns),
        prefetch_confirmed_overlap_ns: load(&COUNTERS.prefetch_confirmed_overlap_ns),
        prefetch_bytes_submitted_while_compute_live: load(
            &COUNTERS.prefetch_bytes_submitted_while_compute_live,
        ),
        decode_prefetch_calibrations: load(&COUNTERS.decode_prefetch_calibrations),
        decode_prefetch_candidates: load(&COUNTERS.decode_prefetch_candidates),
        decode_prefetch_vram_resident: load(&COUNTERS.decode_prefetch_vram_resident),
        decode_prefetch_host_loading: load(&COUNTERS.decode_prefetch_host_loading),
        decode_prefetch_ram_experts: load(&COUNTERS.decode_prefetch_ram_experts),
        decode_prefetch_ram_bytes: load(&COUNTERS.decode_prefetch_ram_bytes),
        decode_prefetch_ram_ns: load(&COUNTERS.decode_prefetch_ram_ns),
        decode_prefetch_ssd_jobs: load(&COUNTERS.decode_prefetch_ssd_jobs),
        decode_prefetch_ssd_blocks: load(&COUNTERS.decode_prefetch_ssd_blocks),
        decode_prefetch_deadline_stops: load(&COUNTERS.decode_prefetch_deadline_stops),
        decode_prefetch_overruns: load(&COUNTERS.decode_prefetch_overruns),
        decode_prefetch_target_finishes: load(&COUNTERS.decode_prefetch_target_finishes),
        decode_prefetch_target_finish_ns: load(&COUNTERS.decode_prefetch_target_finish_ns),
        decode_prefetch_quiesce_lock_waits: load(&COUNTERS.decode_prefetch_quiesce_lock_waits),
        decode_prefetch_quiesce_lock_wait_ns: load(&COUNTERS.decode_prefetch_quiesce_lock_wait_ns),

        gpu_copies: load(&COUNTERS.gpu_copies),
        gpu_copy_bytes: load(&COUNTERS.gpu_copy_bytes),
        dedicated_transfer_submits: load(&COUNTERS.dedicated_transfer_submits),
        dedicated_transfer_regions: load(&COUNTERS.dedicated_transfer_regions),
        dedicated_transfer_bytes: load(&COUNTERS.dedicated_transfer_bytes),
        dedicated_transfer_submit_cpu_ns: load(&COUNTERS.dedicated_transfer_submit_cpu_ns),
        dedicated_transfer_gpu_timed_submits: load(&COUNTERS.dedicated_transfer_gpu_timed_submits),
        dedicated_transfer_gpu_timed_bytes: load(&COUNTERS.dedicated_transfer_gpu_timed_bytes),
        dedicated_transfer_gpu_ns: load(&COUNTERS.dedicated_transfer_gpu_ns),
        dedicated_transfer_slot_waits: load(&COUNTERS.dedicated_transfer_slot_waits),
        dedicated_transfer_slot_wait_ns: load(&COUNTERS.dedicated_transfer_slot_wait_ns),
        dedicated_transfer_timeline_waits: load(&COUNTERS.dedicated_transfer_timeline_waits),
        dedicated_transfer_timeline_wait_ns: load(&COUNTERS.dedicated_transfer_timeline_wait_ns),

        queue_submits: load(&COUNTERS.queue_submits),
        queue_submit_ns: load(&COUNTERS.queue_submit_ns),
        submitted_dispatches: load(&COUNTERS.submitted_dispatches),
        command_record_segments: load(&COUNTERS.command_record_segments),
        command_record_dispatches: load(&COUNTERS.command_record_dispatches),
        command_record_ns: load(&COUNTERS.command_record_ns),
        command_recorder_acquires: load(&COUNTERS.command_recorder_acquires),
        command_recorder_acquire_ns: load(&COUNTERS.command_recorder_acquire_ns),
        backend_executes: load(&COUNTERS.backend_executes),
        backend_execute_ns: load(&COUNTERS.backend_execute_ns),
        backend_setups: load(&COUNTERS.backend_setups),
        backend_setup_ns: load(&COUNTERS.backend_setup_ns),
        backend_setup_layout_ns: load(&COUNTERS.backend_setup_layout_ns),
        backend_setup_phase_scratch_ns: load(&COUNTERS.backend_setup_phase_scratch_ns),
        backend_setup_rope_ns: load(&COUNTERS.backend_setup_rope_ns),
        backend_setup_fusion_ns: load(&COUNTERS.backend_setup_fusion_ns),
        backend_setup_paged_moe_scan_ns: load(&COUNTERS.backend_setup_paged_moe_scan_ns),

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

        ple_gathers: load(&COUNTERS.ple_gathers),
        ple_tokens: load(&COUNTERS.ple_tokens),
        ple_row_requests: load(&COUNTERS.ple_row_requests),
        ple_unique_rows: load(&COUNTERS.ple_unique_rows),
        ple_logical_pages: load(&COUNTERS.ple_logical_pages),
        ple_output_bytes: load(&COUNTERS.ple_output_bytes),
        ple_parallel_gathers: load(&COUNTERS.ple_parallel_gathers),
        ple_plan_ns: load(&COUNTERS.ple_plan_ns),
        ple_work_ns: load(&COUNTERS.ple_work_ns),
        ple_waits: load(&COUNTERS.ple_waits),
        ple_wait_ns: load(&COUNTERS.ple_wait_ns),
        ple_uploads: load(&COUNTERS.ple_uploads),
        ple_upload_bytes: load(&COUNTERS.ple_upload_bytes),
        ple_upload_ns: load(&COUNTERS.ple_upload_ns),

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
        "host memcpy / ReBAR push: count={} bytes={} time={} bw={:.2} GiB/s",
        s.memcpys,
        fmt_bytes(s.memcpy_bytes),
        fmt_ns(s.memcpy_ns),
        bandwidth_gib_per_sec(s.memcpy_bytes, s.memcpy_ns),
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
        "hit-first demand overlap: windows={} bytes={} prepare={} enqueue={} confirmed_host_overlap_lower_bound={} compute_live=start {} ({:.1}%) / after_prepare {} ({:.1}%) / after_enqueue {} ({:.1}%) bytes_enqueued_before_compute_done={}",
        s.prefetch_windows,
        fmt_bytes(s.prefetch_push_bytes),
        fmt_ns(s.prefetch_push_prepare_ns),
        fmt_ns(s.prefetch_push_enqueue_ns),
        fmt_ns(s.prefetch_confirmed_overlap_ns),
        s.prefetch_compute_live_at_start,
        if s.prefetch_windows == 0 {
            0.0
        } else {
            100.0 * s.prefetch_compute_live_at_start as f64 / s.prefetch_windows as f64
        },
        s.prefetch_compute_live_after_prepare,
        if s.prefetch_windows == 0 {
            0.0
        } else {
            100.0 * s.prefetch_compute_live_after_prepare as f64 / s.prefetch_windows as f64
        },
        s.prefetch_compute_live_after_enqueue,
        if s.prefetch_windows == 0 {
            0.0
        } else {
            100.0 * s.prefetch_compute_live_after_enqueue as f64 / s.prefetch_windows as f64
        },
        fmt_bytes(s.prefetch_bytes_submitted_while_compute_live),
    );
    let _ = writeln!(
        out,
        "decode expert prefetch: calibrations={} candidates={} vram_resident={} host_loading={} ram_experts={} ram_bytes={} ram_transfer_wall={} ram_effective_bw={:.2} GiB/s ssd_jobs={} ssd_blocks={} deadline_stops={} overruns={} target_finishes={} finish_wall={} quiesce_lock_waits={} quiesce_lock_wait={}",
        s.decode_prefetch_calibrations,
        s.decode_prefetch_candidates,
        s.decode_prefetch_vram_resident,
        s.decode_prefetch_host_loading,
        s.decode_prefetch_ram_experts,
        fmt_bytes(s.decode_prefetch_ram_bytes),
        fmt_ns(s.decode_prefetch_ram_ns),
        bandwidth_gib_per_sec(s.decode_prefetch_ram_bytes, s.decode_prefetch_ram_ns),
        s.decode_prefetch_ssd_jobs,
        s.decode_prefetch_ssd_blocks,
        s.decode_prefetch_deadline_stops,
        s.decode_prefetch_overruns,
        s.decode_prefetch_target_finishes,
        fmt_ns(s.decode_prefetch_target_finish_ns),
        s.decode_prefetch_quiesce_lock_waits,
        fmt_ns(s.decode_prefetch_quiesce_lock_wait_ns),
    );
    let _ = writeln!(
        out,
        "gpu upload: logical_copies={} bytes={}",
        s.gpu_copies,
        fmt_bytes(s.gpu_copy_bytes),
    );
    let _ = writeln!(
        out,
        "dedicated DMA: submits={} regions={} bytes={} cpu_submit_time={} gpu_timed_submits={} gpu_timed_bytes={} gpu_copy_time={} gpu_bw={:.2} GiB/s slot_reuse_waits={} slot_reuse_wait={} explicit_timeline_waits={} explicit_timeline_wait={}",
        s.dedicated_transfer_submits,
        s.dedicated_transfer_regions,
        fmt_bytes(s.dedicated_transfer_bytes),
        fmt_ns(s.dedicated_transfer_submit_cpu_ns),
        s.dedicated_transfer_gpu_timed_submits,
        fmt_bytes(s.dedicated_transfer_gpu_timed_bytes),
        fmt_ns(s.dedicated_transfer_gpu_ns),
        bandwidth_gib_per_sec(
            s.dedicated_transfer_gpu_timed_bytes,
            s.dedicated_transfer_gpu_ns,
        ),
        s.dedicated_transfer_slot_waits,
        fmt_ns(s.dedicated_transfer_slot_wait_ns),
        s.dedicated_transfer_timeline_waits,
        fmt_ns(s.dedicated_transfer_timeline_wait_ns),
    );
    let timeline = device_timeline_snapshot();
    let _ = writeln!(
        out,
        "device queue timeline: span={} main_intervals={} main_busy={} dma_intervals={} dma_busy={} concurrent_overlap={} dma_hidden={:.1}% visible_dma_tail={} outside_timestamped_intervals={}",
        fmt_ns(timeline.span_ns),
        timeline.main_intervals,
        fmt_ns(timeline.main_busy_ns),
        timeline.dma_intervals,
        fmt_ns(timeline.dma_busy_ns),
        fmt_ns(timeline.overlap_ns),
        if timeline.dma_busy_ns == 0 {
            0.0
        } else {
            100.0 * timeline.overlap_ns as f64 / timeline.dma_busy_ns as f64
        },
        fmt_ns(timeline.dma_busy_ns.saturating_sub(timeline.overlap_ns)),
        fmt_ns(timeline.outside_intervals_ns),
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
        "host orchestration: backend_executes={} execute_wall={} avg_execute={} setup_calls={} setup_time={} avg_setup={} recorder_acquires={} acquire_time={} avg_acquire={} recorded_segments={} record_lifetime={} record_dispatches={} avg_record_per_segment={}",
        s.backend_executes,
        fmt_ns(s.backend_execute_ns),
        fmt_ns(s.backend_execute_ns.checked_div(s.backend_executes).unwrap_or(0)),
        s.backend_setups,
        fmt_ns(s.backend_setup_ns),
        fmt_ns(s.backend_setup_ns.checked_div(s.backend_setups).unwrap_or(0)),
        s.command_recorder_acquires,
        fmt_ns(s.command_recorder_acquire_ns),
        fmt_ns(
            s.command_recorder_acquire_ns
                .checked_div(s.command_recorder_acquires)
                .unwrap_or(0)
        ),
        s.command_record_segments,
        fmt_ns(s.command_record_ns),
        s.command_record_dispatches,
        fmt_ns(
            s.command_record_ns
                .checked_div(s.command_record_segments)
                .unwrap_or(0)
        ),
    );
    let _ = writeln!(
        out,
        "backend setup detail: layout={} phase_scratch={} rope={} fusion={} paged_moe_scan={} other={}",
        fmt_ns(s.backend_setup_layout_ns),
        fmt_ns(s.backend_setup_phase_scratch_ns),
        fmt_ns(s.backend_setup_rope_ns),
        fmt_ns(s.backend_setup_fusion_ns),
        fmt_ns(s.backend_setup_paged_moe_scan_ns),
        fmt_ns(s.backend_setup_ns.saturating_sub(
            s.backend_setup_layout_ns
                .saturating_add(s.backend_setup_phase_scratch_ns)
                .saturating_add(s.backend_setup_rope_ns)
                .saturating_add(s.backend_setup_fusion_ns)
                .saturating_add(s.backend_setup_paged_moe_scan_ns)
        )),
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
        "qwen3.8 PLE: gathers={} tokens={} row_requests={} unique_rows={} duplicate_rows={} logical_4KiB_pages={} output={} parallel_gathers={} plan={} mmap_dequant={} residual_waits={} residual_wait={} uploads={} upload_bytes={} upload_time={}",
        s.ple_gathers,
        s.ple_tokens,
        s.ple_row_requests,
        s.ple_unique_rows,
        s.ple_row_requests.saturating_sub(s.ple_unique_rows),
        s.ple_logical_pages,
        fmt_bytes(s.ple_output_bytes),
        s.ple_parallel_gathers,
        fmt_ns(s.ple_plan_ns),
        fmt_ns(s.ple_work_ns),
        s.ple_waits,
        fmt_ns(s.ple_wait_ns),
        s.ple_uploads,
        fmt_bytes(s.ple_upload_bytes),
        fmt_ns(s.ple_upload_ns),
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

fn bandwidth_gib_per_sec(bytes: u64, ns: u64) -> f64 {
    if ns == 0 {
        0.0
    } else {
        const GIB: f64 = (1u64 << 30) as f64;
        bytes as f64 * 1_000_000_000.0 / ns as f64 / GIB
    }
}

/// Fold all recorded queue intervals into cumulative busy/overlap totals. Callers may subtract two
/// snapshots taken at idle cohort boundaries to attribute device work without resetting the
/// process-wide profiler.
pub fn device_timeline_snapshot() -> DeviceTimelineStats {
    let intervals = DEVICE_INTERVALS.lock().unwrap();
    if intervals.is_empty() {
        return DeviceTimelineStats::default();
    }
    let valid_bits = intervals.iter().map(|i| i.valid_bits).min().unwrap_or(0);
    let period_ns = intervals[0].period_ns;
    if valid_bits == 0
        || intervals
            .iter()
            .any(|i| (i.period_ns - period_ns).abs() > f32::EPSILON * period_ns.abs().max(1.0))
    {
        return DeviceTimelineStats::default();
    }
    let mask = if valid_bits >= 64 {
        u64::MAX as u128
    } else {
        (1u128 << valid_bits) - 1
    };
    let modulus = mask + 1;
    let normalize = |interval: &DeviceInterval| {
        let start = interval.start_tick as u128 & mask;
        let mut end = interval.end_tick as u128 & mask;
        if end < start {
            end += modulus;
        }
        let start_ns = (start as f64 * period_ns as f64) as u128;
        let end_ns = (end as f64 * period_ns as f64) as u128;
        (start_ns, end_ns)
    };
    let main_raw: Vec<_> = intervals
        .iter()
        .filter(|i| i.kind == DeviceIntervalKind::MainQueue)
        .map(normalize)
        .collect();
    let dma_raw: Vec<_> = intervals
        .iter()
        .filter(|i| i.kind == DeviceIntervalKind::DedicatedTransfer)
        .map(normalize)
        .collect();
    let main_intervals = main_raw.len();
    let dma_intervals = dma_raw.len();
    let main = merge_intervals(main_raw);
    let dma = merge_intervals(dma_raw);
    let main_busy = interval_duration(&main);
    let dma_busy = interval_duration(&dma);
    let overlap = interval_intersection_duration(&main, &dma);
    let first = main
        .iter()
        .chain(dma.iter())
        .map(|&(start, _)| start)
        .min()
        .unwrap_or(0);
    let last = main
        .iter()
        .chain(dma.iter())
        .map(|&(_, end)| end)
        .max()
        .unwrap_or(first);
    let span = last.saturating_sub(first);
    let union_busy = main_busy.saturating_add(dma_busy).saturating_sub(overlap);
    DeviceTimelineStats {
        main_intervals,
        dma_intervals,
        span_ns: span.min(u64::MAX as u128) as u64,
        main_busy_ns: main_busy.min(u64::MAX as u128) as u64,
        dma_busy_ns: dma_busy.min(u64::MAX as u128) as u64,
        overlap_ns: overlap.min(u64::MAX as u128) as u64,
        outside_intervals_ns: span.saturating_sub(union_busy).min(u64::MAX as u128) as u64,
    }
}

fn merge_intervals(mut intervals: Vec<(u128, u128)>) -> Vec<(u128, u128)> {
    intervals.retain(|(start, end)| end >= start);
    intervals.sort_unstable_by_key(|&(start, end)| (start, end));
    let mut merged: Vec<(u128, u128)> = Vec::with_capacity(intervals.len());
    for (start, end) in intervals {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}

fn interval_duration(intervals: &[(u128, u128)]) -> u128 {
    intervals
        .iter()
        .map(|&(start, end)| end.saturating_sub(start))
        .sum()
}

fn interval_intersection_duration(a: &[(u128, u128)], b: &[(u128, u128)]) -> u128 {
    let (mut ai, mut bi, mut total) = (0usize, 0usize, 0u128);
    while ai < a.len() && bi < b.len() {
        let start = a[ai].0.max(b[bi].0);
        let end = a[ai].1.min(b[bi].1);
        total = total.saturating_add(end.saturating_sub(start));
        if a[ai].1 <= b[bi].1 {
            ai += 1;
        } else {
            bi += 1;
        }
    }
    total
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
    fn bandwidth_uses_binary_gib_per_second() {
        assert_eq!(bandwidth_gib_per_sec(2u64 << 30, 1_000_000_000), 2.0);
        assert_eq!(bandwidth_gib_per_sec(10, 0), 0.0);
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

    #[test]
    fn device_interval_union_and_overlap_are_not_double_counted() {
        let a = merge_intervals(vec![(0, 10), (8, 20), (30, 40)]);
        let b = merge_intervals(vec![(5, 12), (18, 35)]);
        assert_eq!(a, vec![(0, 20), (30, 40)]);
        assert_eq!(interval_duration(&a), 30);
        assert_eq!(interval_duration(&b), 24);
        assert_eq!(interval_intersection_duration(&a, &b), 14);
    }
}
