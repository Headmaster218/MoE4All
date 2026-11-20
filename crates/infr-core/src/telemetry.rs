//! Wire DTOs for the engine's out-of-band telemetry endpoint (`GET /internal/stats`).
//!
//! These live in `infr-core` because the producer (`infr-vulkan` / `infr-llama`) and the consumer
//! (`infr-server`) both depend on it and neither may depend on the other.
//!
//! Every field is a point-in-time read of a counter that already exists. Sampling this must never
//! take the GPU baton or drain a statistics window: a monitoring poll that stalls decode, or steals
//! the terminal reporter's interval, is worse than no monitoring at all.

use serde::Serialize;

/// One process's resource occupancy.
#[derive(Clone, Debug, Default, Serialize)]
pub struct EngineReport {
    /// Device-local budget as the driver reports it. `None` on a backend without a Vulkan device.
    pub vram: Option<VramReport>,
    /// This process's resident set size. `None` on a platform with no working-set probe — an
    /// absent block, never a zero that reads as a measurement.
    pub rss_bytes: Option<u64>,
    pub experts: ExpertReport,
    /// The paged-KV frame corridor. `None` when `kv.paged` is off.
    pub kv_pool: Option<KvPoolReport>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct VramReport {
    pub total: u64,
    pub available: u64,
    /// Bytes this process has allocated through the backend's own accounting.
    pub tracked_bytes: u64,
    /// False when the driver has no `VK_EXT_memory_budget`, so `available` is the whole heap
    /// rather than a live free figure.
    pub live_budget: bool,
    pub uma: bool,
}

/// MoE expert residency. Counts are EXPERT SLOTS, not bytes: one slot holds one expert of one
/// size class, so pools with different `slot_bytes` are summed as slots and reported separately
/// nowhere — a consumer that wants bytes multiplies by the class size it already knows.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ExpertReport {
    /// False for a model whose experts all fit in VRAM (no pager session at all).
    pub paged: bool,
    pub vram_resident: u64,
    pub vram_slots: u64,
    /// Bounded-RAM tier shadows. Zero when the permanent full host store is used instead.
    pub ram_resident: u64,
    pub ram_slots: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Residency authority is on the GPU, so the host LRU counters above are deferred to the last
    /// layout handoff rather than live. Reading them anyway is honest only with this flag set.
    pub gpu_owned: bool,
    /// The whole expert payload is permanently in RAM (no bounded host tier), so `ram_resident`
    /// cannot describe it — the two `host_store_*` fields are the counts that do.
    pub full_host_store: bool,
    pub host_store_bytes: u64,
    /// Expert tensors the permanent host store holds (every layer × every role).
    pub host_store_experts: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct KvPoolReport {
    pub page_tokens: u64,
    pub frame_bytes: u64,
    pub tiers: Vec<KvTierReport>,
}

impl KvPoolReport {
    /// Token capacity still free in the VRAM tier — the number an operator asks for as "how much
    /// context is left in the pool".
    pub fn free_vram_tokens(&self) -> u64 {
        self.tiers
            .iter()
            .find(|t| t.tier == "vram")
            .map_or(0, |t| t.free_frames.saturating_mul(self.page_tokens))
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct KvTierReport {
    /// `vram` / `ram` / `disk`.
    pub tier: &'static str,
    pub capacity_frames: u64,
    pub held_frames: u64,
    pub free_frames: u64,
}