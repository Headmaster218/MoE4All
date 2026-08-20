//! Backend-neutral accounting for memory-managed runtime resources.
//!
//! This is deliberately policy-free.  Model implementations register their independently
//! evictable units here; the future unified VRAM/RAM/SSD manager can rank snapshots and invoke
//! model-specific transitions without teaching it about chat, embeddings, or vision.

use std::{
    sync::atomic::{AtomicU64, AtomicU8, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

/// What a managed allocation is used for.  Kept model-agnostic so all engines can share a pager.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    ModelWeights,
    EmbeddingWeights,
    VisionWeights,
    KvCache,
    RuntimeScratch,
}

/// The storage tier currently holding the resource's authoritative bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MemoryTier {
    Vram = 0,
    Ram = 1,
    Ssd = 2,
}

impl MemoryTier {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Vram,
            1 => Self::Ram,
            2 => Self::Ssd,
            _ => Self::Ram,
        }
    }
}

/// A cheap point-in-time view consumed by a future unified eviction policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceSnapshot {
    pub id: String,
    pub kind: ResourceKind,
    pub logical_bytes: u64,
    pub resident_bytes: u64,
    pub tier: MemoryTier,
    pub active_requests: u64,
    pub last_access_ms: u64,
    /// Estimated bytes that must be read to make the resource usable after eviction.
    pub reload_bytes: u64,
}

/// Live heat/in-use accounting attached to one independently managed resource.
///
/// It does not evict anything yet.  The important first-step contract is that every model type
/// exposes the same identity, size, tier, heat and active-request protection fields before a
/// cross-model policy is introduced.
pub struct ResourceTracker {
    id: String,
    kind: ResourceKind,
    logical_bytes: u64,
    resident_bytes: AtomicU64,
    tier: AtomicU8,
    active_requests: AtomicU64,
    last_access_ms: AtomicU64,
    reload_bytes: u64,
}

impl ResourceTracker {
    pub fn new(
        id: impl Into<String>,
        kind: ResourceKind,
        logical_bytes: u64,
        resident_bytes: u64,
        tier: MemoryTier,
        reload_bytes: u64,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            logical_bytes,
            resident_bytes: AtomicU64::new(resident_bytes),
            tier: AtomicU8::new(tier as u8),
            active_requests: AtomicU64::new(0),
            last_access_ms: AtomicU64::new(now_ms()),
            reload_bytes,
        }
    }

    /// Mark a request active.  Eviction policy must never move a resource with a non-zero count.
    pub fn acquire(&self) -> ResourceLease<'_> {
        self.last_access_ms.store(now_ms(), Ordering::Relaxed);
        self.active_requests.fetch_add(1, Ordering::AcqRel);
        ResourceLease { tracker: self }
    }

    pub fn set_residency(&self, tier: MemoryTier, resident_bytes: u64) {
        self.tier.store(tier as u8, Ordering::Release);
        self.resident_bytes.store(resident_bytes, Ordering::Release);
        self.last_access_ms.store(now_ms(), Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> ResourceSnapshot {
        ResourceSnapshot {
            id: self.id.clone(),
            kind: self.kind,
            logical_bytes: self.logical_bytes,
            resident_bytes: self.resident_bytes.load(Ordering::Acquire),
            tier: MemoryTier::from_u8(self.tier.load(Ordering::Acquire)),
            active_requests: self.active_requests.load(Ordering::Acquire),
            last_access_ms: self.last_access_ms.load(Ordering::Relaxed),
            reload_bytes: self.reload_bytes,
        }
    }
}

/// RAII pin for an active request.  Dropping it makes the resource evictable again.
pub struct ResourceLease<'a> {
    tracker: &'a ResourceTracker,
}

impl Drop for ResourceLease<'_> {
    fn drop(&mut self) {
        self.tracker.active_requests.fetch_sub(1, Ordering::AcqRel);
        self.tracker
            .last_access_ms
            .store(now_ms(), Ordering::Relaxed);
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_protects_resource_until_drop() {
        let tracker = ResourceTracker::new(
            "embedding:test",
            ResourceKind::EmbeddingWeights,
            100,
            80,
            MemoryTier::Vram,
            100,
        );
        assert_eq!(tracker.snapshot().active_requests, 0);
        {
            let _lease = tracker.acquire();
            assert_eq!(tracker.snapshot().active_requests, 1);
        }
        assert_eq!(tracker.snapshot().active_requests, 0);
    }
}
