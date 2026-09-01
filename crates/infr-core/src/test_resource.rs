//! Process-wide resource-probe overrides for destructive capacity testing.
//!
//! This module is deliberately separate from normal configuration. A profile can only reduce
//! what the platform probes report, and it is installed once before any backend or worker thread
//! exists. Production callers never install one and retain the exact platform behavior.

use std::sync::OnceLock;

/// A synthetic machine shape used by the long-context resource matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TestResourceProfile {
    pub vram_total: u64,
    pub vram_used: u64,
    pub ram_total: u64,
    pub ram_used: u64,
}

impl TestResourceProfile {
    pub fn new(
        vram_total: u64,
        vram_used: u64,
        ram_total: u64,
        ram_used: u64,
    ) -> Result<Self, &'static str> {
        if vram_total == 0 || ram_total == 0 {
            return Err("resource totals must be greater than zero");
        }
        if vram_used >= vram_total {
            return Err("vram_used must be smaller than vram_total");
        }
        if ram_used >= ram_total {
            return Err("ram_used must be smaller than ram_total");
        }
        Ok(Self {
            vram_total,
            vram_used,
            ram_total,
            ram_used,
        })
    }

    pub fn vram_available(&self) -> u64 {
        self.vram_total.saturating_sub(self.vram_used)
    }

    pub fn ram_available(&self) -> u64 {
        self.ram_total.saturating_sub(self.ram_used)
    }

    /// Apply the profile to a Vulkan snapshot. The synthetic room also subtracts allocations
    /// tracked by this backend, so it remains a hard ceiling even when the driver lacks
    /// VK_EXT_memory_budget or the real card is larger than the simulated one.
    pub fn cap_vram(
        &self,
        observed_total: u64,
        observed_available: u64,
        tracked_used: u64,
    ) -> (u64, u64) {
        let total = observed_total.min(self.vram_total);
        let synthetic_available = total
            .saturating_sub(self.vram_used)
            .saturating_sub(tracked_used);
        (total, observed_available.min(synthetic_available))
    }

    /// Apply the profile to a host-memory snapshot. The real probe remains an independent upper
    /// bound, so a profile can never invent RAM that the machine does not currently have.
    pub fn cap_ram(&self, observed_total: u64, observed_available: u64) -> (u64, u64) {
        let total = observed_total.min(self.ram_total);
        let synthetic_available = total.saturating_sub(self.ram_used);
        (total, observed_available.min(synthetic_available))
    }
}

static PROFILE: OnceLock<TestResourceProfile> = OnceLock::new();

/// Install one test profile before any resource probe or backend is constructed.
pub fn install(profile: TestResourceProfile) -> Result<(), String> {
    match PROFILE.set(profile) {
        Ok(()) => Ok(()),
        Err(requested) if PROFILE.get() == Some(&requested) => Ok(()),
        Err(_) => Err("a different test resource profile is already active".into()),
    }
}

/// The active test profile, absent on every normal invocation.
pub fn active() -> Option<TestResourceProfile> {
    PROFILE.get().copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn invalid_shapes_are_rejected() {
        assert!(TestResourceProfile::new(0, 0, 32 * GIB, 10 * GIB).is_err());
        assert!(TestResourceProfile::new(16 * GIB, 16 * GIB, 32 * GIB, 10 * GIB).is_err());
        assert!(TestResourceProfile::new(16 * GIB, 2 * GIB, 32 * GIB, 32 * GIB).is_err());
    }

    #[test]
    fn vram_cap_tracks_backend_allocations_and_never_inflates() {
        let p = TestResourceProfile::new(16 * GIB, 2 * GIB, 32 * GIB, 10 * GIB).unwrap();
        assert_eq!(p.cap_vram(24 * GIB, 20 * GIB, 0), (16 * GIB, 14 * GIB));
        assert_eq!(p.cap_vram(24 * GIB, 15 * GIB, 5 * GIB), (16 * GIB, 9 * GIB));
        assert_eq!(p.cap_vram(12 * GIB, 7 * GIB, 1 * GIB), (12 * GIB, 7 * GIB));
    }

    #[test]
    fn ram_cap_uses_both_real_and_synthetic_headroom() {
        let p = TestResourceProfile::new(16 * GIB, 2 * GIB, 32 * GIB, 10 * GIB).unwrap();
        assert_eq!(p.cap_ram(64 * GIB, 54 * GIB), (32 * GIB, 22 * GIB));
        assert_eq!(p.cap_ram(64 * GIB, 18 * GIB), (32 * GIB, 18 * GIB));
        assert_eq!(p.cap_ram(24 * GIB, 20 * GIB), (24 * GIB, 14 * GIB));
    }
}
