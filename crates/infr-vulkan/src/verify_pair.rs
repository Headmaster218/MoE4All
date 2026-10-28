//! Bounded VERIFY pair metadata, separate from the resident journal and notice ABI.
//! Phase-retained sidecars with one execute lease; never reset live metadata on the host.
use std::sync::Arc;

use infr_core::{backend::Buffer, error::Result, DType};

use crate::{as_vk_buf, be, unified::UnifiedVramClass, VulkanBackend};

#[cfg(test)]
#[path = "verify_pair_tests.rs"]
mod contract_tests;

pub(crate) const WORDS: usize = 128;
pub(crate) const BYTES: usize = WORDS * 4;
pub(crate) const COUNTS: usize = 64;
pub(crate) const VALID: usize = 67;
pub(crate) const COMMANDS: usize = 68;
pub(crate) const CONFIG: usize = 104;
pub(crate) const MAGIC: u32 = 0x5650_0001;

pub(crate) fn shared_iq2_fusion_staged_policy(
    enabled: bool,
    baseline_nr: u32,
    sg_selected: bool,
    buffer_selected: bool,
) -> bool {
    enabled && baseline_nr == 8 && !sg_selected && !buffer_selected
}

/// Baseline-only optimization, independently controlled from pair reuse. Module performance
/// and bit-exactness qualified on RX 7900 XTX; no performance claim for other devices.
pub(crate) fn shared_iq2_nr8_eligible(
    enabled: bool,
    causal_verify: bool,
    dtype: DType,
    shape: [usize; 3], // rows, input width, output width
    routed_used: usize,
    x_per_slot: bool,
    selected_baseline: &str,
) -> bool {
    enabled
        && causal_verify
        && dtype == DType::Iq2S
        && shape == [4, 2560, 640]
        && routed_used == 10
        && !x_per_slot
        && selected_baseline == "native_idm_iq2s_paged_shexp"
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    Gate = 0,
    Up = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Full = 0,
    Hits = 1,
    Misses = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Path {
    Baseline = 0,
    Pair = 1,
}

pub(crate) fn command_byte_offset(role: Role, phase: Phase, path: Path) -> u64 {
    ((COMMANDS + role as usize * 18 + phase as usize * 6 + path as usize * 3) * 4) as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cohort {
    pub(crate) verify: bool,
    pub(crate) independent_rows: bool,
    pub(crate) rows: usize,
    pub(crate) used: usize,
    pub(crate) experts: usize,
    pub(crate) in_f: usize,
    pub(crate) out_f: usize,
    pub(crate) shared_q8: bool,
}

impl Cohort {
    pub(crate) fn eligible(self) -> bool {
        self.verify
            && !self.independent_rows
            && self.rows == 4
            && self.used == 10
            && self.experts == 512
            && self.in_f == 2560
            && self.out_f == 640
    }

    fn physical_used(self) -> usize {
        self.used + usize::from(self.shared_q8)
    }
}

/// Capture the already selected baseline, after grid-buffer and SG precedence is resolved.
/// Never infer this from the device's NR setting or change that setting to enroll a role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RolePolicy {
    pub(crate) nr: u32,
    pub(crate) threshold: u32,
    pub(crate) shared_q8: bool,
    groups: [u32; 2],
}

impl RolePolicy {
    pub(crate) fn measured(dtype: DType, selected_baseline: &str, cohort: Cohort) -> Option<Self> {
        if !cohort.eligible() {
            return None;
        }
        let (nr, threshold) = match (dtype, selected_baseline, cohort.shared_q8) {
            (DType::Iq2S, "native_idm_iq2s_nr8_paged", false) => (8, 10),
            (DType::Iq3S, "native_idm_iq3s_paged", false)
            | (DType::Iq3S, "native_idm_iq3s_paged_shexp", true) => (1, 6),
            // Shared IQ2_S NR8 is qualified only as a baseline, not as a paired kernel.
            // Its independently controlled NR1 fallback also remains unpaired.
            _ => return None,
        };
        let groups =
            (cohort.rows * cohort.physical_used() * cohort.out_f.div_ceil(nr as usize)) as u32;
        Some(Self {
            nr,
            threshold,
            shared_q8: cohort.shared_q8,
            groups: [groups, 1],
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayerPlan {
    cohort: Cohort,
    roles: [Option<RolePolicy>; 2],
}

impl LayerPlan {
    pub(crate) fn new(cohort: Cohort, roles: [Option<RolePolicy>; 2]) -> Option<Self> {
        if !cohort.eligible() || roles.iter().all(Option::is_none) {
            return None;
        }
        // A policy belongs to this exact physical layout, not a nearby graph's cached plan.
        if roles.iter().flatten().any(|role| {
            role.shared_q8 != cohort.shared_q8
                || role.groups
                    != [
                        (cohort.rows
                            * cohort.physical_used()
                            * cohort.out_f.div_ceil(role.nr as usize))
                            as u32,
                        1,
                    ]
        }) {
            return None;
        }
        Some(Self { cohort, roles })
    }

    fn initial_words(self) -> [u32; WORDS] {
        let mut words = [0; WORDS];
        words[..64].fill(u32::MAX);
        // GPU-owned mates/counts/commands start invalid/NOP. Only this tail is host-written.
        words[CONFIG..CONFIG + 5].copy_from_slice(&[
            MAGIC,
            self.cohort.rows as u32,
            self.cohort.used as u32,
            self.cohort.experts as u32,
            (1 << self.cohort.used) - 1,
        ]);
        for (role, policy) in self.roles.into_iter().enumerate() {
            if let Some(policy) = policy {
                let base = CONFIG + 5 + role * 3;
                words[base..base + 3].copy_from_slice(&[
                    policy.threshold,
                    policy.groups[0],
                    policy.groups[1],
                ]);
            }
        }
        words[CONFIG + 11] = self.cohort.physical_used() as u32;
        words
    }
}

/// Allocate ALL layer sidecars before candidate begin/freeze can promise slot ownership.
/// A window is phase-retained and leased to one blocking execute, never a journal ordinal.
pub(crate) struct Layer {
    pub(crate) buffer: Arc<dyn Buffer>,
    pub(crate) cohort: Cohort,
    pub(crate) roles: [Option<RolePolicy>; 2],
}

pub(crate) struct VerifyPairWindow {
    layers: Vec<Option<Arc<Layer>>>,
}

struct CachedWindow {
    plans: Vec<Option<LayerPlan>>,
    window: Arc<VerifyPairWindow>,
}

#[derive(Default)]
pub(crate) struct WindowCache {
    entries: Vec<CachedWindow>,
}

impl WindowCache {
    const CAPACITY: usize = 4;

    /// Called only during execute setup, under RuntimePhaseArena and the unified execution
    /// gate. Previous executes have drained; Arc checks additionally reject outstanding CPU
    /// leases or recorder keepalives, but are not a replacement for that GPU completion proof.
    pub(crate) fn acquire(
        &mut self,
        plans: &[Option<LayerPlan>],
        frozen: bool,
        create: impl FnOnce() -> Result<Option<Arc<VerifyPairWindow>>>,
    ) -> Result<Option<Arc<VerifyPairWindow>>> {
        if plans.iter().all(Option::is_none) {
            return Ok(None);
        }
        if let Some(index) = self.entries.iter().position(|entry| entry.plans == plans) {
            let entry = &self.entries[index];
            if Arc::strong_count(&entry.window) != 1
                || entry.window.layers.iter().flatten().any(|layer| {
                    Arc::strong_count(layer) != 1 || Arc::strong_count(&layer.buffer) != 1
                })
            {
                return Err(be(
                    "VERIFY pair window still has an execute lease or queued reader",
                ));
            }
            let entry = self.entries.remove(index);
            let lease = Arc::clone(&entry.window);
            self.entries.push(entry);
            return Ok(Some(lease));
        }
        if frozen {
            return Err(be("VERIFY pair cache miss after candidate freeze"));
        }
        // Keep the previous cache intact on failed allocation or a panicking constructor.
        let Some(window) = create()? else {
            return Ok(None);
        };
        if self.entries.len() == Self::CAPACITY {
            self.entries.remove(0);
        }
        self.entries.push(CachedWindow {
            plans: plans.to_vec(),
            window: Arc::clone(&window),
        });
        Ok(Some(window))
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

impl VerifyPairWindow {
    pub(crate) fn allocate(
        backend: &VulkanBackend,
        plans: &[Option<LayerPlan>],
    ) -> Result<Option<Self>> {
        if !backend.cfg().kernels.vulkan.verify_pair || plans.iter().all(Option::is_none) {
            return Ok(None);
        }
        let mut layers = Vec::with_capacity(plans.len());
        for plan in plans {
            let Some(plan) = plan else {
                layers.push(None);
                continue;
            };
            let buffer: Arc<dyn Buffer> =
                Arc::from(backend.alloc_unified(BYTES, UnifiedVramClass::LlmRuntime)?);
            let mapped = as_vk_buf(buffer.as_ref())?
                .mapped_ptr()
                .ok_or_else(|| be("VERIFY pair sidecar needs coherent unified metadata"))?;
            let words = plan.initial_words();
            unsafe {
                std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), mapped, BYTES);
            }
            layers.push(Some(Arc::new(Layer {
                buffer,
                cohort: plan.cohort,
                roles: plan.roles,
            })));
        }
        Ok(Some(Self { layers }))
    }

    /// Clone this Arc into every recorder that produces or consumes the metadata, including
    /// CPU-miss resume. Keep this window with the execute until all resumed work is recorded.
    /// Recorder::retain_buffer keeps submitted readers alive through their fence retirement.
    pub(crate) fn layer(&self, layer: usize) -> Option<Arc<Layer>> {
        self.layers.get(layer)?.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_iq2_fusion_preserves_staged_baseline_priority_and_opt_out() {
        assert!(shared_iq2_fusion_staged_policy(true, 8, false, false));
        for nr in [1, 2, 4, 8] {
            for enabled in [false, true] {
                for sg in [false, true] {
                    for buffer in [false, true] {
                        assert_eq!(
                            shared_iq2_fusion_staged_policy(enabled, nr, sg, buffer),
                            enabled && nr == 8 && !sg && !buffer
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn shared_iq2_nr8_only_replaces_exact_causal_verify_nr1_baseline() {
        let baseline = "native_idm_iq2s_paged_shexp";
        let eligible = |enabled, verify, dtype, shape, used, per_slot, name| {
            shared_iq2_nr8_eligible(enabled, verify, dtype, shape, used, per_slot, name)
        };
        assert!(eligible(
            true,
            true,
            DType::Iq2S,
            [4, 2560, 640],
            10,
            false,
            baseline
        ));
        assert!(!eligible(
            false,
            true,
            DType::Iq2S,
            [4, 2560, 640],
            10,
            false,
            baseline
        ));
        // Ordinary and independent rows pass false, even when their dimensions match.
        assert!(!eligible(
            true,
            false,
            DType::Iq2S,
            [4, 2560, 640],
            10,
            false,
            baseline
        ));
        assert!(!eligible(
            true,
            true,
            DType::Iq3S,
            [4, 2560, 640],
            10,
            false,
            baseline
        ));
        assert!(!eligible(
            true,
            true,
            DType::Iq2S,
            [4, 2560, 640],
            8,
            false,
            baseline
        ));
        assert!(!eligible(
            true,
            true,
            DType::Iq2S,
            [4, 2560, 640],
            10,
            true,
            baseline
        ));
        for shape in [
            [1, 2560, 640],
            [2, 2560, 640],
            [3, 2560, 640],
            [8, 2560, 640],
            [4, 2048, 640],
            [4, 2560, 512],
            [4, 640, 2560],
        ] {
            assert!(!eligible(
                true,
                true,
                DType::Iq2S,
                shape,
                10,
                false,
                baseline
            ));
        }
        for name in [
            "native_idm_iq2s_buffer_paged_shexp",
            "native_idm_iq2s_buffer_nr8_paged_shexp",
            "native_idm_iq2s_sg8_paged_shexp",
            "native_idm_iq2s_nr8_paged_shexp",
            "native_idm_iq2s_paged",
        ] {
            assert!(!eligible(
                true,
                true,
                DType::Iq2S,
                [4, 2560, 640],
                10,
                false,
                name
            ));
        }
    }

    #[test]
    fn shared_iq2_nr8_spv_getter_does_not_expand_other_variants() {
        for (dtype, nr) in [
            (DType::Iq2S, 1),
            (DType::Iq2S, 4),
            (DType::Iq3S, 8),
            (DType::Iq4Nl, 8),
        ] {
            assert!(crate::gemm::native_idm_grid_nr_paged_shared_build_spv(dtype, nr).is_none());
        }
        let (name, words) =
            crate::gemm::native_idm_grid_nr_paged_shared_build_spv(DType::Iq2S, 8).unwrap();
        assert_eq!(name, "native_idm_iq2s_nr8_paged_shexp");
        assert_eq!(words.first().copied(), Some(0x0723_0203));
    }

    fn cohort() -> Cohort {
        Cohort {
            verify: true,
            independent_rows: false,
            rows: 4,
            used: 10,
            experts: 512,
            in_f: 2560,
            out_f: 640,
            shared_q8: false,
        }
    }

    #[test]
    fn only_exact_measured_baseline_variants_enroll() {
        let c = cohort();
        assert_eq!(
            RolePolicy::measured(DType::Iq2S, "native_idm_iq2s_nr8_paged", c)
                .unwrap()
                .threshold,
            10
        );
        assert_eq!(
            RolePolicy::measured(DType::Iq3S, "native_idm_iq3s_paged", c)
                .unwrap()
                .threshold,
            6
        );
        for name in [
            "native_idm_iq2s_paged",
            "native_idm_iq2s_nr2_paged",
            "native_idm_iq2s_nr4_paged",
            "native_idm_iq2s_buffer_nr8_paged",
            "native_idm_iq2s_sg8_paged",
        ] {
            assert!(
                RolePolicy::measured(DType::Iq2S, name, c).is_none(),
                "{name}"
            );
        }
        for name in [
            "native_idm_iq3s_buffer_paged",
            "native_idm_iq3s_buffer_nr8_paged",
            "native_idm_iq3s_sg2_paged",
            "native_idm_iq3s_nr8_paged",
        ] {
            assert!(
                RolePolicy::measured(DType::Iq3S, name, c).is_none(),
                "{name}"
            );
        }
        assert!(RolePolicy::measured(DType::Iq4Nl, "native_idm_iq3s_paged", c).is_none());
    }

    #[test]
    fn graph_scope_does_not_expand_implicitly() {
        for c in [
            Cohort {
                verify: false,
                ..cohort()
            },
            Cohort {
                independent_rows: true,
                ..cohort()
            },
            Cohort {
                rows: 3,
                ..cohort()
            },
            Cohort {
                used: 8,
                ..cohort()
            },
            Cohort {
                experts: 256,
                ..cohort()
            },
            Cohort {
                in_f: 2048,
                ..cohort()
            },
            Cohort {
                out_f: 512,
                ..cohort()
            },
        ] {
            assert!(RolePolicy::measured(DType::Iq2S, "native_idm_iq2s_nr8_paged", c).is_none());
        }
    }

    #[test]
    fn mixed_roles_keep_independent_thresholds_and_dispatch_geometry() {
        let c = cohort();
        let gate = RolePolicy::measured(DType::Iq2S, "native_idm_iq2s_nr8_paged", c);
        let up = RolePolicy::measured(DType::Iq3S, "native_idm_iq3s_paged", c);
        let words = LayerPlan::new(c, [gate, up]).unwrap().initial_words();
        assert_eq!(&words[109..115], &[10, 3200, 1, 6, 25600, 1]);
        assert_eq!(words[115], 10);
        assert!(words[COMMANDS..CONFIG].iter().all(|&v| v == 0));
        assert_eq!(words[VALID], 0);
    }

    #[test]
    fn shared_stride_is_not_the_routed_id_stride() {
        let c = Cohort {
            shared_q8: true,
            ..cohort()
        };
        assert!(RolePolicy::measured(DType::Iq2S, "native_idm_iq2s_paged_shexp", c).is_none());
        assert!(RolePolicy::measured(DType::Iq2S, "native_idm_iq2s_nr8_paged_shexp", c).is_none());
        let up = RolePolicy::measured(DType::Iq3S, "native_idm_iq3s_paged_shexp", c);
        let words = LayerPlan::new(c, [None, up]).unwrap().initial_words();
        assert_eq!(&words[109..115], &[0, 0, 0, 6, 28160, 1]);
        assert_eq!(words[106], 10);
        assert_eq!(words[115], 11);
        let unshared = RolePolicy::measured(DType::Iq3S, "native_idm_iq3s_paged", cohort());
        assert!(LayerPlan::new(c, [None, unshared]).is_none());
    }

    #[test]
    fn command_triples_are_disjoint_aligned_and_outside_journal() {
        let mut offsets = Vec::new();
        for role in [Role::Gate, Role::Up] {
            for phase in [Phase::Full, Phase::Hits, Phase::Misses] {
                for path in [Path::Baseline, Path::Pair] {
                    let offset = command_byte_offset(role, phase, path);
                    assert_eq!(offset % 4, 0);
                    assert!(offset >= (COMMANDS * 4) as u64);
                    assert!(offset + 12 <= (CONFIG * 4) as u64);
                    offsets.push(offset);
                }
            }
        }
        offsets.sort_unstable();
        offsets.dedup();
        assert_eq!(offsets.len(), 12);
        assert_eq!(BYTES, 512);
    }

    fn cache_key(layers: usize) -> Vec<Option<LayerPlan>> {
        let c = cohort();
        let role = RolePolicy::measured(DType::Iq3S, "native_idm_iq3s_paged", c);
        vec![LayerPlan::new(c, [role, role]); layers]
    }

    fn cpu_window() -> Result<Option<Arc<VerifyPairWindow>>> {
        Ok(Some(Arc::new(VerifyPairWindow { layers: Vec::new() })))
    }

    #[test]
    fn cache_reuses_exact_keys_without_reinitializing_or_allocating() {
        let mut cache = WindowCache::default();
        let key = cache_key(47);
        let first = cache.acquire(&key, false, cpu_window).unwrap().unwrap();
        let identity = Arc::as_ptr(&first);
        drop(first);
        for _ in 0..32 {
            let next = cache
                .acquire(&key, false, || {
                    panic!("warm hit allocated or reset metadata")
                })
                .unwrap()
                .unwrap();
            assert_eq!(Arc::as_ptr(&next), identity);
        }
        assert_eq!(cache.entries.len(), 1);
        let mut changed = key.clone();
        changed[0].as_mut().unwrap().roles[0] = None;
        assert_ne!(key, changed);
        let other = cache.acquire(&changed, false, cpu_window).unwrap().unwrap();
        assert_ne!(Arc::as_ptr(&other), identity);
        let mut shifted = key.clone();
        shifted.insert(0, None);
        assert_ne!(
            key, shifted,
            "exact op positions are part of the structural key"
        );
    }

    #[test]
    fn cache_frozen_miss_and_live_lease_never_allocate_or_evict() {
        let mut cache = WindowCache::default();
        let key = cache_key(1);
        let lease = cache.acquire(&key, false, cpu_window).unwrap().unwrap();
        assert!(cache
            .acquire(&key, false, || panic!("live lease allocated"))
            .is_err());
        drop(lease);
        assert!(cache
            .acquire(&cache_key(47), true, || panic!("frozen miss allocated"))
            .is_err());
        assert_eq!(cache.entries.len(), 1);
        assert!(cache
            .acquire(&key, true, || panic!("frozen warm hit allocated"))
            .unwrap()
            .is_some());
    }

    #[test]
    fn cache_retains_split_graphs_and_is_bounded_lru() {
        let mut cache = WindowCache::default();
        let a = cache
            .acquire(&cache_key(1), false, cpu_window)
            .unwrap()
            .unwrap();
        let a_id = Arc::as_ptr(&a);
        let a_weak = Arc::downgrade(&a);
        drop(a);
        let b = cache
            .acquire(&cache_key(47), false, cpu_window)
            .unwrap()
            .unwrap();
        let b_id = Arc::as_ptr(&b);
        let b_weak = Arc::downgrade(&b);
        drop(b);
        for _ in 0..16 {
            for (key, identity) in [(cache_key(1), a_id), (cache_key(47), b_id)] {
                let lease = cache
                    .acquire(&key, false, || panic!("split graph thrashed"))
                    .unwrap()
                    .unwrap();
                assert_eq!(Arc::as_ptr(&lease), identity);
            }
        }
        for count in [2, 3] {
            drop(cache.acquire(&cache_key(count), false, cpu_window).unwrap());
        }
        drop(
            cache
                .acquire(&cache_key(1), false, || panic!("LRU hit missed"))
                .unwrap(),
        );
        drop(cache.acquire(&cache_key(4), false, cpu_window).unwrap());
        assert_eq!(cache.entries.len(), WindowCache::CAPACITY);
        assert!(a_weak.upgrade().is_some());
        assert!(
            b_weak.upgrade().is_none(),
            "least-recently-used key should be evicted"
        );
    }

    #[test]
    fn cache_failed_or_panicking_creation_keeps_warm_entries() {
        let mut cache = WindowCache::default();
        let lease = cache
            .acquire(&cache_key(1), false, cpu_window)
            .unwrap()
            .unwrap();
        let weak = Arc::downgrade(&lease);
        drop(lease);
        assert!(cache
            .acquire(&cache_key(2), false, || Err(be(
                "injected allocation failure"
            )))
            .is_err());
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = cache.acquire(&cache_key(2), false, || panic!("injected allocation panic"));
        }));
        assert!(panic.is_err());
        assert_eq!(cache.entries.len(), 1);
        assert!(weak.upgrade().is_some());
    }

    #[test]
    fn phase_transition_and_explicit_release_clear_cache_but_not_live_lease() {
        use crate::adapter::{RuntimePhase, RuntimePhaseArena};
        let mut arena = RuntimePhaseArena::default();
        let layout = Vec::new();
        arena.begin_execute(RuntimePhase::Decode, &layout);
        let lease = arena
            .verify_pairs
            .acquire(&cache_key(1), false, cpu_window)
            .unwrap()
            .unwrap();
        let weak = Arc::downgrade(&lease);
        arena.begin_execute(RuntimePhase::Decode, &layout);
        assert_eq!(arena.verify_pairs.entries.len(), 1);
        arena.begin_execute(RuntimePhase::Prefill, &layout);
        assert!(arena.verify_pairs.entries.is_empty());
        assert!(
            weak.upgrade().is_some(),
            "execute lease must survive phase-cache release"
        );
        drop(lease);
        assert!(weak.upgrade().is_none());
        arena.begin_execute(RuntimePhase::Decode, &layout);
        drop(
            arena
                .verify_pairs
                .acquire(&cache_key(47), false, cpu_window)
                .unwrap(),
        );
        arena.release_phase();
        assert!(arena.verify_pairs.entries.is_empty());
    }

    #[test]
    fn unwinding_phase_owner_releases_cached_windows() {
        let mut weak = std::sync::Weak::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut arena = crate::adapter::RuntimePhaseArena::default();
            let lease = arena
                .verify_pairs
                .acquire(&cache_key(1), false, cpu_window)
                .unwrap()
                .unwrap();
            weak = Arc::downgrade(&lease);
            drop(lease);
            panic!("injected execute unwind");
        }));
        assert!(result.is_err());
        assert!(weak.upgrade().is_none());
    }
}
