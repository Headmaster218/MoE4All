//! Bounded GPU-selected placements; the ordinary Pager remains the host replacement owner.
use super::*;

const CANDIDATES_PER_POOL: usize = 128;
pub(crate) const JOBS_PER_LAYER: usize = infr_core::backend::MAX_DECODE_ROWS * 16 * 3;
const JOB_WORDS: usize = 8;
static COPIED_LAYERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COPIED_BLOCKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn completed_copies() -> (u64, u64) {
    use std::sync::atomic::Ordering;
    (
        COPIED_LAYERS.load(Ordering::Relaxed),
        COPIED_BLOCKS.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
#[path = "pager_gpu_copy_overlap_tests.rs"]
mod overlap_tests;
#[cfg(test)]
#[path = "pager_gpu_copy_tests.rs"]
mod tests;
const JOB_COUNT_WORD: usize = crate::recorder::RESIDENT_JOURNAL_JOB_COUNT_WORD;

pub(crate) struct GpuCopySources {
    words: Vec<u32>,
    keepalive: Vec<Arc<dyn Buffer>>,
    pools: usize,
}

#[derive(Clone)]
struct Candidate {
    pool: usize,
    slot: u32,
    owner: u32,
    address: u64,
}

#[derive(Default)]
struct Epoch {
    candidates: Vec<Option<Candidate>>,
    generation: u64,
    armed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CopyEpochToken(u64);

impl Epoch {
    fn next_token(&self) -> Result<CopyEpochToken> {
        if self.armed {
            return Err(be("GPU copy epoch reused before retirement"));
        }
        self.generation
            .checked_add(1)
            .map(CopyEpochToken)
            .ok_or_else(|| be("GPU copy epoch generation overflow"))
    }

    fn retire(&mut self, token: CopyEpochToken) -> bool {
        if !self.armed || self.generation != token.0 {
            return false;
        }
        self.armed = false;
        true
    }

    fn candidate(&self, index: usize) -> Result<&Candidate> {
        self.candidates
            .get(index)
            .and_then(Option::as_ref)
            .ok_or_else(|| be("GPU copy candidate is padding or out of range"))
    }
}

fn candidate_range(pool: usize, pools: usize) -> Result<(u32, u32)> {
    if pool >= pools {
        return Err(be("GPU copy pool layout changed"));
    }
    let end = pool
        .checked_add(1)
        .and_then(|n| n.checked_mul(CANDIDATES_PER_POOL))
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| be("GPU copy candidate range overflow"))?;
    Ok((end - CANDIDATES_PER_POOL as u32, CANDIDATES_PER_POOL as u32))
}

fn candidate_words(candidates: &[Option<Candidate>]) -> Vec<u32> {
    candidates
        .iter()
        .flat_map(|candidate| match candidate {
            Some(c) => [c.address as u32, (c.address >> 32) as u32, c.owner, 0],
            None => [0, 0, NOT_RESIDENT, 1],
        })
        .collect()
}

fn effective_copy_groups(requested: usize, _wide_copy: bool) -> u32 {
    if requested == 0 {
        8
    } else {
        requested.clamp(1, 8) as u32
    }
}

pub(crate) struct GpuCopyBuffers {
    pub(crate) sources: Arc<dyn Buffer>,
    pub(crate) candidates: Arc<dyn Buffer>,
    pub(crate) overlay: Arc<dyn Buffer>,
    pub(crate) jobs: Arc<dyn Buffer>,
    pub(crate) miss_predicates: Option<Arc<dyn Buffer>>,
    pub(crate) wide_copy: bool,
    pub(crate) copy_groups: u32,
    _sources: Vec<Arc<dyn Buffer>>,
    source_words: Vec<u32>,
    pools: usize,
    epoch: std::sync::Mutex<Epoch>,
}

/// This guard covers only host publication, never submission. A stale token cannot retire
/// another window; a discarded, never-activated command packet never owns a token at all.
struct PreparedCopyActivation<'a> {
    copy: &'a GpuCopyBuffers,
    session: &'a mut MoePagerSession,
    token: CopyEpochToken,
    committed: bool,
}

impl Drop for PreparedCopyActivation<'_> {
    fn drop(&mut self) {
        if !self.committed && self.copy.epoch.lock().unwrap().retire(self.token) {
            self.session.clear_frozen_lut_slots();
        }
    }
}

fn mapped_write(buffer: &dyn Buffer, bytes: &[u8]) -> Result<()> {
    if bytes.len() > buffer.len_bytes() {
        return Err(be("GPU copy metadata overflow"));
    }
    let ptr = as_vk_buf(buffer)?
        .mapped_ptr()
        .ok_or_else(|| be("GPU copy metadata must be mapped"))?;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    Ok(())
}

impl MoePagerSession {
    pub(crate) fn gpu_copy_sources(&self) -> Result<Option<GpuCopySources>> {
        if self._host_tier.is_some() {
            return Ok(None);
        }
        let mut words = vec![0u32; self.role_stride * 3 * 4];
        let mut owners = HashMap::new();
        let mut mapped = 0;
        for (_, pool, source) in self.sources.values() {
            let Some(chunk) = source.host_chunk else {
                return Ok(None);
            };
            if source.stride_bytes == 0
                || !source.stride_bytes.is_multiple_of(16)
                || source.stride_bytes > u32::MAX as usize
            {
                continue;
            }
            for expert in 0..source.bank_bytes / source.stride_bytes {
                let id = source.block_base as usize + expert;
                let offset = source.host_offset + expert * source.stride_bytes;
                let bytes = self.host_store[chunk]
                    .range(offset, source.stride_bytes)
                    .ok_or_else(|| be("GPU copy source exceeds host store"))?;
                words[id * 4 + 2] = source.stride_bytes as u32;
                words[id * 4 + 3] = *pool as u32;
                if let Some((owner, address)) = self.transfer_plan.imported_device_source(bytes) {
                    words[id * 4] = address as u32;
                    words[id * 4 + 1] = (address >> 32) as u32;
                    owners
                        .entry(Arc::as_ptr(&owner) as *const () as usize)
                        .or_insert(owner);
                    mapped += 1;
                }
            }
        }
        if mapped == 0 {
            return Ok(None);
        }
        tracing::info!(
            mapped_role_blocks = mapped,
            "GPU serial-copy source directory ready"
        );
        Ok(Some(GpuCopySources {
            words,
            keepalive: owners.into_values().collect(),
            pools: self.pools.len(),
        }))
    }
}

impl GpuCopyBuffers {
    #[cfg(test)]
    pub(crate) fn test_stub(buffer: Arc<dyn Buffer>) -> Arc<Self> {
        Arc::new(Self {
            sources: buffer.clone(),
            candidates: buffer.clone(),
            overlay: buffer.clone(),
            jobs: buffer,
            miss_predicates: None,
            wide_copy: false,
            copy_groups: effective_copy_groups(0, false),
            _sources: Vec::new(),
            source_words: Vec::new(),
            pools: 0,
            epoch: std::sync::Mutex::new(Epoch::default()),
        })
    }

    pub(crate) fn new(
        be_: &VulkanBackend,
        inputs: GpuCopySources,
        layers: usize,
    ) -> Result<Option<Arc<Self>>> {
        let alloc = |bytes| -> Result<Arc<dyn Buffer>> {
            let buffer: Arc<dyn Buffer> =
                Arc::from(be_.alloc_unified(bytes, UnifiedVramClass::LlmRuntime)?);
            if as_vk_buf(buffer.as_ref())?.mapped_ptr().is_none() {
                return Err(be("GPU serial-copy metadata has no coherent host mapping"));
            }
            Ok(buffer)
        };
        let sources = alloc(inputs.words.len() * 4)?;
        let (last_start, count) = candidate_range(inputs.pools.saturating_sub(1), inputs.pools)?;
        let candidates = alloc((last_start as usize + count as usize) * 16)?;
        let overlay = alloc(inputs.words.len() * 4)?;
        let jobs = alloc(layers * JOBS_PER_LAYER * JOB_WORDS * 4)?;
        let miss_predicates = if be_.cfg().kernels.vulkan.decode_gpu_resident_copy_overlap {
            match be_.alloc_mapped_predicate(layers * crate::recorder::GPU_COPY_PREDICATE_WORDS * 4)
            {
                Ok(buffer) => Some(Arc::from(buffer)),
                Err(error) => {
                    tracing::warn!(%error, "GPU copy overlap predicates unavailable; retaining serial copy");
                    None
                }
            }
        } else {
            None
        };
        mapped_write(sources.as_ref(), bytemuck::cast_slice(&inputs.words))?;
        let properties = unsafe {
            be_.shared
                .instance
                .get_physical_device_properties(be_.shared.physical_device)
        };
        let wide_copy = properties.limits.max_compute_work_group_invocations >= 1024
            && properties.limits.max_compute_work_group_size[0] >= 1024;
        let copy_groups = effective_copy_groups(
            be_.cfg().kernels.vulkan.decode_gpu_resident_copy_groups,
            wide_copy,
        );
        Ok(Some(Arc::new(Self {
            sources,
            candidates,
            overlay,
            jobs,
            miss_predicates,
            wide_copy,
            copy_groups,
            _sources: inputs.keepalive,
            source_words: inputs.words,
            pools: inputs.pools,
            epoch: std::sync::Mutex::new(Epoch::default()),
        })))
    }

    /// Called only after the preceding resident window's commands and notices have drained.
    pub(crate) fn begin(&self, session: &mut MoePagerSession) -> Result<()> {
        self.begin_epoch(session).map(|_| ())
    }

    fn begin_epoch(&self, session: &mut MoePagerSession) -> Result<CopyEpochToken> {
        let mut epoch = self.epoch.lock().unwrap();
        let token = epoch.next_token()?;
        if session.pools.len() != self.pools {
            return Err(be("GPU copy pool layout changed"));
        }
        let mut candidates = vec![None; self.pools * CANDIDATES_PER_POOL];
        for (pool_index, pool) in session.pools.iter().enumerate() {
            let start = pool_index * CANDIDATES_PER_POOL;
            let mut count = 0;
            let budget = CANDIDATES_PER_POOL.min(
                pool.pager
                    .enabled_slots()
                    .saturating_sub(pool.min_enabled_slots),
            );
            for (owner, slot) in pool.pager.pager.replacement_slots() {
                if count >= budget {
                    break;
                }
                let frozen = session.frozen_lut_slots.contains(&ExpertSlotId {
                    pool: pool_index,
                    slot: slot as usize,
                }) || session
                    .resident_lut_cache
                    .as_ref()
                    .is_some_and(|cache| cache.frozen[pool_index][slot as usize] != 0);
                if !pool.pager.pager.slot_enabled(slot) || frozen {
                    continue;
                }
                candidates[start + count] = Some(Candidate {
                    pool: pool_index,
                    slot,
                    owner: owner.unwrap_or(NOT_RESIDENT),
                    address: pool.pager.slot_addr(slot)?,
                });
                count += 1;
            }
        }
        let words = candidate_words(&candidates);
        mapped_write(self.candidates.as_ref(), bytemuck::cast_slice(&words))?;
        for candidate in candidates.iter().flatten() {
            session.frozen_lut_slots.insert(ExpertSlotId {
                pool: candidate.pool,
                slot: candidate.slot as usize,
            });
        }
        *epoch = Epoch {
            candidates,
            generation: token.0,
            armed: true,
        };
        Ok(token)
    }

    /// ACK-time activation at execute entry, after old readers and their pins have drained.
    /// `publish` may update LUT windows/events, but must not submit any command.
    pub(crate) fn activate_prepared<T>(
        &self,
        session: &mut MoePagerSession,
        publish: impl FnOnce(&mut MoePagerSession) -> Result<T>,
    ) -> Result<T> {
        if !session.frozen_lut_slots.is_empty()
            || session
                .resident_lut_cache
                .as_ref()
                .is_some_and(|cache| cache.frozen.iter().flatten().any(|&frozen| frozen != 0))
        {
            return Err(be(
                "prepared GPU copy activation requires drained LUT windows",
            ));
        }
        let token = self.begin_epoch(session)?;
        let mut activation = PreparedCopyActivation {
            copy: self,
            session,
            token,
            committed: false,
        };
        let result = publish(&mut *activation.session)?;
        activation.committed = true;
        Ok(result)
    }

    pub(crate) fn role_params(
        &self,
        session: &MoePagerSession,
        roles: &[usize; 3],
    ) -> Result<[u32; 9]> {
        if session.pools.len() != self.pools {
            return Err(be("GPU copy pool layout changed"));
        }
        let mut params = [0u32; 9];
        for (index, role) in roles.iter().enumerate() {
            let (_, pool, source) = session
                .sources
                .get(role)
                .ok_or_else(|| be("GPU copy role missing"))?;
            params[index] = source.block_base;
            (params[index + 3], params[index + 6]) = candidate_range(*pool, self.pools)?;
        }
        Ok(params)
    }

    /// Serial notices follow payload completion. Overlap notices admit ordered ownership early;
    /// candidates stay frozen until the payload/readers drain, or abort invalidates their owners.
    pub(crate) fn replay(
        &self,
        session: &mut MoePagerSession,
        ordinal: usize,
        roles: &[usize; 3],
        ids: &[u32],
        hit_masks: &[u32],
        used: usize,
        words: &[u32; crate::recorder::RESIDENT_JOURNAL_WORDS],
    ) -> Result<()> {
        let count = words[JOB_COUNT_WORD] as usize;
        if count == 0 || count > JOBS_PER_LAYER {
            return Err(be("GPU copy journal has invalid job count"));
        }
        let offset = ordinal * JOBS_PER_LAYER * JOB_WORDS * 4;
        if offset + count * JOB_WORDS * 4 > self.jobs.len_bytes() {
            return Err(be("GPU copy jobs exceed epoch"));
        }
        let ptr = as_vk_buf(self.jobs.as_ref())?
            .mapped_ptr()
            .ok_or_else(|| be("GPU copy job journal unmapped"))?;
        let mut job_words = vec![0u32; count * JOB_WORDS];
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.add(offset),
                job_words.as_mut_ptr().cast(),
                job_words.len() * 4,
            )
        };
        let epoch = self.epoch.lock().unwrap();
        if !epoch.armed {
            return Err(be("GPU copy notice outlived its epoch"));
        }
        let mut placements = HashMap::new();
        let mut candidate_ids = std::collections::HashSet::new();
        for job in job_words.chunks_exact(JOB_WORDS) {
            let candidate = epoch.candidate(job[6] as usize)?;
            let address = job[2] as u64 | ((job[3] as u64) << 32);
            let matching = roles.iter().any(|role| {
                let (_, pool, source) = &session.sources[role];
                *pool == candidate.pool
                    && source.stride_bytes == job[4] as usize
                    && ids
                        .iter()
                        .any(|&id| source.block_base.checked_add(id) == Some(job[5]))
            });
            let source = self
                .source_words
                .get(job[5] as usize * 4..job[5] as usize * 4 + 4);
            if address != candidate.address
                || !matching
                || source.is_none_or(|source| source[..3] != [job[0], job[1], job[4]])
                || !candidate_ids.insert(job[6])
                || placements.insert(job[5], candidate).is_some()
            {
                return Err(be(
                    "GPU copy placement journal disagrees with admission snapshot",
                ));
            }
        }
        session.begin_role_batches(roles)?;
        let hits: Vec<u32> = ids
            .iter()
            .enumerate()
            .filter_map(|(slot, &id)| {
                (hit_masks[slot / used] & (1 << (slot % used)) != 0).then_some(id)
            })
            .collect();
        session.touch_routed_hits(roles, &hits)?;
        for &role in roles {
            let (_, pool_index, source) = session.sources[&role].clone();
            for (index, &expert) in ids.iter().enumerate() {
                if hit_masks[index / used] & (1 << (index % used)) != 0 {
                    continue;
                }
                let block = source.block_base + expert;
                let pool = &mut session.pools[pool_index].pager;
                if let Some(&candidate) = placements.get(&block) {
                    let resolution =
                        pool.pager
                            .admit_at(block, candidate.slot)
                            .map_err(|error| {
                                be(format!("GPU copy host admission disagrees: {error:?}"))
                            })?;
                    if let Resolution::Miss { slot, evicted } = resolution {
                        pool.record_placement(block, slot, evicted);
                    }
                } else if pool.pager.slot_of(block).is_some() {
                    let _ = pool.pager.touch(block);
                } else {
                    return Err(be("GPU copy completed without a required role placement"));
                }
            }
        }
        let layers = COPIED_LAYERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        COPIED_BLOCKS.fetch_add(count as u64, std::sync::atomic::Ordering::Relaxed);
        if layers == 1 || layers.is_multiple_of(512) {
            let (_, blocks) = completed_copies();
            tracing::info!(layers, blocks, "GPU copy cache placements admitted");
        }
        Ok(())
    }

    pub(crate) fn complete(&self) {
        self.epoch.lock().unwrap().armed = false;
    }

    /// An errored GPU window may have overwritten a victim without publishing its final notice.
    /// Invalidate host ownership only; never write into still-running GPU metadata here.
    pub(crate) fn abort(&self, session: &mut MoePagerSession) {
        let mut epoch = self.epoch.lock().unwrap();
        if !epoch.armed {
            return;
        }
        for candidate in epoch.candidates.iter().flatten() {
            let pager = &mut session.pools[candidate.pool].pager;
            if let Some(id) = pager
                .lut_host
                .iter()
                .position(|&slot| slot == candidate.slot)
            {
                pager.pager.evict(id as u32);
                pager.lut_host[id] = NOT_RESIDENT;
                pager.lut_dirty = true;
            }
        }
        epoch.armed = false;
    }
}

#[cfg(test)]
mod prepared_tests {
    use super::*;

    #[test]
    fn gpu_copy_groups_auto_and_clamp_preserve_both_payload_variants() {
        assert_eq!(effective_copy_groups(0, true), 8);
        assert_eq!(effective_copy_groups(0, false), 8);
        for wide in [false, true] {
            for (requested, expected) in [
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (8, 8),
                (9, 8),
                (usize::MAX, 8),
            ] {
                assert_eq!(effective_copy_groups(requested, wide), expected);
            }
        }
    }

    fn candidate(pool: usize, slot: u32) -> Candidate {
        Candidate {
            pool,
            slot,
            owner: NOT_RESIDENT,
            address: 0x1000 + u64::from(slot) * 64,
        }
    }

    #[test]
    fn fixed_ranges_are_disjoint_or_exact_pool_aliases() {
        let ranges = [0, 1, 0].map(|pool| candidate_range(pool, 2).unwrap());
        assert_eq!(ranges, [(0, 128), (128, 128), (0, 128)]);
        assert!(candidate_range(2, 2).is_err());
        assert!(candidate_range(0, 0).is_err());
        assert!(candidate_range(usize::MAX - 1, usize::MAX).is_err());
        assert!(candidate_range(u32::MAX as usize / 128, usize::MAX).is_err());
    }

    #[test]
    fn padded_indices_survive_unequal_counts_and_reject_holes() {
        let mut epoch = Epoch {
            candidates: vec![None; 2 * CANDIDATES_PER_POOL],
            ..Epoch::default()
        };
        epoch.candidates[0] = Some(candidate(0, 3));
        epoch.candidates[128] = Some(candidate(1, 7));
        epoch.candidates[129] = Some(candidate(1, 9));
        let words = candidate_words(&epoch.candidates);
        assert_eq!(words.len(), 256 * 4);
        assert_eq!(words[128 * 4], epoch.candidate(128).unwrap().address as u32);
        assert_eq!(epoch.candidate(129).unwrap().slot, 9);
        for index in [1, 127, 130, 255, 256, usize::MAX] {
            assert!(epoch.candidate(index).is_err(), "index={index}");
        }
        // Abort walks the same flattened snapshot, never a NOT_RESIDENT padding slot.
        let slots: Vec<_> = epoch.candidates.iter().flatten().map(|c| c.slot).collect();
        assert_eq!(slots, [3, 7, 9]);
    }

    #[test]
    fn shrinking_epoch_rewrites_the_entire_padded_table() {
        let mut candidates = vec![None; 2 * CANDIDATES_PER_POOL];
        candidates[127] = Some(candidate(0, 12));
        candidates[128] = Some(candidate(1, 13));
        let old = candidate_words(&candidates);
        candidates.fill(None);
        candidates[0] = Some(candidate(0, 1));
        let new = candidate_words(&candidates);
        assert_eq!(new.len(), old.len());
        for index in 1..256 {
            assert_eq!(&new[index * 4..index * 4 + 4], &[0, 0, NOT_RESIDENT, 1]);
        }
        assert_ne!(&old[128 * 4..128 * 4 + 4], &new[128 * 4..128 * 4 + 4]);
    }

    #[test]
    fn only_the_armed_generation_can_retire_an_epoch() {
        let mut epoch = Epoch::default();
        let first = epoch.next_token().unwrap();
        epoch.generation = first.0;
        epoch.armed = true;
        assert!(epoch.next_token().is_err());
        assert!(!epoch.retire(CopyEpochToken(0)));
        assert!(epoch.armed);
        assert!(epoch.retire(first));
        assert!(!epoch.retire(first));
        let second = epoch.next_token().unwrap();
        assert_ne!(first, second);
        epoch.generation = second.0;
        epoch.armed = true;
        assert!(!epoch.retire(first));
        assert!(epoch.armed);
        assert!(epoch.retire(second));
    }

    #[test]
    fn epoch_generation_never_wraps_into_a_stale_owner() {
        let epoch = Epoch {
            generation: u64::MAX,
            ..Epoch::default()
        };
        assert!(epoch.next_token().is_err());
    }

    struct CpuBuffer;

    impl Buffer for CpuBuffer {
        fn len_bytes(&self) -> usize {
            0
        }
        fn device_addr(&self) -> Option<u64> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn discarding_retained_copy_handles_does_not_retire_live_epoch() {
        let source: Arc<dyn Buffer> = Arc::new(CpuBuffer);
        let copy = Arc::new(GpuCopyBuffers {
            sources: Arc::new(CpuBuffer),
            candidates: Arc::new(CpuBuffer),
            overlay: Arc::new(CpuBuffer),
            jobs: Arc::new(CpuBuffer),
            miss_predicates: None,
            wide_copy: false,
            copy_groups: 8,
            _sources: vec![Arc::clone(&source)],
            source_words: Vec::new(),
            pools: 1,
            epoch: std::sync::Mutex::new(Epoch {
                generation: 7,
                armed: true,
                ..Epoch::default()
            }),
        });
        let packet = Arc::clone(&copy);
        assert_eq!(Arc::strong_count(&source), 2);
        drop(packet);
        assert!(copy.epoch.lock().unwrap().armed);
        assert_eq!(copy.epoch.lock().unwrap().generation, 7);
        drop(copy);
        assert_eq!(Arc::strong_count(&source), 1);
    }

    #[test]
    #[ignore = "requires an explicitly selected idle Vulkan GPU; activation metadata only"]
    fn prepared_copy_activation_success_discard_and_failure() {
        let mut cfg = infr_core::config::Config::default();
        cfg.device.dev =
            Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select the test GPU explicitly"));
        cfg.kernels.vulkan.decode_gpu_resident_fast_lut = true;
        let be_ = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
        const POOL_SLOTS: usize = 64; // 4 MiB physical arena, including the runtime reserve.
        let session = MoePagerSession::new(
            &be_,
            MoePagerLayout {
                load_reserve_bytes: 0,
                n_blocks: 4,
                pools: vec![MoePoolSpec {
                    slot_bytes: 64 * 1024,
                    n_slots: POOL_SLOTS,
                    min_enabled_slots: 1,
                }],
                host_tier: None,
                host_store_io: None,
                dynamic_state_reserve_bytes: 0,
                dynamic_state_max_allocation_bytes: 0,
                prefill_min_lane_bytes: 0,
                runtime_reserve_bytes: 1024 * 1024,
                host_chunks: vec![MoeHostChunkSpec {
                    base_offset: 0,
                    bytes: 64 * 1024,
                }],
                prefill_target_lanes: 1,
                prefill_cache_bytes: 2 * 1024 * 1024,
            },
        )
        .unwrap();
        assert!(session.resident_lut_cache.is_some());
        // Metadata claims retire Expert filler through the backend's installed pager.
        *be_.moe_pager().lock().unwrap() = Some(session);
        let copy = GpuCopyBuffers::new(
            &be_,
            GpuCopySources {
                words: vec![0; 4 * 3 * 4],
                keepalive: Vec::new(),
                pools: 1,
            },
            2,
        )
        .unwrap()
        .unwrap();
        let mut session = be_.moe_pager().lock().unwrap().take().unwrap();

        drop(Arc::clone(&copy));
        assert!(!copy.epoch.lock().unwrap().armed);
        let owners = session.pools[0].pager.lut_host.clone();
        let failed: Result<()> = copy.activate_prepared(&mut session, |session| {
            session.resident_lut_cache.as_mut().unwrap().frozen[0][0] = 1;
            Err(be("injected publication failure"))
        });
        assert!(failed.is_err());
        assert!(!copy.epoch.lock().unwrap().armed);
        assert!(session.frozen_lut_slots.is_empty());
        assert!(session.resident_lut_cache.as_ref().unwrap().frozen[0]
            .iter()
            .all(|&v| v == 0));
        assert_eq!(
            session.pools[0].pager.lut_host, owners,
            "unsubmitted rollback must not evict"
        );

        copy.activate_prepared(&mut session, |_| Ok(())).unwrap();
        {
            let epoch = copy.epoch.lock().unwrap();
            let valid = epoch.candidates.iter().flatten().count();
            assert_eq!(epoch.candidates.len(), CANDIDATES_PER_POOL);
            assert!(valid > 0 && valid <= POOL_SLOTS);
            assert!(epoch.candidates[..valid].iter().all(Option::is_some));
            assert!(epoch.candidates[valid..].iter().all(Option::is_none));
            for candidate in epoch.candidates.iter().flatten() {
                assert_eq!(candidate.pool, 0);
                assert!((candidate.slot as usize) < POOL_SLOTS);
                assert_ne!(candidate.address, 0);
            }
            let words = candidate_words(&epoch.candidates);
            assert!(words[valid * 4..]
                .chunks_exact(4)
                .all(|word| word == [0, 0, NOT_RESIDENT, 1]));
            assert_eq!(session.frozen_lut_slots.len(), valid);
        }
        let token = CopyEpochToken(copy.epoch.lock().unwrap().generation);
        assert!(copy.epoch.lock().unwrap().armed);
        let frozen = session.frozen_lut_slots.clone();
        let blocked: Result<()> = copy.activate_prepared(&mut session, |_| {
            panic!("a second activation must not publish")
        });
        assert!(blocked.is_err());
        assert_eq!(session.frozen_lut_slots, frozen);
        assert_eq!(copy.epoch.lock().unwrap().generation, token.0);
        copy.complete();
        session.clear_frozen_lut_slots();

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<()> =
                copy.activate_prepared(&mut session, |_| panic!("injected activation unwind"));
        }));
        assert!(panicked.is_err());
        assert!(!copy.epoch.lock().unwrap().armed);
        assert!(session.frozen_lut_slots.is_empty());
        copy.activate_prepared(&mut session, |_| Ok(())).unwrap();
        copy.complete();
        session.clear_frozen_lut_slots();
    }
}
