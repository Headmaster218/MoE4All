//! GPU copy metadata for host-managed windows and persistent GPU-owned Decode.
use super::*;

#[cfg(test)]
impl GpuCopySources {
    pub(crate) fn test_cpu_only_source(&mut self, block: usize) {
        assert!(self.words[block * 4 + 2] != 0);
        self.words[block * 4..block * 4 + 2].fill(0);
    }
}

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

fn gpu_copy_workgroup_supported(max_invocations: u32, max_size_x: u32) -> bool {
    // The small payload kernel is WG256; wide-copy selection is not a smaller-device fallback.
    max_invocations >= 256 && max_size_x >= 256
}

fn copy_overlap_predicates_requested(requested: bool, mtp_enabled: bool) -> bool {
    requested && !mtp_enabled
}

fn effective_copy_groups(requested: usize, _wide_copy: bool) -> u32 {
    if requested == 0 {
        8
    } else {
        requested.clamp(1, 8) as u32
    }
}

fn all_registered_sources_mapped(words: &[u32]) -> bool {
    if !words.len().is_multiple_of(4) {
        return false;
    }
    let mut registered = false;
    for source in words.chunks_exact(4).filter(|source| source[2] != 0) {
        registered = true;
        if source[0] == 0 && source[1] == 0 {
            return false;
        }
    }
    registered
}

fn source_unmapped_prefix(words: &[u32], pools: usize) -> Vec<usize> {
    if !words.len().is_multiple_of(4) {
        return Vec::new();
    }
    let mut prefix = Vec::with_capacity(words.len() / 4 + 1);
    prefix.push(0);
    for entry in words.chunks_exact(4) {
        let address = u64::from(entry[0]) | (u64::from(entry[1]) << 32);
        let mapped = address != 0
            && address & 15 == 0
            && entry[2] != 0
            && entry[2].is_multiple_of(16)
            && (entry[3] as usize) < pools
            && address.checked_add(u64::from(entry[2]) - 1).is_some();
        prefix.push(prefix.last().copied().unwrap() + usize::from(!mapped));
    }
    prefix
}

fn source_banks_mapped(prefix: &[usize], bases: [u32; 3], experts: u32) -> bool {
    experts != 0
        && bases.into_iter().all(|base| {
            let start = base as usize;
            start
                .checked_add(experts as usize)
                .and_then(|end| Some((prefix.get(start)?, prefix.get(end)?)))
                .is_some_and(|(start, end)| start == end)
        })
}

pub(crate) struct GpuCopyBuffers {
    pub(super) owned: Option<super::gpu_owner::OwnedState>,
    pub(crate) sources: Arc<dyn Buffer>,
    pub(crate) candidates: Arc<dyn Buffer>,
    pub(crate) overlay: Arc<dyn Buffer>,
    pub(crate) jobs: Arc<dyn Buffer>,
    pub(crate) miss_predicates: Option<Arc<dyn Buffer>>,
    pub(crate) wide_copy: bool,
    pub(crate) copy_groups: u32,
    all_sources_mapped: bool,
    // Immutable owner/source-directory certificate: prefix count of blocks without GPU sources.
    unmapped_source_prefix: Vec<usize>,
    // Allocated before ownership starts; seed refresh never allocates GPU storage or reenters
    // unified admission while the caller holds the pager/execution lease.
    metadata_staging: Option<std::sync::Mutex<crate::VkBuffer>>,
    _sources: Vec<Arc<dyn Buffer>>,
    pub(super) source_words: Vec<u32>,
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

struct PreparedOwnedActivation<'a> {
    in_forward: &'a std::sync::atomic::AtomicBool,
    previous: bool,
    committed: bool,
}

impl<'a> PreparedOwnedActivation<'a> {
    fn new(in_forward: &'a std::sync::atomic::AtomicBool) -> Self {
        Self {
            in_forward,
            previous: in_forward.load(std::sync::atomic::Ordering::Acquire),
            committed: false,
        }
    }
}

impl Drop for PreparedOwnedActivation<'_> {
    fn drop(&mut self) {
        if !self.committed {
            // Publication cannot submit work. Keep the valid active owner, but roll back this
            // attempted forward on both Result errors and unwinding; an outer forward may exist.
            self.in_forward
                .store(self.previous, std::sync::atomic::Ordering::Release);
        }
    }
}

impl Drop for PreparedCopyActivation<'_> {
    fn drop(&mut self) {
        if !self.committed && self.copy.epoch.lock().unwrap().retire(self.token) {
            self.session.clear_frozen_lut_slots();
        }
    }
}

fn direct_metadata_mapping(mapped: bool, properties: vk::MemoryPropertyFlags) -> bool {
    mapped
        && properties.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
}

fn coherent_metadata_mapping(buffer: &dyn Buffer) -> Result<bool> {
    let buffer = as_vk_buf(buffer)?;
    let properties = match &buffer.backing {
        crate::Backing::Pooled(allocation) => allocation.memory_properties(),
        // A suballocation inherits physical memory properties, not merely its parent's pointer.
        crate::Backing::UnifiedSub(handle) => {
            return Ok(buffer.mapped_ptr().is_some() && coherent_metadata_mapping(handle.buffer())?);
        }
        crate::Backing::BdaSub(block) => {
            return Ok(buffer.mapped_ptr().is_some() && coherent_metadata_mapping(&block.buf)?);
        }
        // Dedicated mapped allocations retain no property flags. All current constructors
        // explicitly select HOST_VISIBLE|HOST_COHERENT: ReBAR, UMA/KV spill and host import.
        crate::Backing::Vram { .. } | crate::Backing::ImportedHost { .. } => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        crate::Backing::Device { .. } | crate::Backing::External { .. } => {
            vk::MemoryPropertyFlags::empty()
        }
        crate::Backing::RecordingAlias { .. } => {
            return Err(be("recording-only buffer cannot publish GPU copy metadata"));
        }
    };
    Ok(direct_metadata_mapping(
        buffer.mapped_ptr().is_some(),
        properties,
    ))
}

pub(super) fn mapped_write(buffer: &dyn Buffer, bytes: &[u8]) -> Result<()> {
    if bytes.len() > buffer.len_bytes() {
        return Err(be("GPU copy metadata overflow"));
    }
    if !coherent_metadata_mapping(buffer)? {
        return Err(be(
            "GPU copy metadata direct write requires a coherent host mapping",
        ));
    }
    let ptr = as_vk_buf(buffer)?
        .mapped_ptr()
        .ok_or_else(|| be("GPU copy metadata must be mapped"))?;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    Ok(())
}

fn metadata_upload_layout(entries: &[(usize, bool)]) -> Result<(usize, Vec<Option<usize>>)> {
    let mut bytes = 0usize;
    let mut offsets = Vec::with_capacity(entries.len());
    for &(size, mapped) in entries {
        if size == 0 || size > isize::MAX as usize || !size.is_multiple_of(4) {
            return Err(be(
                "GPU owner metadata has invalid upload alignment or size",
            ));
        }
        offsets.push(if mapped {
            None
        } else {
            let offset = bytes;
            bytes = bytes
                .checked_add(size)
                .filter(|&n| n <= isize::MAX as usize)
                .ok_or_else(|| be("GPU owner metadata staging size overflow"))?;
            Some(offset)
        });
    }
    Ok((bytes, offsets))
}

fn new_metadata_staging(
    backend: &VulkanBackend,
    buffers: &[&dyn Buffer],
) -> Result<Option<std::sync::Mutex<crate::VkBuffer>>> {
    let entries = buffers
        .iter()
        .map(|buffer| Ok((buffer.len_bytes(), coherent_metadata_mapping(*buffer)?)))
        .collect::<Result<Vec<_>>>()?;
    let (bytes, _) = metadata_upload_layout(&entries)?;
    if bytes == 0 {
        return Ok(None);
    }
    let staging = backend.make_buf(
        bytes,
        gpu_allocator::MemoryLocation::CpuToGpu,
        "gpu-owner-seed-upload",
    )?;
    let crate::Backing::Pooled(allocation) = &staging.backing else {
        return Err(be("GPU owner upload requires pooled host staging memory"));
    };
    if staging.mapped_ptr().is_none()
        || !allocation.memory_properties().contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    {
        return Err(be("GPU owner upload requires coherent host staging memory"));
    }
    Ok(Some(std::sync::Mutex::new(staging)))
}

fn upload_metadata_batch(
    backend: &VulkanBackend,
    retained: Option<&std::sync::Mutex<crate::VkBuffer>>,
    uploads: &[(&dyn Buffer, &[u8])],
) -> Result<()> {
    // Validate the whole batch before changing any mapped bytes or queuing a partial seed.
    let mut entries = Vec::with_capacity(uploads.len());
    for &(buffer, data) in uploads {
        let vk_buffer = as_vk_buf(buffer)?;
        if !Arc::ptr_eq(&vk_buffer.shared, &backend.shared) || data.len() > buffer.len_bytes() {
            return Err(be("GPU owner upload device or destination size mismatch"));
        }
        entries.push((data.len(), coherent_metadata_mapping(buffer)?));
    }
    let (bytes, offsets) = metadata_upload_layout(&entries)?;
    if bytes == 0 {
        for &(buffer, data) in uploads {
            mapped_write(buffer, data)?;
        }
        return Ok(());
    }
    let staging = retained
        .ok_or_else(|| be("GPU owner upload has no retained staging allocation"))?
        .lock()
        .unwrap();
    if bytes > staging.size || !Arc::ptr_eq(&staging.shared, &backend.shared) {
        return Err(be("GPU owner upload exceeds retained staging geometry"));
    }
    let ptr = staging
        .mapped_ptr()
        .ok_or_else(|| be("GPU owner retained staging is not mapped"))?;
    for (&(buffer, data), &offset) in uploads.iter().zip(&offsets) {
        if let Some(offset) = offset {
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len()) };
        } else {
            mapped_write(buffer, data)?;
        }
    }
    // Caller owns the forward/admission lease and has drained old directory users. Do not use
    // Backend::upload/one_shot: their phase/admission hooks may reenter the locked pager.
    let recorder = backend.recorder()?;
    recorder.seed_barrier();
    recorder.host_transfer_barrier();
    for (&(buffer, data), offset) in uploads.iter().zip(offsets) {
        if let Some(offset) = offset {
            recorder.copy(&*staging, offset, buffer, 0, data.len());
        }
    }
    recorder.arena_stream_barrier();
    // Retain staging and destination borrows through submission and draining on every path.
    recorder.finish_nowait()?.wait()
}

impl MoePagerSession {
    pub(crate) fn gpu_copy_sources(&self) -> Result<Option<GpuCopySources>> {
        let mut words = vec![0u32; self.role_stride * 3 * 4];
        let mut owners = HashMap::new();
        let mut mapped = 0;
        for (_, pool, source) in self.sources.values() {
            if source.stride_bytes == 0
                || !source.stride_bytes.is_multiple_of(16)
                || source.stride_bytes > u32::MAX as usize
            {
                continue;
            }
            for expert in 0..source.bank_bytes / source.stride_bytes {
                let id = source.block_base as usize + expert;
                let offset = source.host_offset + expert * source.stride_bytes;
                words[id * 4 + 2] = source.stride_bytes as u32;
                words[id * 4 + 3] = *pool as u32;
                let imported = source.host_chunk.and_then(|chunk| {
                    self.host_store[chunk]
                        .range(offset, source.stride_bytes)
                        .and_then(|bytes| self.transfer_plan.imported_device_source(bytes))
                });
                if let Some((owner, address)) = imported {
                    words[id * 4] = address as u32;
                    words[id * 4 + 1] = (address >> 32) as u32;
                    owners
                        .entry(Arc::as_ptr(&owner) as *const () as usize)
                        .or_insert(owner);
                    mapped += 1;
                }
            }
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
    pub(super) fn upload_owned_seed(
        &self,
        backend: &VulkanBackend,
        nodes: &[u32],
        blocks: &[u32],
    ) -> Result<()> {
        upload_metadata_batch(
            backend,
            self.metadata_staging.as_ref(),
            &[
                (self.candidates.as_ref(), bytemuck::cast_slice(nodes)),
                (self.overlay.as_ref(), bytemuck::cast_slice(blocks)),
            ],
        )
    }

    pub(crate) fn owned_self_service(&self) -> bool {
        self.is_owned() && self.all_sources_mapped
    }

    pub(crate) fn owned_roles_self_service(&self, bases: [u32; 3], experts: u32) -> bool {
        // The registered-bank geometry prevents certifying an arbitrary subrange or padding.
        // This certificate lives with this exact owner Arc and its retained imported sources;
        // mutable residency/generation validation remains in begin/activate_prepared and shaders.
        self.owned_role_geometry(bases, experts).is_some()
            && source_banks_mapped(&self.unmapped_source_prefix, bases, experts)
    }

    #[cfg(test)]
    pub(crate) fn test_stub(buffer: Arc<dyn Buffer>) -> Arc<Self> {
        Arc::new(Self {
            owned: None,
            sources: buffer.clone(),
            candidates: buffer.clone(),
            overlay: buffer.clone(),
            jobs: buffer,
            miss_predicates: None,
            wide_copy: false,
            copy_groups: effective_copy_groups(0, false),
            all_sources_mapped: false,
            unmapped_source_prefix: Vec::new(),
            metadata_staging: None,
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
        Self::new_inner(be_, inputs, layers, None)
    }

    pub(crate) fn new_owned(
        be_: &VulkanBackend,
        inputs: GpuCopySources,
        layers: usize,
        seed: super::gpu_owner::OwnedSeed,
    ) -> Result<Option<Arc<Self>>> {
        Self::new_inner(be_, inputs, layers, Some(seed))
    }

    fn new_inner(
        be_: &VulkanBackend,
        inputs: GpuCopySources,
        layers: usize,
        seed: Option<super::gpu_owner::OwnedSeed>,
    ) -> Result<Option<Arc<Self>>> {
        let properties = unsafe {
            be_.shared
                .instance
                .get_physical_device_properties(be_.shared.physical_device)
        };
        if !gpu_copy_workgroup_supported(
            properties.limits.max_compute_work_group_invocations,
            properties.limits.max_compute_work_group_size[0],
        ) {
            // Decline both host-managed and owned copying before metadata allocation/upload.
            return Ok(None);
        }
        if let Some(seed) = &seed {
            seed.state.validate_source_words(&inputs.words)?;
        }
        let alloc = |bytes: usize, host_access_required: bool| -> Result<Arc<dyn Buffer>> {
            let buffer = be_.alloc_unified(bytes, UnifiedVramClass::LlmRuntime)?;
            if !host_access_required || coherent_metadata_mapping(buffer.as_ref())? {
                return Ok(Arc::from(buffer));
            }
            // A GPU-visible host allocation keeps the protocol usable without mapped VRAM.
            drop(buffer);
            let host = be_.alloc_uninit(bytes, BufferUsage::Staging)?;
            if !coherent_metadata_mapping(host.as_ref())? {
                return Err(be("GPU cache metadata has no coherent host mapping"));
            }
            Ok(Arc::from(host))
        };
        let all_sources_mapped = all_registered_sources_mapped(&inputs.words);
        let unmapped_source_prefix = source_unmapped_prefix(&inputs.words, inputs.pools);
        let sources = alloc(inputs.words.len() * 4, false)?;
        let (last_start, count) = candidate_range(inputs.pools.saturating_sub(1), inputs.pools)?;
        let candidates = alloc(
            seed.as_ref()
                .map_or((last_start as usize + count as usize) * 16, |s| {
                    s.nodes.len() * 4
                }),
            seed.is_none(),
        )?;
        let overlay = alloc(inputs.words.len() * 4, seed.is_none())?;
        let jobs = alloc(layers * JOBS_PER_LAYER * JOB_WORDS * 4, true)?;
        let miss_predicates = if copy_overlap_predicates_requested(
            be_.cfg().kernels.vulkan.decode_gpu_resident_copy_overlap,
            be_.cfg().spec.mtp,
        ) {
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
        let metadata_staging = new_metadata_staging(
            be_,
            &[sources.as_ref(), candidates.as_ref(), overlay.as_ref()],
        )?;
        let mut uploads = vec![(sources.as_ref(), bytemuck::cast_slice(&inputs.words))];
        if let Some(seed) = &seed {
            uploads.push((candidates.as_ref(), bytemuck::cast_slice(&seed.nodes)));
            uploads.push((overlay.as_ref(), bytemuck::cast_slice(&seed.blocks)));
        }
        upload_metadata_batch(be_, metadata_staging.as_ref(), &uploads)?;
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
            all_sources_mapped,
            unmapped_source_prefix,
            metadata_staging,
            _sources: inputs.keepalive,
            source_words: inputs.words,
            pools: inputs.pools,
            epoch: std::sync::Mutex::new(Epoch::default()),
            owned: seed.map(|s| s.state),
        })))
    }

    /// Called only after the preceding resident window's commands and notices have drained.
    pub(crate) fn begin(
        &self,
        backend: &VulkanBackend,
        session: &mut MoePagerSession,
    ) -> Result<()> {
        if self.is_owned() {
            return self.begin_owned(backend, session);
        }
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
        backend: &VulkanBackend,
        session: &mut MoePagerSession,
        publish: impl FnOnce(&mut MoePagerSession) -> Result<T>,
    ) -> Result<T> {
        if let Some(state) = &self.owned {
            let mut activation = PreparedOwnedActivation::new(&state.in_forward);
            self.begin_owned(backend, session)?;
            let result = publish(session)?;
            activation.committed = true;
            return Ok(result);
        }
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
            if !self.is_owned() {
                (params[index + 3], params[index + 6]) = candidate_range(*pool, self.pools)?;
            }
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
        if self.is_owned() {
            return;
        }
        self.epoch.lock().unwrap().armed = false;
    }

    /// An errored GPU window may have overwritten a victim without publishing its final notice.
    /// Invalidate host ownership only; never write into still-running GPU metadata here.
    pub(crate) fn abort(&self, session: &mut MoePagerSession) {
        // GPU ownership is session-wide; a chunk drop must never replay or resurrect the stale
        // CPU snapshot. Pending placements are discarded at a drained ownership handoff.
        if self.is_owned() {
            return;
        }
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
    fn per_bank_mapping_certificate_covers_aliases_holes_and_invalid_ranges() {
        let mut words = Vec::new();
        // Three registered banks: the middle bank has one CPU-only expert.
        for block in 0..12 {
            let address = if block == 5 { 0 } else { 0x1000 + block * 16 };
            words.extend([address, 0, 16, 0]);
        }
        let prefix = source_unmapped_prefix(&words, 1);
        assert!(source_banks_mapped(&prefix, [0, 8, 0], 4));
        for bases in [[4, 0, 8], [0, 4, 8], [0, 8, 4], [3, 8, 0]] {
            assert!(!source_banks_mapped(&prefix, bases, 4));
        }
        assert!(!source_banks_mapped(&prefix, [0, 0, 0], 0));
        assert!(!source_banks_mapped(&prefix, [0, 8, 9], 4));
        assert!(!source_banks_mapped(&prefix, [u32::MAX; 3], 4));
        assert!(!source_banks_mapped(
            &source_unmapped_prefix(&words[..47], 1),
            [0; 3],
            4
        ));
        // Host source mutation changes a NEW owner's certificate, never the old retained one.
        words[5 * 4] = 0x2000;
        assert!(!source_banks_mapped(&prefix, [4; 3], 4));
        let replacement = source_unmapped_prefix(&words, 1);
        assert!(source_banks_mapped(&replacement, [0, 4, 8], 4));
        for bad in [
            [0, 0, 0, 0],
            [1, 0, 16, 0],
            [16, 0, 17, 0],
            [16, 0, 16, 1],
            [0xfffffff0, u32::MAX, 32, 0],
        ] {
            assert!(!source_banks_mapped(
                &source_unmapped_prefix(&bad, 1),
                [0; 3],
                1
            ));
        }
    }

    #[test]
    fn self_service_requires_a_nonempty_fully_mapped_registered_directory() {
        for words in [
            vec![],
            vec![0; 4],
            vec![1, 0, 0, 0],
            vec![0, 0, 16, 0],
            vec![1, 0, 16],
        ] {
            assert!(!all_registered_sources_mapped(&words));
        }
        assert!(all_registered_sources_mapped(&[0, 0, 0, 0, 1, 0, 16, 0]));
        assert!(all_registered_sources_mapped(&[0, 1, 16, 0, 2, 0, 32, 1]));
        assert!(!all_registered_sources_mapped(&[1, 0, 16, 0, 0, 0, 32, 1]));
    }

    #[test]
    fn metadata_upload_layout_preserves_mapped_and_disjoint_unmapped_ranges() {
        assert_eq!(metadata_upload_layout(&[]).unwrap(), (0, vec![]));
        assert_eq!(
            metadata_upload_layout(&[(16, true), (160, false), (64, false), (32, true)]).unwrap(),
            (224, vec![None, Some(0), Some(160), None])
        );
        // Constructor reserves all three directories; reseed uploads only nodes + overlay.
        let (capacity, _) =
            metadata_upload_layout(&[(64, false), (160, false), (64, false)]).unwrap();
        let (reseed, offsets) = metadata_upload_layout(&[(160, false), (64, false)]).unwrap();
        assert!(reseed <= capacity);
        assert_eq!(offsets, [Some(0), Some(160)]);
        for mask in 0..8 {
            let entries = [16, 160, 64]
                .into_iter()
                .enumerate()
                .map(|(i, size)| (size, mask & (1 << i) != 0))
                .collect::<Vec<_>>();
            let (size, offsets) = metadata_upload_layout(&entries).unwrap();
            let mut cursor = 0;
            for ((length, mapped), offset) in entries.into_iter().zip(offsets) {
                if mapped {
                    assert_eq!(offset, None);
                } else {
                    assert_eq!(offset, Some(cursor));
                    cursor += length;
                }
            }
            assert_eq!(cursor, size);
        }
    }

    #[test]
    fn metadata_direct_mapping_requires_both_pointer_and_actual_coherent_visible_flags() {
        use vk::MemoryPropertyFlags as F;
        for mapped in [false, true] {
            for bits in 0..32 {
                let properties = F::from_raw(bits);
                let expected = mapped
                    && bits & F::HOST_VISIBLE.as_raw() != 0
                    && bits & F::HOST_COHERENT.as_raw() != 0;
                assert_eq!(direct_metadata_mapping(mapped, properties), expected);
            }
        }
        assert!(!direct_metadata_mapping(
            true,
            F::DEVICE_LOCAL | F::HOST_VISIBLE
        ));
        assert!(!direct_metadata_mapping(
            true,
            F::HOST_VISIBLE | F::HOST_CACHED
        ));
        assert!(direct_metadata_mapping(
            true,
            F::HOST_VISIBLE | F::HOST_COHERENT
        ));
        assert!(!direct_metadata_mapping(
            false,
            F::HOST_VISIBLE | F::HOST_COHERENT
        ));
        // Noncoherent mapped directories remain in device memory and reserve staging, whereas
        // coherent mapped directories use no staging bytes. Host accessibility alone is not proof.
        assert_eq!(
            metadata_upload_layout(&[
                (64, direct_metadata_mapping(true, F::HOST_VISIBLE)),
                (
                    160,
                    direct_metadata_mapping(true, F::HOST_VISIBLE | F::HOST_COHERENT)
                ),
            ])
            .unwrap(),
            (64, vec![Some(0), None])
        );
    }

    #[test]
    fn metadata_upload_layout_rejects_partial_words_and_overflow() {
        for mapped in [false, true] {
            for size in [0, 1, 2, 3, 5, usize::MAX] {
                assert!(metadata_upload_layout(&[(size, mapped)]).is_err());
            }
        }
        let largest = (isize::MAX as usize) & !3;
        assert!(metadata_upload_layout(&[(largest, false), (4, false)]).is_err());
        assert_eq!(
            metadata_upload_layout(&[(largest, true), (4, false)]).unwrap(),
            (4, vec![None, Some(0)])
        );
    }

    #[test]
    #[ignore = "requires an explicitly selected idle Vulkan GPU; unmapped metadata upload only"]
    fn gpu_owned_unmapped_metadata_reuses_staging_without_pager_reentry() {
        let mut cfg = infr_core::config::Config::default();
        cfg.device.dev =
            Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select the test GPU explicitly"));
        let backend = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
        // Force real private DEVICE_LOCAL buffers even on ReBAR hardware. UMA may have no private
        // memory type, in which case only the mapped path is available and this probe skips.
        let buffers = [64, 160, 64]
            .into_iter()
            .map(|bytes| backend.alloc_device_local_arena_bda(bytes).unwrap().0)
            .collect::<Vec<_>>();
        if buffers
            .iter()
            .any(|buffer| as_vk_buf(buffer.as_ref()).unwrap().mapped_ptr().is_some())
        {
            eprintln!("unmapped metadata probe unavailable: this device has no private VRAM type");
            return;
        }
        let mapped = backend.alloc_uninit(16, BufferUsage::Staging).unwrap();
        assert!(coherent_metadata_mapping(mapped.as_ref()).unwrap());
        assert!(buffers
            .iter()
            .all(|buffer| !coherent_metadata_mapping(buffer.as_ref()).unwrap()));
        let retained = new_metadata_staging(
            &backend,
            &buffers.iter().map(|b| b.as_ref()).collect::<Vec<_>>(),
        )
        .unwrap();
        let identity = retained.as_ref().unwrap().lock().unwrap().buffer;
        for generation in [1u32, 2, 17] {
            let words = [16, 40, 16]
                .into_iter()
                .enumerate()
                .map(|(index, count)| {
                    (0..count)
                        .map(|word| generation * 1000 + index as u32 * 100 + word)
                        .collect::<Vec<u32>>()
                })
                .collect::<Vec<_>>();
            let mapped_words = [generation; 4];
            let mut uploads = buffers
                .iter()
                .zip(&words)
                .map(|(buffer, words)| (buffer.as_ref(), bytemuck::cast_slice(words)))
                .collect::<Vec<_>>();
            uploads.push((mapped.as_ref(), bytemuck::cast_slice(&mapped_words)));
            // The real caller holds the pager during ownership activation. Any admission hook,
            // phase activation or generic upload would deadlock against this mutex.
            {
                let _pager = backend.moe_pager().lock().unwrap();
                upload_metadata_batch(&backend, retained.as_ref(), &uploads).unwrap();
            }
            assert_eq!(identity, retained.as_ref().unwrap().lock().unwrap().buffer);
            for (buffer, expected) in buffers.iter().zip(&words) {
                let mut actual = vec![0u32; expected.len()];
                backend
                    .download(buffer.as_ref(), bytemuck::cast_slice_mut(&mut actual))
                    .unwrap();
                assert_eq!(&actual, expected);
            }
            let mut actual_mapped = [0u32; 4];
            backend
                .download(
                    mapped.as_ref(),
                    bytemuck::cast_slice_mut(&mut actual_mapped),
                )
                .unwrap();
            assert_eq!(actual_mapped, mapped_words);
            assert!(upload_metadata_batch(&backend, None, &uploads).is_err());
            let bad = [0u8; 3];
            assert!(
                upload_metadata_batch(&backend, retained.as_ref(), &[(mapped.as_ref(), &bad)],)
                    .is_err()
            );
        }
    }

    #[test]
    fn owned_activation_failure_restores_the_prior_forward_flag() {
        use std::sync::atomic::{AtomicBool, Ordering};
        for previous in [false, true] {
            let in_forward = AtomicBool::new(previous);
            let failed = || -> Result<()> {
                let _activation = PreparedOwnedActivation::new(&in_forward);
                in_forward.store(true, Ordering::Release);
                Err(be("injected owned publication failure"))
            };
            assert!(failed().is_err());
            assert_eq!(in_forward.load(Ordering::Acquire), previous);
            let panicked = std::panic::catch_unwind(|| {
                let _activation = PreparedOwnedActivation::new(&in_forward);
                in_forward.store(true, Ordering::Release);
                panic!("injected owned publication unwind");
            });
            assert!(panicked.is_err());
            assert_eq!(in_forward.load(Ordering::Acquire), previous);
        }
    }

    #[test]
    fn owned_activation_success_keeps_the_forward_active() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let in_forward = AtomicBool::new(false);
        {
            let mut activation = PreparedOwnedActivation::new(&in_forward);
            in_forward.store(true, Ordering::Release);
            activation.committed = true;
        }
        assert!(in_forward.load(Ordering::Acquire));
    }

    #[test]
    fn gpu_copy_workgroup_floor_rejects_either_sub256_limit() {
        for (invocations, size_x) in [
            (0, 0),
            (128, 128),
            (255, 256),
            (256, 255),
            (128, 1024),
            (1024, 128),
            (255, u32::MAX),
            (u32::MAX, 255),
        ] {
            assert!(
                !gpu_copy_workgroup_supported(invocations, size_x),
                "must decline before allocation: invocations={invocations} size_x={size_x}"
            );
        }
    }

    #[test]
    fn gpu_copy_workgroup_floor_accepts_small_and_wide_devices() {
        for (invocations, size_x) in [
            (256, 256),
            (256, 1024),
            (1024, 256),
            (1023, 1024),
            (1024, 1023),
            (1024, 1024),
            (u32::MAX, u32::MAX),
        ] {
            assert!(gpu_copy_workgroup_supported(invocations, size_x));
        }
    }

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
            owned: None,
            sources: Arc::new(CpuBuffer),
            candidates: Arc::new(CpuBuffer),
            overlay: Arc::new(CpuBuffer),
            jobs: Arc::new(CpuBuffer),
            miss_predicates: None,
            wide_copy: false,
            copy_groups: 8,
            all_sources_mapped: false,
            unmapped_source_prefix: Vec::new(),
            metadata_staging: None,
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
        let failed: Result<()> = copy.activate_prepared(&be_, &mut session, |session| {
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

        copy.activate_prepared(&be_, &mut session, |_| Ok(()))
            .unwrap();
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
        let blocked: Result<()> = copy.activate_prepared(&be_, &mut session, |_| {
            panic!("a second activation must not publish")
        });
        assert!(blocked.is_err());
        assert_eq!(session.frozen_lut_slots, frozen);
        assert_eq!(copy.epoch.lock().unwrap().generation, token.0);
        copy.complete();
        session.clear_frozen_lut_slots();

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<()> = copy
                .activate_prepared(&be_, &mut session, |_| panic!("injected activation unwind"));
        }));
        assert!(panicked.is_err());
        assert!(!copy.epoch.lock().unwrap().armed);
        assert!(session.frozen_lut_slots.is_empty());
        copy.activate_prepared(&be_, &mut session, |_| Ok(()))
            .unwrap();
        copy.complete();
        session.clear_frozen_lut_slots();
    }
}
