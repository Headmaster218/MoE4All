//! Session-wide GPU residency authority. Host state is imported/exported only at layout changes.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const NONE: u32 = u32::MAX;
const FREE: u32 = 0;
const READY: u32 = 1;
const COPYING: u32 = 2;
const DISABLED: u32 = 3;
const HEADER_NODES: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OwnedBlockSource {
    pool: usize,
    stride_bytes: usize,
    host_chunk: Option<usize>,
    host_offset: usize,
}

struct DirectoryGeometry {
    pool_base: usize,
    slot_base: usize,
    address_base: usize,
    queue_base: usize,
    words: usize,
}

fn record_words(records: usize) -> Result<usize> {
    u32::try_from(records).map_err(|_| be("GPU owner directory exceeds u32"))?;
    let words = records
        .checked_mul(4)
        .ok_or_else(|| be("GPU owner directory word count overflow"))?;
    words
        .checked_mul(std::mem::size_of::<u32>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| be("GPU owner directory byte count overflow"))?;
    Ok(words)
}

fn directory_geometry(pools: usize, total: usize) -> Result<DirectoryGeometry> {
    if pools == 0 || total == 0 {
        return Err(be("GPU owner requires nonempty pool geometry"));
    }
    let slot_base = pools
        .checked_mul(2)
        .and_then(|n| HEADER_NODES.checked_add(n))
        .ok_or_else(|| be("GPU owner pool directory overflow"))?;
    let address_base = slot_base
        .checked_add(total)
        .ok_or_else(|| be("GPU owner slot directory overflow"))?;
    let queue_base = address_base
        .checked_add(total)
        .ok_or_else(|| be("GPU owner address directory overflow"))?;
    let records = total
        .checked_mul(2)
        .and_then(|logs| queue_base.checked_add(pools)?.checked_add(logs))
        .ok_or_else(|| be("GPU owner FIFO directory overflow"))?;
    Ok(DirectoryGeometry {
        pool_base: HEADER_NODES,
        slot_base,
        address_base,
        queue_base,
        words: record_words(records)?,
    })
}

fn registered_blocks<'a>(
    blocks: usize,
    pool_bytes: &[usize],
    sources: impl IntoIterator<Item = (usize, &'a RegisteredExpertSource)>,
) -> Result<Vec<Option<OwnedBlockSource>>> {
    record_words(blocks)?;
    blocks
        .checked_mul(std::mem::size_of::<Option<OwnedBlockSource>>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| be("GPU owner source directory byte count overflow"))?;
    let mut directory = vec![None; blocks];
    for (pool, source) in sources {
        if pool_bytes.get(pool).copied() != Some(source.stride_bytes)
            || source.stride_bytes == 0
            || !source.stride_bytes.is_multiple_of(16)
            || u32::try_from(source.stride_bytes).is_err()
            || source.bank_bytes == 0
            || !source.bank_bytes.is_multiple_of(source.stride_bytes)
        {
            return Err(be("GPU owner registered source has invalid pool or size"));
        }
        let first = source.block_base as usize;
        let end = first
            .checked_add(source.bank_bytes / source.stride_bytes)
            .filter(|&end| end <= blocks && end <= NONE as usize)
            .ok_or_else(|| be("GPU owner registered source exceeds block directory"))?;
        source
            .host_offset
            .checked_add(source.bank_bytes)
            .ok_or_else(|| be("GPU owner registered source offset overflow"))?;
        for (expert, entry) in directory[first..end].iter_mut().enumerate() {
            if entry.is_some() {
                return Err(be("GPU owner registered block ranges overlap"));
            }
            *entry = Some(OwnedBlockSource {
                pool,
                stride_bytes: source.stride_bytes,
                host_chunk: source.host_chunk,
                host_offset: source.host_offset + expert * source.stride_bytes,
            });
        }
    }
    Ok(directory)
}

fn validate_block_owner(
    directory: &[Option<OwnedBlockSource>],
    block: BlockId,
    pool: usize,
    seen: &mut HashSet<BlockId>,
) -> Result<()> {
    let source = directory
        .get(block as usize)
        .and_then(Option::as_ref)
        .ok_or_else(|| be("GPU owner block has no registered source"))?;
    if source.pool != pool || !seen.insert(block) {
        return Err(be(
            "GPU owner block has duplicate or incorrect pool ownership",
        ));
    }
    Ok(())
}

fn validate_physical_slot(address: &[u32], actual: u64, pool: usize, slot: u32) -> Result<()> {
    if address != [actual as u32, (actual >> 32) as u32, pool as u32, slot] {
        return Err(be(
            "GPU owner address directory differs from its physical slot",
        ));
    }
    Ok(())
}

pub(crate) struct OwnedSeed {
    pub nodes: Vec<u32>,
    pub blocks: Vec<u32>,
    pub state: OwnedState,
}

pub(crate) struct OwnedState {
    pub active: AtomicBool,
    pub in_forward: AtomicBool,
    pub generation: AtomicU64,
    pool_starts: Vec<usize>,
    slot_counts: Vec<usize>,
    source_blocks: Vec<Option<OwnedBlockSource>>,
    bank_geometry: HashMap<(u32, u32), [u32; 4]>,
    disabled_slots: std::sync::Mutex<Vec<Vec<bool>>>,
    layout_generation: AtomicU64,
    readback: std::sync::Mutex<Option<crate::VkBuffer>>,
}

impl OwnedState {
    fn role_geometry(&self, bases: [u32; 3], experts: u32) -> Option<[u32; 12]> {
        let mut geometry = [0; 12];
        for (role, base) in bases.into_iter().enumerate() {
            geometry[role * 4..role * 4 + 4]
                .copy_from_slice(self.bank_geometry.get(&(base, experts))?);
        }
        Some(geometry)
    }

    pub(super) fn validate_source_words(&self, words: &[u32]) -> Result<()> {
        if words.len() != record_words(self.source_blocks.len())? {
            return Err(be(
                "GPU owner source directory size differs from its certificate",
            ));
        }
        for (block, source) in self.source_blocks.iter().enumerate() {
            let entry = &words[block * 4..block * 4 + 4];
            let Some(source) = source else {
                if entry != [0; 4] {
                    return Err(be(
                        "GPU owner source directory populates an unregistered block",
                    ));
                }
                continue;
            };
            let address = u64::from(entry[0]) | (u64::from(entry[1]) << 32);
            if source.stride_bytes == 0
                || entry[2] as usize != source.stride_bytes
                || entry[3] as usize != source.pool
                || (address != 0
                    && (address & 15 != 0
                        || address
                            .checked_add(source.stride_bytes as u64 - 1)
                            .is_none()))
            {
                return Err(be(
                    "GPU owner source class or address differs from its certificate",
                ));
            }
        }
        Ok(())
    }
}

pub(crate) struct OwnedHostLease {
    host: Arc<InclusiveHostCache>,
    block: BlockId,
}

impl Drop for OwnedHostLease {
    fn drop(&mut self) {
        self.host.release_gpu_blocks(&[self.block]);
    }
}

/// Validated immutable CPU sources and the exact GPU-selected destinations. No pager lock
/// is needed during execution; the active forward keeps residency ownership unchanged.
type BoundedTransfers = (
    Arc<InclusiveHostCache>,
    Vec<(BlockId, DeviceTransferTarget)>,
);

pub(crate) struct OwnedUploadPlan {
    transfer_plan: Arc<SessionTransferPlan>,
    uploads: Vec<(Arc<AlignedHostBuffer>, usize, DeviceTransferTarget)>,
    bounded: Vec<BoundedTransfers>,
    requested: usize,
    bytes: usize,
    class: OwnedUploadClass,
}

struct OwnedUploadClass {
    mapped: bool,
    no_imported_sources: std::sync::OnceLock<bool>,
}

impl OwnedUploadClass {
    fn from_targets(has_bounded_sources: bool, targets: impl Iterator<Item = bool>) -> Self {
        let mut targets = targets.peekable();
        Self {
            mapped: !has_bounded_sources
                && targets.peek().is_some()
                && targets.all(|mapped| mapped),
            no_imported_sources: std::sync::OnceLock::new(),
        }
    }

    fn host_ready_eligible(&self, no_imported_sources: impl FnOnce() -> bool) -> bool {
        self.mapped && *self.no_imported_sources.get_or_init(no_imported_sources)
    }
}

impl OwnedUploadPlan {
    pub(crate) fn host_ready_eligible(&self) -> bool {
        // This cache belongs only to this plan's retained source/target ranges. Target mappings
        // never change; SessionTransferPlan can only shed imports, never add them. Therefore a
        // cached non-imported batch stays non-imported, while false remains conservative.
        // Classification is lazy so no import lock is acquired under plan_owned_upload's pager lock.
        self.class.host_ready_eligible(|| {
            self.uploads.iter().all(|(source, offset, target)| {
                !self
                    .transfer_plan
                    .contains_imported_upload(unsafe { source.slice(*offset, target.len()) })
            })
        })
    }

    pub(crate) fn execute_mapped_cpu(self, serial_max_bytes: usize) -> Result<()> {
        if !self.host_ready_eligible() {
            return Err(be("host-ready upload is not a certified CPU-only batch"));
        }
        crate::transfer::copy_owned_mapped_batch(
            self.uploads.iter().map(|(source, offset, target)| {
                (unsafe { source.slice(*offset, target.len()) }, target)
            }),
            serial_max_bytes,
        )
    }

    pub(crate) fn record_while_uploading(&self) -> bool {
        self.class.mapped
    }

    pub(crate) fn execute<E: TransferExecutor>(
        self,
        executor: &E,
    ) -> Result<(PreparedHostPush, Vec<OwnedHostLease>)> {
        let _execute = infr_core::timeline::background_span("resident_owned_upload_execute");
        let mut transfer = PreparedTransfer::default();
        let mut leases = Vec::new();
        self.transfer_plan.prepare_upload_batch(
            executor,
            self.uploads.iter().map(|(source, offset, target)| {
                // The complete batch was range-checked before the Started notification.
                (unsafe { source.slice(*offset, target.len()) }, target)
            }),
            &mut transfer,
        )?;
        for (host, jobs) in self.bounded {
            let _host = infr_core::timeline::background_span("resident_owned_upload_host_cache");
            leases.extend(jobs.iter().map(|&(block, _)| OwnedHostLease {
                host: Arc::clone(&host),
                block,
            }));
            let requests = jobs
                .iter()
                .enumerate()
                .map(|(i, &(id, _))| (id, None, i))
                .collect::<Vec<_>>();
            let collected = std::sync::Mutex::new(PreparedTransfer::default());
            host.promote_batch(&requests, |data, index| {
                let mut part = PreparedTransfer::default();
                self.transfer_plan
                    .prepare_upload(executor, data, &jobs[index].1, &mut part)?;
                collected.lock().unwrap().append(part);
                Ok(())
            })?;
            transfer.append(collected.into_inner().unwrap());
        }
        Ok((
            PreparedHostPush {
                requested: self.requested,
                bytes: self.bytes,
                transfer,
            },
            leases,
        ))
    }
}

#[cfg(test)]
fn read_mapped_words(buffer: &dyn Buffer) -> Result<Vec<u32>> {
    if !buffer.len_bytes().is_multiple_of(4) {
        return Err(be("GPU owner metadata has a partial word"));
    }
    read_mapped_range(buffer, 0, buffer.len_bytes() / 4)
}

fn readback_owner_snapshot(
    backend: &VulkanBackend,
    nodes: &dyn Buffer,
    overlay: &dyn Buffer,
    retained: &std::sync::Mutex<Option<crate::VkBuffer>>,
) -> Result<(Vec<u32>, Vec<u32>)> {
    if !Arc::ptr_eq(&as_vk_buf(nodes)?.shared, &backend.shared)
        || !Arc::ptr_eq(&as_vk_buf(overlay)?.shared, &backend.shared)
    {
        return Err(be(
            "GPU owner snapshot belongs to a different Vulkan device",
        ));
    }
    let node_bytes = nodes.len_bytes();
    let overlay_bytes = overlay.len_bytes();
    let bytes = node_bytes
        .checked_add(overlay_bytes)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| be("GPU owner snapshot size overflow"))?;
    if node_bytes == 0 || overlay_bytes == 0 || (node_bytes | overlay_bytes) & 3 != 0 {
        return Err(be("GPU owner snapshot has invalid directory sizes"));
    }
    // Bypass unified allocation and retain the immutable-sized buffer: dropping a temporary
    // VkBuffer at each handoff would invalidate otherwise valid prepared command prefixes.
    let mut retained = retained.lock().unwrap();
    if retained.is_none() {
        *retained = Some(backend.make_buf(
            bytes,
            gpu_allocator::MemoryLocation::GpuToCpu,
            "gpu-owner-handoff-readback",
        )?);
    }
    let staging = retained.as_ref().expect("GPU owner readback was allocated");
    if staging.size != bytes || !Arc::ptr_eq(&staging.shared, &backend.shared) {
        return Err(be("GPU owner retained snapshot geometry or device changed"));
    }
    let crate::Backing::Pooled(allocation) = &staging.backing else {
        return Err(be(
            "GPU owner snapshot requires pooled host readback memory",
        ));
    };
    // gpu-allocator 0.27 requires coherence for GpuToCpu (and prefers HOST_CACHED).
    // Enforce that contract here instead of silently relying on it for mapped CPU reads.
    if !allocation
        .memory_properties()
        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
    {
        return Err(be(
            "GPU owner snapshot requires coherent host readback memory",
        ));
    }
    let recorder = backend.recorder()?;
    // Explicit COMPUTE/SHADER_WRITE -> TRANSFER_READ dependency, including BDA writes.
    recorder.arena_stream_barrier();
    recorder.copy(nodes, 0, staging, 0, node_bytes);
    recorder.copy(overlay, 0, staging, node_bytes, overlay_bytes);
    // Explicit TRANSFER_WRITE -> HOST_READ publication before reading cacheable staging.
    recorder.publish_host_reads();
    recorder.finish_nowait()?.wait()?;
    Ok((
        read_mapped_range(staging, 0, node_bytes / 4)?,
        read_mapped_range(staging, node_bytes / 4, overlay_bytes / 4)?,
    ))
}

fn validate_handoff_geometry(
    nodes: &[u32],
    state: &OwnedState,
    counts: &[usize],
) -> Result<(usize, usize)> {
    if nodes.len() < HEADER_NODES * 4
        || !nodes.len().is_multiple_of(4)
        || nodes[0] != 4
        || counts.is_empty()
        || nodes[1] as usize != counts.len()
        || state.pool_starts.len() != counts.len()
        || state.slot_counts.as_slice() != counts
    {
        return Err(be("invalid GPU owner handoff header or pool geometry"));
    }
    let generation = u64::from(nodes[3]) | (u64::from(nodes[4]) << 32);
    if generation != state.generation.load(Ordering::Acquire) {
        return Err(be("stale GPU owner handoff generation"));
    }
    let mut total = 0usize;
    for (p, &count) in counts.iter().enumerate() {
        if count == 0 || state.pool_starts[p] != total {
            return Err(be("invalid GPU owner handoff pool range"));
        }
        total = total
            .checked_add(count)
            .ok_or_else(|| be("GPU owner handoff slot count overflow"))?;
    }
    let geometry = directory_geometry(counts.len(), total)?;
    // Version 4 appends one FIFO header per pool and two log records per physical slot.
    // Validate the complete geometry before traversing any GPU-provided offsets.
    if nodes[2] as usize != total
        || nodes[5] as usize != geometry.pool_base
        || nodes[6] as usize != geometry.slot_base
        || nodes[7] as usize != geometry.address_base
        || nodes.len() != geometry.words
    {
        return Err(be("invalid GPU owner handoff directory offsets or bounds"));
    }
    Ok((geometry.pool_base, geometry.slot_base))
}

fn read_mapped_range(buffer: &dyn Buffer, start: usize, count: usize) -> Result<Vec<u32>> {
    if start
        .checked_add(count)
        .and_then(|n| n.checked_mul(4))
        .is_none_or(|end| end > buffer.len_bytes())
    {
        return Err(be("GPU owner mapped read exceeds metadata"));
    }
    let vk = as_vk_buf(buffer)?;
    let ptr = vk
        .mapped_ptr()
        .ok_or_else(|| be("GPU owner metadata is not mapped"))?;
    let mut words = vec![0u32; count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            ptr.cast::<u32>().add(start),
            words.as_mut_ptr(),
            words.len(),
        )
    };
    Ok(words)
}

fn live_queue_slots(
    nodes: &[u32],
    queue_header: usize,
    slot_base: usize,
    start: usize,
    capacity: usize,
    residents: usize,
    clock: u64,
) -> Result<Vec<usize>> {
    let records = nodes.len() / 4;
    let log_capacity = capacity
        .checked_mul(2)
        .ok_or_else(|| be("GPU owner FIFO capacity overflow"))?;
    let end = start
        .checked_add(capacity)
        .ok_or_else(|| be("GPU owner FIFO physical range overflow"))?;
    if !nodes.len().is_multiple_of(4)
        || capacity == 0
        || residents > capacity
        || queue_header >= records
        || queue_header
            .checked_add(1)
            .and_then(|base| base.checked_add(log_capacity))
            .is_none_or(|last| last > records)
        || slot_base.checked_add(end).is_none_or(|last| last > records)
    {
        return Err(be("GPU owner FIFO exceeds its directory"));
    }
    let header = &nodes[queue_header * 4..queue_header * 4 + 4];
    let head = header[0] as usize;
    let tail = header[1] as usize;
    if header[2] as usize != log_capacity
        || header[3] != 0
        || head > tail
        || tail > log_capacity
        || residents > tail - head
    {
        return Err(be("invalid GPU owner FIFO header"));
    }
    let mut live = Vec::with_capacity(residents);
    let mut seen = HashSet::with_capacity(residents);
    let mut previous = None;
    for index in head..tail {
        let entry = &nodes[(queue_header + 1 + index) * 4..][..4];
        let cached = u64::from(entry[0]) | (u64::from(entry[1]) << 32);
        let slot = entry[2] as usize;
        if entry[3] != 0 || cached == 0 || cached > clock || slot < start || slot >= end {
            return Err(be("invalid GPU owner FIFO log record"));
        }
        let actual = &nodes[(slot_base + slot) * 4..][..4];
        let stamp = u64::from(actual[0]) | (u64::from(actual[1]) << 32);
        if !matches!(actual[3], READY | COPYING) {
            continue;
        }
        if stamp == 0 || stamp > clock || cached > stamp {
            return Err(be("GPU owner FIFO key exceeds its live timestamp"));
        }
        if cached != stamp {
            continue;
        }
        let key = (stamp, slot);
        if !seen.insert(slot) || previous.is_some_and(|old| old >= key) {
            return Err(be("GPU owner FIFO repeats or misorders a live slot"));
        }
        previous = Some(key);
        live.push(slot);
    }
    if live.len() != residents {
        return Err(be("GPU owner FIFO lost a resident slot"));
    }
    Ok(live)
}

pub(crate) fn seed(session: &MoePagerSession, generation: u64) -> Result<OwnedSeed> {
    let counts: Vec<_> = session.pools.iter().map(|p| p.pager.n_slots()).collect();
    let mut starts = Vec::with_capacity(counts.len());
    let mut total = 0usize;
    for &count in &counts {
        if count == 0 {
            return Err(be("GPU owner requires nonempty physical pools"));
        }
        starts.push(total);
        total = total
            .checked_add(count)
            .ok_or_else(|| be("GPU owner slot count overflow"))?;
    }
    let geometry = directory_geometry(counts.len(), total)?;
    let block_count = session
        .role_stride
        .checked_mul(3)
        .ok_or_else(|| be("GPU owner block count overflow"))?;
    let block_words = record_words(block_count)?;
    let pool_bytes = session
        .pools
        .iter()
        .map(|pool| pool.pager.slot_bytes)
        .collect::<Vec<_>>();
    let source_blocks = registered_blocks(
        block_count,
        &pool_bytes,
        session
            .sources
            .values()
            .map(|(_, pool, source)| (*pool, source)),
    )?;
    for source in source_blocks.iter().flatten() {
        if let Some(chunk) = source.host_chunk {
            if session
                .host_store
                .get(chunk)
                .and_then(|store| store.range(source.host_offset, source.stride_bytes))
                .is_none()
            {
                return Err(be("GPU owner registered source exceeds host store"));
            }
        }
    }
    let DirectoryGeometry {
        pool_base,
        slot_base,
        address_base,
        queue_base,
        words,
    } = geometry;
    let mut nodes = vec![0u32; words];
    let checked = |n: usize| u32::try_from(n).map_err(|_| be("GPU owner directory exceeds u32"));
    nodes[..8].copy_from_slice(&[
        4,
        checked(counts.len())?,
        checked(total)?,
        generation as u32,
        (generation >> 32) as u32,
        checked(pool_base)?,
        checked(slot_base)?,
        checked(address_base)?,
    ]);
    let mut blocks = vec![0u32; block_words];
    let mut all_blocks = HashSet::new();
    let mut physical_ranges = Vec::with_capacity(total);
    let mut disabled_slots = Vec::with_capacity(counts.len());
    for (pool_id, pool) in session.pools.iter().enumerate() {
        let snapshot = pool
            .pager
            .pager
            .snapshot_for_handoff()
            .map_err(|e| be(format!("GPU owner handoff: {e:?}")))?;
        disabled_slots.push(snapshot.disabled().to_vec());
        let start = starts[pool_id];
        let queue_header = queue_base + pool_id + start * 2;
        let queue_capacity = counts[pool_id] * 2;
        let queue_word = queue_header * 4;
        nodes[queue_word..queue_word + 4].copy_from_slice(&[
            0,
            checked(snapshot.resident_lru().len())?,
            checked(queue_capacity)?,
            0,
        ]);
        let global = |slot: u32| checked(start + slot as usize);
        let mut free = Vec::new();
        for slot in 0..counts[pool_id] {
            let g = start + slot;
            let node = (slot_base + g) * 4;
            nodes[node..node + 4].copy_from_slice(&[
                NONE,
                NONE,
                NONE,
                if pool.pager.pager.slot_enabled(slot as u32) {
                    FREE
                } else {
                    DISABLED
                },
            ]);
            let addr = pool.pager.slot_addr(slot as u32)?;
            if pool.pager.pager.slot_enabled(slot as u32) {
                let end = addr
                    .checked_add(pool.pager.slot_bytes as u64)
                    .ok_or_else(|| be("GPU owner physical slot address overflow"))?;
                if addr == 0
                    || addr & 15 != 0
                    || pool.pager.slot_bytes == 0
                    || pool.pager.slot_bytes & 15 != 0
                {
                    return Err(be("GPU owner requires aligned nonempty expert slots"));
                }
                physical_ranges.push((addr, end));
            }
            let a = (address_base + g) * 4;
            nodes[a..a + 4].copy_from_slice(&[
                addr as u32,
                (addr >> 32) as u32,
                pool_id as u32,
                slot as u32,
            ]);
        }
        for (index, &(block, slot)) in snapshot.resident_lru().iter().enumerate() {
            validate_block_owner(&source_blocks, block, pool_id, &mut all_blocks)?;
            let g = global(slot)?;
            let stamp = u64::try_from(index + 1).map_err(|_| be("GPU owner clock overflow"))?;
            let node = (slot_base + g as usize) * 4;
            nodes[node..node + 4].copy_from_slice(&[
                stamp as u32,
                (stamp >> 32) as u32,
                block,
                READY,
            ]);
            let entry = (queue_header + 1 + index) * 4;
            nodes[entry..entry + 4].copy_from_slice(&[stamp as u32, (stamp >> 32) as u32, g, 0]);
            let addr = pool.pager.slot_addr(slot)?;
            let b = block as usize * 4;
            if b + 4 > blocks.len() {
                return Err(be("GPU owner block exceeds registered directory"));
            }
            blocks[b..b + 4].copy_from_slice(&[addr as u32, (addr >> 32) as u32, READY, g]);
        }
        for &slot in snapshot.free().iter().rev() {
            free.push(global(slot)?);
        }
        for (i, &g) in free.iter().enumerate() {
            nodes[(slot_base + g as usize) * 4 + 1] = free.get(i + 1).copied().unwrap_or(NONE);
        }
        let p = (pool_base + pool_id * 2) * 4;
        nodes[p..p + 4].copy_from_slice(&[
            checked(queue_header)?,
            checked(snapshot.resident_lru().len())?,
            free.first().copied().unwrap_or(NONE),
            pool.pager.pager.enabled_slots() as u32,
        ]);
        let clock = (nodes[10] as u64 | (u64::from(nodes[11]) << 32))
            .max(snapshot.resident_lru().len() as u64);
        nodes[10] = clock as u32;
        nodes[11] = (clock >> 32) as u32;
    }
    physical_ranges.sort_unstable_by_key(|range| range.0);
    if physical_ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(be("GPU owner expert slot payloads overlap"));
    }
    let mut bank_geometry = HashMap::new();
    for (_, pool, source) in session.sources.values() {
        let key = (
            source.block_base,
            checked(source.bank_bytes / source.stride_bytes)?,
        );
        let value = [
            checked(*pool)?,
            checked(source.stride_bytes)?,
            checked(starts[*pool])?,
            checked(counts[*pool])?,
        ];
        if bank_geometry
            .insert(key, value)
            .is_some_and(|old| old != value)
        {
            return Err(be("GPU owner bank has inconsistent immutable geometry"));
        }
    }
    Ok(OwnedSeed {
        nodes,
        blocks,
        state: OwnedState {
            active: AtomicBool::new(false),
            in_forward: AtomicBool::new(false),
            generation: AtomicU64::new(generation),
            pool_starts: starts,
            slot_counts: counts,
            source_blocks,
            bank_geometry,
            disabled_slots: std::sync::Mutex::new(disabled_slots),
            layout_generation: AtomicU64::new(session.unified_pool.generation()),
            readback: std::sync::Mutex::new(None),
        },
    })
}

impl MoePagerSession {
    pub(crate) fn gpu_decode_layer_capacity(&self) -> usize {
        self.sources
            .values()
            .map(|(_, _, source)| source.layer_base)
            .collect::<HashSet<_>>()
            .len()
    }

    /// Call only at an execution boundary, before changing any expert addresses or arena owners.
    pub(crate) fn handoff_gpu_decode(&mut self, backend: &VulkanBackend) -> Result<()> {
        let Some(copy) = self.gpu_decode_owner.clone() else {
            return Ok(());
        };
        copy.handoff_owned(self, backend)
    }

    pub(crate) fn gpu_decode_in_forward(&self) -> bool {
        self.gpu_decode_owner
            .as_ref()
            .is_some_and(|copy| copy.owned_in_forward())
    }

    pub(crate) fn finish_gpu_decode_forward(&self) {
        if let Some(copy) = &self.gpu_decode_owner {
            copy.finish_owned_forward();
        }
    }
}

impl super::gpu_copy::GpuCopyBuffers {
    pub(crate) fn owned_statistics(
        &self,
        backend: &VulkanBackend,
        session: &MoePagerSession,
    ) -> Result<Option<Vec<PagerStats>>> {
        let Some(state) = self.owned.as_ref().filter(|_| self.owned_active()) else {
            return Ok(None);
        };
        if state.in_forward.load(Ordering::Acquire) {
            return Err(be("GPU owner statistics require a drained token boundary"));
        }
        // Diagnostics reuse the explicit snapshot transport, never import its LRU or clear active.
        let (nodes, _) = readback_owner_snapshot(
            backend,
            self.candidates.as_ref(),
            self.overlay.as_ref(),
            &state.readback,
        )?;
        let counts = session
            .pools
            .iter()
            .map(|pool| pool.pager.n_slots())
            .collect::<Vec<_>>();
        let (pool_base, _) = validate_handoff_geometry(&nodes, state, &counts)?;
        session
            .pools
            .iter()
            .enumerate()
            .map(|(pool, entry)| {
                let mut stats = entry.pager.stats();
                let offset = (pool_base + pool * 2 + 1) * 4;
                stats.hits = stats
                    .hits
                    .checked_add(u64::from(nodes[offset]))
                    .ok_or_else(|| be("GPU owner statistics hit count overflow"))?;
                stats.misses = stats
                    .misses
                    .checked_add(u64::from(nodes[offset + 1]))
                    .ok_or_else(|| be("GPU owner statistics miss count overflow"))?;
                stats.evictions = stats
                    .evictions
                    .checked_add(u64::from(nodes[offset + 2]))
                    .ok_or_else(|| be("GPU owner statistics eviction count overflow"))?;
                Ok(stats)
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    pub(crate) fn owned_role_geometry(&self, bases: [u32; 3], experts: u32) -> Option<[u32; 12]> {
        self.owned.as_ref()?.role_geometry(bases, experts)
    }

    pub(crate) fn is_owned(&self) -> bool {
        self.owned.is_some()
    }

    pub(crate) fn owned_in_forward(&self) -> bool {
        self.owned
            .as_ref()
            .is_some_and(|s| s.in_forward.load(Ordering::Acquire))
    }

    pub(crate) fn owned_active(&self) -> bool {
        self.owned
            .as_ref()
            .is_some_and(|s| s.active.load(Ordering::Acquire))
    }

    pub(crate) fn finish_owned_forward(&self) {
        if let Some(state) = &self.owned {
            state.in_forward.store(false, Ordering::Release);
        }
    }

    pub(super) fn begin_owned(
        &self,
        backend: &VulkanBackend,
        session: &MoePagerSession,
    ) -> Result<()> {
        let state = self
            .owned
            .as_ref()
            .ok_or_else(|| be("missing GPU ownership state"))?;
        if !state.active.load(Ordering::Acquire) {
            let generation = state
                .generation
                .load(Ordering::Relaxed)
                .checked_add(1)
                .ok_or_else(|| be("GPU owner generation overflow"))?;
            let initial = seed(session, generation)?;
            if initial.nodes.len() * 4 != self.candidates.len_bytes()
                || initial.blocks.len() * 4 != self.overlay.len_bytes()
                || initial.state.pool_starts != state.pool_starts
                || initial.state.slot_counts != state.slot_counts
                || initial.state.source_blocks != state.source_blocks
                || initial.state.bank_geometry != state.bank_geometry
            {
                return Err(be(
                    "GPU owner directory geometry or registered sources changed",
                ));
            }
            // Bounded RAM retains its own cache policy. Old GPU shadow pins no longer express
            // VRAM ownership; new upload sources are leased only until their transfers finish.
            for pool in &session.pools {
                if let Some(host) = &pool.host {
                    let ids = pool
                        .pager
                        .pager
                        .resident_slots_lru()
                        .iter()
                        .map(|&(id, _)| id)
                        .collect::<Vec<_>>();
                    host.release_gpu_blocks(&ids);
                }
            }
            self.upload_owned_seed(backend, &initial.nodes, &initial.blocks)?;
            state
                .layout_generation
                .store(session.unified_pool.generation(), Ordering::Relaxed);
            *state.disabled_slots.lock().unwrap() =
                initial.state.disabled_slots.into_inner().unwrap();
            state.generation.store(generation, Ordering::Relaxed);
            state.active.store(true, Ordering::Release);
        } else if state.layout_generation.load(Ordering::Relaxed)
            != session.unified_pool.generation()
        {
            // The allocator generation also counts unrelated runtime frees. Those do not move
            // immutable expert addresses; automatic slot restoration is deferred while GPU-owned.
            let disabled = state.disabled_slots.lock().unwrap();
            let unchanged = session.pools.len() == disabled.len()
                && session
                    .pools
                    .iter()
                    .zip(disabled.iter())
                    .all(|(pool, mask)| {
                        pool.pager.n_slots() == mask.len()
                            && pool.pager.pager.disabled() == mask.as_slice()
                    });
            if !unchanged {
                return Err(be("GPU-owned expert layout changed without a handoff"));
            }
            state
                .layout_generation
                .store(session.unified_pool.generation(), Ordering::Relaxed);
        }
        state.in_forward.store(true, Ordering::Release);
        Ok(())
    }

    pub(crate) fn handoff_owned(
        &self,
        session: &mut MoePagerSession,
        backend: &VulkanBackend,
    ) -> Result<()> {
        let Some(state) = &self.owned else {
            return Ok(());
        };
        if !state.active.load(Ordering::Acquire) {
            return Ok(());
        }
        if state.in_forward.load(Ordering::Acquire) {
            return Err(be(
                "cannot change unified VRAM layout during a GPU-owned forward; prepare its resources before execution",
            ));
        }
        let (nodes, overlay) = readback_owner_snapshot(
            backend,
            self.candidates.as_ref(),
            self.overlay.as_ref(),
            &state.readback,
        )?;
        let counts = session
            .pools
            .iter()
            .map(|pool| pool.pager.n_slots())
            .collect::<Vec<_>>();
        let (pool_base, slot_base) = validate_handoff_geometry(&nodes, state, &counts)?;
        if overlay.len() != record_words(state.source_blocks.len())? {
            return Err(be("GPU owner block directory geometry changed"));
        }
        let clock = u64::from(nodes[10]) | (u64::from(nodes[11]) << 32);
        let queue_base = nodes[7] as usize + nodes[2] as usize;
        let mut all_blocks = HashSet::new();
        let mut snapshots = Vec::with_capacity(session.pools.len());
        for (p, pool) in session.pools.iter().enumerate() {
            let original = pool
                .pager
                .pager
                .snapshot_for_handoff()
                .map_err(|e| be(format!("GPU owner snapshot: {e:?}")))?;
            let mut resident_lru = Vec::new();
            let mut free = Vec::new();
            let start = state.pool_starts[p];
            let end = start + state.slot_counts[p];
            let h = (pool_base + p * 2) * 4;
            for slot in 0..state.slot_counts[p] {
                let a = (nodes[7] as usize + start + slot) * 4;
                validate_physical_slot(
                    &nodes[a..a + 4],
                    pool.pager.slot_addr(slot as u32)?,
                    p,
                    slot as u32,
                )?;
            }
            let mut seen = HashSet::new();
            let mut ordered = Vec::new();
            let resident_count = nodes[h + 1] as usize;
            let queue_header = queue_base + p + start * 2;
            if nodes[h] as usize != queue_header
                || resident_count > state.slot_counts[p]
                || nodes[h + 3] as usize != pool.pager.pager.enabled_slots()
                || (resident_count == nodes[h + 3] as usize && nodes[h + 2] != NONE)
            {
                return Err(be("GPU owner FIFO geometry disagrees"));
            }
            let live = live_queue_slots(
                &nodes,
                queue_header,
                slot_base,
                start,
                state.slot_counts[p],
                resident_count,
                clock,
            )?;
            for g in live {
                if g < start || g >= end || !seen.insert(g) {
                    return Err(be("GPU owner FIFO repeats a slot or crosses pools"));
                }
                let n = (slot_base + g) * 4;
                let stamp = u64::from(nodes[n]) | (u64::from(nodes[n + 1]) << 32);
                if !matches!(nodes[n + 3], READY | COPYING)
                    || stamp == 0
                    || stamp > clock
                    || original.disabled()[g - start]
                {
                    return Err(be("invalid GPU owner resident timestamp or membership"));
                }
                validate_block_owner(&state.source_blocks, nodes[n + 2], p, &mut all_blocks)?;
                let b = nodes[n + 2] as usize * 4;
                let a = (nodes[7] as usize + g) * 4;
                if b.checked_add(4).is_none_or(|end| end > overlay.len())
                    || overlay[b..b + 2] != nodes[a..a + 2]
                    || overlay[b + 2] != nodes[n + 3]
                    || overlay[b + 3] as usize != g
                    || nodes[a + 2] as usize != p
                    || nodes[a + 3] as usize != g - start
                {
                    return Err(be(
                        "GPU owner resident directory disagrees with its physical slot",
                    ));
                }
                ordered.push((stamp, nodes[n + 2], (g - start) as u32, nodes[n + 3]));
            }
            ordered.sort_unstable_by_key(|entry| entry.0);
            if ordered.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err(be("GPU owner has ambiguous LRU timestamps"));
            }
            for (_, block, slot, status) in ordered {
                if status == READY {
                    resident_lru.push((block, slot));
                } else {
                    free.push(slot);
                }
            }
            let mut cursor = nodes[h + 2];
            let mut free_order = Vec::new();
            while cursor != NONE {
                let g = cursor as usize;
                if g < start || g >= end || !seen.insert(g) {
                    return Err(be("GPU owner free chain is cyclic or crosses pools"));
                }
                let n = (slot_base + g) * 4;
                if nodes[n + 3] != FREE || original.disabled()[g - start] {
                    return Err(be("GPU owner free chain names an occupied slot"));
                }
                free_order.push((g - start) as u32);
                cursor = nodes[n + 1];
            }
            free_order.reverse();
            free.extend(free_order);
            if seen.len() != pool.pager.pager.enabled_slots() {
                return Err(be("GPU owner lost an enabled slot"));
            }
            let mut stats = original.stats();
            stats.hits = stats
                .hits
                .checked_add(u64::from(nodes[h + 4]))
                .ok_or_else(|| be("GPU owner hit counter overflow"))?;
            stats.misses = stats
                .misses
                .checked_add(u64::from(nodes[h + 5]))
                .ok_or_else(|| be("GPU owner miss counter overflow"))?;
            stats.evictions = stats
                .evictions
                .checked_add(u64::from(nodes[h + 6]))
                .ok_or_else(|| be("GPU owner eviction counter overflow"))?;
            if resident_lru
                .iter()
                .any(|&(block, _)| block as usize >= pool.pager.lut_host.len())
            {
                return Err(be("GPU owner block exceeds the host LUT"));
            }
            snapshots.push(infr_core::pager::PagerSnapshot::from_parts(
                original.disabled().to_vec(),
                free,
                resident_lru,
                stats,
            ));
        }
        for (block, entry) in overlay.chunks_exact(4).enumerate() {
            if entry[2] > COPYING || (entry[2] != 0 && !all_blocks.contains(&(block as u32))) {
                return Err(be("GPU owner block directory has an unowned resident"));
            }
        }
        // All pools validate before any host state changes. The core import also validates first.
        for (pool, snapshot) in session.pools.iter().zip(&snapshots) {
            pool.pager
                .pager
                .validate_handoff(snapshot)
                .map_err(|e| be(format!("GPU owner validation: {e:?}")))?;
        }
        for (pool, snapshot) in session.pools.iter_mut().zip(&snapshots) {
            pool.pager
                .pager
                .restore_handoff(snapshot)
                .map_err(|e| be(format!("GPU owner restore: {e:?}")))?;
            pool.pager.lut_host.fill(NOT_RESIDENT);
            for &(block, slot) in snapshot.resident_lru() {
                let entry = pool
                    .pager
                    .lut_host
                    .get_mut(block as usize)
                    .ok_or_else(|| be("GPU handoff block exceeds host LUT"))?;
                *entry = slot;
            }
            pool.pager.lut_dirty = true;
        }
        state.active.store(false, Ordering::Release);
        session.clear_frozen_lut_slots();
        Ok(())
    }

    /// CPU services exactly the GPU-reserved destinations. It never touches host residency/LRU.
    pub(crate) fn prepare_owned_upload<E: TransferExecutor>(
        &self,
        session: &MoePagerSession,
        executor: &E,
        ordinal: usize,
        words: &[u32; crate::recorder::RESIDENT_JOURNAL_WORDS],
    ) -> Result<(PreparedHostPush, Vec<OwnedHostLease>)> {
        let _prepare = infr_core::timeline::background_span("resident_owned_upload_prepare");
        self.plan_owned_upload(session, ordinal, words)?
            .execute(executor)
    }

    pub(crate) fn plan_owned_upload(
        &self,
        session: &MoePagerSession,
        ordinal: usize,
        words: &[u32; crate::recorder::RESIDENT_JOURNAL_WORDS],
    ) -> Result<OwnedUploadPlan> {
        let _validate = infr_core::timeline::background_span("resident_owned_upload_validate");
        let state = self
            .owned
            .as_ref()
            .ok_or_else(|| be("CPU upload has no GPU owner"))?;
        if !state.active.load(Ordering::Acquire) || !state.in_forward.load(Ordering::Acquire) {
            return Err(be("CPU upload outlived the GPU-owned forward"));
        }
        let generation = u64::from(words[crate::recorder::RESIDENT_JOURNAL_JOB_COUNT_WORD + 1])
            | (u64::from(words[crate::recorder::RESIDENT_JOURNAL_JOB_COUNT_WORD + 2]) << 32);
        if words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] != 5
            || generation != state.generation.load(Ordering::Acquire)
        {
            return Err(be("CPU upload received a stale or non-service GPU notice"));
        }
        let count = words[crate::recorder::RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize;
        let start = ordinal
            .checked_mul(super::gpu_copy::JOBS_PER_LAYER * 8)
            .ok_or_else(|| be("owned upload offset overflow"))?;
        if count == 0 || count > super::gpu_copy::JOBS_PER_LAYER {
            return Err(be("invalid GPU CPU-upload job count"));
        }
        let jobs = {
            let _read = infr_core::timeline::background_span("resident_owned_upload_jobs_read");
            read_mapped_range(self.jobs.as_ref(), start, count * 8)?
        };
        let total = state
            .pool_starts
            .last()
            .zip(state.slot_counts.last())
            .and_then(|(&start, &count)| start.checked_add(count))
            .ok_or_else(|| be("invalid GPU owner upload pool geometry"))?;
        let mut uploads = Vec::with_capacity(count);
        let mut bounded = BTreeMap::<usize, Vec<(BlockId, DeviceTransferTarget)>>::new();
        let mut bytes = 0usize;
        let mut requested = 0;
        let mut seen = HashSet::with_capacity(count);
        let mut destinations = HashSet::with_capacity(count);
        for job in jobs.chunks_exact(8) {
            let block = job[5];
            let g = job[6] as usize;
            if g >= total || !seen.insert(block) || !destinations.insert(g) {
                return Err(be("GPU CPU-upload repeated an invalid destination"));
            }
            let source = state
                .source_blocks
                .get(block as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| be("GPU CPU-upload block has no registered source"))?;
            // The leased notice is the GPU owner's request, not a second admission vote.
            // Validate its destination against immutable host geometry without reading live LRU.
            let pool_index = source.pool;
            let slot = state
                .pool_starts
                .get(pool_index)
                .and_then(|&start| g.checked_sub(start))
                .and_then(|slot| u32::try_from(slot).ok())
                .ok_or_else(|| be("GPU CPU-upload destination is outside its source pool"))?;
            let b = (block as usize)
                .checked_mul(4)
                .ok_or_else(|| be("GPU CPU-upload source offset overflow"))?;
            let source_words = self
                .source_words
                .get(
                    b..b.checked_add(4)
                        .ok_or_else(|| be("GPU CPU-upload source end overflow"))?,
                )
                .ok_or_else(|| be("GPU CPU-upload source directory is truncated"))?;
            if job[4] as usize != source.stride_bytes
                || source_words != [job[0], job[1], job[4], pool_index as u32]
                || state
                    .slot_counts
                    .get(pool_index)
                    .is_none_or(|&count| slot as usize >= count)
                || state
                    .pool_starts
                    .get(pool_index)
                    .and_then(|start| start.checked_add(slot as usize))
                    != Some(g)
            {
                return Err(be("GPU CPU-upload source geometry differs"));
            }
            let pool = session
                .pools
                .get(pool_index)
                .ok_or_else(|| be("GPU CPU-upload pool is missing"))?;
            if !pool.pager.pager.slot_enabled(slot) {
                return Err(be("GPU CPU-upload targets a disabled slot"));
            }
            validate_physical_slot(
                &[job[2], job[3], pool_index as u32, slot],
                pool.pager.slot_addr(slot)?,
                pool_index,
                slot,
            )?;
            if job[0] != 0 || job[1] != 0 {
                continue;
            }
            let target = pool.pager.slot_copy_target(slot)?;
            if let Some(chunk) = source.host_chunk {
                let offset = source.host_offset;
                if session
                    .host_store
                    .get(chunk)
                    .and_then(|store| store.range(offset, source.stride_bytes))
                    .is_none()
                {
                    return Err(be("GPU CPU-upload exceeds host store"));
                }
                uploads.push((Arc::clone(&session.host_store[chunk].bytes), offset, target));
            } else {
                if session.pools[pool_index].host.is_none() {
                    return Err(be("GPU CPU-upload has no RAM or SSD source"));
                }
                bounded.entry(pool_index).or_default().push((block, target));
            }
            bytes = bytes
                .checked_add(source.stride_bytes)
                .ok_or_else(|| be("GPU CPU-upload byte count overflow"))?;
            requested += 1;
        }
        if requested == 0 {
            return Err(be("GPU requested CPU service without an unmapped expert"));
        }
        let class = OwnedUploadClass::from_targets(
            !bounded.is_empty(),
            uploads
                .iter()
                .map(|(_, _, target)| target.mapped_ptr().is_some()),
        );
        let bounded = bounded
            .into_iter()
            .map(|(pool, jobs)| (Arc::clone(session.pools[pool].host.as_ref().unwrap()), jobs))
            .collect();
        Ok(OwnedUploadPlan {
            transfer_plan: Arc::clone(&session.transfer_plan),
            uploads,
            bounded,
            requested,
            bytes,
            class,
        })
    }
}

#[cfg(test)]
#[path = "pager_owned_upload_cpu_tests.rs"]
mod owned_upload_cpu_tests;

#[cfg(test)]
mod geometry_tests {
    use super::*;

    fn fixture() -> (Vec<u32>, OwnedState) {
        let mut nodes = vec![0; 29 * 4];
        nodes[..8].copy_from_slice(&[4, 2, 5, 42, 0, 3, 7, 12]);
        let state = OwnedState {
            active: AtomicBool::new(true),
            in_forward: AtomicBool::new(false),
            generation: AtomicU64::new(42),
            pool_starts: vec![0, 2],
            slot_counts: vec![2, 3],
            source_blocks: Vec::new(),
            bank_geometry: HashMap::new(),
            disabled_slots: std::sync::Mutex::new(vec![vec![false; 2], vec![false; 3]]),
            layout_generation: AtomicU64::new(9),
            readback: std::sync::Mutex::new(None),
        };
        (nodes, state)
    }

    #[test]
    fn handoff_geometry_checks_every_truncation_before_traversal() {
        let (nodes, state) = fixture();
        assert_eq!(
            validate_handoff_geometry(&nodes, &state, &[2, 3]).unwrap(),
            (3, 7)
        );
        for length in 0..nodes.len() {
            assert!(validate_handoff_geometry(&nodes[..length], &state, &[2, 3]).is_err());
        }
        let mut oversized = nodes.clone();
        oversized.extend_from_slice(&[0; 4]);
        assert!(validate_handoff_geometry(&oversized, &state, &[2, 3]).is_err());
        assert!(state.active.load(Ordering::Acquire));
        assert!(!state.in_forward.load(Ordering::Acquire));
        assert_eq!(state.generation.load(Ordering::Acquire), 42);
    }

    #[test]
    fn handoff_geometry_rejects_corrupt_counts_offsets_and_generation() {
        let (nodes, state) = fixture();
        for index in 0..8 {
            let mut bad = nodes.clone();
            bad[index] = u32::MAX;
            assert!(
                validate_handoff_geometry(&bad, &state, &[2, 3]).is_err(),
                "header word {index}"
            );
        }
        for index in [5, 6, 7] {
            let mut bad = nodes.clone();
            bad[index] = 0;
            assert!(validate_handoff_geometry(&bad, &state, &[2, 3]).is_err());
        }
        assert!(validate_handoff_geometry(&nodes, &state, &[3, 2]).is_err());
        assert!(validate_handoff_geometry(&nodes, &state, &[]).is_err());
    }

    #[test]
    fn handoff_geometry_rejects_bad_host_ranges_and_overflow() {
        let (nodes, mut state) = fixture();
        state.pool_starts.pop();
        assert!(validate_handoff_geometry(&nodes, &state, &[2, 3]).is_err());
        state.pool_starts = vec![0, 1];
        assert!(validate_handoff_geometry(&nodes, &state, &[2, 3]).is_err());
        state.pool_starts = vec![0, usize::MAX];
        state.slot_counts = vec![usize::MAX, 1];
        assert!(validate_handoff_geometry(&nodes, &state, &[usize::MAX, 1]).is_err());
        state.pool_starts = vec![0, 0];
        state.slot_counts = vec![0, 5];
        assert!(validate_handoff_geometry(&nodes, &state, &[0, 5]).is_err());
    }

    #[test]
    fn directory_sizes_reject_overflow_before_allocation() {
        let geometry = directory_geometry(2, 5).unwrap();
        assert_eq!(
            (
                geometry.pool_base,
                geometry.slot_base,
                geometry.address_base,
                geometry.queue_base,
                geometry.words
            ),
            (3, 7, 12, 17, 116)
        );
        assert!(directory_geometry(usize::MAX, 1).is_err());
        assert!(directory_geometry(1, usize::MAX).is_err());
        assert!(directory_geometry(1, u32::MAX as usize / 3).is_err());
        assert!(directory_geometry(0, 1).is_err());
        assert!(record_words(usize::MAX).is_err());
        assert!(registered_blocks(usize::MAX, &[], std::iter::empty()).is_err());
    }

    fn fifo_fixture() -> (Vec<u32>, usize, usize) {
        let geometry = directory_geometry(1, 4).unwrap();
        let mut nodes = vec![0; geometry.words];
        let q = geometry.queue_base;
        nodes[q * 4..q * 4 + 4].copy_from_slice(&[1, 5, 8, 0]);
        for (slot, entry) in [
            [2, 0, 100, READY],
            [5, 0, 101, COPYING],
            [0, NONE, NONE, FREE],
            [NONE, NONE, NONE, DISABLED],
        ]
        .into_iter()
        .enumerate()
        {
            let word = (geometry.slot_base + slot) * 4;
            nodes[word..word + 4].copy_from_slice(&entry);
        }
        for (index, entry) in [
            [NONE; 4],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [4, 0, 1, 0],
            [5, 0, 1, 0],
            [NONE; 4],
        ]
        .into_iter()
        .enumerate()
        {
            let word = (q + 1 + index) * 4;
            nodes[word..word + 4].copy_from_slice(&entry);
        }
        (nodes, q, geometry.slot_base)
    }

    #[test]
    fn fifo_handoff_filters_stale_logs_and_retains_copying_in_exact_live_order() {
        let (nodes, q, slots) = fifo_fixture();
        assert_eq!(
            live_queue_slots(&nodes, q, slots, 0, 4, 2, 9).unwrap(),
            [0, 1]
        );
        // Dead prefix/suffix bytes are intentionally malformed and must not be read.
        let mut carried = nodes;
        let high = u64::from(u32::MAX) + 9;
        for slot in 0..2 {
            carried[(slots + slot) * 4 + 1] = 1;
        }
        for index in 1..5 {
            carried[(q + 1 + index) * 4 + 1] = 1;
        }
        assert_eq!(
            live_queue_slots(&carried, q, slots, 0, 4, 2, high).unwrap(),
            [0, 1]
        );
    }

    #[test]
    fn fifo_handoff_rejects_malformed_live_coverage_bounds_keys_and_order() {
        let (nodes, q, slots) = fifo_fixture();
        for corruption in 0..14 {
            let mut bad = nodes.clone();
            let first = (q + 1 + 2) * 4;
            let second = (q + 1 + 4) * 4;
            match corruption {
                0 => bad[q * 4] = 6,
                1 => bad[q * 4 + 1] = 9,
                2 => bad[q * 4 + 2] = 7,
                3 => bad[q * 4 + 3] = 1,
                4 => bad[first] = 0,
                5 => bad[first] = 10,
                6 => bad[first + 2] = NONE,
                7 => bad[first + 3] = 1,
                8 => {
                    let entry = bad[first..first + 4].to_vec();
                    bad[first + 4..first + 8].copy_from_slice(&entry);
                }
                9 => {
                    let a = bad[first..first + 4].to_vec();
                    let b = bad[second..second + 4].to_vec();
                    bad[first..first + 4].copy_from_slice(&b);
                    bad[second..second + 4].copy_from_slice(&a);
                }
                10 => bad[second] = 4,
                11 => bad[(q + 1 + 3) * 4] = 6,
                12 => bad[slots * 4] = 10,
                _ => bad[(slots + 1) * 4 + 3] = FREE,
            }
            assert!(
                live_queue_slots(&bad, q, slots, 0, 4, 2, 9).is_err(),
                "FIFO corruption {corruption}"
            );
        }
        for length in 0..nodes.len() {
            assert!(live_queue_slots(&nodes[..length], q, slots, 0, 4, 2, 9).is_err());
        }
        assert!(live_queue_slots(&nodes, usize::MAX, slots, 0, 4, 2, 9).is_err());
        assert!(live_queue_slots(&nodes, q, usize::MAX, 0, 4, 2, 9).is_err());
        assert!(live_queue_slots(&nodes, q, slots, usize::MAX, 4, 2, 9).is_err());
        assert!(live_queue_slots(&nodes, q, slots, 0, usize::MAX, 2, 9).is_err());
        assert!(live_queue_slots(&nodes, q, slots, 0, 4, 5, 9).is_err());
    }

    fn source(first: u32, bytes: usize) -> RegisteredExpertSource {
        RegisteredExpertSource {
            stride_bytes: 16,
            layer_base: 0,
            block_base: first,
            host_chunk: Some(0),
            host_offset: 64,
            bank_bytes: bytes,
        }
    }

    #[test]
    fn registered_directory_preserves_sparse_ids_and_rejects_bad_ownership() {
        let a = source(1, 32);
        let b = source(7, 16);
        let directory = registered_blocks(12, &[16, 16], [(0, &a), (1, &b)]).unwrap();
        assert!(directory[0].is_none());
        assert!(directory[3..7].iter().all(Option::is_none));
        assert_eq!(directory[2].unwrap().host_offset, 80);
        let mut seen = HashSet::new();
        validate_block_owner(&directory, 1, 0, &mut seen).unwrap();
        assert!(validate_block_owner(&directory, 1, 0, &mut seen).is_err());
        assert!(validate_block_owner(&directory, 7, 0, &mut seen).is_err());
        validate_block_owner(&directory, 7, 1, &mut seen).unwrap();
        assert!(validate_block_owner(&directory, 4, 0, &mut seen).is_err());
        assert!(validate_block_owner(&directory, NONE, 0, &mut seen).is_err());
    }

    #[test]
    fn immutable_source_certificate_checks_classes_holes_spans_and_exact_banks() {
        let (_, mut state) = fixture();
        state.source_blocks = vec![
            Some(OwnedBlockSource {
                pool: 0,
                stride_bytes: 16,
                host_chunk: None,
                host_offset: 0,
            }),
            None,
            Some(OwnedBlockSource {
                pool: 1,
                stride_bytes: 32,
                host_chunk: None,
                host_offset: 0,
            }),
        ];
        let words = [0x1000u32, 1, 16, 0, 0, 0, 0, 0, 0, 0, 32, 1];
        state.validate_source_words(&words).unwrap();
        for (index, value) in [
            (0, 0x1001),
            (2, 32),
            (3, 1),
            (4, 16),
            (6, 16),
            (10, 16),
            (11, 0),
        ] {
            let mut bad = words;
            bad[index] = value;
            assert!(state.validate_source_words(&bad).is_err(), "word {index}");
        }
        let mut overflow = words;
        overflow[0] = u32::MAX - 15;
        overflow[1] = u32::MAX;
        overflow[2] = 32;
        state.source_blocks[0].as_mut().unwrap().stride_bytes = 32;
        assert!(state.validate_source_words(&overflow).is_err());
        assert!(state.validate_source_words(&words[..11]).is_err());
        state.bank_geometry.insert((0, 4), [0, 16, 0, 2]);
        state.bank_geometry.insert((8, 4), [1, 32, 2, 3]);
        assert_eq!(
            state.role_geometry([0, 0, 8], 4).unwrap(),
            [0, 16, 0, 2, 0, 16, 0, 2, 1, 32, 2, 3]
        );
        assert!(state.role_geometry([0, 0, 8], 3).is_none());
        assert!(state.role_geometry([0, 1, 8], 4).is_none());
    }

    #[test]
    fn registered_directory_rejects_overlaps_ranges_sizes_and_offsets() {
        let a = source(1, 32);
        let b = source(2, 16);
        assert!(registered_blocks(12, &[16, 16], [(0, &a), (1, &b)]).is_err());
        assert!(registered_blocks(2, &[16], [(0, &a)]).is_err());
        assert!(registered_blocks(12, &[16], [(1, &a)]).is_err());
        assert!(registered_blocks(12, &[32], [(0, &a)]).is_err());
        for bytes in [0, 17] {
            assert!(registered_blocks(12, &[16], [(0, &source(1, bytes))]).is_err());
        }
        let mut bad = a.clone();
        bad.stride_bytes = 0;
        assert!(registered_blocks(12, &[0], [(0, &bad)]).is_err());
        bad = a.clone();
        bad.host_offset = usize::MAX;
        assert!(registered_blocks(12, &[16], [(0, &bad)]).is_err());
        bad = a;
        bad.block_base = NONE;
        assert!(registered_blocks(12, &[16], [(0, &bad)]).is_err());
    }

    #[test]
    fn physical_slot_validation_uses_host_address_not_matching_gpu_copies() {
        let actual = 0x1234_0000_0040u64;
        let valid = [actual as u32, (actual >> 32) as u32, 1, 3];
        validate_physical_slot(&valid, actual, 1, 3).unwrap();
        for index in 0..4 {
            let mut bad = valid;
            bad[index] ^= 16;
            assert!(validate_physical_slot(&bad, actual, 1, 3).is_err());
        }
        assert!(validate_physical_slot(&valid[..3], actual, 1, 3).is_err());
    }

    #[test]
    fn fifo_exact_lru_survives_aliases_protection_compaction_and_clock_carry() {
        use std::collections::VecDeque;
        const SLOTS: usize = 32;
        let mut clock = u64::from(u32::MAX) - SLOTS as u64;
        let mut owners = (0..SLOTS as u32).collect::<Vec<_>>();
        let mut stamps = (0..SLOTS)
            .map(|i| clock - SLOTS as u64 + i as u64 + 1)
            .collect::<Vec<_>>();
        let mut log = stamps
            .iter()
            .enumerate()
            .map(|(slot, &stamp)| (stamp, slot))
            .collect::<Vec<_>>();
        let mut reference = owners.iter().copied().collect::<VecDeque<_>>();
        let mut rng = 17u32;
        let mut compactions = 0;
        for step in 0..5000 {
            let count = 1 + step % 24;
            let request = (0..count)
                .map(|i| {
                    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                    if i > 0 && i % 3 == 0 {
                        rng % 16
                    } else {
                        rng % 96
                    }
                })
                .collect::<Vec<_>>();
            let protected = request.iter().copied().collect::<HashSet<_>>();
            let mut last = HashMap::new();
            for (q, &block) in request.iter().enumerate() {
                last.insert(block, clock + q as u64 + 1);
            }
            // All existing hits, including the last role's, are protected before admission.
            for (slot, &block) in owners.iter().enumerate() {
                if let Some(&stamp) = last.get(&block) {
                    stamps[slot] = stamp;
                }
            }
            for &block in &request {
                if owners.contains(&block) {
                    continue;
                }
                let slot = log
                    .iter()
                    .find_map(|&(key, slot)| {
                        (key == stamps[slot] && !protected.contains(&owners[slot])).then_some(slot)
                    })
                    .unwrap();
                let victim = reference
                    .iter()
                    .position(|id| !protected.contains(id))
                    .unwrap();
                assert_eq!(owners[slot], reference.remove(victim).unwrap());
                assert!(!protected.contains(&owners[slot]));
                owners[slot] = block;
                stamps[slot] = last[&block];
                reference.push_back(block);
            }
            // Append one record per distinct touched slot, in last-request order. Existing
            // records become stale; compaction retains exactly the live ordered coverage.
            let mut touched = last
                .iter()
                .map(|(&block, &stamp)| (stamp, owners.iter().position(|&id| id == block).unwrap()))
                .collect::<Vec<_>>();
            touched.sort_unstable();
            if log.len() + touched.len() > 2 * SLOTS {
                log.retain(|&(stamp, slot)| stamp == stamps[slot]);
                compactions += 1;
            }
            assert!(log.len() + touched.len() <= 2 * SLOTS);
            log.extend(touched);
            for &block in &request {
                reference.retain(|&id| id != block);
                reference.push_back(block);
            }
            clock += request.len() as u64;
            let mut actual = (0..SLOTS).collect::<Vec<_>>();
            actual.sort_unstable_by_key(|&slot| stamps[slot]);
            assert_eq!(
                actual.iter().map(|&slot| owners[slot]).collect::<Vec<_>>(),
                reference.iter().copied().collect::<Vec<_>>()
            );
            let live = log
                .iter()
                .filter(|&&(stamp, slot)| stamp == stamps[slot])
                .map(|&(_, slot)| slot)
                .collect::<Vec<_>>();
            assert_eq!(live, actual);
        }
        assert!(clock > u64::from(u32::MAX));
        assert!(compactions > 0);
    }
}

#[cfg(test)]
#[path = "pager_gpu_owner_tests.rs"]
mod tests;
