//! The tensor-parallel ALL-REDUCE — sum each rank's partial `[tokens, n_embd]` output of a
//! row-parallel projection (attn-O / ffn-down) across all ranks so every rank ends with the full
//! sum, the residual stream identical everywhere. This is the comms-critical piece of TP: two of
//! these run per transformer layer.
//!
//! # Data path — host-LESS P2P dma-buf
//!
//! Each rank `p` publishes its partial into an EXPORTED device-local buffer `E_p`
//! ([`VulkanBackend::p2p_export`]); every other rank imports `E_p` ([`p2p_import`]) so it can read
//! `p`'s VRAM directly over PCIe (no host bounce — the campaign measured 12.0-25.3 GiB/s vs 3.56
//! host-bounce). A rank then copies each peer's published partial into a local scratch and sums
//! `own + Σ peers` with an on-device `Op::Add` chain. The reduction is a FIXED rank-order sum, so it
//! is deterministic run-to-run (bit-reproducible), and — crucially for correctness — every rank
//! computes the SAME set of addends in the SAME order, so all ranks agree bit-for-bit after the
//! reduce (the residual stream can't drift between devices). A device pair that can't share a
//! dma-buf falls back to a host bounce for that transport, reported as [`AllReduceMode::Host`].
//!
//! # Sync — external semaphore vs host fence
//!
//! The gather must not read `E_p` before rank `p` finished writing it. v1 orders this with the
//! backend's host fence (`copy_buffer`'s `queue_wait_idle`): correct, but each all-reduce host-stalls
//! — 2 per layer, so on a deep model the host round-trips dominate. The zero-host-stall optimization
//! (`VK_KHR_external_semaphore_fd`: rank `p` signals a semaphore its readers wait on, GPU-side) is
//! the "don't skimp" target; when the device pair supports it the mode is
//! [`AllReduceMode::P2pSemaphore`], else [`AllReduceMode::P2pHostFence`]. See [`AllReduce::mode`].
//!
//! # Generality
//!
//! The `publish → gather-all-peers → sum` schedule is correct for any world size `W` (each rank
//! reads `W-1` peers into `W-1` scratches, an all-to-all-ish exchange of `O(W²)` total bandwidth).
//! A ring/tree schedule (`O(W)` bandwidth) is the >2-device throughput optimization, deferred.

use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Plan};
use infr_core::error::Result;
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};

use crate::{be, P2pExport, P2pHandleType, TpExportSemaphore, TpImportSemaphore, VulkanBackend};

/// How the all-reduce moves + orders data across the device pair (for the report).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllReduceMode {
    /// Host-less P2P dma-buf data path, cross-device ordering by `VK_KHR_external_semaphore_fd`
    /// (no host round-trip). The optimized target.
    P2pSemaphore,
    /// Host-less P2P dma-buf data path, cross-device ordering by the host fence (`queue_wait_idle`).
    /// Correct but host-stalls twice per layer — the reported perf gap when the semaphore path is
    /// unavailable.
    P2pHostFence,
    /// No usable cross-device dma-buf: partials bounce through host RAM. Slowest; correctness only.
    Host,
}

/// Element size (bytes) of a boundary/activation dtype — the single shared helper for the
/// tensor-parallel and expert-parallel boundary sizing (hoisted out of `tp.rs`/`ep.rs`, which had
/// byte-identical copies). Boundary tensors are plain f32 activations, so this is just the element
/// size (`block_layout` reports `(1, bytes)` for the scalar float dtypes).
pub(crate) fn dtype_bytes(dt: DType) -> usize {
    let (elems, bytes) = infr_gguf::block_layout(dt);
    bytes / elems.max(1)
}

/// Element count of a `bytes`-byte boundary in `dtype`, rejecting any boundary the reduce Add chain
/// can't sum. The reduce is compiled over the f32 elementwise `add` shader (see
/// [`build_reduce_graph`]), so an f16 (or any non-f32) boundary would have that f32 Add read raw
/// halves as garbage — reject it CLEANLY at construction instead of mis-summing (or hard-erroring
/// later in [`AllReduce::reduce`] via the `elems != bytes/4` mismatch). Carrying a true f16 Add chain
/// is deferred until the reduce graph gains an f16 `add`; today f32 is the only correct boundary.
pub(crate) fn allreduce_elems(bytes: usize, dtype: DType) -> Result<usize> {
    if dtype != DType::F32 {
        return Err(be(format!(
            "tp all-reduce: boundary dtype {dtype:?} is unsupported — the reduce Add chain is \
             f32-only, so only an f32 boundary can be all-reduced (an f16 boundary would sum raw \
             halves as garbage)"
        )));
    }
    let esize = dtype_bytes(dtype); // 4 for f32
    if bytes == 0 || !bytes.is_multiple_of(esize) {
        return Err(be(format!(
            "tp all-reduce: boundary size {bytes} is not a positive multiple of {esize} ({dtype:?})"
        )));
    }
    Ok(bytes / esize)
}

/// Require every boundary tensor to be the SAME byte size and return it. The transport compiles ONE
/// reduce plan of a fixed `elems`, and [`AllReduce::reduce`] requires `elems == self.elems` exactly,
/// so sizing to the `max` of differing boundaries would silently break a model whose row-parallel
/// boundaries differ in width. Holds trivially today (every boundary is `[tokens, n_embd]`); this
/// makes the assumption explicit and fails loudly if a future model violates it.
pub(crate) fn uniform_boundary_bytes(sizes: &[usize]) -> Result<usize> {
    let mut it = sizes.iter().copied();
    let first = it.next().unwrap_or(0);
    for s in it {
        if s != first {
            return Err(be(format!(
                "tp all-reduce: boundary tensors differ in size ({first} vs {s} bytes) — the \
                 single reduce transport requires all boundaries be the same width (per-size \
                 transports are not implemented)"
            )));
        }
    }
    Ok(first)
}

/// The per-rank all-reduce transport for ONE boundary size, set up once and reused every layer.
pub struct AllReduce {
    mode: AllReduceMode,
    bytes: usize,
    /// f32 element count of the boundary tensor (`bytes / 4`).
    elems: usize,
    /// Rank `p`'s exported publish buffer (`Some` in a P2P mode). Indexed by producer rank.
    exports: Vec<Option<P2pExport>>,
    /// `imported[r][p]` — rank `r`'s import alias of rank `p`'s export (`p != r`; `None` at `p==r`).
    /// Only populated in a P2P mode.
    imported: Vec<Vec<Option<Box<dyn Buffer>>>>,
    /// `scratch[r][p]` — rank `r`'s local buffer holding peer `p`'s partial before the sum
    /// (`None` at `p==r`).
    scratch: Vec<Vec<Option<Box<dyn Buffer>>>>,
    /// Rank `p`'s exported timeline semaphore (`Some` in `P2pSemaphore` mode). Signalled `value` on
    /// `p`'s publish submit; peers wait it on their gather submit. Indexed by producer rank.
    export_sems: Vec<Option<TpExportSemaphore>>,
    /// `import_sems[r][p]` — rank `r`'s import of rank `p`'s timeline semaphore (`p != r`; shared
    /// payload). Only populated in `P2pSemaphore` mode.
    import_sems: Vec<Vec<Option<TpImportSemaphore>>>,
    /// Monotonic timeline value: incremented each all-reduce so every semaphore signal is strictly
    /// increasing (a timeline requirement).
    step: AtomicU64,
    /// Per-rank compiled Add-chain plan (`own += Σ peers`).
    reduce_plans: Vec<Box<dyn Plan>>,
    /// The reduce graph's tensor handles: `sub` (own partial, in/out) + one scratch input per peer.
    sub_tid: TensorId,
    scratch_tids: Vec<TensorId>,
    /// Host-bounce scratch, ONE buffer per producer rank (Host mode only). Each producer is
    /// downloaded ONCE into `host_bufs[p]` then uploaded to every peer's scratch, instead of
    /// re-downloading the same producer `W-1` times (the dominant PCIe read otherwise multiplied).
    host_bufs: std::sync::Mutex<Vec<Vec<u8>>>,
}

// The P2P exports/imports are Send/Sync under the same whole-backend discipline as the buffers.
unsafe impl Send for AllReduce {}
unsafe impl Sync for AllReduce {}

impl AllReduce {
    /// The transport mode chosen for the report.
    pub fn mode(&self) -> AllReduceMode {
        self.mode
    }

    /// Build the all-reduce transport over `ranks` for a boundary of `bytes` bytes in `dtype`. Sets
    /// up the P2P export/import ring (when every rank supports dma-buf and `use_p2p`) + per-rank
    /// scratches + the per-rank Add-chain reduce plan. `dtype` MUST be f32 (the reduce Add is f32);
    /// any other boundary is rejected up front by [`allreduce_elems`].
    pub fn new(ranks: &[VulkanBackend], bytes: usize, dtype: DType, use_p2p: bool) -> Result<Self> {
        let w = ranks.len();
        let elems = allreduce_elems(bytes, dtype)?;

        // ── choose the data + sync mode ────────────────────────────────────────────────────────
        let all_dma = use_p2p && ranks.iter().all(|b| b.p2p_supported(P2pHandleType::DmaBuf));

        let mut exports: Vec<Option<P2pExport>> = (0..w).map(|_| None).collect();
        let mut imported: Vec<Vec<Option<Box<dyn Buffer>>>> =
            (0..w).map(|_| (0..w).map(|_| None).collect()).collect();
        let mut p2p_ok = all_dma;

        if all_dma {
            // Publish buffer per producer rank.
            for (p, ex) in exports.iter_mut().enumerate() {
                match ranks[p].p2p_export(bytes, P2pHandleType::DmaBuf) {
                    Ok(e) => *ex = Some(e),
                    Err(e) => {
                        tracing::warn!(
                            "tp all-reduce: p2p_export on rank {p} failed ({e}); host bounce"
                        );
                        p2p_ok = false;
                        break;
                    }
                }
            }
            // Cross-import: rank r imports every other rank's export.
            if p2p_ok {
                'outer: for r in 0..w {
                    for p in 0..w {
                        if p == r {
                            continue;
                        }
                        let export = exports[p].as_ref().expect("export set");
                        match ranks[r].p2p_import(export) {
                            Ok(buf) => imported[r][p] = Some(buf),
                            Err(e) => {
                                tracing::warn!(
                                    "tp all-reduce: p2p import rank{p}->rank{r} rejected ({e}); \
                                     host bounce"
                                );
                                p2p_ok = false;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        if !p2p_ok {
            exports = (0..w).map(|_| None).collect();
            imported = (0..w).map(|_| (0..w).map(|_| None).collect()).collect();
        }

        // The external-semaphore ordering rides on the P2P data path. Attempt it (mode
        // P2pSemaphore) only when the P2P path is live AND every rank supports the fd; if any
        // cross-device semaphore import is rejected (a valid hardware finding), fall back to the
        // host fence and report the exact failure.
        let mut export_sems: Vec<Option<TpExportSemaphore>> = (0..w).map(|_| None).collect();
        let mut import_sems: Vec<Vec<Option<TpImportSemaphore>>> =
            (0..w).map(|_| (0..w).map(|_| None).collect()).collect();
        let mut sem_ok = p2p_ok && w > 1 && ranks.iter().all(|b| b.external_semaphore_supported());
        if sem_ok {
            for (p, es) in export_sems.iter_mut().enumerate() {
                match ranks[p].tp_export_timeline() {
                    Ok(e) => *es = Some(e),
                    Err(e) => {
                        tracing::warn!(
                            "tp all-reduce: semaphore export on rank {p} failed ({e}); host-fence"
                        );
                        sem_ok = false;
                        break;
                    }
                }
            }
        }
        if sem_ok {
            'outer: for r in 0..w {
                for p in 0..w {
                    if p == r {
                        continue;
                    }
                    let exp = export_sems[p].as_ref().expect("export sem set");
                    match ranks[r].tp_import_timeline(exp) {
                        Ok(imp) => import_sems[r][p] = Some(imp),
                        Err(e) => {
                            tracing::warn!(
                                "tp all-reduce: cross-device semaphore import rank{p}->rank{r} \
                                 rejected ({e}); host-fence"
                            );
                            sem_ok = false;
                            break 'outer;
                        }
                    }
                }
            }
        }
        if !sem_ok {
            export_sems = (0..w).map(|_| None).collect();
            import_sems = (0..w).map(|_| (0..w).map(|_| None).collect()).collect();
        }
        let mode = if !p2p_ok {
            AllReduceMode::Host
        } else if sem_ok {
            AllReduceMode::P2pSemaphore
        } else {
            AllReduceMode::P2pHostFence
        };
        if w > 1 {
            let sync = match mode {
                AllReduceMode::P2pSemaphore => "external-semaphore (no host round-trip)",
                AllReduceMode::P2pHostFence => "host-fence (queue_wait_idle)",
                AllReduceMode::Host => "host-bounce (no cross-device dma-buf)",
            };
            let data = if p2p_ok { "P2P dma-buf" } else { "host RAM" };
            tracing::info!(
                "tp all-reduce: {w}-way, {} bytes/boundary — data path: {data}, cross-device sync: {sync}",
                bytes
            );
        }

        // ── per-rank scratches (hold each peer's partial before the sum) ───────────────────────
        let mut scratch: Vec<Vec<Option<Box<dyn Buffer>>>> = Vec::with_capacity(w);
        #[allow(clippy::needless_range_loop)] // r indexes ranks AND builds the r-th scratch row
        for r in 0..w {
            let mut row = Vec::with_capacity(w);
            for p in 0..w {
                if p == r {
                    row.push(None);
                } else {
                    row.push(Some(ranks[r].alloc(bytes, BufferUsage::Activations)?));
                }
            }
            scratch.push(row);
        }

        // ── per-rank Add-chain reduce plan: sub += Σ scratch[peer] ─────────────────────────────
        let (reduce_graph, sub_tid, scratch_tids) = build_reduce_graph(elems, w);
        let mut reduce_plans = Vec::with_capacity(w);
        for b in ranks {
            reduce_plans.push(b.compile(&reduce_graph)?);
        }

        Ok(Self {
            mode,
            bytes,
            elems,
            exports,
            imported,
            scratch,
            export_sems,
            import_sems,
            step: AtomicU64::new(0),
            reduce_plans,
            sub_tid,
            scratch_tids,
            host_bufs: std::sync::Mutex::new((0..w).map(|_| vec![0u8; bytes]).collect()),
        })
    }

    /// All-reduce: `bufs[r]` holds rank `r`'s partial on entry; on return every `bufs[r]` holds the
    /// full sum `Σ_r bufs[r]`. `elems` is the live element count (`<= self.elems`; a smaller live
    /// count than the allocated boundary would need a smaller Add — v1 keeps them equal).
    pub fn reduce(
        &self,
        ranks: &[VulkanBackend],
        bufs: &[Box<dyn Buffer>],
        elems: usize,
    ) -> Result<()> {
        if elems != self.elems {
            return Err(be(format!(
                "tp all-reduce: live elems {elems} != transport elems {} (boundary size must match \
                 the compiled reduce plan)",
                self.elems
            )));
        }
        let w = ranks.len();
        if w == 1 {
            return Ok(()); // world=1: the partial IS the full sum, nothing to reduce.
        }
        match self.mode {
            AllReduceMode::P2pSemaphore => self.reduce_p2p_semaphore(ranks, bufs),
            AllReduceMode::P2pHostFence => self.reduce_p2p_hostfence(ranks, bufs),
            AllReduceMode::Host => self.reduce_host(ranks, bufs),
        }
    }

    /// P2P data path with GPU-side cross-device ordering (`VK_KHR_external_semaphore_fd`): the host
    /// issues every rank's publish (signal) and gather (wait) submit back-to-back WITHOUT waiting on a
    /// peer's GPU — the timeline semaphores enforce publish→gather ordering on the devices, so the
    /// GPUs pipeline. The ONLY host waits are a single `queue_wait_idle` per rank after the gather (a
    /// memory barrier so the reduce dispatch safely reads scratch) plus the reduce's own sync — no
    /// cross-device host round-trip. Eliminates the host-fence path's serial "wait producer, then let
    /// consumer read" stall (its dominant per-layer cost).
    fn reduce_p2p_semaphore(
        &self,
        ranks: &[VulkanBackend],
        bufs: &[Box<dyn Buffer>],
    ) -> Result<()> {
        let w = ranks.len();
        // Strictly-increasing timeline value for this all-reduce (first call = 1 > initial 0).
        let v = self.step.fetch_add(1, Ordering::SeqCst) + 1;

        // Collected submitted command buffers. EVERY buffer pushed here was SUCCESSFULLY submitted
        // (a failed submit frees its own cmd inside `tp_submit_*`), so on ANY outcome we must wait
        // for them to complete and free them — a mid-loop `?` must not abandon in-flight GPU work or
        // leak the long-lived pool's command buffers (finding: tp_sem error-path leaks).
        let mut pub_cmds: Vec<(usize, vk::CommandBuffer)> = Vec::with_capacity(w);
        let mut gat_cmds: Vec<(usize, vk::CommandBuffer)> = Vec::with_capacity(w);

        let mut submit = || -> Result<()> {
            // ── PUBLISH (no host wait): each rank copies its partial → its export buffer, signalling
            //    its timeline semaphore = v when the copy completes (+ releases the export to
            //    QUEUE_FAMILY_EXTERNAL for the cross-device read). ──────────────────────────────────
            for p in 0..w {
                let ex = self.exports[p].as_ref().expect("p2p export");
                let sem = self.export_sems[p].as_ref().expect("export sem");
                let cmd = ranks[p].tp_submit_copy_signal(
                    bufs[p].as_ref(),
                    ex.buffer(),
                    self.bytes,
                    sem,
                    v,
                )?;
                pub_cmds.push((p, cmd));
            }

            // ── GATHER (no host wait): each rank acquires + copies every peer's export → its scratch,
            //    its submit WAITING GPU-side on the peer's semaphore ≥ v (so it can't read a partial
            //    before it is published). ─────────────────────────────────────────────────────────
            #[allow(clippy::needless_range_loop)]
            // r indexes ranks, imported[r]/scratch[r]/import_sems[r]
            for r in 0..w {
                let mut copies: Vec<(&dyn Buffer, &dyn Buffer, usize)> = Vec::with_capacity(w - 1);
                let mut waits: Vec<(&TpImportSemaphore, u64)> = Vec::with_capacity(w - 1);
                for p in 0..w {
                    if p == r {
                        continue;
                    }
                    let imp = self.imported[r][p].as_ref().expect("p2p import");
                    let sc = self.scratch[r][p].as_ref().expect("scratch");
                    copies.push((imp.as_ref(), sc.as_ref(), self.bytes));
                    waits.push((self.import_sems[r][p].as_ref().expect("import sem"), v));
                }
                let cmd = ranks[r].tp_submit_copies_wait(&copies, &waits)?;
                gat_cmds.push((r, cmd));
            }
            Ok(())
        };
        let outcome = submit();

        // ── residual host wait: one queue_wait_idle per rank (the memory barrier for the reduce
        //    read + the point past which the collected cmds are complete and free-able). All
        //    cross-device ordering already happened GPU-side above. Run this on EVERY outcome so an
        //    error mid-submit still drains + frees what was already submitted. ─────────────────────
        let mut wait_err: Option<infr_core::error::Error> = None;
        for rank in ranks {
            if let Err(e) = rank.tp_queue_wait_idle() {
                wait_err.get_or_insert(e);
            }
        }
        for (p, cmd) in pub_cmds {
            ranks[p].tp_free_cmds(&[cmd]);
        }
        for (r, cmd) in gat_cmds {
            ranks[r].tp_free_cmds(&[cmd]);
        }
        outcome?;
        if let Some(e) = wait_err {
            return Err(e);
        }

        // ── reduce: each rank sums own + all peers' scratches into its own partial. ──────────────
        self.run_reduce_plans(ranks, bufs)
    }

    /// P2P data path ordered by the HOST fence (`queue_wait_idle`) — correct but the host serializes
    /// "wait producer, then let consumer read" per exchange (the per-layer stall the semaphore path
    /// removes). Used when the device pair can't share an external semaphore.
    fn reduce_p2p_hostfence(
        &self,
        ranks: &[VulkanBackend],
        bufs: &[Box<dyn Buffer>],
    ) -> Result<()> {
        let w = ranks.len();
        // ── publish: each rank copies its partial into its exported buffer, then RELEASES that
        //    buffer's queue-family ownership to QUEUE_FAMILY_EXTERNAL (the cross-device read). ─────
        for p in 0..w {
            let ex = self.exports[p].as_ref().expect("p2p export");
            ranks[p].p2p_publish_copy(bufs[p].as_ref(), ex.buffer(), self.bytes)?;
        }
        // ── gather: each rank ACQUIRES every peer's exported buffer from QUEUE_FAMILY_EXTERNAL then
        //    reads it into a local scratch. (Reads only — race-free after publish.) ───────────────
        #[allow(clippy::needless_range_loop)]
        // r indexes ranks, imported[r] and scratch[r] together
        for r in 0..w {
            for p in 0..w {
                if p == r {
                    continue;
                }
                let imp = self.imported[r][p].as_ref().expect("p2p import");
                let sc = self.scratch[r][p].as_ref().expect("scratch");
                ranks[r].p2p_gather_copy(imp.as_ref(), sc.as_ref(), self.bytes)?;
            }
        }
        // ── reduce: each rank sums own + all peers' scratches into its own partial ──────────────
        self.run_reduce_plans(ranks, bufs)
    }

    /// Host-bounce fallback: download each producer's partial through host RAM ONCE, then upload it
    /// to every peer's scratch — instead of re-downloading each producer `W-1` times (the dominant
    /// PCIe read). Same summed bytes, same order, so the reduced value is unchanged.
    fn reduce_host(&self, ranks: &[VulkanBackend], bufs: &[Box<dyn Buffer>]) -> Result<()> {
        let w = ranks.len();
        let mut host = self.host_bufs.lock().expect("tp host bufs poisoned");
        // ── download each producer once ────────────────────────────────────────────────────────
        for p in 0..w {
            ranks[p].download(bufs[p].as_ref(), &mut host[p])?;
        }
        // ── fan each producer's bytes out to every other rank's scratch ────────────────────────
        #[allow(clippy::needless_range_loop)] // r indexes ranks AND scratch[r] together
        for r in 0..w {
            for p in 0..w {
                if p == r {
                    continue;
                }
                let sc = self.scratch[r][p].as_ref().expect("scratch");
                ranks[r].upload(sc.as_ref(), &host[p])?;
            }
        }
        drop(host);
        self.run_reduce_plans(ranks, bufs)
    }

    /// Run each rank's compiled Add-chain: `bufs[r] += Σ scratch[r][peer]`.
    fn run_reduce_plans(&self, ranks: &[VulkanBackend], bufs: &[Box<dyn Buffer>]) -> Result<()> {
        let w = ranks.len();
        for r in 0..w {
            let mut b = Bindings::new();
            b.bind(self.sub_tid, bufs[r].as_ref());
            // scratch_tids are ordered by peer index skipping self; bind them to this rank's peers
            // in the SAME (ascending peer) order so every rank sums the same addends deterministically.
            let mut k = 0usize;
            for p in 0..w {
                if p == r {
                    continue;
                }
                let sc = self.scratch[r][p].as_ref().expect("scratch");
                b.bind(self.scratch_tids[k], sc.as_ref());
                k += 1;
            }
            ranks[r].execute(self.reduce_plans[r].as_ref(), &b)?;
            ranks[r].sync()?;
        }
        Ok(())
    }
}

/// Build the reduce graph: `sub` (own partial, `[elems]` f32, in/out) plus `world-1` scratch inputs,
/// chained `sub = sub + scratch_k`. The Adds are sequential (each reads the running `sub`), which is
/// a deterministic fixed-order sum.
fn build_reduce_graph(elems: usize, world: usize) -> (Graph, TensorId, Vec<TensorId>) {
    let mut g = Graph::new();
    let sub = g.input(TensorDesc::new(vec![elems], DType::F32));
    let scratch_tids: Vec<TensorId> = (0..world.saturating_sub(1))
        .map(|_| g.input(TensorDesc::new(vec![elems], DType::F32)))
        .collect();
    for &s in &scratch_tids {
        g.push(Op::Add {
            a: sub,
            b: s,
            dst: sub,
            n: elems as u32,
        });
    }
    (g, sub, scratch_tids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_bytes_scalar_floats() {
        assert_eq!(dtype_bytes(DType::F32), 4);
        assert_eq!(dtype_bytes(DType::F16), 2);
        assert_eq!(dtype_bytes(DType::Bf16), 2);
    }

    #[test]
    fn allreduce_elems_f32_ok() {
        // elems derived from the boundary dtype's element size, not a hardwired /4.
        assert_eq!(allreduce_elems(4096, DType::F32).unwrap(), 1024);
        assert_eq!(allreduce_elems(4, DType::F32).unwrap(), 1);
    }

    #[test]
    fn allreduce_elems_rejects_zero_and_unaligned() {
        assert!(allreduce_elems(0, DType::F32).is_err());
        assert!(allreduce_elems(6, DType::F32).is_err()); // not a multiple of 4
    }

    #[test]
    fn allreduce_elems_rejects_non_f32_boundary() {
        // An f16 boundary is cleanly rejected at construction (the reduce Add chain is f32-only) —
        // no longer a mysterious `numel != bytes/4` hard-error inside reduce().
        let err = allreduce_elems(4096, DType::F16).unwrap_err();
        assert!(format!("{err}").contains("f32"), "err was: {err}");
        assert!(allreduce_elems(4096, DType::Bf16).is_err());
    }

    #[test]
    fn uniform_boundary_bytes_agrees_or_rejects() {
        assert_eq!(uniform_boundary_bytes(&[4096, 4096, 4096]).unwrap(), 4096);
        assert_eq!(uniform_boundary_bytes(&[]).unwrap(), 0);
        assert!(uniform_boundary_bytes(&[4096, 2048]).is_err());
    }
}
