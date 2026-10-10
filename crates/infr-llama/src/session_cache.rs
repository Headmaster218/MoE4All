//! Opt-in cold storage for idle Vulkan conversation state.
//!
//! The scheduler owns policy (which resident slot to evict and which prefix to restore). This
//! module owns the durable representation, model/geometry validation, streaming device transfers,
//! and directory maintenance. No code on the ordinary generation path reaches this module when
//! `kv.session_cache_dir` is unset.

use crate::mtp::{Qwen4MtpCacheState, Qwen4MtpCheckpointState, Qwen4MtpSession};
use crate::sampling::StepGate;
use crate::seam::{SeamKv, SessionBufferKey, SessionStateMeta};
use crate::{Config, EngineConfig};
use anyhow::{anyhow, Context, Result};
use infr_core::backend::Backend;
use infr_core::{DType, SizeSpec};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, FileTimes, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAGIC: [u8; 8] = *b"INFRKV01";
const VERSION: u32 = 2;
const HEADER_BYTES: u64 = 112;
const RECORD_HEADER_BYTES: u64 = 16;
const CHECKSUM_BYTES: u64 = 32;
const MTP_MAGIC: [u8; 8] = *b"INFRMT01";
const MTP_VERSION: u32 = 1;
const MTP_HEADER_BYTES: u64 = 112;
const MTP_FLAG_AGENT_CHECKPOINT: u32 = 1;
const MTP_FLAG_EDIT_CHECKPOINT: u32 = 2;
const MTP_FLAG_PENDING: u32 = 4;
const MTP_FLAG_PENDING_EMITTED: u32 = 8;
const MTP_FLAGS: u32 = MTP_FLAG_AGENT_CHECKPOINT
    | MTP_FLAG_EDIT_CHECKPOINT
    | MTP_FLAG_PENDING
    | MTP_FLAG_PENDING_EMITTED;
const STREAM_BYTES: usize = 16 * 1024 * 1024;
const MAX_RECORDS: u32 = 16_384;
const FLAG_AGENT_CHECKPOINT: u32 = 1;
const FLAG_EDIT_CHECKPOINT: u32 = 2;
const CHECKPOINT_FLAGS: u32 = FLAG_AGENT_CHECKPOINT | FLAG_EDIT_CHECKPOINT;
const MIN_STALE_TEMP_AGE: Duration = Duration::from_secs(24 * 60 * 60);

static FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) struct SessionCache {
    root: PathBuf,
    model_dir: PathBuf,
    fingerprint: [u8; 32],
    max_bytes: u64,
    ttl: Option<Duration>,
    max_ctx: usize,
    k_fmt: DType,
    v_fmt: DType,
    entries: Vec<ColdEntry>,
}

pub(crate) struct ColdEntry {
    path: PathBuf,
    saved_at: u64,
    meta: SessionStateMeta,
    file_bytes: u64,
}

pub(crate) struct MtpCacheSource<'a> {
    pub(crate) head: &'a Qwen4MtpSession,
    pub(crate) state: &'a Qwen4MtpCacheState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Header {
    fingerprint: [u8; 32],
    saved_at: u64,
    max_ctx: u64,
    committed_tokens: u64,
    cached_count: u64,
    checkpoint_counts: [u64; crate::seam::TURN_CHECKPOINT_COUNT],
    record_count: u32,
    k_fmt: DType,
    v_fmt: DType,
    data_bytes: u64,
    has_checkpoints: [bool; crate::seam::TURN_CHECKPOINT_COUNT],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MtpHeader {
    fingerprint: [u8; 32],
    max_ctx: u64,
    cached_count: u64,
    h_width: u64,
    checkpoint_counts: [u64; crate::seam::TURN_CHECKPOINT_COUNT],
    pending_id: u32,
    kv_bytes_per_side: u64,
    data_bytes: u64,
    has_checkpoints: [bool; crate::seam::TURN_CHECKPOINT_COUNT],
    has_pending: bool,
    pending_emitted: bool,
}

impl SessionCache {
    pub(crate) fn open(
        cfg: &EngineConfig,
        gguf: &infr_gguf::Gguf,
        slot: &SessionStateMeta,
    ) -> Result<Option<Self>> {
        let Some(root) = cfg.kv.session_cache_dir.as_ref() else {
            return Ok(None);
        };
        if root.as_os_str().is_empty() {
            return Ok(None);
        }
        let max_bytes = match cfg.kv.session_cache_max {
            SizeSpec::Bytes(bytes) => bytes,
            SizeSpec::Percent(_) => {
                return Err(anyhow!(
                    "kv.session_cache_max does not accept percentages; use an absolute GiB/MiB size"
                ));
            }
        };
        if max_bytes == 0 {
            tracing::info!("cold KV session cache disabled by a zero size limit");
            return Ok(None);
        }

        fs::create_dir_all(root)
            .with_context(|| format!("create cold KV cache directory {}", root.display()))?;
        let ttl = (cfg.kv.session_cache_ttl_hours != 0)
            .then(|| Duration::from_secs(cfg.kv.session_cache_ttl_hours.saturating_mul(60 * 60)));
        gc_root(root, max_bytes, ttl)?;

        let fingerprint = model_fingerprint(gguf)?;
        let model_dir = root.join(hex_digest(&fingerprint));
        fs::create_dir_all(&model_dir).with_context(|| {
            format!(
                "create model cold KV cache directory {}",
                model_dir.display()
            )
        })?;

        let mut entries = Vec::new();
        for item in fs::read_dir(&model_dir)
            .with_context(|| format!("scan cold KV cache directory {}", model_dir.display()))?
        {
            let item = match item {
                Ok(item) => item,
                Err(error) => {
                    tracing::warn!("cold KV cache: skip unreadable directory entry: {error}");
                    continue;
                }
            };
            let Ok(file_type) = item.file_type() else {
                continue;
            };
            let path = item.path();
            if !file_type.is_file() || !is_cache_file(&path) {
                continue;
            }
            match read_catalog_entry(&path, fingerprint, slot.max_ctx, slot.k_fmt, slot.v_fmt) {
                Ok(Some(entry)) => entries.push(entry),
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(
                        path = %path.display(),
                        "cold KV cache: removing invalid catalog entry: {error}"
                    );
                    remove_cache_pair(&path);
                }
            }
        }
        let bytes = entries.iter().map(|entry| entry.file_bytes).sum::<u64>();
        tracing::info!(
            sessions = entries.len(),
            bytes,
            max_bytes,
            idle_secs = cfg.kv.session_idle_secs,
            directory = %model_dir.display(),
            "cold KV session cache ready"
        );
        Ok(Some(Self {
            root: root.clone(),
            model_dir,
            fingerprint,
            max_bytes,
            ttl,
            max_ctx: slot.max_ctx,
            k_fmt: slot.k_fmt,
            v_fmt: slot.v_fmt,
            entries,
        }))
    }

    pub(crate) fn best_continuation_len(&self, prompt: &[u32]) -> usize {
        self.entries
            .iter()
            .filter_map(|entry| entry.meta.continuation_prefix_len(prompt))
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn take_best_continuation(&mut self, prompt: &[u32]) -> Option<ColdEntry> {
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                entry
                    .meta
                    .continuation_prefix_len(prompt)
                    .map(|prefix| (index, prefix, entry.saved_at))
            })
            .max_by_key(|&(_, prefix, saved_at)| (prefix, saved_at))?
            .0;
        Some(self.entries.swap_remove(index))
    }

    pub(crate) fn return_entry(&mut self, entry: ColdEntry) {
        if entry.path.is_file() {
            self.entries.push(entry);
        }
    }

    pub(crate) fn spill(
        &mut self,
        kv: &mut SeamKv,
        backend: &dyn Backend,
        model_cfg: &Config,
        mtp: Option<MtpCacheSource<'_>>,
    ) -> Result<bool> {
        self.spill_with_gate(kv, backend, model_cfg, mtp, None)
    }

    // A private catalog lets the idle worker write without holding the shared restore index.
    pub(crate) fn spill_writer(&self) -> Self {
        Self {
            root: self.root.clone(),
            model_dir: self.model_dir.clone(),
            fingerprint: self.fingerprint,
            max_bytes: self.max_bytes,
            ttl: self.ttl,
            max_ctx: self.max_ctx,
            k_fmt: self.k_fmt,
            v_fmt: self.v_fmt,
            entries: Vec::new(),
        }
    }

    pub(crate) fn publish_spills(&mut self, writer: Self) {
        self.entries.retain(|entry| entry.path.is_file());
        for entry in writer.entries {
            self.return_entry(entry);
        }
    }

    pub(crate) fn spill_with_gate(
        &mut self,
        kv: &mut SeamKv,
        backend: &dyn Backend,
        model_cfg: &Config,
        mtp: Option<MtpCacheSource<'_>>,
        gate: Option<&StepGate>,
    ) -> Result<bool> {
        let meta = kv.session_state_meta();
        if meta.cached.is_empty() {
            return Ok(false);
        }
        if !kv.can_release_session_state() {
            return Err(anyhow!(
                "cold KV sessions require the dynamic segmented KV allocator"
            ));
        }
        if meta.max_ctx != self.max_ctx || meta.k_fmt != self.k_fmt || meta.v_fmt != self.v_fmt {
            return Err(anyhow!(
                "resident slot geometry changed after cache initialization"
            ));
        }

        let started = Instant::now();

        {
            let _gate = gate.map(StepGate::enter);
            backend
                .sync()
                .map_err(|error| anyhow!("sync before cold KV spill: {error}"))?;
        }
        let buffers = kv.session_state_buffers(backend)?;
        let data_bytes = buffers.iter().try_fold(0u64, |total, buffer| {
            total
                .checked_add(buffer.committed_bytes as u64)
                .ok_or_else(|| anyhow!("cold KV payload size overflow"))
        })?;
        let header = Header::new(self.fingerprint, &meta, buffers.len(), data_bytes)?;
        let main_file_bytes = checked_file_bytes(&header)?;
        let mtp_header = mtp
            .as_ref()
            .map(|source| {
                MtpHeader::new(
                    self.fingerprint,
                    self.max_ctx,
                    source.state,
                    source.head.kv_prefix_bytes(source.state.cached.len())?,
                )
            })
            .transpose()?;
        let mtp_file_bytes = mtp_header
            .as_ref()
            .map(checked_mtp_file_bytes)
            .transpose()?
            .unwrap_or(0);
        let file_bytes = main_file_bytes
            .checked_add(mtp_file_bytes)
            .ok_or_else(|| anyhow!("cold KV snapshot byte count overflow"))?;
        if file_bytes > self.max_bytes {
            return Err(anyhow!(
                "one cold KV session needs {:.2} GiB, exceeding kv.session_cache_max {:.2} GiB",
                file_bytes as f64 / (1u64 << 30) as f64,
                self.max_bytes as f64 / (1u64 << 30) as f64,
            ));
        }

        let paths = self.new_paths();
        if let (Some(source), Some(mtp_header)) = (mtp.as_ref(), mtp_header.as_ref()) {
            if let Err(error) = write_mtp_file(
                &paths.mtp_temporary,
                mtp_header,
                source.state,
                source.head,
                backend,
                gate,
            ) {
                let _ = fs::remove_file(&paths.mtp_temporary);
                return Err(error);
            }
        }
        let write_result = write_session_file(
            &paths.main_temporary,
            &header,
            &meta,
            &buffers,
            backend,
            gate,
        );
        drop(buffers);
        if let Err(error) = write_result {
            let _ = fs::remove_file(&paths.main_temporary);
            let _ = fs::remove_file(&paths.mtp_temporary);
            return Err(error);
        }
        if mtp_header.is_some() {
            if let Err(error) = fs::rename(&paths.mtp_temporary, &paths.mtp_final) {
                let _ = fs::remove_file(&paths.main_temporary);
                let _ = fs::remove_file(&paths.mtp_temporary);
                return Err(error).with_context(|| {
                    format!(
                        "publish cold MTP sidecar {} -> {}",
                        paths.mtp_temporary.display(),
                        paths.mtp_final.display()
                    )
                });
            }
        }
        if let Err(error) = fs::rename(&paths.main_temporary, &paths.main_final) {
            let _ = fs::remove_file(&paths.main_temporary);
            let _ = fs::remove_file(&paths.mtp_final);
            return Err(error).with_context(|| {
                format!(
                    "publish cold KV session {} -> {}",
                    paths.main_temporary.display(),
                    paths.main_final.display()
                )
            });
        }
        let _gate = gate.map(StepGate::enter);
        if let Err(error) = kv.release_session_state(backend, model_cfg) {
            let _ = fs::remove_file(&paths.main_final);
            let _ = fs::remove_file(&paths.mtp_final);
            return Err(error.context("release resident KV after durable spill"));
        }
        if let Some(source) = mtp.as_ref() {
            if let Err(error) = source.head.release_kv(backend) {
                tracing::warn!(
                    "cold KV cache: main state was released but MTP head KV release failed: {error}"
                );
            }
        }
        self.entries.push(ColdEntry {
            path: paths.main_final,
            saved_at: header.saved_at,
            meta,
            file_bytes,
        });
        tracing::info!(
            tokens = header.cached_count,
            bytes = file_bytes,
            gib = file_bytes as f64 / (1u64 << 30) as f64,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "spilled conversation KV to cold storage"
        );
        Ok(true)
    }

    pub(crate) fn restore(
        &mut self,
        mut entry: ColdEntry,
        kv: &mut SeamKv,
        backend: &dyn Backend,
        model_cfg: &Config,
        mtp_head: Option<&Qwen4MtpSession>,
    ) -> Result<Option<Qwen4MtpCacheState>> {
        let started = Instant::now();
        let tokens = entry.meta.cached.len();
        let file_bytes = entry.file_bytes;
        let path = entry.path.clone();
        let result = restore_session_file(
            &path,
            self.fingerprint,
            self.max_ctx,
            self.k_fmt,
            self.v_fmt,
            kv,
            backend,
            model_cfg,
        );
        if result.is_err() {
            // A restore publishes metadata only after checksum validation. Releasing here removes
            // any segments allocated for an incomplete/corrupt file before normal prefill resumes.
            if let Err(error) = kv.release_session_state(backend, model_cfg) {
                tracing::warn!("cold KV cache: cleanup after failed restore also failed: {error}");
                kv.reset();
            }
        }
        if result.is_ok() {
            let mtp_path = mtp_path_for(&path);
            let mtp_state = if let Some(head) = mtp_head.filter(|_| mtp_path.is_file()) {
                match restore_mtp_file(
                    &mtp_path,
                    self.fingerprint,
                    self.max_ctx,
                    &entry.meta.cached,
                    head,
                    backend,
                ) {
                    Ok(state) => Some(state),
                    Err(error) => {
                        tracing::warn!(
                            path = %mtp_path.display(),
                            "cold KV cache: MTP sidecar restore failed; preserving the restored main KV and using ordinary decode: {error}"
                        );
                        let sidecar_bytes = fs::metadata(&mtp_path).map_or(0, |meta| meta.len());
                        entry.file_bytes = entry.file_bytes.saturating_sub(sidecar_bytes);
                        let _ = fs::remove_file(&mtp_path);
                        None
                    }
                }
            } else {
                None
            };
            // Cold entries are immutable snapshots. Keep a validated entry available so several
            // conversations can branch from the same system/tool checkpoint instead of consuming
            // it on the first restore. Size and age GC remain the eviction authority.
            entry.saved_at = unix_secs();
            if let Ok(file) = OpenOptions::new().write(true).open(&path) {
                let _ = file.set_times(FileTimes::new().set_modified(SystemTime::now()));
            }
            if let Ok(file) = OpenOptions::new().write(true).open(&mtp_path) {
                let _ = file.set_times(FileTimes::new().set_modified(SystemTime::now()));
            }
            self.return_entry(entry);
            tracing::info!(
                tokens,
                bytes = file_bytes,
                gib = file_bytes as f64 / (1u64 << 30) as f64,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "restored reusable conversation KV from cold storage"
            );
            return Ok(mtp_state);
        } else {
            remove_cache_pair(&path);
        }
        result.map(|()| None)
    }

    pub(crate) fn gc(&mut self) -> Result<()> {
        gc_root(&self.root, self.max_bytes, self.ttl)?;
        self.entries.retain(|entry| entry.path.is_file());
        Ok(())
    }

    fn new_paths(&self) -> SessionPaths {
        let now = unix_secs();
        let sequence = FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let stem = format!(
            "session-{now:016x}-{:08x}-{sequence:016x}",
            std::process::id()
        );
        SessionPaths {
            main_temporary: self.model_dir.join(format!(".{stem}.kv.tmp")),
            mtp_temporary: self.model_dir.join(format!(".{stem}.mtp.tmp")),
            main_final: self.model_dir.join(format!("{stem}.infrkv")),
            mtp_final: self.model_dir.join(format!("{stem}.infrmtp")),
        }
    }
}

struct SessionPaths {
    main_temporary: PathBuf,
    mtp_temporary: PathBuf,
    main_final: PathBuf,
    mtp_final: PathBuf,
}

impl Header {
    fn new(
        fingerprint: [u8; 32],
        meta: &SessionStateMeta,
        record_count: usize,
        data_bytes: u64,
    ) -> Result<Self> {
        Ok(Self {
            fingerprint,
            saved_at: unix_secs(),
            max_ctx: meta.max_ctx.try_into().context("max context exceeds u64")?,
            committed_tokens: meta
                .committed_tokens
                .try_into()
                .context("committed token count exceeds u64")?,
            cached_count: meta
                .cached
                .len()
                .try_into()
                .context("cached token count exceeds u64")?,
            checkpoint_counts: [
                meta.checkpoint_tokens[0].as_ref().map_or(Ok(0), |tokens| {
                    tokens
                        .len()
                        .try_into()
                        .context("agent checkpoint token count exceeds u64")
                })?,
                meta.checkpoint_tokens[1].as_ref().map_or(Ok(0), |tokens| {
                    tokens
                        .len()
                        .try_into()
                        .context("edit checkpoint token count exceeds u64")
                })?,
            ],
            record_count: record_count
                .try_into()
                .context("cold KV record count exceeds u32")?,
            k_fmt: meta.k_fmt,
            v_fmt: meta.v_fmt,
            data_bytes,
            has_checkpoints: meta
                .checkpoint_tokens
                .each_ref()
                .map(|tokens| tokens.is_some()),
        })
    }

    fn meta(
        &self,
        cached: Vec<u32>,
        checkpoints: [Option<Vec<u32>>; crate::seam::TURN_CHECKPOINT_COUNT],
    ) -> Result<SessionStateMeta> {
        Ok(SessionStateMeta {
            max_ctx: self
                .max_ctx
                .try_into()
                .context("max context exceeds usize")?,
            k_fmt: self.k_fmt,
            v_fmt: self.v_fmt,
            committed_tokens: self
                .committed_tokens
                .try_into()
                .context("committed token count exceeds usize")?,
            cached,
            checkpoint_tokens: checkpoints,
        })
    }
}

impl MtpHeader {
    fn new(
        fingerprint: [u8; 32],
        max_ctx: usize,
        state: &Qwen4MtpCacheState,
        kv_bytes_per_side: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            !state.cached.is_empty() && state.cached.len() <= max_ctx,
            "cold MTP state has an invalid cached-token depth"
        );
        let h_width = state.last_h.len();
        anyhow::ensure!(h_width != 0, "cold MTP state has an empty hidden frontier");
        for checkpoint in state.turn_checkpoints.iter().flatten() {
            anyhow::ensure!(
                !checkpoint.tokens.is_empty()
                    && state.cached.starts_with(&checkpoint.tokens)
                    && checkpoint.last_h.len() == h_width,
                "cold MTP checkpoint is not a complete prefix of its live state"
            );
        }
        anyhow::ensure!(
            !state.pending_emitted || state.pending_id.is_some(),
            "cold MTP state marks a missing frontier as emitted"
        );
        let checkpoint_counts = state.turn_checkpoints.each_ref().map(|checkpoint| {
            checkpoint
                .as_ref()
                .map_or(Ok(0), |checkpoint| checkpoint.tokens.len().try_into())
                .context("cold MTP checkpoint token count exceeds u64")
        });
        let [agent_checkpoint_count, edit_checkpoint_count] = checkpoint_counts;
        let checkpoint_counts = [agent_checkpoint_count?, edit_checkpoint_count?];
        let cached_count: u64 = state
            .cached
            .len()
            .try_into()
            .context("cold MTP token count exceeds u64")?;
        let h_width: u64 = h_width
            .try_into()
            .context("cold MTP hidden width exceeds u64")?;
        let kv_bytes_per_side: u64 = kv_bytes_per_side
            .try_into()
            .context("cold MTP KV byte count exceeds u64")?;
        let checkpoint_tokens = checkpoint_counts
            .iter()
            .try_fold(0u64, |total, &count| total.checked_add(count))
            .ok_or_else(|| anyhow!("cold MTP checkpoint metadata size overflow"))?;
        let hidden_rows = 1u64
            .checked_add(state.turn_checkpoints.iter().flatten().count() as u64)
            .ok_or_else(|| anyhow!("cold MTP hidden row count overflow"))?;
        let data_bytes = cached_count
            .checked_add(checkpoint_tokens)
            .and_then(|tokens| tokens.checked_mul(4))
            .and_then(|bytes| {
                h_width
                    .checked_mul(hidden_rows)
                    .and_then(|values| values.checked_mul(4))
                    .and_then(|hidden| bytes.checked_add(hidden))
            })
            .and_then(|bytes| {
                kv_bytes_per_side
                    .checked_mul(2)
                    .and_then(|kv| bytes.checked_add(kv))
            })
            .ok_or_else(|| anyhow!("cold MTP payload size overflow"))?;
        Ok(Self {
            fingerprint,
            max_ctx: max_ctx.try_into().context("max context exceeds u64")?,
            cached_count,
            h_width,
            checkpoint_counts,
            pending_id: state.pending_id.unwrap_or(0),
            kv_bytes_per_side,
            data_bytes,
            has_checkpoints: state
                .turn_checkpoints
                .each_ref()
                .map(|checkpoint| checkpoint.is_some()),
            has_pending: state.pending_id.is_some(),
            pending_emitted: state.pending_emitted,
        })
    }
}

fn checked_mtp_file_bytes(header: &MtpHeader) -> Result<u64> {
    MTP_HEADER_BYTES
        .checked_add(header.data_bytes)
        .and_then(|bytes| bytes.checked_add(CHECKSUM_BYTES))
        .ok_or_else(|| anyhow!("cold MTP file size overflow"))
}

fn encode_mtp_header(header: &MtpHeader) -> [u8; MTP_HEADER_BYTES as usize] {
    let mut out = Vec::with_capacity(MTP_HEADER_BYTES as usize);
    out.extend_from_slice(&MTP_MAGIC);
    out.extend_from_slice(&MTP_VERSION.to_le_bytes());
    let checkpoint_flags = header
        .has_checkpoints
        .iter()
        .enumerate()
        .fold(0u32, |flags, (index, &present)| {
            flags | (u32::from(present) << index)
        });
    let flags = checkpoint_flags
        | (u32::from(header.has_pending) * MTP_FLAG_PENDING)
        | (u32::from(header.pending_emitted) * MTP_FLAG_PENDING_EMITTED);
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&header.fingerprint);
    out.extend_from_slice(&header.max_ctx.to_le_bytes());
    out.extend_from_slice(&header.cached_count.to_le_bytes());
    out.extend_from_slice(&header.h_width.to_le_bytes());
    for count in header.checkpoint_counts {
        out.extend_from_slice(&count.to_le_bytes());
    }
    out.extend_from_slice(&header.pending_id.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&header.kv_bytes_per_side.to_le_bytes());
    out.extend_from_slice(&header.data_bytes.to_le_bytes());
    out.try_into().expect("fixed cold MTP header size")
}

fn decode_mtp_header(bytes: &[u8; MTP_HEADER_BYTES as usize]) -> Result<MtpHeader> {
    let mut cursor = 0usize;
    if take::<8>(bytes, &mut cursor)? != MTP_MAGIC {
        return Err(anyhow!("not an infr cold MTP file"));
    }
    let version = u32::from_le_bytes(take(bytes, &mut cursor)?);
    if version != MTP_VERSION {
        return Err(anyhow!("unsupported cold MTP format version {version}"));
    }
    let flags = u32::from_le_bytes(take(bytes, &mut cursor)?);
    if flags & !MTP_FLAGS != 0 {
        return Err(anyhow!("cold MTP header has unknown flags"));
    }
    let fingerprint = take(bytes, &mut cursor)?;
    let max_ctx = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let cached_count = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let h_width = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let checkpoint_counts = [
        u64::from_le_bytes(take(bytes, &mut cursor)?),
        u64::from_le_bytes(take(bytes, &mut cursor)?),
    ];
    let pending_id = u32::from_le_bytes(take(bytes, &mut cursor)?);
    let reserved = u32::from_le_bytes(take(bytes, &mut cursor)?);
    if reserved != 0 {
        return Err(anyhow!("cold MTP header has non-zero reserved bytes"));
    }
    let kv_bytes_per_side = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let data_bytes = u64::from_le_bytes(take(bytes, &mut cursor)?);
    debug_assert_eq!(cursor, MTP_HEADER_BYTES as usize);
    Ok(MtpHeader {
        fingerprint,
        max_ctx,
        cached_count,
        h_width,
        checkpoint_counts,
        pending_id,
        kv_bytes_per_side,
        data_bytes,
        has_checkpoints: [
            flags & MTP_FLAG_AGENT_CHECKPOINT != 0,
            flags & MTP_FLAG_EDIT_CHECKPOINT != 0,
        ],
        has_pending: flags & MTP_FLAG_PENDING != 0,
        pending_emitted: flags & MTP_FLAG_PENDING_EMITTED != 0,
    })
}

fn validate_mtp_header(header: &MtpHeader, file_bytes: u64) -> Result<()> {
    let invalid_checkpoint = header
        .checkpoint_counts
        .iter()
        .zip(header.has_checkpoints)
        .any(|(&count, present)| count > header.cached_count || present != (count != 0));
    if header.cached_count == 0
        || header.cached_count > header.max_ctx
        || header.h_width == 0
        || header.h_width > 1_048_576
        || invalid_checkpoint
        || (header.pending_emitted && !header.has_pending)
    {
        return Err(anyhow!("cold MTP header has inconsistent state geometry"));
    }
    let checkpoint_tokens = header
        .checkpoint_counts
        .iter()
        .try_fold(0u64, |total, &count| total.checked_add(count))
        .ok_or_else(|| anyhow!("cold MTP checkpoint metadata size overflow"))?;
    let hidden_rows = 1u64
        + header
            .has_checkpoints
            .iter()
            .filter(|&&present| present)
            .count() as u64;
    let expected_data_bytes = header
        .cached_count
        .checked_add(checkpoint_tokens)
        .and_then(|tokens| tokens.checked_mul(4))
        .and_then(|bytes| {
            header
                .h_width
                .checked_mul(hidden_rows)
                .and_then(|values| values.checked_mul(4))
                .and_then(|hidden| bytes.checked_add(hidden))
        })
        .and_then(|bytes| {
            header
                .kv_bytes_per_side
                .checked_mul(2)
                .and_then(|kv| bytes.checked_add(kv))
        })
        .ok_or_else(|| anyhow!("cold MTP payload size overflow"))?;
    if expected_data_bytes != header.data_bytes {
        return Err(anyhow!("cold MTP header has an inconsistent payload size"));
    }
    if checked_mtp_file_bytes(header)? != file_bytes {
        return Err(anyhow!(
            "cold MTP file size is {file_bytes}, expected {} from its header",
            checked_mtp_file_bytes(header)?
        ));
    }
    Ok(())
}

fn write_session_file(
    path: &Path,
    header: &Header,
    meta: &SessionStateMeta,
    buffers: &[crate::seam::SessionBuffer<'_>],
    backend: &dyn Backend,
    gate: Option<&StepGate>,
) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create cold KV temporary file {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    let mut hasher = Sha256::new();
    write_hashed(&mut writer, &mut hasher, &encode_header(header))?;
    write_tokens(&mut writer, &mut hasher, &meta.cached)?;
    for tokens in meta.checkpoint_tokens.iter().flatten() {
        write_tokens(&mut writer, &mut hasher, tokens)?;
    }
    let mut scratch = vec![0u8; STREAM_BYTES];
    for buffer in buffers {
        let record = encode_record(buffer.key, buffer.committed_bytes as u64);
        write_hashed(&mut writer, &mut hasher, &record)?;
        write_device_payload(
            &mut writer,
            &mut hasher,
            buffer.buffer,
            buffer.committed_bytes,
            backend,
            gate,
            &mut scratch,
        )
        .with_context(|| format!("spill cold KV {:?}", buffer.key))?;
    }
    let checksum = hasher.finalize();
    writer.write_all(checksum.as_ref())?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn write_device_payload(
    writer: &mut impl Write,
    hasher: &mut Sha256,
    buffer: &dyn infr_core::backend::Buffer,
    bytes: usize,
    backend: &dyn Backend,
    gate: Option<&StepGate>,
    scratch: &mut [u8],
) -> Result<()> {
    anyhow::ensure!(!scratch.is_empty(), "cold KV transfer scratch is empty");
    let mut offset = 0;
    while offset < bytes {
        let count = (bytes - offset).min(scratch.len());
        {
            let _gate = gate.map(StepGate::enter);
            backend
                .download_range(buffer, offset, &mut scratch[..count])
                .map_err(|error| anyhow!("download cold KV payload at {offset}: {error}"))?;
        }
        write_hashed(writer, hasher, &scratch[..count])?;
        offset += count;
    }
    Ok(())
}

fn write_mtp_file(
    path: &Path,
    header: &MtpHeader,
    state: &Qwen4MtpCacheState,
    head: &Qwen4MtpSession,
    backend: &dyn Backend,
    gate: Option<&StepGate>,
) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create cold MTP temporary file {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    let mut hasher = Sha256::new();
    write_hashed(&mut writer, &mut hasher, &encode_mtp_header(header))?;
    write_tokens(&mut writer, &mut hasher, &state.cached)?;
    for checkpoint in state.turn_checkpoints.iter().flatten() {
        write_tokens(&mut writer, &mut hasher, &checkpoint.tokens)?;
    }
    write_f32s(&mut writer, &mut hasher, &state.last_h)?;
    for checkpoint in state.turn_checkpoints.iter().flatten() {
        write_f32s(&mut writer, &mut hasher, &checkpoint.last_h)?;
    }
    let kv = {
        let _gate = gate.map(StepGate::enter);
        head.kv_prefix(backend, state.cached.len())?
    };
    anyhow::ensure!(
        kv.bytes_per_side() as u64 == header.kv_bytes_per_side,
        "cold MTP KV byte count changed while spilling"
    );
    let mut scratch = vec![0u8; STREAM_BYTES];
    for (name, buffer) in [("K", kv.k()), ("V", kv.v())] {
        write_device_payload(
            &mut writer,
            &mut hasher,
            buffer,
            kv.bytes_per_side(),
            backend,
            gate,
            &mut scratch,
        )
        .with_context(|| format!("spill cold MTP {name}"))?;
    }
    let checksum = hasher.finalize();
    writer.write_all(checksum.as_ref())?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn restore_mtp_file(
    path: &Path,
    fingerprint: [u8; 32],
    max_ctx: usize,
    main_cached: &[u32],
    head: &Qwen4MtpSession,
    backend: &dyn Backend,
) -> Result<Qwen4MtpCacheState> {
    let file =
        File::open(path).with_context(|| format!("open cold MTP state {}", path.display()))?;
    let file_bytes = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut header_bytes = [0u8; MTP_HEADER_BYTES as usize];
    read_hashed(&mut reader, &mut hasher, &mut header_bytes)?;
    let header = decode_mtp_header(&header_bytes)?;
    validate_mtp_header(&header, file_bytes)?;
    if header.fingerprint != fingerprint
        || header.max_ctx != max_ctx as u64
        || header.h_width != head.h_width() as u64
    {
        return Err(anyhow!(
            "cold MTP file belongs to a different model or slot geometry"
        ));
    }
    let cached = read_tokens_hashed(&mut reader, &mut hasher, header.cached_count, max_ctx)?;
    if cached != main_cached {
        return Err(anyhow!(
            "cold MTP token prefix differs from the restored main KV"
        ));
    }
    let mut turn_checkpoints = std::array::from_fn(|_| None);
    for (index, checkpoint) in turn_checkpoints.iter_mut().enumerate() {
        if header.has_checkpoints[index] {
            let tokens = read_tokens_hashed(
                &mut reader,
                &mut hasher,
                header.checkpoint_counts[index],
                max_ctx,
            )?;
            if tokens.is_empty() || !cached.starts_with(&tokens) {
                return Err(anyhow!(
                    "cold MTP checkpoint {index} is not a live-prefix checkpoint"
                ));
            }
            *checkpoint = Some(Qwen4MtpCheckpointState {
                tokens,
                last_h: Vec::new(),
            });
        }
    }
    let h_width: usize = header
        .h_width
        .try_into()
        .context("cold MTP hidden width exceeds usize")?;
    let last_h = read_f32s_hashed(&mut reader, &mut hasher, h_width)?;
    for checkpoint in turn_checkpoints.iter_mut().flatten() {
        checkpoint.last_h = read_f32s_hashed(&mut reader, &mut hasher, h_width)?;
    }
    let kv = head.kv_prefix(backend, cached.len())?;
    anyhow::ensure!(
        kv.bytes_per_side() as u64 == header.kv_bytes_per_side,
        "cold MTP KV geometry differs from its target slot"
    );
    let mut scratch = vec![0u8; STREAM_BYTES];
    for (name, buffer) in [("K", kv.k()), ("V", kv.v())] {
        let mut offset = 0usize;
        while offset < kv.bytes_per_side() {
            let count = (kv.bytes_per_side() - offset).min(scratch.len());
            read_hashed(&mut reader, &mut hasher, &mut scratch[..count])?;
            backend
                .upload_range(buffer, offset, &scratch[..count])
                .map_err(|error| anyhow!("upload cold MTP {name} at {offset}: {error}"))?;
            offset += count;
        }
    }
    let mut stored_checksum = [0u8; CHECKSUM_BYTES as usize];
    reader.read_exact(&mut stored_checksum)?;
    let calculated = hasher.finalize();
    if calculated.as_slice() != stored_checksum {
        return Err(anyhow!("cold MTP checksum mismatch"));
    }
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(anyhow!("cold MTP file has trailing data"));
    }
    backend
        .sync()
        .map_err(|error| anyhow!("sync restored cold MTP state: {error}"))?;
    Ok(Qwen4MtpCacheState {
        cached,
        last_h,
        turn_checkpoints,
        pending_id: header.has_pending.then_some(header.pending_id),
        pending_emitted: header.pending_emitted,
    })
}

#[allow(clippy::too_many_arguments)]
fn restore_session_file(
    path: &Path,
    fingerprint: [u8; 32],
    max_ctx: usize,
    k_fmt: DType,
    v_fmt: DType,
    kv: &mut SeamKv,
    backend: &dyn Backend,
    model_cfg: &Config,
) -> Result<()> {
    let file =
        File::open(path).with_context(|| format!("open cold KV session {}", path.display()))?;
    let file_bytes = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut header_bytes = [0u8; HEADER_BYTES as usize];
    read_hashed(&mut reader, &mut hasher, &mut header_bytes)?;
    let header = decode_header(&header_bytes)?;
    validate_header(&header, file_bytes)?;
    if header.fingerprint != fingerprint
        || header.max_ctx != max_ctx as u64
        || header.k_fmt != k_fmt
        || header.v_fmt != v_fmt
    {
        return Err(anyhow!(
            "cold KV file belongs to a different model or slot geometry"
        ));
    }
    let cached = read_tokens_hashed(&mut reader, &mut hasher, header.cached_count, max_ctx)?;
    let mut checkpoints = std::array::from_fn(|_| None);
    for (index, checkpoint) in checkpoints.iter_mut().enumerate() {
        if header.has_checkpoints[index] {
            *checkpoint = Some(read_tokens_hashed(
                &mut reader,
                &mut hasher,
                header.checkpoint_counts[index],
                max_ctx,
            )?);
        }
    }
    let meta = header.meta(cached, checkpoints)?;
    if meta.cached.len() > meta.committed_tokens {
        return Err(anyhow!(
            "cold KV token depth exceeds its committed physical depth"
        ));
    }
    kv.prepare_session_restore(backend, model_cfg, &meta)?;

    let mut expected = kv
        .session_restore_buffer_specs(backend)?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    if expected.len() != header.record_count as usize {
        return Err(anyhow!(
            "cold KV record count {} does not match the slot's {} buffers",
            header.record_count,
            expected.len()
        ));
    }
    let mut data_bytes = 0u64;
    let mut scratch = vec![0u8; STREAM_BYTES];
    for _ in 0..header.record_count {
        let mut record_bytes = [0u8; RECORD_HEADER_BYTES as usize];
        read_hashed(&mut reader, &mut hasher, &mut record_bytes)?;
        let (key, len) = decode_record(&record_bytes)?;
        let expected_len = expected
            .remove(&key)
            .ok_or_else(|| anyhow!("cold KV contains duplicate or unexpected buffer {key:?}"))?;
        if len != expected_len as u64 {
            return Err(anyhow!(
                "cold KV buffer {key:?} has {len} bytes; the slot expects {expected_len}"
            ));
        }
        let target = kv
            .session_state_buffer(key)
            .ok_or_else(|| anyhow!("cold KV target buffer {key:?} is absent"))?;
        let mut offset = 0usize;
        while offset < expected_len {
            let count = (expected_len - offset).min(scratch.len());
            read_hashed(&mut reader, &mut hasher, &mut scratch[..count])?;
            backend
                .upload_range(target, offset, &scratch[..count])
                .map_err(|error| anyhow!("upload cold KV {key:?} at {offset}: {error}"))?;
            offset += count;
        }
        data_bytes = data_bytes
            .checked_add(len)
            .ok_or_else(|| anyhow!("cold KV restored byte count overflow"))?;
    }
    if !expected.is_empty() || data_bytes != header.data_bytes {
        return Err(anyhow!("cold KV record set is incomplete"));
    }
    let mut stored_checksum = [0u8; CHECKSUM_BYTES as usize];
    reader.read_exact(&mut stored_checksum)?;
    let calculated = hasher.finalize();
    if calculated.as_slice() != stored_checksum {
        return Err(anyhow!("cold KV checksum mismatch"));
    }
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(anyhow!("cold KV file has trailing data"));
    }
    backend
        .sync()
        .map_err(|error| anyhow!("sync restored cold KV state: {error}"))?;
    kv.finish_session_restore(meta)
}

fn read_catalog_entry(
    path: &Path,
    fingerprint: [u8; 32],
    max_ctx: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> Result<Option<ColdEntry>> {
    let file = File::open(path)?;
    let main_file_bytes = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut bytes = [0u8; HEADER_BYTES as usize];
    reader.read_exact(&mut bytes)?;
    let header = decode_header(&bytes)?;
    validate_header(&header, main_file_bytes)?;
    if header.fingerprint != fingerprint
        || header.max_ctx != max_ctx as u64
        || header.k_fmt != k_fmt
        || header.v_fmt != v_fmt
    {
        return Ok(None);
    }
    let cached = read_tokens(&mut reader, header.cached_count, max_ctx)?;
    let mut checkpoints = std::array::from_fn(|_| None);
    for (index, checkpoint) in checkpoints.iter_mut().enumerate() {
        if header.has_checkpoints[index] {
            *checkpoint = Some(read_tokens(
                &mut reader,
                header.checkpoint_counts[index],
                max_ctx,
            )?);
        }
    }
    Ok(Some(ColdEntry {
        path: path.to_path_buf(),
        saved_at: header.saved_at,
        meta: header.meta(cached, checkpoints)?,
        file_bytes: main_file_bytes
            .saturating_add(fs::metadata(mtp_path_for(path)).map_or(0, |metadata| metadata.len())),
    }))
}

fn validate_header(header: &Header, file_bytes: u64) -> Result<()> {
    if header.record_count > MAX_RECORDS {
        return Err(anyhow!("cold KV record count is implausibly large"));
    }
    let invalid_checkpoint = header
        .checkpoint_counts
        .iter()
        .zip(header.has_checkpoints)
        .any(|(&count, present)| count > header.max_ctx || present != (count != 0));
    if header.cached_count > header.max_ctx
        || header.committed_tokens > header.max_ctx
        || invalid_checkpoint
    {
        return Err(anyhow!("cold KV header has inconsistent token counts"));
    }
    let expected = checked_file_bytes(header)?;
    if expected != file_bytes {
        return Err(anyhow!(
            "cold KV file size is {file_bytes}, expected {expected} from its header"
        ));
    }
    Ok(())
}

fn checked_file_bytes(header: &Header) -> Result<u64> {
    let checkpoint_tokens = header
        .checkpoint_counts
        .iter()
        .try_fold(0u64, |total, &count| total.checked_add(count))
        .ok_or_else(|| anyhow!("cold KV checkpoint metadata size overflow"))?;
    let token_bytes = header
        .cached_count
        .checked_add(checkpoint_tokens)
        .and_then(|tokens| tokens.checked_mul(4))
        .ok_or_else(|| anyhow!("cold KV token metadata size overflow"))?;
    HEADER_BYTES
        .checked_add(token_bytes)
        .and_then(|bytes| bytes.checked_add(header.record_count as u64 * RECORD_HEADER_BYTES))
        .and_then(|bytes| bytes.checked_add(header.data_bytes))
        .and_then(|bytes| bytes.checked_add(CHECKSUM_BYTES))
        .ok_or_else(|| anyhow!("cold KV file size overflow"))
}

fn encode_header(header: &Header) -> [u8; HEADER_BYTES as usize] {
    let mut out = Vec::with_capacity(HEADER_BYTES as usize);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    let flags = header
        .has_checkpoints
        .iter()
        .enumerate()
        .fold(0u32, |flags, (index, &present)| {
            flags | (u32::from(present) << index)
        });
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&header.fingerprint);
    out.extend_from_slice(&header.saved_at.to_le_bytes());
    out.extend_from_slice(&header.max_ctx.to_le_bytes());
    out.extend_from_slice(&header.committed_tokens.to_le_bytes());
    out.extend_from_slice(&header.cached_count.to_le_bytes());
    for count in header.checkpoint_counts {
        out.extend_from_slice(&count.to_le_bytes());
    }
    out.extend_from_slice(&header.record_count.to_le_bytes());
    out.extend_from_slice(&encode_dtype(header.k_fmt).to_le_bytes());
    out.extend_from_slice(&encode_dtype(header.v_fmt).to_le_bytes());
    out.extend_from_slice(&header.data_bytes.to_le_bytes());
    out.try_into().expect("fixed cold KV header size")
}

fn decode_header(bytes: &[u8; HEADER_BYTES as usize]) -> Result<Header> {
    let mut cursor = 0usize;
    let magic = take::<8>(bytes, &mut cursor)?;
    if magic != MAGIC {
        return Err(anyhow!("not an infr cold KV file"));
    }
    let version = u32::from_le_bytes(take(bytes, &mut cursor)?);
    if version != VERSION {
        return Err(anyhow!("unsupported cold KV format version {version}"));
    }
    let flags = u32::from_le_bytes(take(bytes, &mut cursor)?);
    if flags & !CHECKPOINT_FLAGS != 0 {
        return Err(anyhow!("cold KV header has unknown flags"));
    }
    let fingerprint = take(bytes, &mut cursor)?;
    let saved_at = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let max_ctx = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let committed_tokens = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let cached_count = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let checkpoint_counts = [
        u64::from_le_bytes(take(bytes, &mut cursor)?),
        u64::from_le_bytes(take(bytes, &mut cursor)?),
    ];
    let record_count = u32::from_le_bytes(take(bytes, &mut cursor)?);
    let k_fmt = decode_dtype(u16::from_le_bytes(take(bytes, &mut cursor)?))?;
    let v_fmt = decode_dtype(u16::from_le_bytes(take(bytes, &mut cursor)?))?;
    let data_bytes = u64::from_le_bytes(take(bytes, &mut cursor)?);
    debug_assert_eq!(cursor, HEADER_BYTES as usize);
    Ok(Header {
        fingerprint,
        saved_at,
        max_ctx,
        committed_tokens,
        cached_count,
        checkpoint_counts,
        record_count,
        k_fmt,
        v_fmt,
        data_bytes,
        has_checkpoints: [
            flags & FLAG_AGENT_CHECKPOINT != 0,
            flags & FLAG_EDIT_CHECKPOINT != 0,
        ],
    })
}

fn encode_record(key: SessionBufferKey, len: u64) -> [u8; RECORD_HEADER_BYTES as usize] {
    let (kind, layer) = match key {
        SessionBufferKey::K(layer) => (0, layer),
        SessionBufferKey::V(layer) => (1, layer),
        SessionBufferKey::QsaRaw(layer) => (2, layer),
        SessionBufferKey::QsaBlock(layer) => (3, layer),
        SessionBufferKey::PleState => (4, u32::MAX),
        SessionBufferKey::CheckpointK(checkpoint, layer) => (5 + checkpoint * 3, layer),
        SessionBufferKey::CheckpointV(checkpoint, layer) => (6 + checkpoint * 3, layer),
        SessionBufferKey::CheckpointPle(checkpoint) => (7 + checkpoint * 3, u32::MAX),
    };
    let mut out = [0u8; RECORD_HEADER_BYTES as usize];
    out[0] = kind;
    out[4..8].copy_from_slice(&layer.to_le_bytes());
    out[8..16].copy_from_slice(&len.to_le_bytes());
    out
}

fn decode_record(bytes: &[u8; RECORD_HEADER_BYTES as usize]) -> Result<(SessionBufferKey, u64)> {
    if bytes[1..4] != [0; 3] {
        return Err(anyhow!("cold KV record has non-zero reserved bytes"));
    }
    let layer = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let key = match bytes[0] {
        0 => SessionBufferKey::K(layer),
        1 => SessionBufferKey::V(layer),
        2 => SessionBufferKey::QsaRaw(layer),
        3 => SessionBufferKey::QsaBlock(layer),
        4 if layer == u32::MAX => SessionBufferKey::PleState,
        5 => SessionBufferKey::CheckpointK(0, layer),
        6 => SessionBufferKey::CheckpointV(0, layer),
        7 if layer == u32::MAX => SessionBufferKey::CheckpointPle(0),
        8 => SessionBufferKey::CheckpointK(1, layer),
        9 => SessionBufferKey::CheckpointV(1, layer),
        10 if layer == u32::MAX => SessionBufferKey::CheckpointPle(1),
        kind => return Err(anyhow!("unknown cold KV record kind {kind}")),
    };
    let len = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    Ok((key, len))
}

fn write_tokens(writer: &mut impl Write, hasher: &mut Sha256, tokens: &[u32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(tokens.len().min(8192) * 4);
    for chunk in tokens.chunks(8192) {
        bytes.clear();
        for &token in chunk {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        write_hashed(writer, hasher, &bytes)?;
    }
    Ok(())
}

fn write_f32s(writer: &mut impl Write, hasher: &mut Sha256, values: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len().min(8192) * 4);
    for chunk in values.chunks(8192) {
        bytes.clear();
        for &value in chunk {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        write_hashed(writer, hasher, &bytes)?;
    }
    Ok(())
}

fn read_tokens(reader: &mut impl Read, count: u64, max_ctx: usize) -> Result<Vec<u32>> {
    let count: usize = count.try_into().context("token count exceeds usize")?;
    if count > max_ctx {
        return Err(anyhow!("cold KV token count exceeds context capacity"));
    }
    let byte_count = count
        .checked_mul(4)
        .ok_or_else(|| anyhow!("cold KV token byte count overflow"))?;
    let mut bytes = vec![0u8; byte_count];
    reader.read_exact(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect())
}

fn read_tokens_hashed(
    reader: &mut impl Read,
    hasher: &mut Sha256,
    count: u64,
    max_ctx: usize,
) -> Result<Vec<u32>> {
    let count: usize = count.try_into().context("token count exceeds usize")?;
    if count > max_ctx {
        return Err(anyhow!("cold KV token count exceeds context capacity"));
    }
    let byte_count = count
        .checked_mul(4)
        .ok_or_else(|| anyhow!("cold KV token byte count overflow"))?;
    let mut bytes = vec![0u8; byte_count];
    read_hashed(reader, hasher, &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect())
}

fn read_f32s_hashed(reader: &mut impl Read, hasher: &mut Sha256, count: usize) -> Result<Vec<f32>> {
    let byte_count = count
        .checked_mul(4)
        .ok_or_else(|| anyhow!("cold MTP hidden byte count overflow"))?;
    let mut bytes = vec![0u8; byte_count];
    read_hashed(reader, hasher, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_bits(u32::from_le_bytes(bytes.try_into().unwrap())))
        .collect())
}

fn write_hashed(writer: &mut impl Write, hasher: &mut Sha256, bytes: &[u8]) -> Result<()> {
    writer.write_all(bytes)?;
    hasher.update(bytes);
    Ok(())
}

fn read_hashed(reader: &mut impl Read, hasher: &mut Sha256, bytes: &mut [u8]) -> Result<()> {
    reader.read_exact(bytes)?;
    hasher.update(bytes);
    Ok(())
}

fn take<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Result<[u8; N]> {
    let end = cursor
        .checked_add(N)
        .ok_or_else(|| anyhow!("cold KV header cursor overflow"))?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| anyhow!("truncated cold KV header"))?
        .try_into()
        .unwrap();
    *cursor = end;
    Ok(value)
}

fn model_fingerprint(gguf: &infr_gguf::Gguf) -> Result<[u8; 32]> {
    const SAMPLE: usize = 64 * 1024;
    let shards = gguf.shards();
    let mut hasher = Sha256::new();
    hasher.update(b"infr-cold-kv-model-v1");
    hasher.update((shards.len() as u64).to_le_bytes());
    let mut sample = vec![0u8; SAMPLE];
    for (path, declared_len) in shards {
        let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let label = canonical.to_string_lossy();
        hasher.update((label.len() as u64).to_le_bytes());
        hasher.update(label.as_bytes());
        hasher.update(declared_len.to_le_bytes());
        let metadata = fs::metadata(path)
            .with_context(|| format!("fingerprint model shard {}", path.display()))?;
        if metadata.len() != declared_len {
            return Err(anyhow!(
                "model shard {} changed size while opening the cold KV cache",
                path.display()
            ));
        }
        if let Ok(modified) = metadata.modified() {
            let stamp = modified.duration_since(UNIX_EPOCH).unwrap_or_default();
            hasher.update(stamp.as_secs().to_le_bytes());
            hasher.update(stamp.subsec_nanos().to_le_bytes());
        }
        let mut file = File::open(path)?;
        let head = declared_len.min(SAMPLE as u64) as usize;
        file.read_exact(&mut sample[..head])?;
        hasher.update(&sample[..head]);
        if declared_len > SAMPLE as u64 {
            let tail = declared_len.min(SAMPLE as u64) as usize;
            file.seek(SeekFrom::End(-(tail as i64)))?;
            file.read_exact(&mut sample[..tail])?;
            hasher.update(&sample[..tail]);
        }
    }
    Ok(hasher.finalize().into())
}

fn gc_root(root: &Path, max_bytes: u64, ttl: Option<Duration>) -> Result<()> {
    remove_stale_temporary_files(
        root,
        ttl.unwrap_or(MIN_STALE_TEMP_AGE).max(MIN_STALE_TEMP_AGE),
    )?;
    remove_orphan_mtp_files(root)?;
    let mut files = collect_cache_files(root)?;
    let now = SystemTime::now();
    if let Some(ttl) = ttl {
        for file in &mut files {
            if now.duration_since(file.modified).unwrap_or_default() > ttl
                && remove_cache_file(&file.path)
            {
                file.removed = true;
            }
        }
    }
    let mut total = files
        .iter()
        .filter(|file| !file.removed)
        .fold(0u64, |total, file| total.saturating_add(file.len));
    files.sort_by(|left, right| {
        left.modified
            .cmp(&right.modified)
            .then_with(|| left.path.cmp(&right.path))
    });
    for file in files.iter_mut().filter(|file| !file.removed) {
        if total <= max_bytes {
            break;
        }
        if remove_cache_file(&file.path) {
            file.removed = true;
            total = total.saturating_sub(file.len);
        }
    }
    Ok(())
}

fn remove_stale_temporary_files(root: &Path, max_age: Duration) -> Result<()> {
    let now = SystemTime::now();
    for item in
        fs::read_dir(root).with_context(|| format!("scan cold KV cache root {}", root.display()))?
    {
        let item = match item {
            Ok(item) => item,
            Err(error) => {
                tracing::warn!("cold KV cache: skip unreadable root entry: {error}");
                continue;
            }
        };
        let Ok(file_type) = item.file_type() else {
            continue;
        };
        if file_type.is_file() {
            remove_stale_temporary_file(&item.path(), now, max_age);
        } else if file_type.is_dir() {
            let Ok(children) = fs::read_dir(item.path()) else {
                continue;
            };
            for child in children.flatten() {
                if child.file_type().is_ok_and(|kind| kind.is_file()) {
                    remove_stale_temporary_file(&child.path(), now, max_age);
                }
            }
        }
    }
    Ok(())
}

fn remove_stale_temporary_file(path: &Path, now: SystemTime, max_age: Duration) {
    if !is_session_temporary_file(path) {
        return;
    }
    let Ok(metadata) = fs::metadata(path) else {
        return;
    };
    let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
    if now.duration_since(modified).unwrap_or_default() > max_age {
        let _ = remove_cache_file(path);
    }
}

struct CacheFile {
    path: PathBuf,
    len: u64,
    modified: SystemTime,
    removed: bool,
}

fn collect_cache_files(root: &Path) -> Result<Vec<CacheFile>> {
    let mut files = Vec::new();
    for item in
        fs::read_dir(root).with_context(|| format!("scan cold KV cache root {}", root.display()))?
    {
        let item = match item {
            Ok(item) => item,
            Err(error) => {
                tracing::warn!("cold KV cache: skip unreadable root entry: {error}");
                continue;
            }
        };
        let Ok(file_type) = item.file_type() else {
            continue;
        };
        if file_type.is_file() {
            push_cache_file(&mut files, item.path());
        } else if file_type.is_dir() {
            let Ok(children) = fs::read_dir(item.path()) else {
                continue;
            };
            for child in children.flatten() {
                if child.file_type().is_ok_and(|kind| kind.is_file()) {
                    push_cache_file(&mut files, child.path());
                }
            }
        }
    }
    Ok(files)
}

fn push_cache_file(files: &mut Vec<CacheFile>, path: PathBuf) {
    if !is_cache_file(&path) {
        return;
    }
    let Ok(metadata) = fs::metadata(&path) else {
        return;
    };
    files.push(CacheFile {
        len: metadata
            .len()
            .saturating_add(fs::metadata(mtp_path_for(&path)).map_or(0, |meta| meta.len())),
        path,
        modified: metadata.modified().unwrap_or(UNIX_EPOCH),
        removed: false,
    });
}

fn remove_cache_file(path: &Path) -> bool {
    let removed = match fs::remove_file(path) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
        Err(error) => {
            tracing::warn!(path = %path.display(), "cold KV cache: cannot remove expired entry: {error}");
            false
        }
    };
    if removed && is_cache_file(path) {
        let sidecar = mtp_path_for(path);
        if let Err(error) = fs::remove_file(&sidecar) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = %sidecar.display(), "cold KV cache: cannot remove paired MTP sidecar: {error}");
            }
        }
    }
    removed
}

fn remove_cache_pair(path: &Path) {
    let _ = remove_cache_file(path);
}

fn mtp_path_for(path: &Path) -> PathBuf {
    path.with_extension("infrmtp")
}

fn remove_orphan_mtp_files(root: &Path) -> Result<()> {
    for item in
        fs::read_dir(root).with_context(|| format!("scan cold KV cache root {}", root.display()))?
    {
        let item = match item {
            Ok(item) => item,
            Err(_) => continue,
        };
        if item.file_type().is_ok_and(|kind| kind.is_dir()) {
            let Ok(children) = fs::read_dir(item.path()) else {
                continue;
            };
            for child in children.flatten() {
                let path = child.path();
                if path
                    .extension()
                    .is_some_and(|extension| extension == "infrmtp")
                    && !path.with_extension("infrkv").is_file()
                {
                    let _ = fs::remove_file(path);
                }
            }
        }
    }
    Ok(())
}

fn is_cache_file(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == "infrkv")
}

fn is_session_temporary_file(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "tmp")
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(".session-"))
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn hex_digest(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &byte in digest {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn encode_dtype(dtype: DType) -> u16 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::Bf16 => 2,
        DType::I32 => 3,
        DType::U32 => 4,
        DType::Q4_0 => 5,
        DType::Q4_1 => 6,
        DType::Q5_0 => 7,
        DType::Q5_1 => 8,
        DType::Q8_0 => 9,
        DType::Q2K => 10,
        DType::Q3K => 11,
        DType::Q4K => 12,
        DType::Q5K => 13,
        DType::Q6K => 14,
        DType::Iq1S => 15,
        DType::Iq1M => 16,
        DType::Iq2Xxs => 17,
        DType::Iq2Xs => 18,
        DType::Iq2S => 19,
        DType::Iq3Xxs => 20,
        DType::Iq3S => 21,
        DType::Iq4Nl => 22,
        DType::Iq4Xs => 23,
        DType::Tq1_0 => 24,
        DType::Tq2_0 => 25,
        DType::I2S => 26,
        DType::Q2_0 => 27,
        DType::Mxfp4 => 28,
        DType::Nvfp4 => 29,
        DType::Turbo2 => 30,
        DType::Turbo3 => 31,
        DType::Turbo4 => 32,
    }
}

fn decode_dtype(value: u16) -> Result<DType> {
    Ok(match value {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::Bf16,
        3 => DType::I32,
        4 => DType::U32,
        5 => DType::Q4_0,
        6 => DType::Q4_1,
        7 => DType::Q5_0,
        8 => DType::Q5_1,
        9 => DType::Q8_0,
        10 => DType::Q2K,
        11 => DType::Q3K,
        12 => DType::Q4K,
        13 => DType::Q5K,
        14 => DType::Q6K,
        15 => DType::Iq1S,
        16 => DType::Iq1M,
        17 => DType::Iq2Xxs,
        18 => DType::Iq2Xs,
        19 => DType::Iq2S,
        20 => DType::Iq3Xxs,
        21 => DType::Iq3S,
        22 => DType::Iq4Nl,
        23 => DType::Iq4Xs,
        24 => DType::Tq1_0,
        25 => DType::Tq2_0,
        26 => DType::I2S,
        27 => DType::Q2_0,
        28 => DType::Mxfp4,
        29 => DType::Nvfp4,
        30 => DType::Turbo2,
        31 => DType::Turbo3,
        32 => DType::Turbo4,
        _ => return Err(anyhow!("unknown cold KV dtype id {value}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    struct GateCheckingWriter {
        gate: std::sync::Arc<StepGate>,
        bytes: Vec<u8>,
        writes: usize,
        fail_after: Option<usize>,
    }

    impl Write for GateCheckingWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let gate = self.gate.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            let thread = std::thread::spawn(move || {
                let _pass = gate.enter();
                let _ = tx.send(());
            });
            rx.recv_timeout(Duration::from_secs(2))
                .map_err(|_| std::io::Error::other("SSD write held the inference gate"))?;
            thread.join().unwrap();
            if self.fail_after == Some(self.writes) {
                return Err(std::io::Error::other("injected SSD write failure"));
            }
            self.writes += 1;
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn spill_payload_yields_inference_gate_between_downloads_and_ssd_writes() {
        use infr_core::backend::BufferUsage;
        let backend = infr_cpu::CpuBackend::new_with(std::sync::Arc::new(EngineConfig::default()));
        let gate = std::sync::Arc::new(StepGate::new());
        let payload: Vec<_> = (0..83).collect();
        let buffer = backend.alloc(payload.len(), BufferUsage::KvCache).unwrap();
        backend.upload(buffer.as_ref(), &payload).unwrap();
        let mut writer = GateCheckingWriter {
            gate: gate.clone(),
            bytes: Vec::new(),
            writes: 0,
            fail_after: None,
        };
        let mut hasher = Sha256::new();
        write_device_payload(
            &mut writer,
            &mut hasher,
            buffer.as_ref(),
            payload.len(),
            &backend,
            Some(&gate),
            &mut [0; 17],
        )
        .unwrap();
        assert_eq!(writer.bytes, payload);
        assert_eq!(writer.writes, 5);
        assert_eq!(hasher.finalize(), Sha256::digest(&payload));
        writer.fail_after = Some(5);
        assert!(write_device_payload(
            &mut writer,
            &mut Sha256::new(),
            buffer.as_ref(),
            payload.len(),
            &backend,
            Some(&gate),
            &mut [0; 17]
        )
        .is_err());
        assert!(write_device_payload(
            &mut Vec::new(),
            &mut Sha256::new(),
            buffer.as_ref(),
            payload.len() + 17,
            &backend,
            Some(&gate),
            &mut [0; 17]
        )
        .is_err());
        let _pass = gate.enter();
    }

    #[test]
    fn gated_spill_file_preserves_layout_checkpoints_and_checksum() {
        use infr_core::backend::BufferUsage;
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("roundtrip.infrkv");
        let backend = infr_cpu::CpuBackend::new_with(std::sync::Arc::new(EngineConfig::default()));
        let payload: Vec<u8> = (0..127).collect();
        let buffer = backend.alloc(payload.len(), BufferUsage::KvCache).unwrap();
        backend.upload(buffer.as_ref(), &payload).unwrap();
        let meta = SessionStateMeta {
            max_ctx: 32768,
            k_fmt: DType::Q8_0,
            v_fmt: DType::Q8_0,
            committed_tokens: 32768,
            cached: vec![1, 2, 3],
            checkpoint_tokens: [Some(vec![1]), Some(vec![1, 2])],
        };
        let header = Header::new([7; 32], &meta, 1, payload.len() as u64).unwrap();
        let records = [crate::seam::SessionBuffer {
            key: SessionBufferKey::K(0),
            buffer: buffer.as_ref(),
            committed_bytes: payload.len(),
        }];
        write_session_file(
            &path,
            &header,
            &meta,
            &records,
            &backend,
            Some(&StepGate::new()),
        )
        .unwrap();
        let file = fs::read(&path).unwrap();
        assert_eq!(file.len() as u64, checked_file_bytes(&header).unwrap());
        let data_end = file.len() - CHECKSUM_BYTES as usize;
        assert_eq!(
            &file[data_end..],
            Sha256::digest(&file[..data_end]).as_slice()
        );
        assert_eq!(&file[data_end - payload.len()..data_end], payload);
        let entry = read_catalog_entry(&path, [7; 32], meta.max_ctx, meta.k_fmt, meta.v_fmt)
            .unwrap()
            .unwrap();
        assert_eq!(entry.meta.cached, meta.cached);
        assert_eq!(entry.meta.checkpoint_tokens, meta.checkpoint_tokens);
    }

    #[test]
    fn detached_spill_publication_preserves_existing_entries_and_prunes_gc_victims() {
        let temp = tempfile::tempdir().unwrap();
        let make_entry = |name: &str, token| {
            let path = temp.path().join(name);
            fs::write(&path, [0u8; 10]).unwrap();
            ColdEntry {
                path,
                saved_at: 0,
                file_bytes: 10,
                meta: SessionStateMeta {
                    max_ctx: 32768,
                    k_fmt: DType::Q8_0,
                    v_fmt: DType::Q8_0,
                    committed_tokens: 32768,
                    cached: vec![token],
                    checkpoint_tokens: [None, None],
                },
            }
        };
        let mut cache = SessionCache {
            root: temp.path().to_path_buf(),
            model_dir: temp.path().to_path_buf(),
            fingerprint: [0; 32],
            max_bytes: 1000,
            ttl: None,
            max_ctx: 32768,
            k_fmt: DType::Q8_0,
            v_fmt: DType::Q8_0,
            entries: vec![make_entry("old.infrkv", 1), make_entry("victim.infrkv", 2)],
        };
        let mut writer = cache.spill_writer();
        assert!(writer.entries.is_empty());
        writer.entries.push(make_entry("new.infrkv", 3));
        fs::remove_file(&cache.entries[1].path).unwrap();
        cache.publish_spills(writer);
        assert_eq!(cache.entries.len(), 2);
        assert!(cache.entries.iter().any(|entry| entry.meta.cached == [1]));
        assert!(cache.entries.iter().any(|entry| entry.meta.cached == [3]));
    }

    #[test]
    fn header_round_trips_and_has_a_checked_size() {
        let header = Header {
            fingerprint: [7; 32],
            saved_at: 123,
            max_ctx: 131_072,
            committed_tokens: 65_536,
            cached_count: 42,
            checkpoint_counts: [17, 29],
            record_count: 12,
            k_fmt: DType::Q8_0,
            v_fmt: DType::F16,
            data_bytes: 98_765,
            has_checkpoints: [true, true],
        };
        assert_eq!(decode_header(&encode_header(&header)).unwrap(), header);
        assert_eq!(
            checked_file_bytes(&header).unwrap(),
            HEADER_BYTES + (42 + 17 + 29) * 4 + 12 * RECORD_HEADER_BYTES + 98_765 + CHECKSUM_BYTES
        );
    }

    #[test]
    fn mtp_header_round_trips_and_prices_the_complete_sidecar() {
        let state = Qwen4MtpCacheState {
            cached: vec![1, 2, 3, 4],
            last_h: vec![0.25; 8],
            turn_checkpoints: [
                Some(Qwen4MtpCheckpointState {
                    tokens: vec![1, 2],
                    last_h: vec![0.5; 8],
                }),
                None,
            ],
            pending_id: Some(9),
            pending_emitted: true,
        };
        let header = MtpHeader::new([3; 32], 4096, &state, 8192).unwrap();
        assert_eq!(
            decode_mtp_header(&encode_mtp_header(&header)).unwrap(),
            header
        );
        let expected_payload = (4 + 2) * 4 + (8 + 8) * 4 + 2 * 8192;
        assert_eq!(header.data_bytes, expected_payload);
        assert_eq!(
            checked_mtp_file_bytes(&header).unwrap(),
            MTP_HEADER_BYTES + expected_payload + CHECKSUM_BYTES
        );
        validate_mtp_header(&header, checked_mtp_file_bytes(&header).unwrap()).unwrap();
    }

    #[test]
    fn record_keys_round_trip() {
        let keys = [
            SessionBufferKey::K(4),
            SessionBufferKey::V(5),
            SessionBufferKey::QsaRaw(6),
            SessionBufferKey::QsaBlock(7),
            SessionBufferKey::PleState,
            SessionBufferKey::CheckpointK(0, 8),
            SessionBufferKey::CheckpointV(0, 9),
            SessionBufferKey::CheckpointPle(0),
            SessionBufferKey::CheckpointK(1, 10),
            SessionBufferKey::CheckpointV(1, 11),
            SessionBufferKey::CheckpointPle(1),
        ];
        for key in keys {
            assert_eq!(decode_record(&encode_record(key, 99)).unwrap(), (key, 99));
        }
    }

    #[test]
    fn gc_caps_only_cache_files() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("model");
        fs::create_dir(&model).unwrap();
        for index in 0..3 {
            fs::write(model.join(format!("{index}.infrkv")), [0u8; 10]).unwrap();
        }
        fs::write(model.join("keep.txt"), [0u8; 100]).unwrap();
        gc_root(temp.path(), 15, None).unwrap();
        let cache_bytes = collect_cache_files(temp.path())
            .unwrap()
            .iter()
            .map(|file| file.len)
            .sum::<u64>();
        assert!(cache_bytes <= 15);
        assert!(model.join("keep.txt").is_file());
    }

    #[test]
    fn gc_counts_and_removes_main_and_mtp_as_one_snapshot() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("model");
        fs::create_dir(&model).unwrap();
        let main = model.join("session.infrkv");
        let mtp = model.join("session.infrmtp");
        fs::write(&main, [0u8; 10]).unwrap();
        fs::write(&mtp, [0u8; 20]).unwrap();
        let files = collect_cache_files(temp.path()).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].len, 30);
        gc_root(temp.path(), 0, None).unwrap();
        assert!(!main.exists());
        assert!(!mtp.exists());
    }

    #[test]
    fn temporary_file_classifier_is_narrow() {
        assert!(is_session_temporary_file(Path::new(
            ".session-0000000000000000-00000000-0000000000000000.tmp"
        )));
        assert!(!is_session_temporary_file(Path::new("session.tmp")));
        assert!(!is_session_temporary_file(Path::new(
            ".session-data.infrkv"
        )));
        assert!(!is_session_temporary_file(Path::new("unrelated.tmp")));
    }

    #[test]
    fn every_dtype_has_a_stable_round_trip_id() {
        let mut ids = BTreeSet::new();
        for id in 0..=32 {
            let dtype = decode_dtype(id).unwrap();
            assert_eq!(encode_dtype(dtype), id);
            assert!(ids.insert(id));
        }
    }
}
