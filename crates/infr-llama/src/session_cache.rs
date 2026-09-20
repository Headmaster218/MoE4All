//! Opt-in cold storage for idle Vulkan conversation state.
//!
//! The scheduler owns policy (which resident slot to evict and which prefix to restore). This
//! module owns the durable representation, model/geometry validation, streaming device transfers,
//! and directory maintenance. No code on the ordinary generation path reaches this module when
//! `kv.session_cache_dir` is unset.

use crate::seam::{SeamKv, SessionBufferKey, SessionStateMeta};
use crate::{Config, EngineConfig};
use anyhow::{anyhow, Context, Result};
use infr_core::backend::Backend;
use infr_core::{DType, SizeSpec};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAGIC: [u8; 8] = *b"INFRKV01";
const VERSION: u32 = 1;
const HEADER_BYTES: u64 = 104;
const RECORD_HEADER_BYTES: u64 = 16;
const CHECKSUM_BYTES: u64 = 32;
const STREAM_BYTES: usize = 16 * 1024 * 1024;
const MAX_RECORDS: u32 = 16_384;
const FLAG_CHECKPOINT: u32 = 1;
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct Header {
    fingerprint: [u8; 32],
    saved_at: u64,
    max_ctx: u64,
    committed_tokens: u64,
    cached_count: u64,
    checkpoint_count: u64,
    record_count: u32,
    k_fmt: DType,
    v_fmt: DType,
    data_bytes: u64,
    has_checkpoint: bool,
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
                    let _ = fs::remove_file(&path);
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

        backend
            .sync()
            .map_err(|error| anyhow!("sync before cold KV spill: {error}"))?;
        let buffers = kv.session_state_buffers(backend)?;
        let data_bytes = buffers.iter().try_fold(0u64, |total, buffer| {
            total
                .checked_add(buffer.committed_bytes as u64)
                .ok_or_else(|| anyhow!("cold KV payload size overflow"))
        })?;
        let header = Header::new(self.fingerprint, &meta, buffers.len(), data_bytes)?;
        let file_bytes = checked_file_bytes(&header)?;
        if file_bytes > self.max_bytes {
            return Err(anyhow!(
                "one cold KV session needs {:.2} GiB, exceeding kv.session_cache_max {:.2} GiB",
                file_bytes as f64 / (1u64 << 30) as f64,
                self.max_bytes as f64 / (1u64 << 30) as f64,
            ));
        }

        let (temporary, final_path) = self.new_paths();
        let write_result = write_session_file(&temporary, &header, &meta, &buffers, backend);
        drop(buffers);
        if let Err(error) = write_result {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        if let Err(error) = fs::rename(&temporary, &final_path) {
            let _ = fs::remove_file(&temporary);
            return Err(error).with_context(|| {
                format!(
                    "publish cold KV session {} -> {}",
                    temporary.display(),
                    final_path.display()
                )
            });
        }
        if let Err(error) = kv.release_session_state(backend, model_cfg) {
            let _ = fs::remove_file(&final_path);
            return Err(error.context("release resident KV after durable spill"));
        }
        self.entries.push(ColdEntry {
            path: final_path,
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
        entry: ColdEntry,
        kv: &mut SeamKv,
        backend: &dyn Backend,
        model_cfg: &Config,
    ) -> Result<()> {
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
        if let Err(error) = fs::remove_file(&path) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = %path.display(), "cold KV cache: cannot remove consumed entry: {error}");
            }
        }
        if result.is_ok() {
            tracing::info!(
                tokens,
                bytes = file_bytes,
                gib = file_bytes as f64 / (1u64 << 30) as f64,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "restored conversation KV from cold storage"
            );
        }
        result
    }

    pub(crate) fn gc(&mut self) -> Result<()> {
        gc_root(&self.root, self.max_bytes, self.ttl)?;
        self.entries.retain(|entry| entry.path.is_file());
        Ok(())
    }

    fn new_paths(&self) -> (PathBuf, PathBuf) {
        let now = unix_secs();
        let sequence = FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let stem = format!(
            "session-{now:016x}-{:08x}-{sequence:016x}",
            std::process::id()
        );
        (
            self.model_dir.join(format!(".{stem}.tmp")),
            self.model_dir.join(format!("{stem}.infrkv")),
        )
    }
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
            checkpoint_count: meta.checkpoint_tokens.as_ref().map_or(Ok(0), |tokens| {
                tokens
                    .len()
                    .try_into()
                    .context("checkpoint token count exceeds u64")
            })?,
            record_count: record_count
                .try_into()
                .context("cold KV record count exceeds u32")?,
            k_fmt: meta.k_fmt,
            v_fmt: meta.v_fmt,
            data_bytes,
            has_checkpoint: meta.checkpoint_tokens.is_some(),
        })
    }

    fn meta(&self, cached: Vec<u32>, checkpoint: Option<Vec<u32>>) -> Result<SessionStateMeta> {
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
            checkpoint_tokens: checkpoint,
        })
    }
}

fn write_session_file(
    path: &Path,
    header: &Header,
    meta: &SessionStateMeta,
    buffers: &[crate::seam::SessionBuffer<'_>],
    backend: &dyn Backend,
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
    if let Some(tokens) = meta.checkpoint_tokens.as_deref() {
        write_tokens(&mut writer, &mut hasher, tokens)?;
    }
    let mut scratch = vec![0u8; STREAM_BYTES];
    for buffer in buffers {
        let record = encode_record(buffer.key, buffer.committed_bytes as u64);
        write_hashed(&mut writer, &mut hasher, &record)?;
        let mut offset = 0usize;
        while offset < buffer.committed_bytes {
            let count = (buffer.committed_bytes - offset).min(scratch.len());
            backend
                .download_range(buffer.buffer, offset, &mut scratch[..count])
                .map_err(|error| {
                    anyhow!("download cold KV {:?} at {offset}: {error}", buffer.key)
                })?;
            write_hashed(&mut writer, &mut hasher, &scratch[..count])?;
            offset += count;
        }
    }
    let checksum = hasher.finalize();
    writer.write_all(checksum.as_ref())?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
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
    let checkpoint = if header.has_checkpoint {
        Some(read_tokens_hashed(
            &mut reader,
            &mut hasher,
            header.checkpoint_count,
            max_ctx,
        )?)
    } else {
        None
    };
    let meta = header.meta(cached, checkpoint)?;
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
    let file_bytes = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut bytes = [0u8; HEADER_BYTES as usize];
    reader.read_exact(&mut bytes)?;
    let header = decode_header(&bytes)?;
    validate_header(&header, file_bytes)?;
    if header.fingerprint != fingerprint
        || header.max_ctx != max_ctx as u64
        || header.k_fmt != k_fmt
        || header.v_fmt != v_fmt
    {
        return Ok(None);
    }
    let cached = read_tokens(&mut reader, header.cached_count, max_ctx)?;
    let checkpoint = if header.has_checkpoint {
        Some(read_tokens(&mut reader, header.checkpoint_count, max_ctx)?)
    } else {
        None
    };
    Ok(Some(ColdEntry {
        path: path.to_path_buf(),
        saved_at: header.saved_at,
        meta: header.meta(cached, checkpoint)?,
        file_bytes,
    }))
}

fn validate_header(header: &Header, file_bytes: u64) -> Result<()> {
    if header.record_count > MAX_RECORDS {
        return Err(anyhow!("cold KV record count is implausibly large"));
    }
    if header.cached_count > header.max_ctx
        || header.checkpoint_count > header.max_ctx
        || header.committed_tokens > header.max_ctx
        || (header.has_checkpoint != (header.checkpoint_count != 0))
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
    let token_bytes = header
        .cached_count
        .checked_add(header.checkpoint_count)
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
    let flags = if header.has_checkpoint {
        FLAG_CHECKPOINT
    } else {
        0
    };
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&header.fingerprint);
    out.extend_from_slice(&header.saved_at.to_le_bytes());
    out.extend_from_slice(&header.max_ctx.to_le_bytes());
    out.extend_from_slice(&header.committed_tokens.to_le_bytes());
    out.extend_from_slice(&header.cached_count.to_le_bytes());
    out.extend_from_slice(&header.checkpoint_count.to_le_bytes());
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
    if flags & !FLAG_CHECKPOINT != 0 {
        return Err(anyhow!("cold KV header has unknown flags"));
    }
    let fingerprint = take(bytes, &mut cursor)?;
    let saved_at = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let max_ctx = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let committed_tokens = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let cached_count = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let checkpoint_count = u64::from_le_bytes(take(bytes, &mut cursor)?);
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
        checkpoint_count,
        record_count,
        k_fmt,
        v_fmt,
        data_bytes,
        has_checkpoint: flags & FLAG_CHECKPOINT != 0,
    })
}

fn encode_record(key: SessionBufferKey, len: u64) -> [u8; RECORD_HEADER_BYTES as usize] {
    let (kind, layer) = match key {
        SessionBufferKey::K(layer) => (0, layer),
        SessionBufferKey::V(layer) => (1, layer),
        SessionBufferKey::QsaRaw(layer) => (2, layer),
        SessionBufferKey::QsaBlock(layer) => (3, layer),
        SessionBufferKey::PleState => (4, u32::MAX),
        SessionBufferKey::CheckpointK(layer) => (5, layer),
        SessionBufferKey::CheckpointV(layer) => (6, layer),
        SessionBufferKey::CheckpointPle => (7, u32::MAX),
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
        5 => SessionBufferKey::CheckpointK(layer),
        6 => SessionBufferKey::CheckpointV(layer),
        7 if layer == u32::MAX => SessionBufferKey::CheckpointPle,
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
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
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
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
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
        path,
        len: metadata.len(),
        modified: metadata.modified().unwrap_or(UNIX_EPOCH),
        removed: false,
    });
}

fn remove_cache_file(path: &Path) -> bool {
    match fs::remove_file(path) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
        Err(error) => {
            tracing::warn!(path = %path.display(), "cold KV cache: cannot remove expired entry: {error}");
            false
        }
    }
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

    #[test]
    fn header_round_trips_and_has_a_checked_size() {
        let header = Header {
            fingerprint: [7; 32],
            saved_at: 123,
            max_ctx: 131_072,
            committed_tokens: 65_536,
            cached_count: 42,
            checkpoint_count: 17,
            record_count: 12,
            k_fmt: DType::Q8_0,
            v_fmt: DType::F16,
            data_bytes: 98_765,
            has_checkpoint: true,
        };
        assert_eq!(decode_header(&encode_header(&header)).unwrap(), header);
        assert_eq!(
            checked_file_bytes(&header).unwrap(),
            HEADER_BYTES + (42 + 17) * 4 + 12 * RECORD_HEADER_BYTES + 98_765 + CHECKSUM_BYTES
        );
    }

    #[test]
    fn record_keys_round_trip() {
        let keys = [
            SessionBufferKey::K(4),
            SessionBufferKey::V(5),
            SessionBufferKey::QsaRaw(6),
            SessionBufferKey::QsaBlock(7),
            SessionBufferKey::PleState,
            SessionBufferKey::CheckpointK(8),
            SessionBufferKey::CheckpointV(9),
            SessionBufferKey::CheckpointPle,
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
