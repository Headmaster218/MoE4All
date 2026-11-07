//! One blob's download: choosing between a single stream and a ranged fan-out, resume for both,
//! a progress bar, a size cap for the unverified case, and the end-of-body sha256 gate that decides
//! whether the bytes become a content-addressed blob at all.
//!
//! Everything here is about ONE file. Which files a model is made of, and how many are fetched at a
//! time, belongs to [`crate::pull`]; how one file is split across connections belongs to
//! [`crate::ranged`], and what an interrupted split leaves on disk to [`crate::parts`].

use crate::http::{download_client, token, ConnBudget};
use crate::parts;
use crate::ranged::{self, RangedError};
use indicatif::{MultiProgress, ProgressBar};
use infr_core::error::{Error, Result};
use infr_core::progress::{self, Unit};
use reqwest::blocking::Response;
use sha2::{Digest, Sha256};
#[cfg(not(target_os = "windows"))]
use std::os::unix::io::AsRawFd;
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tracing::{debug, info};

/// Stream `url` into `blobs/<sha256>` (HF's content-addressed blob name), resuming a prior partial if
/// present. Returns `(blob_path, hex_digest, total_bytes)`. On a transport error the partial temp file
/// is KEPT so a later call resumes from where it stopped.
///
/// When `expected_sha` is `Some`, it is HF's advertised LFS sha256 and is used two ways: the download
/// is skipped entirely if that content-addressed blob is already on disk, and the downloaded bytes are
/// verified against it before the blob is committed — a mismatch discards the temp and errors (a
/// corrupt/truncated body, or a resume of a stale partial from a since-changed file, must never be
/// linked as the model). `None` (non-LFS file / no digest available) proceeds without verification.
///
/// `max_bytes` caps the FINAL size of the file (resumed prefix included) and is enforced INSIDE the
/// streaming loop, so an over-long body is aborted mid-transfer instead of after it has already
/// landed on disk — the whole point of a cap on an unverified download. `None` means uncapped and
/// is what the model-blob path passes: a GGUF is legitimately multi-GiB, and its integrity comes
/// from `expected_sha` instead. See [`crate::pull::MAX_COMPANION_BYTES`] for the only capped
/// caller. The advertised `Content-Length` is checked first as a courtesy (fail before writing a
/// byte) but is never trusted on its own — it is attacker-controlled and may be absent under
/// chunked encoding, which is exactly why the loop counts for itself.
///
/// `progress` owns the bar's line. Several of these run at once (one per shard), so the bar joins a
/// group rather than addressing the terminal itself — see [`infr_core::progress::group`]. There is
/// ONE bar per file however many connections carry it: a ranged download's chunk workers all
/// advance the same line, because "which slice of shard 3 is at 40%" is not a thing anyone wants
/// eight lines of.
///
/// `budget` is the pull's whole allowance of simultaneous connections ([`ConnBudget`]). A single
/// large file spends what the file fan-out left over on fetching ITSELF in parallel; when there is
/// nothing spare — `hub.pull_jobs = 1`, or a 236-shard repo with every permit already on a file —
/// the download is one stream, exactly as before.
///
/// The HF bearer token ([`crate::http::token`]) is attached when the environment carries one, so a
/// gated repo works the same on this path as on the metadata calls.
pub(crate) fn download_to_blob(
    url: &str,
    blobs: &Path,
    label: &str,
    expected_sha: Option<&str>,
    max_bytes: Option<u64>,
    budget: &ConnBudget,
    progress: &MultiProgress,
) -> Result<(PathBuf, String, u64)> {
    fs::create_dir_all(blobs).map_err(Error::from)?;
    // Content-addressed short-circuit: if we already know the sha and hold that blob, we're done.
    if let Some(sha) = expected_sha {
        let blob = blobs.join(sha);
        if blob.exists() {
            let size = fs::metadata(&blob).map(|m| m.len()).unwrap_or(0);
            debug!("blob {sha} already present ({size} bytes); skipping download of {label}");
            return Ok((blob, sha.to_string(), size));
        }
    } else {
        debug!("no expected sha256 for {label}; download will not be integrity-checked");
    }
    let obj = Object {
        url,
        label,
        oid: expected_sha,
    };
    let partial = partial_name(label);
    let temps = Temps {
        stream: blobs.join(&partial),
        meta: blobs.join(format!("{partial}.meta")), // If-Range validator for a stream partial
        ranged: blobs.join(ranged_partial_name(label)),
    };

    // Serialize concurrent pulls of the SAME blob (auto-pull racing a manual `pull`, two `run`s).
    // An advisory `flock` on a per-blob lockfile is chosen over unique-per-process temp names ON
    // PURPOSE: it PRESERVES resume — the one shared temp keeps accumulating across processes instead
    // of each starting a fresh partial from byte 0. The lock releases when `_lock` drops. It is
    // named off the STREAM partial for both modes, so two processes that picked different modes for
    // the same blob still serialise.
    let _lock = FileLock::acquire(&blobs.join(format!("{partial}.lock")))?;
    // Re-check the content-addressed short-circuit now that we hold the lock: a racing process may
    // have finished this exact blob while we waited.
    if let Some(sha) = expected_sha {
        let blob = blobs.join(sha);
        if blob.exists() {
            let size = fs::metadata(&blob).map(|m| m.len()).unwrap_or(0);
            return Ok((blob, sha.to_string(), size));
        }
    }

    // Ranged first when there is a ranged partial to continue, or when a fresh download has budget
    // to split; `None` means "not this one" and falls through to the single stream below.
    let (tmp, pb) = match fetch_ranged(&obj, &temps, max_bytes, budget, progress)? {
        Some(done) => done,
        None => fetch_stream(&obj, &temps, max_bytes, progress)?,
    };
    commit(&obj, &temps, &tmp, blobs, &pb, progress)
}

/// The object being fetched: where it is, what to call it on screen, and the LFS sha256 HF says it
/// hashes to. Passed as one value because every stage of a download needs all three — the request,
/// the progress line, the identity a partial is bound to and the gate at the end.
pub(crate) struct Object<'a> {
    pub(crate) url: &'a str,
    pub(crate) label: &'a str,
    /// HF's advertised LFS sha256, or `None` for a non-LFS file (which is then unverifiable).
    pub(crate) oid: Option<&'a str>,
}

/// The three on-disk names one download can leave behind, all derived from [`temp_stem`]: the
/// single-stream partial, its `If-Range` validator, and the ranged partial (whose `.plan` sidecar
/// hangs off it in [`crate::parts`]).
struct Temps {
    stream: PathBuf,
    meta: PathBuf,
    ranged: PathBuf,
}

/// Hash the assembled temp file, gate it on the advertised sha256, and — only then — commit it as
/// the content-addressed blob.
///
/// This is the same last line of defence it has always been, and it is deliberately the ONE place
/// that decides: whether the bytes arrived on one connection or on eight, whether they were resumed
/// or not, they become a blob here or not at all.
fn commit(
    obj: &Object,
    temps: &Temps,
    tmp: &Path,
    blobs: &Path,
    pb: &ProgressBar,
    progress: &MultiProgress,
) -> Result<(PathBuf, String, u64)> {
    let label = obj.label;
    // Hash the COMPLETE file ONCE at the end. The old code folded the on-disk prefix into the digest
    // on every resume (O(K·size) over K flaky-link retries); since the whole body is sha-verified
    // here anyway, a single final pass is equivalent and cheaper.
    let mut hasher = Sha256::new();
    hash_file(tmp, &mut hasher)?;
    let hex: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let size = fs::metadata(tmp).map(|m| m.len()).unwrap_or(0);

    // Integrity gate: the body MUST hash to HF's advertised LFS sha256. On mismatch discard the temp
    // (do NOT keep it for resume — a resumed corrupt prefix stays corrupt) and fail loudly.
    if let Err(e) = verify_sha(label, &hex, obj.oid) {
        pb.abandon_with_message(format!("⚠ {label} sha256 mismatch"));
        let _ = fs::remove_file(tmp);
        let _ = fs::remove_file(&temps.meta);
        parts::discard(&temps.ranged);
        return Err(e);
    }
    pb.finish_with_message(format!("✓ {label} ({} MiB)", size / (1024 * 1024)));

    let blob = blobs.join(&hex); // HF blob name = bare sha256 hex
    fs::rename(tmp, &blob).map_err(Error::from)?;
    let _ = fs::remove_file(&temps.meta); // partial committed; validator no longer needed
    let _ = fs::remove_file(parts::plan_path(&temps.ranged));
    progress.suspend(|| info!("Saved blob: {blob:?}"));
    Ok((blob, hex, size))
}

/// Try to fetch the object as a grid of ranges, returning `Ok(None)` when it should be a single
/// stream instead: no budget for a second connection, a capped (companion) download, an object of
/// one chunk or less, or a server that will not serve ranges.
///
/// A stream partial on disk also keeps its own mode: it is an append-only prefix, and continuing it
/// with sparse writers is not something the sidecar can describe. Nothing is lost by that — the
/// next fresh download of that object splits.
fn fetch_ranged(
    obj: &Object,
    temps: &Temps,
    max_bytes: Option<u64>,
    budget: &ConnBudget,
    progress: &MultiProgress,
) -> Result<Option<(PathBuf, ProgressBar)>> {
    let (label, ranged_tmp) = (obj.label, temps.ranged.as_path());
    let existing = ranged_tmp
        .exists()
        .then(|| parts::load(ranged_tmp))
        .flatten();
    if existing.is_none() {
        // Junk from an interrupted run whose sidecar is gone or unreadable: it cannot be resumed by
        // either mode, and leaving it would let a later run mistake it for progress. The sidecar
        // goes too, since without its file it describes nothing.
        if ranged_tmp.exists() {
            debug!("{label}: ranged partial has no usable plan; starting over");
        }
        parts::discard(ranged_tmp);
        let stream_have = fs::metadata(&temps.stream).map(|m| m.len()).unwrap_or(0);
        // A capped download is a sub-1 MiB companion — one request is already the whole of it.
        if max_bytes.is_some() || stream_have > 0 || budget.available() == 0 {
            return Ok(None);
        }
    }

    let Some(probe) = ranged::probe(obj)? else {
        if existing.is_some() {
            // The partial is a grid of holes and no single-stream resume can use it.
            ranged::warn_restart(
                progress,
                &format!("{label}: the server no longer serves ranges; restarting as one stream"),
            );
            parts::discard(ranged_tmp);
        }
        return Ok(None);
    };
    if existing.is_none() && probe.total <= parts::chunk_bytes() {
        return Ok(None); // one cell: splitting it would be one request either way
    }

    let pb = progress::bar_in(progress, Some(probe.total), label, Unit::Bytes);
    let mut existing = existing;
    let mut restarted = false;
    loop {
        match ranged::run(obj, ranged_tmp, existing.take(), &probe, budget, &pb) {
            Ok(()) => return Ok(Some((ranged_tmp.to_path_buf(), pb))),
            // The object is not the one the partial belongs to — the case that must never become
            // "continue anyway". Throw those bytes away and try once from nothing.
            Err(RangedError::Changed(why)) if !restarted => {
                restarted = true;
                parts::discard(ranged_tmp);
                ranged::warn_restart(
                    progress,
                    &format!("{label}: {why}; the partial was discarded, downloading it again"),
                );
                if probe.total <= parts::chunk_bytes() {
                    progress.remove(&pb);
                    return Ok(None);
                }
            }
            // Twice in one run means the object is being rewritten under us; stop rather than loop.
            Err(RangedError::Changed(why)) => {
                parts::discard(ranged_tmp);
                pb.abandon_with_message(format!("⚠ {label} keeps changing upstream"));
                return Err(Error::Other(format!("{label}: {why} (again) — giving up")));
            }
            Err(RangedError::NoRanges(why)) => {
                ranged::warn_restart(
                    progress,
                    &format!("{label}: {why}; falling back to a single stream"),
                );
                parts::discard(ranged_tmp);
                progress.remove(&pb);
                return Ok(None);
            }
            Err(RangedError::Fatal(e)) => {
                pb.abandon_with_message(format!("⚠ {label} interrupted (resumable)"));
                return Err(e);
            }
        }
    }
}

/// The single-stream download: one connection, appending, resumed from the temp file's own length.
/// Unchanged by ranged parallelism on purpose — it is what every server that will not range, every
/// companion file, every object below the split threshold and `hub.pull_jobs = 1` still get.
fn fetch_stream(
    obj: &Object,
    temps: &Temps,
    max_bytes: Option<u64>,
    progress: &MultiProgress,
) -> Result<(PathBuf, ProgressBar)> {
    let (url, label) = (obj.url, obj.label);
    let (tmp, meta) = (temps.stream.as_path(), temps.meta.as_path());
    parts::discard(&temps.ranged); // nothing here can use a ranged partial
    let have = fs::metadata(tmp).map(|m| m.len()).unwrap_or(0);
    // Validator captured when THIS partial was first written; only meaningful with bytes on disk.
    let validator = (have > 0)
        .then(|| fs::read_to_string(meta).ok())
        .flatten()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    debug!("GET {url}{}", if have > 0 { " (resume)" } else { "" });
    let mut req = download_client()?.get(url);
    if let Some(t) = token() {
        req = req.bearer_auth(t);
    }
    // Resume with `If-Range`: if the object changed since the partial was written, the server
    // ignores the Range and sends a full 200 → we restart clean instead of splicing new bytes onto a
    // stale prefix (an undetectable corruption without the end-of-body sha check).
    if let Some((range, if_range)) = resume_headers(have, validator.as_deref()) {
        req = req.header(reqwest::header::RANGE, range);
        if let Some(v) = if_range {
            req = req.header(reqwest::header::IF_RANGE, v);
        }
    }
    let resp = req
        .send()
        .map_err(|e| Error::Other(format!("HTTP request: {e}")))?;
    if !resp.status().is_success() {
        return Err(Error::Other(format!(
            "download failed: HTTP {}",
            resp.status()
        )));
    }
    // The server honours the Range only with 206; on 200 it sends the whole file → restart clean.
    let resuming = have > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let remaining = resp.content_length();
    let total = remaining.map(|r| if resuming { have + r } else { r });

    // Cheap pre-flight: refuse an announced size over the cap before opening the temp file at all.
    // Advisory only — the loop below is the enforcement (see the fn docs).
    if let (Some(cap), Some(t)) = (max_bytes, total) {
        if over_cap(t, max_bytes) {
            return Err(Error::Other(format!(
                "{label}: advertised size {t} bytes exceeds the {cap}-byte cap"
            )));
        }
    }

    // Persist the validator for a FUTURE resume — before streaming, so an interrupt mid-body still
    // leaves a usable `If-Range` for the next attempt. On a 206 the stored validator still matches
    // (the server accepted it), so only refresh it on a fresh/200 body.
    if !resuming {
        match response_validator(resp.headers()) {
            Some(v) => {
                let _ = fs::write(meta, v);
            }
            None => {
                let _ = fs::remove_file(meta);
            }
        }
    }

    let mut file = if resuming {
        // Through `suspend`, like every log line emitted while bars are on screen: `tracing` writes
        // straight to stderr and would otherwise land in the middle of the bar block, which is the
        // legibility problem several concurrent downloads make N times worse.
        progress.suspend(|| info!("resuming {label} at {have} bytes"));
        fs::OpenOptions::new()
            .append(true)
            .open(tmp)
            .map_err(Error::from)?
    } else {
        fs::File::create(tmp).map_err(Error::from)? // truncates any stale partial (changed object)
    };
    let start = if resuming { have } else { 0 };

    let pb = progress::bar_in(progress, total, label, Unit::Bytes);
    pb.set_position(start);
    // A resumed bar jumps from 0 to `start` in no time at all, and the rate/ETA estimator counts
    // that jump as transferred bytes — a 4 GiB partial reads as "77 TiB/s ETA 0s" until real bytes
    // dilute it. Reset the estimator so the numbers describe THIS transfer from its first byte.
    pb.reset_eta();

    match stream_into(resp, &mut file, &pb, max_bytes, start) {
        Ok(()) => {}
        // Over the cap: the body is not something we will ever accept, so the partial is DELETED
        // rather than kept. Keeping it would leave up-to-cap bytes in the cache forever and make
        // the next attempt resume from a prefix that is already at the limit.
        Err(StreamError::TooLarge { cap }) => {
            pb.abandon_with_message(format!("⚠ {label} exceeds {cap} bytes"));
            drop(file);
            let _ = fs::remove_file(tmp);
            let _ = fs::remove_file(meta);
            return Err(Error::Other(format!(
                "{label}: body exceeds the {cap}-byte cap; download aborted"
            )));
        }
        // A transport failure is transient: keep the partial so the next call resumes it.
        Err(StreamError::Io(e)) => {
            pb.abandon_with_message(format!("⚠ {label} interrupted (resumable)"));
            return Err(Error::Other(format!(
                "download failed (partial kept for resume): {e}"
            )));
        }
    }
    drop(file); // flush + close before re-reading for the digest
    Ok((tmp.to_path_buf(), pb))
}

/// Build the resume request directives: the `Range` value and an optional `If-Range` value. Returns
/// `None` when there is nothing on disk to resume (`have == 0`). `If-Range` is omitted when no
/// validator was stored (a partial from before this feature) — the server may then splice, but the
/// end-of-download sha256 verification still catches it.
fn resume_headers(have: u64, validator: Option<&str>) -> Option<(String, Option<String>)> {
    if have == 0 {
        return None;
    }
    Some((format!("bytes={have}-"), validator.map(str::to_string)))
}

/// The value to persist as the `If-Range` validator for a partial: the strong `ETag` if present,
/// else `Last-Modified`. Either is an opaque object identity the server compares on the next resume.
fn response_validator(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(reqwest::header::ETAG)
        .or_else(|| headers.get(reqwest::header::LAST_MODIFIED))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
}

/// An advisory exclusive `flock` on a lockfile, released when dropped. Serializes concurrent
/// downloads of the same blob so two processes can't interleave writes into the shared temp.
#[cfg(not(target_os = "windows"))]
struct FileLock {
    _file: fs::File,
}

#[cfg(not(target_os = "windows"))]
impl FileLock {
    fn acquire(path: &Path) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
            .map_err(Error::from)?;
        // Blocks until any other holder releases. `flock` is process-associated and auto-releases if
        // the holder dies (crash-safe — no stale lock).
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        if rc != 0 {
            return Err(Error::Other(format!(
                "flock {path:?}: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(FileLock { _file: file })
    }
}

#[cfg(not(target_os = "windows"))]
impl Drop for FileLock {
    fn drop(&mut self) {
        // Closing the fd releases the lock; the explicit unlock is belt-and-suspenders.
        unsafe { libc::flock(self._file.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[cfg(target_os = "windows")]
struct FileLock {
    _file: fs::File,
}

#[cfg(target_os = "windows")]
impl FileLock {
    fn acquire(path: &Path) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
            .map_err(Error::from)?;
        lock_file_exclusive(&file)
            .map_err(|e| Error::Other(format!("LockFileEx {path:?}: {e}")))?;
        Ok(FileLock { _file: file })
    }
}

#[cfg(target_os = "windows")]
impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = unlock_file(&self._file);
    }
}

#[cfg(target_os = "windows")]
#[repr(C)]
union OverlappedOffset {
    offset: OffsetPair,
    pointer: *mut std::ffi::c_void,
}

#[cfg(target_os = "windows")]
#[repr(C)]
#[derive(Copy, Clone)]
struct OffsetPair {
    offset: u32,
    offset_high: u32,
}

#[cfg(target_os = "windows")]
#[repr(C)]
struct Overlapped {
    internal: usize,
    internal_high: usize,
    offset_or_pointer: OverlappedOffset,
    h_event: *mut std::ffi::c_void,
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LockFileEx(
        h_file: *mut std::ffi::c_void,
        dw_flags: u32,
        dw_reserved: u32,
        n_number_of_bytes_to_lock_low: u32,
        n_number_of_bytes_to_lock_high: u32,
        lp_overlapped: *mut Overlapped,
    ) -> i32;
    fn UnlockFileEx(
        h_file: *mut std::ffi::c_void,
        dw_reserved: u32,
        n_number_of_bytes_to_unlock_low: u32,
        n_number_of_bytes_to_unlock_high: u32,
        lp_overlapped: *mut Overlapped,
    ) -> i32;
}

#[cfg(target_os = "windows")]
fn zero_overlapped() -> Overlapped {
    Overlapped {
        internal: 0,
        internal_high: 0,
        offset_or_pointer: OverlappedOffset {
            offset: OffsetPair {
                offset: 0,
                offset_high: 0,
            },
        },
        h_event: std::ptr::null_mut(),
    }
}

#[cfg(target_os = "windows")]
fn lock_file_exclusive(file: &fs::File) -> std::io::Result<()> {
    use std::os::windows::io::AsRawHandle;

    const LOCKFILE_EXCLUSIVE_LOCK: u32 = 0x0000_0002;
    const LOCK_LEN_LOW: u32 = u32::MAX;
    const LOCK_LEN_HIGH: u32 = u32::MAX;

    let mut overlapped = zero_overlapped();
    let ok = unsafe {
        LockFileEx(
            file.as_raw_handle(),
            LOCKFILE_EXCLUSIVE_LOCK,
            0,
            LOCK_LEN_LOW,
            LOCK_LEN_HIGH,
            &mut overlapped,
        )
    };
    if ok == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(target_os = "windows")]
fn unlock_file(file: &fs::File) -> std::io::Result<()> {
    use std::os::windows::io::AsRawHandle;

    const LOCK_LEN_LOW: u32 = u32::MAX;
    const LOCK_LEN_HIGH: u32 = u32::MAX;

    let mut overlapped = zero_overlapped();
    let ok = unsafe {
        UnlockFileEx(
            file.as_raw_handle(),
            0,
            LOCK_LEN_LOW,
            LOCK_LEN_HIGH,
            &mut overlapped,
        )
    };
    if ok == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Assert the downloaded digest `hex` matches the `expected` LFS sha256 (case-insensitive). `None`
/// means no digest was available (non-LFS file) → verification is skipped.
fn verify_sha(label: &str, hex: &str, expected: Option<&str>) -> Result<()> {
    match expected {
        Some(exp) if !hex.eq_ignore_ascii_case(exp) => Err(Error::Other(format!(
            "sha256 mismatch for {label}: expected {exp}, got {hex} — corrupt download discarded"
        ))),
        _ => Ok(()),
    }
}

/// Read an existing file fully through `hasher` (to continue a resumed digest).
fn hash_file(path: &Path, hasher: &mut Sha256) -> Result<()> {
    let mut f = fs::File::open(path).map_err(Error::from)?;
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf).map_err(Error::from)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(())
}

/// The size-cap predicate, shared by the `Content-Length` pre-flight and the streaming loop so the
/// two can never disagree about the boundary. `None` = uncapped (the model-blob path). The
/// comparison is STRICTLY greater: a body of exactly `cap` bytes is accepted, so a cap chosen as
/// "the largest size I will tolerate" means what it says.
fn over_cap(total: u64, cap: Option<u64>) -> bool {
    matches!(cap, Some(c) if total > c)
}

/// Why the streaming loop fails, because the two outcomes need OPPOSITE cleanup: an I/O error is
/// transient and the partial is kept so the next call resumes it, while an over-cap body is a
/// response we will never accept and its partial must be deleted. Collapsing both into
/// `io::Error` would leave up-to-cap junk in the blob dir after every rejected download.
enum StreamError {
    Io(std::io::Error),
    TooLarge { cap: u64 },
}

impl From<std::io::Error> for StreamError {
    fn from(e: std::io::Error) -> Self {
        StreamError::Io(e)
    }
}

/// Stream the response body into `file`, advancing the progress bar. The digest is computed in a
/// single final pass over the completed file (see [`download_to_blob`]), not here.
///
/// `cap` (when set) bounds the file's FINAL size — `written` seeds the counter with the bytes a
/// resume already has on disk, so a cap can't be walked past one resumed chunk at a time. The
/// check is per read, BEFORE the buffer is written, so at most one 64 KiB chunk beyond the limit
/// is ever touched and nothing over-long reaches the disk. Doing this here rather than after
/// `stream_into` returns is the whole point: a hostile or broken endpoint answering an unverified
/// request with an endless body must be cut off mid-transfer, not measured once it has already
/// filled the user's cache.
fn stream_into(
    mut resp: Response,
    file: &mut fs::File,
    pb: &ProgressBar,
    cap: Option<u64>,
    written: u64,
) -> std::result::Result<(), StreamError> {
    let mut total = written;
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        total = total.saturating_add(n as u64);
        if over_cap(total, cap) {
            return Err(StreamError::TooLarge {
                cap: cap.unwrap_or(u64::MAX),
            });
        }
        file.write_all(&buf[..n])?;
        pb.inc(n as u64);
    }
    file.flush()?;
    Ok(())
}

/// The per-download temp/lock stem for `label` (an HF `rfilename`): the readable [`sanitise`]d name
/// plus a short digest of the FULL original.
///
/// The digest is what makes it INJECTIVE, and that is the point. `sanitise` alone maps `/` to `_`,
/// so `UD-Q4_K_XL/m.gguf` and `UD-Q4_K_XL_m.gguf` — both plausible in one repo now that
/// subdirectory `rfilename`s are supported — collide on the same `.dl-` partial and the same
/// `.lock`. The `flock` serialises them so there is no interleaved write, but the SECOND download
/// then "resumes" onto the first file's partial bytes. That splice is caught by the final sha256
/// only for an LFS file; a non-LFS file has no `expected_sha` and its verification is skipped
/// entirely, so the corrupt splice would be committed as a blob. The readable prefix is kept so a
/// leftover partial in `blobs/` is still identifiable by eye.
fn temp_stem(label: &str) -> String {
    let digest = Sha256::digest(label.as_bytes());
    let short: String = digest[..4].iter().map(|b| format!("{b:02x}")).collect();
    format!("{}-{short}", sanitise(label))
}

/// `label`'s partial download, named relative to the blobs dir. The `.meta` validator and the
/// `.lock` hang off it, so this is the ONE place the three names are derived from — and it is what
/// a test plants a half-finished download at.
pub(crate) fn partial_name(label: &str) -> String {
    format!(".dl-{}", temp_stem(label))
}

/// `label`'s RANGED partial download, and the stem its `.plan` sidecar hangs off.
///
/// A different name from [`partial_name`], and that is the point rather than tidiness. The two
/// partials mean incompatible things: a stream partial is a prefix, so its LENGTH is how much of
/// the file it holds, while a ranged partial is a sparse file whose length says nothing at all —
/// its last chunk landing first makes it full-length with holes. Share one name and a run that
/// loses the sidecar reads a hole-ridden file as a complete prefix and resumes past it, which is a
/// corrupt blob that only the sha256 gate stands between and the model directory — and a non-LFS
/// file has no gate. Separate names make that confusion unrepresentable.
pub(crate) fn ranged_partial_name(label: &str) -> String {
    format!(".dlr-{}", temp_stem(label))
}

/// Replace characters unsafe in a filename with `_`.
fn sanitise(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pull::MAX_COMPANION_BYTES;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    #[test]
    fn resume_headers_build() {
        // Nothing on disk → no resume directives.
        assert_eq!(resume_headers(0, Some("etag")), None);
        assert_eq!(resume_headers(0, None), None);
        // Bytes on disk with a stored validator → Range + If-Range.
        assert_eq!(
            resume_headers(100, Some("\"abc\"")),
            Some(("bytes=100-".to_string(), Some("\"abc\"".to_string())))
        );
        // Bytes but no validator (pre-feature partial) → Range only.
        assert_eq!(
            resume_headers(100, None),
            Some(("bytes=100-".to_string(), None))
        );
    }

    #[test]
    fn response_validator_prefers_etag() {
        use reqwest::header::{HeaderMap, HeaderValue, ETAG, LAST_MODIFIED};
        let mut h = HeaderMap::new();
        assert_eq!(response_validator(&h), None);
        h.insert(
            LAST_MODIFIED,
            HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
        );
        assert_eq!(
            response_validator(&h).as_deref(),
            Some("Wed, 21 Oct 2026 07:28:00 GMT")
        );
        h.insert(ETAG, HeaderValue::from_static("\"deadbeef\""));
        assert_eq!(response_validator(&h).as_deref(), Some("\"deadbeef\""));
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn file_lock_is_exclusive() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("blob.lock");
        let guard = FileLock::acquire(&path).unwrap();
        // A second exclusive lock on the same file (separate fd) must NOT be grantable while held.
        let other = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .unwrap();
        let rc = unsafe { libc::flock(other.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_ne!(rc, 0, "second flock should fail while the first is held");
        // After the first releases, the lock is grantable again.
        drop(guard);
        let rc = unsafe { libc::flock(other.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(rc, 0, "flock should succeed once the holder drops");
        unsafe { libc::flock(other.as_raw_fd(), libc::LOCK_UN) };
    }

    /// `sanitise` maps `/` to `_`, so the sanitised name ALONE is not a unique temp/lock key once
    /// subdirectory filenames exist: `a/b.gguf` and `a_b.gguf` would share a `.dl-` partial and one
    /// would resume onto the other's bytes (undetectable for a non-LFS file, which is never
    /// sha-verified). The appended digest of the full name is what separates them.
    #[test]
    fn temp_stem_is_collision_free() {
        assert_ne!(temp_stem("a/b.gguf"), temp_stem("a_b.gguf"));
        assert_eq!(sanitise("a/b.gguf"), sanitise("a_b.gguf")); // …the collision it protects against
        assert_eq!(temp_stem("a/b.gguf"), temp_stem("a/b.gguf")); // stable → resume still works
        assert!(
            temp_stem("UD-Q4_K_XL/m.gguf").starts_with("UD-Q4_K_XL_m.gguf-"),
            "readable prefix is kept for debuggability: {}",
            temp_stem("UD-Q4_K_XL/m.gguf")
        );
    }

    /// The cap boundary, pinned once for both the `Content-Length` pre-flight and the streaming
    /// loop (they share this predicate on purpose — a mismatch between them would mean a body
    /// rejected in one place and accepted in the other).
    #[test]
    fn over_cap_boundary() {
        // Uncapped (the model-blob path): nothing is ever too large.
        assert!(!over_cap(0, None));
        assert!(!over_cap(u64::MAX, None));
        // Strictly greater — exactly at the cap is accepted.
        assert!(!over_cap(1023, Some(1024)));
        assert!(!over_cap(1024, Some(1024)));
        assert!(over_cap(1025, Some(1024)));
        // A resumed prefix already at the cap means the very next byte trips it.
        assert!(over_cap(MAX_COMPANION_BYTES + 1, Some(MAX_COMPANION_BYTES)));
    }

    #[test]
    fn verify_sha_gate() {
        // Match (case-insensitive) passes.
        assert!(verify_sha("f", SHA_A, Some(SHA_A)).is_ok());
        assert!(verify_sha("f", SHA_A, Some(&SHA_A.to_ascii_uppercase())).is_ok());
        // No expected digest → skipped (non-LFS best-effort).
        assert!(verify_sha("f", SHA_A, None).is_ok());
        // Mismatch fails loudly.
        let other = "b".repeat(64);
        assert!(verify_sha("f", SHA_A, Some(&other)).is_err());
    }
}
