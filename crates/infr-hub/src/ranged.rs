//! Intra-file parallelism: ONE object fetched over several connections at once, as a grid of
//! byte ranges, with [`crate::parts`]'s sidecar recording which cells have landed.
//!
//! The reason this exists is the same measurement that motivated fetching several shards at once —
//! a single connection to the HF CDN, not the link, is what caps a download — except that a model
//! shipped as one file (`unsloth/DeepSeek-V3.2-GGUF`'s 161 GiB `UD-TQ1_0`) has no shards to spread
//! over connections. So the FILE is spread instead: worker `k` asks for `Range: bytes=A-B` and
//! `pwrite`s the reply at offset `A`, and the assembled file is byte-identical to what one stream
//! would have written.
//!
//! Three things are load-bearing, and all three are about not producing a plausible-sized file
//! assembled from two different uploads — the failure that once read as "this quant is broken" for
//! a day (memory `verify-gguf-downloads`):
//!
//!   * **Identity is checked before a resumed byte is written.** The sidecar records HF's LFS
//!     sha256 for the object it belongs to, and a run that finds a different one throws the partial
//!     away instead of continuing it. That is a content hash, so unlike an `ETag` it cannot differ
//!     merely because a different CDN edge answered.
//!   * **Every chunk request carries `If-Range` and must come back `206`** with a `Content-Range`
//!     naming exactly the bytes asked for, out of a total that still matches the plan. A `200` means
//!     the server declined the range and is offering the whole object from byte 0 — never something
//!     to write into the middle of a file.
//!   * **The sha256 gate is untouched.** Everything here is about where bytes land; whether the
//!     assembled file is the one HF advertises is still decided once, at the end, by
//!     [`crate::download`], and a mismatch still discards the temp and refuses to link it.

use crate::download::Object;
use crate::http::{api_client, download_client, token, ConnBudget, Permit};
use crate::parts::{self, Plan};
use indicatif::{MultiProgress, ProgressBar};
use infr_core::error::{Error, Result};
use reqwest::header::{
    ACCEPT_RANGES, CONTENT_LENGTH, CONTENT_RANGE, ETAG, IF_RANGE, LAST_MODIFIED, RANGE,
};
use reqwest::StatusCode;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use tracing::{debug, warn};

/// What a one-byte probe learned about the object: how big it is, and the validator to present on
/// each range request.
pub(crate) struct Probe {
    pub(crate) total: u64,
    pub(crate) validator: Option<String>,
}

/// Ask the object how big it is and whether it can be fetched in pieces, with a `HEAD`.
///
/// `Ok(None)` is the graceful fallback and covers everything uncertain: a server that does not
/// advertise `Accept-Ranges: bytes`, one that will not say how long the object is, an error status.
/// The object is then downloaded as a single stream exactly as it was before ranges existed.
///
/// A `HEAD` rather than a one-byte ranged `GET` because it costs no body at all — a server that
/// declines the range would answer a ranged `GET` with the WHOLE object, and the client abandoning
/// that response has still made the origin start sending 161 GiB down a socket it is about to close.
/// The cost of asking this way is that `Accept-Ranges` is an advertisement rather than a
/// demonstration; the demonstration isevery chunk request, each of which must come back `206` naming
/// the bytes it asked for, and a server that advertised ranges and then declines one falls back to
/// the single stream from there.
pub(crate) fn probe(obj: &Object) -> Result<Option<Probe>> {
    let (url, label) = (obj.url, obj.label);
    let mut req = api_client()?.head(url);
    if let Some(t) = token() {
        req = req.bearer_auth(t);
    }
    let resp = req
        .send()
        .map_err(|e| Error::Other(format!("HEAD {url}: {e}")))?;
    if !resp.status().is_success() {
        debug!("{label}: range probe answered HTTP {}", resp.status());
        return Ok(None);
    }
    let ranges = resp
        .headers()
        .get(ACCEPT_RANGES)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|t| t.trim().eq_ignore_ascii_case("bytes")));
    if !ranges {
        debug!("{label}: server does not advertise byte ranges; not splitting");
        return Ok(None);
    }
    // The HEADER, not `Response::content_length()` — that reports the length of the body being
    // received, which for a `HEAD` is zero by definition and would read as "no length advertised".
    let Some(total) = resp
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|n| *n > 0)
    else {
        debug!("{label}: no length advertised; not splitting");
        return Ok(None);
    };
    Ok(Some(Probe {
        total,
        validator: validator_of(resp.headers()),
    }))
}

/// The `If-Range` validator a response offers: the `ETag` if present, else `Last-Modified`.
fn validator_of(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(ETAG)
        .or_else(|| headers.get(LAST_MODIFIED))
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .filter(|s| !s.is_empty())
}

/// `bytes 200-299/1000` → `(200, 299, 1000)`. `None` for any other shape, including the `*/N`
/// unsatisfied form and an unknown total (`.../*`), neither of which can plan a grid.
fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let rest = value.trim().strip_prefix("bytes ")?;
    let (span, total) = rest.split_once('/')?;
    let (first, last) = span.split_once('-')?;
    Some((
        first.trim().parse().ok()?,
        last.trim().parse().ok()?,
        total.trim().parse().ok()?,
    ))
}

/// Why a ranged download stopped, split by what the caller must DO about it — the three outcomes
/// need three different treatments of the bytes already on disk, and collapsing them is how a
/// partial from a since-replaced upload gets continued.
pub(crate) enum RangedError {
    /// The object is not the one this partial belongs to (a re-upload, or a resize). The caller
    /// must DISCARD the partial and start over — continuing it is the splice.
    Changed(String),
    /// The server will not serve ranges after all. The caller must discard the partial — it has
    /// holes in it and no single-stream resume can use one — and fall back to a single stream.
    NoRanges(String),
    /// Transport or I/O. The partial and its sidecar are KEPT, so the next run resumes the cells
    /// that did land.
    Fatal(Error),
}

/// Fetch `obj` into `tmp` over several connections, resuming `existing` when it describes this same
/// object.
///
/// The partial is bound to [`Object::oid`] — HF's advertised LFS sha256 — which is what decides
/// whether `existing` may be continued at all. `budget` is the pull's whole connection allowance:
/// this call takes what is spare and gives it back as its workers finish, so the total in flight
/// across every file of the model stays at `hub.pull_jobs`.
pub(crate) fn run(
    obj: &Object,
    tmp: &Path,
    existing: Option<Plan>,
    probe: &Probe,
    budget: &ConnBudget,
    pb: &ProgressBar,
) -> std::result::Result<(), RangedError> {
    let label = obj.label;
    // Both failures below leave the partial for the CALLER to discard: `Changed` can also come out
    // of a worker mid-download, and one owner of "throw these bytes away" is one place to get it
    // right rather than two places to get it inconsistent.
    let plan = reconcile(existing, probe, obj.oid).map_err(RangedError::Changed)?;
    // The temp file first, then the sidecar that describes it — never a plan pointing at a file
    // that does not exist, which the next run would have to reason about.
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(tmp)
        .map_err(|e| RangedError::Fatal(Error::from(e)))?;
    parts::save(tmp, &plan).map_err(RangedError::Fatal)?;

    pb.set_position(plan.completed_bytes());
    // A resume jumps the bar forward instantly, and the rate estimator would read that jump as
    // transferred bytes; the numbers must describe THIS transfer.
    pb.reset_eta();

    // One connection per remaining chunk at most, and never more than the budget has spare. The
    // calling thread is one of the workers, so it asks for one fewer than the connections it wants.
    let permits = budget.acquire_up_to(plan.remaining().saturating_sub(1));
    let conns = permits.len() + 1;
    let job = Job {
        url: obj.url,
        label,
        tmp,
        file,
        size: plan.size,
        chunk: plan.chunk,
        chunks: plan.chunks(),
        validator: plan.validator.clone(),
        plan: Mutex::new(plan),
        cursor: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
        err: Mutex::new(None),
        pb,
    };
    pb.suspend(|| {
        debug!(
            "{label}: {} of {} chunks left, {conns} connection(s)",
            job.plan.lock().unwrap().remaining(),
            job.chunks
        )
    });

    let job = &job;
    std::thread::scope(|scope| {
        for permit in permits {
            spawn_worker(scope, job, budget, permit);
        }
        worker(scope, job, budget);
    });

    if let Some(e) = job.err.lock().unwrap_or_else(|e| e.into_inner()).take() {
        return Err(e);
    }
    let plan = job.plan.lock().unwrap_or_else(|e| e.into_inner());
    if !plan.all_done() {
        return Err(RangedError::Fatal(Error::Other(format!(
            "{label}: ranged download ended with {} chunk(s) missing",
            plan.remaining()
        ))));
    }
    // The grid tiles the file exactly, so a complete plan means a complete file — assert it rather
    // than trust it, because the alternative is committing a blob with a hole in it.
    let got = fs::metadata(tmp)
        .map(|m| m.len())
        .map_err(|e| RangedError::Fatal(Error::from(e)))?;
    if got != plan.size {
        return Err(RangedError::Fatal(Error::Other(format!(
            "{label}: assembled {got} bytes but the plan says {}",
            plan.size
        ))));
    }
    Ok(())
}

/// Decide whether `existing` may be resumed against the object the probe just described, returning
/// the plan to run — or the reason the partial has to go.
///
/// The identity test prefers the LFS oid over the `ETag` deliberately: the oid is a hash of the
/// content, so it answers "are these the same bytes?" directly and identically from every edge,
/// whereas an `ETag` mismatch can only say "something the server tracks differs". The `ETag` is
/// still used when there is no oid (a non-LFS object), and is refreshed either way because
/// `If-Range` has to be presented in the server's own terms.
fn reconcile(
    existing: Option<Plan>,
    probe: &Probe,
    expected_sha: Option<&str>,
) -> std::result::Result<Plan, String> {
    let Some(mut plan) = existing else {
        return Ok(Plan::fresh(
            probe.total,
            parts::chunk_bytes(),
            expected_sha.map(str::to_string),
            probe.validator.clone(),
        ));
    };
    if plan.size != probe.total {
        return Err(format!(
            "the object is now {} bytes, the partial was planned for {}",
            probe.total, plan.size
        ));
    }
    match (plan.oid.as_deref(), expected_sha) {
        (Some(had), Some(now)) if !had.eq_ignore_ascii_case(now) => {
            return Err(format!("the LFS sha256 changed ({had} → {now})"));
        }
        // No oid on either side to compare: fall back to the opaque validator, which is weaker (an
        // edge that reports a different one costs a restart) but is all a non-LFS object has.
        (None, None) => match (plan.validator.as_deref(), probe.validator.as_deref()) {
            (Some(had), Some(now)) if had != now => {
                return Err(format!("the validator changed ({had} → {now})"));
            }
            _ => {}
        },
        _ => {}
    }
    if plan.oid.is_none() {
        plan.oid = expected_sha.map(str::to_string);
    }
    plan.validator = probe.validator.clone();
    Ok(plan)
}

/// Everything the chunk workers share. Immutable but for the plan (behind its lock), the claim
/// cursor and the first error.
struct Job<'a> {
    url: &'a str,
    label: &'a str,
    tmp: &'a Path,
    file: fs::File,
    size: u64,
    chunk: u64,
    chunks: usize,
    validator: Option<String>,
    plan: Mutex<Plan>,
    cursor: AtomicUsize,
    stop: AtomicBool,
    err: Mutex<Option<RangedError>>,
    pb: &'a ProgressBar,
}

impl Job<'_> {
    /// Chunk `i`'s `(start, len)`, without taking the plan's lock — the grid is fixed for the whole
    /// run, so the numbers cannot change under a worker.
    fn range(&self, i: usize) -> (u64, u64) {
        let start = (i as u64) * self.chunk;
        (start, self.chunk.min(self.size - start))
    }
}

#[cfg(not(target_os = "windows"))]
fn write_all_at(file: &fs::File, buf: &[u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;

    file.write_all_at(buf, offset)
}

#[cfg(target_os = "windows")]
fn write_all_at(file: &fs::File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    use std::io::ErrorKind;
    use std::os::windows::fs::FileExt;

    while !buf.is_empty() {
        let n = file.seek_write(buf, offset)?;
        if n == 0 {
            return Err(io::Error::new(
                ErrorKind::WriteZero,
                "failed to write the full chunk at its offset",
            ));
        }
        buf = &buf[n..];
        offset += n as u64;
    }
    Ok(())
}

/// Start one more chunk worker, holding `permit` for as long as it runs. The permit rides with the
/// thread so the budget gets it back the moment this worker runs out of chunks to claim, even if it
/// returns early or panics.
fn spawn_worker<'s, 'e>(
    scope: &'s std::thread::Scope<'s, 'e>,
    job: &'s Job<'s>,
    budget: &'s ConnBudget,
    permit: Permit<'s>,
) {
    scope.spawn(move || {
        let _permit = permit;
        worker(scope, job, budget);
    });
}

/// Claim cells until the grid runs out or somebody fails. One shared cursor, so each chunk is
/// fetched by exactly one worker and the file is still filled roughly front to back.
///
/// Between chunks a worker also tries to GROW the fan-out. Permits come free while a download runs
/// — most often because another file of the model finished and its worker gave one back — and a
/// download that fixed its connection count at its first chunk would leave them idle. That is the
/// tail of every shard set: once fewer files remain than there are connections, the last few would
/// otherwise crawl on one connection each while the rest of the allowance sits unused. Growth is
/// bounded by the same budget as everything else, so it can only ever fill the allowance, never
/// exceed it.
fn worker<'s>(scope: &'s std::thread::Scope<'s, '_>, job: &'s Job<'s>, budget: &'s ConnBudget) {
    while !job.stop.load(Ordering::Relaxed) {
        // Only worth another connection if there is a cell for it to claim after this one.
        if job.cursor.load(Ordering::Relaxed) + 1 < job.chunks {
            if let Some(permit) = budget.try_acquire() {
                spawn_worker(scope, job, budget, permit);
            }
        }
        let i = job.cursor.fetch_add(1, Ordering::Relaxed);
        if i >= job.chunks {
            return;
        }
        if job.plan.lock().unwrap().is_done(i) {
            continue; // landed in an earlier run
        }
        if let Err(e) = fetch_chunk(job, i) {
            job.stop.store(true, Ordering::Relaxed);
            let mut slot = job.err.lock().unwrap();
            if slot.is_none() {
                *slot = Some(e); // first failure wins; the rest are consequences of `stop`
            }
            return;
        }
    }
}

/// Fetch chunk `i` and record it as landed — in that order, and only if every byte of it arrived.
fn fetch_chunk(job: &Job, i: usize) -> std::result::Result<(), RangedError> {
    let (start, len) = job.range(i);
    let end = start + len - 1;
    let mut req = download_client()
        .map_err(RangedError::Fatal)?
        .get(job.url)
        .header(RANGE, format!("bytes={start}-{end}"));
    if let Some(v) = &job.validator {
        req = req.header(IF_RANGE, v.clone());
    }
    if let Some(t) = token() {
        req = req.bearer_auth(t);
    }
    let mut resp = req.send().map_err(|e| {
        RangedError::Fatal(Error::Other(format!(
            "{}: chunk {i} ({start}..={end}): {e} (partial kept for resume)",
            job.label
        )))
    })?;
    if resp.status() == StatusCode::OK {
        // The range was declined and the whole object is on its way — writing that at `start` is
        // exactly the splice this design exists to prevent.
        return Err(RangedError::NoRanges(format!(
            "{}: the server answered chunk {i} with the whole object",
            job.label
        )));
    }
    if resp.status() != StatusCode::PARTIAL_CONTENT {
        return Err(RangedError::Fatal(Error::Other(format!(
            "{}: chunk {i}: HTTP {}",
            job.label,
            resp.status()
        ))));
    }
    // The reply must describe the bytes that were asked for, out of the size the plan was built on.
    match resp
        .headers()
        .get(CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_content_range)
    {
        Some((a, b, total)) if a == start && b == end && total == job.size => {}
        Some((a, b, total)) => {
            return Err(RangedError::Changed(format!(
                "{}: asked for bytes {start}-{end}/{} and was served {a}-{b}/{total}",
                job.label, job.size
            )));
        }
        None => {
            return Err(RangedError::NoRanges(format!(
                "{}: chunk {i} came back 206 with no usable Content-Range",
                job.label
            )));
        }
    }
    if let Some(n) = resp.content_length() {
        if n != len {
            return Err(RangedError::Fatal(Error::Other(format!(
                "{}: chunk {i} announced {n} bytes, expected {len}",
                job.label
            ))));
        }
    }

    let mut buf = vec![0u8; 1 << 16];
    let mut at = start;
    let mut left = len;
    while left > 0 {
        let want = buf.len().min(left as usize);
        let n = resp.read(&mut buf[..want]).map_err(|e| {
            RangedError::Fatal(Error::Other(format!(
                "{}: chunk {i} at {at}: {e} (partial kept for resume)",
                job.label
            )))
        })?;
        if n == 0 {
            // A short body would otherwise leave a hole under a chunk marked done.
            return Err(RangedError::Fatal(Error::Other(format!(
                "{}: chunk {i} ended {left} bytes early (partial kept for resume)",
                job.label
            ))));
        }
        write_all_at(&job.file, &buf[..n], at).map_err(|e| RangedError::Fatal(Error::from(e)))?;
        at += n as u64;
        left -= n as u64;
        job.pb.inc(n as u64);
    }
    // Data first, then the record of it: a sidecar that outlived its bytes across a crash would
    // claim a cell of zeros had landed, and only the sha256 gate would notice — after the whole
    // object had been downloaded again for nothing.
    job.file
        .sync_data()
        .map_err(|e| RangedError::Fatal(Error::from(e)))?;
    let mut plan = job.plan.lock().unwrap();
    plan.mark_done(i);
    parts::save(job.tmp, &plan).map_err(RangedError::Fatal)?;
    Ok(())
}

/// Log a restart where the user will see it: a discarded partial is minutes or hours of transfer,
/// and the reason it happened is the difference between "the repo was updated" and "something is
/// wrong with this mirror".
pub(crate) fn warn_restart(progress: &MultiProgress, why: &str) {
    progress.suspend(|| warn!("{why}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_range_parses_the_forms_that_matter() {
        assert_eq!(parse_content_range("bytes 0-0/1000"), Some((0, 0, 1000)));
        assert_eq!(
            parse_content_range("bytes 200-299/173458321408"),
            Some((200, 299, 173458321408))
        );
        // Shapes that cannot plan a grid.
        for bad in [
            "bytes */1000", // unsatisfied
            "bytes 0-99/*", // unknown total
            "items 0-99/100",
            "0-99/100",
            "bytes 0-99",
            "",
        ] {
            assert!(parse_content_range(bad).is_none(), "accepted {bad:?}");
        }
    }

    fn probe_of(total: u64, validator: &str) -> Probe {
        Probe {
            total,
            validator: Some(validator.to_string()),
        }
    }

    /// The re-upload rule, which is the whole reason the sidecar records an oid: same size, same
    /// server, different content — resume it and the blob is half of one upload and half of
    /// another, at exactly the right length, and only the sha256 gate stands between that and the
    /// user's model directory.
    #[test]
    fn a_changed_oid_refuses_the_partial() {
        let old = "a".repeat(64);
        let new = "b".repeat(64);
        let plan = Plan::fresh(1000, 100, Some(old.clone()), Some("\"v1\"".into()));

        let same = reconcile(Some(plan.clone()), &probe_of(1000, "\"v1\""), Some(&old))
            .expect("the same object must resume");
        assert_eq!(same.oid.as_deref(), Some(old.as_str()));

        let err = reconcile(Some(plan.clone()), &probe_of(1000, "\"v1\""), Some(&new))
            .expect_err("a re-upload must NOT resume");
        assert!(err.contains("LFS sha256 changed"), "{err}");

        // A resize is the same call for the same reason, and is caught even when no oid is known.
        let err = reconcile(Some(plan.clone()), &probe_of(999, "\"v1\""), Some(&old))
            .expect_err("a resized object must NOT resume");
        assert!(err.contains("999"), "{err}");
    }

    /// Without an oid the opaque validator is all there is, so it decides — and it is refreshed on
    /// every run, because `If-Range` has to be presented in the server's current terms.
    #[test]
    fn the_validator_decides_when_there_is_no_oid() {
        let plan = Plan::fresh(1000, 100, None, Some("\"v1\"".into()));
        let err = reconcile(Some(plan.clone()), &probe_of(1000, "\"v2\""), None)
            .expect_err("a changed validator must not resume an unhashed object");
        assert!(err.contains("validator changed"), "{err}");
        // …but an oid, once known, outranks it: same content, new edge, new ETag → resume.
        let sha = "c".repeat(64);
        let hashed = Plan::fresh(1000, 100, Some(sha.clone()), Some("\"v1\"".into()));
        let ok = reconcile(Some(hashed), &probe_of(1000, "\"v2\""), Some(&sha)).unwrap();
        assert_eq!(
            ok.validator.as_deref(),
            Some("\"v2\""),
            "validator refreshed"
        );
    }

    /// A fresh plan takes its whole shape from the probe.
    #[test]
    fn no_partial_plans_from_the_probe() {
        let sha = "d".repeat(64);
        let p = reconcile(None, &probe_of(4096, "\"v9\""), Some(&sha)).unwrap();
        assert_eq!(p.size, 4096);
        assert_eq!(p.chunk, parts::chunk_bytes());
        assert_eq!(p.oid.as_deref(), Some(sha.as_str()));
        assert_eq!(p.validator.as_deref(), Some("\"v9\""));
        assert_eq!(p.completed_bytes(), 0);
    }
}
