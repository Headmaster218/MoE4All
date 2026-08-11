//! Model download from the HuggingFace hub over plain HTTP (reqwest) — no external CLI
//! (`huggingface-cli`) is ever invoked. Downloads stream into the **shared HF Hub cache**
//! (`~/.cache/huggingface/hub`, the same dir llama.cpp / `huggingface_hub` use) with **resume**
//! (HTTP Range) + a progress bar, and land in HF's `models--<org>--<repo>/{blobs,snapshots,refs}`
//! layout so the result is interchangeable with a `llama-cli -hf` download.
//!
//! This module decides WHICH files a model is and how many are fetched at a time; one file's bytes
//! are [`crate::download`]'s business.

use crate::download::download_to_blob;
use crate::http::{api_client, head_client, token, ConnBudget};
use crate::{model_ref::ModelRef, store::Store};
use indicatif::MultiProgress;
use infr_core::config::Config;
use infr_core::error::{Error, Result};
use infr_core::progress;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::{
    fs,
    os::unix::fs::symlink,
    path::{Component, Path, PathBuf},
};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Download a model into the HF Hub cache, returning the resolved GGUF path (a `snapshots/` symlink).
/// Idempotent: a model already cached is returned without re-downloading. Does NOT check for updates —
/// any cached snapshot satisfies it (the fast path for `infr run`). For an update check see
/// [`pull_latest`].
pub fn pull(r: &ModelRef, cfg: &Config) -> Result<PathBuf> {
    match r {
        ModelRef::Path(p) => Ok(p.clone()),
        ModelRef::Repo { repo, sel } => pull_repo(repo, sel.as_deref(), cfg),
    }
}

/// Like [`pull`] but ALWAYS queries HF for the repo's current `main` commit first and downloads when
/// the cached snapshot is missing or stale (the remote commit moved). A no-op when already up to date
/// (one cheap API call). On any network/API error, falls back to the cached copy if there is one
/// (offline-friendly). This is what `infr pull` runs so a re-pull actually picks up repo updates.
pub fn pull_latest(r: &ModelRef, cfg: &Config) -> Result<PathBuf> {
    match r {
        ModelRef::Path(p) => Ok(p.clone()),
        ModelRef::Repo { repo, sel } => pull_repo_latest(repo, sel.as_deref(), cfg),
    }
}

/// The HuggingFace origin every `resolve`/API URL is built from.
///
/// A parameter rather than a hardcoded literal at each `format!` so the download path can be
/// pointed at a local HTTP server in tests — the only way to exercise resume, the sha256 gate and
/// the concurrency bound against real sockets without reaching huggingface.co from `cargo test`.
/// Production has exactly one value and passes it from here.
const HF_ORIGIN: &str = "https://huggingface.co";

/// The `resolve` URL for one file of a repo: what a browser would download.
fn resolve_url(origin: &str, repo: &str, filename: &str) -> String {
    format!("{origin}/{repo}/resolve/main/{filename}")
}

fn pull_repo_latest(repo: &str, sel: Option<&str>, cfg: &Config) -> Result<PathBuf> {
    let store = Store::discover()?;
    // Ask HF for the current commit + concrete gguf filename. If the API is unreachable (offline),
    // serve whatever is cached rather than failing.
    let (commit, filename, siblings) = match repo_info(HF_ORIGIN, repo, sel) {
        Ok(x) => x,
        Err(e) => {
            return match store.resolve_repo(repo, sel) {
                Some(p) => {
                    info!("hf:{repo}: update check failed ({e}); using cached copy");
                    Ok(p)
                }
                None => Err(e),
            };
        }
    };

    let repo_dir = store.repo_dir(repo);
    let blobs = repo_dir.join("blobs");
    let snap = repo_dir.join("snapshots").join(&commit);
    // A sharded GGUF needs its WHOLE `-NNNNN-of-MMMMM` set — one shard fails at load.
    let shards = crate::store::shard_set(&filename);
    let primary = snap.join(&shards[0]);
    let progress = progress::group();
    // Up to date when THIS commit's snapshot already links every present shard blob.
    if shards.iter().all(|f| snap.join(f).exists()) {
        info!("hf:{repo}:{filename} already up to date ({commit})");
        // Still ensure companions — a snapshot pulled before this feature won't have them yet.
        fetch_companions(HF_ORIGIN, repo, &blobs, &snap, &siblings, &progress);
        return Ok(primary);
    }

    // Repoint refs/main + (re)build the snapshot. `fetch_and_link` content-addresses each shard, so a
    // commit that only moved a sibling (file bytes unchanged) relinks the cached blob — no re-download.
    info!("Updating hf:{repo}:{filename} → {commit}");
    write_text(&repo_dir.join("refs").join("main"), &commit)?;
    fs::create_dir_all(&snap).map_err(Error::from)?;
    fetch_all(
        HF_ORIGIN,
        &blobs,
        &snap,
        repo,
        &shards,
        cfg.hub.pull_jobs,
        &progress,
    )?;
    fetch_companions(HF_ORIGIN, repo, &blobs, &snap, &siblings, &progress);
    Ok(primary)
}

/// Download one file (or reuse its content-addressed blob) into `snap` as a symlink into `blobs`.
/// Returns the snapshot symlink path. HEADs for the LFS sha256 so a present blob is relinked without
/// a download and a fresh download is verified against it.
///
/// `filename` is an HF `rfilename` and may name a SUBDIRECTORY (`UD-Q4_K_XL/model.gguf`, how
/// unsloth ships its Dynamic quants), so the link's parent is created first — otherwise `symlink`
/// fails ENOENT and the whole pull dies. It is validated by [`is_safe_relative`] before any join:
/// the name comes from a remote API response, and a `..` component in it would make `snap.join`
/// plant a symlink anywhere the user can write.
///
/// Runs on a worker thread when several files are in flight ([`fetch_all`]). Everything it touches
/// is per-FILE — its own `.dl-` temp, its own lockfile, its own progress line — except the two
/// `create_dir_all`s, which are idempotent, and the blob directory, which is content-addressed.
fn fetch_and_link(
    origin: &str,
    blobs: &Path,
    snap: &Path,
    repo: &str,
    filename: &str,
    budget: &ConnBudget,
    progress: &MultiProgress,
) -> Result<PathBuf> {
    let filename = check_relative(filename)?;
    let url = resolve_url(origin, repo, filename);
    let want = head_lfs_sha(origin, repo, filename).ok().flatten();
    let hex = match &want {
        Some(sha) if blobs.join(sha).exists() => {
            debug!("hf:{repo}:{filename} blob already present; linking → {sha}");
            sha.clone()
        }
        _ => {
            download_to_blob(
                &url,
                blobs,
                filename,
                want.as_deref(),
                None, // model blobs are legitimately multi-GB — never capped
                budget,
                progress,
            )?
            .1
        }
    };
    let link = snap.join(filename);
    if let Some(parent) = link.parent() {
        fs::create_dir_all(parent).map_err(Error::from)?; // subdirectory rfilename
    }
    let _ = fs::remove_file(&link); // replace a stale/dangling link
    let target = blob_link_target(filename, &hex);
    symlink(&target, &link).map_err(Error::from)?;
    debug!("linked {link:?} -> {target}");
    Ok(link)
}

/// Fetch and link every file in `files`, over at most `jobs` connections IN TOTAL.
///
/// One HTTPS connection to the HF CDN — not the link — is what caps a download (see
/// [`infr_core::config::HubCfg::pull_jobs`] for the measurement), and there are two ways to spend
/// more of them: on several FILES at once, and on several RANGES of one file
/// ([`crate::ranged`]). `jobs` bounds the two TOGETHER, through one [`ConnBudget`] shared by
/// everything this pull starts — the alternative, a bound per axis, multiplies into `jobs × jobs`
/// sockets on a repo whose shard count is the publisher's choice (`DeepSeek-V3.2-REAP` ships 236).
///
/// How the allowance divides is decided by the work rather than by a setting. Each file worker
/// holds one permit for its lifetime, so `min(files, jobs)` of them run and a 236-shard repo has
/// nothing left over and splits nothing; a single-file repo starts one worker and its file takes
/// the other seven permits as ranges; and in between, a worker that runs out of files returns its
/// permit to whichever download is still going. Range workers only ever TRY to take a permit, so
/// they can never keep a file worker waiting.
///
/// The work queue is one shared cursor, so each file is handed to exactly one worker and the order
/// is still shard 1 first. A failure sets `stop`, which keeps the other workers from CLAIMING more
/// files (an in-flight transfer is not interruptible and runs to its own end) — without it, a
/// missing shard 1 would be followed by cheerfully downloading the other 200 GB. That matches the
/// sequential loop this replaced, which stopped at the first `?`.
///
/// `jobs = 0` and `jobs = 1` both mean strictly one connection: one file at a time, unsplit, which
/// is the pre-concurrency behaviour and stays available for a metered link.
fn fetch_all(
    origin: &str,
    blobs: &Path,
    snap: &Path,
    repo: &str,
    files: &[String],
    jobs: usize,
    progress: &MultiProgress,
) -> Result<()> {
    let budget = ConnBudget::new(jobs.max(1));
    // One permit per file worker, taken up front — so `workers` is what the budget actually
    // granted rather than a number computed alongside it that could drift from it.
    let primaries = budget.acquire_up_to(files.len().min(jobs.max(1)));
    let workers = primaries.len();
    if files.len() > 1 {
        info!(
            "hf:{repo}: {} files to fetch, {workers} at a time",
            files.len()
        );
    }
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let failures: Vec<Error> = std::thread::scope(|scope| {
        let handles: Vec<_> = primaries
            .into_iter()
            .map(|permit| {
                scope.spawn(|| {
                    // Dropped when this worker runs out of files, handing its connection to
                    // whichever download is still going — the tail of a shard set is one file with
                    // the whole budget, not one file on one connection.
                    let _permit = permit;
                    while !stop.load(Ordering::Relaxed) {
                        let Some(f) = files.get(next.fetch_add(1, Ordering::Relaxed)) else {
                            break;
                        };
                        if let Err(e) =
                            fetch_and_link(origin, blobs, snap, repo, f, &budget, progress)
                        {
                            stop.store(true, Ordering::Relaxed);
                            return Some(e);
                        }
                    }
                    None
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| match h.join() {
                Ok(e) => e,
                // A worker panicked: report it rather than let `join`'s `Err` vanish into a
                // successful-looking pull.
                Err(_) => Some(Error::Other(
                    "download worker panicked; see the panic message above".into(),
                )),
            })
            .collect()
    });
    // Each worker returns at most one error, and more than one can fail in the window before `stop`
    // latches. Report one — in worker order, which says nothing about which failed first — and keep
    // the others visible rather than dropping them silently.
    let mut failures = failures.into_iter();
    match failures.next() {
        None => Ok(()),
        Some(first) => {
            for e in failures {
                debug!("hf:{repo}: further download failure: {e}");
            }
            Err(first)
        }
    }
}

/// The RELATIVE symlink target stored in a snapshot for the blob `hex`, as `huggingface_hub` writes
/// it: enough `..` segments to climb from the link's own directory back to `models--<org>--<repo>/`,
/// then `blobs/<hex>`.
///
/// `snapshots/<commit>/<file>` is two levels down from the repo dir → `../../blobs/<hex>`; every
/// extra directory in `rel` (`snapshots/<commit>/UD-Q4_K_XL/<file>`) adds one more `..`. Hardcoding
/// `../../` — as this used to — silently produces a DANGLING link for a subdirectory GGUF, which
/// reads as "not cached" and re-downloads multiple GB on every run. The link must also stay valid
/// when the whole hub dir is moved, and stay byte-identical to what `huggingface_hub` / llama.cpp
/// would write, since the two share this cache (see the module header).
pub(crate) fn blob_link_target(rel: &str, hex: &str) -> String {
    // Depth of the link BELOW the repo dir = 2 (`snapshots/<commit>`) + directories inside `rel`.
    let dirs = Path::new(rel).components().count().saturating_sub(1);
    let mut target = String::new();
    for _ in 0..(2 + dirs) {
        target.push_str("../");
    }
    target.push_str("blobs/");
    target.push_str(hex);
    target
}

/// True when `name` may be `join`ed onto a local directory without escaping it: non-empty, relative,
/// and built purely of ordinary path components. Subdirectories are ALLOWED (`UD-Q4_K_XL/m.gguf` is
/// a real HF `rfilename`); `..`, `.`, a leading `/`, and a Windows prefix/drive are not.
///
/// The check is component-based rather than a substring scan on purpose: `Path`'s own parser is what
/// `join` will use, so agreeing with it is correct by construction, and a name like `a..b.gguf` (no
/// `..` component) is not falsely rejected. Every snapshot-side path here is built from a filename
/// the remote model API handed us, so an unvalidated join is an arbitrary-file-write primitive for
/// a malicious or compromised API response.
fn is_safe_relative(name: &str) -> bool {
    !name.is_empty()
        && Path::new(name)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

/// [`is_safe_relative`] as a guard, naming the rejected value so a genuinely odd repo is debuggable.
fn check_relative(name: &str) -> Result<&str> {
    if is_safe_relative(name) {
        Ok(name)
    } else {
        Err(Error::Other(format!(
            "refusing unsafe remote filename {name:?}: must be a relative path with no '..' components"
        )))
    }
}

/// HEAD the resolve URL to read the file's LFS sha256 (HF's `X-Linked-Etag`) WITHOUT downloading the
/// body — so a commit bump that left the file unchanged can relink the cached blob, and so the
/// download can verify the body against it. Returns `Ok(Some(sha))` for an LFS file, `Ok(None)` for a
/// non-LFS file (no `X-Linked-Etag`), and `Err` only on a transport failure. The plain `ETag` is a
/// quoted md5, NOT the content sha256, so it is deliberately never used here — treating it as a sha
/// would defeat both the relink fast-path and integrity verification. Redirects are disabled because
/// the sha header is on huggingface.co's 302, not the CDN's final 200.
fn head_lfs_sha(origin: &str, repo: &str, filename: &str) -> Result<Option<String>> {
    let url = resolve_url(origin, repo, filename);
    let mut req = head_client()?.head(&url);
    if let Some(t) = token() {
        req = req.bearer_auth(t);
    }
    let resp = req
        .send()
        .map_err(|e| Error::Other(format!("HEAD {url}: {e}")))?;
    let sha = resp
        .headers()
        .get("x-linked-etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"').to_ascii_lowercase())
        .filter(|s| is_sha256(s));
    Ok(sha)
}

/// True for a lowercase hex sha256 digest: exactly 64 hex digits.
fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

// ---------------------------------------------------------------------------
// HuggingFace
// ---------------------------------------------------------------------------

fn pull_repo(repo: &str, sel: Option<&str>, cfg: &Config) -> Result<PathBuf> {
    let store = Store::discover()?;
    // Already cached (any matching snapshot)?
    if let Some(p) = store.resolve_repo(repo, sel) {
        debug!("hf:{repo} ({}) already cached", sel.unwrap_or("default"));
        return Ok(p);
    }

    // Resolve the repo's main commit + the concrete gguf filename for `sel` via the HF model API.
    let (commit, filename, siblings) = repo_info(HF_ORIGIN, repo, sel)?;
    info!("Pulling hf:{repo}:{filename}");

    let repo_dir = store.repo_dir(repo);
    let blobs = repo_dir.join("blobs");
    // Write the HF Hub pointers: refs/main = commit, snapshots/<commit>/<file> -> ../../blobs/<sha>.
    write_text(&repo_dir.join("refs").join("main"), &commit)?;
    let snap = repo_dir.join("snapshots").join(&commit);
    fs::create_dir_all(&snap).map_err(Error::from)?;
    // A sharded GGUF (`-NNNNN-of-MMMMM`) needs its WHOLE set downloaded/linked — a lone shard 1 fails
    // at load. A non-sharded file is a singleton set.
    let shards = crate::store::shard_set(&filename);
    let progress = progress::group();
    fetch_all(
        HF_ORIGIN,
        &blobs,
        &snap,
        repo,
        &shards,
        cfg.hub.pull_jobs,
        &progress,
    )?;
    fetch_companions(HF_ORIGIN, repo, &blobs, &snap, &siblings, &progress);
    Ok(snap.join(&shards[0]))
}

/// Query the HF model API for `repo`: return `(commit_sha, gguf_filename, sibling_filenames)` for
/// selector `sel`. The sibling list lets the caller fetch companion files (see [`fetch_companions`])
/// without a second API round-trip.
fn repo_info(origin: &str, repo: &str, sel: Option<&str>) -> Result<(String, String, Vec<String>)> {
    let url = format!("{origin}/api/models/{repo}");
    debug!("GET {url}");
    let mut req = api_client()?.get(&url);
    if let Some(t) = token() {
        req = req.bearer_auth(t);
    }
    let resp = req
        .send()
        .map_err(|e| Error::Other(format!("HF API request: {e}")))?;
    if !resp.status().is_success() {
        return Err(Error::Other(format!(
            "HF API failed: HTTP {}",
            resp.status()
        )));
    }

    #[derive(serde::Deserialize)]
    struct Sibling {
        rfilename: String,
    }
    #[derive(serde::Deserialize)]
    struct ModelInfo {
        sha: String,
        siblings: Vec<Sibling>,
    }
    let info: ModelInfo = resp
        .json()
        .map_err(|e| Error::Other(format!("parsing HF API response: {e}")))?;

    let names: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();
    let file = crate::store::pick_gguf(&names, sel).ok_or_else(|| {
        Error::Other(match sel {
            Some(s) => format!("no .gguf matching '{s}' in {repo}"),
            None => format!("no .gguf files found in {repo}"),
        })
    })?;
    // Reject a traversing `rfilename` at the boundary where it enters the system, before it is ever
    // joined onto a local path (`shard_set` derives more names from it, all sharing its directory).
    check_relative(&file)?;
    // Same for the commit: it names `snapshots/<commit>/` and comes from the same response.
    check_relative(&info.sha)?;
    Ok((info.sha, file, names))
}

/// Small non-GGUF sibling files worth caching NEXT TO the GGUF. `generation_config.json` carries the
/// model's own recommended sampling (temperature/top_k/top_p) — the CLI reads it beside the model to
/// seed `infr run`/`serve` defaults (see infr-cli's `model_sampling_defaults`). Kept deliberately
/// tiny: only files the engine actually consumes belong here.
const COMPANIONS: &[&str] = &["generation_config.json"];

/// Hard ceiling on a companion download, enforced DURING streaming (see
/// [`crate::download::download_to_blob`]).
///
/// A companion is the one file in the snapshot that may legitimately arrive unverified: HF only
/// advertises an `X-Linked-Etag` sha256 for LFS objects, and a small `generation_config.json` is
/// usually stored as a plain (non-LFS) blob, so there is nothing to check the body against. Without
/// a cap, "unverified" also means "unbounded": a hostile mirror, a misrouted redirect or simply a
/// broken proxy answering with a multi-GB body writes all of it into the user's HF cache before
/// anyone notices, and the failure is disk exhaustion, not a bad config.
///
/// 1 MiB is chosen as generous-but-finite. The real files are under 2 KB (a handful of sampling
/// keys), so this is ~500x headroom — no plausible companion is ever refused — while bounding the
/// worst case to a single megabyte of wasted I/O that is then deleted. The cap deliberately does
/// NOT apply to the model blob path: a GGUF is legitimately multi-GB and is sha256-verified.
pub(crate) const MAX_COMPANION_BYTES: u64 = 1 << 20;

/// Download any [`COMPANIONS`] the repo lists into `snap` (the GGUF's snapshot dir, so they sit
/// beside it), content-addressed + symlinked exactly like the GGUF. Idempotent (skips a present
/// link) and STRICTLY NON-FATAL: a companion that's absent, unlisted, or fails to download never
/// fails the model pull — it's a convenience, not a requirement. `siblings` is the repo file list
/// already fetched by [`repo_info`], so an absent companion costs zero network calls.
///
/// A companion is HEADed for its LFS sha256 exactly like the GGUF ([`fetch_and_link`]) and the
/// digest is passed through to the download, so an LFS companion is integrity-checked to the same
/// standard as the model blob it sits next to. That symmetry is the point: `generation_config.json`
/// feeds the CLI's sampling defaults (temperature/top_k/top_p), and having ONE unverified file in
/// a snapshot whose every other byte is content-addressed is the gap, not the blast radius. A
/// NON-LFS companion has no `X-Linked-Etag` and therefore stays unverified — that is expected and
/// fine, and [`MAX_COMPANION_BYTES`] is what keeps that case bounded.
///
/// Both new failure modes stay inside the non-fatal contract: a sha mismatch or an over-cap body
/// is a `debug!` and a skipped companion, never a failed model pull.
///
/// Deliberately NOT part of the [`fetch_all`] fan-out: this is one sub-2-KB file, so a connection
/// spent on it buys nothing, and it runs after the shards precisely so it cannot take a slot from
/// them.
fn fetch_companions(
    origin: &str,
    repo: &str,
    blobs: &Path,
    snap: &Path,
    siblings: &[String],
    progress: &MultiProgress,
) {
    for &name in COMPANIONS {
        if !siblings.iter().any(|s| s == name) {
            continue; // repo doesn't ship it
        }
        // Belt-and-braces: COMPANIONS is a compile-time list of plain filenames, but this is a join
        // of a name that was matched against remote data — keep the guard where the join is.
        if check_relative(name).is_err() {
            continue;
        }
        let link = snap.join(name);
        if link.exists() {
            continue; // already cached
        }
        let url = resolve_url(origin, repo, name);
        // Same integrity path as the GGUF: HEAD for the LFS sha256 and hand it to the download so
        // the body is verified (and a present blob relinked without a transfer). `Ok(None)` = the
        // file is not LFS, so no digest exists; a HEAD transport error degrades to the same
        // unverified-but-capped path rather than skipping the companion.
        let want = head_lfs_sha(origin, repo, name).ok().flatten();
        // No connection budget: a companion is a sub-2-KB file and splitting it across ranges
        // would be more requests than bytes.
        let dl = download_to_blob(
            &url,
            blobs,
            name,
            want.as_deref(),
            Some(MAX_COMPANION_BYTES),
            &ConnBudget::new(0),
            progress,
        );
        match dl {
            Ok((_, hex, _)) => {
                let _ = fs::remove_file(&link);
                match symlink(blob_link_target(name, &hex), &link) {
                    Ok(()) => info!("hf:{repo}: cached companion {name}"),
                    Err(e) => debug!("hf:{repo}: companion {name} symlink failed: {e}"),
                }
            }
            Err(e) => debug!("hf:{repo}: companion {name} not cached ({e})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write `text` to `path`, creating parent directories.
fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(Error::from)?;
    }
    fs::write(path, text).map_err(Error::from)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parts;
    use crate::testhttp::TestHub;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    /// A throwaway HF-cache repo dir with the REAL layout (`blobs/` beside `snapshots/<commit>/`).
    /// The layout is load-bearing for these tests: [`blob_link_target`] writes `../../blobs/<sha>`
    /// from the file's own depth, so a flat scratch directory would leave every snapshot link
    /// dangling and the assertions would be measuring the fixture.
    struct TempRepo(tempfile::TempDir);

    impl TempRepo {
        fn new() -> Self {
            let repo = Self(tempfile::tempdir().unwrap());
            fs::create_dir_all(repo.blobs()).unwrap();
            fs::create_dir_all(repo.snapshot()).unwrap();
            repo
        }
        fn blobs(&self) -> PathBuf {
            self.0.path().join("models--org--repo").join("blobs")
        }
        fn snapshot(&self) -> PathBuf {
            self.0
                .path()
                .join("models--org--repo")
                .join("snapshots")
                .join("c1")
        }
    }

    #[test]
    fn is_sha256_shape() {
        assert!(is_sha256(SHA_A));
        assert!(is_sha256(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        ));
        assert!(!is_sha256(&"a".repeat(63))); // too short
        assert!(!is_sha256(&"a".repeat(65))); // too long
        assert!(!is_sha256(&"g".repeat(64))); // non-hex
        assert!(!is_sha256("d41d8cd98f00b204e9800998ecf8427e")); // md5 (32 chars)
        assert!(!is_sha256(""));
    }

    /// The `..` depth of a snapshot symlink target must follow the file's OWN depth, or the link
    /// dangles: `snapshots/<commit>/<file>` is 2 below the repo dir, `snapshots/<commit>/<dir>/<file>`
    /// is 3. Verified end-to-end (an actual dangling-vs-readable link) in store.rs's
    /// `resolve_subdirectory_gguf_is_cached`.
    #[test]
    fn blob_link_target_depth_follows_filename() {
        assert_eq!(
            blob_link_target("model.gguf", SHA_A),
            format!("../../blobs/{SHA_A}")
        );
        assert_eq!(
            blob_link_target("UD-Q4_K_XL/model.gguf", SHA_A),
            format!("../../../blobs/{SHA_A}")
        );
        assert_eq!(
            blob_link_target("a/b/model.gguf", SHA_A),
            format!("../../../../blobs/{SHA_A}")
        );
    }

    /// Subdirectories WORK, traversal does NOT — one rule, both directions. The names come from a
    /// remote API response, so a `..` component reaching `Path::join` is an arbitrary-file-write.
    #[test]
    fn safe_relative_allows_subdirs_rejects_traversal() {
        for ok in [
            "model.gguf",
            "UD-Q4_K_XL/model.gguf",
            "a/b/c/model.gguf",
            "weird..name.gguf",   // `..` in a component is not a `..` component
            "-leading-dash.gguf", // odd but harmless
            "a/./b.gguf",         // `Path` normalises an interior `.` away; still inside `snap`
        ] {
            assert!(is_safe_relative(ok), "{ok} should be accepted");
            assert!(check_relative(ok).is_ok());
        }
        for bad in [
            "",
            "..",
            "../evil.gguf",
            "a/../../evil.gguf",
            "/etc/cron.d/evil",
            ".",
            "./model.gguf",
        ] {
            assert!(!is_safe_relative(bad), "{bad:?} should be rejected");
            let err = check_relative(bad).unwrap_err().to_string();
            assert!(err.contains("unsafe remote filename"), "{err}");
        }
    }

    /// The companion cap is generous relative to what these files actually are (~2 KB of
    /// sampling keys) but finite — the point is bounding an UNVERIFIED download, not policing a
    /// plausible size. Pinned so a later edit to the constant is a deliberate decision.
    #[test]
    fn companion_cap_is_generous_but_finite() {
        assert_eq!(MAX_COMPANION_BYTES, 1 << 20); // ~500x a real generation_config.json
    }

    /// A shard set fetched concurrently lands COMPLETE: every blob content-addressed by its own
    /// sha256, every snapshot symlink readable and carrying that shard's own bytes. A fan-out that
    /// crossed two downloads' streams would show up here as a shard holding another's content.
    #[test]
    fn concurrent_shards_all_land() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-Q4_K_M", 5, 40_000);
        let cache = TempRepo::new();
        let (blobs, snap) = (cache.blobs(), cache.snapshot());

        fetch_all(
            &hub.origin(),
            &blobs,
            &snap,
            "org/repo",
            &files,
            4,
            &progress::group(),
        )
        .unwrap();

        for f in &files {
            let got = fs::read(snap.join(f)).expect("snapshot link must resolve to its blob");
            assert_eq!(got, hub.body(f), "{f} has the wrong bytes");
            // …and it is stored under the sha256 of exactly those bytes.
            assert!(blobs.join(hub.sha(f)).exists(), "{f} blob is not addressed");
        }
    }

    /// The BOUND binds. Ten files with a limit of three must never put a fourth request in flight —
    /// asserted on the peak the server itself observed, so an unbounded implementation (spawn one
    /// worker per file) fails here instead of passing for looking concurrent.
    #[test]
    fn concurrency_never_exceeds_the_bound() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-Q8_0", 10, 20_000);
        let cache = TempRepo::new();
        let (blobs, snap) = (cache.blobs(), cache.snapshot());

        fetch_all(
            &hub.origin(),
            &blobs,
            &snap,
            "org/repo",
            &files,
            3,
            &progress::group(),
        )
        .unwrap();

        let peak = hub.peak_concurrent_bodies();
        assert!(peak <= 3, "peak {peak} bodies in flight, bound was 3");
        // …and it really did overlap, or "never exceeded 3" would also be true of the sequential
        // loop this replaced and the assertion above would prove nothing.
        assert!(peak >= 2, "peak {peak}: nothing ran concurrently at all");
    }

    /// `jobs = 1` is the pre-concurrency path and must stay strictly sequential — the setting a
    /// metered link or a fussy proxy reaches for.
    #[test]
    fn one_job_is_sequential() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-seq", 4, 20_000);
        let cache = TempRepo::new();
        let (blobs, snap) = (cache.blobs(), cache.snapshot());

        fetch_all(
            &hub.origin(),
            &blobs,
            &snap,
            "org/repo",
            &files,
            1,
            &progress::group(),
        )
        .unwrap();

        assert_eq!(hub.peak_concurrent_bodies(), 1);
        for f in &files {
            assert_eq!(fs::read(snap.join(f)).unwrap(), hub.body(f));
        }
    }

    /// A file of one chunk or less is fetched by ONE connection however high the bound is set —
    /// the "small files are not split at all" rule, whose threshold is the chunk size itself
    /// ([`crate::parts::CHUNK_BYTES`]) so that there is one number rather than two.
    ///
    /// The assertion is on the CONNECTION count, not the worker count, and that is deliberate:
    /// `min(files, jobs)` spawning one worker is today's reason, but an extra worker that finds the
    /// queue empty would be harmless, whereas a second connection would not.
    #[test]
    fn a_small_file_is_not_split() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-solo", 1, parts::chunk_bytes() as usize - 1);
        let cache = TempRepo::new();
        let (blobs, snap) = (cache.blobs(), cache.snapshot());

        fetch_all(
            &hub.origin(),
            &blobs,
            &snap,
            "org/repo",
            &files,
            8,
            &progress::group(),
        )
        .unwrap();

        assert_eq!(hub.peak_concurrent_bodies(), 1);
        assert_eq!(fs::read(snap.join(&files[0])).unwrap(), hub.body(&files[0]));
    }

    /// Resume survives the fan-out: a partial `.dl-` left by an interrupted pull is continued with
    /// a `Range` request while its siblings download alongside it, and the finished blob still
    /// hashes to the whole file. Per-file temps are what make this safe, so a shared one would show
    /// up as a corrupt (sha-refused) blob rather than as a silent pass.
    #[test]
    fn resume_works_under_concurrency() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-resume", 4, 60_000);
        let cache = TempRepo::new();
        let (blobs, snap) = (cache.blobs(), cache.snapshot());

        // Interrupted earlier: the first 25 000 bytes of shard 3 are already on disk.
        let partial = &files[2];
        let head = 25_000;
        fs::write(
            blobs.join(crate::download::partial_name(partial)),
            &hub.body(partial)[..head],
        )
        .unwrap();

        fetch_all(
            &hub.origin(),
            &blobs,
            &snap,
            "org/repo",
            &files,
            4,
            &progress::group(),
        )
        .unwrap();

        assert_eq!(
            hub.ranges_served(partial),
            vec![head as u64],
            "the partial must be CONTINUED, not restarted"
        );
        for f in &files {
            assert_eq!(fs::read(snap.join(f)).unwrap(), hub.body(f), "{f}");
        }
    }

    /// A partial whose object CHANGED upstream is thrown away, not continued. The stored `If-Range`
    /// validator no longer matches, so the server answers `200` with the whole body and the prefix
    /// is truncated instead of being spliced onto. That splice — old head, new tail, plausible
    /// size — is what read as "this quant is broken" for a day (memory `verify-gguf-downloads`),
    /// and it is undetectable in the bytes: only restarting, or the sha256 gate, catches it.
    #[test]
    fn a_stale_partial_is_restarted_not_spliced() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-stale", 2, 40_000);
        let cache = TempRepo::new();
        let (blobs, snap) = (cache.blobs(), cache.snapshot());

        // What an interrupted download of the PREVIOUS upload left behind: a prefix that is not a
        // prefix of the current file, plus the validator that upload was served with.
        let stale = &files[0];
        let partial = crate::download::partial_name(stale);
        fs::write(blobs.join(&partial), vec![0xAB; 20_000]).unwrap();
        fs::write(blobs.join(format!("{partial}.meta")), "\"v0\"").unwrap();

        fetch_all(
            &hub.origin(),
            &blobs,
            &snap,
            "org/repo",
            &files,
            2,
            &progress::group(),
        )
        .unwrap();

        assert_eq!(
            hub.ranges_served(stale),
            vec![20_000],
            "the resume was attempted (and refused), so this is the splice path"
        );
        assert_eq!(fs::read(snap.join(stale)).unwrap(), hub.body(stale));
        assert!(blobs.join(hub.sha(stale)).exists());
    }

    /// A body that does not hash to the advertised LFS oid is REFUSED and never linked — the
    /// regression that cost a day of "the quant is broken" debugging when a resumed download
    /// spliced two different uploads (memory `verify-gguf-downloads`). Concurrency must not have
    /// weakened it.
    #[test]
    fn a_mismatched_sha_is_refused_and_unlinked() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-corrupt", 3, 20_000);
        // Shard 2's HEAD advertises a digest its body does not have — what a mid-flight re-upload
        // or a lying mirror looks like from here.
        hub.lie_about_sha(&files[1], SHA_A);
        let cache = TempRepo::new();
        let (blobs, snap) = (cache.blobs(), cache.snapshot());

        let err = fetch_all(
            &hub.origin(),
            &blobs,
            &snap,
            "org/repo",
            &files,
            3,
            &progress::group(),
        )
        .expect_err("a sha256 mismatch must fail the pull")
        .to_string();
        assert!(err.contains("sha256 mismatch"), "{err}");

        // Nothing corrupt was kept: no blob under the advertised digest, no link for that shard,
        // and no partial left to resume the bad bytes from.
        assert!(!blobs.join(SHA_A).exists(), "a mismatched blob was stored");
        assert!(!blobs.join(hub.sha(&files[1])).exists());
        assert!(!snap.join(&files[1]).exists(), "a mismatched shard linked");
        assert!(!blobs
            .join(crate::download::partial_name(&files[1]))
            .exists());
    }

    /// A failure stops the fan-out CLAIMING more work. Without it a repo whose first shard 404s
    /// still pulls every other shard — 200 GB for a model that cannot load.
    #[test]
    fn a_failure_stops_the_remaining_fetches() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-404", 8, 20_000);
        hub.remove(&files[0]);
        let cache = TempRepo::new();
        let (blobs, snap) = (cache.blobs(), cache.snapshot());

        fetch_all(
            &hub.origin(),
            &blobs,
            &snap,
            "org/repo",
            &files,
            2,
            &progress::group(),
        )
        .expect_err("a missing shard must fail the pull");

        let fetched = files.iter().filter(|f| snap.join(f).exists()).count();
        assert!(
            fetched < files.len() - 1,
            "{fetched} of {} shards were fetched anyway",
            files.len()
        );
    }

    // -----------------------------------------------------------------------
    // Ranged (intra-file) parallelism
    // -----------------------------------------------------------------------

    /// A file big enough to be split: ten cells of the test grid.
    fn big() -> usize {
        10 * parts::chunk_bytes() as usize
    }

    /// The chunk requests the hub honoured for `f`, as `(start, end)`, IN THE ORDER THEY ARRIVED —
    /// so a test can slice off "what the second run asked for".
    ///
    /// The one-byte planning probe (`bytes=0-0`) is filtered out: it is how the size and the
    /// server's range support are learned, not a piece of the file.
    fn chunk_spans(hub: &TestHub, f: &str) -> Vec<(u64, u64)> {
        hub.served_bodies(f)
            .into_iter()
            .filter(|s| s.honoured && s.range != Some((0, Some(0))))
            .filter_map(|s| s.range)
            .map(|(a, b)| (a, b.expect("a chunk request is a closed range")))
            .collect()
    }

    /// The ranged partial and its sidecar for `f` inside `blobs`.
    fn partial_of(blobs: &Path, f: &str) -> PathBuf {
        blobs.join(crate::download::ranged_partial_name(f))
    }

    /// The starts of the cells a partial has already got, per its own sidecar.
    fn landed_starts(blobs: &Path, f: &str) -> Vec<u64> {
        let plan = parts::load(&partial_of(blobs, f)).expect("an interrupted run leaves a plan");
        (0..plan.chunks())
            .filter(|i| plan.is_done(*i))
            .map(|i| plan.range(i).0)
            .collect()
    }

    fn pull_one(hub: &TestHub, cache: &TempRepo, files: &[String], jobs: usize) -> Result<()> {
        fetch_all(
            &hub.origin(),
            &cache.blobs(),
            &cache.snapshot(),
            "org/repo",
            files,
            jobs,
            &progress::group(),
        )
    }

    /// The core claim: a file assembled from N ranges is byte-for-byte the file one stream would
    /// have fetched, and is stored under the sha256 of exactly those bytes.
    ///
    /// Both halves are downloaded from the same origin in the same test so the comparison is
    /// between the two PATHS and not between a fixture and itself — and the request log is asserted
    /// on too, because "identical to the single stream" is only interesting if the ranged side
    /// really did use ranges.
    #[test]
    fn a_ranged_file_equals_the_single_stream_fetch() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-wide", 1, big());
        let f = &files[0];
        let (ranged, single) = (TempRepo::new(), TempRepo::new());

        pull_one(&hub, &ranged, &files, 8).unwrap();
        let spans = chunk_spans(&hub, f);
        assert_eq!(
            spans.len(),
            10,
            "the file should have been fetched in cells"
        );
        // The cells tile the file exactly: no gap, no overlap, no byte fetched twice.
        let mut sorted = spans.clone();
        sorted.sort();
        let mut want = 0u64;
        for (a, b) in &sorted {
            assert_eq!(*a, want, "cells must tile: {sorted:?}");
            want = b + 1;
        }
        assert_eq!(want, big() as u64, "cells must cover the file: {sorted:?}");

        pull_one(&hub, &single, &files, 1).unwrap();

        let a = fs::read(ranged.snapshot().join(f)).unwrap();
        let b = fs::read(single.snapshot().join(f)).unwrap();
        assert_eq!(a, b, "the ranged file differs from the single-stream file");
        assert_eq!(a, hub.body(f));
        assert!(ranged.blobs().join(hub.sha(f)).exists());
        assert!(single.blobs().join(hub.sha(f)).exists());
        // Nothing is left behind once the blob is committed.
        assert!(!partial_of(&ranged.blobs(), f).exists());
        assert!(parts::load(&partial_of(&ranged.blobs(), f)).is_none());
    }

    /// Killed mid-flight, the cells that LANDED stay landed: the next run fetches the rest and
    /// nothing else, and the assembled file still verifies.
    ///
    /// This is the case a sparse download cannot do without the sidecar — the temp file's length
    /// says nothing about how much of it is real — and "it resumed" is asserted as "it did not
    /// re-request what it already had", since a run that quietly started over would also produce a
    /// correct file and prove nothing.
    #[test]
    fn a_ranged_download_resumes_the_cells_it_missed() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-cut", 1, big());
        let f = &files[0];
        let cache = TempRepo::new();

        // The probe plus three cells get through; everything after that is truncated.
        hub.cut_after(4);
        pull_one(&hub, &cache, &files, 4).expect_err("a cut body must fail the pull");
        let landed = landed_starts(&cache.blobs(), f);
        assert!(
            (1..10).contains(&landed.len()),
            "the interrupted run should have kept some but not all cells, got {landed:?}"
        );

        hub.heal();
        let before = chunk_spans(&hub, f).len();
        pull_one(&hub, &cache, &files, 4).unwrap();

        let second = chunk_spans(&hub, f).split_off(before);
        assert!(!second.is_empty(), "the second run fetched nothing");
        for start in &landed {
            assert!(
                !second.iter().any(|(a, _)| a == start),
                "cell at {start} was already on disk and was fetched again: {second:?}"
            );
        }
        assert_eq!(
            fs::read(cache.snapshot().join(f)).unwrap(),
            hub.body(f),
            "the resumed file is not the object"
        );
        assert!(cache.blobs().join(hub.sha(f)).exists());
    }

    /// `hub.pull_jobs` may change between an interrupted run and the next one. The grid is fixed by
    /// the OBJECT (size and chunk size, both recorded in the sidecar) rather than by the worker
    /// count, so a different job count claims different cells of the same plan instead of
    /// re-planning — which is the step that could otherwise write a cell at an offset the file was
    /// never planned on.
    #[test]
    fn the_plan_survives_a_changed_job_count() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-jobs", 1, big());
        let f = &files[0];
        let cache = TempRepo::new();

        hub.cut_after(3);
        pull_one(&hub, &cache, &files, 2).expect_err("a cut body must fail the pull");
        let plan = parts::load(&partial_of(&cache.blobs(), f)).expect("a plan survives the kill");
        let landed = landed_starts(&cache.blobs(), f);
        assert!(!landed.is_empty(), "nothing landed, so nothing is resumed");

        hub.heal();
        pull_one(&hub, &cache, &files, 7).unwrap();

        let after = parts::load(&partial_of(&cache.blobs(), f));
        assert!(after.is_none(), "the sidecar outlived the download");
        assert_eq!(fs::read(cache.snapshot().join(f)).unwrap(), hub.body(f));
        assert!(cache.blobs().join(hub.sha(f)).exists());
        // The second run used the FIRST run's grid, not one sized for seven workers.
        assert_eq!(plan.chunk, parts::chunk_bytes());
        assert_eq!(plan.chunks(), 10);
    }

    /// The object is REPLACED between two runs, at the same length. Continuing the partial would
    /// splice half of one upload onto half of another at exactly the right size — the failure that
    /// read as "this quant is broken" for a day (memory `verify-gguf-downloads`).
    ///
    /// The partial must be thrown away and every cell fetched again, and the blob must be the NEW
    /// object. Asserting on the request log is what makes this test able to fail: a spliced result
    /// would be caught by the sha256 gate as an error, but a resume that happened to re-fetch
    /// everything anyway would pass a bytes-only assertion while leaving the bug in place.
    #[test]
    fn a_reupload_between_runs_restarts_rather_than_splicing() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-reup", 1, big());
        let f = &files[0];
        let cache = TempRepo::new();
        let old_sha = hub.sha(f);

        hub.cut_after(4);
        pull_one(&hub, &cache, &files, 4).expect_err("a cut body must fail the pull");
        let landed = landed_starts(&cache.blobs(), f);
        assert!(
            !landed.is_empty(),
            "nothing landed, so nothing could splice"
        );

        hub.heal();
        hub.reupload(f); // same size, new bytes, new ETag, new advertised sha256
        let before = chunk_spans(&hub, f).len();
        pull_one(&hub, &cache, &files, 4).unwrap();
        let second = chunk_spans(&hub, f).split_off(before);

        // Every cell of the new object was fetched — the stale ones were NOT kept.
        let mut starts: Vec<u64> = second.iter().map(|(a, _)| *a).collect();
        starts.sort();
        starts.dedup();
        assert_eq!(
            starts.len(),
            10,
            "a re-upload must re-fetch the whole object, got {starts:?}"
        );
        let got = fs::read(cache.snapshot().join(f)).unwrap();
        assert_eq!(got, hub.body(f), "the blob is not the new object");
        assert_ne!(hub.sha(f), old_sha, "the fixture did not actually change");
        assert!(cache.blobs().join(hub.sha(f)).exists());
        assert!(!cache.blobs().join(&old_sha).exists());
    }

    /// The other way a plan stops describing its object: the file is re-uploaded at a DIFFERENT
    /// length. The partial's bytes have to GO, not just its flags — a temp file left longer than
    /// the object it is now supposed to hold assembles to the wrong size even when every cell of
    /// the new plan lands.
    #[test]
    fn a_resized_object_discards_the_partial() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-resize", 1, big());
        let f = &files[0];
        let cache = TempRepo::new();

        hub.cut_after(4);
        pull_one(&hub, &cache, &files, 4).expect_err("a cut body must fail the pull");
        assert!(!landed_starts(&cache.blobs(), f).is_empty());

        hub.heal();
        hub.reupload_resized(f, big() / 2); // half the size, all new bytes
        pull_one(&hub, &cache, &files, 4).unwrap();

        let got = fs::read(cache.snapshot().join(f)).unwrap();
        assert_eq!(got.len(), big() / 2, "the old partial's length survived");
        assert_eq!(got, hub.body(f));
        assert!(cache.blobs().join(hub.sha(f)).exists());
    }

    /// The sha256 gate still fires on the ranged path, and takes the sidecar with it: a partial
    /// whose bytes are known bad must not be resumable, or the next run "continues" corruption.
    #[test]
    fn a_ranged_sha_mismatch_is_refused_and_unlinked() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-wrong", 1, big());
        let f = &files[0];
        hub.lie_about_sha(f, SHA_A);
        let cache = TempRepo::new();

        let err = pull_one(&hub, &cache, &files, 6)
            .expect_err("a sha256 mismatch must fail the pull")
            .to_string();
        assert!(err.contains("sha256 mismatch"), "{err}");

        let blobs = cache.blobs();
        assert!(!blobs.join(SHA_A).exists(), "a mismatched blob was stored");
        assert!(!blobs.join(hub.sha(f)).exists());
        assert!(!cache.snapshot().join(f).exists(), "a bad file was linked");
        assert!(!partial_of(&blobs, f).exists(), "bad bytes kept for resume");
        assert!(parts::load(&partial_of(&blobs, f)).is_none());
    }

    /// A server that does not do ranges gets a single stream, and the file still lands. The
    /// fallback is what keeps this feature from being a hard dependency on an origin behaviour
    /// nothing guarantees.
    #[test]
    fn a_server_that_refuses_ranges_falls_back_to_one_stream() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-noranges", 1, big());
        let f = &files[0];
        hub.refuse_ranges();
        let cache = TempRepo::new();

        pull_one(&hub, &cache, &files, 8).unwrap();

        assert_eq!(fs::read(cache.snapshot().join(f)).unwrap(), hub.body(f));
        assert!(cache.blobs().join(hub.sha(f)).exists());
        assert!(chunk_spans(&hub, f).is_empty(), "ranges were used anyway");
        assert_eq!(
            hub.peak_concurrent_bodies(),
            1,
            "a range-refusing server must not be hammered with connections"
        );
        assert!(!partial_of(&cache.blobs(), f).exists());
    }

    /// A server that ADVERTISES ranges and then serves the whole object anyway is caught mid-flight
    /// and the download falls back to one stream.
    ///
    /// The `206`-per-chunk requirement is what catches it: a `200` carries the object from byte 0,
    /// and writing that at a chunk's offset is a spliced file of exactly the right length. So the
    /// bytes are refused, the holed partial is thrown away rather than left for a resume that could
    /// not use it, and the file is fetched again in one piece.
    #[test]
    fn a_server_that_lies_about_ranges_falls_back_mid_flight() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-liar", 1, big());
        let f = &files[0];
        hub.lie_about_ranges();
        let cache = TempRepo::new();

        pull_one(&hub, &cache, &files, 4).unwrap();

        assert_eq!(fs::read(cache.snapshot().join(f)).unwrap(), hub.body(f));
        assert!(cache.blobs().join(hub.sha(f)).exists());
        assert!(
            chunk_spans(&hub, f).is_empty(),
            "a 200 was treated as a chunk"
        );
        assert!(
            !partial_of(&cache.blobs(), f).exists(),
            "a holed partial was left behind for a resume that cannot use it"
        );
    }

    /// The TOTAL bound holds across both axes at once. Three files of ten cells each with a bound
    /// of four may never put a fifth body in flight — not `4 files × 4 ranges`, and not one
    /// connection per cell.
    ///
    /// An implementation that bounded the two axes separately would peak at 16 here, and one that
    /// split without a budget at 30, so the ceiling is what this asserts; the floor is asserted too,
    /// or "never exceeded four" would also be true of a sequential download.
    #[test]
    fn the_bound_covers_files_and_ranges_together() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-bound", 3, big());
        let cache = TempRepo::new();

        pull_one(&hub, &cache, &files, 4).unwrap();

        let peak = hub.peak_concurrent_bodies();
        assert!(peak <= 4, "peak {peak} bodies in flight, bound was 4");
        assert!(peak >= 3, "peak {peak}: the fan-out did not happen");
        for f in &files {
            assert_eq!(
                fs::read(cache.snapshot().join(f)).unwrap(),
                hub.body(f),
                "{f}"
            );
        }
    }

    /// ONE file spends the WHOLE allowance on itself — the case this slice exists for, since a
    /// 161 GB single-file model has no shards to spread over connections.
    #[test]
    fn one_big_file_uses_the_whole_budget() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-alone", 1, big());
        let cache = TempRepo::new();

        pull_one(&hub, &cache, &files, 4).unwrap();

        let peak = hub.peak_concurrent_bodies();
        assert_eq!(peak, 4, "one file should have used all four connections");
        assert_eq!(
            fs::read(cache.snapshot().join(&files[0])).unwrap(),
            hub.body(&files[0])
        );
    }

    /// A file that finishes hands its connection to the one still going.
    ///
    /// This is the tail of every shard set: once fewer files remain than there are connections, a
    /// download that fixed its fan-out when it started would crawl on one connection while the rest
    /// of the allowance sat idle. Here the small file takes a worker and the big one starts with
    /// only ONE spare connection — the assertion is that the big file's OWN peak reaches the whole
    /// bound anyway, which it can only do by picking up the permit the small file gave back.
    #[test]
    fn a_finished_file_hands_over_its_connection() {
        let hub = TestHub::start();
        let small = hub.add_shards("m-tail-small", 1, 1024)[0].clone();
        let large = hub.add_shards("m-tail-large", 1, big())[0].clone();
        let files = vec![small.clone(), large.clone()];
        let cache = TempRepo::new();

        pull_one(&hub, &cache, &files, 3).unwrap();

        // Two file workers, so the large file could only take ONE spare connection at its start.
        assert_eq!(
            hub.peak_for(&large),
            3,
            "the large file never grew past {} connections",
            hub.peak_for(&large)
        );
        assert!(hub.peak_concurrent_bodies() <= 3, "the bound was 3");
        assert_eq!(
            fs::read(cache.snapshot().join(&large)).unwrap(),
            hub.body(&large)
        );
        assert_eq!(
            fs::read(cache.snapshot().join(&small)).unwrap(),
            hub.body(&small)
        );
    }

    /// `hub.pull_jobs = 1` stays ONE connection and one request per file: no splitting, and not even
    /// the planning probe — the setting for a metered link or a proxy that objects to parallelism
    /// must not quietly cost an extra round trip.
    #[test]
    fn one_job_never_splits_a_file() {
        let hub = TestHub::start();
        let files = hub.add_shards("m-metered", 1, big());
        let f = &files[0];
        let cache = TempRepo::new();

        pull_one(&hub, &cache, &files, 1).unwrap();

        assert_eq!(hub.peak_concurrent_bodies(), 1);
        assert_eq!(
            hub.served_bodies(f).len(),
            1,
            "one job means one request: {:?}",
            hub.served_bodies(f)
        );
        assert_eq!(fs::read(cache.snapshot().join(f)).unwrap(), hub.body(f));
    }
}
