//! The persisted per-range progress sidecar that makes a RANGED download resumable.
//!
//! A single-stream download needs no such thing: one appending writer means `metadata(tmp).len()`
//! is exactly "how many bytes do I have". The moment N workers `pwrite` into disjoint slices of the
//! same file that stops being true — a file whose last chunk landed first is full-length with holes
//! in it — so what has actually been downloaded has to be written down somewhere.
//!
//! The plan is a **fixed grid**: chunk `i` is `[i·chunk, min((i+1)·chunk, size))`, and a chunk is
//! either wholly downloaded or not started as far as resume is concerned. Two consequences are the
//! reason for that shape rather than "split into one range per worker":
//!
//!   * **The plan does not depend on how many workers there are.** Changing `hub.pull_jobs` between
//!     an interrupted run and the next one changes nothing about the grid, so there is no
//!     re-planning step to get wrong — the workers just claim different cells of the same grid.
//!   * **The chunk size is READ BACK from the sidecar, never assumed.** A partial written by a build
//!     with a different [`CHUNK_BYTES`] resumes on its own grid instead of being reinterpreted on a
//!     new one, which is how a resumed download would otherwise write a chunk into the wrong place.
//!
//! What the sidecar deliberately does NOT do is decide integrity. It records progress; whether the
//! assembled bytes are the file HF advertises is still settled by the end-of-download sha256 gate in
//! [`crate::download`], which stays the last line of defence.

use infr_core::error::{Error, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// The grid's chunk size in production: 64 MiB.
///
/// It trades two costs against each other. Each chunk is one HTTP request — and against HF each is
/// a redirect from `huggingface.co` to the CDN as well — so a smaller chunk spends more of the
/// transfer on request setup: a 161 GiB object at 64 MiB is ~2 570 requests, which at ~100 ms of
/// setup spread over eight connections is tens of seconds against a transfer measured in tens of
/// minutes. A larger chunk costs the other way: an interrupted download loses the in-flight chunks,
/// so the worst case thrown away is `pull_jobs × CHUNK_BYTES` — half a gigabyte at eight
/// connections, seconds of re-transfer, and bounded no matter how large the object is.
///
/// It is ALSO the threshold for splitting at all, and deliberately the same number rather than a
/// second constant: an object of one chunk or less has exactly one cell in its grid, so it is
/// fetched by one request on one connection with no sidecar involved. That is what keeps every
/// small file — a `generation_config.json`, a 20 MiB tokenizer, a tiny GGUF — on precisely the path
/// it used before ranges existed.
pub(crate) const CHUNK_BYTES: u64 = 64 << 20;

/// The chunk size a FRESH plan is built with. Tests use a small grid so that a fan-out over many
/// chunks costs a few hundred KiB of loopback traffic instead of gigabytes; every byte offset here
/// is `u64` arithmetic that does not care which value it is given, and the production constant is
/// what the real-object pull exercises.
#[cfg(not(test))]
pub(crate) fn chunk_bytes() -> u64 {
    CHUNK_BYTES
}
#[cfg(test)]
pub(crate) fn chunk_bytes() -> u64 {
    32 << 10
}

/// First line of the sidecar. The version is bumped if the format ever changes meaning; an
/// unrecognised one is not parsed, and an unparsable sidecar means "discard the partial and start
/// over", which is always safe.
const MAGIC: &str = "infr-dl-plan 1";

/// One interrupted ranged download's state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Plan {
    /// The object's total size, from the `Content-Range` of the probe that planned it.
    pub(crate) size: u64,
    /// The grid's cell size. Read back from disk on resume rather than taken from [`chunk_bytes`].
    pub(crate) chunk: u64,
    /// HF's advertised LFS sha256 for the object this partial belongs to, when there is one.
    ///
    /// This is the IDENTITY check, and it is a content hash rather than an opaque validator on
    /// purpose: an `ETag` is whatever the CDN edge that answered says it is, while the oid changes
    /// if and only if the bytes change. A re-upload between two runs is therefore caught here,
    /// deterministically, before a single byte of the new object is written next to the old one's.
    pub(crate) oid: Option<String>,
    /// The `If-Range` validator (`ETag`, else `Last-Modified`) to present on each chunk request.
    /// Refreshed from the current probe on every run — it is the server's view of the object, not
    /// part of the plan's identity.
    pub(crate) validator: Option<String>,
    /// One flag per grid cell: downloaded in full, or not.
    done: Vec<bool>,
}

impl Plan {
    /// A plan for an object of `size` bytes on a `chunk`-sized grid, nothing downloaded yet.
    pub(crate) fn fresh(
        size: u64,
        chunk: u64,
        oid: Option<String>,
        validator: Option<String>,
    ) -> Self {
        let cells = cell_count(size, chunk);
        Plan {
            size,
            chunk,
            oid,
            validator,
            done: vec![false; cells],
        }
    }

    pub(crate) fn chunks(&self) -> usize {
        self.done.len()
    }

    /// Chunk `i`'s `(start, len)`. The last cell is short whenever `size` is not a multiple of
    /// `chunk`.
    pub(crate) fn range(&self, i: usize) -> (u64, u64) {
        let start = (i as u64) * self.chunk;
        let len = self.chunk.min(self.size - start);
        (start, len)
    }

    pub(crate) fn is_done(&self, i: usize) -> bool {
        self.done[i]
    }

    pub(crate) fn mark_done(&mut self, i: usize) {
        self.done[i] = true;
    }

    pub(crate) fn all_done(&self) -> bool {
        self.done.iter().all(|d| *d)
    }

    pub(crate) fn remaining(&self) -> usize {
        self.done.iter().filter(|d| !**d).count()
    }

    /// Bytes already on disk — what the progress bar starts from on a resume.
    pub(crate) fn completed_bytes(&self) -> u64 {
        (0..self.chunks())
            .filter(|i| self.done[*i])
            .map(|i| self.range(i).1)
            .sum()
    }

    fn encode(&self) -> String {
        let done: String = self
            .done
            .iter()
            .map(|d| if *d { '1' } else { '0' })
            .collect();
        format!(
            "{MAGIC}\nsize {}\nchunk {}\noid {}\nvalidator {}\ndone {done}\n",
            self.size,
            self.chunk,
            self.oid.as_deref().unwrap_or("-"),
            self.validator.as_deref().unwrap_or("-"),
        )
    }

    /// Parse a sidecar, or `None` if it is not one this build understands or is internally
    /// inconsistent. Every rejection means the same thing to the caller — throw the partial away and
    /// download the object again — so being strict here costs at most one restart and never risks
    /// writing a chunk at an offset the file was not planned on.
    fn parse(text: &str) -> Option<Self> {
        let mut lines = text.lines();
        if lines.next()? != MAGIC {
            return None;
        }
        let (mut size, mut chunk, mut oid, mut validator, mut done) =
            (None, None, None, None, None);
        for line in lines {
            let (key, value) = line.split_once(' ')?;
            match key {
                // `Last-Modified` contains spaces, so the value is the whole rest of the line.
                "size" => size = Some(value.parse::<u64>().ok()?),
                "chunk" => chunk = Some(value.parse::<u64>().ok()?),
                "oid" => oid = Some(value.to_string()),
                "validator" => validator = Some(value.to_string()),
                "done" => done = Some(value.to_string()),
                _ => return None, // an unknown key means an unknown format
            }
        }
        let (size, chunk, done) = (size?, chunk?, done?);
        if size == 0 || chunk == 0 || done.len() != cell_count(size, chunk) {
            return None; // a grid whose flags don't cover its own file is not a plan
        }
        let flags: Option<Vec<bool>> = done
            .chars()
            .map(|c| match c {
                '0' => Some(false),
                '1' => Some(true),
                _ => None,
            })
            .collect();
        Some(Plan {
            size,
            chunk,
            oid: dash_to_none(oid?),
            validator: dash_to_none(validator?),
            done: flags?,
        })
    }
}

/// `-` is how an absent oid/validator is written, since the format is one value per line and an
/// empty value would be indistinguishable from a truncated line.
fn dash_to_none(s: String) -> Option<String> {
    (s != "-").then_some(s)
}

/// Cells in the grid for `size` bytes at `chunk` bytes each — `size` rounded UP, so the final short
/// cell is included.
fn cell_count(size: u64, chunk: u64) -> usize {
    size.div_ceil(chunk) as usize
}

/// The sidecar's path for a ranged partial.
pub(crate) fn plan_path(tmp: &Path) -> PathBuf {
    let mut name = tmp.as_os_str().to_os_string();
    name.push(".plan");
    PathBuf::from(name)
}

/// Read the sidecar beside `tmp`, or `None` when there is none or it is not usable.
pub(crate) fn load(tmp: &Path) -> Option<Plan> {
    let text = fs::read_to_string(plan_path(tmp)).ok()?;
    Plan::parse(&text)
}

/// Write the sidecar beside `tmp`, atomically.
///
/// Written to a neighbouring temp and `rename`d, so a crash between the two leaves either the old
/// plan or the new one and never a half-written line that would read as "this file has no plan" and
/// throw away a 100 GiB partial. `sync_all` before the rename is what makes that a promise rather
/// than a hope — a rename can otherwise reach the disk before the bytes it points at.
pub(crate) fn save(tmp: &Path, plan: &Plan) -> Result<()> {
    let path = plan_path(tmp);
    let mut staging = path.as_os_str().to_os_string();
    staging.push(".new");
    let staging = PathBuf::from(staging);
    {
        use std::io::Write;
        let mut f = fs::File::create(&staging).map_err(Error::from)?;
        f.write_all(plan.encode().as_bytes()).map_err(Error::from)?;
        f.sync_all().map_err(Error::from)?;
    }
    fs::rename(&staging, &path).map_err(Error::from)?;
    Ok(())
}

/// Remove a ranged partial and its sidecar. Used when the object turns out not to be the one the
/// partial belongs to — the ONE case where deleting downloaded bytes is the correct answer, because
/// keeping them is what splices two uploads into one plausible-sized corrupt file.
pub(crate) fn discard(tmp: &Path) {
    let _ = fs::remove_file(tmp);
    let _ = fs::remove_file(plan_path(tmp));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> Plan {
        Plan::fresh(250, 100, Some("a".repeat(64)), Some("\"v1\"".to_string()))
    }

    /// The grid covers the file exactly: every byte in exactly one cell, the last one short.
    #[test]
    fn the_grid_tiles_the_file() {
        let p = plan();
        assert_eq!(p.chunks(), 3);
        assert_eq!(p.range(0), (0, 100));
        assert_eq!(p.range(1), (100, 100));
        assert_eq!(p.range(2), (200, 50)); // short tail
        let covered: u64 = (0..p.chunks()).map(|i| p.range(i).1).sum();
        assert_eq!(covered, p.size);
        // An exact multiple has no short tail and no empty extra cell.
        let exact = Plan::fresh(200, 100, None, None);
        assert_eq!(exact.chunks(), 2);
        assert_eq!(exact.range(1), (100, 100));
        // A file smaller than one chunk is a single cell — the "not split at all" case.
        assert_eq!(Plan::fresh(1, 100, None, None).chunks(), 1);
    }

    #[test]
    fn completed_bytes_counts_only_finished_cells() {
        let mut p = plan();
        assert_eq!(p.completed_bytes(), 0);
        assert_eq!(p.remaining(), 3);
        p.mark_done(2); // the SHORT cell — a count that assumed uniform cells would say 100
        assert_eq!(p.completed_bytes(), 50);
        assert!(!p.all_done());
        p.mark_done(0);
        p.mark_done(1);
        assert_eq!(p.completed_bytes(), 250);
        assert_eq!(p.remaining(), 0);
        assert!(p.all_done());
    }

    #[test]
    fn a_plan_round_trips_through_the_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let partial = tmp.path().join(".dlr-x");
        let mut p = plan();
        p.mark_done(1);
        save(&partial, &p).unwrap();
        assert_eq!(load(&partial).as_ref(), Some(&p));
        // …including the values that are allowed to be absent, and a validator containing spaces.
        let q = Plan::fresh(10, 4, None, Some("Wed, 21 Oct 2026 07:28:00 GMT".into()));
        save(&partial, &q).unwrap();
        assert_eq!(load(&partial).as_ref(), Some(&q));
        discard(&partial);
        assert!(load(&partial).is_none());
    }

    /// A sidecar that does not describe the file it sits next to must be REFUSED, not repaired: a
    /// flag string shorter than the grid would make `is_done` panic or (worse, if we padded it)
    /// declare a never-downloaded cell finished, which is a hole in the committed blob.
    #[test]
    fn an_inconsistent_sidecar_is_rejected() {
        let good = "infr-dl-plan 1\nsize 250\nchunk 100\noid -\nvalidator -\ndone 010\n";
        assert!(Plan::parse(good).is_some());
        for bad in [
            "infr-dl-plan 2\nsize 250\nchunk 100\noid -\nvalidator -\ndone 010\n", // future version
            "size 250\nchunk 100\noid -\nvalidator -\ndone 010\n",                 // no magic
            "infr-dl-plan 1\nsize 250\nchunk 100\noid -\nvalidator -\ndone 01\n",  // too few flags
            "infr-dl-plan 1\nsize 250\nchunk 100\noid -\nvalidator -\ndone 0101\n", // too many
            "infr-dl-plan 1\nsize 250\nchunk 100\noid -\nvalidator -\ndone 0x0\n", // not a flag
            "infr-dl-plan 1\nsize 0\nchunk 100\noid -\nvalidator -\ndone \n",      // empty object
            "infr-dl-plan 1\nsize 250\nchunk 0\noid -\nvalidator -\ndone 010\n",   // no grid
            "infr-dl-plan 1\nsize 250\nchunk 100\noid -\nvalidator -\n",           // no flags
            "infr-dl-plan 1\nsize 250\nchunk 100\noid -\nvalidator -\ndone 010\nwat 1\n", // unknown
            "infr-dl-plan 1\nsize x\nchunk 100\noid -\nvalidator -\ndone 010\n",   // not a number
        ] {
            assert!(Plan::parse(bad).is_none(), "accepted {bad:?}");
        }
        // A truncated write (the crash the atomic save exists to prevent) is also refused rather
        // than half-believed.
        assert!(Plan::parse("infr-dl-plan 1\nsize 250\nchunk 10").is_none());
    }

    /// The production chunk size, pinned. It is also the split threshold (see [`CHUNK_BYTES`]), so
    /// a later edit changes both at once and is a deliberate decision.
    #[test]
    fn the_production_chunk_is_64_mib() {
        assert_eq!(CHUNK_BYTES, 64 * 1024 * 1024);
    }
}
