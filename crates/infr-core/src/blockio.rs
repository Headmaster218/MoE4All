//! Reading one pageable block's bytes off the model file — the bottom tier of the weight pager
//! (`docs/disk-streaming-plan.md` §3.2, §3.5).
//!
//! What a block IS lives in [`crate::pager`] (an opaque `BlockId` plus residency bookkeeping);
//! what a block's bytes ARE lives here: a [`BlockDesc`] naming one or more byte ranges of the
//! model file, in upload order, and a [`BlockIo`] that fills a caller's slot from them.
//!
//! Positioned reads, never a shared cursor: [`FileBlockIo`] reads at an explicit offset
//! (`pread`/`seek_read`), so any number of reader threads share one open file with no seek race
//! and no lock. This is the deliberate alternative to reaching through the GGUF mmap — the page
//! cache evicts by recency, which is the pathological policy for the cyclic sweep a forward pass
//! performs (measured: `docs/perf/results.md`, "Weights that do not fit memory").
//!
//! The file can also change under a live run. A mapping makes that a `SIGBUS` or silently
//! different bytes; explicit reads make it detectable, so [`FileBlockIo`] stamps the file at open
//! and [`FileBlockIo::verify_unchanged`] re-checks the SAME descriptor (never the path — `infr
//! pull` renames into place, which leaves this fd on the intact old inode; see backlog B30).

use crate::error::{Error, Result};
use crate::pager::BlockId;
use std::fs::File;
use std::path::Path;

/// One contiguous byte range of the model file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockExtent {
    /// Absolute offset in the file (the tensor-data region's start already added in).
    pub offset: u64,
    pub len: usize,
}

/// A block's identity plus where its bytes live, in the order they must be laid down.
///
/// A fused weight group (qkv, gate+up) lists one extent per component tensor, so the concatenation
/// happens directly into the destination slot and is never materialized on the side — the same
/// property `infr_vulkan::pager::DenseSource`'s segment list has for the mmap path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockDesc {
    pub id: BlockId,
    pub extents: Vec<BlockExtent>,
}

impl BlockDesc {
    /// Total bytes this block occupies in a slot — the sum of its extents. Not stored alongside
    /// them: a stored total is a second source of truth that can disagree with the extents it
    /// claims to describe, and every caller that needs it has the extents in hand.
    pub fn nbytes(&self) -> usize {
        self.extents.iter().map(|e| e.len).sum()
    }
}

/// Fills a slot with one block's bytes. The tier's only I/O surface, so a test can drive the whole
/// pager off an in-memory implementation (see `infr-testkit`) and inject short reads and errors
/// that a real file will not produce on demand.
pub trait BlockIo: Send + Sync {
    /// Write `desc`'s extents, in order, into the front of `dst`.
    ///
    /// `dst` may be longer than the block (slots are a padded stride); bytes past `desc.nbytes()`
    /// are left alone. Fails if `dst` is SHORTER — a truncated block is silent wrong output, so it
    /// is never a partial success.
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()>;
}

/// The file identity [`FileBlockIo`] stamps at open, to notice the model being replaced under a
/// live run. Read from the held descriptor, so it follows the inode this reader actually reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    /// Modification time as (secs, nanos) since the epoch. `None` when the platform's metadata
    /// carries no mtime, in which case only the length is compared.
    mtime: Option<(i64, u32)>,
}

impl FileStamp {
    fn of(file: &File) -> Result<Self> {
        let md = file.metadata()?;
        let mtime = md.modified().ok().map(|t| {
            match t.duration_since(std::time::UNIX_EPOCH) {
                Ok(d) => (d.as_secs() as i64, d.subsec_nanos()),
                // Pre-epoch mtimes are legal; carry them as a negative second count rather than
                // dropping the stamp (a dropped stamp silently weakens the check to length only).
                Err(e) => {
                    let d = e.duration();
                    (-(d.as_secs() as i64), d.subsec_nanos())
                }
            }
        });
        Ok(Self {
            len: md.len(),
            mtime,
        })
    }
}

/// Reads blocks from an open model file with positioned reads.
pub struct FileBlockIo {
    file: File,
    stamp: FileStamp,
    /// Path kept for error messages only — every read and every re-stat goes through `file`.
    path: String,
}

impl FileBlockIo {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        Ok(Self {
            stamp: FileStamp::of(&file)?,
            file,
            path: path.display().to_string(),
        })
    }

    /// The file's length at open — what a caller bounds-checks its extents against.
    pub fn len(&self) -> u64 {
        self.stamp.len
    }

    pub fn is_empty(&self) -> bool {
        self.stamp.len == 0
    }

    /// Fail if the file changed since [`Self::open`].
    ///
    /// Callers run this at a coarse boundary (once per forward pass), not per read: it is one
    /// `fstat`, and the failure it catches — the model being rewritten mid-generation — turns what
    /// would be silently different weights into an error. The known blind spot is a same-length
    /// in-place write whose mtime is restored; catching that means hashing gigabytes per check
    /// (backlog B30 records the same limit for `WeightWatch`).
    pub fn verify_unchanged(&self) -> Result<()> {
        let now = FileStamp::of(&self.file)?;
        if now == self.stamp {
            return Ok(());
        }
        Err(Error::Loader(format!(
            "model file changed while it was being streamed: {} (was {} bytes, now {} bytes) — \
             weights read after this point would not match the ones already loaded",
            self.path, self.stamp.len, now.len
        )))
    }
}

impl BlockIo for FileBlockIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
        let need = desc.nbytes();
        if dst.len() < need {
            return Err(Error::backend(format!(
                "block {} needs {need} bytes, slot holds {}",
                desc.id,
                dst.len()
            )));
        }
        let mut at = 0usize;
        for e in &desc.extents {
            read_exact_at(&self.file, e.offset, &mut dst[at..at + e.len]).map_err(|err| {
                Error::Loader(format!(
                    "reading block {} at {}+{} of {}: {err}",
                    desc.id, e.offset, e.len, self.path
                ))
            })?;
            at += e.len;
        }
        Ok(())
    }
}

/// Read exactly `buf.len()` bytes at `offset`, looping over short reads.
///
/// A positioned read is allowed to return fewer bytes than asked for even mid-file, so the loop is
/// not optional; a partially-filled slot is the silent-wrong-output case this whole tier exists to
/// avoid. Hitting EOF before the buffer is full is an error, not a short success.
fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;
    #[cfg(windows)]
    use std::os::windows::fs::FileExt;

    let mut done = 0usize;
    while done < buf.len() {
        #[cfg(unix)]
        let n = file.read_at(&mut buf[done..], offset + done as u64)?;
        #[cfg(windows)]
        let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
        #[cfg(not(any(unix, windows)))]
        compile_error!("positioned reads need a unix or windows FileExt");
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "short read: wanted {} bytes at {offset}, got {done}",
                    buf.len()
                ),
            ));
        }
        done += n;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A file of `n` bytes whose byte at index `i` is `i as u8` — so any wrong offset, any
    /// mis-ordered extent and any short read shows up as a value mismatch rather than as a length
    /// that happens to be right. The pattern repeats every 256 bytes, so a test that must
    /// distinguish two ranges has to pick offsets that differ modulo 256.
    fn ramp_file(n: usize) -> (tempfile::NamedTempFile, Vec<u8>) {
        let bytes: Vec<u8> = (0..n).map(|i| i as u8).collect();
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(&bytes).expect("write");
        f.flush().expect("flush");
        (f, bytes)
    }

    #[test]
    fn reads_one_extent_at_its_offset() {
        let (f, bytes) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        let desc = BlockDesc {
            id: 7,
            extents: vec![BlockExtent {
                offset: 1000,
                len: 256,
            }],
        };
        let mut dst = vec![0u8; 256];
        io.read_block(&desc, &mut dst).expect("read");
        assert_eq!(dst, bytes[1000..1256]);
    }

    /// A fused group: the extents must land back to back in the order listed, which is what makes
    /// the concatenation free. Reversing the two extents here must NOT produce the same bytes.
    #[test]
    fn concatenates_extents_in_order() {
        let (f, bytes) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        // Offsets differ modulo 256, so the two ranges hold different bytes and the reversed
        // order below cannot coincidentally match.
        let fwd = BlockDesc {
            id: 1,
            extents: vec![
                BlockExtent {
                    offset: 2048,
                    len: 64,
                },
                BlockExtent {
                    offset: 100,
                    len: 64,
                },
            ],
        };
        let mut dst = vec![0u8; 128];
        io.read_block(&fwd, &mut dst).expect("read");
        assert_eq!(&dst[..64], &bytes[2048..2112]);
        assert_eq!(&dst[64..], &bytes[100..164]);

        let rev = BlockDesc {
            id: 1,
            extents: fwd.extents.iter().rev().copied().collect(),
        };
        let mut other = vec![0u8; 128];
        io.read_block(&rev, &mut other).expect("read");
        assert_ne!(dst, other, "extent order must decide the layout");
    }

    /// A slot is a PADDED stride, so a longer destination is normal and the tail must be left
    /// alone — the pager reuses slots, and clobbering past the block would corrupt nothing today
    /// but would hide a sizing bug tomorrow.
    #[test]
    fn a_longer_slot_keeps_its_tail() {
        let (f, _) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        let desc = BlockDesc {
            id: 0,
            extents: vec![BlockExtent { offset: 0, len: 32 }],
        };
        let mut dst = vec![0xAAu8; 64];
        io.read_block(&desc, &mut dst).expect("read");
        assert_eq!(&dst[32..], &[0xAAu8; 32], "padding was overwritten");
    }

    /// A slot too small is a caller bug that must fail, not truncate: a short block is wrong
    /// output with no error attached.
    #[test]
    fn a_short_slot_is_rejected() {
        let (f, _) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        let desc = BlockDesc {
            id: 3,
            extents: vec![BlockExtent {
                offset: 0,
                len: 100,
            }],
        };
        let mut dst = vec![0u8; 99];
        let err = io.read_block(&desc, &mut dst).expect_err("must reject");
        assert!(
            err.to_string().contains("slot holds 99"),
            "unexpected error: {err}"
        );
    }

    /// Reading past EOF must error rather than silently leaving the slot half-filled.
    #[test]
    fn reading_past_the_end_errors() {
        let (f, _) = ramp_file(512);
        let io = FileBlockIo::open(f.path()).expect("open");
        let desc = BlockDesc {
            id: 4,
            extents: vec![BlockExtent {
                offset: 256,
                len: 512,
            }],
        };
        let mut dst = vec![0u8; 512];
        let err = io.read_block(&desc, &mut dst).expect_err("must reject");
        assert!(err.to_string().contains("short read"), "unexpected: {err}");
    }

    /// The file-replaced check: unchanged is silent, and a rewrite through the SAME path (the
    /// `cp new.gguf live.gguf` shape, which truncates in place) is caught.
    #[test]
    fn verify_unchanged_catches_a_rewrite() {
        let (f, _) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        io.verify_unchanged().expect("unchanged file must pass");

        std::fs::write(f.path(), vec![0u8; 8192]).expect("rewrite");
        let err = io.verify_unchanged().expect_err("rewrite must be caught");
        assert!(
            err.to_string()
                .contains("changed while it was being streamed"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn nbytes_is_the_extent_sum() {
        let desc = BlockDesc {
            id: 0,
            extents: vec![
                BlockExtent { offset: 0, len: 10 },
                BlockExtent {
                    offset: 100,
                    len: 5,
                },
            ],
        };
        assert_eq!(desc.nbytes(), 15);
    }
}
