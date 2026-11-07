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
//! A `gguf-split` model is several files rather than one, and they are addressed end to end in
//! shard order ([`FileBlockIo::open_shards`]) so a [`BlockExtent`] stays one number. A single-file
//! model is the one-shard case of that, at base 0, and reads exactly as it did before.
//!
//! That freedom is also what makes a block's read CONCURRENT rather than one syscall: an NVMe
//! reaches its bandwidth only with several requests in flight, so one block is split across
//! [`IO_FANOUT`] positioned reads (see [`read_pieces`]). This is what puts the tier's reader on
//! even footing with the mapping it replaces, whose faults the kernel already issues in parallel.
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
    /// Absolute offset in the model's file address space (the tensor-data region's start already
    /// added in). A `gguf-split` model is several files addressed end to end in shard order — see
    /// [`FileBlockIo::open_shards`] — and a single file is the one-shard case of that, at base 0.
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

/// One file of the model, and where its bytes sit in the address space the extents are written in.
struct ShardIo {
    file: File,
    /// Windows synchronous `ReadFile` calls issued through one file handle are serialized by the
    /// kernel even when their byte ranges are independent. Keep additional handles opened on the
    /// same file object generation so `read_pieces` can actually put several requests in flight.
    /// Unix positioned reads do not need this: one descriptor already permits concurrent pread.
    #[cfg(windows)]
    fanout_files: Vec<File>,
    /// Round-robin cursor shared by concurrent block reads. Without it, every block's first piece
    /// lands on the primary handle and every second piece on handle one, serializing same-lane
    /// work even though the remaining handles are idle.
    #[cfg(windows)]
    next_file: std::sync::atomic::AtomicUsize,
    stamp: FileStamp,
    /// Path kept for error messages only — every read and every re-stat goes through `file`.
    path: String,
    /// First offset of the concatenated address space this file serves. Zero for the first shard,
    /// hence for every single-file model.
    base: u64,
}

/// Reads blocks from the open model file (or shard set) with positioned reads.
pub struct FileBlockIo {
    /// The model's files in shard order, each covering `[base, base + stamp.len)` of the address
    /// space [`BlockExtent::offset`] is written in. A single-file model has exactly one.
    shards: Vec<ShardIo>,
}

impl FileBlockIo {
    /// Open a single-file model — the one-shard case of [`Self::open_shards`], with no length to
    /// check because nothing before it computed offsets against one.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = Self::open_one(path.as_ref(), None, 0)?;
        Ok(Self { shards: vec![file] })
    }

    /// Open a `gguf-split` shard set, addressing the files end to end in the given order.
    ///
    /// Each entry is `(path, length)` as `infr_gguf::Gguf::shards` reports it: the length its
    /// tensor offsets were computed against. It is checked, not assumed — a shard that grew or
    /// shrank between the load and this open shifts every offset in every LATER shard, which is
    /// not a read that fails but a read that silently returns other weights. `verify_unchanged`
    /// covers the same file changing after this point; this covers it having changed before.
    pub fn open_shards(shards: &[(impl AsRef<Path>, u64)]) -> Result<Self> {
        let mut out = Vec::with_capacity(shards.len());
        let mut base = 0u64;
        for (path, expect_len) in shards {
            let s = Self::open_one(path.as_ref(), Some(*expect_len), base)?;
            base = base.checked_add(s.stamp.len).ok_or_else(|| {
                Error::Loader("model shard set is larger than u64 bytes".to_string())
            })?;
            out.push(s);
        }
        Ok(Self { shards: out })
    }

    fn open_one(path: &Path, expect_len: Option<u64>, base: u64) -> Result<ShardIo> {
        let file = File::open(path)?;
        let stamp = FileStamp::of(&file)?;
        if let Some(expect) = expect_len {
            if stamp.len != expect {
                return Err(Error::Loader(format!(
                    "model shard changed size before it was streamed: {} is {} bytes, but the \
                     weights were loaded against {expect} — every offset past it would name the \
                     wrong bytes",
                    path.display(),
                    stamp.len
                )));
            }
        }
        #[cfg(windows)]
        let fanout_files = {
            let mut out = Vec::with_capacity(IO_FANOUT.saturating_sub(1));
            for _ in 1..IO_FANOUT {
                let extra = File::open(path)?;
                let extra_stamp = FileStamp::of(&extra)?;
                if extra_stamp != stamp {
                    return Err(Error::Loader(format!(
                        "model file changed while opening parallel readers: {}",
                        path.display()
                    )));
                }
                out.push(extra);
            }
            out
        };
        Ok(ShardIo {
            file,
            #[cfg(windows)]
            fanout_files,
            #[cfg(windows)]
            next_file: std::sync::atomic::AtomicUsize::new(0),
            stamp,
            path: path.display().to_string(),
            base,
        })
    }

    /// The model's total length at open — what a caller bounds-checks its extents against. The sum
    /// over a shard set, which is the space those extents are written in.
    pub fn len(&self) -> u64 {
        self.shards.last().map_or(0, |s| s.base + s.stamp.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fail if any of the model's files changed since it was opened.
    ///
    /// Callers run this at a coarse boundary (once per forward pass), not per read: it is one
    /// `fstat` per shard, and the failure it catches — the model being rewritten mid-generation —
    /// turns what would be silently different weights into an error. The known blind spot is a
    /// same-length in-place write whose mtime is restored; catching that means hashing gigabytes
    /// per check (backlog B30 records the same limit for `WeightWatch`).
    pub fn verify_unchanged(&self) -> Result<()> {
        for s in &self.shards {
            let now = FileStamp::of(&s.file)?;
            if now != s.stamp {
                return Err(Error::Loader(format!(
                    "model file changed while it was being streamed: {} (was {} bytes, now {} \
                     bytes) — weights read after this point would not match the ones already \
                     loaded",
                    s.path, s.stamp.len, now.len
                )));
            }
        }
        Ok(())
    }

    /// The shard holding `[offset, offset + len)`, and that range's offset within it.
    ///
    /// A range that runs off the end of its shard is refused rather than continued into the next
    /// one: the shards are separate files laid end to end for addressing only, so bytes never
    /// actually span the join, and a request that claims they do is a corrupt descriptor.
    fn locate(&self, offset: u64, len: usize) -> Result<(&ShardIo, u64)> {
        // Ordered by `base`, each covering its whole file, so the owner is the last shard starting
        // at or before `offset`. One shard at base 0 → always index 0.
        let idx = self
            .shards
            .partition_point(|s| s.base <= offset)
            .saturating_sub(1);
        let shard = self.shards.get(idx).ok_or_else(|| {
            Error::Loader(format!("read at {offset}+{len} of a model with no files"))
        })?;
        let local = offset - shard.base;
        // `checked_add`: a descriptor carrying a near-`u64::MAX` length must be refused, never wrap
        // the sum into a range that looks in-bounds.
        if local
            .checked_add(len as u64)
            .is_none_or(|e| e > shard.stamp.len)
        {
            return Err(Error::Loader(format!(
                "read at {offset}+{len} runs past the end of {} ({} bytes at offset {})",
                shard.path, shard.stamp.len, shard.base
            )));
        }
        Ok((shard, local))
    }
}

impl ShardIo {
    /// Pick an independent Windows handle for one concurrent piece. The primary descriptor is
    /// lane zero and remains the identity/mtime authority; every extra descriptor was opened and
    /// stamp-checked alongside it. On Unix every lane intentionally shares the positioned-read fd.
    fn read_file(&self, lane: usize) -> &File {
        #[cfg(windows)]
        {
            let _ = lane;
            let n_files = self.fanout_files.len() + 1;
            let selected = self
                .next_file
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                % n_files;
            if selected == 0 {
                &self.file
            } else {
                &self.fanout_files[selected - 1]
            }
        }
        #[cfg(not(windows))]
        {
            let _ = lane;
            &self.file
        }
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
        // Resolve every piece to the file that holds it BEFORE any read runs, so a descriptor
        // naming bytes outside the model fails by name instead of being read from a neighbour.
        let pieces = plan_pieces(&desc.extents)
            .into_iter()
            .map(|p| {
                let (shard, local) = self.locate(p.offset, p.len)?;
                Ok((
                    shard,
                    BlockExtent {
                        offset: local,
                        len: p.len,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        read_pieces(pieces, &mut dst[..need]).map_err(|(shard, e, err)| {
            Error::Loader(format!(
                "reading block {} at {}+{} of {}: {err}",
                desc.id, e.offset, e.len, shard.path
            ))
        })
    }
}

/// How many positioned reads one block is split across.
///
/// A drive reaches its bandwidth on queue depth, not on request size: measured on this workspace's
/// NVMe over 16-128 MiB blocks, one read sustains 1.12-1.40 GiB/s and the device tops out at
/// 2.05 GiB/s, which two to four concurrent reads already reach — eight and sixteen buy nothing and
/// cost threads. The gap between those two figures is the whole reason this exists; a serial reader
/// loses to the mapping it replaces, whose faults the kernel issues in parallel for free.
///
/// Those figures are from Linux on NVMe, which is also where the end-to-end gain was measured
/// (`docs/perf/results.md`). The reads stay CORRECT everywhere — each carries its own offset — but
/// the speedup is not portable-by-construction: on Windows `seek_read` issues `ReadFile` with an
/// `OVERLAPPED` offset, and a handle not opened `FILE_FLAG_OVERLAPPED` has its concurrent
/// operations serialized by the kernel, so the fanout may buy nothing there until the file is
/// opened for overlapped I/O. Unverified on Windows and macOS; see backlog B35.
const IO_FANOUT: usize = 4;

/// A piece below this is not worth its own thread — the fanout is for keeping the drive busy, and
/// small blocks (the norm/bias tensors of a fused group) are latency-bound rather than
/// bandwidth-bound, where an extra thread is pure overhead.
const MIN_CHUNK: usize = 4 << 20;

/// Cut `extents` into the file ranges one block's concurrent reads will each cover, in the order
/// their bytes appear in the destination slot.
///
/// Aims for [`IO_FANOUT`] pieces overall, never smaller than [`MIN_CHUNK`]. Extent boundaries can
/// push the count slightly above the fanout on a fused group; that is harmless — the measured
/// bandwidth curve is flat from 2 to 16 concurrent reads — and keeping every piece inside one
/// extent is what lets each be a single contiguous range of both the file and the slot.
///
/// Separate from the reading so the split can be checked on its own: a piece list that silently
/// stayed one element would leave every read serial and every byte still correct.
fn plan_pieces(extents: &[BlockExtent]) -> Vec<BlockExtent> {
    let total: usize = extents.iter().map(|e| e.len).sum();
    let target = total.div_ceil(IO_FANOUT).max(MIN_CHUNK);
    let mut out = Vec::new();
    for e in extents {
        let mut at = 0usize;
        while at < e.len {
            let n = target.min(e.len - at);
            out.push(BlockExtent {
                offset: e.offset + at as u64,
                len: n,
            });
            at += n;
        }
    }
    out
}

/// Fill `dst` with `pieces`, in order, using up to [`IO_FANOUT`] concurrent positioned reads.
///
/// Each piece is already resolved to the shard that holds it and carries that shard's LOCAL offset
/// (see [`FileBlockIo::locate`]); their total must be exactly `dst.len()`, which the caller has
/// already length-checked. On failure returns the piece that failed and its shard alongside the
/// error — a sub-range of one extent, which names the failing bytes more precisely than the extent
/// enclosing them would.
///
/// Ordering is preserved by construction rather than by sequencing the reads: each piece is handed
/// a disjoint `&mut` sub-slice of `dst` carved at the position that piece belongs at, so the reads
/// may complete in any order and still land where they must. Nothing is shared between them — the
/// files are read at explicit offsets, so there is no cursor to race on.
fn read_pieces<'a>(
    pieces: Vec<(&'a ShardIo, BlockExtent)>,
    dst: &mut [u8],
) -> std::result::Result<(), (&'a ShardIo, BlockExtent, std::io::Error)> {
    debug_assert_eq!(
        pieces.iter().map(|(_, e)| e.len).sum::<usize>(),
        dst.len(),
        "dst must be exactly the pieces' total"
    );
    let mut rest = dst;
    let mut placed: Vec<(usize, &ShardIo, BlockExtent, &mut [u8])> =
        Vec::with_capacity(pieces.len());
    for (lane, (shard, e)) in pieces.into_iter().enumerate() {
        let (mine, tail) = rest.split_at_mut(e.len);
        rest = tail;
        placed.push((lane, shard, e, mine));
    }
    debug_assert!(rest.is_empty(), "the pieces must cover dst exactly");

    // The calling thread takes the last piece rather than parking on a join, so a single-piece
    // block spawns nothing at all and the common case costs no threads.
    let Some((last_lane, last_shard, last, last_dst)) = placed.pop() else {
        return Ok(());
    };
    let mut first_err = None;
    std::thread::scope(|s| {
        let handles: Vec<_> = placed
            .into_iter()
            .map(|(lane, shard, e, buf)| {
                s.spawn(move || {
                    read_exact_at(shard.read_file(lane), e.offset, buf)
                        .map_err(|err| (shard, e, err))
                })
            })
            .collect();
        first_err = read_exact_at(last_shard.read_file(last_lane), last.offset, last_dst)
            .map_err(|err| (last_shard, last, err))
            .err();
        for h in handles {
            // A panicking reader is a bug in this function, not an I/O condition: propagate it
            // rather than reporting a block that was never filled as read.
            if let Err(e) = h.join().expect("block reader thread panicked") {
                first_err.get_or_insert(e);
            }
        }
    });
    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
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

    /// An extent naming bytes the file does not have must error rather than silently leaving the
    /// slot half-filled. It is refused by `locate` before any read is issued: with a shard set the
    /// next file's bytes sit immediately after this one in the address space, so "runs off the end
    /// of shard k" has to be a refusal and not something the reader could paper over.
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
        assert!(
            err.to_string().contains("runs past the end of"),
            "unexpected: {err}"
        );
    }

    /// The other half of that: an extent that WAS in range when the model was loaded, on a file
    /// that has since been truncated. `locate` passes (it checks the length stamped at open), so
    /// this is the path where `read_exact_at`'s EOF branch is what stops a half-filled slot.
    #[test]
    fn a_file_truncated_after_open_short_reads() {
        let (f, _) = ramp_file(512);
        let io = FileBlockIo::open(f.path()).expect("open");
        std::fs::File::create(f.path())
            .expect("truncate")
            .set_len(64)
            .expect("set_len");
        let desc = BlockDesc {
            id: 4,
            extents: vec![BlockExtent {
                offset: 256,
                len: 128,
            }],
        };
        let mut dst = vec![0u8; 128];
        let err = io.read_block(&desc, &mut dst).expect_err("must reject");
        assert!(err.to_string().contains("short read"), "unexpected: {err}");
    }

    /// Two shards addressed end to end: an extent past the first file's length must be read from
    /// the SECOND file, at its own offset. Both files hold the same ramp pattern, so an extent
    /// resolved against the wrong file would still return readable bytes of the right length —
    /// what tells them apart here is that shard 2's ramp is offset by one.
    #[test]
    fn open_shards_reads_across_the_join() {
        let (a, a_bytes) = ramp_file(4096);
        let b_bytes: Vec<u8> = (0..4096usize).map(|i| (i + 1) as u8).collect();
        let mut b = tempfile::NamedTempFile::new().expect("temp file");
        b.write_all(&b_bytes).expect("write");
        b.flush().expect("flush");

        let io = FileBlockIo::open_shards(&[(a.path(), 4096), (b.path(), 4096)]).expect("open set");
        assert_eq!(
            io.len(),
            8192,
            "the set is addressed as one 8192-byte space"
        );

        // One extent inside shard 1, one inside shard 2, in one block — the fused-group shape.
        let desc = BlockDesc {
            id: 9,
            extents: vec![
                BlockExtent {
                    offset: 100,
                    len: 64,
                },
                BlockExtent {
                    offset: 4096 + 100,
                    len: 64,
                },
            ],
        };
        let mut dst = vec![0u8; 128];
        io.read_block(&desc, &mut dst).expect("read");
        assert_eq!(&dst[..64], &a_bytes[100..164]);
        assert_eq!(&dst[64..], &b_bytes[100..164]);
    }

    /// A read that would run off the end of one shard is refused, never continued into the next —
    /// the shards are separate files laid end to end for ADDRESSING, and no tensor spans the join.
    #[test]
    fn a_read_never_spans_the_shard_join() {
        let (a, _) = ramp_file(4096);
        let (b, _) = ramp_file(4096);
        let io = FileBlockIo::open_shards(&[(a.path(), 4096), (b.path(), 4096)]).expect("open set");
        let desc = BlockDesc {
            id: 3,
            extents: vec![BlockExtent {
                offset: 4032,
                len: 128,
            }],
        };
        let mut dst = vec![0u8; 128];
        let err = io.read_block(&desc, &mut dst).expect_err("must reject");
        assert!(
            err.to_string().contains("runs past the end of"),
            "unexpected: {err}"
        );
    }

    /// A shard whose length is no longer the one the weights were loaded against shifts every
    /// offset in every LATER shard — not a read that fails, a read that returns other weights. It
    /// must be refused at open, naming the file and both lengths.
    #[test]
    fn open_shards_refuses_a_shard_of_the_wrong_length() {
        let (a, _) = ramp_file(4096);
        let (b, _) = ramp_file(4096);
        let msg = match FileBlockIo::open_shards(&[(a.path(), 2048), (b.path(), 4096)]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a resized shard must be refused"),
        };
        assert!(
            msg.contains("changed size") && msg.contains("4096") && msg.contains("2048"),
            "unexpected: {msg}"
        );
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

    /// The split itself, checked without touching a disk: a block big enough to be worth
    /// parallelising must come back as several pieces that tile its extent exactly and in order.
    /// Asserting only on the bytes a read produced would pass just as well if the fanout silently
    /// collapsed to one piece and every read stayed serial.
    #[test]
    fn a_big_block_is_cut_into_concurrent_pieces() {
        // Sized against MIN_CHUNK alone. Deriving it from IO_FANOUT and then asserting the count
        // against IO_FANOUT would move both sides of the comparison together, so it would hold for
        // any fanout — including 1, which is the case this exists to rule out.
        let len = 16 * MIN_CHUNK;
        let pieces = plan_pieces(&[BlockExtent { offset: 100, len }]);
        assert!(
            pieces.len() > 1,
            "a {len}-byte block must be split to be read concurrently, got {}",
            pieces.len()
        );
        let mut at = 100u64;
        for p in &pieces {
            assert_eq!(p.offset, at, "pieces must tile the extent in order");
            at += p.len as u64;
        }
        assert_eq!(at, 100 + len as u64, "pieces must cover the whole extent");
        assert!(
            pieces.iter().all(|p| p.len >= MIN_CHUNK),
            "no piece may fall below the minimum: {pieces:?}"
        );
    }

    /// The other half of the same rule: a block that is only worth one read must not be split, or
    /// every small tensor pays thread-spawn overhead to no purpose.
    ///
    /// Sized as a concrete small tensor — 64 KiB, the order of a norm vector — rather than as
    /// `MIN_CHUNK - 1`, which sits below the threshold by construction and so would hold no matter
    /// how small the minimum became.
    #[test]
    fn a_small_block_stays_one_read() {
        let pieces = plan_pieces(&[BlockExtent {
            offset: 0,
            len: 64 << 10,
        }]);
        assert_eq!(pieces.len(), 1, "small block must not be split: {pieces:?}");
    }

    /// A piece never straddles an extent boundary — it could not, since the two sides come from
    /// different offsets in the file. A fused group of large components splits within each.
    #[test]
    fn pieces_never_straddle_an_extent_boundary() {
        // Lengths deliberately NOT a whole multiple of the piece size: if they divided evenly, the
        // last piece of each extent would land exactly on the boundary and a splitter that ignored
        // the extent's end would produce the same list as one that respected it.
        let a = BlockExtent {
            offset: 0,
            len: 5 * MIN_CHUNK + 12345,
        };
        let b = BlockExtent {
            offset: 1 << 30,
            len: 5 * MIN_CHUNK + 999,
        };
        let pieces = plan_pieces(&[a, b]);
        for p in &pieces {
            let end = p.offset + p.len as u64;
            let within_a = p.offset >= a.offset && end <= a.offset + a.len as u64;
            let within_b = p.offset >= b.offset && end <= b.offset + b.len as u64;
            assert!(within_a || within_b, "piece {p:?} straddles a boundary");
        }
        assert_eq!(
            pieces.iter().map(|p| p.len).sum::<usize>(),
            a.len + b.len,
            "pieces must cover both extents"
        );
    }

    /// End to end over a real file, at a size that actually engages the concurrent path: the bytes
    /// must be identical to what a serial read would have produced, and the two extents must still
    /// land in the order they were listed even though their pieces complete in any order.
    #[test]
    fn a_concurrently_read_block_matches_its_file() {
        // Two extents, each several pieces, at offsets that differ modulo 256 so a swapped piece
        // shows up as a value mismatch rather than a coincidentally equal ramp.
        let each = 3 * MIN_CHUNK;
        let (f, bytes) = ramp_file(2 * each + 1000);
        let io = FileBlockIo::open(f.path()).expect("open");
        let desc = BlockDesc {
            id: 9,
            extents: vec![
                BlockExtent {
                    offset: each as u64 + 1000,
                    len: each,
                },
                BlockExtent {
                    offset: 0,
                    len: each,
                },
            ],
        };
        assert!(
            plan_pieces(&desc.extents).len() > 1,
            "this test must engage the concurrent path"
        );
        let mut dst = vec![0u8; 2 * each];
        io.read_block(&desc, &mut dst).expect("read");
        assert_eq!(&dst[..each], &bytes[each + 1000..2 * each + 1000]);
        assert_eq!(&dst[each..], &bytes[..each]);
    }

    /// A failure in ANY piece must surface, including one the calling thread did not read itself —
    /// a spawned reader whose error was dropped would leave a partly-filled slot reported as read.
    ///
    /// The FIRST extent is the one that runs off the end and the last is in range, because the
    /// calling thread keeps the last piece: a failure placed at the end would be caught by the
    /// caller's own read and would say nothing about whether a spawned reader's error is joined.
    ///
    /// The file is TRUNCATED after opening so the descriptor stays inside the length `locate`
    /// checks (an extent naming bytes past the stamped length never reaches a reader at all) while
    /// the read itself still hits EOF — which is the condition a spawned reader has to report.
    #[test]
    fn a_failing_piece_fails_the_block() {
        let each = 2 * MIN_CHUNK;
        let (f, _) = ramp_file(2 * each);
        let io = FileBlockIo::open(f.path()).expect("open");
        std::fs::File::options()
            .write(true)
            .open(f.path())
            .expect("reopen")
            .set_len(each as u64)
            .expect("truncate");
        let desc = BlockDesc {
            id: 11,
            extents: vec![
                BlockExtent {
                    offset: each as u64, // starts exactly at EOF
                    len: each,
                },
                BlockExtent {
                    offset: 0,
                    len: each,
                },
            ],
        };
        let pieces = plan_pieces(&desc.extents);
        let last = pieces.last().expect("pieces");
        assert!(
            last.offset + last.len as u64 <= each as u64,
            "the caller's own piece must be the readable one, else this proves nothing"
        );
        let mut dst = vec![0u8; 2 * each];
        let err = io.read_block(&desc, &mut dst).expect_err("must reject");
        assert!(err.to_string().contains("short read"), "unexpected: {err}");
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
