//! GGUF loader — `WeightSource` impl.
//!
//! Parses the GGUF binary format (little-endian) by mmap-ping the file and
//! walking a byte cursor through header → metadata KV pairs → tensor directory.
//! Quantised weight blocks are returned as-is; the backend owns dequantisation.
//!
//! References:
//!   `~/Projects/llama.cpp/ggml/include/gguf.h`  (format spec)
//!   `~/Projects/llama.cpp/ggml/include/ggml.h`  (ggml_type enum)
//!   `~/Projects/llama.cpp/conversion/diffusion_gemma.py` (tensor names / keys)

pub mod dequant;
pub mod watch;

use infr_core::{
    error::{Error, Result},
    gguf_split::{parse_shard, shard_set},
    loader::{MetaValue, Metadata, TensorInfo},
    tensor::DType,
    WeightSource,
};
use memmap2::Mmap;
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

// ─── constants ────────────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // b"GGUF" little-endian
const DEFAULT_ALIGNMENT: usize = 32; // GGUF_DEFAULT_ALIGNMENT

/// Upper bound on `general.alignment`.
///
/// The lower bound (positive, power of two) is not enough on its own: `1 << 62` satisfies
/// both and then makes `r.pos.div_ceil(alignment) * alignment` overflow — a panic in a debug
/// build, a wrapped-to-garbage `data_region_start` in a release one, which silently reads
/// every tensor from the wrong file offset. 16 MiB is arbitrary-but-justified: the spec's
/// default (and every real producer) uses 32, GPU/page-level constraints top out in the
/// kilobytes, so this is ~19 binary orders of magnitude of headroom over anything legitimate,
/// while keeping `div_ceil(a) * a` far below `usize::MAX` for any in-file cursor position.
const MAX_ALIGNMENT: usize = 16 << 20;

/// Maximum number of dimensions a GGUF tensor may declare — ggml's `GGML_MAX_DIMS`.
///
/// `n_dims` is read straight from the file as a `u32`. Without this bound a bogus value just
/// runs the shape loop until `read_u64` hits EOF, so a malformed file reports "unexpected end
/// of file" instead of the actual defect, and a value near `u32::MAX` spends the whole read
/// budget getting there. Naming the real problem at the point it is detectable is the whole
/// win; the reservation is already clamped by `r.remaining()`, so this is about diagnostics.
const MAX_TENSOR_DIMS: usize = 4;

/// Hard ceiling on how deeply metadata ARRAY values (GGUF type 9) may nest.
///
/// `read_meta_value` is recursive: an ARRAY reads an `elem_type` and then parses `count`
/// values of that type, and `elem_type` may itself be ARRAY. Each nesting level costs the
/// attacker only 12 bytes of file (u32 elem_type + u64 count) but one full stack frame of
/// ours, so a ~1 MB crafted GGUF nests ~85 000 deep and blows the stack — a SIGSEGV/abort
/// the caller cannot catch, not a `Result` it can report. Since `Gguf::open` is reachable
/// from "open this model I downloaded", that is a remote DoS on the whole process.
///
/// 64 is far above anything real: GGUF metadata in the wild is at most 2 levels deep
/// (arrays of strings, arrays of arrays for token-type tables), and llama.cpp's own reader
/// uses a comparable bound. Well-formed files never see this guard.
const MAX_META_DEPTH: usize = 64;

/// The `gguf-split` metadata keys (llama.cpp `LLM_KV_SPLIT_*`), written on EVERY shard of a split
/// model. `split.no` is ZERO-based while the filename's `-NNNNN-` field is one-based; the two must
/// agree, which is one of the checks [`Gguf::open`] makes.
const KEY_SPLIT_NO: &str = "split.no";
const KEY_SPLIT_COUNT: &str = "split.count";
const KEY_SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

// ─── public struct ────────────────────────────────────────────────────────────

/// A parsed, mmap-backed GGUF model — one file, or a whole `gguf-split` shard set.
///
/// Each `Mmap` handle keeps its shard's backing memory alive for the lifetime of this struct;
/// `tensor_bytes` returns slices directly into the mapping of the shard that holds the tensor.
pub struct Gguf {
    /// The files backing this model, in shard order (`split.no` 0, 1, …), each mapped whole. A
    /// single-file model has exactly one, and then every field below reads as it did before shard
    /// sets existed.
    shards: Vec<ShardFile>,
    /// Metadata from the FIRST shard, which is the only one carrying the model's own keys —
    /// llama.cpp's `gguf-split` copies the source KV store into shard 1 and writes only the three
    /// `split.*` keys into the rest.
    metadata: Metadata,
    /// Every shard's tensors, concatenated in shard order. `TensorInfo::offset` is an offset into
    /// the SET's concatenated address space (see [`Gguf::tensor_file_range`]), so it alone locates
    /// the tensor: the shard is the one whose byte range contains it.
    tensors: Vec<TensorInfo>,
}

/// One shard's mapping and where its bytes sit in the set's concatenated address space.
struct ShardFile {
    mmap: Arc<Mmap>,
    /// The path this shard was opened from, for a reader that wants the FILE rather than this
    /// mapping — see [`Gguf::shards`].
    path: PathBuf,
    /// Offset of this shard's first byte in the concatenated address space: the sum of every
    /// earlier shard's file length. Zero for the first shard, hence for every single-file model.
    file_base: u64,
}

/// An owning, ref-counted view of a tensor's bytes in the mmap'd file — a zero-copy `[u8]` slice that
/// keeps the whole `Mmap` alive via `Arc`, so it can outlive the borrow of `&Gguf` (e.g. a CPU
/// backend buffer that reads weights straight from the mapping with no `memcpy`).
#[derive(Clone)]
pub struct TensorBytes {
    mmap: Arc<Mmap>,
    off: usize,
    len: usize,
    /// The owning shard's [`ShardFile::file_base`], so [`Self::file_range`] can report the range in
    /// the concatenated address space rather than in one shard's file.
    file_base: u64,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl std::ops::Deref for TensorBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.mmap[self.off..self.off + self.len]
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl AsRef<[u8]> for TensorBytes {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl TensorBytes {
    /// This view's `(offset, len)` in the model's concatenated file address space, for a reader
    /// that opens the files instead of using the mapping (the pager's disk tier — see
    /// [`Gguf::tensor_file_range`], which reports the same numbers by tensor name).
    ///
    /// For a single-file model the offset is a plain file offset, because `Gguf::open` maps the
    /// whole file from byte 0 so the mapping and the file agree; `mapping_covers_the_whole_file`
    /// pins that. For a shard set it is that offset PLUS the lengths of the earlier shards, which
    /// is the same space `infr_core::blockio::FileBlockIo::open_shards` reads.
    pub fn file_range(&self) -> (u64, usize) {
        (self.file_base + self.off as u64, self.len)
    }
}

// ─── byte-cursor ──────────────────────────────────────────────────────────────

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Bytes left to read from the current cursor. Used to clamp attacker-controlled
    /// `with_capacity` reservations to a sane upper bound (each element needs ≥1 byte).
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn ensure(&self, n: usize) -> Result<()> {
        // `checked_add`: a bogus near-`usize::MAX` length prefix must error, never
        // overflow `pos + n` (debug panic / release wrap → out-of-bounds slice panic).
        let end = self.pos.checked_add(n).ok_or_else(|| {
            Error::Loader(format!(
                "GGUF: length {n} at offset {} overflows usize",
                self.pos
            ))
        })?;
        if end > self.buf.len() {
            Err(Error::Loader(format!(
                "GGUF: unexpected EOF at offset {} (need {n} more bytes, file is {} bytes)",
                self.pos,
                self.buf.len()
            )))
        } else {
            Ok(())
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        self.ensure(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        self.ensure(2)?;
        let v = u16::from_le_bytes(self.buf[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        self.ensure(4)?;
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        self.ensure(8)?;
        let v = u64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        self.ensure(4)?;
        let v = f32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_f64(&mut self) -> Result<f64> {
        self.ensure(8)?;
        let v = f64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    /// Read a GGUF string: u64 length prefix then UTF-8 bytes (no NUL).
    fn read_gguf_str(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        self.ensure(len)?;
        let s = std::str::from_utf8(&self.buf[self.pos..self.pos + len])
            .map_err(|e| {
                Error::Loader(format!(
                    "GGUF: invalid UTF-8 string at offset {}: {e}",
                    self.pos
                ))
            })?
            .to_owned();
        self.pos += len;
        Ok(s)
    }

    /// Parse a metadata value given its GGUF type tag.
    ///
    /// Entry point for a top-level KV value; the recursion (nested ARRAYs) runs through
    /// [`Self::read_meta_value_at_depth`], which refuses to nest past [`MAX_META_DEPTH`].
    fn read_meta_value(&mut self, vtype: u32) -> Result<MetaValue> {
        self.read_meta_value_at_depth(vtype, 0)
    }

    /// Recursively parse a metadata value given its GGUF type tag, tracking how many ARRAY
    /// levels deep we already are so a crafted file cannot recurse us off the stack. See
    /// [`MAX_META_DEPTH`] for why the ceiling exists and why 64 is safe for real files.
    fn read_meta_value_at_depth(&mut self, vtype: u32, depth: usize) -> Result<MetaValue> {
        match vtype {
            0 => Ok(MetaValue::U64(self.read_u8()? as u64)), // UINT8
            1 => Ok(MetaValue::I64(self.read_i8()? as i64)), // INT8
            2 => Ok(MetaValue::U64(self.read_u16()? as u64)), // UINT16
            3 => Ok(MetaValue::I64(self.read_i16()? as i64)), // INT16
            4 => Ok(MetaValue::U64(self.read_u32()? as u64)), // UINT32
            5 => Ok(MetaValue::I64(self.read_i32()? as i64)), // INT32
            6 => Ok(MetaValue::F64(self.read_f32()? as f64)), // FLOAT32
            7 => Ok(MetaValue::Bool(self.read_bool()?)),     // BOOL
            8 => Ok(MetaValue::Str(self.read_gguf_str()?)),  // STRING
            9 => {
                // ARRAY: u32 elem_type, u64 count, then count × elem
                // Refuse before descending: an ARRAY-of-ARRAY chain costs 12 bytes per level
                // on disk and one stack frame here, so without this the parse dies by
                // stack overflow (uncatchable) instead of returning an error.
                if depth >= MAX_META_DEPTH {
                    return Err(Error::Loader(format!(
                        "GGUF: metadata array nesting exceeds the maximum depth of {MAX_META_DEPTH} at offset {}",
                        self.pos
                    )));
                }
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                // Clamp the reservation: an attacker-controlled `count` on a tiny file
                // must not trigger a multi-GB `with_capacity` abort before we read.
                let mut arr = Vec::with_capacity(count.min(self.remaining()));
                for _ in 0..count {
                    arr.push(self.read_meta_value_at_depth(elem_type, depth + 1)?);
                }
                Ok(MetaValue::Arr(arr))
            }
            10 => Ok(MetaValue::U64(self.read_u64()?)), // UINT64
            11 => Ok(MetaValue::I64(self.read_i64()?)), // INT64
            12 => Ok(MetaValue::F64(self.read_f64()?)), // FLOAT64
            _ => Err(Error::Loader(format!("GGUF: unknown value type {vtype}"))),
        }
    }
}

// ─── ggml_type → DType + block sizing ────────────────────────────────────────

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn ggml_type_to_dtype(t: u32) -> Result<DType> {
    match t {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::Q4_0), // GGML_TYPE_Q4_0: 32 elems, 18 bytes/block
        3 => Ok(DType::Q4_1), // GGML_TYPE_Q4_1: 32 elems, 20 bytes/block
        // 4,5 removed from ggml
        6 => Ok(DType::Q5_0),    // GGML_TYPE_Q5_0: 32 elems, 22 bytes/block
        7 => Ok(DType::Q5_1),    // GGML_TYPE_Q5_1: 32 elems, 24 bytes/block
        8 => Ok(DType::Q8_0),    // GGML_TYPE_Q8_0: 32 elems, 34 bytes/block
        10 => Ok(DType::Q2K),    // GGML_TYPE_Q2_K: 256 elems, 84 bytes/block
        11 => Ok(DType::Q3K),    // GGML_TYPE_Q3_K: 256 elems, 110 bytes/block
        12 => Ok(DType::Q4K),    // GGML_TYPE_Q4_K: 256 elems, 144 bytes/block
        13 => Ok(DType::Q5K),    // GGML_TYPE_Q5_K: 256 elems, 176 bytes/block
        14 => Ok(DType::Q6K),    // GGML_TYPE_Q6_K: 256 elems, 210 bytes/block
        16 => Ok(DType::Iq2Xxs), // GGML_TYPE_IQ2_XXS: 256 elems, 66 bytes/block
        17 => Ok(DType::Iq2Xs),  // GGML_TYPE_IQ2_XS: 256 elems, 74 bytes/block
        18 => Ok(DType::Iq3Xxs), // GGML_TYPE_IQ3_XXS: 256 elems, 98 bytes/block
        19 => Ok(DType::Iq1S),   // GGML_TYPE_IQ1_S: 256 elems, 50 bytes/block
        20 => Ok(DType::Iq4Nl),  // GGML_TYPE_IQ4_NL: 32 elems, 18 bytes/block
        21 => Ok(DType::Iq3S),   // GGML_TYPE_IQ3_S: 256 elems, 110 bytes/block
        22 => Ok(DType::Iq2S),   // GGML_TYPE_IQ2_S: 256 elems, 82 bytes/block
        23 => Ok(DType::Iq4Xs),  // GGML_TYPE_IQ4_XS: 256 elems, 136 bytes/block
        29 => Ok(DType::Iq1M),   // GGML_TYPE_IQ1_M: 256 elems, 56 bytes/block
        30 => Ok(DType::Bf16),   // GGML_TYPE_BF16
        34 => Ok(DType::Tq1_0),  // GGML_TYPE_TQ1_0: 256 elems, 54 bytes/block
        35 => Ok(DType::Tq2_0),  // GGML_TYPE_TQ2_0: 256 elems, 66 bytes/block
        39 => Ok(DType::Mxfp4),  // GGML_TYPE_MXFP4: 32 elems, 17 bytes/block
        40 => Ok(DType::Nvfp4),  // GGML_TYPE_NVFP4: 64 elems, 36 bytes/block
        42 => Ok(DType::Q2_0),   // GGML_TYPE_Q2_0: 64 elems, 18 bytes/block (Bonsai ternary)
        36 => Ok(DType::I2S), // GGML_TYPE_I2_S (microsoft/BitNet): ternary 2bpw + per-tensor f32 scale
        _ => Err(Error::Unsupported(format!("ggml type {t}"))),
    }
}

/// Returns `(elements_per_block, bytes_per_block)` for the GGUF block layout.
///
/// Thin re-export of [`infr_core::decode_spec::block_layout`], the single source of truth for block
/// geometry (sizes from llama.cpp `ggml/src/ggml.c` `type_traits[]` `.blck_size` / `.type_size`;
/// each arm's defining formula is cross-checked there by `block_bytes_match_the_ggml_formulas`).
/// GGUF dim order: `ne[0]` is the fastest-varying axis (innermost / columns).
/// Public so placement code (dense layer streaming) can align streamed slot strides to whole
/// quant blocks — a slot base that isn't a whole number of blocks breaks the kernels'
/// element-offset weight addressing.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn block_layout(dtype: DType) -> (usize, usize) {
    infr_core::decode_spec::block_layout(dtype)
}

/// Bytes for `numel` elements of `dtype`, ROUNDED UP to a whole number of blocks.
///
/// `numel` is required to be a whole number of blocks — every real caller either owns a whole
/// tensor (validated at load, see [`Gguf::open`]'s `block_elems` check) or deliberately sizes a
/// block-aligned PREFIX. The rounding is what happens if that precondition is ever broken, and the
/// direction matters: plain `(numel / be) * bb` TRUNCATES, so a numel one element short of a block
/// boundary silently reports a SHORT byte count — for `Q4_K` with `numel = 100` it reports ZERO —
/// and every bounds check downstream (`resolve` against the file size, a `&b[..pb]` prefix slice)
/// then passes trivially while dequant/upload code works from a slice that doesn't cover the data
/// it is about to address. Rounding up can only ever produce a range that is too LARGE, which the
/// same bounds checks catch loudly instead. `debug_assert` states the contract so a new caller that
/// breaks it trips in tests rather than inheriting the rounding.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn checked_tensor_nbytes(dtype: DType, numel: usize) -> Option<usize> {
    // i2_s stores `numel/4` bytes of 2-bit ternary codes followed by ONE per-tensor f32 scale (see
    // `DType::I2S`). The scale is part of the tensor's data region (ggml rounds the next tensor's
    // offset up to `general.alignment`, so the on-disk stride is `numel/4 + 4` padded to 32B, but
    // the readable tensor bytes end after the scale). Size the slice to include it so
    // `dequant_codebook` can read the scale from the tail.
    if dtype == DType::I2S {
        debug_assert!(
            numel.is_multiple_of(4),
            "i2_s packs 4 ternary codes per byte; numel {numel} is not a multiple of 4"
        );
        return numel.div_ceil(4).checked_add(4);
    }
    let (be, bb) = block_layout(dtype);
    debug_assert!(
        numel.is_multiple_of(be),
        "{dtype:?} blocks hold {be} elements; numel {numel} is not a whole number of blocks"
    );
    numel.div_ceil(be).checked_mul(bb)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn tensor_nbytes(dtype: DType, numel: usize) -> usize {
    checked_tensor_nbytes(dtype, numel).expect("tensor byte count overflows usize")
}

/// Bytes occupied by `numel` elements of `dtype` in its GGUF block layout (`numel` must be a whole
/// number of blocks). Public helper so backends can size a block-aligned prefix (e.g. a quantized
/// KV cache: dequant only the first `kv_len` rows).
///
/// This is the second entry point into [`tensor_nbytes`] — the first being tensor sizing in
/// [`Gguf::open`], which REJECTS a misaligned `numel` outright. This one cannot: its callers pass a
/// prefix length they computed themselves, so there is no tensor name to name and no `Result` in
/// the signatures of the CPU kernels that call it (`kvquant::quantize_row`, the KV dequant prefix
/// in `infr-cpu`, the seam runner's per-row weight stride). Making it fallible would push a
/// `Result` into those hot loops for a condition their own callers already guarantee — the
/// block-quant KV path is gated on `kv_q8_layout_ok`, and a weight row is `n_embd` elements wide.
/// The contract is instead enforced by `tensor_nbytes`'s `debug_assert` plus its round-UP
/// behaviour, so a violation is a loud test failure or an out-of-range slice panic, never a
/// silently short buffer.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn nbytes(dtype: DType, numel: usize) -> usize {
    tensor_nbytes(dtype, numel)
}

// ─── Gguf::open ───────────────────────────────────────────────────────────────

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Gguf {
    /// Open and parse a GGUF file.
    ///
    /// The file is memory-mapped; no tensor bytes are copied into RAM.
    ///
    /// UNENFORCED INVARIANT: nothing else may write to or truncate the file while this `Mmap` is
    /// live. A file-backed mapping is not immutable memory — another process editing the file
    /// mutates the bytes under the `&[u8]` slices `tensor_bytes` hands out, which is a data race
    /// on memory Rust believes is frozen, and truncating it turns a resident page into `SIGBUS`.
    /// The `SAFETY` note below is a claim about the environment, not a guarantee this code makes.
    ///
    /// TODO: hold an exclusive advisory lock on the fd for the mapping's whole lifetime — i.e.
    /// keep the `File` in the struct, `flock(LOCK_EX | LOCK_NB)` (Windows: `LockFileEx` with
    /// `LOCKFILE_EXCLUSIVE_LOCK`) at open, and drop it with the mapping — so a concurrent writer
    /// is refused rather than trusted. Two things to settle before doing it: advisory locks bind
    /// only cooperating processes (`flock` does not stop a writer that never asks for the lock,
    /// so this narrows the window rather than closing it), and taking a lock EXCLUSIVE would stop
    /// two `infr` processes from sharing one model, which is a normal thing to want and the
    /// reason a shared lock is probably the right call for readers.
    ///
    /// Copying the file into an anonymous mapping DOES close the hole, and was tried — it is not
    /// affordable. Measured on a 16 GiB Qwen3.6-27B with a warm page cache: warm load 1.87 s →
    /// 10.5 s, and 14 GiB of evictable page cache became 20.2 GiB of anonymous RSS. Weights that
    /// the kernel could previously reclaim under pressure become swap-only, so a model larger
    /// than RAM stops being merely slow and starts being unrunnable.
    ///
    /// # Shard sets
    /// A model written by llama.cpp's `gguf-split` is N files, each a complete GGUF carrying part
    /// of the tensor index at offsets relative to its OWN data region, with only the first holding
    /// the model's metadata. Opening any one of them yields the WHOLE model: `split.count` names
    /// the set, [`infr_core::gguf_split::shard_set`] names its files, and each is mapped
    /// separately — every shard keeps the shared file mapping described above, which is the point
    /// (the alternative, one concatenated anonymous buffer, is the copy this doc rejects, at five
    /// times the size).
    ///
    /// Everything the set claims about itself is checked rather than trusted, because a
    /// downloaded model's metadata and filenames are remote input: the filename's `-of-MMMMM`
    /// total must equal `split.count`, every shard's zero-based `split.no` must equal the position
    /// its own filename claims, every shard must agree on `split.count`, and the assembled tensor
    /// index must be exactly `split.tensors.count` entries with no name appearing twice. A missing
    /// shard fails naming the file.
    pub fn open(path: &Path) -> Result<Self> {
        let first = Self::map_and_parse(path)?;
        match first.metadata.u64(KEY_SPLIT_COUNT) {
            // `gguf-split` writes `split.count` only when it actually splits, and a count of one
            // names this file and nothing else — so anything below two is the ordinary single-file
            // path, byte for byte what it was before shard sets existed.
            Some(count) if count > 1 => Self::open_split(first, count),
            _ => Self::assemble(vec![first], None),
        }
    }

    /// Follow the opened shard's `split.count` to the rest of the set, then assemble the model.
    ///
    /// `opened` is the shard the CALLER named, which need not be the first: a later shard carries
    /// no model metadata at all (no architecture, no hyperparameters — `gguf-split` copies the KV
    /// store into shard 1 and writes only the three `split.*` keys into the rest), so loading one
    /// on its own could never produce a runnable model. The set is therefore re-rooted at shard 1,
    /// which the filename convention names without needing anything else from the user, and
    /// `infr run <any shard>` loads the whole model. The shard already mapped is placed at its own
    /// index rather than mapped a second time.
    fn open_split(opened: ShardParts, count: u64) -> Result<Self> {
        let (names, dir, opened_idx) = {
            let fname = opened
                .path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| {
                    Error::Loader(format!(
                        "GGUF: {} declares {KEY_SPLIT_COUNT} = {count}, but its filename is not \
                         usable text, so the other shards cannot be named",
                        opened.path.display()
                    ))
                })?;
            // The filename is what BOUNDS the set: `parse_shard` refuses a field wider than
            // `gguf-split` writes, so a hostile `split.count` cannot make us enumerate (or open)
            // more files than a five-digit total allows.
            let Some(shard) = parse_shard(fname) else {
                return Err(Error::Loader(format!(
                    "GGUF: {fname} declares {KEY_SPLIT_COUNT} = {count}, but its name is not a \
                     `<base>-NNNNN-of-MMMMM.gguf` shard name, so the rest of the set cannot be \
                     found — rename the shards to the `gguf-split` convention"
                )));
            };
            if u64::from(shard.total) != count {
                return Err(Error::Loader(format!(
                    "GGUF: {fname} is named as one of {} shards but declares \
                     {KEY_SPLIT_COUNT} = {count} — the file's name and its metadata disagree \
                     about how many shards this model has",
                    shard.total
                )));
            }
            // The index is 1-based, so `0` is not a position at all (and would underflow the
            // conversion below), and a shard numbered past the total is not in the set it names.
            if shard.index < 1 || u64::from(shard.index) > count {
                return Err(Error::Loader(format!(
                    "GGUF: {fname} is numbered {} of {count} — a shard's index runs from 1 to the \
                     total, so this file is not a member of the set its own name describes",
                    shard.index
                )));
            }
            (
                shard_set(fname),
                opened.path.parent().unwrap_or(Path::new(".")).to_path_buf(),
                shard.index as usize - 1,
            )
        };
        // `shard.index` is 1-based and `split.no` is 0-based; this is the check that they agree for
        // the shard we were handed, and it is what makes `opened_idx` trustworthy as a position.
        Self::check_shard_position(&opened, opened_idx, count)?;

        let mut opened = Some(opened);
        let mut parts = Vec::with_capacity(names.len());
        for (i, name) in names.iter().enumerate() {
            let part = match opened.take_if(|_| i == opened_idx) {
                Some(p) => p,
                None => {
                    let path = dir.join(name);
                    let p = Self::map_and_parse(&path).map_err(|e| {
                        Error::Loader(format!(
                            "GGUF: shard {} of {count} ({}): {e}",
                            i + 1,
                            path.display()
                        ))
                    })?;
                    Self::check_shard_position(&p, i, count)?;
                    p
                }
            };
            parts.push(part);
        }

        // Written by `gguf-split` on every shard; read from the FIRST, which is the shard whose
        // metadata this model is loaded with. Its absence leaves nothing to check the assembled
        // index against, which for a format that always carries it means the file is not what it
        // says it is.
        let expect = parts[0]
            .metadata
            .u64(KEY_SPLIT_TENSORS_COUNT)
            .ok_or_else(|| {
                Error::Loader(format!(
                    "GGUF: {} declares {KEY_SPLIT_COUNT} = {count} but carries no \
                     {KEY_SPLIT_TENSORS_COUNT}, so the assembled tensor index cannot be checked \
                     for completeness",
                    parts[0].path.display()
                ))
            })?;
        Self::assemble(parts, Some(expect))
    }

    /// Check one shard's `split.no` / `split.count` against the position its FILENAME claims.
    ///
    /// Both numbers are how the set is stitched together, so a disagreement means these files are
    /// not the set they say they are — a renamed file, a mixed-up download, one shard from a
    /// different quantisation — and every tensor in the offending shard would be read from the
    /// wrong file. `index0` is the ZERO-based position, matching `split.no`.
    fn check_shard_position(parts: &ShardParts, index0: usize, count: u64) -> Result<()> {
        let path = parts.path.display();
        let Some(no) = parts.metadata.u64(KEY_SPLIT_NO) else {
            return Err(Error::Loader(format!(
                "GGUF: {path} is named as shard {} of {count} but carries no {KEY_SPLIT_NO} — it \
                 is not part of a `gguf-split` set",
                index0 + 1
            )));
        };
        if no != index0 as u64 {
            return Err(Error::Loader(format!(
                "GGUF: {path} is named as shard {} of {count} but declares {KEY_SPLIT_NO} = {no} \
                 (shard {}) — its name and its metadata disagree about which shard it is",
                index0 + 1,
                no + 1
            )));
        }
        let Some(declared) = parts.metadata.u64(KEY_SPLIT_COUNT) else {
            return Err(Error::Loader(format!(
                "GGUF: {path} is named as shard {} of {count} but carries no {KEY_SPLIT_COUNT}",
                index0 + 1
            )));
        };
        if declared != count {
            return Err(Error::Loader(format!(
                "GGUF: {path} declares {KEY_SPLIT_COUNT} = {declared} but the rest of the set \
                 declares {count} — these files are not one model"
            )));
        }
        Ok(())
    }

    /// Place mapped shards end to end into one model: shard 1's metadata, every shard's tensors in
    /// order, and each tensor's offset rewritten into the set's concatenated address space.
    ///
    /// `expect_tensors` is `split.tensors.count` for a shard set, `None` for a single file (which
    /// has nothing to compare against — its header's own `tensor_count` already drove the parse).
    fn assemble(parts: Vec<ShardParts>, expect_tensors: Option<u64>) -> Result<Self> {
        let mut shards: Vec<ShardFile> = Vec::with_capacity(parts.len());
        let mut tensors: Vec<TensorInfo> = Vec::new();
        let mut metadata: Option<Metadata> = None;
        let mut file_base: u64 = 0;

        for part in parts {
            let ShardParts {
                mmap,
                metadata: md,
                tensors: mut ts,
                data_region_start,
                path,
            } = part;
            // Rewrite each tensor's shard-local data-region offset into the SET's concatenated
            // address space ONCE, here. That single number then locates the tensor completely —
            // `resolve` finds the owning shard from it — so there is no second per-tensor table
            // that could drift out of step with the tensor list it describes.
            let this_base = file_base;
            let data_base = this_base
                .checked_add(data_region_start as u64)
                .ok_or_else(|| Error::Loader("GGUF: shard set is larger than u64 bytes".into()))?;
            for t in ts.iter_mut() {
                t.offset = data_base.checked_add(t.offset).ok_or_else(|| {
                    Error::Loader(format!("tensor '{}' offset overflows u64", t.name))
                })?;
            }
            file_base = this_base
                .checked_add(mmap.len() as u64)
                .ok_or_else(|| Error::Loader("GGUF: shard set is larger than u64 bytes".into()))?;
            tensors.append(&mut ts);
            // Only the first shard's metadata is kept: the rest carry nothing but `split.*`.
            metadata.get_or_insert(md);
            shards.push(ShardFile {
                mmap: Arc::new(mmap),
                path,
                file_base: this_base,
            });
        }

        // Duplicate tensor names are malformed input and must not be silently accepted.
        // `resolve` looks a tensor up with a linear `find`, so with two entries sharing a
        // name the FIRST always wins and the second is unreachable — different offsets,
        // different shapes, possibly a different dtype, and nothing anywhere reports that
        // half the file was ignored. Rejecting at load is the only place the ambiguity is
        // visible; afterwards every consumer just gets a plausible-looking answer. One
        // `HashSet` pass over the (few thousand, at most) names costs nothing measurable.
        // Across a shard set it also catches the same tensor arriving from two shards.
        {
            let mut seen: std::collections::HashSet<&str> =
                std::collections::HashSet::with_capacity(tensors.len());
            for t in &tensors {
                if !seen.insert(t.name.as_str()) {
                    return Err(Error::Loader(format!(
                        "GGUF: duplicate tensor name '{}' — the file declares it more than \
                         once, so which entry a lookup resolves to is ambiguous",
                        t.name
                    )));
                }
            }
        }

        if let Some(expect) = expect_tensors {
            if tensors.len() as u64 != expect {
                return Err(Error::Loader(format!(
                    "GGUF: the {} shards hold {} tensors between them but the model declares \
                     {KEY_SPLIT_TENSORS_COUNT} = {expect} — the set is incomplete or mismatched",
                    shards.len(),
                    tensors.len()
                )));
            }
        }

        Ok(Gguf {
            shards,
            metadata: metadata.expect("a model is assembled from at least one shard"),
            tensors,
        })
    }

    /// Map and parse ONE GGUF file: its metadata, its own tensor index (offsets still relative to
    /// its own data region), and where that data region starts. The unit [`Self::assemble`] places
    /// into a model, whether the model is one file or a shard set.
    fn map_and_parse(path: &Path) -> Result<ShardParts> {
        let file = File::open(path)?;
        // SAFETY: the file is not modified while this Mmap is live — see the invariant above.
        let mmap = unsafe { Mmap::map(&file) }?;

        // Best-effort madvise hints. All are advisory: `advise()` returns an
        // `io::Result`, but a rejected/unsupported hint must never fail the load —
        // weight correctness does not depend on any hint landing, so we swallow errors.
        // NOTE: we deliberately do NOT use `Advice::Sequential`. Decode re-reads the
        // ENTIRE model every token, so `MADV_SEQUENTIAL`'s drop-behind eviction would
        // throw away pages we need on the very next token — wrong for this access
        // pattern. Only `WillNeed`/`HugePage` fit.
        #[cfg(unix)]
        {
            // Populate/readahead the whole mapping into the page cache now, front-loading
            // the weight read at load instead of faulting it in lazily on the first token.
            let _ = mmap.advise(memmap2::Advice::WillNeed);
            // Linux only: request 2 MB transparent huge pages to cut dTLB page-walks over
            // the multi-GB sequential weight stream. On file-backed mmaps this is frequently
            // a no-op (THP-for-filesystem is not always enabled) — hence best-effort.
            #[cfg(target_os = "linux")]
            let _ = mmap.advise(memmap2::Advice::HugePage);
        }

        // All parsing happens in this block so the borrow of `mmap` ends
        // before we move `mmap` into the returned struct.
        let (metadata, tensors, data_region_start) = {
            let buf: &[u8] = &mmap;
            let mut r = Reader::new(buf);

            // ── header ────────────────────────────────────────────────────────
            let magic = r.read_u32()?;
            if magic != GGUF_MAGIC {
                return Err(Error::Loader(format!(
                    "not a GGUF file (magic 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X})"
                )));
            }

            let version = r.read_u32()?;
            if version != 2 && version != 3 {
                return Err(Error::Loader(format!(
                    "unsupported GGUF version {version} (supported: 2, 3)"
                )));
            }

            let tensor_count = r.read_u64()? as usize;
            let kv_count = r.read_u64()? as usize;

            // ── metadata KV pairs ─────────────────────────────────────────────
            // Clamp the reservation to the bytes actually left in the file so a bogus
            // `kv_count` can't force a huge allocation before the parse fails on EOF.
            let mut kv: HashMap<String, MetaValue> =
                HashMap::with_capacity(kv_count.min(r.remaining()));
            for _ in 0..kv_count {
                let key = r.read_gguf_str()?;
                let vtype = r.read_u32()?;
                let val = r.read_meta_value(vtype)?;
                kv.insert(key, val);
            }
            let metadata = Metadata { kv };

            // alignment from metadata, defaulting to GGUF_DEFAULT_ALIGNMENT (32).
            // Reject non-positive / non-power-of-two: `general.alignment == 0` would make
            // `pos.div_ceil(alignment)` panic with a div-by-zero, and GGUF requires a
            // power-of-two alignment.
            let alignment = metadata
                .u64("general.alignment")
                .unwrap_or(DEFAULT_ALIGNMENT as u64) as usize;
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(Error::Loader(format!(
                    "GGUF: invalid general.alignment {alignment} (must be a positive power of two)"
                )));
            }
            // …and bound it from ABOVE too. A power-of-two check alone still admits values like
            // `1 << 62`, and `pos.div_ceil(alignment) * alignment` then overflows the multiply —
            // a panic in debug, a silently wrapped (tiny or zero) `data_region_start` in release,
            // which points every tensor at the wrong bytes. Real GGUF files use 32; the cap is set
            // several orders of magnitude above that so no plausible producer is rejected, while
            // still leaving `div_ceil * alignment` unable to come near `usize::MAX`.
            if alignment > MAX_ALIGNMENT {
                return Err(Error::Loader(format!(
                    "GGUF: invalid general.alignment {alignment} (exceeds the {MAX_ALIGNMENT}-byte cap)"
                )));
            }

            // ── tensor info entries ───────────────────────────────────────────
            // Collect raw fields first (name, shape, ggml_type, offset); then
            // convert after parsing so errors are reported before we consume the
            // remaining header bytes.
            let mut raw: Vec<(String, Vec<usize>, u32, u64)> =
                Vec::with_capacity(tensor_count.min(r.remaining()));
            for _ in 0..tensor_count {
                let name = r.read_gguf_str()?;
                let n_dims = r.read_u32()? as usize;
                if n_dims > MAX_TENSOR_DIMS {
                    return Err(Error::Loader(format!(
                        "GGUF: tensor '{name}' declares {n_dims} dimensions (max {MAX_TENSOR_DIMS})"
                    )));
                }
                let mut shape = Vec::with_capacity(n_dims.min(r.remaining()));
                for _ in 0..n_dims {
                    shape.push(r.read_u64()? as usize);
                }
                let ggml_type = r.read_u32()?;
                let offset = r.read_u64()?;
                raw.push((name, shape, ggml_type, offset));
            }

            // ── data region: align cursor position to `alignment` ─────────────
            let data_region_start = r.pos.div_ceil(alignment) * alignment;

            // ── convert raw tensor entries to TensorInfo ─────────────────────
            let mut tensors: Vec<TensorInfo> = Vec::with_capacity(raw.len());
            for (name, shape, ggml_type, offset) in raw {
                let dtype = ggml_type_to_dtype(ggml_type)?;
                // `checked_mul`: a crafted shape must error, not overflow the product.
                let numel: usize = shape
                    .iter()
                    .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                    .ok_or_else(|| {
                        Error::Loader(format!(
                            "GGUF: tensor '{name}' shape {shape:?} overflows usize"
                        ))
                    })?;
                // A quantised tensor's element count MUST be a whole number of blocks — llama.cpp
                // checks `ne[0] % blck_size == 0` at load for the same reason. Without this a
                // crafted or corrupt file (say Q4_K with numel = 100, i.e. 0 whole 256-element
                // blocks) sizes the tensor at FEWER bytes than the data it describes; `resolve`'s
                // file-size bounds check then passes trivially on that short-or-empty range and
                // dequant/upload code proceeds to address elements the slice does not cover. This
                // is the load-time place to catch it: the tensor's name and shape are still in
                // hand, so the error can say WHICH tensor is malformed instead of surfacing later
                // as an unexplained out-of-range panic (or worse, plausible-looking garbage) deep
                // in a backend.
                if dtype == DType::I2S {
                    // i2_s packs 4 ternary codes per byte and carries a single trailing f32 scale;
                    // a numel that isn't a multiple of 4 has no whole-byte code layout at all
                    // (numel = 3 would size to 4 bytes: all scale, no codes).
                    if !numel.is_multiple_of(4) {
                        return Err(Error::Loader(format!(
                            "GGUF: tensor '{name}' shape {shape:?} has {numel} elements, not a \
                             multiple of 4 (i2_s packs 4 2-bit codes per byte)"
                        )));
                    }
                } else {
                    let (block_elems, _) = block_layout(dtype);
                    if !numel.is_multiple_of(block_elems) {
                        return Err(Error::Loader(format!(
                            "GGUF: tensor '{name}' shape {shape:?} has {numel} elements, not a \
                             whole number of {dtype:?} blocks ({block_elems} elements per block)"
                        )));
                    }
                }
                let nbytes = checked_tensor_nbytes(dtype, numel).ok_or_else(|| {
                    Error::Loader(format!(
                        "GGUF: tensor '{name}' shape {shape:?} byte size overflows usize"
                    ))
                })?;
                tensors.push(TensorInfo {
                    name,
                    shape,
                    dtype,
                    offset,
                    nbytes,
                });
            }

            (metadata, tensors, data_region_start)
        }; // ← borrow of `mmap` ends here

        Ok(ShardParts {
            mmap,
            metadata,
            tensors,
            data_region_start,
            path: path.to_path_buf(),
        })
    }

    /// Zero-copy, ref-counted view of a tensor's raw bytes (keeps the `Mmap` alive via `Arc`). Unlike
    /// [`WeightSource::tensor_bytes`] the result is not borrow-bound to `&self`, so a backend can hold
    /// it as a weight buffer and read straight from the mapping — no `memcpy` into owned RAM.
    pub fn tensor_bytes_arc(&self, name: &str) -> Result<TensorBytes> {
        let (shard, off, len) = self.resolve(name)?;
        Ok(TensorBytes {
            mmap: Arc::clone(&shard.mmap),
            off,
            len,
            file_base: shard.file_base,
        })
    }

    /// One tensor's byte range in the model's concatenated file address space — the same
    /// `(absolute_offset, len)` the mapping uses, for a reader that opens the files itself instead
    /// of going through these mappings.
    ///
    /// That is the weight pager's disk tier (`infr_core::blockio`): it reads at explicit offsets so
    /// residency is its own decision rather than the page cache's. Bounds-checked identically to
    /// every mapped read, so a descriptor built from this can never name bytes outside the model.
    ///
    /// For a single file the offset IS a file offset. For a shard set the shards are addressed end
    /// to end in `split.no` order, and `infr_core::blockio::FileBlockIo::open_shards` — given the
    /// same files in the same order by [`Self::shards`] — maps an offset back to the shard that
    /// holds it. A tensor never straddles that boundary: each lives wholly inside one shard.
    pub fn tensor_file_range(&self, name: &str) -> Result<(u64, usize)> {
        let (shard, off, len) = self.resolve(name)?;
        Ok((shard.file_base + off as u64, len))
    }

    /// The files this model was opened from, in shard order, each with the byte length its offsets
    /// were computed against — what a second reader (the pager's disk tier) opens.
    ///
    /// Paths, not descriptors, and that is a real difference: re-opening one can land on a
    /// DIFFERENT inode if the model was replaced meanwhile. `infr_core::blockio::FileBlockIo`
    /// stamps whatever it opens and re-checks that, so a mid-run replacement is caught rather than
    /// silently read (the same guarantee, and the same blind spot, as `watch::WeightWatch`). The
    /// lengths are what makes that check possible ACROSS a shard set: a shard whose size changed
    /// would shift every offset in every later shard, so `open_shards` refuses a file whose length
    /// is no longer the one these offsets were built on.
    pub fn shards(&self) -> Vec<(&Path, u64)> {
        self.shards
            .iter()
            .map(|s| (s.path.as_path(), s.mmap.len() as u64))
            .collect()
    }

    /// Look up a tensor by name and return the shard holding it plus its `(offset, len)` within
    /// THAT shard's mapping, bounds-checked against the shard's size with `checked_add` so a
    /// crafted offset/length can't overflow. Shared by [`Self::tensor_bytes_arc`] and
    /// [`WeightSource::tensor_bytes`] so the lookup + overflow-safe bounds check lives
    /// in exactly one place.
    fn resolve(&self, name: &str) -> Result<(&ShardFile, usize, usize)> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::Loader(format!("tensor not found: '{name}'")))?;
        // `shards` is ordered by `file_base` and each covers its whole file, so the shard holding
        // an offset is the last one starting at or before it. A single-file model has one shard at
        // base 0, so this is always index 0 and the arithmetic below is the identity it used to be.
        let idx = self
            .shards
            .partition_point(|s| s.file_base <= info.offset)
            .saturating_sub(1);
        let shard = &self.shards[idx];
        let off = usize::try_from(info.offset - shard.file_base)
            .map_err(|_| Error::Loader(format!("tensor '{name}' offset overflows usize")))?;
        let end = off
            .checked_add(info.nbytes)
            .ok_or_else(|| Error::Loader(format!("tensor '{name}' byte range overflows usize")))?;
        if end > shard.mmap.len() {
            return Err(Error::Loader(format!(
                "tensor '{name}' byte range {off}..{end} exceeds the {} bytes of {}",
                shard.mmap.len(),
                shard.path.display()
            )));
        }
        Ok((shard, off, info.nbytes))
    }
}

/// One shard parsed on its own, before the set is assembled: its position in the concatenated
/// address space is not known until every earlier shard has been mapped, so its tensor offsets are
/// still relative to its OWN data region.
struct ShardParts {
    mmap: Mmap,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    /// Absolute byte offset into `mmap` where this shard's tensor data begins.
    data_region_start: usize,
    path: PathBuf,
}

// ─── WeightSource impl ────────────────────────────────────────────────────────

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl WeightSource for Gguf {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Returns a slice into the mmap'd data region of the shard holding the named tensor.
    ///
    /// The slice lifetime is tied to `&self` (i.e. the `Gguf` struct keeps
    /// every shard's `Mmap` alive).
    fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let (shard, start, len) = self.resolve(name)?;
        Ok(&shard.mmap[start..start + len])
    }

    fn chat_template(&self) -> Option<&str> {
        self.metadata.str("tokenizer.chat_template")
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ── fixture builder ───────────────────────────────────────────────────────

    fn u32_le(v: u32) -> [u8; 4] {
        v.to_le_bytes()
    }
    fn u64_le(v: u64) -> [u8; 8] {
        v.to_le_bytes()
    }

    fn push_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&u32_le(v));
    }
    fn push_u64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&u64_le(v));
    }
    fn push_gguf_str(buf: &mut Vec<u8>, s: &str) {
        push_u64(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }

    /// Build a minimal valid GGUF v3 file in memory: 2 metadata KVs
    /// (`test.block_count` = UINT32 30, `general.architecture` = STRING "diffusion-gemma")
    /// and 1 tensor (`tensor0`, F32, shape [4], 16 bytes of zeros).
    fn build_fixture() -> Vec<u8> {
        let mut b: Vec<u8> = Vec::new();

        // header
        push_u32(&mut b, GGUF_MAGIC); // magic
        push_u32(&mut b, 3); // version
        push_u64(&mut b, 1); // tensor_count
        push_u64(&mut b, 2); // kv_count

        // KV 1: "test.block_count" = UINT32 30
        push_gguf_str(&mut b, "test.block_count");
        push_u32(&mut b, 4); // GGUF_TYPE_UINT32
        push_u32(&mut b, 30);

        // KV 2: "general.architecture" = STRING "diffusion-gemma"
        push_gguf_str(&mut b, "general.architecture");
        push_u32(&mut b, 8); // GGUF_TYPE_STRING
        push_gguf_str(&mut b, "diffusion-gemma");

        // tensor info: "tensor0" F32 [4] offset=0
        push_gguf_str(&mut b, "tensor0");
        push_u32(&mut b, 1); // n_dims
        push_u64(&mut b, 4); // dim[0] = 4 elements
        push_u32(&mut b, 0); // ggml_type = F32
        push_u64(&mut b, 0); // offset = 0

        // pad to 32-byte alignment
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }

        // tensor data: 4 × f32 = 16 bytes
        b.extend_from_slice(&[0u8; 16]);

        b
    }

    // ── helper: write fixture to a named temp file ────────────────────────────

    fn write_temp_gguf(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(bytes).expect("write fixture");
        f.flush().expect("flush");
        f
    }

    // ── offline fixture test ──────────────────────────────────────────────────

    #[test]
    fn fixture_gguf_round_trip() {
        let bytes = build_fixture();
        let tmp = write_temp_gguf(&bytes);
        let gguf = Gguf::open(tmp.path()).expect("open fixture");

        // metadata
        assert_eq!(
            gguf.metadata().u64("test.block_count"),
            Some(30),
            "test.block_count should be 30"
        );
        assert_eq!(
            gguf.metadata().str("general.architecture"),
            Some("diffusion-gemma"),
            "general.architecture should be 'diffusion-gemma'"
        );

        // tensor directory
        let tensors = gguf.tensors();
        assert_eq!(tensors.len(), 1, "should have 1 tensor");
        let t = &tensors[0];
        assert_eq!(t.name, "tensor0");
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.shape, vec![4]);
        assert_eq!(t.nbytes, 16);

        // data region
        let data = gguf.tensor_bytes("tensor0").expect("tensor_bytes");
        assert_eq!(data.len(), 16, "tensor bytes should be 16");

        // chat_template absent → None
        assert!(gguf.chat_template().is_none());
    }

    // ── madvise hints are transparent ─────────────────────────────────────────

    /// The best-effort `madvise` hints applied to the weight mmap in `Gguf::open`
    /// (`WillNeed`, and `HugePage` on Linux) must not change what the mapping reads
    /// back. Open the fixture and assert the tensor bytes are byte-for-byte identical
    /// to the raw tensor-data region of the source buffer — i.e. the hint is a no-op
    /// on observable content. This is the only path that exercises the advise() calls.
    ///
    /// Deliberately NOT asserted here: that the mapping keeps reading the ORIGINAL bytes
    /// after the file is rewritten. It does not, and cannot — see the unenforced invariant
    /// on [`Gguf::open`].
    #[test]
    fn madvise_hints_are_transparent() {
        let bytes = build_fixture();
        let tmp = write_temp_gguf(&bytes);
        let gguf = Gguf::open(tmp.path()).expect("open fixture");

        // tensor0 is F32 [4] at data offset 0 → the last 16 bytes of the buffer.
        let expected = &bytes[bytes.len() - 16..];
        let data = gguf.tensor_bytes("tensor0").expect("tensor_bytes");
        assert_eq!(
            data, expected,
            "mmap read must be byte-for-byte with the source"
        );
    }

    /// The mapping starts at file byte 0, which is what lets a mapped view's offset double as a
    /// FILE offset ([`TensorBytes::file_range`]). `Mmap::map` maps the whole file, so this holds by
    /// construction — but it is the assumption every descriptor the disk tier builds rests on, and
    /// a future partial mapping would break it silently, shifting every streamed weight.
    #[test]
    fn mapping_covers_the_whole_file() {
        let bytes = build_fixture();
        let tmp = write_temp_gguf(&bytes);
        let gguf = Gguf::open(tmp.path()).expect("open fixture");
        let view = gguf.tensor_bytes_arc("tensor0").expect("view");
        let (off, len) = view.file_range();
        assert_eq!(
            (off, len),
            gguf.tensor_file_range("tensor0").expect("range"),
            "a view's file range must agree with the by-name range"
        );
        // And it really is a file offset: the same bytes are there when read through the file.
        assert_eq!(&bytes[off as usize..off as usize + len], &view[..]);
    }

    /// The pager's disk tier reads a tensor by `(offset, len)` taken from
    /// [`Gguf::tensor_file_range`] and opens the files [`Gguf::shards`] names. Both halves must
    /// land on exactly the bytes the MAPPING serves — an off-by-one against `data_region_start`
    /// here is weights shifted by a few bytes, which decodes to plausible garbage rather than
    /// failing.
    #[test]
    fn the_file_range_reads_what_the_mapping_reads() {
        use std::io::{Read, Seek};
        let bytes = build_fixture();
        let tmp = write_temp_gguf(&bytes);
        let gguf = Gguf::open(tmp.path()).expect("open fixture");

        // A single-file model is one shard, at base 0, of its own full length.
        assert_eq!(gguf.shards().len(), 1);
        assert_eq!(gguf.shards()[0], (tmp.path(), bytes.len() as u64));

        let mapped = gguf.tensor_bytes("tensor0").expect("tensor_bytes").to_vec();
        let (off, len) = gguf
            .tensor_file_range("tensor0")
            .expect("tensor_file_range");
        assert_eq!(len, mapped.len());

        let mut f = std::fs::File::open(gguf.shards()[0].0).expect("re-open by path");
        let mut from_file = vec![0u8; len];
        f.seek(std::io::SeekFrom::Start(off)).expect("seek");
        f.read_exact(&mut from_file).expect("read");
        assert_eq!(
            from_file, mapped,
            "a file read at the reported range must match the mapping"
        );

        // A name the file does not carry is an error, not a range into nothing.
        assert!(gguf.tensor_file_range("nope").is_err());
    }

    // ── malformed / truncated header hardening ────────────────────────────────

    /// A bogus near-`usize::MAX` string length prefix must surface as `Error::Loader`,
    /// never overflow `pos + n` (debug panic / release wrap → slice panic).
    #[test]
    fn oversized_string_length_errors() {
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3); // version
        push_u64(&mut b, 0); // tensor_count
        push_u64(&mut b, 1); // kv_count
        push_u64(&mut b, u64::MAX); // key string length = absurd
        let tmp = write_temp_gguf(&b);
        let err = Gguf::open(tmp.path()).map(|_| ());
        assert!(
            matches!(err, Err(Error::Loader(_))),
            "oversized string length should be Error::Loader, got {err:?}"
        );
    }

    /// `general.alignment == 0` must be rejected, not cause a div-by-zero panic in
    /// `pos.div_ceil(alignment)`.
    #[test]
    fn zero_alignment_errors() {
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3); // version
        push_u64(&mut b, 0); // tensor_count
        push_u64(&mut b, 1); // kv_count
        push_gguf_str(&mut b, "general.alignment");
        push_u32(&mut b, 4); // GGUF_TYPE_UINT32
        push_u32(&mut b, 0); // alignment = 0
        let tmp = write_temp_gguf(&b);
        let err = Gguf::open(tmp.path()).map(|_| ());
        assert!(
            matches!(err, Err(Error::Loader(_))),
            "zero alignment should be Error::Loader, got {err:?}"
        );
    }

    /// A non-power-of-two alignment must be rejected.
    #[test]
    fn non_pow2_alignment_errors() {
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3);
        push_u64(&mut b, 0);
        push_u64(&mut b, 1);
        push_gguf_str(&mut b, "general.alignment");
        push_u32(&mut b, 4);
        push_u32(&mut b, 33); // not a power of two
        let tmp = write_temp_gguf(&b);
        let err = Gguf::open(tmp.path()).map(|_| ());
        assert!(
            matches!(err, Err(Error::Loader(_))),
            "non-pow2 alignment should be Error::Loader, got {err:?}"
        );
    }

    /// An absurdly LARGE alignment is a power of two and so passes the lower-bound checks,
    /// but `pos.div_ceil(alignment) * alignment` then overflows: a panic in debug, a wrapped
    /// `data_region_start` (every tensor read from the wrong offset) in release. It must be
    /// rejected with an `Error::Loader` naming the value and the cap.
    #[test]
    fn oversized_alignment_errors() {
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3);
        push_u64(&mut b, 0);
        push_u64(&mut b, 1);
        push_gguf_str(&mut b, "general.alignment");
        push_u32(&mut b, 10); // GGUF_TYPE_UINT64
        push_u64(&mut b, 1u64 << 62); // power of two, but nonsense
        let tmp = write_temp_gguf(&b);
        let err = Gguf::open(tmp.path()).map(|_| ());
        let Err(Error::Loader(msg)) = err else {
            panic!("oversized alignment should be Error::Loader, got {err:?}");
        };
        assert!(msg.contains(&(1u64 << 62).to_string()), "{msg}");
        assert!(msg.contains(&MAX_ALIGNMENT.to_string()), "{msg}");
    }

    /// `n_dims` is an unbounded `u32` on disk but ggml caps a tensor at `GGML_MAX_DIMS` = 4.
    /// Without the check the shape loop just runs until EOF, so the reported error is
    /// "unexpected end of file" rather than the actual defect.
    #[test]
    fn oversized_n_dims_errors() {
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3);
        push_u64(&mut b, 1); // tensor_count = 1
        push_u64(&mut b, 0); // kv_count = 0
        push_gguf_str(&mut b, "tensor0");
        push_u32(&mut b, 5); // n_dims = 5 > GGML_MAX_DIMS
        for _ in 0..5 {
            push_u64(&mut b, 2);
        }
        push_u32(&mut b, 0); // F32
        push_u64(&mut b, 0); // offset
        let tmp = write_temp_gguf(&b);
        let err = Gguf::open(tmp.path()).map(|_| ());
        let Err(Error::Loader(msg)) = err else {
            panic!("n_dims > 4 should be Error::Loader, got {err:?}");
        };
        assert!(msg.contains("tensor0") && msg.contains('5'), "{msg}");
    }

    /// The boundary of the rule above: `GGML_MAX_DIMS` = 4 dimensions is LEGAL and must still
    /// load. Pins the cap as `> 4`, not `>= 4`.
    #[test]
    fn four_dims_is_accepted() {
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3);
        push_u64(&mut b, 1); // tensor_count
        push_u64(&mut b, 0); // kv_count
        push_gguf_str(&mut b, "tensor0");
        push_u32(&mut b, 4); // n_dims = GGML_MAX_DIMS
        for d in [4u64, 1, 1, 1] {
            push_u64(&mut b, d);
        }
        push_u32(&mut b, 0); // F32
        push_u64(&mut b, 0); // offset
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        b.extend_from_slice(&[0u8; 16]);
        let tmp = write_temp_gguf(&b);
        let gguf = Gguf::open(tmp.path()).expect("4-dim tensor must load");
        assert_eq!(gguf.tensors()[0].shape, vec![4, 1, 1, 1]);
    }

    /// Two tensors sharing a name make `resolve`'s linear `find` silently serve the first and
    /// leave the second unreachable. That is malformed input being accepted, so `open` rejects
    /// it and names the duplicate.
    #[test]
    fn duplicate_tensor_name_errors() {
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3);
        push_u64(&mut b, 2); // tensor_count = 2
        push_u64(&mut b, 0); // kv_count = 0
        for offset in [0u64, 16] {
            push_gguf_str(&mut b, "tensor0"); // same name twice
            push_u32(&mut b, 1); // n_dims
            push_u64(&mut b, 4); // dim[0]
            push_u32(&mut b, 0); // F32
            push_u64(&mut b, offset);
        }
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        b.extend_from_slice(&[0u8; 32]);
        let tmp = write_temp_gguf(&b);
        let err = Gguf::open(tmp.path()).map(|_| ());
        let Err(Error::Loader(msg)) = err else {
            panic!("duplicate tensor name should be Error::Loader, got {err:?}");
        };
        assert!(msg.contains("tensor0"), "{msg}");
        assert!(msg.contains("duplicate"), "{msg}");
    }

    /// A huge `tensor_count` on a tiny file must error gracefully — no multi-GB
    /// `with_capacity` reservation / capacity-overflow abort.
    #[test]
    fn huge_tensor_count_errors_gracefully() {
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3); // version
        push_u64(&mut b, u64::MAX); // tensor_count = absurd
        push_u64(&mut b, 0); // kv_count
        let tmp = write_temp_gguf(&b);
        let err = Gguf::open(tmp.path()).map(|_| ());
        assert!(
            matches!(err, Err(Error::Loader(_))),
            "huge tensor_count should be Error::Loader, got {err:?}"
        );
    }

    /// A huge `kv_count` on a tiny file must error gracefully.
    #[test]
    fn huge_kv_count_errors_gracefully() {
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3); // version
        push_u64(&mut b, 0); // tensor_count
        push_u64(&mut b, u64::MAX); // kv_count = absurd
        let tmp = write_temp_gguf(&b);
        let err = Gguf::open(tmp.path()).map(|_| ());
        assert!(
            matches!(err, Err(Error::Loader(_))),
            "huge kv_count should be Error::Loader, got {err:?}"
        );
    }

    /// A metadata ARRAY whose element type is itself ARRAY recurses `read_meta_value`. Each
    /// level costs the file only 12 bytes (u32 elem_type + u64 count), so a small crafted
    /// file nests tens of thousands deep and overflows the stack — a SIGSEGV/abort the
    /// caller cannot catch, and reachable from "open this downloaded model". The depth
    /// ceiling must turn that into a plain `Error::Loader`.
    #[test]
    fn deeply_nested_metadata_array_errors() {
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3); // version
        push_u64(&mut b, 0); // tensor_count
        push_u64(&mut b, 1); // kv_count
        push_gguf_str(&mut b, "bomb");
        push_u32(&mut b, 9); // value type = ARRAY
                             // Each pair opens one more ARRAY level holding exactly one element, which is itself
                             // an ARRAY: 8192 levels in ~96 KB of file.
        for _ in 0..8192 {
            push_u32(&mut b, 9); // elem_type = ARRAY
            push_u64(&mut b, 1); // count = 1
        }
        // Innermost terminator (never reached — the guard fires long before).
        push_u32(&mut b, 0); // elem_type = UINT8
        push_u64(&mut b, 0); // count = 0

        let tmp = write_temp_gguf(&b);
        let err = Gguf::open(tmp.path()).map(|_| ());
        match err {
            Err(Error::Loader(msg)) => assert!(
                msg.contains("nesting"),
                "should fail on the depth guard, not incidentally on EOF: {msg}"
            ),
            other => panic!("deeply nested array should be Error::Loader, got {other:?}"),
        }
    }

    /// The depth guard must not change well-formed files: real GGUF metadata nests arrays
    /// (e.g. an array of arrays of ints), so a 2-level value still parses to the same
    /// `MetaValue::Arr` tree it did before the ceiling existed.
    #[test]
    fn shallow_nested_metadata_array_still_parses() {
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3); // version
        push_u64(&mut b, 0); // tensor_count
        push_u64(&mut b, 1); // kv_count
        push_gguf_str(&mut b, "nested");
        push_u32(&mut b, 9); // value type = ARRAY
        push_u32(&mut b, 9); // elem_type = ARRAY
        push_u64(&mut b, 2); // 2 inner arrays
        for _ in 0..2 {
            push_u32(&mut b, 4); // elem_type = UINT32
            push_u64(&mut b, 1); // 1 element
            push_u32(&mut b, 7);
        }
        let tmp = write_temp_gguf(&b);
        let gguf = Gguf::open(tmp.path()).expect("2-level nesting must still parse");
        let Some(MetaValue::Arr(outer)) = gguf.metadata().kv.get("nested") else {
            panic!("expected an array value");
        };
        assert_eq!(outer.len(), 2, "outer array should hold 2 inner arrays");
        assert!(
            matches!(&outer[0], MetaValue::Arr(inner) if inner.len() == 1),
            "inner element should be a 1-element array"
        );
    }

    // ── block-alignment hardening ─────────────────────────────────────────────

    #[test]
    fn tensor_byte_count_overflow_errors() {
        let b = build_typed_fixture(0, 1u64 << 62, 0);
        let tmp = write_temp_gguf(&b);
        let err = Gguf::open(tmp.path()).map(|_| ());
        assert!(
            matches!(err, Err(Error::Loader(_))),
            "overflowing F32 byte count should be Error::Loader, got {err:?}"
        );
    }

    /// A GGUF holding exactly one tensor of `ggml_type` with the given 1-D length, and `data_len`
    /// bytes of zeros in the data region. The round-trip fixture uses F32 (`block_elems == 1`), so
    /// it structurally cannot express a misaligned element count — this builder takes the ggml type
    /// so a quantised one can.
    fn build_typed_fixture(ggml_type: u32, dim0: u64, data_len: usize) -> Vec<u8> {
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3); // version
        push_u64(&mut b, 1); // tensor_count
        push_u64(&mut b, 0); // kv_count
        push_gguf_str(&mut b, "tensor0");
        push_u32(&mut b, 1); // n_dims
        push_u64(&mut b, dim0);
        push_u32(&mut b, ggml_type);
        push_u64(&mut b, 0); // offset = 0
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        b.resize(b.len() + data_len, 0);
        b
    }

    /// A quantised tensor whose element count is not a whole number of blocks must be rejected at
    /// LOAD. `(numel / block_elems) * block_bytes` truncates, so Q4_K with numel = 100 sizes to
    /// ZERO bytes; `resolve`'s file-size bounds check then passes vacuously and the backend
    /// dequants from an empty slice. llama.cpp validates `ne[0] % blck_size == 0` at load for
    /// exactly this reason.
    #[test]
    fn misaligned_quant_numel_errors() {
        // ggml type 12 = Q4_K: 256 elements per 144-byte block. 100 is not a whole block.
        let b = build_typed_fixture(12, 100, 144);
        let tmp = write_temp_gguf(&b);
        let err = Gguf::open(tmp.path()).map(|_| ());
        match err {
            Err(Error::Loader(msg)) => {
                assert!(
                    msg.contains("tensor0") && msg.contains("256"),
                    "error should name the tensor and its block size: {msg}"
                );
            }
            other => panic!("misaligned quant numel should be Error::Loader, got {other:?}"),
        }
    }

    /// The alignment guard must not reject legitimate quantised tensors: one whole Q4_K block
    /// (256 elements → 144 bytes) still opens and sizes exactly.
    #[test]
    fn block_aligned_quant_numel_still_opens() {
        let b = build_typed_fixture(12, 256, 144);
        let tmp = write_temp_gguf(&b);
        let gguf = Gguf::open(tmp.path()).expect("one whole Q4_K block must open");
        let t = &gguf.tensors()[0];
        assert_eq!(t.dtype, DType::Q4K);
        assert_eq!(t.nbytes, 144, "one Q4_K block is 144 bytes");
        assert_eq!(gguf.tensor_bytes("tensor0").expect("bytes").len(), 144);
    }

    /// i2_s has no `block_layout` block to divide by — it packs 4 2-bit codes per byte plus ONE
    /// trailing per-tensor f32 scale. `numel = 3` would size to 4 bytes: all scale, no codes.
    #[test]
    fn misaligned_i2s_numel_errors() {
        // ggml type 36 = GGML_TYPE_I2_S.
        let b = build_typed_fixture(36, 3, 64);
        let tmp = write_temp_gguf(&b);
        let err = Gguf::open(tmp.path()).map(|_| ());
        match err {
            Err(Error::Loader(msg)) => assert!(
                msg.contains("tensor0") && msg.contains("i2_s"),
                "error should name the tensor and the i2_s packing: {msg}"
            ),
            other => panic!("misaligned i2_s numel should be Error::Loader, got {other:?}"),
        }
    }

    /// The F32 round-trip fixture (`block_elems == 1`) must be unaffected — every element count is
    /// trivially a whole number of blocks for an unquantised dtype.
    #[test]
    fn unquantised_numel_is_never_rejected() {
        let tmp = write_temp_gguf(&build_fixture());
        Gguf::open(tmp.path()).expect("F32 tensors have no block-alignment constraint");
    }

    // ── multi-shard (`gguf-split`) fixtures ───────────────────────────────────

    /// Build ONE shard of a synthetic `gguf-split` set.
    ///
    /// Mirrors what llama.cpp's `gguf-split` writes: the three `split.*` keys on every shard (with
    /// `split.no`/`split.count` as UINT16 and `split.tensors.count` as INT32, the types it uses),
    /// the model's own metadata on the first shard only, and a tensor directory whose offsets are
    /// relative to THIS shard's data region. Every tensor is 1-D F32 so the fixture can carry
    /// distinguishable values; each is padded to the 32-byte alignment the next offset assumes.
    ///
    /// `split_no` and `split_count` are written verbatim rather than derived, because the
    /// validation tests exist to make a shard LIE about its own position.
    fn build_split_shard(
        split_no: u64,
        split_count: u64,
        tensors_total: u64,
        tensors: &[(&str, &[f32])],
    ) -> Vec<u8> {
        let first = split_no == 0;
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3); // version
        push_u64(&mut b, tensors.len() as u64);
        push_u64(&mut b, if first { 4 } else { 3 }); // kv_count

        if first {
            // Only shard 1 carries the model's own metadata — the rest hold `split.*` and nothing
            // else, which is why a later shard can never be loaded on its own.
            push_gguf_str(&mut b, "general.architecture");
            push_u32(&mut b, 8); // STRING
            push_gguf_str(&mut b, "shardtest");
        }
        push_gguf_str(&mut b, KEY_SPLIT_NO);
        push_u32(&mut b, 2); // UINT16
        b.extend_from_slice(&(split_no as u16).to_le_bytes());
        push_gguf_str(&mut b, KEY_SPLIT_COUNT);
        push_u32(&mut b, 2); // UINT16
        b.extend_from_slice(&(split_count as u16).to_le_bytes());
        push_gguf_str(&mut b, KEY_SPLIT_TENSORS_COUNT);
        push_u32(&mut b, 5); // INT32
        push_u32(&mut b, tensors_total as u32);

        let mut off = 0u64;
        for (name, vals) in tensors {
            push_gguf_str(&mut b, name);
            push_u32(&mut b, 1); // n_dims
            push_u64(&mut b, vals.len() as u64);
            push_u32(&mut b, 0); // F32
            push_u64(&mut b, off);
            off += (vals.len() * 4).next_multiple_of(32) as u64;
        }

        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        for (_, vals) in tensors {
            let start = b.len();
            for v in *vals {
                b.extend_from_slice(&v.to_le_bytes());
            }
            while !(b.len() - start).is_multiple_of(32) {
                b.push(0);
            }
        }
        b
    }

    /// Write `shards` into a fresh temp dir under `gguf-split`'s `-NNNNN-of-MMMMM.gguf` naming.
    /// Returns the dir (which must outlive the paths) and the shard paths in order.
    fn write_shard_set(shards: &[Vec<u8>]) -> (tempfile::TempDir, Vec<std::path::PathBuf>) {
        let dir = tempfile::tempdir().expect("tempdir");
        let total = shards.len();
        let paths = shards
            .iter()
            .enumerate()
            .map(|(i, bytes)| {
                let p = dir
                    .path()
                    .join(format!("m-Q4_K_M-{:05}-of-{total:05}.gguf", i + 1));
                std::fs::write(&p, bytes).expect("write shard");
                p
            })
            .collect();
        (dir, paths)
    }

    /// The ordinary two-shard set: `a`/`b` in shard 1, `c` in shard 2, all with distinct values.
    fn two_shard_fixture() -> (tempfile::TempDir, Vec<std::path::PathBuf>) {
        write_shard_set(&[
            build_split_shard(0, 2, 3, &[("a", &[1.0, 2.0, 3.0, 4.0]), ("b", &[5.0, 6.0])]),
            build_split_shard(1, 2, 3, &[("c", &[7.0, 8.0, 9.0])]),
        ])
    }

    fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn open_err(path: &Path) -> String {
        match Gguf::open(path).map(|_| ()) {
            Err(Error::Loader(msg)) => msg,
            other => panic!("expected an Error::Loader, got {other:?}"),
        }
    }

    /// Opening shard 1 of a split model yields the WHOLE model: shard 1's metadata, every shard's
    /// tensors, and — the part a presence-only check would miss — a shard-2 tensor reading back its
    /// OWN bytes rather than whatever sits at that offset in shard 1.
    #[test]
    fn a_shard_set_reads_each_tensor_from_its_own_file() {
        let (_dir, paths) = two_shard_fixture();
        let gguf = Gguf::open(&paths[0]).expect("open shard 1");

        assert_eq!(
            gguf.metadata().str("general.architecture"),
            Some("shardtest")
        );
        let names: Vec<&str> = gguf.tensors().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b", "c"]);

        assert_eq!(
            f32s(gguf.tensor_bytes("a").expect("a")),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(f32s(gguf.tensor_bytes("b").expect("b")), vec![5.0, 6.0]);
        // `c` lives in shard 2, and shard 1 is long enough that `c`'s offset is inside it too — so
        // a `c` resolved against the wrong mapping comes back readable and the length check passes.
        // Asserting the VALUES is what tells the two files apart.
        assert_eq!(
            f32s(gguf.tensor_bytes("c").expect("c")),
            vec![7.0, 8.0, 9.0]
        );

        // Both shards are mapped, laid end to end at their own file lengths.
        let shard1_len = std::fs::metadata(&paths[0]).unwrap().len();
        assert_eq!(
            gguf.shards(),
            vec![
                (paths[0].as_path(), shard1_len),
                (
                    paths[1].as_path(),
                    std::fs::metadata(&paths[1]).unwrap().len()
                ),
            ]
        );
        // `c`'s reported file range is in the CONCATENATED space, i.e. past all of shard 1.
        let (off, len) = gguf.tensor_file_range("c").expect("range");
        assert_eq!(len, 12);
        assert!(
            off >= shard1_len,
            "a shard-2 tensor must be addressed past shard 1 ({off} < {shard1_len})"
        );
    }

    /// The pager's disk tier reads those same concatenated offsets through
    /// `FileBlockIo::open_shards`. This is the seam where a disagreement about the address space
    /// would surface as silently wrong weights, so it is checked end to end: the bytes the block
    /// reader returns for a shard-2 tensor must be the bytes the mapping serves.
    #[test]
    fn the_block_reader_and_the_mapping_agree_across_shards() {
        use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo, FileBlockIo};
        let (_dir, paths) = two_shard_fixture();
        let gguf = Gguf::open(&paths[0]).expect("open shard 1");
        let io = FileBlockIo::open_shards(&gguf.shards()).expect("open shard set");

        for name in ["a", "b", "c"] {
            let mapped = gguf.tensor_bytes(name).expect("mapped").to_vec();
            let (offset, len) = gguf.tensor_file_range(name).expect("range");
            let mut got = vec![0u8; len];
            io.read_block(
                &BlockDesc {
                    id: 0,
                    extents: vec![BlockExtent { offset, len }],
                },
                &mut got,
            )
            .expect("read block");
            assert_eq!(
                got, mapped,
                "block read for '{name}' must match the mapping"
            );
        }
    }

    /// A later shard carries no model metadata at all, so opening one alone could only ever fail.
    /// Its filename names the whole set, so the loader re-roots at shard 1 and yields the same
    /// model — `infr run <any shard>` works.
    #[test]
    fn opening_a_later_shard_loads_the_whole_model() {
        let (_dir, paths) = two_shard_fixture();
        let from_first = Gguf::open(&paths[0]).expect("open shard 1");
        let from_second = Gguf::open(&paths[1]).expect("open shard 2");

        assert_eq!(
            from_second.metadata().str("general.architecture"),
            Some("shardtest"),
            "the metadata must come from shard 1 even when shard 2 was named"
        );
        assert_eq!(from_second.tensors().len(), 3);
        for name in ["a", "b", "c"] {
            assert_eq!(
                from_second.tensor_bytes(name).expect("bytes"),
                from_first.tensor_bytes(name).expect("bytes"),
                "'{name}' must read identically whichever shard was opened"
            );
        }
        assert_eq!(from_second.shards(), from_first.shards());
    }

    /// A shard named by `split.count` that is not on disk must fail naming the file, not silently
    /// load a model missing a third of its weights.
    #[test]
    fn a_missing_shard_errors_by_name() {
        let (_dir, paths) = two_shard_fixture();
        std::fs::remove_file(&paths[1]).expect("remove shard 2");
        let msg = open_err(&paths[0]);
        assert!(
            msg.contains("m-Q4_K_M-00002-of-00002.gguf") && msg.contains("shard 2 of 2"),
            "{msg}"
        );
    }

    /// A shard whose `split.no` disagrees with the position its filename claims means the files
    /// are not the set they say they are — every tensor in it would be read from the wrong file.
    #[test]
    fn a_shard_no_disagreeing_with_its_filename_errors() {
        // Shard 2's file says `split.no = 2` (i.e. the third shard) while being named `-00002-`.
        let (_dir, paths) = write_shard_set(&[
            build_split_shard(0, 2, 3, &[("a", &[1.0]), ("b", &[2.0])]),
            build_split_shard(2, 2, 3, &[("c", &[3.0])]),
        ]);
        let msg = open_err(&paths[0]);
        assert!(
            msg.contains("m-Q4_K_M-00002-of-00002.gguf")
                && msg.contains(KEY_SPLIT_NO)
                && msg.contains("shard 2 of 2"),
            "{msg}"
        );
    }

    /// Two shards disagreeing about how many shards the model has are not one model.
    #[test]
    fn a_split_count_mismatch_between_shards_errors() {
        let (_dir, paths) = write_shard_set(&[
            build_split_shard(0, 2, 3, &[("a", &[1.0]), ("b", &[2.0])]),
            build_split_shard(1, 5, 3, &[("c", &[3.0])]),
        ]);
        let msg = open_err(&paths[0]);
        assert!(
            msg.contains("m-Q4_K_M-00002-of-00002.gguf")
                && msg.contains(KEY_SPLIT_COUNT)
                && msg.contains('5'),
            "{msg}"
        );
    }

    /// The filename's `-of-MMMMM` is what BOUNDS the set — it is the reason a hostile
    /// `split.count` cannot make the loader enumerate millions of files — so it must agree with
    /// the metadata rather than be quietly overridden by it.
    #[test]
    fn a_filename_total_disagreeing_with_split_count_errors() {
        let (_dir, paths) = write_shard_set(&[
            build_split_shard(0, 7, 3, &[("a", &[1.0]), ("b", &[2.0])]),
            build_split_shard(1, 7, 3, &[("c", &[3.0])]),
        ]);
        let msg = open_err(&paths[0]);
        assert!(
            msg.contains("m-Q4_K_M-00001-of-00002.gguf") && msg.contains('7'),
            "{msg}"
        );
    }

    /// `split.count` on a file that is not named as a shard leaves the rest of the set unnameable.
    #[test]
    fn a_split_file_without_a_shard_name_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("renamed.gguf");
        std::fs::write(&p, build_split_shard(0, 2, 3, &[("a", &[1.0])])).expect("write");
        let msg = open_err(&p);
        assert!(
            msg.contains("renamed.gguf") && msg.contains("gguf-split"),
            "{msg}"
        );
    }

    /// The assembled index must be exactly as long as the model says it is: a shard set that
    /// parses cleanly but is short a tensor is the failure this whole path exists to prevent.
    #[test]
    fn a_tensor_total_disagreeing_with_split_tensors_count_errors() {
        let (_dir, paths) = write_shard_set(&[
            build_split_shard(0, 2, 4, &[("a", &[1.0]), ("b", &[2.0])]),
            build_split_shard(1, 2, 4, &[("c", &[3.0])]),
        ]);
        let msg = open_err(&paths[0]);
        assert!(
            msg.contains(KEY_SPLIT_TENSORS_COUNT) && msg.contains('4') && msg.contains('3'),
            "{msg}"
        );
    }

    /// Without `split.tensors.count` there is nothing to check the assembled index against, and
    /// `gguf-split` always writes it — so its absence is the file not being what it claims.
    #[test]
    fn a_split_model_without_a_tensor_total_errors() {
        // Hand-built: the first shard's KV block minus `split.tensors.count`.
        let mut b: Vec<u8> = Vec::new();
        push_u32(&mut b, GGUF_MAGIC);
        push_u32(&mut b, 3);
        push_u64(&mut b, 0); // tensor_count
        push_u64(&mut b, 2); // kv_count
        push_gguf_str(&mut b, KEY_SPLIT_NO);
        push_u32(&mut b, 2);
        b.extend_from_slice(&0u16.to_le_bytes());
        push_gguf_str(&mut b, KEY_SPLIT_COUNT);
        push_u32(&mut b, 2);
        b.extend_from_slice(&2u16.to_le_bytes());
        let (_dir, paths) = write_shard_set(&[b, build_split_shard(1, 2, 0, &[])]);
        let msg = open_err(&paths[0]);
        assert!(msg.contains(KEY_SPLIT_TENSORS_COUNT), "{msg}");
    }

    /// The duplicate-name check must span the SET, not each shard: one name arriving from two
    /// shards is the same ambiguity, and `resolve`'s linear `find` would silently serve the first.
    #[test]
    fn a_tensor_name_repeated_across_shards_errors() {
        let (_dir, paths) = write_shard_set(&[
            build_split_shard(0, 2, 3, &[("a", &[1.0]), ("b", &[2.0])]),
            build_split_shard(1, 2, 3, &[("a", &[3.0])]),
        ]);
        let msg = open_err(&paths[0]);
        assert!(msg.contains("duplicate tensor name 'a'"), "{msg}");
    }

    /// A shard numbered outside the set its own name describes — `-00000-` (there is no shard
    /// zero; the index is 1-based) or one past the total — is not a member of it.
    #[test]
    fn a_shard_index_outside_its_own_set_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, shown) in [
            ("m-00000-of-00002.gguf", "numbered 0 of 2"),
            ("m-00003-of-00002.gguf", "numbered 3 of 2"),
        ] {
            let p = dir.path().join(name);
            std::fs::write(&p, build_split_shard(0, 2, 1, &[("a", &[1.0])])).expect("write");
            let msg = open_err(&p);
            assert!(msg.contains(name) && msg.contains(shown), "{name}: {msg}");
        }
    }

    /// A file with a shard-shaped NAME but no `split.*` metadata is an ordinary single-file model
    /// — the loader follows metadata, never the filename, so nothing goes looking for siblings.
    #[test]
    fn a_shard_shaped_name_without_split_metadata_is_one_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("m-00001-of-00003.gguf");
        std::fs::write(&p, build_fixture()).expect("write");
        let gguf = Gguf::open(&p).expect("a plain GGUF must load whatever it is named");
        assert_eq!(gguf.tensors().len(), 1);
        assert_eq!(gguf.shards().len(), 1);
    }

    // ── gated real-model test (skipped offline) ───────────────────────────────

    #[test]
    fn real_model_if_path_set() {
        let path = match std::env::var("INFR_TEST_GGUF") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => return, // skip when env var is not set
        };
        let gguf = Gguf::open(&path).expect("open real GGUF");
        assert!(
            !gguf.tensors().is_empty(),
            "real model should have at least one tensor"
        );
        assert!(
            !gguf.metadata().kv.is_empty(),
            "real model should have non-empty metadata"
        );
    }
}
