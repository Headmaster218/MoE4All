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
    loader::{MetaValue, Metadata, TensorInfo},
    tensor::DType,
    WeightSource,
};
use memmap2::Mmap;
use std::{collections::HashMap, fs::File, path::Path, sync::Arc};

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

// ─── public struct ────────────────────────────────────────────────────────────

/// A parsed, mmap-backed GGUF file.
///
/// The `Mmap` handle keeps the backing memory alive for the lifetime of this
/// struct; `tensor_bytes` returns slices directly into that region.
pub struct Gguf {
    mmap: Arc<Mmap>,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    /// Absolute byte offset into `mmap` where tensor data begins.
    data_region_start: usize,
}

/// An owning, ref-counted view of a tensor's bytes in the mmap'd file — a zero-copy `[u8]` slice that
/// keeps the whole `Mmap` alive via `Arc`, so it can outlive the borrow of `&Gguf` (e.g. a CPU
/// backend buffer that reads weights straight from the mapping with no `memcpy`).
#[derive(Clone)]
pub struct TensorBytes {
    mmap: Arc<Mmap>,
    off: usize,
    len: usize,
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
    pub fn open(path: &Path) -> Result<Self> {
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

            // Duplicate tensor names are malformed input and must not be silently accepted.
            // `resolve` looks a tensor up with a linear `find`, so with two entries sharing a
            // name the FIRST always wins and the second is unreachable — different offsets,
            // different shapes, possibly a different dtype, and nothing anywhere reports that
            // half the file was ignored. Rejecting at load is the only place the ambiguity is
            // visible; afterwards every consumer just gets a plausible-looking answer. One
            // `HashSet` pass over the (few thousand, at most) names costs nothing measurable.
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

            (metadata, tensors, data_region_start)
        }; // ← borrow of `mmap` ends here

        Ok(Gguf {
            mmap: Arc::new(mmap),
            metadata,
            tensors,
            data_region_start,
        })
    }

    /// Zero-copy, ref-counted view of a tensor's raw bytes (keeps the `Mmap` alive via `Arc`). Unlike
    /// [`WeightSource::tensor_bytes`] the result is not borrow-bound to `&self`, so a backend can hold
    /// it as a weight buffer and read straight from the mapping — no `memcpy` into owned RAM.
    pub fn tensor_bytes_arc(&self, name: &str) -> Result<TensorBytes> {
        let (off, len) = self.resolve(name)?;
        Ok(TensorBytes {
            mmap: Arc::clone(&self.mmap),
            off,
            len,
        })
    }

    /// Look up a tensor by name and return its `(absolute_offset, len)` in the mmap,
    /// bounds-checked against the file size with `checked_add` so a crafted
    /// offset/length can't overflow. Shared by [`Self::tensor_bytes_arc`] and
    /// [`WeightSource::tensor_bytes`] so the lookup + overflow-safe bounds check lives
    /// in exactly one place.
    fn resolve(&self, name: &str) -> Result<(usize, usize)> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::Loader(format!("tensor not found: '{name}'")))?;
        let off = self
            .data_region_start
            .checked_add(info.offset as usize)
            .ok_or_else(|| Error::Loader(format!("tensor '{name}' offset overflows usize")))?;
        let end = off
            .checked_add(info.nbytes)
            .ok_or_else(|| Error::Loader(format!("tensor '{name}' byte range overflows usize")))?;
        if end > self.mmap.len() {
            return Err(Error::Loader(format!(
                "tensor '{name}' byte range {off}..{end} exceeds file size {}",
                self.mmap.len()
            )));
        }
        Ok((off, info.nbytes))
    }
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

    /// Returns a slice into the mmap'd data region for the named tensor.
    ///
    /// The slice lifetime is tied to `&self` (i.e. the `Gguf` struct keeps
    /// the `Mmap` alive).
    fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let (start, len) = self.resolve(name)?;
        Ok(&self.mmap[start..start + len])
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
