//! The weight-source seam — formats (GGUF now, safetensors later) implement this.

use crate::error::Result;
use crate::tensor::DType;
use std::collections::HashMap;

/// A typed metadata value from a model file's KV store.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    Bool(bool),
    U64(u64),
    I64(i64),
    F64(f64),
    Str(String),
    Arr(Vec<MetaValue>),
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetaValue::U64(v) => Some(*v),
            // A NEGATIVE `I64` is NOT a valid unsigned count/size — reject it (`None`) rather than
            // wrapping (`-1 as u64` == `u64::MAX`), which would drive a downstream alloc/loop into
            // OOM/overflow instead of a clean "invalid field" rejection.
            MetaValue::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetaValue::F64(v) => Some(*v),
            MetaValue::U64(v) => Some(*v as f64),
            MetaValue::I64(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[MetaValue]> {
        match self {
            MetaValue::Arr(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Metadata {
    pub kv: HashMap<String, MetaValue>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Metadata {
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.kv.get(key)
    }
    pub fn u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(MetaValue::as_u64)
    }
    pub fn str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(MetaValue::as_str)
    }
}

/// Where a tensor lives in the backing file.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Byte offset of this tensor's bytes in the weight source's file address space — the data
    /// region's start already added in. A `gguf-split` model is several files, addressed end to
    /// end in shard order, so this is an offset into that concatenated space (see
    /// `infr_gguf::Gguf::tensor_file_range`, which reports it, and
    /// `crate::blockio::FileBlockIo::open_shards`, which reads it).
    pub offset: u64,
    pub nbytes: usize,
}

/// A model weight file. GGUF is the MVP impl (`infr-gguf`).
pub trait WeightSource: Send + Sync {
    fn metadata(&self) -> &Metadata;
    fn tensors(&self) -> &[TensorInfo];
    /// Raw bytes for a named tensor (quantized blocks returned as-is).
    fn tensor_bytes(&self, name: &str) -> Result<&[u8]>;
    /// Embedded jinja chat template, if present.
    fn chat_template(&self) -> Option<&str>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `as_u64` must reject a NEGATIVE `I64` (a count/size field can't be negative) instead of
    /// wrapping it into a huge `u64` that drives a downstream alloc/loop into OOM.
    #[test]
    fn as_u64_rejects_negative_i64() {
        assert_eq!(MetaValue::I64(-1).as_u64(), None);
        assert_eq!(MetaValue::I64(i64::MIN).as_u64(), None);
        // A non-negative `I64` still reads through unchanged.
        assert_eq!(MetaValue::I64(0).as_u64(), Some(0));
        assert_eq!(MetaValue::I64(42).as_u64(), Some(42));
        assert_eq!(MetaValue::I64(i64::MAX).as_u64(), Some(i64::MAX as u64));
        // A native `U64` is untouched (including values above `i64::MAX`).
        assert_eq!(MetaValue::U64(u64::MAX).as_u64(), Some(u64::MAX));
        // Non-integer variants stay `None`.
        assert_eq!(MetaValue::Bool(true).as_u64(), None);
    }
}
