//! `infr-core` — shared types + the four pluggable seams:
//! [`backend::Backend`] (GPU), [`loader::WeightSource`] (format), plus the
//! [`graph::Graph`]/[`tensor`] vocabulary the model layer builds against.
//!
//! Nothing here is GPU- or model-specific. See docs/plan.md.

pub mod backend;
pub mod blockio;
pub mod budget;
/// The layered `INFR_*` replacement: a typed, explicitly-passed [`config::Config`].
pub mod config;
pub mod decode_spec;
pub mod error;
pub mod exec;
pub mod fusion;
/// llama.cpp's `gguf-split` filename convention — shared by the hub (is the set complete?) and the
/// GGUF loader (which files make up this model?).
pub mod gguf_split;
pub mod graph;
pub mod hostmem;
pub mod hostpager;
pub mod iquant_grids;
/// The shared on-disk cache for compiled kernel artifacts (Vulkan pipeline blobs) —
/// envelope, durability, and the poisoned-blob tripwire, once.
pub mod kernel_cache;
pub mod loader;
pub mod pager;
pub mod pager_profile;
/// The per-op profiling seam every backend reports through — one collector, one label grammar,
/// one report, one on/off predicate (`prof.per_op()` ANDed with the warmup-suppression flag).
pub mod prof;
pub mod progress;
pub mod resource;
pub mod shutdown;
pub mod tensor;
pub mod test_resource;
pub mod tier;
pub mod timeline;

pub use backend::{
    initial_submit_dispatch_cap, integrated_ubatch_rows, submit_cap_from_measurement,
    submit_cap_from_measurement_with_budget, Backend, Bindings, Buffer, BufferUsage, Capabilities,
    GraphPlan, Plan, COOPMAT_TILE_16, COOPMAT_TILE_8, SUBMIT_BUDGET_NS, SUBMIT_DANGER_NS,
};
pub use error::{Error, Result};
pub use graph::{Activation, AttnMask, Graph, Op, TensorDecl, TensorKind};
pub use loader::{MetaValue, Metadata, TensorInfo, WeightSource};
pub use pager::{BlockId, Pager, PagerStats, Resolution, NOT_RESIDENT};
pub use resource::{MemoryTier, ResourceKind, ResourceLease, ResourceSnapshot, ResourceTracker};
pub use tensor::{DType, Shape, TensorDesc, TensorId};

/// Which part of an autoregressive request is currently making progress.
///
/// This deliberately describes model work rather than rendered text. A reasoning token and an
/// ordinary answer token are both [`Decode`](Self::Decode), even when a protocol adapter buffers
/// or routes their decoded text differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationPhase {
    Prefill,
    Decode,
}

/// Cumulative, backend-independent progress for one generation request.
///
/// Counts are token counts reported by the model runner, never estimates derived from UTF-8 text
/// pieces. Callbacks receive a snapshot after each natural work unit (a Prefill chunk or generated
/// token), so consumers can throttle presentation without losing accounting accuracy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationProgress {
    pub phase: GenerationPhase,
    /// The complete rendered prompt, including the part served from an existing KV prefix.
    pub prompt_tokens: u64,
    /// Prompt tokens served from an existing KV prefix this turn.
    pub cached_prompt_tokens: u64,
    /// Prompt tokens actually evaluated so far this turn.
    pub prefill_tokens: u64,
    /// Tokens generated so far, including reasoning tokens.
    pub completion_tokens: u64,
    /// Logical context reached so far: cached + evaluated prompt during Prefill, then the complete
    /// prompt + generated tokens during Decode.
    pub context_tokens: u64,
    /// Capacity of this request's KV/session slot.
    pub context_limit: u64,
}

/// Optional observer installed only by frontends that need live request telemetry. Keeping the
/// callback behind an `Option` in the runner leaves run/bench with one predictable branch and no
/// allocation, atomic update, or formatting on each token.
pub type GenerationProgressCallback =
    std::sync::Arc<dyn Fn(GenerationProgress) + Send + Sync + 'static>;

/// A parsed human size/count value: an absolute amount, or a percentage the CALLER resolves
/// against the knob-appropriate base (device-local VRAM for GPU budgets; total physical RAM for
/// the process-wide `device.ram_budget`; available system RAM for automatic host-cache policy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizeSpec {
    /// Absolute amount in the knob's base unit (bytes for sizes, tokens for counts).
    Bytes(u64),
    /// Fraction in `(0.0, ..]` of the caller's base (e.g. `80%` → `0.8`).
    Percent(f64),
}

impl SizeSpec {
    /// Resolve against `base` (the device-appropriate total — see the enum doc). Absolute values
    /// pass through untouched; percentages scale `base`.
    pub fn resolve(self, base: u64) -> u64 {
        match self {
            SizeSpec::Bytes(b) => b,
            SizeSpec::Percent(f) => (base as f64 * f) as u64,
        }
    }
}

/// Parse a human size/count string: a plain number is the base unit (bytes for sizes, tokens for
/// counts); an optional case-insensitive IEC suffix scales by 1024-powers — `kib`, `mib`, `gib`,
/// `tib`. The short forms `k`, `m`, `g`, `t` and historical `kb`, `mb`, `gb`, `tb` spellings stay
/// accepted as compatibility aliases; a bare `b` is the base unit, so `INFR_CACHE=256b` == `256`.
/// A `%` suffix yields [`SizeSpec::Percent`], which the caller resolves against the
/// device-appropriate base ([`SizeSpec::resolve`]). Fractional mantissas work (`1.5GiB`, `12.5%`).
/// `None` on anything else — callers treat unparseable values as unset rather than guessing.
///
/// This is the shared grammar for every size/count env the engine reads (`INFR_CACHE`,
/// `INFR_CTX`, ...): one parser so `256m` never means something different between knobs.
pub fn parse_size(s: &str) -> Option<SizeSpec> {
    let s = s.trim().to_ascii_lowercase();
    if let Some(head) = s.strip_suffix('%') {
        let v: f64 = head.trim().parse().ok()?;
        if !v.is_finite() || v <= 0.0 {
            return None;
        }
        return Some(SizeSpec::Percent(v / 100.0));
    }
    let (num, mult): (&str, u64) = match s.as_bytes() {
        [head @ .., b'k', b'i', b'b'] | [head @ .., b'k', b'b'] | [head @ .., b'k'] => {
            (std::str::from_utf8(head).ok()?, 1 << 10)
        }
        [head @ .., b'm', b'i', b'b'] | [head @ .., b'm', b'b'] | [head @ .., b'm'] => {
            (std::str::from_utf8(head).ok()?, 1 << 20)
        }
        [head @ .., b'g', b'i', b'b'] | [head @ .., b'g', b'b'] | [head @ .., b'g'] => {
            (std::str::from_utf8(head).ok()?, 1 << 30)
        }
        [head @ .., b't', b'i', b'b'] | [head @ .., b't', b'b'] | [head @ .., b't'] => {
            (std::str::from_utf8(head).ok()?, 1 << 40)
        }
        [head @ .., b'b'] => (std::str::from_utf8(head).ok()?, 1),
        _ => (s.as_str(), 1),
    };
    let num = num.trim();
    if num.is_empty() {
        return None;
    }
    let v: f64 = num.parse().ok()?;
    if !v.is_finite() || v < 0.0 {
        return None;
    }
    Some(SizeSpec::Bytes((v * mult as f64) as u64))
}

#[cfg(test)]
mod parse_size_tests {
    use super::{parse_size, SizeSpec};

    #[test]
    fn parse_size_grammar() {
        let b = |v: u64| Some(SizeSpec::Bytes(v));
        assert_eq!(parse_size("256"), b(256));
        assert_eq!(parse_size("256b"), b(256));
        assert_eq!(parse_size("256k"), b(256 << 10));
        assert_eq!(parse_size("256K"), b(256 << 10));
        assert_eq!(parse_size("256kb"), b(256 << 10));
        assert_eq!(parse_size("256KiB"), b(256 << 10));
        assert_eq!(parse_size("3m"), b(3 << 20));
        assert_eq!(parse_size("3MiB"), b(3 << 20));
        assert_eq!(parse_size("19g"), b(19u64 << 30));
        assert_eq!(parse_size("19GiB"), b(19u64 << 30));
        assert_eq!(parse_size("1t"), b(1 << 40));
        assert_eq!(parse_size("1TiB"), b(1 << 40));
        assert_eq!(parse_size("1.5GiB"), b((1.5 * (1u64 << 30) as f64) as u64));
        assert_eq!(parse_size(" 8G "), b(8u64 << 30));
        assert_eq!(parse_size("0"), b(0));
        assert_eq!(parse_size("80%"), Some(SizeSpec::Percent(0.8)));
        assert_eq!(parse_size("12.5 %"), Some(SizeSpec::Percent(0.125)));
        assert_eq!(parse_size("0%"), None);
        assert_eq!(parse_size("-5%"), None);
        assert_eq!(parse_size(""), None);
        assert_eq!(parse_size("g"), None);
        assert_eq!(parse_size("-1g"), None);
        assert_eq!(parse_size("abc"), None);
        assert_eq!(parse_size("1q"), None);
        assert_eq!(SizeSpec::Percent(0.5).resolve(24 << 30), 12 << 30);
        assert_eq!(SizeSpec::Bytes(42).resolve(24 << 30), 42);
    }
}
