//! `infr-testkit` — the **shared backend parity harness**.
//!
//! One place that knows how to (a) synthesize a valid block-quantized weight of ANY GGUF format,
//! (b) drive a one-op [`Graph`] through the agnostic [`Backend`] seam, and (c) score the result
//! against the host decode oracle (`infr_gguf::dequant::dequant_block`) plus a reference GEMV — so
//! cpu / vulkan / metal / rocm all get the same decode coverage from one source instead of each
//! maintaining its own `lcg_bytes` + `synth_q4k` + `ref_linear` + bind-run-download boilerplate.
//!
//! ## Why a crate and not a `cfg(test)` module in `infr-core`
//!
//! The oracle IS `infr_gguf::dequant_block`, and `infr-gguf` depends on `infr-core` — so an
//! infr-core module reaching for it would be a dependency cycle, cargo feature or not. A separate
//! crate also keeps the harness out of every production dependency graph by construction (it is a
//! `[dev-dependencies]` entry everywhere), which a `feature = "test-harness"` cannot guarantee
//! once workspace feature unification is in play.
//!
//! ## What it does NOT do
//!
//! It does not re-implement block decode. The spec ([`infr_core::decode_spec`]) says where a
//! block's scale fields are; `dequant_block` says what the bytes mean; the device kernels say what
//! the GPU computes. The harness only makes the three meet.
//!
//! ## Shape of a parity test
//!
//! ```ignore
//! use infr_testkit::{sweep_linear_on, weight_quant_cases};
//! let be = MyBackend::new().unwrap();
//! let cases = weight_quant_cases(2, 256, 8);           // all 24 weight quants, m=2
//! sweep_linear_on("MyBackend Linear m=2", &be, &cases, |_| 2e-2).assert_ok();
//! ```
//!
//! Use [`sweep_linear`] instead when the backend keys a dequantized-weight cache by the weight
//! buffer's raw device ADDRESS (ROCm does) — it takes a closure that mints a fresh backend per
//! case, so a recycled VRAM address cannot serve the previous format's rows out of a stale cache.
//!
//! Every device-side entry point takes `&dyn Backend`, so the harness needs no GPU-specific API
//! and a backend gets coverage the moment it implements the seam.

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::decode_spec::{block_spec, WEIGHT_QUANTS};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};

// ─── deterministic payload generation ────────────────────────────────────────────────────────

/// Deterministic LCG byte stream — an arbitrary but reproducible payload for the code / nibble /
/// grid-index fields of a quant block, all of which decode to FINITE values for ANY bit pattern.
/// (Only a block's scale slots must be sane; [`synth_weight`] writes those from the spec.)
pub fn lcg_bytes(mut seed: u32, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as u8
        })
        .collect()
}

/// Deterministic small-magnitude f32 stream — activations that stay well inside f16 range so a
/// backend that rounds activations to f16 is not being tested on overflow.
pub fn gen_f32(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
        .collect()
}

/// The per-format `(d, min)` magnitudes [`synth_weight`] writes into a block's scale slots.
///
/// These are NOT arbitrary: a synthetic weight has to land in a realistic magnitude band or the
/// parity comparison degenerates. Formats whose codes are wide (Q8_0's ±127, IQ3_S's grid) take a
/// small `d`; the low-bit-width ternary/codebook formats take a larger one. Values match the
/// magnitudes the pre-existing ROCm and Metal parity suites had converged on independently, so
/// moving those suites onto the harness does not change what they measure.
pub fn synth_scales(dtype: DType) -> (f32, f32) {
    match dtype {
        DType::Q8_0 => (0.01, 0.0),
        DType::Q4_0 | DType::Q5_0 => (0.04, 0.0),
        DType::Q4_1 | DType::Q5_1 => (0.04, -0.30),
        DType::Q2K | DType::Q4K | DType::Q5K => (0.05, 0.10),
        DType::Q3K | DType::Q6K => (0.03, 0.0),
        DType::Iq4Nl => (0.004, 0.0),
        DType::Iq4Xs => (0.06, 0.0),
        DType::Iq2Xxs | DType::Iq2Xs | DType::Iq2S => (0.015, 0.0),
        DType::Iq3Xxs => (0.008, 0.0),
        DType::Iq3S => (0.002, 0.0),
        DType::Iq1S | DType::Iq1M => (0.03, 0.0),
        DType::Tq1_0 | DType::Tq2_0 | DType::Q2_0 => (0.05, 0.0),
        // MXFP4's E8M0 is a pure power of two; NVFP4's UE4M3 sub-scales spread around `d`.
        DType::Mxfp4 => (0.25, 0.0),
        DType::Nvfp4 => (0.5, 0.0),
        DType::Turbo2 | DType::Turbo3 | DType::Turbo4 => (0.05, 0.0),
        _ => (1.0, 0.0),
    }
}

/// Synthesize a **valid** block-quantized weight of `n_elem` elements in `dtype`, entirely from
/// [`infr_core::decode_spec`]: LCG payload bytes (finite-decoding for any pattern) with every
/// declared scale slot overwritten by [`BlockSpec::write_scales`](infr_core::decode_spec::BlockSpec::write_scales)
/// at this format's [`synth_scales`] magnitudes.
///
/// This is the payoff of having a spec: ONE builder covers all 24 weight quants, including the
/// three with non-f16 scale encodings (MXFP4's E8M0, NVFP4's four UE4M3 sub-scales, IQ1_M's `d`
/// split across the top nibbles of its four scale words) that previously needed bespoke per-format
/// builders duplicated in two backends' test suites.
///
/// Panics if `n_elem` is not a whole number of blocks.
pub fn synth_weight(dtype: DType, n_elem: usize, seed: u32) -> Vec<u8> {
    let s = block_spec(dtype);
    assert_eq!(
        n_elem % s.block_elems,
        0,
        "{dtype:?}: {n_elem} elements is not a whole number of {}-element blocks",
        s.block_elems
    );
    let nblk = n_elem / s.block_elems;
    let (d, min) = synth_scales(dtype);
    let mut out = Vec::with_capacity(nblk * s.block_bytes);
    for b in 0..nblk {
        let mut blk = lcg_bytes(seed ^ b as u32, s.block_bytes);
        s.write_scales(&mut blk, d, min);
        out.extend_from_slice(&blk);
    }
    out
}

/// f16 little-endian bytes for an f32 slice (an F16 weight/table upload).
pub fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
        .collect()
}

/// Raw little-endian bytes for an f32 slice.
pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

// ─── the host oracle ─────────────────────────────────────────────────────────────────────────

/// Host decode oracle: `infr_gguf::dequant::dequant_block`. Named so a parity test reads as
/// "compare against the oracle", and so there is exactly one place to change if the oracle moves.
pub fn dequant_oracle(dtype: DType, bytes: &[u8]) -> Vec<f32> {
    infr_gguf::dequant::dequant_block(dtype, bytes)
        .unwrap_or_else(|e| panic!("host dequant oracle failed for {dtype:?}: {e}"))
}

/// Reference GEMV/GEMM: `dst[r, o] = Σ_i x[r, i] · w[o, i]`, `w` row-major `[out_f, in_f]`.
/// Accumulated in f64 and rounded once, so the reference itself contributes no reassociation
/// error to the comparison — the measured gap is the DEVICE's, not the oracle's.
pub fn ref_linear(x: &[f32], w: &[f32], m: usize, in_f: usize, out_f: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * out_f];
    for r in 0..m {
        for o in 0..out_f {
            let mut acc = 0f64;
            for i in 0..in_f {
                acc += x[r * in_f + i] as f64 * w[o * in_f + i] as f64;
            }
            out[r * out_f + o] = acc as f32;
        }
    }
    out
}

// ─── driving a backend through the seam ──────────────────────────────────────────────────────

/// Bind raw byte buffers to graph handles, run `g` once on `be`, and download `out` as f32.
///
/// The bind-run-download boilerplate every backend's op-parity test re-spells by hand. `bound`
/// gives the raw bytes for each Input/Weight handle; every other declared handle is allocated
/// zeroed. Uses only [`Backend`] methods, so it works for any backend on the seam.
pub fn run_graph(
    be: &dyn Backend,
    g: &Graph,
    bound: &[(TensorId, Vec<u8>)],
    out: TensorId,
    out_n: usize,
) -> Vec<f32> {
    let plan = be.compile(g).expect("compile");
    let mut bufs = Vec::new();
    let mut binds = Bindings::new();
    for (id, bytes) in bound {
        let usage = match g.tensors[id.0 as usize].kind {
            infr_core::graph::TensorKind::Weight => BufferUsage::Weights,
            _ => BufferUsage::Activations,
        };
        let b = be.alloc(bytes.len().max(4), usage).expect("alloc input");
        be.upload(b.as_ref(), bytes).expect("upload");
        bufs.push((*id, b));
    }
    let ob = be
        .alloc(out_n * 4, BufferUsage::Readback)
        .expect("alloc output");
    for (id, b) in &bufs {
        binds.bind(*id, b.as_ref());
    }
    binds.bind(out, ob.as_ref());
    be.execute(plan.as_ref(), &binds).expect("execute");
    let mut o = vec![0f32; out_n];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .expect("download");
    o
}

/// One `Op::Linear` parity case: `dst[m, out_f] = x[m, in_f] · w[out_f, in_f]ᵀ` with `w` uploaded
/// as its RAW `dtype` bytes.
#[derive(Clone, Copy, Debug)]
pub struct LinearCase {
    pub dtype: DType,
    pub m: usize,
    pub in_f: usize,
    pub out_f: usize,
    /// Seed for both the synthetic weight and the activations, so a case is reproducible.
    pub seed: u32,
}

impl LinearCase {
    pub fn new(dtype: DType, m: usize, in_f: usize, out_f: usize) -> Self {
        Self {
            dtype,
            m,
            in_f,
            out_f,
            seed: 0x5eed,
        }
    }

    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }
}

/// What a parity case measured.
#[derive(Clone, Debug)]
pub struct CaseReport {
    pub dtype: DType,
    pub m: usize,
    pub in_f: usize,
    pub out_f: usize,
    /// Max absolute deviation from the oracle.
    pub abs: f32,
    /// `abs / max|oracle|` — the number tolerances are stated in.
    pub rel: f32,
    /// `max|oracle|`; a case whose oracle is all-zero is VACUOUS, not passing.
    pub ref_mag: f32,
    /// Tolerance the case was scored against.
    pub tol: f32,
}

impl CaseReport {
    /// A case passes when it is non-vacuous AND inside tolerance.
    pub fn ok(&self) -> bool {
        self.ref_mag > 1e-3 && self.rel < self.tol
    }

    pub fn line(&self) -> String {
        let v = if self.ref_mag > 1e-3 {
            ""
        } else {
            "  << VACUOUS (oracle is all-zero)"
        };
        format!(
            "{:8?} m={:<3} {}x{} rel={:.3e} abs={:.3e} max|ref|={:.3e} tol={:.1e}{}{}",
            self.dtype,
            self.m,
            self.in_f,
            self.out_f,
            self.rel,
            self.abs,
            self.ref_mag,
            self.tol,
            if self.rel < self.tol { "" } else { "  << OVER" },
            v
        )
    }
}

/// Build + run one `Op::Linear` case on `be` and score it against the oracle
/// (`dequant_block` of the SAME bytes, then [`ref_linear`]).
///
/// The oracle is deliberately the HOST DECODE, not another backend: comparing two backends only
/// proves they agree, while comparing to `dequant_block` proves the device decoder implements the
/// format. `tol` is a RELATIVE bound against `max|oracle|` — see each call site for its rationale.
/// The synthetic `(weight bytes, activations)` a [`LinearCase`] stands for — the same pair
/// [`check_linear`] scores, exposed so a caller can drive TWO backends (or two modes of one
/// backend) on identical inputs and compare them to each other instead of to the oracle.
pub fn case_inputs(case: LinearCase) -> (Vec<u8>, Vec<f32>) {
    let LinearCase {
        dtype,
        m,
        in_f,
        out_f,
        seed,
    } = case;
    let n = out_f * in_f;
    let w_bytes = if dtype == DType::F16 {
        f16_bytes(&gen_f32(n, seed as usize))
    } else if dtype == DType::F32 {
        f32_bytes(&gen_f32(n, seed as usize))
    } else {
        synth_weight(dtype, n, seed)
    };
    (w_bytes, gen_f32(m * in_f, seed as usize ^ 0x9e37))
}

/// Run one [`LinearCase`] on `be` and return the backend's raw output — [`check_linear`] without
/// the scoring. Use it to compare two backends (e.g. `CpuBackend::new()` vs
/// `CpuBackend::reference()`) on byte-identical inputs.
pub fn run_linear(be: &dyn Backend, case: LinearCase) -> Vec<f32> {
    let LinearCase {
        dtype,
        m,
        in_f,
        out_f,
        ..
    } = case;
    let (w_bytes, x) = case_inputs(case);
    let mut g = Graph::new();
    let xid = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let wid = g.weight(TensorDesc::new(vec![out_f, in_f], dtype));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x: xid,
        weight: wid,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    run_graph(
        be,
        &g,
        &[(xid, f32_bytes(&x)), (wid, w_bytes)],
        dst,
        m * out_f,
    )
}

pub fn check_linear(be: &dyn Backend, case: LinearCase, tol: f32) -> CaseReport {
    let LinearCase {
        dtype,
        m,
        in_f,
        out_f,
        ..
    } = case;
    let (w_bytes, x) = case_inputs(case);
    let w_ref = dequant_oracle(dtype, &w_bytes);
    let want = ref_linear(&x, &w_ref, m, in_f, out_f);

    let mut g = Graph::new();
    let xid = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let wid = g.weight(TensorDesc::new(vec![out_f, in_f], dtype));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x: xid,
        weight: wid,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let got = run_graph(
        be,
        &g,
        &[(xid, f32_bytes(&x)), (wid, w_bytes)],
        dst,
        m * out_f,
    );

    let abs = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let ref_mag = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    CaseReport {
        dtype,
        m,
        in_f,
        out_f,
        abs,
        rel: abs / ref_mag.max(1e-3),
        ref_mag,
        tol,
    }
}

/// Every real GGUF weight quant as a `LinearCase` at the given shape. `in_f * out_f` must be a
/// multiple of 256 (the largest block size) so one shape covers all 24 formats — 256×8 works.
pub fn weight_quant_cases(m: usize, in_f: usize, out_f: usize) -> Vec<LinearCase> {
    assert_eq!(
        (in_f * out_f) % 256,
        0,
        "in_f*out_f must be a multiple of 256 so every block size divides it"
    );
    WEIGHT_QUANTS
        .iter()
        .enumerate()
        .map(|(i, &dt)| LinearCase::new(dt, m, in_f, out_f).with_seed(0x201 + i as u32))
        .collect()
}

/// A whole sweep's worth of reports plus a label, so a test can print every number and then fail
/// ONCE with all the offenders — a per-case `assert!` hides how many formats are broken.
#[derive(Debug)]
pub struct SweepReport {
    pub label: String,
    pub cases: Vec<CaseReport>,
}

impl SweepReport {
    /// Print every case, then panic listing the failures (if any).
    pub fn assert_ok(&self) {
        for c in &self.cases {
            println!("[{}] {}", self.label, c.line());
        }
        let bad: Vec<String> = self
            .cases
            .iter()
            .filter(|c| !c.ok())
            .map(|c| c.line())
            .collect();
        assert!(
            bad.is_empty(),
            "{}: {} of {} parity cases failed:\n  {}",
            self.label,
            bad.len(),
            self.cases.len(),
            bad.join("\n  ")
        );
    }
}

/// Run `cases` on `be`, scoring each with `tol(dtype)`. `fresh` is called before every case to get
/// the backend to run it on — some backends key a dequantized-weight cache by the weight buffer's
/// raw device address, so back-to-back single-op cases must not share one (a freed buffer's
/// address can be recycled, and a stale cache hit would feed the previous format's rows).
pub fn sweep_linear<B: Backend>(
    label: &str,
    cases: &[LinearCase],
    tol: impl Fn(DType) -> f32,
    mut fresh: impl FnMut() -> B,
) -> SweepReport {
    SweepReport {
        label: label.to_string(),
        cases: cases
            .iter()
            .map(|&c| {
                let be = fresh();
                check_linear(&be, c, tol(c.dtype))
            })
            .collect(),
    }
}

/// Run `cases` on ONE long-lived backend. Use when the backend has no address-keyed weight cache
/// (the CPU interpreter, Vulkan) — much faster than re-creating a device per case.
pub fn sweep_linear_on(
    label: &str,
    be: &dyn Backend,
    cases: &[LinearCase],
    tol: impl Fn(DType) -> f32,
) -> SweepReport {
    SweepReport {
        label: label.to_string(),
        cases: cases
            .iter()
            .map(|&c| check_linear(be, c, tol(c.dtype)))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::decode_spec::block_layout;

    /// The spec-driven builder must produce EXACTLY the byte count the spec claims, for every
    /// weight quant. A wrong `block_bytes` shows up here first.
    #[test]
    fn synth_weight_sizes_match_the_spec() {
        for &dt in WEIGHT_QUANTS {
            let (be, bb) = block_layout(dt);
            let n = be * 7;
            assert_eq!(
                synth_weight(dt, n, 1).len(),
                7 * bb,
                "{dt:?}: synth_weight byte count"
            );
        }
    }

    /// **The spec↔oracle cross-check.** Feeding the oracle a spec-sized buffer must yield exactly
    /// `n_blocks * block_elems` finite values — which independently confirms BOTH halves of the
    /// geometry (`block_bytes`, because the oracle consumes the buffer in whole blocks, and
    /// `block_elems`, because that is the output length) against 24 hand-written decoders.
    #[test]
    fn oracle_agrees_with_the_spec_geometry_for_every_weight_quant() {
        for &dt in WEIGHT_QUANTS {
            let (be, _) = block_layout(dt);
            let n = be * 5;
            let out = dequant_oracle(dt, &synth_weight(dt, n, 7));
            assert_eq!(
                out.len(),
                n,
                "{dt:?}: oracle decoded a different element count"
            );
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{dt:?}: oracle produced a non-finite value from a spec-synthesized block"
            );
        }
    }

    /// **The scale-slot cross-check.** Doubling the value written to a format's `D` slot must
    /// (near-)double the decoded magnitude. If the declared offset pointed at a payload field
    /// instead of the real scale, the decoded magnitude would be unchanged (or move randomly) —
    /// so this is what actually pins `ScaleSlot::offset` for all 24 formats against the decoders.
    ///
    /// Exact doubling is not required: MXFP4's E8M0 only represents powers of two, NVFP4's UE4M3
    /// rounds, and IQ1_M's `d` goes through f16. The assertion is that the magnitude tracks the
    /// scale within 25%, which no wrong offset would satisfy.
    #[test]
    fn spec_scale_slots_are_the_real_scales() {
        for &dt in WEIGHT_QUANTS {
            let s = block_spec(dt);
            let (d, min) = synth_scales(dt);
            let n = s.block_elems * 4;
            let base = synth_weight(dt, n, 11);
            // Same payload bytes, `d` (and the affine `min`) doubled.
            let mut scaled = base.clone();
            for blk in scaled.chunks_exact_mut(s.block_bytes) {
                s.write_scales(blk, d * 2.0, min * 2.0);
            }
            let a = dequant_oracle(dt, &base);
            let b = dequant_oracle(dt, &scaled);
            let ma = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let mb = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                ma > 1e-9,
                "{dt:?}: synthesized block decodes to all-zero — synth_scales is degenerate"
            );
            let ratio = mb / ma;
            assert!(
                (1.75..=2.25).contains(&ratio),
                "{dt:?}: doubling the declared D slot changed the decoded magnitude by {ratio:.3}x \
                 (expected ~2x) — ScaleSlot::offset does not point at this format's real scale"
            );
        }
    }

    /// The reference GEMV is the plain triple loop, in f64. Pinned against a hand-computed case so
    /// a "clever" rewrite cannot silently change what every backend is measured against.
    #[test]
    fn ref_linear_is_the_plain_dot_product() {
        let x = [1.0f32, 2.0, 3.0, 4.0]; // m=2, in_f=2
        let w = [1.0f32, 10.0, 100.0, 1000.0]; // out_f=2, in_f=2
        assert_eq!(
            ref_linear(&x, &w, 2, 2, 2),
            vec![21.0, 2100.0, 43.0, 4300.0]
        );
    }

    /// `weight_quant_cases` covers the full roster with distinct seeds (so two formats never share
    /// a payload) and rejects a shape that does not divide by the largest block.
    #[test]
    fn weight_quant_cases_cover_the_roster() {
        let cases = weight_quant_cases(2, 256, 8);
        assert_eq!(cases.len(), WEIGHT_QUANTS.len());
        for (c, &dt) in cases.iter().zip(WEIGHT_QUANTS) {
            assert_eq!(c.dtype, dt);
        }
        let seeds: std::collections::HashSet<u32> = cases.iter().map(|c| c.seed).collect();
        assert_eq!(seeds.len(), cases.len(), "case seeds must be distinct");
    }

    /// A vacuous case (all-zero oracle) must FAIL, not silently pass — the classic way a parity
    /// suite lies about coverage.
    #[test]
    fn vacuous_case_is_not_a_pass() {
        let r = CaseReport {
            dtype: DType::Q4K,
            m: 1,
            in_f: 256,
            out_f: 8,
            abs: 0.0,
            rel: 0.0,
            ref_mag: 0.0,
            tol: 1e-2,
        };
        assert!(!r.ok(), "an all-zero oracle must not count as agreement");
    }
}
