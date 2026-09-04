//! DiffusionGemma self-conditioning: the Phase-A host self-cond MLP + Phase-B in-graph
//! soft-embedding weight builder, and the canvas-denoise request/plan-cache types. Pure-move
//! split of `seam.rs` — see `super` for the module overview.
use super::runner::DecodeHandles;
use crate::Config;
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Buffer, BufferUsage, Plan};

/// Phase-A perf: one (canvas_len, prompt_len)-shaped DiffusionGemma canvas-denoise graph, compiled
/// once and replayed across every denoise step that shares the shape (see the `denoise_req` branch
/// in `generate_dense_backend`). `cc` (canvas length) never changes within a session — it's the
/// model's fixed `canvas_length` — and `p` (prompt length) only changes when a block commits and
/// the next block's prefill grows the prefix. So within one block every step hits this cache: only
/// the canvas hidden/positions get re-uploaded into the buffers held here, the plan itself replays.
pub(super) struct DenoiseCache {
    pub(super) cc: usize,
    pub(super) p: usize,
    /// Phase-B/D perf: whether this plan bakes the in-graph SC subgraph (`gpu_sc: Some(true)` — see
    /// `build`'s doc). Always `false` on CPU (its graph never varies with SC). A DIFFERENT
    /// graph shape from the no-SC plan (extra ops + an extra weight/input), so it's part of the
    /// cache key: step 0 (no SC) and steps 1+ (SC on) hit two separate cached plans instead of one
    /// runtime-gated plan — see docs/diffusion-gemma.md's Phase-B "two-plan" note.
    pub(super) sc: bool,
    pub(super) plan: Box<dyn Plan>,
    pub(super) dh: DecodeHandles,
    pub(super) hidden_buf: Box<dyn Buffer>,
    pub(super) pos_buf: Box<dyn Buffer>,
    /// Perf (Vulkan — docs/diffusion-gemma.md's Phase-B "sc round-trip" elimination): `None` on
    /// Vulkan, which binds `dh.logits` to `SeamKv::sc_ping` (a session-persistent ping-pong pair)
    /// instead of a buffer owned by this per-plan cache — see the denoise call site. `Some` on
    /// Metal/CPU, which keep the original per-plan-owned output buffer.
    pub(super) logits_buf: Option<Box<dyn Buffer>>,
    /// Per-step host-premultiplied previous canvas logits `[cc, vocab]` — `Some` only on the
    /// Metal `sc` path (`plan_sc && !dyn_sc`). Vulkan's `dyn_sc` path reads `SeamKv::sc_ping`
    /// directly instead (no per-plan buffer, no host premultiply) — see the denoise call site.
    pub(super) sc_logits_buf: Option<Box<dyn Buffer>>,
}

/// Phase-A perf: DiffusionGemma's self-conditioning gated-MLP weights, dequantized ONCE (see
/// `SeamKv::self_cond_w`) instead of on every `diffusion_self_cond` call.
pub(super) struct SelfCondWeights {
    pub(super) pre_norm: Vec<f32>,
    pub(super) gate_w: Vec<f32>, // [nff, ne]
    pub(super) up_w: Vec<f32>,   // [nff, ne]
    pub(super) down_w: Vec<f32>, // [ne, nff]
    /// f16 copy of `token_embd` (row-major [vocab, ne], f16 bits) for the SC soft-embed: the
    /// vocab-tiled loop is L3-BANDWIDTH-bound (each canvas row re-reads the tile from L3 —
    /// FMA alone measured neutral), so halving the element width halves the traffic. Same
    /// precision as the reference, whose `sc_embT` is f16 (its CPU matmul then also converts
    /// through f16). ~1.5 GiB for gemma's 262k vocab, built once per session.
    pub(super) emb16: Vec<u16>,
}

/// DiffusionGemma self-conditioning block (Phase 2 — see docs/diffusion-gemma.md's "Self-
/// conditioning is ON by default" and the reference's `dg_canvas_embed`): given the PREVIOUS
/// step's raw canvas logits `[cc, vocab]`, returns the additive signal `sc_sig` (`[cc, ne]`) the
/// caller adds to the scaled canvas embedding before the weightless rms-norm.
/// `sc_sig[row] = sc_down @ (gelu_tanh(sc_gate @ n) * (sc_up @ n))`, `n = rmsnorm(soft,
/// sc_pre_norm)`, `soft = softmax(logits·temp_inv) @ token_embd · √n_embd`.
///
/// Runs entirely on the HOST: `soft` is a `[vocab]`-wide weighted sum of embedding rows, computed
/// directly against the CPU runner's already-dequantized `token_embd` table (a vocab-tiled
/// threaded GEMM — see the `SC_VT` comment in the body) rather than materializing the
/// reference's second on-device transposed-embedding
/// weight (`sc_embT`, ~1.4 GiB). CPU keeps this host path (this function). Phase-B moved the
/// Vulkan denoise path's SC block IN-GRAPH instead (a `sc_embT` device weight + `Op::Softmax` +
/// `Op::Linear`/`Op::GatedAct` — see the SC subgraph in `build` and `build_sc_embt` below) since
/// the host matvec here was ~85% of every Vulkan denoise step's wall time (see
/// `INFR_PROF_STAGES`'s breakdown); Phase-D widened the same in-graph path to Metal (see
/// `gpu_sc`'s call site below — Metal's Softmax/Linear ops already cover this shape and dtype).
///
/// Phase-A perf: `scw` is the ONE-TIME dequant of the four self-cond tensors (see
/// `SeamKv::self_cond_w`) — this used to re-dequantize all four on EVERY call.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(super) fn diffusion_self_cond(
    scw: &SelfCondWeights,
    c: &Config,
    sc_logits: &[f32],
    temp_inv: f32,
    cc: usize,
) -> AResult<Vec<f32>> {
    use rayon::prelude::*;
    let ne = c.n_embd;
    let vocab = c.vocab;
    debug_assert_eq!(sc_logits.len(), cc * vocab);
    let (pre_norm, gate_w, up_w, down_w) = (&scw.pre_norm, &scw.gate_w, &scw.up_w, &scw.down_w);
    let nff = gate_w.len() / ne;
    let sqrt_ne = (ne as f32).sqrt();
    let eps = c.rms_eps;
    // probs = softmax(logits * temp_inv) over the FULL vocab, all rows up front ([cc, vocab] —
    // ~1 MiB/row; materialized so the soft-embed below can be vocab-TILED across rows).
    let mut probs = vec![0f32; cc * vocab];
    probs
        .par_chunks_mut(vocab)
        .enumerate()
        .for_each(|(row, pr)| {
            let logits_row = &sc_logits[row * vocab..(row + 1) * vocab];
            let mx = logits_row
                .iter()
                .fold(f32::NEG_INFINITY, |m, &v| m.max(v * temp_inv));
            let mut denom = 0f32;
            for (p, &v) in pr.iter_mut().zip(logits_row) {
                *p = (v * temp_inv - mx).exp();
                denom += *p;
            }
            for p in pr.iter_mut() {
                *p /= denom;
            }
        });
    // soft = (probs @ token_embd) — a [ne] weighted sum over ALL vocab rows per canvas row
    // (token_embd is row-major [vocab, ne], already fully dequantized in host memory).
    //
    // TILED over vocab: the naive per-row loop streams the whole [vocab, ne] f32 table (~2 GiB for
    // gemma's 262k vocab) once per canvas row — cc=256 rows ≈ 540 GiB of DRAM traffic, which alone
    // was ~47% of every denoise step. Instead each `SC_VT`-row embedding tile (SC_VT·ne·4 B ≈
    // 16 MiB — L3-resident) is consumed by ALL cc rows while hot, so the table streams from DRAM
    // ONCE per step. The accumulation uses FMA (`mul_add`): once the tiling made this loop
    // compute-bound, the unfused mul+add pair was the ceiling — the reference (llama.cpp) runs
    // this very matmul as f16 weights with FMA accumulation, so f32+FMA is strictly MORE precise
    // than upstream. Per-(row, e) accumulation order over v is still fixed (ascending).
    let mut soft_all = vec![0f32; cc * ne];
    const SC_VT: usize = 2048;
    for t0 in (0..vocab).step_by(SC_VT) {
        let t1 = (t0 + SC_VT).min(vocab);
        soft_all
            .par_chunks_mut(ne)
            .enumerate()
            .for_each(|(row, sr)| {
                let pr = &probs[row * vocab..(row + 1) * vocab];
                for v in t0..t1 {
                    let p = pr[v];
                    if p == 0.0 {
                        continue;
                    }
                    let row_e = &scw.emb16[v * ne..v * ne + ne];
                    for (s, &e) in sr.iter_mut().zip(row_e) {
                        *s = half::f16::from_bits(e).to_f32().mul_add(p, *s);
                    }
                }
            });
    }
    drop(probs);
    // sc_pre_norm: a NORMAL (weighted) rmsnorm — unlike the canvas embedding's weightless one.
    let mut normed_all = vec![0f32; cc * ne];
    normed_all
        .par_chunks_mut(ne)
        .enumerate()
        .for_each(|(row, nr)| {
            let mut soft = soft_all[row * ne..(row + 1) * ne].to_vec();
            for s in soft.iter_mut() {
                *s *= sqrt_ne;
            }
            let ms: f32 = soft.iter().map(|&x| x * x).sum::<f32>() / ne as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for ((n, &x), &w) in nr.iter_mut().zip(soft.iter()).zip(pre_norm.iter()) {
                *n = x * inv * w;
            }
        });
    drop(soft_all);
    // Gated-GELU MLP: down(gelu_tanh(gate·normed) * (up·normed)) — WEIGHT-row-major loops, same
    // traffic argument as the soft-embed above: the per-canvas-row version streamed the full
    // gate/up/down tables (~400 MiB f32) once per row. Parallelizing over WEIGHT rows instead
    // streams each table once per step, with the [cc, ne] activations (2 MiB) L3-hot. `act` is
    // kept TRANSPOSED ([nff, cc]) so both phases read/write contiguously. Bit-identical: every
    // per-(row, f) / per-(row, e) dot accumulates in the same ascending element order.
    let mut act_t = vec![0f32; nff * cc];
    act_t.par_chunks_mut(cc).enumerate().for_each(|(f, ar)| {
        let grow = &gate_w[f * ne..f * ne + ne];
        let urow = &up_w[f * ne..f * ne + ne];
        for (row, a) in ar.iter_mut().enumerate() {
            let normed = &normed_all[row * ne..(row + 1) * ne];
            let gd: f32 = grow.iter().zip(normed).map(|(&w, &x)| w * x).sum();
            let ud: f32 = urow.iter().zip(normed).map(|(&w, &x)| w * x).sum();
            // gelu_pytorch_tanh, matching `infr_cpu::act_fn(Activation::Gelu, ..)` exactly.
            let gelu = 0.5 * gd * (1.0 + (0.797_884_6 * (gd + 0.044715 * gd * gd * gd)).tanh());
            *a = gelu * ud;
        }
    });
    drop(normed_all);
    // down phase, also weight-row-major: one [cc]-wide accumulator per output dim `e`, written
    // TRANSPOSED ([ne, cc]) so every read and write streams contiguously; the final [cc, ne]
    // un-transpose is a 2 MiB copy — noise next to the streaming reads it buys.
    let mut sig_t = vec![0f32; ne * cc];
    sig_t.par_chunks_mut(cc).enumerate().for_each(|(e, acc)| {
        let drow = &down_w[e * nff..e * nff + nff];
        for (f, &w) in drow.iter().enumerate() {
            let arow = &act_t[f * cc..(f + 1) * cc];
            for (a, &x) in acc.iter_mut().zip(arow) {
                *a = x.mul_add(w, *a);
            }
        }
    });
    let mut sig = vec![0f32; cc * ne];
    sig.par_chunks_mut(ne).enumerate().for_each(|(row, out)| {
        for (e, o) in out.iter_mut().enumerate() {
            *o = sig_t[e * cc + row];
        }
    });
    Ok(sig)
}

/// Phase-B perf: build the in-graph SC soft-embedding weight ONCE (see `SeamKv::sc_embt`) —
/// `token_embd` (already dequantized to f32 host-side, row-major `[vocab, ne]`) TRANSPOSED to f16
/// `[ne, vocab]` row-major (row `e` holds embedding dim `e` across every vocab token), matching
/// `Op::Linear`'s `weight: [out_f, in_f]` convention exactly like the reference's `sc_embT` /
/// `dg_ensure_sc_embT` — the difference being the reference dequantizes `tok_embd` FROM the GGUF
/// dtype on the host, while this runner already has it in f32 (`token_embd`), so this is a plain
/// transpose+cast. Threaded over embedding rows (each row's inner loop reads `token_embd` with a
/// `ne`-element stride — cache-unfriendly, but this runs ONCE per session).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(super) fn build_sc_embt(
    be: &dyn Backend,
    token_embd: &[f32],
    ne: usize,
    vocab: usize,
) -> AResult<std::sync::Arc<dyn Buffer>> {
    use half::f16;
    use rayon::prelude::*;
    let mut dst = vec![0u16; ne * vocab]; // f16 bits, row-major [ne, vocab]
    dst.par_chunks_mut(vocab).enumerate().for_each(|(e, row)| {
        for (v, out_v) in row.iter_mut().enumerate() {
            *out_v = f16::from_f32(token_embd[v * ne + e]).to_bits();
        }
    });
    let buf = be
        .alloc(dst.len() * 2, BufferUsage::Weights)
        .map_err(|e| anyhow!("{e}"))?;
    be.upload(buf.as_ref(), bytemuck::cast_slice(&dst))
        .map_err(|e| anyhow!("{e}"))?;
    Ok(std::sync::Arc::from(buf))
}

/// Phase-2 DiffusionGemma canvas-denoise request (see docs/diffusion-gemma.md): short-circuits
/// `generate_dense_backend` into ONE forward over the `canvas_tokens.len()` canvas rows, reusing
/// the session's already-prefilled prompt KV (rows `0..P`, `P = state.cached.len()` — the prior
/// causal prefill call's materialized prompt). Mirrors the `verify` early-return below (same
/// short-circuit style) but for the bidirectional canvas mask + decoder scalars + self-cond.
pub(crate) struct DenoiseReq<'a> {
    pub canvas_tokens: &'a [u32],
    /// Previous step's raw (pre-softmax, post-softcap) canvas logits `[C * vocab]`, for self-
    /// conditioning. `None` = SC off (`sc_use = 0`, matching the reference's step-0 gate).
    /// Perf slice 3 (Vulkan `dyn_sc`): the VALUES are only ever read when `!dyn_sc` (the length
    /// check below is scoped accordingly) — on the Vulkan ping path this is `Some` purely as a
    /// presence marker (see `EbReduced`'s doc), so it may be a placeholder slice of any length.
    pub sc_logits: Option<&'a [f32]>,
    /// Self-conditioning softmax temperature divisor (`probs = softmax(sc_logits · temp_inv)`).
    /// Unused when `sc_logits` is `None`.
    pub temp_inv: f32,
    /// Filled with `[C * vocab]` raw logits (pre-softmax, post-softcap) on success — UNLESS the
    /// GPU reducer ran this call (`reduced` comes back `Some`), in which case this is left empty
    /// (no full download — see `u`'s doc).
    pub out_logits: &'a mut Vec<f32>,
    /// Perf slice 3 (docs/diffusion-gemma.md): `Some` when the caller wants THIS step's raw logits
    /// reduced on-GPU into {argmax, entropy, sampled} instead of downloaded whole (see
    /// `Backend::eb_sample_reduce`) — `Some(u)` where `u` is `canvas_tokens.len()` host-drawn
    /// uniform `[0,1)` floats (the seeded CDF-inversion draw, `Rng::next_f32_01` per canvas row).
    /// `None` (CPU/Metal sessions) always takes the ordinary full-logits path. Even when `Some`,
    /// the backend may not support the reducer (falls back to full logits) — check `reduced`.
    pub u: Option<&'a [f32]>,
    /// This step's sampler temperature divisor (`softmax(raw * sample_temp_inv)`, i.e.
    /// `denoise_block`'s local `temp_inv` — NOT `Self::temp_inv`, which is the PREVIOUS step's
    /// self-conditioning divisor). Only meaningful when `u` is `Some`.
    pub sample_temp_inv: f32,
    /// Filled `Some` iff the GPU reducer actually ran this call (`u.is_some()` AND the backend
    /// supports it) — see `u`'s doc. `None` on entry; still `None` on return means the caller
    /// must read the (fully populated) `out_logits` instead.
    pub reduced: &'a mut Option<EbReduced>,
}

/// Perf slice 3 (docs/diffusion-gemma.md, `Backend::eb_sample_reduce`): one denoise step's sampler
/// state reduced entirely on-GPU, one entry per canvas row — the `argmax`/`entropy`/`sampled`
/// triple `diffusion.rs::denoise_block`'s host `per_pos` closure used to compute from the FULL
/// downloaded `[C, vocab]` logits. Mirrors that closure's math exactly (NOT bit-identical — see
/// `Backend::eb_sample_reduce`'s doc); `sampled[pos]` is only ever consumed by the caller for
/// positions this step's entropy bound actually ACCEPTS (see `denoise_block`'s accept loop) — the
/// renoise path never reads GPU data.
// `pub` (not `pub(crate)`): both types below flow through `DiffusionSession::denoise`'s return
// type, and that trait is `pub` (`crate::diffusion`) — a private type in a public trait's
// signature is a hard compile error, not just a lint.
pub struct EbReduced {
    pub argmax: Vec<u32>,
    pub entropy: Vec<f32>,
    pub sampled: Vec<u32>,
}

/// A [`crate::diffusion::DiffusionSession::denoise`] call's outcome — either the ordinary full
/// `[C, vocab]` raw logits (CPU/Metal always; Vulkan when the GPU reducer wasn't requested or
/// wasn't available) or this step's sampler state already reduced on-GPU (Vulkan perf slice 3).
pub enum DenoiseOutcome {
    Logits(Vec<f32>),
    Reduced(EbReduced),
}
