//! B7 slice 3a — BITWISE parity of the decode-only split-K pass 1 (`attn_decode.comp`) with the
//! general `attn_partial` build it specializes.
//!
//! `attn_decode` is `attn_partial_bda`'s hd=128 f16 decode arm with every other arm (window/canvas/
//! ring/Q8/mainline/hd-256/hd-512/small-m rows) deleted, which buys 120 → 96 VGPRs and 4.75 → 2.75
//! KB of LDS. Nothing about the arithmetic changes — same key→lane mapping, same 4-key software
//! pipeline, same PLAIN `subgroupAdd` — so the output must be identical to the LAST BIT, and this
//! file asserts exactly that: `f32::to_bits()`, no tolerance. Bit-identity is what makes the slice
//! safe to ship without re-blessing a single golden.
//!
//! Both production call paths are covered, because they select the kernel independently:
//!  * `attention_kv_split_at` — the STATIC push-constant path (`INFR_SEAM_NO_REPLAY=1`, and
//!    small-m callers at rows==1).
//!  * `attn_live_prologue` + `attention_kv_split_dynac_at` — the record-once REPLAY path, which is
//!    what real decode runs. A variant wired only into the static path is a measured no-op.
//!
//! The two legs are run on two backends differing ONLY in `kernels.vulkan.attn_decode`
//! (`INFR_NO_ATTN_DECODE`), i.e. through the real production selector rather than a test-only
//! entry point — and `built_kernel_names()` is asserted afterwards, so a gate that quietly sent
//! both legs to `attn_partial` fails instead of passing vacuously.
//!
//! Run: `cargo test --release -p infr-vulkan --test attn_decode_parity`
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::config::Config;
use infr_vulkan::VulkanBackend;
use std::sync::Arc;

const HD: usize = 128;

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    }
}

/// `n` f16 elements drawn from [-1, 1). SIGNED: a non-negative K/Q makes every score large and
/// positive, which lets one key dominate the softmax and hides disagreement in the rest.
fn f16_data(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        out.extend_from_slice(&half::f16::from_f32(r.next_f32()).to_bits().to_le_bytes());
    }
    out
}

struct Shape {
    kv_len: usize,
    nh: usize,
    nkv: usize,
    /// Static path: the chunk itself. Replay path: the BAKED minimum, from which the kernel's
    /// SELF_CHUNK arm re-derives `max(chunk, min(max(kv_len/32, 64), 512))`.
    chunk: usize,
    /// Caller scale; 0.0 → the kernel's 1/√hd default (which is the shipped decode case).
    scale: f32,
    /// Planar-Q8 `cap` push constant. It is dead for an f16 cache EXCEPT that `attn_partial`
    /// derives its ring-row modulo from it (`rcap = cap / (nkv*hd)`), which `attn_decode` drops —
    /// so a nonzero, full-context `cap` is the case that proves the host gate's row bound is
    /// right rather than merely untested.
    cap_elems: bool,
    label: &'static str,
}

const SHAPES: &[Shape] = &[
    // The B7 decode shape, scaled down: GQA g=8, four full chunks.
    Shape {
        kv_len: 2048,
        nh: 32,
        nkv: 4,
        chunk: 512,
        scale: 0.0,
        cap_elems: false,
        label: "gqa8 kv2048 c512",
    },
    // Ragged last chunk (1000 = 3*256 + 232) — exercises the un-pipelined key tail.
    Shape {
        kv_len: 1000,
        nh: 32,
        nkv: 4,
        chunk: 256,
        scale: 0.0,
        cap_elems: false,
        label: "gqa8 kv1000 c256 ragged",
    },
    // A chunk holding a SINGLE key (513 = 512 + 1): that workgroup's 4-key pipelined loop never
    // runs and only one of the two waves has a key at all.
    Shape {
        kv_len: 513,
        nh: 16,
        nkv: 2,
        chunk: 512,
        scale: 0.0,
        cap_elems: false,
        label: "gqa8 kv513 c512 single-key chunk",
    },
    // MHA (g == 1, nh == nkv) — the other end of the workgroup → (head, chunk) decomposition.
    Shape {
        kv_len: 1024,
        nh: 8,
        nkv: 8,
        chunk: 512,
        scale: 0.0,
        cap_elems: false,
        label: "mha kv1024 c512",
    },
    // kv_len below one chunk: fewer keys than one pipelined iteration covers (8 per workgroup).
    Shape {
        kv_len: 40,
        nh: 8,
        nkv: 2,
        chunk: 32,
        scale: 0.0,
        cap_elems: false,
        label: "kv40 c32 sub-chunk",
    },
    // Caller-supplied scale (gemma4's 1.0) instead of the 1/√hd default — a different branch of
    // the `pc.scale > 0` select, and the multiply that every score passes through.
    Shape {
        kv_len: 1500,
        nh: 16,
        nkv: 4,
        chunk: 256,
        scale: 1.0,
        cap_elems: false,
        label: "gqa4 kv1500 c256 scale=1.0",
    },
    // Nonzero `cap` (a full-context cache): `attn_partial` computes RROW(j) = j % kv_len, which is
    // the identity here — the condition the host gate checks as `pos < cap/(nkv*hd)`.
    Shape {
        kv_len: 1200,
        nh: 16,
        nkv: 4,
        chunk: 256,
        scale: 0.0,
        cap_elems: true,
        label: "gqa4 kv1200 c256 cap>0",
    },
    // The BENCHMARKED shapes, so no timed configuration goes unverified. `adaptive_chunk` picks
    // 256 at d8192 and 512 at d32768.
    Shape {
        kv_len: 8192,
        nh: 32,
        nkv: 4,
        chunk: 256,
        scale: 0.0,
        cap_elems: false,
        label: "gqa8 kv8192 c256 (benched)",
    },
    Shape {
        kv_len: 32768,
        nh: 32,
        nkv: 4,
        chunk: 512,
        scale: 0.0,
        cap_elems: false,
        label: "gqa8 kv32768 c512 (benched)",
    },
];

/// One shape's two outputs — the static path and the record-once replay path — as raw f32 bits.
struct Legs {
    stat: Vec<u32>,
    replay: Vec<u32>,
}

fn run_shape(be: &VulkanBackend, s: &Shape) -> Legs {
    let (kv_len, nh, nkv, chunk) = (s.kv_len, s.nh, s.nkv, s.chunk);
    let pos = kv_len - 1; // rows == 1: the decode query is position kv_len-1
    let cache_elems = kv_len * nkv * HD;
    let cap = if s.cap_elems { cache_elems } else { 0 };

    let qb = be.alloc(nh * HD * 2, BufferUsage::Activations).unwrap();
    be.upload(qb.as_ref(), &f16_data(nh * HD, 101)).unwrap();
    let kb = be.alloc(cache_elems * 2, BufferUsage::KvCache).unwrap();
    let vb = be.alloc(cache_elems * 2, BufferUsage::KvCache).unwrap();
    be.upload(kb.as_ref(), &f16_data(cache_elems, 11)).unwrap();
    be.upload(vb.as_ref(), &f16_data(cache_elems, 23)).unwrap();
    let ka = kb.device_addr().expect("KvCache K device address");
    let va = vb.device_addr().expect("KvCache V device address");

    // The replay grid derives its own chunk from the live kv_len; the partial scratch must be
    // strided for the widest chunk COUNT either path can produce (the static path's `n_chunks`).
    let n_chunks = kv_len.div_ceil(chunk);
    let pm = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc = be
        .alloc(nh * n_chunks * HD * 4, BufferUsage::Activations)
        .unwrap();
    let o_bytes = nh * HD * 4;
    let bits = |b: &dyn Buffer| -> Vec<u32> {
        let mut out = vec![0u8; o_bytes];
        be.download(b, &mut out).unwrap();
        bytemuck::cast_slice::<u8, u32>(&out).to_vec()
    };

    // ── static path ──────────────────────────────────────────────────────────────────────────
    let o_stat = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.attention_kv_split_at(
        qb.as_ref(),
        kb.as_ref(),
        vb.as_ref(),
        ka,
        va,
        o_stat.as_ref(),
        pm.as_ref(),
        pl.as_ref(),
        pacc.as_ref(),
        1, // rows
        pos,
        kv_len,
        nh,
        nkv,
        HD,
        chunk,
        n_chunks,
        s.scale,
        0,     // window: full causal
        None,  // canvas_lo
        false, // k f16
        false, // v f16
        cap,
        false, // batched
        None,  // kv_ml
    );
    rec.finish().unwrap();
    let stat = bits(o_stat.as_ref());

    // ── record-once replay path ──────────────────────────────────────────────────────────────
    let params = be.alloc(8, BufferUsage::Activations).unwrap();
    be.upload(
        params.as_ref(),
        bytemuck::cast_slice(&[pos as u32, kv_len as u32]),
    )
    .unwrap();
    let args = be.alloc(16, BufferUsage::Activations).unwrap();
    let o_rep = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.attn_live_prologue(params.as_ref(), args.as_ref(), nh, chunk, 0);
    rec.attention_kv_split_dynac_at(
        qb.as_ref(),
        kb.as_ref(),
        vb.as_ref(),
        ka,
        va,
        o_rep.as_ref(),
        pm.as_ref(),
        pl.as_ref(),
        pacc.as_ref(),
        params.as_ref(),
        args.as_ref(),
        nh,
        nkv,
        HD,
        chunk,
        n_chunks,
        s.scale,
        0,     // window: full causal
        false, // q8
        cap,
    );
    rec.finish().unwrap();
    let replay = bits(o_rep.as_ref());
    Legs { stat, replay }
}

/// A backend whose config differs from the default in nothing but the fast-path knob, so the two
/// legs go through the production selector rather than a test-only entry point.
fn backend(attn_decode: bool) -> Option<VulkanBackend> {
    let mut cfg = Config::default();
    cfg.kernels.vulkan.attn_decode = attn_decode;
    VulkanBackend::new_with(Arc::new(cfg)).ok()
}

fn run_all(attn_decode: bool) -> Option<(Vec<Legs>, Vec<&'static str>)> {
    let be = backend(attn_decode)?;
    let out = SHAPES.iter().map(|s| run_shape(&be, s)).collect();
    Some((out, be.built_kernel_names()))
}

/// Non-vacuity: the reference must be finite and mostly non-zero, or "identical" means nothing.
fn assert_real_signal(bits: &[u32], what: &str) {
    let v: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "{what}: reference output has non-finite values"
    );
    let nz = v.iter().filter(|x| x.abs() > 1e-6).count();
    assert!(
        nz * 4 > v.len() * 3,
        "{what}: reference output is mostly zero ({nz}/{}) — the compare would be vacuous",
        v.len()
    );
}

fn compare(want: &[u32], got: &[u32], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length mismatch");
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        assert_eq!(
            w,
            g,
            "{what}: head {} dim {} differs — attn_partial {:#010x} ({}) vs attn_decode {:#010x} \
             ({}). This kernel is required to be BIT-identical; a tolerance is not the fix.",
            i / HD,
            i % HD,
            w,
            f32::from_bits(w),
            g,
            f32::from_bits(g)
        );
    }
}

#[test]
fn attn_decode_is_bit_identical_to_attn_partial() {
    // Reference FIRST, then drop its backend: each leg allocates a 32768-row f16 KV cache, and
    // holding two devices' worth at once is pure waste when the outputs are plain host Vecs.
    let Some((want, ref_kernels)) = run_all(false) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (got, fast_kernels) = run_all(true).expect("second backend");

    // Assert the gate RAN — otherwise both legs are `attn_partial` and every compare below is
    // trivially true.
    for k in ["attn_decode", "attn_decode_dynac"] {
        assert!(
            fast_kernels.contains(&k),
            "the fast path never dispatched {k} (built: {fast_kernels:?}) — the host gate in \
             Recorder rejected the shape, so this test proves nothing"
        );
        assert!(
            !ref_kernels.contains(&k),
            "INFR_NO_ATTN_DECODE did not disable the fast path: the reference leg built {k}"
        );
    }
    for k in ["attn_partial_bda", "attn_partial_dynac_bda"] {
        assert!(
            ref_kernels.contains(&k),
            "the reference leg never dispatched {k} (built: {ref_kernels:?})"
        );
    }

    for (i, s) in SHAPES.iter().enumerate() {
        assert_real_signal(&want[i].stat, s.label);
        assert_real_signal(&want[i].replay, s.label);
        compare(
            &want[i].stat,
            &got[i].stat,
            &format!("{} [static]", s.label),
        );
        compare(
            &want[i].replay,
            &got[i].replay,
            &format!("{} [replay]", s.label),
        );
    }
    eprintln!(
        "attn_decode == attn_partial_bda bit-for-bit across {} shapes x 2 call paths",
        SHAPES.len()
    );
}
