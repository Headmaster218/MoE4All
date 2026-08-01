//! B7 slices 3a/3b — BITWISE parity of the decode-only split-K pass 1 family (`attn_decode.comp`)
//! with the general `attn_partial` builds it specializes.
//!
//! Every member of the family is `attn_partial_bda`'s matching arm with the other arms deleted:
//! `attn_decode` is the hd=128 f16 causal arm (slice 3a, 120 → 96 VGPRs and 5120 → 3072 B LDS), and
//! slice 3b adds `-DSWA -DRING` (sliding-window layers and the ring cache they allocate) and
//! `-DDHD4=64/128` (hd 256 / 512) as further build-time specializations — all twelve measured at
//! the same 96 VGPRs / 3072 B LDS, zero spills.
//!
//! Nothing about the arithmetic changes — same key→lane mapping, same software pipeline, same PLAIN
//! `subgroupAdd` — so the output must be identical to the LAST BIT, and this file asserts exactly
//! that: `f32::to_bits()`, no tolerance. Bit-identity is what makes the slices safe to ship without
//! re-blessing a single golden.
//!
//! The failure mode being guarded is SILENT: a wrong ring row, a dropped key or a mis-derived
//! window floor produces plausible finite numbers, not a crash. So the shape table deliberately
//! includes a ring cache PAST ITS WRAP (`kv_len` several times the ring's row count), windows
//! smaller than / larger than / unaligned to the chunk, ragged last chunks and single-key chunks.
//!
//! Both production call paths are covered, because they select the kernel independently:
//!  * `attention_kv_split_at` — the STATIC push-constant path (`INFR_SEAM_NO_REPLAY=1`, and
//!    small-m callers at rows==1).
//!  * `attn_live_prologue` + `attention_kv_split_dynac_at` — the record-once REPLAY path, which is
//!    what real decode runs. A variant wired only into the static path is a measured no-op.
//!
//! The two legs are run on two backends differing ONLY in `kernels.vulkan.attn_decode`
//! (`INFR_NO_ATTN_DECODE`), i.e. through the real production selector rather than a test-only
//! entry point. Afterwards `built_kernel_names()` is asserted BOTH ways: the fast leg must have
//! built every expected family member and NO `attn_partial*` kernel at all (one shape quietly
//! falling back would otherwise hide behind another shape that took the fast path), and the
//! reference leg must be the exact mirror.
//!
//! Run: `cargo test --release -p infr-vulkan --test attn_decode_parity`
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::config::Config;
use infr_vulkan::VulkanBackend;
use std::collections::BTreeSet;
use std::sync::Arc;

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
    /// 128, 256 or 512 — each has its own build (`-DDHD4=32/64/128`).
    hd: usize,
    /// Static path: the chunk itself. Replay path: the BAKED minimum, from which the kernel's
    /// SELF_CHUNK arm re-derives `max(chunk, min(max(span/32, 64), 512))`.
    chunk: usize,
    /// Caller scale; 0.0 → the kernel's 1/√hd default (which is the shipped decode case).
    scale: f32,
    /// Sliding-window width; 0 = full causal. Nonzero selects the `-DSWA -DRING` build.
    window: usize,
    /// Cache ROW capacity when the layer is a ring (`seam`'s `window + ubatch` sizing). 0 = a
    /// full-context cache of `kv_len` rows. When nonzero and below `kv_len` the ring has WRAPPED,
    /// which is the case that makes `attn_partial`'s `j % rcap` row map observable at all.
    ring_rows: usize,
    /// Push a nonzero planar-Q8 `cap` for a FULL-CONTEXT cache. It is otherwise dead for an f16
    /// cache except that `attn_partial` derives its ring-row modulo from it — so a nonzero,
    /// full-context `cap` is what proves the causal builds' host-gate row bound is right rather
    /// than merely untested. Ignored (forced on) when `ring_rows` is set, since a real ring layer
    /// always pushes its capacity.
    cap_elems: bool,
    label: &'static str,
}

const SHAPES: &[Shape] = &[
    // ── hd 128, full causal (slice 3a's table, unchanged) ────────────────────────────────────
    // The B7 decode shape, scaled down: GQA g=8, four full chunks.
    Shape {
        kv_len: 2048,
        nh: 32,
        nkv: 4,
        hd: 128,
        chunk: 512,
        scale: 0.0,
        window: 0,
        ring_rows: 0,
        cap_elems: false,
        label: "hd128 gqa8 kv2048 c512",
    },
    // Ragged last chunk (1000 = 3*256 + 232) — exercises the un-pipelined key tail.
    Shape {
        kv_len: 1000,
        nh: 32,
        nkv: 4,
        hd: 128,
        chunk: 256,
        scale: 0.0,
        window: 0,
        ring_rows: 0,
        cap_elems: false,
        label: "hd128 gqa8 kv1000 c256 ragged",
    },
    // A chunk holding a SINGLE key (513 = 512 + 1): that workgroup's 4-key pipelined loop never
    // runs and only one of the two waves has a key at all.
    Shape {
        kv_len: 513,
        nh: 16,
        nkv: 2,
        hd: 128,
        chunk: 512,
        scale: 0.0,
        window: 0,
        ring_rows: 0,
        cap_elems: false,
        label: "hd128 gqa8 kv513 c512 single-key chunk",
    },
    // MHA (g == 1, nh == nkv) — the other end of the workgroup → (head, chunk) decomposition.
    Shape {
        kv_len: 1024,
        nh: 8,
        nkv: 8,
        hd: 128,
        chunk: 512,
        scale: 0.0,
        window: 0,
        ring_rows: 0,
        cap_elems: false,
        label: "hd128 mha kv1024 c512",
    },
    // kv_len below one chunk: fewer keys than one pipelined iteration covers (8 per workgroup).
    Shape {
        kv_len: 40,
        nh: 8,
        nkv: 2,
        hd: 128,
        chunk: 32,
        scale: 0.0,
        window: 0,
        ring_rows: 0,
        cap_elems: false,
        label: "hd128 kv40 c32 sub-chunk",
    },
    // Caller-supplied scale (gemma4's 1.0) instead of the 1/√hd default — a different branch of
    // the `pc.scale > 0` select, and the multiply that every score passes through.
    Shape {
        kv_len: 1500,
        nh: 16,
        nkv: 4,
        hd: 128,
        chunk: 256,
        scale: 1.0,
        window: 0,
        ring_rows: 0,
        cap_elems: false,
        label: "hd128 gqa4 kv1500 c256 scale=1.0",
    },
    // Nonzero `cap` (a full-context cache): `attn_partial` computes RROW(j) = j % kv_len, which is
    // the identity here — the condition the host gate checks as `pos < cap/(nkv*hd)`.
    Shape {
        kv_len: 1200,
        nh: 16,
        nkv: 4,
        hd: 128,
        chunk: 256,
        scale: 0.0,
        window: 0,
        ring_rows: 0,
        cap_elems: true,
        label: "hd128 gqa4 kv1200 c256 cap>0",
    },
    // The BENCHMARKED shapes, so no timed configuration goes unverified. `adaptive_chunk` picks
    // 256 at d8192 and 512 at d32768.
    Shape {
        kv_len: 8192,
        nh: 32,
        nkv: 4,
        hd: 128,
        chunk: 256,
        scale: 0.0,
        window: 0,
        ring_rows: 0,
        cap_elems: false,
        label: "hd128 gqa8 kv8192 c256 (benched)",
    },
    Shape {
        kv_len: 32768,
        nh: 32,
        nkv: 4,
        hd: 128,
        chunk: 512,
        scale: 0.0,
        window: 0,
        ring_rows: 0,
        cap_elems: false,
        label: "hd128 gqa8 kv32768 c512 (benched)",
    },
    // ── hd 128, sliding window (slice 3b, `-DSWA -DRING`) ────────────────────────────────────
    // THE ring case: a 1024-row ring four wraps deep. Every attended position j in
    // [kv_len-512, kv_len) maps to row j % 1024, and three quarters of the cache's rows hold data
    // that must NOT be read. Getting this wrong is silent garbage, so it is the shape the
    // deliberate-break run targets.
    Shape {
        kv_len: 4096,
        nh: 32,
        nkv: 4,
        hd: 128,
        chunk: 256,
        scale: 0.0,
        window: 512,
        ring_rows: 1024,
        cap_elems: false,
        label: "hd128 swa512 kv4096 ring1024 (4 wraps)",
    },
    // Window NOT chunk-aligned (300 vs chunk 256) and a ring that wraps: the window floor
    // `lo = kv_len - window` lands mid-chunk, so `j0 = lo + c*chunk` is unaligned to the ring too.
    Shape {
        kv_len: 2000,
        nh: 16,
        nkv: 4,
        hd: 128,
        chunk: 256,
        scale: 0.0,
        window: 300,
        ring_rows: 768,
        cap_elems: false,
        label: "hd128 swa300 kv2000 ring768 unaligned",
    },
    // Window WIDER than the cache is deep: `lo` folds to 0 and the SWA build must degenerate to
    // the causal one, on a full-context cache (no wrap).
    Shape {
        kv_len: 1000,
        nh: 16,
        nkv: 4,
        hd: 128,
        chunk: 256,
        scale: 0.0,
        window: 4096,
        ring_rows: 0,
        cap_elems: true,
        label: "hd128 swa4096 kv1000 window>kv",
    },
    // Window EXACTLY the depth: `qpos1 > window` is false, so `lo` must fold to 0 — the off-by-one
    // boundary of the window floor, on a full-context cache that still pushes a nonzero `cap`.
    Shape {
        kv_len: 512,
        nh: 16,
        nkv: 2,
        hd: 128,
        chunk: 256,
        scale: 0.0,
        window: 512,
        ring_rows: 0,
        cap_elems: true,
        label: "hd128 swa512 kv512 window==kv boundary",
    },
    // Ragged window span with a single-key last chunk: span 513 over chunk 512.
    Shape {
        kv_len: 3000,
        nh: 16,
        nkv: 4,
        hd: 128,
        chunk: 512,
        scale: 0.0,
        window: 513,
        ring_rows: 1024,
        cap_elems: false,
        label: "hd128 swa513 kv3000 ring1024 single-key chunk",
    },
    // ── hd 256 (`-DDHD4=64`) — gemma3-12b / gemma4 layers ────────────────────────────────────
    Shape {
        kv_len: 2048,
        nh: 16,
        nkv: 8,
        hd: 256,
        chunk: 512,
        scale: 0.0,
        window: 0,
        ring_rows: 0,
        cap_elems: true,
        label: "hd256 gqa2 kv2048 c512 causal",
    },
    Shape {
        kv_len: 1000,
        nh: 8,
        nkv: 4,
        hd: 256,
        chunk: 256,
        scale: 1.0,
        window: 0,
        ring_rows: 0,
        cap_elems: false,
        label: "hd256 gqa2 kv1000 c256 ragged scale=1.0",
    },
    Shape {
        kv_len: 4096,
        nh: 16,
        nkv: 8,
        hd: 256,
        chunk: 256,
        scale: 0.0,
        window: 1024,
        ring_rows: 1536,
        cap_elems: false,
        label: "hd256 swa1024 kv4096 ring1536 (wrapped)",
    },
    Shape {
        kv_len: 1500,
        nh: 8,
        nkv: 4,
        hd: 256,
        chunk: 512,
        scale: 0.0,
        window: 513,
        ring_rows: 1024,
        cap_elems: false,
        label: "hd256 swa513 kv1500 ring1024 single-key chunk",
    },
    // ── hd 512 (`-DDHD4=128`) — gemma4 / qwen3.5 layers ──────────────────────────────────────
    Shape {
        kv_len: 1024,
        nh: 8,
        nkv: 2,
        hd: 512,
        chunk: 256,
        scale: 0.0,
        window: 0,
        ring_rows: 0,
        cap_elems: true,
        label: "hd512 gqa4 kv1024 c256 causal",
    },
    Shape {
        kv_len: 700,
        nh: 8,
        nkv: 2,
        hd: 512,
        chunk: 256,
        scale: 1.0,
        window: 0,
        ring_rows: 0,
        cap_elems: false,
        label: "hd512 gqa4 kv700 c256 ragged scale=1.0",
    },
    Shape {
        kv_len: 2048,
        nh: 8,
        nkv: 2,
        hd: 512,
        chunk: 256,
        scale: 0.0,
        window: 512,
        ring_rows: 1024,
        cap_elems: false,
        label: "hd512 swa512 kv2048 ring1024 (wrapped)",
    },
    Shape {
        kv_len: 1500,
        nh: 8,
        nkv: 2,
        hd: 512,
        chunk: 512,
        scale: 0.0,
        window: 513,
        ring_rows: 1024,
        cap_elems: false,
        label: "hd512 swa513 kv1500 ring1024 single-key chunk",
    },
];

impl Shape {
    /// Window floor for the single decode row at position `kv_len-1` — the shader's
    /// `lo = (window > 0 && qpos1 > window) ? qpos1 - window : 0`, and (at rows == 1) also the
    /// adapter's chunk-grid `swa_base`.
    fn lo(&self) -> usize {
        if self.window > 0 && self.kv_len > self.window {
            self.kv_len - self.window
        } else {
            0
        }
    }
    /// Cache row capacity: a ring's declared rows, else the full context.
    fn rows_alloc(&self) -> usize {
        if self.ring_rows > 0 {
            self.ring_rows
        } else {
            self.kv_len
        }
    }
    /// The planar-Q8 `cap` push constant = total cache ELEMENTS. A ring layer always pushes it
    /// (that is where `rcap` comes from); a full-context layer only when the shape says so.
    fn cap(&self) -> usize {
        if self.ring_rows > 0 || self.cap_elems {
            self.rows_alloc() * self.nkv * self.hd
        } else {
            0
        }
    }
    /// The family member this shape must select, in both form factors.
    fn expect_kernels(&self) -> [String; 2] {
        let hd = match self.hd {
            128 => "",
            256 => "_hd256",
            512 => "_hd512",
            other => panic!("shape {}: unhandled hd {other}", self.label),
        };
        let swa = if self.window > 0 { "_swa" } else { "" };
        [
            format!("attn_decode{hd}{swa}"),
            format!("attn_decode{hd}{swa}_dynac"),
        ]
    }
}

/// One shape's two outputs — the static path and the record-once replay path — as raw f32 bits.
struct Legs {
    stat: Vec<u32>,
    replay: Vec<u32>,
}

fn run_shape(be: &VulkanBackend, s: &Shape) -> Legs {
    let (kv_len, nh, nkv, hd, chunk) = (s.kv_len, s.nh, s.nkv, s.hd, s.chunk);
    let pos = kv_len - 1; // rows == 1: the decode query is position kv_len-1
    let cache_elems = s.rows_alloc() * nkv * hd;
    let cap = s.cap();
    let lo = s.lo();

    let qb = be.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
    be.upload(qb.as_ref(), &f16_data(nh * hd, 101)).unwrap();
    let kb = be.alloc(cache_elems * 2, BufferUsage::KvCache).unwrap();
    let vb = be.alloc(cache_elems * 2, BufferUsage::KvCache).unwrap();
    be.upload(kb.as_ref(), &f16_data(cache_elems, 11)).unwrap();
    be.upload(vb.as_ref(), &f16_data(cache_elems, 23)).unwrap();
    let ka = kb.device_addr().expect("KvCache K device address");
    let va = vb.device_addr().expect("KvCache V device address");

    // Chunk grid: a sliding-window layer chunks only the union span `[lo, kv_len)`, exactly as
    // `adapter.rs`'s `swa_base`/`span` do — the shader derives the same base from pos/window, so a
    // grid over the full cache would launch chunks the kernel never maps. The replay path derives
    // its own (never smaller) chunk from the live kv_len, so its live count is <= this one and the
    // shared pm/pl/pacc stride is safe for both.
    let n_chunks = (kv_len - lo).div_ceil(chunk);
    let pm = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc = be
        .alloc(nh * n_chunks * hd * 4, BufferUsage::Activations)
        .unwrap();
    let o_bytes = nh * hd * 4;
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
        hd,
        chunk,
        n_chunks,
        s.scale,
        s.window,
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
    rec.attn_live_prologue(params.as_ref(), args.as_ref(), nh, chunk, s.window);
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
        hd,
        chunk,
        n_chunks,
        s.scale,
        s.window,
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

fn compare(want: &[u32], got: &[u32], hd: usize, what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length mismatch");
    let mut ndiff = 0usize;
    let mut first = None;
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        if w != g {
            ndiff += 1;
            first.get_or_insert((i, w, g));
        }
    }
    if let Some((i, w, g)) = first {
        panic!(
            "{what}: {ndiff}/{} outputs differ; first at head {} dim {} — attn_partial {:#010x} \
             ({}) vs attn_decode {:#010x} ({}). This kernel family is required to be BIT-identical; \
             a tolerance is not the fix.",
            want.len(),
            i / hd,
            i % hd,
            w,
            f32::from_bits(w),
            g,
            f32::from_bits(g)
        );
    }
}

#[test]
fn attn_decode_is_bit_identical_to_attn_partial() {
    // Shape-table sanity, so a typo cannot make a case silently weaker than it reads: every ring
    // shape must actually have WRAPPED, and its row count must cover the window (a ring shorter
    // than its window would be recycling live positions, which is a broken configuration rather
    // than a hard case).
    for s in SHAPES {
        if s.ring_rows > 0 {
            assert!(
                s.kv_len > s.ring_rows,
                "{}: ring_rows {} >= kv_len {} — the ring never wraps, so `j % rcap` is the \
                 identity and this shape proves nothing about the row map",
                s.label,
                s.ring_rows,
                s.kv_len
            );
            assert!(
                s.ring_rows >= s.window,
                "{}: ring_rows {} < window {} — the ring recycles rows the mask still needs",
                s.label,
                s.ring_rows,
                s.window
            );
        }
    }
    // …and every head dim must have at least one WRAPPED ring among its shapes, since the ring row
    // map is per-build code (each `-DDHD4` arm spells its own `RROW(j) * stride` indices).
    for hd in [128usize, 256, 512] {
        assert!(
            SHAPES
                .iter()
                .any(|s| s.hd == hd && s.ring_rows > 0 && s.kv_len > s.ring_rows),
            "no wrapped-ring shape at hd {hd} — that build's row map would go unverified"
        );
    }

    // Reference FIRST, then drop its backend: each leg allocates a 32768-row f16 KV cache, and
    // holding two devices' worth at once is pure waste when the outputs are plain host Vecs.
    let Some((want, ref_kernels)) = run_all(false) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (got, fast_kernels) = run_all(true).expect("second backend");

    // Assert the gate RAN, on BOTH sides. Membership alone is too weak once several shapes map to
    // the same kernel — one shape falling back would hide behind another that did not. The
    // "no attn_partial at all" clause is what makes it per-shape: any fallback anywhere in the
    // table puts an `attn_partial*` name in the fast leg's set.
    let want_fast: BTreeSet<String> = SHAPES.iter().flat_map(|s| s.expect_kernels()).collect();
    let fast: BTreeSet<&str> = fast_kernels.iter().copied().collect();
    for k in &want_fast {
        assert!(
            fast.contains(k.as_str()),
            "the fast path never dispatched {k} (built: {fast_kernels:?}) — the host gate in \
             Recorder rejected a shape, so this test proves less than it claims"
        );
    }
    let leaked: Vec<&str> = fast_kernels
        .iter()
        .copied()
        .filter(|k| k.starts_with("attn_partial"))
        .collect();
    assert!(
        leaked.is_empty(),
        "the fast leg fell back to {leaked:?} for at least one shape — every shape in this table \
         is supposed to be covered by the attn_decode family, and a shape that fell back is \
         compared against ITSELF below"
    );
    let refs: BTreeSet<&str> = ref_kernels.iter().copied().collect();
    for k in &want_fast {
        assert!(
            !refs.contains(k.as_str()),
            "INFR_NO_ATTN_DECODE did not disable the fast path: the reference leg built {k}"
        );
    }
    for k in ["attn_partial_bda", "attn_partial_dynac_bda"] {
        assert!(
            refs.contains(k),
            "the reference leg never dispatched {k} (built: {ref_kernels:?})"
        );
    }

    for (i, s) in SHAPES.iter().enumerate() {
        assert_real_signal(&want[i].stat, s.label);
        assert_real_signal(&want[i].replay, s.label);
        compare(
            &want[i].stat,
            &got[i].stat,
            s.hd,
            &format!("{} [static]", s.label),
        );
        compare(
            &want[i].replay,
            &got[i].replay,
            s.hd,
            &format!("{} [replay]", s.label),
        );
    }
    eprintln!(
        "attn_decode family == attn_partial_bda bit-for-bit across {} shapes x 2 call paths \
         ({} kernels exercised)",
        SHAPES.len(),
        want_fast.len()
    );
}
