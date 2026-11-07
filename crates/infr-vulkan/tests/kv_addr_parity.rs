//! Proves the scalar decode attention kernel (`attention_kv`) reads its K/V cache correctly by
//! 64-bit device address (`kv_addr.glsl`) — the KV-side mirror of `weight_addr_parity.rs`. The
//! `-DKV_BDA` twin (`attention_kv_at`) must compute BIT-IDENTICALLY to the bound-SSBO build
//! (`attention_kv`): the two differ ONLY in where K/V words come from (a bound binding vs a
//! `k_addr`/`v_addr` pointer read), the softmax/accumulation math is the same source, so anything
//! short of bitwise equality is a mis-addressing bug, not a tolerance question.
//!
//! Two assertions per case:
//!  * BOUND-VS-POINTER: `attention_kv` (K/V bound at slots 1/2) vs `attention_kv_at` (K/V at their
//!    own device address, arena offset 0) — identical bits.
//!  * OFFSET-INVARIANCE (the load-bearing one): the SAME K/V bytes parked behind a garbage non-zero
//!    prefix inside a `KvCache` buffer, with `k_addr`/`v_addr` = base+prefix. A twin that ignored
//!    the offset (or mishandled the 64-bit add's non-zero low bits) would read the garbage prefix
//!    and diverge — this is what integration's per-tensor arena offsets need to work.
//!
//! Covers f16 K/V and all three planar-Q8 combos (K-only, V-only, both). Q/K/V bytes are drawn so
//! every f16 decodes finite and non-degenerate (high byte < 0x40 → sign+top-exp bit clear, never
//! NaN/Inf — a NaN output would make a bitwise compare pass vacuously, hiding the mis-addressing
//! this test exists to catch); Q8 codes/scales use the same < 0x40 range.
//!
//! Run: `cargo test -p infr-vulkan --test kv_addr_parity -- --ignored --nocapture`
//! ISA probe: `RADV_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true cargo test -p infr-vulkan \
//!   --test kv_addr_parity kv_isa_probe -- --ignored --nocapture 2> isa.txt` (move the pipeline
//!   cache aside first).
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_vulkan::{Recorder, VulkanBackend};

/// Pseudo-random bytes in `0x00..=0x3F` — Q8 codes (small +ints) and scales (finite +f16).
fn synth_bytes(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7;
            (h % 0x40) as u8
        })
        .collect()
}

/// `n` f16 elements whose bits are masked to `0x3FFF`: sign + top exponent bit clear → every value
/// is finite, non-negative, < 2.0. Never NaN/Inf, so a bitwise output compare is never vacuous.
fn synth_f16(n: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let h = ((i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 5) as u16;
        out.extend_from_slice(&(h & 0x3FFF).to_le_bytes());
    }
    out
}

fn bits(v: &[u8]) -> Vec<u32> {
    bytemuck::cast_slice::<u8, u32>(v).to_vec()
}

#[derive(Clone, Copy)]
struct Shape {
    q_len: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
}

/// Number of f16 elements in one KV-cache side for a full-context (rcap=0) run at pos_offset 0:
/// query row `q_len-1` attends key `q_len-1`, so the cache spans `q_len` rows of `nkv*hd`.
fn kv_elems(s: Shape) -> usize {
    s.q_len * s.nkv * s.hd
}

/// Bytes of one KV side: f16 = 2/elem; planar Q8_0 = codes[cap] (1 byte) + scales[cap/32] (f16).
fn side_bytes(s: Shape, q8: bool) -> usize {
    let cap = kv_elems(s);
    if q8 {
        assert_eq!(
            cap % 32,
            0,
            "q8 cap must be a whole number of 32-elem blocks"
        );
        cap + (cap / 32) * 2
    } else {
        cap * 2
    }
}

/// Synth bytes for one KV side (differ per side via `seed` so a K/V swap would show up).
fn side_data(s: Shape, q8: bool, seed: usize) -> Vec<u8> {
    if q8 {
        synth_bytes(side_bytes(s, q8), seed)
    } else {
        synth_f16(kv_elems(s), seed)
    }
}

/// 256-byte garbage prefix (aligned for f16 & u32-word reads) filled with DIFFERENT bytes, so a
/// kernel that reads from the buffer base instead of base+off gets visibly wrong data.
const OFF: usize = 256;

struct Legs {
    bound: Vec<u32>,
    at0: Vec<u32>,
    atoff: Vec<u32>,
}

/// Runs all three legs (bound, pointer@0, pointer@nonzero-offset) for one shape/dtype combo.
fn run(be: &VulkanBackend, s: Shape, k_q8: bool, v_q8: bool) -> Legs {
    let (q_len, nh, nkv, hd) = (s.q_len, s.nh, s.nkv, s.hd);
    let cap = if k_q8 || v_q8 { kv_elems(s) } else { 0 };
    let scale = 0.0f32; // default 1/sqrt(hd)
    let window = 0usize; // full causal
    let pos = 0usize;

    let q = synth_f16(q_len * nh * hd, 101);
    let q_buf = be.alloc(q.len(), BufferUsage::Activations).unwrap();
    be.upload(q_buf.as_ref(), &q).unwrap();
    let o_bytes = q_len * nh * hd * 4;

    let kd = side_data(s, k_q8, 11);
    let vd = side_data(s, v_q8, 23);

    // ── Leg A: bound SSBOs (K/V at slots 1/2) ────────────────────────────────────────────────
    let kbuf = be.alloc(kd.len(), BufferUsage::KvCache).unwrap();
    let vbuf = be.alloc(vd.len(), BufferUsage::KvCache).unwrap();
    be.upload(kbuf.as_ref(), &kd).unwrap();
    be.upload(vbuf.as_ref(), &vd).unwrap();
    let o_buf = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.attention_kv(
        q_buf.as_ref(),
        kbuf.as_ref(),
        vbuf.as_ref(),
        o_buf.as_ref(),
        q_len,
        q_len,
        nh,
        nkv,
        hd,
        pos,
        window,
        scale,
        k_q8,
        v_q8,
        cap,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; o_bytes];
    be.download(o_buf.as_ref(), &mut out).unwrap();
    let bound = bits(&out);

    // ── Leg B: pointer read, K/V at arena offset 0 ───────────────────────────────────────────
    let ka0 = kbuf
        .device_addr()
        .expect("KvCache K must expose a device address");
    let va0 = vbuf
        .device_addr()
        .expect("KvCache V must expose a device address");
    let rec = be.recorder().unwrap();
    rec.attention_kv_at(
        q_buf.as_ref(),
        kbuf.as_ref(),
        vbuf.as_ref(),
        ka0,
        va0,
        o_buf.as_ref(),
        q_len,
        q_len,
        nh,
        nkv,
        hd,
        pos,
        window,
        scale,
        k_q8,
        v_q8,
        cap,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; o_bytes];
    be.download(o_buf.as_ref(), &mut out).unwrap();
    let at0 = bits(&out);

    // ── Leg C: pointer read, SAME bytes parked behind a garbage prefix in a KvCache buffer ────
    let mut kback = synth_bytes(OFF, 0xBAD);
    kback.extend_from_slice(&kd);
    let mut vback = synth_bytes(OFF, 0xBEEF);
    vback.extend_from_slice(&vd);
    let kbuf2 = be.alloc(kback.len(), BufferUsage::KvCache).unwrap();
    let vbuf2 = be.alloc(vback.len(), BufferUsage::KvCache).unwrap();
    be.upload(kbuf2.as_ref(), &kback).unwrap();
    be.upload(vbuf2.as_ref(), &vback).unwrap();
    let ka = kbuf2.device_addr().unwrap() + OFF as u64;
    let va = vbuf2.device_addr().unwrap() + OFF as u64;
    let rec = be.recorder().unwrap();
    rec.attention_kv_at(
        q_buf.as_ref(),
        kbuf2.as_ref(),
        vbuf2.as_ref(),
        ka,
        va,
        o_buf.as_ref(),
        q_len,
        q_len,
        nh,
        nkv,
        hd,
        pos,
        window,
        scale,
        k_q8,
        v_q8,
        cap,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; o_bytes];
    be.download(o_buf.as_ref(), &mut out).unwrap();
    let atoff = bits(&out);

    Legs { bound, at0, atoff }
}

fn assert_legs(name: &str, l: &Legs) {
    assert!(
        l.bound.iter().any(|&b| b != 0),
        "{name}: bound output is all zeros — the case is not exercising the kernel"
    );
    for (i, (&b, &p)) in l.bound.iter().zip(l.at0.iter()).enumerate() {
        assert_eq!(
            b, p,
            "{name}: pointer@0 differs from bound at out {i}: {} vs {} (bits {b:#010x} vs {p:#010x})",
            f32::from_bits(b),
            f32::from_bits(p)
        );
    }
    for (i, (&b, &p)) in l.bound.iter().zip(l.atoff.iter()).enumerate() {
        assert_eq!(
            b,
            p,
            "{name}: pointer@nonzero-offset differs from bound at out {i}: {} vs {} — the twin is \
             ignoring its K/V base offset, which breaks every KV tensor but one at a shared base",
            f32::from_bits(b),
            f32::from_bits(p)
        );
    }
}

const SHAPES: [Shape; 4] = [
    Shape {
        q_len: 4,
        nh: 8,
        nkv: 2,
        hd: 64,
    },
    Shape {
        q_len: 1,
        nh: 4,
        nkv: 4,
        hd: 128,
    },
    Shape {
        q_len: 3,
        nh: 8,
        nkv: 8,
        hd: 64,
    },
    Shape {
        q_len: 6,
        nh: 8,
        nkv: 1,
        hd: 64,
    },
];

#[test]
#[ignore = "requires a Vulkan GPU"]
fn attention_kv_bda_matches_bound_f16() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    for s in SHAPES {
        let l = run(&be, s, false, false);
        let name = format!(
            "f16 q_len={} nh={} nkv={} hd={}",
            s.q_len, s.nh, s.nkv, s.hd
        );
        assert_legs(&name, &l);
        println!("ok: {name}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn attention_kv_bda_matches_bound_q8() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // K-only, V-only, and both — each Q8 side reads its own k_addr/v_addr through kv_word().
    for (k_q8, v_q8) in [(true, false), (false, true), (true, true)] {
        for s in SHAPES {
            let l = run(&be, s, k_q8, v_q8);
            let name = format!(
                "q8(k={k_q8},v={v_q8}) q_len={} nh={} nkv={} hd={}",
                s.q_len, s.nh, s.nkv, s.hd
            );
            assert_legs(&name, &l);
            println!("ok: {name}");
        }
    }
}

/// ISA-dump vehicle (mirrors `warp_gemm_parity::warp_ag_isa_probe`): dispatches ONLY the `-DKV_BDA`
/// pointer build so `RADV_DEBUG=shaders` emits its ISA. Confirm the inner K/V loads use a
/// scalar/saddr base + 32-bit offset (`global_load_*` with an s[]/saddr base), NOT a per-load
/// `v_add_co_u32`/`v_addc` 64-bit address materialization — the shape that measured the 2.4x
/// regression on the weight side.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn kv_isa_probe() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let s = Shape {
        q_len: 8,
        nh: 8,
        nkv: 2,
        hd: 128,
    };
    let l = run(&be, s, false, false);
    assert_legs("isa-probe f16", &l);
    println!("ok: kv_isa_probe (dispatched attention_kv_bda)");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// #74 slice 2 — flash-decoding split-K (`attn_partial`) bound-vs-pointer parity.
//
// The split-K partial pass (`attention_kv_split`) reads K/V as `f16vec4` (or planar-Q8 words). The
// `-DKV_BDA` twin (`attention_kv_split_at`) must read BIT-IDENTICALLY through `k_addr`/`v_addr`: the
// f16 vec4 is a single `KV2` b64 read (two u32 words unpacked back to the vec4), Q8 stays scalar
// `kv_word`. Same three legs as the scalar test — bound, pointer@0, pointer@nonzero-offset — plus a
// RING-WRAP case (a cache shorter than kv_len so `RROW(j)=j%rcap` recycles rows, exercising the
// wrapped-index device-address math) and a rows-batched (`attn_partial_mrows_c256`) case.

/// `n` planar-Q8 bytes: codes[n] + scales[n/32] (f16). `n` = cache elements (one side).
fn q8_bytes(elems: usize) -> usize {
    assert_eq!(
        elems % 32,
        0,
        "q8 cache must be a whole number of 32-elem blocks"
    );
    elems + (elems / 32) * 2
}

/// Synth bytes for one KV side holding `elems` cache elements (f16 or planar-Q8).
fn side_data_n(elems: usize, q8: bool, seed: usize) -> Vec<u8> {
    if q8 {
        synth_bytes(q8_bytes(elems), seed)
    } else {
        synth_f16(elems, seed)
    }
}

struct SplitCase {
    kv_len: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    rows: usize,
    /// Rows physically present in the cache buffer. `< kv_len` ⇒ a ring cache: keys wrap via
    /// `RROW(j)=j%cap_rows`, and `cap` (the push) is set to the ring's element count.
    cap_rows: usize,
    batched: bool,
}

/// Runs the three addressing legs for one split-K case and returns the combined `o` outputs.
fn run_split(be: &VulkanBackend, c: &SplitCase, k_q8: bool, v_q8: bool) -> Legs {
    let SplitCase {
        kv_len,
        nh,
        nkv,
        hd,
        rows,
        cap_rows,
        batched,
    } = *c;
    let ring = cap_rows < kv_len;
    let cache_elems = cap_rows * nkv * hd;
    // `cap` push: planar-Q8 needs the scales-region base (= total elements) always; f16 sets it only
    // for a ring cache (the ring capacity), 0 for a full-context cache (identity `RROW`).
    let cap = if k_q8 || v_q8 || ring { cache_elems } else { 0 };
    let scale = 0.0f32; // default 1/sqrt(hd)
    let window = 0usize;
    let pos = kv_len - rows; // decode suffix: row i attends up to pos+i (< kv_len)
                             // chunk/n_chunks mirror the adapter: batched clamps to 256, else the ~32-chunk decode policy;
                             // both cases below choose kv_len so n_chunks > 1.
    let chunk = if batched {
        256
    } else {
        (kv_len / 32).clamp(64, 512)
    };
    let n_chunks = kv_len.div_ceil(chunk);
    assert!(
        n_chunks > 1,
        "case must split into >1 chunk to exercise the grid"
    );

    let q = synth_f16(rows * nh * hd, 101);
    let q_buf = be.alloc(q.len(), BufferUsage::Activations).unwrap();
    be.upload(q_buf.as_ref(), &q).unwrap();
    let o_bytes = rows * nh * hd * 4;

    let kd = side_data_n(cache_elems, k_q8, 11);
    let vd = side_data_n(cache_elems, v_q8, 23);

    // Scratch (shared across the three legs — fully written each dispatch before the combine reads).
    let pm = be
        .alloc(rows * nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl = be
        .alloc(rows * nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc = be
        .alloc(rows * nh * n_chunks * hd * 4, BufferUsage::Activations)
        .unwrap();

    let one_leg = |k_addr: Option<(u64, u64)>,
                   kb: &dyn infr_core::backend::Buffer,
                   vb: &dyn infr_core::backend::Buffer|
     -> Vec<u32> {
        let o_buf = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        match k_addr {
            Some((ka, va)) => rec.attention_kv_split_at(
                q_buf.as_ref(),
                kb,
                vb,
                ka,
                va,
                o_buf.as_ref(),
                pm.as_ref(),
                pl.as_ref(),
                pacc.as_ref(),
                rows,
                pos,
                kv_len,
                nh,
                nkv,
                hd,
                chunk,
                n_chunks,
                scale,
                window,
                None,
                k_q8,
                v_q8,
                cap,
                batched,
                None,
            ),
            None => rec.attention_kv_split(
                q_buf.as_ref(),
                kb,
                vb,
                o_buf.as_ref(),
                pm.as_ref(),
                pl.as_ref(),
                pacc.as_ref(),
                rows,
                pos,
                kv_len,
                nh,
                nkv,
                hd,
                chunk,
                n_chunks,
                scale,
                window,
                None,
                k_q8,
                v_q8,
                cap,
                batched,
            ),
        }
        rec.finish().unwrap();
        let mut out = vec![0u8; o_bytes];
        be.download(o_buf.as_ref(), &mut out).unwrap();
        bits(&out)
    };

    // ── Leg A: bound SSBOs ────────────────────────────────────────────────────────────────────
    let kbuf = be.alloc(kd.len(), BufferUsage::KvCache).unwrap();
    let vbuf = be.alloc(vd.len(), BufferUsage::KvCache).unwrap();
    be.upload(kbuf.as_ref(), &kd).unwrap();
    be.upload(vbuf.as_ref(), &vd).unwrap();
    let bound = one_leg(None, kbuf.as_ref(), vbuf.as_ref());

    // ── Leg B: pointer read at arena offset 0 ─────────────────────────────────────────────────
    let ka0 = kbuf
        .device_addr()
        .expect("KvCache K must expose a device address");
    let va0 = vbuf
        .device_addr()
        .expect("KvCache V must expose a device address");
    let at0 = one_leg(Some((ka0, va0)), kbuf.as_ref(), vbuf.as_ref());

    // ── Leg C: pointer read, same bytes behind a garbage prefix ───────────────────────────────
    let mut kback = synth_bytes(OFF, 0xBAD);
    kback.extend_from_slice(&kd);
    let mut vback = synth_bytes(OFF, 0xBEEF);
    vback.extend_from_slice(&vd);
    let kbuf2 = be.alloc(kback.len(), BufferUsage::KvCache).unwrap();
    let vbuf2 = be.alloc(vback.len(), BufferUsage::KvCache).unwrap();
    be.upload(kbuf2.as_ref(), &kback).unwrap();
    be.upload(vbuf2.as_ref(), &vback).unwrap();
    let ka = kbuf2.device_addr().unwrap() + OFF as u64;
    let va = vbuf2.device_addr().unwrap() + OFF as u64;
    let atoff = one_leg(Some((ka, va)), kbuf2.as_ref(), vbuf2.as_ref());

    Legs { bound, at0, atoff }
}

/// f16 split-K: decode (rows=1) at hd=128 (the b64 fast path) and hd=64, full-context + ring-wrap.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn attn_partial_bda_matches_bound_f16() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let cases = [
        // hd=128 decode, full context (hd4==32 4-key fast path).
        SplitCase {
            kv_len: 200,
            nh: 8,
            nkv: 2,
            hd: 128,
            rows: 1,
            cap_rows: 200,
            batched: false,
        },
        // hd=64 decode, full context (general per-key loop + hd4<=32 V).
        SplitCase {
            kv_len: 300,
            nh: 8,
            nkv: 8,
            hd: 64,
            rows: 1,
            cap_rows: 300,
            batched: false,
        },
        // RING-WRAP: cache holds 96 rows, 200 keys attended ⇒ RROW wraps (window=0, so bound and
        // BDA both recycle rows identically — the point is the wrapped device-address math matches).
        SplitCase {
            kv_len: 200,
            nh: 4,
            nkv: 4,
            hd: 128,
            rows: 1,
            cap_rows: 96,
            batched: false,
        },
    ];
    for c in &cases {
        let l = run_split(&be, c, false, false);
        let name = format!(
            "f16 split kv={} nh={} nkv={} hd={} rows={} cap_rows={}",
            c.kv_len, c.nh, c.nkv, c.hd, c.rows, c.cap_rows
        );
        assert_legs(&name, &l);
        println!("ok: {name}");
    }
}

/// Planar-Q8 split-K (K-only, V-only, both) — each side reads its own k_addr/v_addr via kv_word.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn attn_partial_bda_matches_bound_q8() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // hd=128, full-context (cap_rows == kv_len ⇒ RROW identity); Q8 is decode-only (rows==1).
    let c = SplitCase {
        kv_len: 256,
        nh: 8,
        nkv: 2,
        hd: 128,
        rows: 1,
        cap_rows: 256,
        batched: false,
    };
    for (k_q8, v_q8) in [(true, false), (false, true), (true, true)] {
        let l = run_split(&be, &c, k_q8, v_q8);
        let name = format!("q8(k={k_q8},v={v_q8}) split kv={} hd={}", c.kv_len, c.hd);
        assert_legs(&name, &l);
        println!("ok: {name}");
    }
}

// ── Lever 1 (kv-decode-perf-levers #1): mainline low-bit KV decode reads the compact GGUF block
// format INLINE in `attn_partial` (`-DKMAINLINE -DVMAINLINE -DFMT_* -DKV_BDA`), instead of the
// dequant→f16 PREPASS (`dequant_kv_f16` → f16 scratch → f16 `attention_kv_split`). The inline build
// rounds each decoded element to f16 (`dqh`) to reproduce the scratch EXACTLY, so the two paths must
// be BYTE-IDENTICAL — anything else is a decode/addressing bug, not a tolerance question. Two checks:
//   * INLINE-VS-PREPASS: `attention_kv_split_at(kv_ml=Some(dt))` reading the quant cache directly vs
//     the prepass (dequant to f16 scratch, then f16 `attention_kv_split`) — identical output bits.
//   * OFFSET-INVARIANCE (load-bearing): the SAME quant cache parked behind a garbage byte prefix,
//     `k_addr`/`v_addr` = base+prefix — a twin dropping its base offset would read the prefix.

/// Bytes for one mainline-quant KV side holding `elems` cache elements (`block` bytes per 32 elems),
/// crafted so EVERY dequantized value is exactly an integer (hence exactly f16-representable): the
/// per-block scale `d` is forced to 1.0 and any min `m` to 0.0, while the quant codes (nibbles / qh
/// high bits) are varied by `synth`. Then `dq()` yields `(code - bias)` — an integer in a small range,
/// exact in f16 — so the prepass's `float16_t(dq)` store-round is the IDENTITY and the inline
/// (f16-rounded) decode is BIT-IDENTICAL to it. This still exercises the whole inline read/decode path
/// (block addressing, nibble+qh extraction, accumulation); only the non-exact scale-multiply rounding
/// — where in-register f16 and memory f16 legitimately differ by a ULP on this driver — is factored out.
fn mainline_side_bytes(elems: usize, dt: infr_core::DType, block: usize, seed: usize) -> Vec<u8> {
    use infr_core::DType::*;
    assert_eq!(elems % 32, 0, "mainline cache must be whole 32-elem blocks");
    let mut b = synth_bytes((elems / 32) * block, seed);
    let one = 0x3C00u16.to_le_bytes(); // f16 1.0
    let zero = 0u16.to_le_bytes();
    for blk in b.chunks_mut(block) {
        blk[0..2].copy_from_slice(&one); // d = 1.0 (first field of every mainline block)
        if matches!(dt, Q4_1 | Q5_1) {
            blk[2..4].copy_from_slice(&zero); // m = 0.0 (Q4_1/Q5_1 carry a min after d)
        }
    }
    b
}

/// Runs one mainline decode case (rows==1, full context) three ways and asserts byte-identity:
/// prepass (dequant→f16 + f16 split), inline@0, inline@nonzero-offset.
fn run_mainline(
    be: &VulkanBackend,
    dt: infr_core::DType,
    block: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    kv_len: usize,
) {
    let rows = 1usize;
    let pos = kv_len - rows;
    let scale = 0.0f32; // default 1/sqrt(hd)
    let window = 0usize;
    let cap = 0usize; // full-context (RROW identity); mainline ignores the planar-Q8 scales base
    let chunk = (kv_len / 32).clamp(64, 512);
    let n_chunks = kv_len.div_ceil(chunk);
    assert!(
        n_chunks > 1,
        "case must split into >1 chunk to exercise the grid"
    );
    let elems = kv_len * nkv * hd; // full-context cache
    let ne = elems;

    let q = synth_f16(rows * nh * hd, 101);
    let q_buf = be.alloc(q.len(), BufferUsage::Activations).unwrap();
    be.upload(q_buf.as_ref(), &q).unwrap();
    let o_bytes = rows * nh * hd * 4;

    let kd = mainline_side_bytes(elems, dt, block, 11);
    let vd = mainline_side_bytes(elems, dt, block, 23);

    let pm = be
        .alloc(rows * nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl = be
        .alloc(rows * nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc = be
        .alloc(rows * nh * n_chunks * hd * 4, BufferUsage::Activations)
        .unwrap();

    // ── PREPASS reference: the exact production path is dequant→f16 scratch→f16 attention. To keep
    // the reference trustworthy and DETERMINISTIC, materialize the f16 scratch as a CONCRETE uploaded
    // cache: GPU-dequant (its own submit), download the f16 bytes, re-upload as a fresh f16 KvCache,
    // then run f16 split attention on it in a clean submit. This severs the cross-recorder
    // dequant→attention scratch hazard (which left a raw f16-scratch read flaky run-to-run) while
    // using the SAME `dequant_kv_f16` bytes and the SAME `attn_partial` machinery the inline path uses.
    let kq = be.alloc(kd.len(), BufferUsage::KvCache).unwrap();
    let vq = be.alloc(vd.len(), BufferUsage::KvCache).unwrap();
    be.upload(kq.as_ref(), &kd).unwrap();
    be.upload(vq.as_ref(), &vd).unwrap();
    let kf16 = be.alloc(ne * 2, BufferUsage::Activations).unwrap();
    let vf16 = be.alloc(ne * 2, BufferUsage::Activations).unwrap();
    {
        let rec = be.recorder().unwrap();
        rec.dequant_kv_f16(dt, kq.as_ref(), kf16.as_ref(), ne);
        rec.dequant_kv_f16(dt, vq.as_ref(), vf16.as_ref(), ne);
        rec.finish().unwrap();
    }
    let mut kf16_bytes = vec![0u8; ne * 2];
    let mut vf16_bytes = vec![0u8; ne * 2];
    be.download(kf16.as_ref(), &mut kf16_bytes).unwrap();
    be.download(vf16.as_ref(), &mut vf16_bytes).unwrap();
    let kf16c = be.alloc(ne * 2, BufferUsage::KvCache).unwrap();
    let vf16c = be.alloc(ne * 2, BufferUsage::KvCache).unwrap();
    be.upload(kf16c.as_ref(), &kf16_bytes).unwrap();
    be.upload(vf16c.as_ref(), &vf16_bytes).unwrap();
    let o_ref = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
    {
        let rec = be.recorder().unwrap();
        rec.attention_kv_split(
            q_buf.as_ref(),
            kf16c.as_ref(),
            vf16c.as_ref(),
            o_ref.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            rows,
            pos,
            kv_len,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            scale,
            window,
            None,
            false,
            false,
            0,
            false,
        );
        rec.finish().unwrap();
    }
    let mut ref_out = vec![0u8; o_bytes];
    be.download(o_ref.as_ref(), &mut ref_out).unwrap();
    let ref_bits = bits(&ref_out);

    // ── INLINE leg: read the quant cache directly (no scratch), at base offset `prefix`.
    let inline_leg = |prefix: usize| -> Vec<u32> {
        let mut kback = synth_bytes(prefix, 0xBAD);
        kback.extend_from_slice(&kd);
        let mut vback = synth_bytes(prefix, 0xBEEF);
        vback.extend_from_slice(&vd);
        let kbuf = be.alloc(kback.len(), BufferUsage::KvCache).unwrap();
        let vbuf = be.alloc(vback.len(), BufferUsage::KvCache).unwrap();
        be.upload(kbuf.as_ref(), &kback).unwrap();
        be.upload(vbuf.as_ref(), &vback).unwrap();
        let ka = kbuf.device_addr().unwrap() + prefix as u64;
        let va = vbuf.device_addr().unwrap() + prefix as u64;
        let o_buf = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_kv_split_at(
            q_buf.as_ref(),
            kbuf.as_ref(),
            vbuf.as_ref(),
            ka,
            va,
            o_buf.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            rows,
            pos,
            kv_len,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            scale,
            window,
            None,
            false,
            false,
            cap,
            false,
            Some(dt),
        );
        rec.finish().unwrap();
        let mut out = vec![0u8; o_bytes];
        be.download(o_buf.as_ref(), &mut out).unwrap();
        bits(&out)
    };
    let inline0 = inline_leg(0);
    let inlineoff = inline_leg(OFF);

    let name = format!("mainline {dt:?} kv={kv_len} nh={nh} nkv={nkv} hd={hd}");
    assert!(
        ref_bits.iter().any(|&b| b != 0),
        "{name}: prepass reference is all zeros — not exercising the kernel"
    );
    for (i, (&r, &p)) in ref_bits.iter().zip(inline0.iter()).enumerate() {
        assert_eq!(r, p, "{name}: INLINE@0 differs from prepass at out {i}: {} vs {} (bits {r:#010x} vs {p:#010x})",
            f32::from_bits(r), f32::from_bits(p));
    }
    for (i, (&r, &p)) in ref_bits.iter().zip(inlineoff.iter()).enumerate() {
        assert_eq!(
            r, p,
            "{name}: INLINE@nonzero-offset differs from prepass at out {i} — the inline twin \
            is ignoring its K/V base offset",
        );
    }
    println!("ok: {name}");
}

/// Mainline inline-vs-prepass decode parity for q4_0 and q5_0, hd=128 (b64 fast path) and hd=64
/// (general per-key loop). K and V both route through the inline `dq()` (they share the cache dtype).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn attn_partial_mainline_matches_prepass() {
    use infr_core::DType::*;
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    for (dt, block) in [(Q4_0, 18usize), (Q5_0, 22usize)] {
        // hd=128 decode, full context (hd4==32 4-key fast path).
        run_mainline(&be, dt, block, 8, 2, 128, 256);
        // hd=64 decode (general per-key QK loop + hd4<=32 V path).
        run_mainline(&be, dt, block, 8, 8, 64, 288);
    }
}

/// Rows-batched split-K (`attn_partial_mrows_c256`): 4 query rows, one K/V stream, chunk=256.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn attn_partial_mrows_bda_matches_bound() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // rows>=2, hd<=128, chunk<=256, non-ring, f16 — the recorder's batched debug_assert.
    let c = SplitCase {
        kv_len: 400,
        nh: 8,
        nkv: 2,
        hd: 128,
        rows: 4,
        cap_rows: 400,
        batched: true,
    };
    let l = run_split(&be, &c, false, false);
    let name = format!(
        "mrows split kv={} nh={} hd={} rows={}",
        c.kv_len, c.nh, c.hd, c.rows
    );
    assert_legs(&name, &l);
    println!("ok: {name}");
}

/// ISA-dump vehicle for the split-K `-DKV_BDA` build: dispatch `attn_partial_bda` (hd=128, the b64
/// fast path). Under `RADV_DEBUG=shaders` confirm BOTH (1) K/V loads use a scalar/saddr base + a
/// 32-bit VGPR offset (no per-load `v_add_co_u32`/`v_addc` 64-bit address build), and (2) the f16
/// vec4 reads fuse to `global_load_b128`/`b64` (NOT four `global_load_short_d16`).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn attn_partial_isa_probe() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let c = SplitCase {
        kv_len: 256,
        nh: 8,
        nkv: 2,
        hd: 128,
        rows: 1,
        cap_rows: 256,
        batched: false,
    };
    let l = run_split(&be, &c, false, false);
    assert_legs("isa-probe attn_partial f16", &l);
    println!("ok: attn_partial_isa_probe (dispatched attn_partial_bda)");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// #74 slice 3 — KV STORE kernels (write the DEST cache by device address) and DEQUANT READERS
// (read the SOURCE cache by device address) bound-vs-pointer + offset-invariance.
//
// The store/dequant `-DKV_BDA` twins must produce BYTE-IDENTICAL cache/output bytes to the bound
// builds — they differ ONLY in where the KV bytes are written/read (a bound binding vs a k_addr
// pointer). Three legs each, exactly like the read tests:
//   * BOUND-VS-POINTER: bound store/dequant vs the `_at` twin at arena offset 0.
//   * OFFSET-INVARIANCE (load-bearing): the SAME store/read parked behind a 256-byte garbage prefix
//     in a KvCache buffer, base+prefix passed as the address. A twin that dropped its base offset
//     would clobber/read the garbage prefix and diverge.

/// `n` f32 elements as small finite values in [-1, 1) (never NaN/Inf → a byte compare is never
/// vacuous). Distinct per `seed`.
fn synth_f32(n: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let h = (i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 9;
        let v = ((h % 2048) as f32) / 1024.0 - 1.0;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn assert_bytes(name: &str, bound: &[u8], at0: &[u8], atoff: &[u8]) {
    assert!(
        bound.iter().any(|&b| b != 0),
        "{name}: bound leg wrote all zeros — the case is not exercising the kernel"
    );
    assert_eq!(
        bound, at0,
        "{name}: BDA@0 bytes differ from bound — a mis-addressing bug in the pointer twin"
    );
    assert_eq!(
        bound, atoff,
        "{name}: BDA@nonzero-offset bytes differ from bound — the twin is ignoring its KV base \
         offset, which breaks every KV tensor but one at a shared base"
    );
}

/// Runs a STORE 3 ways into a fresh zeroed KvCache dst (bound, pointer@0, pointer@nonzero-offset)
/// and returns the written cache bytes for each. `store(rec, dst, addr)`: `addr` None = bound build,
/// Some = the `-DKV_BDA` `_at` twin writing at that device address (dst still bound for the barrier).
fn store_legs(
    be: &VulkanBackend,
    dst_len: usize,
    store: impl Fn(&Recorder, &dyn Buffer, Option<u64>),
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let leg = |prefix: usize, bda: bool| -> Vec<u8> {
        let d = be.alloc(prefix + dst_len, BufferUsage::KvCache).unwrap();
        be.upload(d.as_ref(), &vec![0u8; prefix + dst_len]).unwrap();
        let addr = if bda {
            Some(
                d.device_addr()
                    .expect("KvCache must expose a device address")
                    + prefix as u64,
            )
        } else {
            None
        };
        let rec = be.recorder().unwrap();
        store(&rec, d.as_ref(), addr);
        rec.finish().unwrap();
        let mut out = vec![0u8; prefix + dst_len];
        be.download(d.as_ref(), &mut out).unwrap();
        out[prefix..].to_vec()
    };
    (leg(0, false), leg(0, true), leg(OFF, true))
}

// ── Slice-5 BDA-only kernels ──────────────────────────────────────────────────────────────────
// store_q8/quant_kv/store_kv_dense/dequant_q8_f16/dequant_kv_f16 went BDA-only in slice 5 (the bound
// twin AND the explicit-address `_at` method were deleted; the base method now reads the KV cache's
// own `device_addr()`). With no explicit-address entry point the parity test can't park data behind a
// byte prefix and pass base+prefix; instead it proves BASE-INVARIANCE: the SAME store/read run into
// two SIMULTANEOUSLY-LIVE KvCache buffers (distinct device-address bases — a throwaway buffer is held
// between them so the second base is offset from the first) must be byte-identical. A twin that
// mishandled its 64-bit base (e.g. dropped low bits of device_addr) would diverge across the two
// bases. gpu_seam goldens (never re-blessed) remain the end-to-end byte-identity proof.

/// Runs a STORE into two simultaneously-live zeroed KvCache dsts at DISTINCT device-address bases and
/// returns both written caches. `store(rec, dst)` records the BDA store (dst read by `device_addr()`).
fn store_offinv(
    be: &VulkanBackend,
    dst_len: usize,
    store: impl Fn(&Recorder, &dyn Buffer),
) -> (Vec<u8>, Vec<u8>) {
    // All three buffers stay alive → the allocator hands out distinct bases (no freed-then-reused
    // address that would make the two legs share a base and the check vacuous).
    let d0 = be.alloc(dst_len, BufferUsage::KvCache).unwrap();
    let _gap = be.alloc(OFF, BufferUsage::KvCache).unwrap();
    let d1 = be.alloc(dst_len, BufferUsage::KvCache).unwrap();
    let run_one = |d: &dyn Buffer| -> Vec<u8> {
        be.upload(d, &vec![0u8; dst_len]).unwrap();
        let rec = be.recorder().unwrap();
        store(&rec, d);
        rec.finish().unwrap();
        let mut out = vec![0u8; dst_len];
        be.download(d, &mut out).unwrap();
        out
    };
    assert_ne!(
        d0.device_addr().unwrap(),
        d1.device_addr().unwrap(),
        "test setup: the two KvCache dsts must have distinct bases for base-invariance to mean anything"
    );
    (run_one(d0.as_ref()), run_one(d1.as_ref()))
}

/// Runs a DEQUANT READER from two simultaneously-live KvCache srcs at DISTINCT device-address bases,
/// returns both f16 outputs. `run(rec, src, dst)` records the BDA read (src read by `device_addr()`).
fn dequant_offinv(
    be: &VulkanBackend,
    src_bytes: &[u8],
    dst_len: usize,
    run: impl Fn(&Recorder, &dyn Buffer, &dyn Buffer),
) -> (Vec<u8>, Vec<u8>) {
    let s0 = be.alloc(src_bytes.len(), BufferUsage::KvCache).unwrap();
    let _gap = be.alloc(OFF, BufferUsage::KvCache).unwrap();
    let s1 = be.alloc(src_bytes.len(), BufferUsage::KvCache).unwrap();
    be.upload(s0.as_ref(), src_bytes).unwrap();
    be.upload(s1.as_ref(), src_bytes).unwrap();
    let run_one = |src: &dyn Buffer| -> Vec<u8> {
        let dst = be.alloc(dst_len, BufferUsage::Activations).unwrap();
        be.upload(dst.as_ref(), &vec![0u8; dst_len]).unwrap();
        let rec = be.recorder().unwrap();
        run(&rec, src, dst.as_ref());
        rec.finish().unwrap();
        let mut out = vec![0u8; dst_len];
        be.download(dst.as_ref(), &mut out).unwrap();
        out
    };
    assert_ne!(
        s0.device_addr().unwrap(),
        s1.device_addr().unwrap(),
        "test setup: the two KvCache srcs must have distinct bases for base-invariance to mean anything"
    );
    (run_one(s0.as_ref()), run_one(s1.as_ref()))
}

/// Asserts a BDA kernel is byte-identical across the two distinct-base legs (and non-vacuous).
fn assert_offinv(name: &str, a: &[u8], b: &[u8]) {
    assert!(
        a.iter().any(|&x| x != 0),
        "{name}: BDA leg produced all zeros — the case is not exercising the kernel"
    );
    assert_eq!(
        a, b,
        "{name}: BDA output differs across two KvCache allocations (distinct device-address bases) — \
         the pointer twin is mishandling its 64-bit base"
    );
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn store_f16_bda_matches_bound() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    for n in [64usize, 256, 512] {
        let sb = synth_f32(n, 7);
        let src = be.alloc(sb.len(), BufferUsage::Activations).unwrap();
        be.upload(src.as_ref(), &sb).unwrap();
        let (b, a0, ao) = store_legs(&be, n * 2, |rec, dst, addr| match addr {
            Some(a) => rec.store_f16_off_at(src.as_ref(), dst, a, n, 0, 0),
            None => rec.store_f16_off(src.as_ref(), dst, n, 0, 0),
        });
        assert_bytes(&format!("store_f16 n={n}"), &b, &a0, &ao);
        println!("ok: store_f16 n={n}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn store_q8_bda_matches_bound() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // f32 V source and f16 K source; planar Q8 dst = codes[cap] + scales[cap/32]·2, cap = n.
    for n in [64usize, 256] {
        let cap = n;
        let dst_len = cap + (cap / 32) * 2;
        // f32 source.
        let sb = synth_f32(n, 9);
        let src = be.alloc(sb.len(), BufferUsage::Activations).unwrap();
        be.upload(src.as_ref(), &sb).unwrap();
        let (a, b) = store_offinv(&be, dst_len, |rec, dst| {
            rec.store_q8(src.as_ref(), dst, n, 0, cap, false, 0)
        });
        assert_offinv(&format!("store_q8(f32) n={n}"), &a, &b);
        // f16 source.
        let sf = synth_f16(n, 21);
        let srcf = be.alloc(sf.len(), BufferUsage::Activations).unwrap();
        be.upload(srcf.as_ref(), &sf).unwrap();
        let (a, b) = store_offinv(&be, dst_len, |rec, dst| {
            rec.store_q8(srcf.as_ref(), dst, n, 0, cap, true, 0)
        });
        assert_offinv(&format!("store_q8(f16) n={n}"), &a, &b);
        println!("ok: store_q8 n={n}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn quant_kv_bda_matches_bound() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    use infr_core::DType::*;
    // (dtype, bytes-per-32-block) for the mainline low-bit KV quants.
    for (dt, blk) in [
        (Q4_0, 18usize),
        (Q4_1, 20),
        (Q5_0, 22),
        (Q5_1, 24),
        (Iq4Nl, 18),
    ] {
        for n in [64usize, 256] {
            let dst_len = (n / 32) * blk;
            let sb = synth_f32(n, 33);
            let src = be.alloc(sb.len(), BufferUsage::Activations).unwrap();
            be.upload(src.as_ref(), &sb).unwrap();
            let (a, b) = store_offinv(&be, dst_len, |rec, dst| {
                rec.quant_kv(dt, src.as_ref(), dst, n, 0, false)
            });
            assert_offinv(&format!("quant_kv {dt:?} n={n}"), &a, &b);
        }
        println!("ok: quant_kv {dt:?}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn store_kv_dense_bda_matches_bound() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    use infr_core::DType::*;
    for (dt, bytes) in [(F32, 4usize), (Bf16, 2)] {
        for n in [64usize, 256] {
            let dst_len = n * bytes;
            let sb = synth_f32(n, 41);
            let src = be.alloc(sb.len(), BufferUsage::Activations).unwrap();
            be.upload(src.as_ref(), &sb).unwrap();
            let (a, b) = store_offinv(&be, dst_len, |rec, dst| {
                rec.store_kv_dense(dt, src.as_ref(), dst, n, 0, false)
            });
            assert_offinv(&format!("store_kv_dense {dt:?} n={n}"), &a, &b);
        }
        println!("ok: store_kv_dense {dt:?}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn dequant_q8_f16_bda_matches_bound() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    for n in [64usize, 256] {
        let cap = n;
        let src_bytes = synth_bytes(cap + (cap / 32) * 2, 13); // planar Q8 codes + scales
        let (a, b) = dequant_offinv(&be, &src_bytes, n * 2, |rec, src, dst| {
            rec.dequant_q8_f16(src, dst, n, cap)
        });
        assert_offinv(&format!("dequant_q8_f16 n={n}"), &a, &b);
        println!("ok: dequant_q8_f16 n={n}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn dequant_kv_f16_bda_matches_bound() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    use infr_core::DType::*;
    for (dt, blk) in [
        (Q4_0, 18usize),
        (Q4_1, 20),
        (Q5_0, 22),
        (Q5_1, 24),
        (Iq4Nl, 18),
    ] {
        for n in [64usize, 256] {
            let src_bytes = synth_bytes((n / 32) * blk, 17); // GGUF blocks
            let (a, b) = dequant_offinv(&be, &src_bytes, n * 2, |rec, src, dst| {
                rec.dequant_kv_f16(dt, src, dst, n)
            });
            assert_offinv(&format!("dequant_kv_f16 {dt:?} n={n}"), &a, &b);
        }
        println!("ok: dequant_kv_f16 {dt:?}");
    }
}

/// qk_norm_rope — the HOT fused-K cache write. Bound `qk_norm_rope` vs `qk_norm_rope_at` writing the
/// same f16 cache by device address, plus offset-invariance. out_base=0 (write cache rows [0,rows)).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn qk_norm_rope_bda_matches_bound() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (rows, nh, hd, rope_dim) = (3usize, 8usize, 64usize, 64usize);
    let n = rows * nh * hd;
    let xb = synth_f32(n, 5);
    let x = be.alloc(xb.len(), BufferUsage::Activations).unwrap();
    be.upload(x.as_ref(), &xb).unwrap();
    let nwb = synth_f32(hd, 55);
    let nw = be.alloc(nwb.len(), BufferUsage::Activations).unwrap();
    be.upload(nw.as_ref(), &nwb).unwrap();
    let (b, a0, ao) = store_legs(&be, n * 2, |rec, dst, addr| match addr {
        Some(a) => rec.qk_norm_rope_at(
            x.as_ref(),
            nw.as_ref(),
            dst,
            a,
            rows,
            nh,
            hd,
            rope_dim,
            10000.0,
            0,
            0,
            1e-6,
            None,
        ),
        None => rec.qk_norm_rope(
            x.as_ref(),
            nw.as_ref(),
            dst,
            rows,
            nh,
            hd,
            rope_dim,
            10000.0,
            0,
            0,
            1e-6,
            None,
        ),
    });
    assert_bytes("qk_norm_rope", &b, &a0, &ao);
    println!("ok: qk_norm_rope");
}
