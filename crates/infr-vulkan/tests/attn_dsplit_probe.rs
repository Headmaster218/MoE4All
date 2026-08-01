//! PROBE (docs/backlog.md B7, slice 2) — decode attention pass 1 at a PARAMETERIZED subgroup
//! reduction width.
//!
//! `attn_partial.comp`'s hd==128 QK loop has all 32 lanes of a wave cooperate on ONE key's 128-dim
//! dot, reduced by a full `subgroupAdd`: reduction width 32, exactly one key in flight per wave.
//! Slice 1's LDS K-tile probe was the opposite extreme (width 1, a whole dot per thread, no
//! cross-lane op) and lost 2.7x. llama.cpp runs this shape (hsk=hsv=128, n_rows=1, f16 KV) on RDNA3
//! at width 8 — `get_fa_tuning_params_scalar`'s `d_split = min(min(subgroup_size, 8), D_lsb/4)`.
//! `attn_partial_dsplit.comp` makes the width a build-time constant so the middle ground neither
//! previous experiment tried can be measured. Nothing in production dispatches it; this file is its
//! only caller.
//!
//! Two tests:
//!  * `dsplit_matches_split_reference` — combined output vs the shipped `attention_kv_split_at` at
//!    several shapes, for EVERY width. Width 32 reproduces the shipped summation order exactly;
//!    the narrower widths reassociate the 128-dim dot, so this is a tight RELATIVE tolerance. The
//!    reference is first asserted finite and mostly non-zero so the compare cannot pass vacuously.
//!  * `dsplit_bench` (`--ignored`) — per-dispatch us for the reference and every width at the B7
//!    target shape (nh=32, nkv=4, hd=128, chunk=512) and kv_len 8192 / 32768. The reference is
//!    measured IN THIS HARNESS, alternated around every probe leg, rather than trusting B7's table.
//!
//! Run: `cargo test --release -p infr-vulkan --test attn_dsplit_probe -- --ignored --nocapture`
//! (the cargo wrapper swallows test stdout — run `target/release/deps/attn_dsplit_probe-*`
//! directly).
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_vulkan::{Recorder, VulkanBackend};

/// `(reduction width, workgroup threads, label)` — every build of `attn_partial_dsplit.comp`.
/// Width 32 + wg 64 is the shipped configuration expressed through the parameterization, and is
/// the self-check that the parameterization is faithful (it must match the reference's timing).
const CFGS: &[(u32, u32, &str)] = &[
    (1, 64, "w=1  wg=64  (32 keys/wave, no cross-lane op)"),
    (2, 64, "w=2  wg=64  (16 keys/wave)"),
    (4, 64, "w=4  wg=64  (8 keys/wave)"),
    (8, 64, "w=8  wg=64  (4 keys/wave, llama.cpp's d_split)"),
    (16, 64, "w=16 wg=64  (2 keys/wave)"),
    (32, 64, "w=32 wg=64  (1 key/wave == SHIPPED behaviour)"),
    (1, 128, "w=1  wg=128 (4 waves/workgroup)"),
    (2, 128, "w=2  wg=128"),
    (4, 128, "w=4  wg=128"),
    (8, 128, "w=8  wg=128 (llama.cpp's d_split AND workgroup)"),
    (16, 128, "w=16 wg=128"),
    (32, 128, "w=32 wg=128"),
];

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
/// positive, which lets one key dominate the softmax and hides disagreement in the rest — the sign
/// is what keeps this comparison discriminating.
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
    chunk: usize,
}

const HD: usize = 128;

/// Allocates one case's buffers and returns `(reference_o, dsplit_o[cfg])`.
fn run_case(be: &VulkanBackend, s: &Shape) -> (Vec<f32>, Vec<Vec<f32>>) {
    let Shape {
        kv_len,
        nh,
        nkv,
        chunk,
    } = *s;
    let n_chunks = kv_len.div_ceil(chunk);
    let pos = kv_len - 1; // rows == 1: the decode query is position kv_len-1
    let cache_elems = kv_len * nkv * HD;

    let qb = be.alloc(nh * HD * 2, BufferUsage::Activations).unwrap();
    be.upload(qb.as_ref(), &f16_data(nh * HD, 101)).unwrap();
    let kb = be.alloc(cache_elems * 2, BufferUsage::KvCache).unwrap();
    let vb = be.alloc(cache_elems * 2, BufferUsage::KvCache).unwrap();
    be.upload(kb.as_ref(), &f16_data(cache_elems, 11)).unwrap();
    be.upload(vb.as_ref(), &f16_data(cache_elems, 23)).unwrap();
    let ka = kb.device_addr().expect("KvCache K device address");
    let va = vb.device_addr().expect("KvCache V device address");

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

    let read = |b: &dyn Buffer| -> Vec<f32> {
        let mut out = vec![0u8; o_bytes];
        be.download(b, &mut out).unwrap();
        bytemuck::cast_slice::<u8, f32>(&out).to_vec()
    };

    let o_ref = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    reference(
        &rec,
        qb.as_ref(),
        kb.as_ref(),
        vb.as_ref(),
        ka,
        va,
        o_ref.as_ref(),
        pm.as_ref(),
        pl.as_ref(),
        pacc.as_ref(),
        pos,
        kv_len,
        nh,
        nkv,
        chunk,
        n_chunks,
    );
    rec.finish().unwrap();
    let want = read(o_ref.as_ref());

    let mut got = Vec::new();
    for &(width, wg, _) in CFGS {
        let o = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_kv_split_dsplit_at(
            qb.as_ref(),
            kb.as_ref(),
            vb.as_ref(),
            ka,
            va,
            o.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            pos,
            kv_len,
            nh,
            nkv,
            HD,
            chunk,
            n_chunks,
            0.0,
            width,
            wg,
        );
        rec.finish().unwrap();
        got.push(read(o.as_ref()));
    }
    (want, got)
}

/// The shipped split-K decode path (`attn_partial_bda` + `attn_combine`), f16 K/V by device
/// address, full causal, no window/canvas/Q8/ring — the exact configuration the probe targets.
#[allow(clippy::too_many_arguments)]
fn reference(
    rec: &Recorder,
    q: &dyn Buffer,
    kc: &dyn Buffer,
    vc: &dyn Buffer,
    ka: u64,
    va: u64,
    o: &dyn Buffer,
    pm: &dyn Buffer,
    pl: &dyn Buffer,
    pacc: &dyn Buffer,
    pos: usize,
    kv_len: usize,
    nh: usize,
    nkv: usize,
    chunk: usize,
    n_chunks: usize,
) {
    rec.attention_kv_split_at(
        q, kc, vc, ka, va, o, pm, pl, pacc, 1, pos, kv_len, nh, nkv, HD, chunk, n_chunks, 0.0, 0,
        None, false, false, 0, false, None,
    );
}

#[test]
fn dsplit_matches_split_reference() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let cases = [
        // The B7 decode shape, scaled down: GQA g=8, several full chunks.
        Shape {
            kv_len: 2048,
            nh: 32,
            nkv: 4,
            chunk: 512,
        },
        // Ragged last chunk (1000 = 3*256 + 232) — exercises the masked key tail.
        Shape {
            kv_len: 1000,
            nh: 32,
            nkv: 4,
            chunk: 256,
        },
        // A chunk holding a SINGLE key (513 = 512 + 1): in that workgroup only one lane group has
        // a valid key, so the whole tail is masked except one cluster.
        Shape {
            kv_len: 513,
            nh: 16,
            nkv: 2,
            chunk: 512,
        },
        // MHA (g == 1, nh == nkv) — the other end of the workgroup→(head, chunk) decomposition.
        Shape {
            kv_len: 1024,
            nh: 8,
            nkv: 8,
            chunk: 512,
        },
        // kv_len below one chunk, and fewer keys than the widest configuration's per-iteration key
        // count (w=1 wg=128 covers 128 keys per iteration) → the pipelined loop never runs.
        Shape {
            kv_len: 40,
            nh: 8,
            nkv: 2,
            chunk: 32,
        },
        // The BENCHMARKED shapes themselves, so no timed configuration goes unverified.
        Shape {
            kv_len: 8192,
            nh: 32,
            nkv: 4,
            chunk: 512,
        },
        Shape {
            kv_len: 8192,
            nh: 32,
            nkv: 4,
            chunk: 256,
        },
        Shape {
            kv_len: 32768,
            nh: 32,
            nkv: 4,
            chunk: 512,
        },
    ];
    let mut worst = 0f32;
    let mut worst_w32 = 0f32;
    for s in &cases {
        let (want, got) = run_case(&be, s);
        // Non-vacuity: the reference itself must be finite and carry real signal.
        assert!(
            want.iter().all(|v| v.is_finite()),
            "reference output has non-finite values (kv_len={})",
            s.kv_len
        );
        let nz = want.iter().filter(|v| v.abs() > 1e-6).count();
        assert!(
            nz * 4 > want.len() * 3,
            "reference output is mostly zero ({nz}/{}) — the compare would be vacuous",
            want.len()
        );
        for (gi, g) in got.iter().enumerate() {
            let (width, wg, name) = CFGS[gi];
            for i in 0..want.len() {
                assert!(
                    g[i].is_finite(),
                    "{name} kv_len={} idx {i} not finite",
                    s.kv_len
                );
                let denom = want[i].abs().max(0.05); // outputs are O(0.1); floor keeps near-zeros sane
                let rel = (want[i] - g[i]).abs() / denom;
                worst = worst.max(rel);
                if width == 32 {
                    worst_w32 = worst_w32.max(rel);
                }
                assert!(
                    rel <= 1e-3,
                    "{name} kv_len={} nh={} chunk={}: head {} dim {} reference {} vs dsplit(w={width}, wg={wg}) {} (rel {rel:.3e})",
                    s.kv_len,
                    s.nh,
                    s.chunk,
                    i / HD,
                    i % HD,
                    want[i],
                    g[i]
                );
            }
        }
    }
    eprintln!(
        "attn_partial_dsplit == attention_kv_split_at across {} shapes x {} configs; \
         worst rel {worst:.3e} (width 32, the shipped summation order: {worst_w32:.3e})",
        cases.len(),
        CFGS.len()
    );
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench); run alone, nothing else on the GPU"]
fn dsplit_bench() {
    let be = VulkanBackend::new().unwrap();
    let (nh, nkv) = (32usize, 4usize);
    let reps = 200usize;
    let rounds = 5usize;

    // (kv_len, chunk). B7's shape is chunk=512 at both depths, but the SHIPPED policy
    // (`adaptive_chunk`, ~32 chunks/head clamped to 64..512) picks 256 at kv_len 8192 and 512 at
    // 32768 — so 8192 is measured BOTH ways: at 512 the reference runs only 512 workgroups on 96
    // CUs, which is not a configuration production ever dispatches.
    for (kv_len, chunk) in [(8192usize, 512usize), (8192, 256), (32768, 512)] {
        let n_chunks = kv_len.div_ceil(chunk);
        let pos = kv_len - 1;
        let cache_elems = kv_len * nkv * HD;
        let qb = be.alloc(nh * HD * 2, BufferUsage::Activations).unwrap();
        be.upload(qb.as_ref(), &f16_data(nh * HD, 101)).unwrap();
        let kb = be.alloc(cache_elems * 2, BufferUsage::KvCache).unwrap();
        let vb = be.alloc(cache_elems * 2, BufferUsage::KvCache).unwrap();
        be.upload(kb.as_ref(), &f16_data(cache_elems, 11)).unwrap();
        be.upload(vb.as_ref(), &f16_data(cache_elems, 23)).unwrap();
        let ka = kb.device_addr().unwrap();
        let va = vb.device_addr().unwrap();
        let pm = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pl = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pacc = be
            .alloc(nh * n_chunks * HD * 4, BufferUsage::Activations)
            .unwrap();
        let o = be.alloc(nh * HD * 4, BufferUsage::Activations).unwrap();

        let time = |f: &dyn Fn(&Recorder)| -> f64 {
            let rec = be.recorder().unwrap(); // warmup: pipeline compile out of the timed region
            f(&rec);
            rec.finish().unwrap();
            let t0 = std::time::Instant::now();
            let rec = be.recorder().unwrap();
            for _ in 0..reps {
                f(&rec);
            }
            rec.finish().unwrap();
            t0.elapsed().as_secs_f64() * 1e6 / reps as f64
        };
        let run_ref = |rec: &Recorder| {
            reference(
                rec,
                qb.as_ref(),
                kb.as_ref(),
                vb.as_ref(),
                ka,
                va,
                o.as_ref(),
                pm.as_ref(),
                pl.as_ref(),
                pacc.as_ref(),
                pos,
                kv_len,
                nh,
                nkv,
                chunk,
                n_chunks,
            );
        };

        // Alternate the reference around every probe leg (perf-ab-methodology: order matters), and
        // take the MEDIAN of `rounds` sweeps — a single sweep on this GPU carries several percent.
        let mut ref_us: Vec<f64> = Vec::new();
        let mut cfg_us: Vec<Vec<f64>> = vec![Vec::new(); CFGS.len()];
        // Warm-up sweep, DISCARDED: `time()`'s own single-dispatch warmup gets the pipeline
        // compiled but not the clocks up, and the first few measured sweeps otherwise land 20-40%
        // high — which showed up as a reference spread far wider than any effect being measured.
        for _ in 0..2 {
            time(&run_ref);
            for &(width, wg, _) in CFGS {
                time(&|rec: &Recorder| {
                    rec.attention_kv_split_dsplit_at(
                        qb.as_ref(),
                        kb.as_ref(),
                        vb.as_ref(),
                        ka,
                        va,
                        o.as_ref(),
                        pm.as_ref(),
                        pl.as_ref(),
                        pacc.as_ref(),
                        pos,
                        kv_len,
                        nh,
                        nkv,
                        HD,
                        chunk,
                        n_chunks,
                        0.0,
                        width,
                        wg,
                    );
                });
            }
        }
        for _ in 0..rounds {
            ref_us.push(time(&run_ref));
            for (i, &(width, wg, _)) in CFGS.iter().enumerate() {
                cfg_us[i].push(time(&|rec: &Recorder| {
                    rec.attention_kv_split_dsplit_at(
                        qb.as_ref(),
                        kb.as_ref(),
                        vb.as_ref(),
                        ka,
                        va,
                        o.as_ref(),
                        pm.as_ref(),
                        pl.as_ref(),
                        pacc.as_ref(),
                        pos,
                        kv_len,
                        nh,
                        nkv,
                        HD,
                        chunk,
                        n_chunks,
                        0.0,
                        width,
                        wg,
                    );
                }));
                ref_us.push(time(&run_ref));
            }
        }
        let med = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let spread = |v: &[f64]| -> f64 {
            let (lo, hi) = v
                .iter()
                .fold((f64::MAX, 0f64), |(l, h), &x| (l.min(x), h.max(x)));
            (hi - lo) / lo * 100.0
        };
        let rspread = spread(&ref_us);
        let r = med(&mut ref_us);
        println!(
            "\n=== kv_len={kv_len}  nh={nh} nkv={nkv} hd={HD} chunk={chunk} n_chunks={n_chunks} \
             ({} workgroups)  reps={reps} rounds={rounds} ===",
            nh * n_chunks
        );
        println!(
            "  {:52} {:>9}  {:>7}  {:>7}",
            "leg (pass1 + attn_combine)", "us/disp", "vs ref", "spread"
        );
        println!(
            "  {:52} {r:9.1}  {:>7}  {rspread:6.1}%",
            "attention_kv_split_at (SHIPPED reference)", "1.00x"
        );
        for (i, (_, _, name)) in CFGS.iter().enumerate() {
            let sp = spread(&cfg_us[i]);
            let m = med(&mut cfg_us[i]);
            println!("  {name:52} {m:9.1}  {:6.2}x  {sp:6.1}%", r / m);
        }
    }
}
