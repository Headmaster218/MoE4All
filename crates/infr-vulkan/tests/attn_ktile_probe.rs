//! PROBE (docs/backlog.md B7, slice 1) — LDS-staged **K-tile** decode attention pass 1.
//!
//! `attn_partial.comp` gives each 32-lane wave ONE key and reduces that key's 128-dim dot with a
//! cross-lane `subgroupAdd`; that reduction ALU scales with keys x heads and is 59% of decode GPU
//! time at d32768 (177 us per layer-token on a 7900 XTX). `attn_ktile.comp` instead stages a tile
//! of K in shared memory with coalesced global reads and gives each THREAD a whole 128-dim dot, so
//! the per-key cross-lane reduction disappears. This file is the ONLY caller of that kernel —
//! nothing in production dispatches it.
//!
//! Two tests:
//!  * `ktile_matches_split_reference` — combined output vs the shipped `attention_kv_split_at` at
//!    several shapes. Bitwise equality is NOT expected (the key→thread mapping, and therefore the
//!    dot summation order, differs by design), so this is a tight RELATIVE tolerance; the reference
//!    is first asserted non-zero and all-finite so the compare cannot pass vacuously.
//!  * `ktile_bench` (`--ignored`) — per-dispatch us for the reference and every k-tile config at
//!    the B7 target shape (nh=32, nkv=4, hd=128, chunk=512) and kv_len 8192 / 32768. The reference
//!    is measured IN THIS HARNESS, alternated around the probe legs, rather than trusting the
//!    numbers in B7.
//!
//! Run: `cargo test --release -p infr-vulkan --test attn_ktile_probe -- --ignored --nocapture`
//! (the cargo wrapper swallows test stdout — run `target/release/deps/attn_ktile_probe-*` directly).
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_vulkan::{Recorder, VulkanBackend};

/// The four `attn_ktile` build configurations, as `Recorder::attention_kv_split_ktile_at`'s `cfg`.
/// LDS figures are the K tile only (`keys * row_stride_words * 4`); each adds ~3.8 KB of
/// `sc`/`qf4`/`red`/`vsh` on top.
const CFGS: &[(u32, &str)] = &[
    (0, "w64      (64-key tile, 68-word rows, 17.0 KB K-LDS)"),
    (1, "w64_nopad(64-key tile, 64-word rows, 16.0 KB K-LDS)"),
    (2, "w128     (128-key tile, 68-word rows, 34.0 KB K-LDS)"),
    (3, "w64_dw32 (64-key tile, half-depth stage, 9.0 KB K-LDS)"),
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

/// `n` f16 elements drawn from [-1, 1). SIGNED (unlike kv_addr_parity's masked-bits helper): a
/// non-negative K/Q makes every score large and positive, which lets one key dominate the softmax
/// and hides disagreement in the rest — the sign is what keeps this comparison discriminating.
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

/// Allocates one case's buffers and returns `(reference_o, ktile_o[cfg])`.
fn run_case(be: &VulkanBackend, s: &Shape, cfgs: &[u32]) -> (Vec<f32>, Vec<Vec<f32>>) {
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
    for &cfg in cfgs {
        let o = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_kv_split_ktile_at(
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
            cfg,
        );
        rec.finish().unwrap();
        got.push(read(o.as_ref()));
    }
    (want, got)
}

/// The shipped split-K decode path (`attn_partial_bda` + `attn_combine`), f16 K/V by device
/// address, full causal, no window/canvas/Q8/ring — the exact configuration `attn_ktile` targets.
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
fn ktile_matches_split_reference() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let cfgs: Vec<u32> = CFGS.iter().map(|(c, _)| *c).collect();
    let cases = [
        // The B7 decode shape, scaled down: GQA g=8, several full chunks.
        Shape {
            kv_len: 2048,
            nh: 32,
            nkv: 4,
            chunk: 512,
        },
        // Ragged last chunk (1000 = 3*256 + 232) — exercises the partial-tile stage guard.
        Shape {
            kv_len: 1000,
            nh: 32,
            nkv: 4,
            chunk: 256,
        },
        // A chunk holding a SINGLE key (513 = 512 + 1): 63 of 64 threads idle in that workgroup.
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
        // kv_len below one tile (64 keys) so the tile loop runs exactly once, mostly masked.
        Shape {
            kv_len: 40,
            nh: 8,
            nkv: 2,
            chunk: 32,
        },
    ];
    let mut worst = 0f32;
    for s in &cases {
        let (want, got) = run_case(&be, s, &cfgs);
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
            let (_, name) = CFGS[gi];
            for i in 0..want.len() {
                assert!(
                    g[i].is_finite(),
                    "{name} kv_len={} idx {i} not finite",
                    s.kv_len
                );
                let denom = want[i].abs().max(0.05); // outputs are O(0.1); floor keeps near-zeros sane
                let rel = (want[i] - g[i]).abs() / denom;
                worst = worst.max(rel);
                assert!(
                    rel <= 1e-3,
                    "{name} kv_len={} nh={} chunk={}: head {} dim {} reference {} vs ktile {} (rel {rel:.3e})",
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
        "attn_ktile == attention_kv_split_at across 5 shapes x 4 configs; worst rel {worst:.3e}"
    );
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench); run alone, nothing else on the GPU"]
fn ktile_bench() {
    let be = VulkanBackend::new().unwrap();
    let (nh, nkv, chunk) = (32usize, 4usize, 512usize);
    let reps = 200usize;
    let rounds = 3usize;

    for kv_len in [8192usize, 32768] {
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

        // Alternate reference around every probe leg (perf-ab-methodology: order matters), and
        // take the MEDIAN of `rounds` sweeps — a single sweep on this GPU carries several percent.
        let mut ref_us: Vec<f64> = Vec::new();
        let mut cfg_us: Vec<Vec<f64>> = vec![Vec::new(); CFGS.len()];
        for _ in 0..rounds {
            ref_us.push(time(&run_ref));
            for (i, (cfg, _)) in CFGS.iter().enumerate() {
                let cfg = *cfg;
                cfg_us[i].push(time(&|rec: &Recorder| {
                    rec.attention_kv_split_ktile_at(
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
                        cfg,
                    );
                }));
                ref_us.push(time(&run_ref));
            }
        }
        let med = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let r = med(&mut ref_us);
        println!(
            "\n=== kv_len={kv_len}  nh={nh} nkv={nkv} hd={HD} chunk={chunk} n_chunks={n_chunks} \
             ({} workgroups)  reps={reps} rounds={rounds} ===",
            nh * n_chunks
        );
        println!(
            "  {:52} {:>9}  {:>7}",
            "leg (pass1 + attn_combine)", "us/disp", "vs ref"
        );
        println!(
            "  {:52} {r:9.1}  {:>7}",
            "attention_kv_split_at (SHIPPED reference)", "1.00x"
        );
        for (i, (_, name)) in CFGS.iter().enumerate() {
            let m = med(&mut cfg_us[i]);
            println!("  {name:52} {m:9.1}  {:6.2}x", r / m);
        }
    }
}
