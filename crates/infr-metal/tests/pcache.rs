//! The persisted `MTLBinaryArchive` pipeline cache, end to end: build → create pipelines → save →
//! reload → create pipelines FROM the archive, with identical kernel results at every step.
//!
//! # What this file is a gate ON
//!
//! Reloading a blob is not the property worth testing — a cache that silently missed on every
//! launch would still round-trip a file and still compute the right answers, just at cold speed
//! forever. So the cold → warm transition is asserted STRUCTURALLY:
//!
//! * cold (nothing on disk) must be `hits == 0`, `misses > 0`, and must leave a non-empty blob;
//! * warm (same key, blob present) must be `misses == 0` — EVERY pipeline out of the archive;
//! * the two runs must serve the same SET OF KERNEL NAMES, so the manifest is proven to describe
//!   the archive and not merely to have the right cardinality;
//! * and an archive-backed pipeline's output must be BITWISE identical to a freshly compiled
//!   one's, which is the failure a checksum structurally cannot see (a blob that loads fine and
//!   computes something subtly different is the poisoned-blob class the seam's tripwire exists
//!   for).
//!
//! Timings are PRINTED, not gated, save for one catastrophic-shape bound — see
//! [`WARM_PSO_CATASTROPHIC_FACTOR`]. A flaky gate on a shared CI runner is worse than no gate; a
//! permanent record in the log of where Metal startup actually goes is worth more than either.
//!
//! # What is checked elsewhere
//!
//! Everything that can be checked without a GPU is checked without one: the payload framing, the
//! cache key and the device token in `src/pcache_blob.rs` (device-free, runnable standalone off a
//! Mac), the debounce and the temp-path shapes in `src/pcache.rs`, and the envelope + checksum +
//! durable write + poisoned-blob tripwire in `infr_core::kernel_cache`. What is LEFT — and what
//! only a real Metal device can answer — is whether Metal accepts an archive we serialized,
//! whether a pipeline created through it behaves identically to a freshly compiled one, and
//! whether the degradation paths actually degrade instead of failing.
//!
//! macOS-only and `#[ignore]`d, like the rest of this crate's device tests:
//!
//!   cargo test -p infr-metal --test pcache -- --include-ignored --nocapture
//!
//! **This file deliberately holds exactly ONE `#[test]`.** It points `XDG_CACHE_HOME` at a private
//! temp directory so it never touches the developer's real `~/.cache/infr` — and mutating process
//! environment is only safe because nothing else runs in this binary. If a second test is ever
//! added here, it must go in its own file or the two will race.
#![cfg(target_os = "macos")]

use std::sync::Arc;
use std::time::Duration;

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::config::Config;
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_metal::{MetalBackend, PipelineCacheStats};

/// The one timing bound this test enforces, and deliberately the ONLY one.
///
/// A warm run creates every pipeline from the archive, so its PSO wall should be well BELOW cold's
/// — but that is not assertable here. `macos-15` runners are shared, contended and thermally
/// unpredictable, Apple keeps its own system shader cache underneath ours (so even a "cold" run may
/// be served from something), and the honest expectation for the win is "somewhere between large
/// and nil". Gating on cold > warm would fail on noise, and a gate that fails on noise gets
/// `#[ignore]`d within a month, at which point the structural assertions above go with it.
///
/// What IS unambiguous is the catastrophic shape: the archive path costing MULTIPLES of the compile
/// it replaces — a re-add (and therefore a recompile) of every kernel because the manifest stopped
/// round-tripping, a re-serialize per PSO because the debounce broke, a linear rescan of the
/// archive per lookup. Each of those puts warm at or far above cold, not near it. 4× is far outside
/// anything runner contention produces on a number that is tens of milliseconds, and far inside
/// anything those bugs produce.
const WARM_PSO_CATASTROPHIC_FACTOR: f64 = 4.0;

/// Floor under that bound, so a very fast cold run cannot make it tiny. Below a few tens of
/// milliseconds the measurement is scheduler jitter, not pipeline creation, and 4× of jitter is
/// jitter.
const WARM_PSO_FLOOR_MS: f64 = 25.0;

/// Deterministic inputs (LCG — no rng dependency), so "the same result" is a byte-exact claim.
fn rand_f32(n: usize, mut s: u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// BITWISE equality of two runs' outputs, reported down to the first differing lane.
///
/// `assert_eq!` on `f32` would be `PartialEq`, which is not the claim being made: it calls `-0.0`
/// equal to `0.0` and every `NaN` unequal to itself, so a pipeline whose sign-of-zero or NaN
/// payload moved would pass while a perfectly correct one containing a `NaN` would fail. The claim
/// is that a pipeline created FROM a stale/rebuilt archive is indistinguishable from a freshly
/// compiled one at the bit level — there is no acceptable epsilon for "the cache changed the
/// answer", which is precisely the failure a checksum over the blob cannot see.
fn assert_bit_identical(what: &str, reference: &[Vec<f32>], actual: &[Vec<f32>]) {
    assert_eq!(
        reference.len(),
        actual.len(),
        "{what}: different number of outputs"
    );
    for (k, (r, a)) in reference.iter().zip(actual).enumerate() {
        assert_eq!(r.len(), a.len(), "{what}: output {k} changed length");
        for (i, (x, y)) in r.iter().zip(a).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what}: output {k} lane {i} differs — {x:?} (0x{:08x}) vs {y:?} (0x{:08x})",
                x.to_bits(),
                y.to_bits(),
            );
        }
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Print the cold/warm breakdown, split into the two compilers Metal startup actually runs.
///
/// This is half the point of the slice. Nobody on this project can run Metal, so the only durable
/// record of what the pipeline cache buys is what the macOS CI log says — which is why the job
/// passes `--nocapture`. The split matters more than the totals: RM caches the BACK end (AIR → GPU
/// ISA, one compile per PSO) and cannot cache the FRONT end (MSL → AIR over ~340 KiB of source),
/// because `MTLLibrary` has no serialize API. If the front end is what dominates a warm launch,
/// then the next lever is a build-time `.metallib`, not anything about this cache — and the log
/// should say so rather than leave the next person to re-derive it.
fn report(blob: &std::path::Path, cold: &PipelineCacheStats, warm: &PipelineCacheStats) {
    let blob_kib = std::fs::metadata(blob).map(|m| m.len()).unwrap_or(0) as f64 / 1024.0;
    let row = |label: &str, c: String, w: String| eprintln!("│ {label:<30} {c:>14} {w:>14}");
    eprintln!("┌── metal pipeline cache ─────────────────────────────────────────");
    row("", "COLD".into(), "WARM".into());
    row(
        "MSL → AIR  front end (uncached)",
        format!("{:.1} ms", ms(cold.library)),
        format!("{:.1} ms", ms(warm.library)),
    );
    row(
        "AIR → ISA  back end (CACHED)",
        format!("{:.1} ms", ms(cold.pso)),
        format!("{:.1} ms", ms(warm.pso)),
    );
    row(
        "pipelines from archive",
        cold.hits.to_string(),
        warm.hits.to_string(),
    );
    row(
        "pipelines compiled",
        cold.misses.to_string(),
        warm.misses.to_string(),
    );
    row(
        "seeded from disk",
        cold.seeded.to_string(),
        warm.seeded.to_string(),
    );
    row("blob", "—".into(), format!("{blob_kib:.1} KiB"));
    let cold_total = ms(cold.library) + ms(cold.pso);
    let warm_total = ms(warm.library) + ms(warm.pso);
    row(
        "total shader startup",
        format!("{cold_total:.1} ms"),
        format!("{warm_total:.1} ms"),
    );
    eprintln!(
        "│ back end saved by the cache: {:.1} ms ({:.0}% of a cold start)",
        ms(cold.pso) - ms(warm.pso),
        if cold_total > 0.0 {
            (ms(cold.pso) - ms(warm.pso)) / cold_total * 100.0
        } else {
            0.0
        },
    );
    // The honest conclusion, stated in the log rather than left to be re-derived.
    if warm_total > 0.0 && ms(warm.library) > ms(warm.pso) {
        eprintln!(
            "│ VERDICT: the UNCACHED front end is {:.0}% of a warm start ({:.1} of {:.1} ms). This \
             cache cannot touch it — the next lever is a build-time .metallib, not a bigger \
             archive.",
            ms(warm.library) / warm_total * 100.0,
            ms(warm.library),
            warm_total,
        );
    } else {
        eprintln!(
            "│ VERDICT: pipeline creation still dominates a warm start ({:.1} of {:.1} ms) — the \
             archive is where the remaining win is.",
            ms(warm.pso),
            warm_total,
        );
    }
    eprintln!("└─────────────────────────────────────────────────────────────────");
}

/// A graph plus its bound inputs (`id → bytes`) and the outputs to read back (`id → f32 count`).
type Case = (Graph, Vec<(TensorId, Vec<u8>)>, Vec<(TensorId, usize)>);

/// Three independent elementwise ops in one graph — three DISTINCT kernels, so the archive holds
/// more than one entry and the manifest round-trip is doing real work. Independent (rather than
/// chained) so every tensor is either bound or an output, with no intermediates to allocate.
fn graph() -> Case {
    const N: usize = 4096;
    let mut g = Graph::new();
    let a = g.input(TensorDesc::new(vec![N], DType::F32));
    let b = g.input(TensorDesc::new(vec![N], DType::F32));
    let sum = g.output(TensorDesc::new(vec![N], DType::F32));
    let scaled = g.output(TensorDesc::new(vec![N], DType::F32));
    let capped = g.output(TensorDesc::new(vec![N], DType::F32));
    g.push(Op::Add {
        a,
        b,
        dst: sum,
        n: N as u32,
    });
    g.push(Op::Scale {
        x: a,
        dst: scaled,
        s: 0.125,
        n: N as u32,
    });
    g.push(Op::Softcap {
        x: b,
        dst: capped,
        cap: 30.0,
        n: N as u32,
    });
    let bound = vec![
        (a, bytemuck::cast_slice(&rand_f32(N, 0xA11CE)).to_vec()),
        (
            b,
            bytemuck::cast_slice(
                &rand_f32(N, 0xB0B)
                    .iter()
                    .map(|v| v * 60.0)
                    .collect::<Vec<_>>(),
            )
            .to_vec(),
        ),
    ];
    (g, bound, vec![(sum, N), (scaled, N), (capped, N)])
}

/// One full backend lifetime: construct, run [`graph`], snapshot the cache stats, then DROP the
/// backend (which is what saves the archive and disarms the tripwire) before returning.
fn run_once(cfg: Config) -> (Vec<Vec<f32>>, PipelineCacheStats) {
    let be = MetalBackend::new_with(Arc::new(cfg)).expect("metal backend");
    let (g, bound, reads) = graph();

    let mut bufs: Vec<(TensorId, Box<dyn Buffer>)> = Vec::new();
    for (id, bytes) in &bound {
        let buf = be.alloc(bytes.len(), BufferUsage::Activations).unwrap();
        be.upload(buf.as_ref(), bytes).unwrap();
        bufs.push((*id, buf));
    }
    for (id, n) in &reads {
        bufs.push((*id, be.alloc(n * 4, BufferUsage::Activations).unwrap()));
    }
    let mut binds = Bindings::new();
    for (id, b) in &bufs {
        binds.bind(*id, b.as_ref());
    }
    let plan = be.compile(&g).unwrap();
    be.execute(plan.as_ref(), &binds).unwrap();
    be.sync().unwrap();

    let out = reads
        .iter()
        .map(|(id, n)| {
            let buf = &bufs.iter().find(|(i, _)| i == id).unwrap().1;
            let mut bytes = vec![0u8; n * 4];
            be.download(buf.as_ref(), &mut bytes).unwrap();
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        })
        .collect();

    let stats = be.pipeline_cache_stats();
    drop(bufs);
    drop(be); // saves the archive through the seam and disarms the tripwire
    (out, stats)
}

#[test]
#[ignore = "requires a Metal GPU"]
fn pipelines_round_trip_through_the_persisted_binary_archive() {
    let dir = std::env::temp_dir().join(format!("infr-metal-pcache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Safe here and ONLY here: this binary holds exactly one test. See the module doc.
    std::env::set_var("XDG_CACHE_HOME", &dir);

    // ── 1. COLD. Nothing on disk: every pipeline is compiled and added to a fresh archive.
    let (cold_out, cold) = run_once(Config::default());
    eprintln!("cold : {cold:?}");
    let blob = match cold.blob.clone() {
        Some(p) => p,
        // A device with no binary-archive support is a legitimate configuration, not a failure —
        // the backend just runs uncached. Nothing below is meaningful there.
        None => {
            eprintln!("no pipeline archive on this device — nothing to test");
            std::env::remove_var("XDG_CACHE_HOME");
            return;
        }
    };
    assert!(
        blob.starts_with(&dir),
        "the test must not write to the real cache dir: {}",
        blob.display()
    );
    assert!(!cold.seeded, "nothing was on disk, so nothing was seeded");
    assert_eq!(cold.hits, 0, "a cold run cannot hit");
    assert!(cold.misses > 0, "the graph must create pipelines");
    assert_eq!(
        cold.served.len() as u64,
        cold.misses,
        "every kernel created cold must be accounted for as a miss: {:?}",
        cold.served
    );
    assert!(
        cold.served.values().all(|from_archive| !from_archive),
        "no kernel can come from an archive that did not exist: {:?}",
        cold.served
    );
    let size = std::fs::metadata(&blob)
        .unwrap_or_else(|e| panic!("no blob at {}: {e}", blob.display()))
        .len();
    assert!(size > 0, "the saved blob is empty");
    eprintln!("blob : {} ({size} bytes)", blob.display());

    // ── 2. WARM. THE GATE. Every pipeline must come out of the archive — not "the file reloaded",
    //      not "the answers still match", but zero back-end compiles. This is the assertion that
    //      fires if the key drifts, the manifest is lost, the binaryArchives binding stops being
    //      passed, or the archive is silently rebuilt from scratch each launch: in every one of
    //      those cases the old test still passed and the cache did nothing.
    let (warm_out, warm) = run_once(Config::default());
    eprintln!("warm : {warm:?}");
    assert!(warm.seeded, "the second run must seed from the blob");
    assert_eq!(
        warm.misses,
        0,
        "every kernel the cold run stored must be found in the reloaded archive — {} of {} were \
         recompiled: {:?}",
        warm.misses,
        warm.served.len(),
        warm.served
            .iter()
            .filter(|(_, from_archive)| !**from_archive)
            .map(|(n, _)| *n)
            .collect::<Vec<_>>(),
    );
    assert_eq!(warm.hits, cold.misses, "the same kernel set, all from disk");
    // The NAMES, not just the count: "warm hits == cold misses" would still hold if the manifest
    // round-tripped a different set of the same size, which is exactly what a truncated name list
    // or a renamed kernel would produce.
    let cold_names: Vec<&str> = cold.served.keys().copied().collect();
    let warm_names: Vec<&str> = warm.served.keys().copied().collect();
    assert_eq!(
        cold_names, warm_names,
        "the warm run must serve the SAME kernels the cold run stored"
    );
    assert_bit_identical(
        "a pipeline created from the archive vs the compiled one",
        &cold_out,
        &warm_out,
    );

    report(&blob, &cold, &warm);

    // The ONE timing assertion — a catastrophic-shape bound, not a performance gate. See
    // `WARM_PSO_CATASTROPHIC_FACTOR` for why it is this loose and why nothing tighter is here.
    let cold_pso_ms = ms(cold.pso);
    let warm_pso_ms = ms(warm.pso);
    let bound_ms = (cold_pso_ms * WARM_PSO_CATASTROPHIC_FACTOR).max(WARM_PSO_FLOOR_MS);
    assert!(
        warm_pso_ms <= bound_ms,
        "creating {} pipelines FROM the archive took {warm_pso_ms:.1} ms against {cold_pso_ms:.1} \
         ms to compile them cold (bound {bound_ms:.1} ms). The cache is not saving the back-end \
         compile, it is paying it twice — check that the manifest still round-trips and that the \
         mid-run save is still debounced.",
        warm.hits,
    );

    // ── 3. A DAMAGED blob must be a miss, never a pipeline. The seam's checksum is what catches
    //      it (tested exhaustively in `infr-core`); what is proven HERE is that the recovery path
    //      through Metal actually runs the backend instead of failing it.
    let mut bytes = std::fs::read(&blob).unwrap();
    let tail = bytes.len() - 16;
    bytes[tail] ^= 0xff;
    std::fs::write(&blob, &bytes).unwrap();
    let (rot_out, rot) = run_once(Config::default());
    eprintln!("rotted: {rot:?}");
    assert!(!rot.seeded, "a bit-rotted blob must not be seeded from");
    assert_eq!(rot.hits, 0, "and nothing may be reported as cached");
    assert_bit_identical(
        "a run recovering from a bit-rotted blob",
        &cold_out,
        &rot_out,
    );

    // ── 4. A blob that is PERFECTLY VALID to the seam but garbage to Metal — the case no checksum
    //      can catch and the one that must not reach the GPU as a pipeline. Corrupt the archive
    //      bytes (the tail of the payload, well past the name manifest) and then repair the
    //      envelope's FNV so the seam hands it over happily.
    //
    //      Envelope: `magic(8) ++ format_version(2) ++ key_len(2) ++ payload_len(8) ++ hash(8)`,
    //      then the key, then the payload — so the hash sits at [20..28] and the payload starts at
    //      `28 + key_len`.
    let mut junk = std::fs::read(&blob).unwrap();
    assert!(junk.len() > 1024, "expected a real blob to mangle");
    let key_len = u16::from_le_bytes(junk[10..12].try_into().unwrap()) as usize;
    let payload_at = 28 + key_len;
    let n = junk.len();
    for b in junk[n - 256..].iter_mut() {
        *b = 0x5a;
    }
    let fixed = infr_core::kernel_cache::fnv1a(&junk[payload_at..]);
    junk[20..28].copy_from_slice(&fixed.to_le_bytes());
    std::fs::write(&blob, &junk).unwrap();
    let (junk_out, junk_stats) = run_once(Config::default());
    // Whether Metal rejects the file outright (expected) or accepts it and finds nothing usable,
    // the ONE thing that may never happen is a different answer.
    eprintln!("junk : {junk_stats:?} (seeded={})", junk_stats.seeded);
    assert_bit_identical(
        "a corrupt-but-checksum-valid archive must never change a kernel's result",
        &cold_out,
        &junk_out,
    );

    // ── 5. TURNED OFF. No archive object, no file touched, identical results — the pre-RM path,
    //      and therefore the REFERENCE every archive-backed pipeline is measured against.
    let before = std::fs::read(&blob).unwrap();
    let mut off = Config::default();
    off.kernels.metal.pipeline_cache = false;
    let (off_out, off_stats) = run_once(off);
    eprintln!("off  : {off_stats:?}");
    assert!(off_stats.blob.is_none(), "a disabled cache has no blob");
    assert!(!off_stats.seeded);
    assert_eq!(off_stats.hits, 0);
    assert_eq!(
        off_stats.served.keys().copied().collect::<Vec<_>>(),
        cold_names,
        "the uncached path must run the same kernels"
    );
    assert!(
        off_stats.served.values().all(|from_archive| !from_archive),
        "a disabled cache cannot serve anything"
    );
    assert_bit_identical("the uncached path is the reference", &cold_out, &off_out);
    // THE numerical-equivalence claim, stated against the two paths that actually differ: `warm`
    // created every one of its pipelines from the persisted archive (asserted `misses == 0`
    // above), `off` compiled every one of its from source with no archive in sight. Same kernels,
    // same inputs, same bits — or the cache is changing results, which no checksum over the blob
    // could ever detect.
    assert_bit_identical(
        "archive-backed pipelines vs freshly compiled ones",
        &warm_out,
        &off_out,
    );
    assert_eq!(
        before,
        std::fs::read(&blob).unwrap(),
        "a disabled cache must not write"
    );

    // ── 6. A key change (a kernel edit, an OS bump, a different GPU) discards the blob WHOLESALE
    //      rather than reusing entries. Simulated by hand-editing the stored key.
    let mut stale = std::fs::read(&blob).unwrap();
    // The envelope is `magic(8) ++ format_version(2) ++ key_len(2) ++ payload_len(8) ++ hash(8)`,
    // then the key: flip the first key byte (the FNV of the MSL source).
    stale[28] ^= 0x01;
    std::fs::write(&blob, &stale).unwrap();
    let (stale_out, stale_stats) = run_once(Config::default());
    eprintln!("stale: {stale_stats:?}");
    assert!(!stale_stats.seeded, "a key mismatch must discard the blob");
    assert_eq!(
        stale_stats.hits, 0,
        "a discarded blob must serve nothing — a key change invalidates WHOLESALE, it does not \
         salvage entries"
    );
    assert_bit_identical("a run after a key change", &cold_out, &stale_out);

    // ...and after all of that, a clean pair of runs still round-trips — same gate as step 2, so
    // that none of the damage above leaves the cache permanently disabled (a `broken` archive that
    // never recovers would sail through every assertion up to here).
    let (_, again_cold) = run_once(Config::default());
    let (again_out, again_warm) = run_once(Config::default());
    assert!(
        again_warm.seeded,
        "the cache must still work: {again_cold:?}"
    );
    assert_eq!(again_warm.misses, 0, "and still serve every kernel warm");
    assert_eq!(
        again_warm.served.keys().copied().collect::<Vec<_>>(),
        cold_names,
        "with the same kernel set it started with"
    );
    assert_bit_identical("the cache after every recovery path", &cold_out, &again_out);

    // No tripwire marker and no temp files may be left behind by a clean run.
    let leftovers: Vec<String> = std::fs::read_dir(blob.parent().unwrap())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n != blob.file_name().unwrap().to_str().unwrap())
        .collect();
    assert!(
        leftovers.is_empty(),
        "a clean run must leave only the blob, found {leftovers:?}"
    );

    std::env::remove_var("XDG_CACHE_HOME");
    let _ = std::fs::remove_dir_all(&dir);
}
