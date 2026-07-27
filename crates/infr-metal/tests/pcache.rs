//! The persisted `MTLBinaryArchive` pipeline cache, end to end: build → create pipelines → save →
//! reload → create pipelines FROM the archive, with identical kernel results at every step.
//!
//! Everything that can be checked without a GPU is checked without one: the payload framing, the
//! cache key and the device token in `src/pcache_blob.rs` (device-free, runnable standalone off a
//! Mac), the debounce and the temp-path shapes in `src/pcache.rs`, and the envelope + checksum +
//! durable write + poisoned-blob tripwire in `infr_core::kernel_cache`. What is LEFT — and what
//! only a real Metal device can answer — is whether Metal accepts an archive we serialized,
//! whether a pipeline created through it behaves identically to a freshly compiled one, and
//! whether the three degradation paths actually degrade instead of failing.
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

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::config::Config;
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_metal::{MetalBackend, PipelineCacheStats};

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
    let size = std::fs::metadata(&blob)
        .unwrap_or_else(|e| panic!("no blob at {}: {e}", blob.display()))
        .len();
    assert!(size > 0, "the saved blob is empty");
    eprintln!("blob : {} ({size} bytes)", blob.display());

    // ── 2. WARM. The same kernel set must come back from the archive, and compute the same thing.
    let (warm_out, warm) = run_once(Config::default());
    eprintln!("warm : {warm:?}");
    assert!(warm.seeded, "the second run must seed from the blob");
    assert_eq!(
        warm.misses, 0,
        "every kernel the cold run stored must be found in the reloaded archive"
    );
    assert_eq!(warm.hits, cold.misses, "the same kernel set, all from disk");
    assert_eq!(
        cold_out, warm_out,
        "a pipeline created from the archive must compute EXACTLY what the compiled one did"
    );
    // Not an assertion (a fast machine can make both numbers noise) but the number the whole slice
    // exists for — `--nocapture` prints it, and a Mac can compare the two.
    eprintln!(
        "pipeline creation: cold {:.1} ms vs warm {:.1} ms; MSL front end (never cached) cold \
         {:.1} ms vs warm {:.1} ms",
        cold.pso.as_secs_f64() * 1e3,
        warm.pso.as_secs_f64() * 1e3,
        cold.library.as_secs_f64() * 1e3,
        warm.library.as_secs_f64() * 1e3,
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
    assert_eq!(cold_out, rot_out, "the run is correct either way");

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
    assert_eq!(
        cold_out, junk_out,
        "a corrupt archive must never change a kernel's result"
    );

    // ── 5. TURNED OFF. No archive object, no file touched, identical results — the pre-RM path.
    let before = std::fs::read(&blob).unwrap();
    let mut off = Config::default();
    off.kernels.metal.pipeline_cache = false;
    let (off_out, off_stats) = run_once(off);
    eprintln!("off  : {off_stats:?}");
    assert!(off_stats.blob.is_none(), "a disabled cache has no blob");
    assert!(!off_stats.seeded);
    assert_eq!(off_stats.hits, 0);
    assert_eq!(cold_out, off_out, "the uncached path is the reference");
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
    assert_eq!(cold_out, stale_out);

    // ...and after all of that, a clean pair of runs still round-trips.
    let (_, again_cold) = run_once(Config::default());
    let (again_out, again_warm) = run_once(Config::default());
    assert!(
        again_warm.seeded,
        "the cache must still work: {again_cold:?}"
    );
    assert_eq!(again_warm.misses, 0);
    assert_eq!(cold_out, again_out);

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
