//! MSL compute kernels and lazy pipeline-state cache.
//!
//! All kernels operate on `float` (f32) buffers — quantized weights are dequantized to f32 on the
//! host before they reach a kernel, so the shaders stay format-agnostic and simple. The full MSL
//! source is compiled once at backend init; individual `MTLComputePipelineState`s are created on
//! first use and cached by function name.
//!
//! Both stages used to re-run from scratch on every process launch. The per-PSO half no longer
//! does: [`pcache`](crate::pcache) persists the compiled pipelines in an `MTLBinaryArchive` through
//! the shared [`infr_core::kernel_cache`] seam, so a later launch creates them from stored ISA. The
//! MSL → AIR compile above it still runs every time — Metal exposes no way to serialize a library
//! — see that module's doc for why.

use crate::be;
use crate::pcache::ArchiveCache;
use infr_core::config::Config;
use infr_core::error::Result;
use metal::{ComputePipelineState, Device, Library};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub struct Pipelines {
    device: Device,
    library: Library,
    cache: Mutex<HashMap<&'static str, ComputePipelineState>>,
    /// The persisted `MTLBinaryArchive` this backend seeds pipeline creation from, or `None` when
    /// there is no cache this run (`kernels.metal.pipeline_cache = false`, a device without binary
    /// archive support, no writable cache dir). `None` is the pre-RM behaviour exactly.
    archive: Option<ArchiveCache>,
    /// `prof.prof` (`INFR_PROF`): print the compile/cache breakdown on drop.
    prof: bool,
    stats: Mutex<PipelineCacheStats>,
}

/// What the compile + pipeline-cache path did this run.
///
/// **This is the measurement hook.** Metal cannot be run on the Linux dev box, so the evidence that
/// the cache is (or is not) earning its keep has to be readable from a Mac or the macOS CI log —
/// either off the `INFR_PROF` summary this feeds, or programmatically via
/// [`MetalBackend::pipeline_cache_stats`](crate::MetalBackend::pipeline_cache_stats), which is what
/// `tests/pcache.rs` asserts on. Counters are kept unconditionally (two `Instant`s against a
/// pipeline creation that costs milliseconds is free); only the printing is gated.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PipelineCacheStats {
    /// `newLibraryWithSource:` — the MSL → AIR FRONT end, which is **not** cached and is paid on
    /// every launch. See [`crate::pcache`] for why.
    pub library: Duration,
    /// Wall time inside pipeline creation, summed over every kernel first used this run.
    pub pso: Duration,
    /// Pipelines the persisted archive already held — the back-end compile this slice saves.
    pub hits: u64,
    /// Pipelines the archive did not hold: compiled now, and added for the next launch.
    pub misses: u64,
    /// The on-disk blob, or `None` when there is no pipeline cache this run.
    pub blob: Option<PathBuf>,
    /// Was the archive seeded from a blob at init? `false` on a cold machine, after any
    /// key-invalidating change, and whenever the cache is off.
    pub seeded: bool,
    /// Every kernel whose pipeline was CREATED this run (i.e. first use — `get`'s in-process hit
    /// does not appear), mapped to whether the persisted archive served it. `hits`/`misses` are
    /// this map's two value counts.
    ///
    /// The names, not just the totals, because "warm hits == cold misses" would still hold if the
    /// manifest round-tripped the WRONG set — a drifted key, a truncated name list, a kernel added
    /// under a different name. `tests/pcache.rs` compares the two runs' key sets, which is the
    /// assertion that fails when the manifest stops describing the archive.
    pub served: BTreeMap<&'static str, bool>,
}

unsafe impl Send for Pipelines {}
unsafe impl Sync for Pipelines {}

/// The complete assembled MSL source — the ONE string the backend compiles. Public so the
/// kernel-name tripwire test resolves names against exactly what the runtime compiles (a
/// separately-maintained file list in the test would drift the same way a duplicated source
/// copy once did).
pub fn msl_source() -> String {
    // The IQ codebook grids (IQ2/IQ3) are generated from `infr_core::iquant_grids` — the SAME
    // tables the CPU dequant reads, so the native kernels stay bit-exact by construction rather
    // than by hand-transcribing 256..1024-entry tables into MSL. They must land before
    // `linear.metal` (whose DEC16_IQ2XXS etc reference them): common + norms, then grids, then
    // the rest in order.
    let mut s = String::with_capacity(256 * 1024);
    s.push_str(MSL_PARTS[0]);
    s.push_str(MSL_PARTS[1]);
    s.push_str(&iquant_grids_msl());
    for part in &MSL_PARTS[2..] {
        s.push_str(part);
    }
    s
}

/// Emit an MSL `constant <ty> NAME[N] = { ... };` from a Rust static, `sfx` the integer-literal
/// suffix that pins the element type (`ul` for ulong/u64, `u` for uint/u32, empty for uchar/u8).
fn emit_grid<T: std::fmt::Display>(s: &mut String, ty: &str, name: &str, arr: &[T], sfx: &str) {
    use std::fmt::Write;
    write!(s, "constant {ty} {name}[{}] = {{", arr.len()).unwrap();
    for (i, v) in arr.iter().enumerate() {
        if i % 8 == 0 {
            s.push('\n');
        }
        write!(s, "{v}{sfx},").unwrap();
    }
    s.push_str("\n};\n");
}

/// Emit the IQ codebook grid + sign tables as MSL `constant` arrays, formatted from the Rust
/// statics in `infr_core::iquant_grids` so there is exactly one copy of each table.
fn iquant_grids_msl() -> String {
    use infr_core::iquant_grids as ig;
    let mut s =
        String::from("// Auto-generated from infr_core::iquant_grids (single source of truth).\n");
    emit_grid(&mut s, "uchar", "KSIGNS_IQ2XS", &ig::KSIGNS_IQ2XS, "");
    emit_grid(&mut s, "ulong", "IQ2XXS_GRID", &ig::IQ2XXS_GRID, "ul");
    emit_grid(&mut s, "ulong", "IQ2XS_GRID", &ig::IQ2XS_GRID, "ul");
    emit_grid(&mut s, "ulong", "IQ2S_GRID", &ig::IQ2S_GRID, "ul");
    emit_grid(&mut s, "uint", "IQ3XXS_GRID", &ig::IQ3XXS_GRID, "u");
    emit_grid(&mut s, "uint", "IQ3S_GRID", &ig::IQ3S_GRID, "u");
    // IQ1_S / IQ1_M share the 2048-entry IQ1S_GRID (u64: 8 signed i8 each).
    emit_grid(&mut s, "ulong", "IQ1S_GRID", &ig::IQ1S_GRID, "ul");
    s
}

impl Pipelines {
    /// Compile the MSL library and open this device's persisted pipeline archive.
    ///
    /// `cfg` is the backend's [`Config`] (`kernels.metal.pipeline_cache` gates the archive,
    /// `prof.prof` the timing line) — handed in rather than read from the environment, like every
    /// other knob this backend takes.
    pub fn build(device: &Device, cfg: &Config) -> Result<Self> {
        let opts = metal::CompileOptions::new();
        // Reference backend: prefer accurate transcendentals (sin/cos/tanh) over fast intrinsics so
        // results stay in tight numeric parity with the CPU interpreter. This is load-bearing for
        // R1 parity AND is folded into the pipeline cache's key — an archive built with fast-math
        // on must never be reloaded here.
        opts.set_fast_math_enabled(false);
        let src = msl_source();
        let t0 = Instant::now();
        let library = device
            .new_library_with_source(&src, &opts)
            .map_err(|e| be(format!("compile MSL library: {e}")))?;
        let lib_wall = t0.elapsed();

        let archive = ArchiveCache::open(device, &src, cfg);
        let prof = cfg.prof.prof;
        if prof {
            match archive.as_ref() {
                Some(a) => eprintln!(
                    "[infr-metal] MSL library compiled in {:.1} ms ({} KiB source); pipeline \
                     archive {} ({})",
                    lib_wall.as_secs_f64() * 1e3,
                    src.len() / 1024,
                    if a.seeded {
                        "SEEDED from disk"
                    } else {
                        "empty (cold)"
                    },
                    a.path().display(),
                ),
                None => eprintln!(
                    "[infr-metal] MSL library compiled in {:.1} ms ({} KiB source); pipeline \
                     archive DISABLED (config off, no binary-archive support, or no cache dir)",
                    lib_wall.as_secs_f64() * 1e3,
                    src.len() / 1024,
                ),
            }
        }
        Ok(Self {
            device: device.clone(),
            library,
            cache: Mutex::new(HashMap::new()),
            prof,
            stats: Mutex::new(PipelineCacheStats {
                library: lib_wall,
                blob: archive.as_ref().map(|a| a.path().to_path_buf()),
                seeded: archive.as_ref().is_some_and(|a| a.seeded),
                ..Default::default()
            }),
            archive,
        })
    }

    /// A snapshot of [`PipelineCacheStats`] — the measurement/CI hook; see that type.
    pub fn cache_stats(&self) -> PipelineCacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get (creating + caching on first use) the compute pipeline for an MSL kernel function.
    ///
    /// When a pipeline archive is live the state is created from a descriptor carrying it, so a
    /// kernel compiled by an earlier launch comes back without re-running the driver's back end;
    /// a miss compiles as before and is added to the archive for next time. EVERY archive failure
    /// falls through to the original `newComputePipelineStateWithFunction:` call — the cache can
    /// cost a cold start, never a run.
    pub fn get(&self, name: &'static str) -> Result<ComputePipelineState> {
        if let Some(p) = self.cache.lock().unwrap().get(name) {
            return Ok(p.clone());
        }
        let func = self
            .library
            .get_function(name, None)
            .map_err(|e| be(format!("get MSL function {name}: {e}")))?;
        let t0 = Instant::now();
        let (cached, hit) = match self.archive.as_ref() {
            Some(a) => a.pipeline(&self.device, name, &func),
            None => (None, false),
        };
        let pso = match cached {
            Some(p) => p,
            None => self
                .device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| be(format!("pipeline for {name}: {e}")))?,
        };
        {
            let mut s = self.stats.lock().unwrap();
            s.pso += t0.elapsed();
            if hit {
                s.hits += 1;
            } else {
                s.misses += 1;
            }
            s.served.insert(name, hit);
        }
        self.cache.lock().unwrap().insert(name, pso.clone());
        Ok(pso)
    }
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        // Final save + TRIPWIRE disarm BEFORE the summary, so the reported blob size is the one
        // this run actually left on disk. `ArchiveCache::finish` is idempotent (its own `Drop`
        // calls it too, for the paths that never get here).
        if let Some(a) = self.archive.as_ref() {
            a.finish();
        }
        if !self.prof {
            return;
        }
        let s = self.stats.lock().unwrap();
        let (added, blob, broken) = match self.archive.as_ref() {
            Some(a) => a.stats(),
            None => (0, 0, false),
        };
        eprintln!(
            "── infr-metal pipelines: library {:.1} ms (front end, NOT cached) │ {} PSOs in \
             {:.1} ms ({} from archive, {} compiled) │ archive +{added}, blob {:.1} KiB{}",
            s.library.as_secs_f64() * 1e3,
            s.hits + s.misses,
            s.pso.as_secs_f64() * 1e3,
            s.hits,
            s.misses,
            blob as f64 / 1024.0,
            if broken {
                " (ARCHIVE DISABLED mid-run)"
            } else {
                ""
            },
        );
    }
}

/// Metal Shading Language source for every kernel, split into domain files under `shaders/`
/// (see each file's header). Concatenated IN ORDER into one string so it compiles as a single
/// library — MSL requires define-before-use, so the file order here is load-bearing (helpers and
/// constant tables in earlier files are referenced by later ones). `include_str!` makes cargo
/// track the files for rebuilds automatically.
///
/// There is deliberately NO other copy of the shader source: an embedded-string duplicate once
/// drifted (the string was restored in a rebase while new kernels landed only in the files),
/// which silently disabled every kernel that existed only in the non-live copy — the pipeline
/// cap-checks treat a missing function as "capability absent" and fall back, so nothing errors.
const MSL_PARTS: [&str; 9] = [
    include_str!("../shaders/common.metal"),
    include_str!("../shaders/elementwise_norms.metal"),
    include_str!("../shaders/linear.metal"),
    include_str!("../shaders/moe.metal"),
    include_str!("../shaders/rope_ffn.metal"),
    include_str!("../shaders/attention.metal"),
    include_str!("../shaders/deltanet.metal"),
    include_str!("../shaders/kv_cache.metal"),
    // Instantiates linear.metal's DEC16_* decode macros — must come after it.
    include_str!("../shaders/embed_gather.metal"),
];
