//! On-disk `MTLBinaryArchive` persistence — the Metal-specific SHELL around
//! [`infr_core::kernel_cache::KernelCache`], the same seam `infr-vulkan/src/pcache.rs` and
//! `infr-metal/src/kernels.rs` ride on.
//!
//! # What this caches, and what it does NOT
//!
//! Turning MSL into something a GPU runs is TWO compilers:
//!
//! 1. the **front end**, MSL → AIR (Apple's IR), which `MTLDevice.newLibraryWithSource:` runs over
//!    the ~340 KB [`msl_source`](crate::msl_source) assembles, and
//! 2. the **back end**, AIR → that GPU's ISA, which every `MTLComputePipelineState` creation runs
//!    for one kernel.
//!
//! **Only the back end is cached here.** `MTLLibrary` exposes no way to serialize the AIR it just
//! produced — `newLibraryWithSource:` is the only public entry point that accepts MSL, and nothing
//! on the resulting library hands the compiled module back out. The only supported way to persist
//! a `.metallib` is to compile it AHEAD of time with `xcrun metal`, which needs the Xcode command
//! line tools installed on the user's machine; that is not an acceptable runtime dependency for a
//! CLI, so the front-end compile is paid on every launch and shows up as `library` in the
//! `INFR_PROF` line. If that turns out to dominate, the answer is a build-time `.metallib` in the
//! release artifact, not a runtime cache — a separate slice.
//!
//! What IS cached is every PSO: `MTLBinaryArchive` collects compiled compute pipelines, serializes
//! to a file, and — passed as `MTLComputePipelineDescriptor.binaryArchives` — lets a later launch
//! create those pipelines from the stored ISA instead of re-running the back end.
//!
//! # Shape: the archive GROWS during a run
//!
//! Metal's archive behaves like Vulkan's `vkPipelineCache`, not like a code object: PSOs are
//! created LAZILY (`Pipelines::get` on first use of each kernel), so the artifact is not finished
//! at init. The lifecycle therefore mirrors `infr-vulkan/src/pcache.rs` exactly — seed at
//! [`Pipelines::build`](crate::shaders::Pipelines::build), add each pipeline as it is created, save
//! DEBOUNCED during the run (so a `serve` killed with SIGKILL still leaves a useful blob), save
//! once more on drop, and [`disarm`](KernelCache::disarm) the tripwire on a clean exit.
//!
//! # Bytes through the seam, not a bare file
//!
//! `MTLBinaryArchive` only speaks URLs: it loads from one and serializes to one. The seam is
//! byte-oriented — and the seam is where the envelope, the payload checksum, the atomic+durable
//! write and the poisoned-blob TRIPWIRE live, none of which we want a second, weaker copy of. So
//! the archive is bounced through a temp file at each boundary (serialize → read the bytes →
//! [`KernelCache::store`]; [`KernelCache::load`] → write the bytes → hand Metal that URL) and the
//! temp is removed. The seeding temp outlives the load call deliberately — see [`ArchiveCache`].
//!
//! The payload is not the raw archive: it carries a manifest of the function names inside it, so a
//! launch can tell "already compiled" from "must add" without re-adding (and thus re-compiling)
//! every kernel. That framing, and the cache key, are in [`pcache_blob`](crate::pcache_blob),
//! which is device-free so it can be tested off a Mac.
//!
//! # Every failure is a fallback, never an error
//!
//! A device without binary-archive support, an OS that refuses the archive we saved, a serialize
//! that fails, a blob that fails its checksum, a manifest that will not parse, a pipeline the
//! archive path cannot produce: each one drops back to plain
//! `newComputePipelineStateWithFunction:` and the run continues at cold speed.
//! `MTLPipelineOption::FailOnBinaryArchiveMiss` is deliberately NEVER used — a miss must
//! RECOMPILE, not fail.
//!
//! `kernels.metal.pipeline_cache = false` turns the whole thing off (no directory, no file, no
//! archive object).

use crate::pcache_blob;
use infr_core::config::Config;
use infr_core::kernel_cache::{fnv1a, KernelCache};
use metal::foreign_types::ForeignType;
use metal::{BinaryArchive, BinaryArchiveDescriptor, ComputePipelineDescriptor};
use metal::{ComputePipelineDescriptorRef, ComputePipelineState, Device, FunctionRef};
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use std::collections::BTreeSet;
use std::ffi::{c_char, c_void, CStr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

/// Producer magic + on-disk layout version, in the shared envelope. `INFRMPC1` is the first: the
/// payload is `pcache_blob`'s name manifest followed by the `MTLBinaryArchive` file's bytes. Bump
/// it on any change to what the payload MEANS and every old file is rejected cleanly rather than
/// misread.
const MAGIC: [u8; 8] = *b"INFRMPC1";

/// Debounce for mid-run saves. PSO creation comes in bursts (the first forward walks most of the
/// kernel set), and serializing the archive is not free — one save per burst-second, with the drop
/// save catching the tail. Same value and same reasoning as Vulkan's.
const SAVE_DEBOUNCE_SECS: u64 = 1;

/// How long a leftover `.archive-in`/`.archive-out` temp may sit in the cache dir before a later
/// launch sweeps it. Generous enough that a CONCURRENT run's live temp is never touched (its file
/// is seconds old), short enough that a crashed run leaves no permanent litter.
const TEMP_TTL: Duration = Duration::from_secs(3600);

/// Per-process counter making each temp path unique, so two backends alive in one process (a
/// CPU/Metal parity test, an MTP trunk+head pair) never write the same temp file.
static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Live state behind one lock: what the archive contains, whether it needs saving, and whether we
/// have given up on it. One mutex, never nested, so the debounced save cannot deadlock against a
/// concurrent `get()`.
struct State {
    /// Function names known to be IN the archive — seeded from the blob's manifest and extended as
    /// pipelines are added. Sorted (a `BTreeSet`) so an unchanged run re-serializes byte-identical
    /// manifests instead of churning the file.
    present: BTreeSet<String>,
    /// A pipeline was added since the last save.
    dirty: bool,
    /// Metal refused to add to or serialize this archive. Stop trying: the run is correct either
    /// way, and retrying per-PSO would turn one failure into hundreds.
    broken: bool,
    last_save: Instant,
    /// Size of the archive last written (or the one seeded from), for the `INFR_PROF` line.
    blob_bytes: usize,
    /// Pipelines added to the archive this run (i.e. back-end compiles that the NEXT launch will
    /// not pay).
    added: u64,
}

/// One device's persisted pipeline archive: the shared on-disk cache, the live `MTLBinaryArchive`,
/// and the bookkeeping that keeps the two in step.
pub(crate) struct ArchiveCache {
    disk: KernelCache,
    archive: BinaryArchive,
    /// The blob path, for deriving temp paths. (`disk.path()` is `Some` whenever we exist — a
    /// disabled cache never gets this far — but keeping our own copy avoids re-unwrapping it.)
    blob: PathBuf,
    /// The temp file we handed `newBinaryArchiveWithDescriptor:`, held for the whole run and
    /// removed on drop. Metal does not document whether it reads the archive eagerly or maps it
    /// lazily, and deleting a file it is still mapping is the kind of bug that shows up as a hang
    /// on someone else's machine — so it stays until we are done with the archive.
    seed_temp: Option<PathBuf>,
    /// True when this run started from a blob on disk (for the `INFR_PROF` line).
    pub(crate) seeded: bool,
    state: Mutex<State>,
}

// MTLBinaryArchive is used only under `state`'s lock (adds and serialization are serialized by it);
// the metal-rs wrapper is !Send/!Sync purely because it holds a raw obj-c pointer.
unsafe impl Send for ArchiveCache {}
unsafe impl Sync for ArchiveCache {}

impl ArchiveCache {
    /// Open this device's archive: seeded from `~/.cache/infr/metal-pipelines-<device>.bin` when a
    /// valid blob is there, otherwise empty.
    ///
    /// `None` — meaning "run without a pipeline cache", never an error — when the config disables
    /// it, the device has no binary-archive support, there is no durable cache directory, the
    /// device has no usable name, or Metal will not give us an archive object at all.
    pub(crate) fn open(device: &Device, src: &str, cfg: &Config) -> Option<Self> {
        if !cfg.kernels.metal.pipeline_cache {
            return None;
        }
        if !supports_binary_archives(device) {
            return None;
        }
        // Like Vulkan (and unlike ROCm) this file is REWRITTEN throughout the run, so a temp-dir
        // fallback is not worth it: report "no persistence available" instead.
        infr_core::kernel_cache::cache_dir()?;
        let token = pcache_blob::device_token(device.name());
        if token.is_empty() {
            return None;
        }

        objc::rc::autoreleasepool(|| {
            let key = pcache_blob::KeyInputs {
                src_hash: fnv1a(src.as_bytes()),
                src_len: src.len() as u64,
                device: device.name(),
                architecture: &architecture_name(device),
                os: &os_version(),
                // `Pipelines::build` pins this OFF for CPU numeric parity; see its doc.
                fast_math: false,
            }
            .compose();
            let disk = KernelCache::open(&format!("metal-pipelines-{token}.bin"), MAGIC, key, true);
            let blob = disk.path()?.to_path_buf();
            sweep_stale_temps(&blob);

            // TRIPWIRE + envelope + checksum, all in the seam. A `Some` here means the payload
            // passed every check AND the marker is armed before Metal ever sees the bytes.
            let loaded = disk.load().and_then(|payload| {
                let decoded = pcache_blob::decode(&payload);
                if decoded.is_none() {
                    // Checksum-valid but structurally wrong: a bug on OUR side, not bit-rot. Same
                    // remedy either way.
                    eprintln!(
                        "[infr-metal] the cached pipeline archive's manifest did not parse — \
                         discarding {} and rebuilding.",
                        blob.display()
                    );
                    disk.invalidate();
                }
                decoded
            });

            let (archive, present, seed_temp, was_seeded, blob_bytes) = match loaded {
                Some((names, bytes)) => match seed_archive(device, &blob, &bytes) {
                    Ok((a, tmp)) => (a, names.into_iter().collect(), Some(tmp), true, bytes.len()),
                    Err(e) => {
                        // The OS moved under a key that did not see it, or the archive is subtly
                        // wrong. Recompiling is the correct answer, not a failure — but say so,
                        // because a cache that silently misses every launch is a perf bug nobody
                        // would ever notice.
                        eprintln!(
                            "[infr-metal] the cached pipeline archive was rejected by Metal ({e}) \
                             — discarding it and rebuilding."
                        );
                        disk.invalidate();
                        (empty_archive(device)?, BTreeSet::new(), None, false, 0)
                    }
                },
                None => (empty_archive(device)?, BTreeSet::new(), None, false, 0),
            };

            Some(Self {
                disk,
                archive,
                blob,
                seed_temp,
                seeded: was_seeded,
                state: Mutex::new(State {
                    present,
                    dirty: false,
                    broken: false,
                    last_save: Instant::now(),
                    blob_bytes,
                    added: 0,
                }),
            })
        })
    }

    /// Create `name`'s pipeline THROUGH the archive, and add it if the archive did not already
    /// have it. Returns `(pipeline, served_from_the_archive)`; a `None` pipeline means the caller
    /// must fall back to `newComputePipelineStateWithFunction:` — which is a slow path, not a
    /// failure.
    ///
    /// The descriptor carries nothing but the function and the archive list, so it is equivalent
    /// to `newComputePipelineStateWithFunction:` in every respect that reaches the GPU
    /// (`maxTotalThreadsPerThreadgroup = 0` ⇒ device default,
    /// `threadGroupSizeIsMultipleOfThreadExecutionWidth = NO`). R1: the numerics cannot move.
    ///
    /// `FailOnBinaryArchiveMiss` is NOT passed — on a miss Metal must compile the pipeline, which
    /// is exactly what we want; the manifest, not an error code, tells us whether it was a hit.
    pub(crate) fn pipeline(
        &self,
        device: &Device,
        name: &str,
        func: &FunctionRef,
    ) -> (Option<ComputePipelineState>, bool) {
        objc::rc::autoreleasepool(|| {
            let mut st = self.state.lock().unwrap();
            let hit = st.present.contains(name);

            let desc = ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(func));
            desc.set_binary_archives(&[&self.archive]);
            let pso = match pipeline_from_descriptor(device, &desc) {
                Ok(p) => Some(p),
                Err(e) => {
                    if !st.broken {
                        eprintln!(
                            "[infr-metal] creating a pipeline through the binary archive failed \
                             ({e}) — falling back to uncached pipeline creation for this run."
                        );
                        st.broken = true;
                    }
                    None
                }
            };

            if !hit && !st.broken {
                // A SEPARATE descriptor: `addComputePipelineFunctionsWithDescriptor:` populates an
                // archive, and handing it one that already points at that same archive is not a
                // combination Apple documents.
                let add = ComputePipelineDescriptor::new();
                add.set_compute_function(Some(func));
                match self
                    .archive
                    .add_compute_pipeline_functions_with_descriptor(&add)
                {
                    Ok(true) => {
                        st.present.insert(name.to_string());
                        st.added += 1;
                        st.dirty = true;
                    }
                    other => {
                        let why = match other {
                            Err(e) => e,
                            _ => "refused without an error".to_string(),
                        };
                        eprintln!(
                            "[infr-metal] this Metal device would not add {name} to a binary \
                             archive ({why}) — running without a persisted pipeline cache."
                        );
                        st.broken = true;
                    }
                }
            }
            drop(st);
            self.maybe_save();
            (pso, hit)
        })
    }

    /// Debounced mid-run save — the archive grows lazily, so a long-lived process that never drops
    /// cleanly (a `serve` under SIGKILL) must still leave a useful blob behind.
    fn maybe_save(&self) {
        let mut st = self.state.lock().unwrap();
        if !st.save_due() {
            return;
        }
        self.save_locked(&mut st);
    }

    /// TRIPWIRE step 2 (see [`infr_core::kernel_cache`]): the run is over and we came back alive,
    /// so whatever we seeded from disk did not hang the GPU. Final save, then disarm this
    /// instance's marker. Idempotent — a second call saves nothing and removes an absent file.
    pub(crate) fn finish(&self) {
        let mut st = self.state.lock().unwrap();
        self.save_locked(&mut st);
        drop(st);
        self.disk.disarm();
    }

    /// Serialize the live archive and push the bytes through the seam. Best-effort throughout: a
    /// full disk, a read-only cache dir or a Metal refusal must never fail a run that has perfectly
    /// good pipelines in hand.
    fn save_locked(&self, st: &mut State) {
        if !st.dirty || st.broken {
            return;
        }
        let out = temp_path(&self.blob, "archive-out");
        // `serializeToURL:` will not overwrite, and the path is unique per (process, save) anyway —
        // but a leftover from a previous run with our pid would silently poison every save.
        let _ = std::fs::remove_file(&out);
        let Some(url) = file_url(&out) else {
            st.broken = true;
            return;
        };
        let serialized = objc::rc::autoreleasepool(|| self.archive.serialize_to_url(&url));
        let ok = match serialized {
            Ok(true) => true,
            Ok(false) => false,
            Err(ref e) => {
                eprintln!(
                    "[infr-metal] could not serialize the pipeline archive ({e}) — this run keeps \
                     its pipelines, but the next launch will compile them again."
                );
                false
            }
        };
        if !ok {
            st.broken = true;
            let _ = std::fs::remove_file(&out);
            return;
        }
        match std::fs::read(&out) {
            Ok(bytes) if !bytes.is_empty() => {
                let names: Vec<String> = st.present.iter().cloned().collect();
                // The seam owns the envelope, the checksum, and the atomic+durable rename.
                let _ = self.disk.store(&pcache_blob::encode(&names, &bytes));
                st.blob_bytes = bytes.len();
                st.dirty = false;
                st.last_save = Instant::now();
            }
            // Metal said it wrote the archive and there is nothing there: do not store an empty
            // payload (which `pcache_blob::decode` would reject on the next launch anyway).
            _ => st.broken = true,
        }
        let _ = std::fs::remove_file(&out);
    }

    /// `(pipelines added this run, last blob size, gave up)` — for the `INFR_PROF` summary.
    pub(crate) fn stats(&self) -> (u64, usize, bool) {
        let st = self.state.lock().unwrap();
        (st.added, st.blob_bytes, st.broken)
    }

    /// Where the blob lives, for diagnostics.
    pub(crate) fn path(&self) -> &Path {
        &self.blob
    }
}

impl State {
    /// Is a mid-run save due? Split out so a test can exercise the debounce without a device.
    fn save_due(&self) -> bool {
        self.dirty && !self.broken && self.last_save.elapsed().as_secs() >= SAVE_DEBOUNCE_SECS
    }
}

impl Drop for ArchiveCache {
    fn drop(&mut self) {
        // Belt and braces: `Pipelines::drop` calls this first (so the profile line can report the
        // final blob), and a second call is a no-op.
        self.finish();
        if let Some(t) = self.seed_temp.take() {
            let _ = std::fs::remove_file(t);
        }
    }
}

/// Write `bytes` where Metal can open them and build the archive from that URL. The temp path is
/// RETURNED, not removed: the caller holds it for the archive's lifetime (see
/// [`ArchiveCache::seed_temp`]).
fn seed_archive(
    device: &Device,
    blob: &Path,
    bytes: &[u8],
) -> std::result::Result<(BinaryArchive, PathBuf), String> {
    let tmp = temp_path(blob, "archive-in");
    std::fs::write(&tmp, bytes).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    let url = match file_url(&tmp) {
        Some(u) => u,
        None => {
            let _ = std::fs::remove_file(&tmp);
            return Err("cache path is not valid UTF-8".to_string());
        }
    };
    let desc = BinaryArchiveDescriptor::new();
    desc.set_url(&url);
    match device.new_binary_archive_with_descriptor(&desc) {
        Ok(a) => Ok((a, tmp)),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// `-[MTLDevice newComputePipelineStateWithDescriptor:options:reflection:error:]` — the ONE
/// `MTLDevice` entry point that takes a descriptor, and therefore the only way to pass
/// `binaryArchives`.
///
/// Hand-rolled rather than metal-rs's `Device::new_compute_pipeline_state`, which messages
/// `newComputePipelineStateWithDescriptor:error:` — a selector `MTLDevice` does not declare (the
/// descriptor form has always carried `options:reflection:`). An unrecognized selector raises an
/// Objective-C exception, i.e. an abort on every Mac, and nothing off a Mac can catch that: metal-rs
/// 0.33 is deprecated and its untested bindings are exactly this kind of trap. The other
/// descriptor-taking wrapper, `new_compute_pipeline_state_with_reflection`, sends the right message
/// but then `retain`s and wraps the reflection out-param unconditionally — with
/// [`MTLPipelineOption::None`] that pointer is nil, and `foreign_types` wraps pointers in a
/// `NonNull`. So: the raw call, `options = None`, `reflection = NULL`.
///
/// **`FailOnBinaryArchiveMiss` is deliberately not among the options** — a miss must recompile.
fn pipeline_from_descriptor(
    device: &Device,
    desc: &ComputePipelineDescriptorRef,
) -> std::result::Result<ComputePipelineState, String> {
    let dev: &metal::DeviceRef = device;
    unsafe {
        let mut err: *mut Object = std::ptr::null_mut();
        // `new…` ⇒ the returned object is +1 and owned by us; `ComputePipelineState`'s Drop
        // releases it.
        let pso: *mut Object = msg_send![dev,
            newComputePipelineStateWithDescriptor: desc
            options: metal::MTLPipelineOption::None.bits()
            reflection: std::ptr::null_mut::<*mut Object>()
            error: &mut err];
        if pso.is_null() {
            let why = if err.is_null() {
                "no pipeline and no error".to_string()
            } else {
                let d: *mut Object = msg_send![err, localizedDescription];
                nsstring_to_string(d)
            };
            return Err(why);
        }
        Ok(ComputePipelineState::from_ptr(pso.cast()))
    }
}

/// A fresh, empty archive (a descriptor with no URL). `None` if this device will not make one,
/// which is simply "no pipeline cache this run".
fn empty_archive(device: &Device) -> Option<BinaryArchive> {
    let desc = BinaryArchiveDescriptor::new();
    device.new_binary_archive_with_descriptor(&desc).ok()
}

/// `<cache dir>/<blob stem>.<kind>.<pid>.<seq>` — deliberately NOT of the shape
/// `<stem>.tmp.<pid>.<seq>` (the seam's own save temp) nor `<stem>.seeded.<pid>-<nonce>` (its
/// tripwire marker), so neither the seam's sweep nor ours can mistake one for the other.
fn temp_path(blob: &Path, kind: &str) -> PathBuf {
    blob.with_extension(format!(
        "{kind}.{}.{}",
        std::process::id(),
        TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Remove `.archive-in`/`.archive-out` temps older than [`TEMP_TTL`]. A run that dies between
/// writing a temp and removing it would otherwise leave a copy of the archive in the cache dir
/// forever; an age bound (rather than a liveness probe) keeps this dependency-free and cannot
/// delete a concurrent run's file, which is always seconds old.
fn sweep_stale_temps(blob: &Path) {
    let (Some(dir), Some(stem)) = (blob.parent(), blob.file_stem().and_then(|s| s.to_str())) else {
        return;
    };
    let prefix = format!("{stem}.archive-");
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let name = e.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with(&prefix) {
            continue;
        }
        let stale = e
            .metadata()
            .and_then(|m| m.modified())
            .map(|t| SystemTime::now().duration_since(t).unwrap_or_default() > TEMP_TTL)
            .unwrap_or(false);
        if stale {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

/// A `file://` NSURL for a local path.
///
/// `NSURL fileURLWithPath:` rather than metal-rs's `URL::new_with_string` because the latter goes
/// through `URLWithString:`, which needs an already-percent-encoded absolute URL — a cache dir
/// under a `$HOME` with a space in it would silently produce a nil/relative URL. The result is
/// autoreleased, so it is retained before being wrapped in an owning `metal::URL` (whose `Drop`
/// releases it).
///
/// Carries its OWN autorelease pool: this is reached from `Drop` (the final save) as well as from
/// `pipeline()`, and both the `NSString` and the `NSURL` are autoreleased. The explicit `retain`
/// is what lets the URL outlive the drain.
fn file_url(path: &Path) -> Option<metal::URL> {
    let s = path.to_str()?;
    objc::rc::autoreleasepool(|| unsafe {
        let ns = nsstring(s);
        let url: *mut Object = msg_send![class!(NSURL), fileURLWithPath: ns];
        if url.is_null() {
            return None;
        }
        let url: *mut Object = msg_send![url, retain];
        Some(metal::URL::from_ptr(url.cast()))
    })
}

/// Does this `MTLDevice` do binary archives at all?
///
/// metal-rs 0.33 puts `supports_binary_archive` on `MTLFeatureSet`, not on the device, and the
/// feature-set tables are the legacy (deprecated, iOS-shaped) capability model — so ask the runtime
/// directly. `newBinaryArchiveWithDescriptor:error:` is macOS 11 / iOS 14; on anything older the
/// selector is absent and messaging it would raise an Objective-C exception rather than return an
/// error, which is precisely what `respondsToSelector:` is for.
fn supports_binary_archives(device: &Device) -> bool {
    responds_to(device, sel!(newBinaryArchiveWithDescriptor:error:))
}

/// `-[NSObject respondsToSelector:]`, with the `BOOL` marshalling done properly: `objc`'s `BOOL` is
/// `c_schar` on x86_64 and `bool` on aarch64, so it must be compared against `NO` rather than
/// transmuted into a Rust `bool` (which has exactly two valid bit patterns).
fn responds_to(device: &Device, sel: objc::runtime::Sel) -> bool {
    let dev: &metal::DeviceRef = device;
    let answer: objc::runtime::BOOL = unsafe { msg_send![dev, respondsToSelector: sel] };
    answer != objc::runtime::NO
}

/// `MTLDevice.architecture.name` — the actual codegen target ("applegpu_g14p"), which separates two
/// GPUs that share a marketing name. macOS 14+; an older OS does not answer the selector and gets
/// an empty string, which simply contributes nothing to the key (the OS version, also in the key,
/// covers the upgrade that would introduce it).
fn architecture_name(device: &Device) -> String {
    if !responds_to(device, sel!(architecture)) {
        return String::new();
    }
    let dev: &metal::DeviceRef = device;
    // Self-contained pool: the result is an OWNED String, so nothing here outlives the drain.
    objc::rc::autoreleasepool(|| unsafe {
        let arch: *mut Object = msg_send![dev, architecture];
        if arch.is_null() {
            return String::new();
        }
        let name: *mut Object = msg_send![arch, name];
        nsstring_to_string(name)
    })
}

/// `NSProcessInfo.operatingSystemVersionString`. Apple documents this as human-readable and not for
/// parsing — which is exactly right here: it is hashed verbatim, never interpreted, and only has to
/// CHANGE when the OS (and with it Metal's compiler) does.
fn os_version() -> String {
    objc::rc::autoreleasepool(|| unsafe {
        let info: *mut Object = msg_send![class!(NSProcessInfo), processInfo];
        if info.is_null() {
            return String::new();
        }
        let s: *mut Object = msg_send![info, operatingSystemVersionString];
        nsstring_to_string(s)
    })
}

/// An autoreleased `NSString` from a Rust `&str` (metal-rs keeps its own copy private).
///
/// # Safety
/// Caller must be inside an autorelease pool; the returned object is valid until it drains.
unsafe fn nsstring(s: &str) -> *mut Object {
    /// `NSUTF8StringEncoding`.
    const UTF8: usize = 4;
    let obj: *mut Object = msg_send![class!(NSString), alloc];
    let obj: *mut Object = msg_send![
        obj,
        initWithBytes: s.as_ptr() as *const c_void
        length: s.len()
        encoding: UTF8
    ];
    let _: *mut c_void = msg_send![obj, autorelease];
    obj
}

/// Copy an `NSString`'s UTF-8 out into an owned `String` (empty for nil).
///
/// # Safety
/// `s` must be nil or a valid `NSString`.
unsafe fn nsstring_to_string(s: *mut Object) -> String {
    if s.is_null() {
        return String::new();
    }
    let bytes: *const c_char = msg_send![s, UTF8String];
    if bytes.is_null() {
        return String::new();
    }
    CStr::from_ptr(bytes).to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The payload framing, the key and the device token are `pcache_blob`'s and tested there (off
    /// a Mac, standalone). The envelope, the checksum, the durable write and the tripwire are
    /// `infr_core::kernel_cache`'s and tested there. What is left here that needs no device is the
    /// mid-run save debounce and the temp-path shapes.
    fn state(dirty: bool, broken: bool, age: Duration) -> State {
        State {
            present: BTreeSet::new(),
            dirty,
            broken,
            last_save: Instant::now() - age,
            blob_bytes: 0,
            added: 0,
        }
    }

    /// PSO creation comes in bursts, so a burst must write the file ONCE, not once per pipeline —
    /// and a run that added nothing must not rewrite an identical blob at all.
    #[test]
    fn mid_run_saves_are_debounced() {
        let old = Duration::from_secs(SAVE_DEBOUNCE_SECS + 1);
        assert!(!state(true, false, Duration::ZERO).save_due(), "just saved");
        assert!(state(true, false, old).save_due(), "past the window");
        assert!(!state(false, false, old).save_due(), "nothing new to save");
        assert!(
            !state(true, true, old).save_due(),
            "a broken archive must not be re-serialized"
        );
    }

    /// Our temp files sit in the SAME directory as the seam's own bookkeeping. They must not look
    /// like `<stem>.tmp.<pid>.<seq>` (the seam's save temp) or `<stem>.seeded.<pid>-<nonce>` (its
    /// tripwire marker), or one sweep would eat the other's files — and the tripwire marker is the
    /// one piece of state whose loss reintroduces the poisoned-blob hang.
    #[test]
    fn temp_paths_cannot_be_confused_with_the_seams_own_files() {
        let blob = PathBuf::from("/tmp/infr/metal-pipelines-applem2.bin");
        let a = temp_path(&blob, "archive-in");
        let b = temp_path(&blob, "archive-out");
        let c = temp_path(&blob, "archive-in");
        assert_ne!(a, c, "two temps in one process must differ");
        for t in [&a, &b, &c] {
            let name = t.file_name().unwrap().to_str().unwrap();
            assert!(name.starts_with("metal-pipelines-applem2.archive-"));
            assert!(
                !name.starts_with("metal-pipelines-applem2.seeded."),
                "{name} would be swept as a tripwire marker"
            );
            assert!(
                !name.starts_with("metal-pipelines-applem2.tmp."),
                "{name} collides with the seam's save temp"
            );
            assert_eq!(t.parent(), blob.parent(), "temps live beside the blob");
        }
        // ...and conversely, the seam's files must not match OUR sweep prefix.
        for theirs in [
            "metal-pipelines-applem2.tmp.1.0",
            "metal-pipelines-applem2.seeded.1-0",
        ] {
            assert!(!theirs.starts_with("metal-pipelines-applem2.archive-"));
        }
    }
}
