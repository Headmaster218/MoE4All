//! A shared on-disk cache for COMPILED KERNEL ARTIFACTS — the backend-agnostic half of what
//! `infr-vulkan/src/pcache.rs` grew for `vkPipelineCache`, now also carrying the ROCm hiprtc code
//! object.
//!
//! Every GPU backend pays the same tax: turning our kernel source into machine code is expensive,
//! one-time, and re-runs on every process launch unless somebody writes the result down. Measured
//! on this box: ~5 s of `vkCreateComputePipelines` across a cold DG forward's kernel set (Vulkan),
//! and ~9.2 s of `hiprtcCompileProgram` on a cold comgr cache — plus ~0.25 s of hiprtc even when
//! comgr's own lower-level cache is hot, on EVERY launch (ROCm). The artifacts differ; the
//! bookkeeping around them does not, and that bookkeeping is the part that is easy to get subtly,
//! dangerously wrong. So it lives here, once.
//!
//! # What a cached artifact is allowed to do
//!
//! Nothing here is a general-purpose blob store. What lands in these files is MACHINE CODE that we
//! hand straight back to a driver, and every driver's contract for "cached binaries" is the same:
//! data it cannot validate is allowed to produce UNDEFINED BEHAVIOR. On a GPU that reads as a hung
//! ring, not a clean error. **A cache must never be able to produce a wrong result silently** — so
//! the rule is that a damaged, stale, or merely SUSPECT file must die HERE, at the cost of one cold
//! recompile, rather than reach the device.
//!
//! Three layers enforce that, and a fourth catches what none of them can:
//!
//! - **The KEY.** [`open`](KernelCache::open) takes an opaque byte string that is compared
//!   VERBATIM on load. The caller folds in everything that would make an old artifact wrong:
//!   Vulkan carries `driverVersion ++ pipelineCacheUUID ++ SHADER_SET_FINGERPRINT`, ROCm carries
//!   `FNV(hip_source()) ++ gfx arch ++ HIP runtime/driver version ++ compile options`. A mismatch
//!   discards the file WHOLESALE, which is also why the key is not a hash we compute for the
//!   caller: only the caller knows what its artifact depends on.
//! - **The ENVELOPE.** `magic ++ format_version ++ key_len ++ payload_len ++ payload_hash`, so a
//!   file from another producer, an older layout, or a truncated write is rejected on sight. The
//!   `magic` is the producer's identity AND its layout version — bump it and every old file is
//!   rejected cleanly instead of being misread.
//! - **The CHECKSUM.** FNV-1a over the payload, verified on load, because `rename` is atomic only
//!   with respect to a concurrent READER: on a crash/power-loss it can publish a name over data
//!   blocks that were never flushed (ext4 delayed allocation), i.e. a valid-looking file over
//!   garbage. [`store`](KernelCache::store) `fsync`s to keep that from happening at all; the
//!   checksum is what catches it if it happens anyway.
//! - **The TRIPWIRE** — see below.
//!
//! # Durability
//!
//! [`store`](KernelCache::store) writes to a UNIQUE per-save temp path (`<path>.tmp.<pid>.<seq>`,
//! from a monotonic per-process counter, so two concurrent saves never share a path and neither
//! clobbers the other's in-flight bytes), `fsync`s the file, `rename`s it over the target, then
//! `fsync`s the DIRECTORY so the new entry survives a crash too.
//!
//! # ── THE TRIPWIRE ─────────────────────────────────────────────────────────────────────────────
//!
//! None of the above can catch the failure that actually happened: a blob that is perfectly
//! WELL-FORMED but contains a kernel binary that HANGS THE GPU. (Real incident: a hung
//! `native_idm_q5k_sg2` sat in the Vulkan pipeline cache and was re-seeded on every launch, so a
//! 35B MoE returned all-zero logits or a device-lost — surviving reboots, reproducing at CI-green
//! commits, and ignoring every code knob, because the poison was on DISK, not in the tree.) A
//! checksum sees only the bytes we wrote, and they are exactly the bytes we wrote.
//!
//! So instead of validating the CONTENT, watch what HAPPENS after we hand it to the driver — the
//! same dirty-bit trick a filesystem uses:
//!
//! 1. When a run seeds a driver from a loaded blob, drop a marker file next to it
//!    ([`load`](KernelCache::load) arms it BEFORE returning the payload, so a hang in the very call
//!    that consumes it is still attributable).
//! 2. On a clean exit, delete the marker ([`disarm`](KernelCache::disarm)).
//! 3. If a marker from a DEAD process is found at startup, that run seeded from this blob and then
//!    died without a clean exit. The blob is not trustworthy: delete it and recompile.
//!
//! That closes the loop the incident was stuck in — a poisoned blob hangs the GPU, the run dies
//! with a lost device, the marker survives, and the NEXT launch throws the blob away. One slow
//! start instead of a hang that reproduces forever.
//!
//! The false positive is deliberate and cheap: any unclean death (a crash, a power cut, SIGKILL)
//! discards a perfectly good cache and costs one cold compile. The asymmetry is total — the other
//! mistake is an undebuggable GPU hang, so we take the recompile every time.
//!
//! Markers are keyed PER-INSTANCE (`<path>.seeded.<pid>-<nonce>`) and a marker is only stale if its
//! PROCESS is GONE, so neither a second `infr` running concurrently (a `serve` alongside a CLI run)
//! NOR a sibling backend in the same process (an MTP bench loop, a Vulkan+ROCm parity run) is
//! mistaken for a crashed one — and one instance's clean [`disarm`](KernelCache::disarm) only
//! removes ITS OWN marker, never a live sibling's.
//!
//! # Disabled
//!
//! A cache opened with `enabled = false` (`kernels.vulkan.pipeline_cache_disk = false`,
//! `kernels.rocm.module_cache = false`) is a TOTAL no-op: it never creates the cache directory,
//! never reads, never writes, never sweeps. Every method returns the miss/OK answer immediately.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic per-process sequence for save temp files (`<path>.tmp.<pid>.<seq>`). Two threads
/// saving concurrently (a `serve` compiling different kernels) must never write+rename the SAME
/// temp path, or one's write/fsync races the other's rename → a torn write or a failed rename.
static SAVE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Monotonic per-process sequence handed to each [`KernelCache`] as its tripwire-marker nonce.
/// Two caches over the SAME file in ONE process (an `infr bench` MTP loop, a CPU/GPU parity run)
/// must get DISTINCT markers: keying the marker by PID alone let the first instance's clean
/// `disarm()` delete the shared marker while a sibling was still live, so a later unclean death
/// then left a poisoned blob uncaught — the exact case the tripwire exists for.
static INSTANCE_NONCE: AtomicU64 = AtomicU64::new(0);

/// Envelope layout version, INSIDE the producer's `magic`. A producer that changes its payload
/// meaning bumps its own magic (`INFRVPC2` → `INFRVPC3`); this bumps only if the header itself
/// changes shape, which would invalidate every producer at once.
const FORMAT_VERSION: u16 = 1;

/// `magic(8) ++ format_version(2) ++ key_len(2) ++ payload_len(8) ++ payload_hash(8)`, followed by
/// the key bytes and then the payload.
const HEADER_LEN: usize = 28;

/// Suffix for a tripwire marker: `<path stem>.seeded.<pid>-<nonce>`. See the module doc.
const MARKER_EXT: &str = "seeded";

/// FNV-1a over the blob — the same hash `infr-vulkan/build.rs` uses for `SHADER_SET_FINGERPRINT`
/// (which it must recompute standalone: a build script cannot depend on the crate graph it builds).
/// Not a cryptographic checksum and not meant to be one: it guards against truncation/bit-rot on a
/// file only this process family writes, not against an adversary who can already write to `$HOME`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// `$XDG_CACHE_HOME/infr` → `$HOME/.cache/infr`, or `None` when neither is set. Callers that want
/// a cache no matter what get the temp-dir fallback [`KernelCache::open`] applies; callers that
/// want to report "no persistence available" (Vulkan's `pipeline_cache_disk` probe) ask here.
///
/// Does NOT create the directory — [`KernelCache::open`] does that, and only when enabled.
pub fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("infr"))
}

/// Is `pid` still running? `kill(pid, 0)` delivers no signal and only asks the kernel whether the
/// process exists: `Ok` = alive, `EPERM` = alive but not ours (a foreign process reusing the pid —
/// treat as alive, i.e. do NOT discard the cache on it), anything else (`ESRCH`) = gone.
#[cfg(unix)]
fn pid_alive(pid: i32) -> bool {
    // SAFETY: `kill` with signal 0 performs only an existence/permission check.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Without a cheap liveness probe the tripwire cannot tell a corpse from a concurrent run, and
/// guessing WRONG discards a live process's cache. Report "alive" — the sweep then never fires,
/// which loses the tripwire but never misbehaves. (No such target ships today; this keeps
/// `infr-core` buildable everywhere.)
#[cfg(not(unix))]
fn pid_alive(_pid: i32) -> bool {
    true
}

/// One persisted compiled-artifact file: where it lives, what key it must match, and this
/// instance's tripwire identity. See the module doc — the invariants are the point of this type.
pub struct KernelCache {
    /// `None` ⇔ disabled. Every method short-circuits, so a disabled cache never touches the FS.
    path: Option<PathBuf>,
    /// Producer identity + on-disk layout version (e.g. `INFRVPC3`, `INFRRMC1`).
    magic: [u8; 8],
    /// Compared VERBATIM on load; a mismatch discards the file. See the module doc.
    key: Vec<u8>,
    /// Per-instance nonce (from [`INSTANCE_NONCE`]) folded into this instance's marker name so
    /// sibling caches in one process own DISTINCT markers.
    nonce: u64,
}

impl KernelCache {
    /// Open the per-device artifact file `<cache dir>/<file_name>`.
    ///
    /// `file_name` must be unique per DEVICE-CLASS (Vulkan keys it by `vendor_id`/`device_id`,
    /// ROCm by gfx arch) so a multi-GPU box never hands one device another's binaries. `magic`
    /// identifies the producer AND its layout version. `key` is whatever makes an old artifact
    /// wrong; it is compared verbatim, never interpreted.
    ///
    /// `enabled = false` yields a total no-op — no directory is created, nothing is read or
    /// written.
    pub fn open(file_name: &str, magic: [u8; 8], key: Vec<u8>, enabled: bool) -> Self {
        let path = enabled
            .then(|| {
                // A box with neither XDG_CACHE_HOME nor HOME (a container, a systemd unit) still
                // gets a cache: the temp dir is not durable across reboots, but a recompile saved
                // within one boot is the whole point.
                let dir = cache_dir().unwrap_or_else(|| std::env::temp_dir().join("infr"));
                std::fs::create_dir_all(&dir).ok()?;
                Some(dir.join(file_name))
            })
            .flatten();
        Self {
            path,
            magic,
            key,
            nonce: INSTANCE_NONCE.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// The file this cache reads and writes, or `None` when disabled. For diagnostics.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Read + validate the persisted artifact. Any mismatch (magic, format version, key,
    /// truncation, or a payload that fails its checksum) is a MISS — the stale/damaged file is
    /// replaced by the next [`store`](Self::store), at the cost of one cold compile.
    ///
    /// TRIPWIRE steps 1+3: sweeps dead processes' markers FIRST (which may discard the blob), and
    /// ARMS this instance's marker before handing the payload back, so a hang inside the driver
    /// call that consumes it is attributable on the next launch.
    pub fn load(&self) -> Option<Vec<u8>> {
        let path = self.path.as_ref()?;
        self.discard_if_a_seeded_run_died();
        let payload = self.load_validated()?;
        // Armed BEFORE the payload reaches the driver — if THAT call is what hangs the GPU, the
        // marker has to already be on disk for the next launch to find.
        let _ = std::fs::File::create(self.marker(path));
        Some(payload)
    }

    /// Write `payload` atomically AND durably: temp file → `fsync` → `rename` → `fsync` the
    /// directory. See the module doc for why plain `write` + `rename` is not enough.
    ///
    /// `Ok(())` on a disabled cache (nothing to do). A failed write leaves the previous file — and
    /// the temp file — untouched/removed, never a half-written target.
    pub fn store(&self, payload: &[u8]) -> std::io::Result<()> {
        let Some(path) = self.path.as_ref() else {
            return Ok(());
        };
        let key_len: u16 = self.key.len().try_into().map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "kernel-cache key exceeds 65535 bytes",
            )
        })?;
        let mut out = Vec::with_capacity(HEADER_LEN + self.key.len() + payload.len());
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&key_len.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&fnv1a(payload).to_le_bytes());
        out.extend_from_slice(&self.key);
        out.extend_from_slice(payload);

        let tmp = tmp_path(path);
        match write_durable(&tmp, &out).and_then(|()| std::fs::rename(&tmp, path)) {
            Ok(()) => {
                // The rename itself is a directory metadata change: sync the directory so the new
                // entry survives a crash too (the payload it points at is already on disk).
                if let Some(dir) = path.parent() {
                    if let Ok(d) = std::fs::File::open(dir) {
                        let _ = d.sync_all();
                    }
                }
                Ok(())
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                Err(e)
            }
        }
    }

    /// TRIPWIRE step 2: this run seeded from the blob and came back ALIVE. Removes THIS instance's
    /// marker and no other — a live sibling's marker must survive (see the module doc).
    pub fn disarm(&self) {
        if let Some(path) = self.path.as_ref() {
            let _ = std::fs::remove_file(self.marker(path));
        }
    }

    /// The driver REJECTED this artifact (`vkCreatePipelineCache` refused it, `hipModuleLoadData`
    /// returned an error), or this run lost the device: delete the file. Not an error — the caller
    /// compiles from source and stores the fresh result. This instance's marker goes too: it
    /// accuses a file that no longer exists, and leaving it would cost the NEXT run its cache.
    pub fn invalidate(&self) {
        if let Some(path) = self.path.as_ref() {
            let _ = std::fs::remove_file(path);
        }
        self.disarm();
    }

    /// This instance's tripwire marker: `<path stem>.seeded.<pid>-<nonce>`.
    fn marker(&self, path: &Path) -> PathBuf {
        path.with_extension(format!(
            "{MARKER_EXT}.{}-{}",
            std::process::id(),
            self.nonce
        ))
    }

    /// TRIPWIRE step 3 (see the module doc): a marker left behind by a process that is GONE means
    /// that run seeded from this blob and then died without a clean exit — a hung GPU is exactly
    /// how that happens, and the blob is the prime suspect. Discard it and recompile.
    ///
    /// Markers whose process is still ALIVE belong to a concurrently-running `infr` (or a sibling
    /// backend here) and are left alone. Returns true when the blob was discarded (for the tests;
    /// the caller does not care — a discarded blob simply reads as a miss).
    fn discard_if_a_seeded_run_died(&self) -> bool {
        let Some(path) = self.path.as_ref() else {
            return false;
        };
        let (Some(dir), Some(stem)) = (path.parent(), path.file_stem().and_then(|s| s.to_str()))
        else {
            return false;
        };
        // `foo.bin` → markers are `foo.seeded.<pid>-<nonce>` (`with_extension` replaced `.bin`).
        let prefix = format!("{stem}.{MARKER_EXT}.");
        let Ok(entries) = std::fs::read_dir(dir) else {
            return false;
        };
        let mut dirty = false;
        for e in entries.flatten() {
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(tag) = name.strip_prefix(&prefix) else {
                continue;
            };
            // `<pid>-<nonce>`: the pid is what determines liveness; the nonce only distinguishes
            // sibling instances of the same pid. (`split` also accepts a legacy bare `<pid>`.)
            let Ok(pid) = tag.split('-').next().unwrap_or(tag).parse::<i32>() else {
                continue;
            };
            if pid_alive(pid) {
                continue; // a concurrent infr, not a corpse
            }
            let _ = std::fs::remove_file(e.path());
            dirty = true;
        }
        if dirty {
            eprintln!(
                "[infr] a previous run seeded the GPU kernel cache and then died without a clean \
                 exit (a hung GPU does exactly that, and a cached kernel binary is the prime \
                 suspect) — discarding {} and recompiling. This costs one slow start.",
                path.display()
            );
            let _ = std::fs::remove_file(path);
        }
        dirty
    }

    /// The envelope checks alone (no tripwire) — split out so the tests can exercise them without
    /// a live process's marker bookkeeping.
    fn load_validated(&self) -> Option<Vec<u8>> {
        let path = self.path.as_ref()?;
        let data = std::fs::read(path).ok()?;
        if data.len() < HEADER_LEN || data[..8] != self.magic {
            return None;
        }
        let ver = u16::from_le_bytes(data[8..10].try_into().unwrap());
        let key_len = u16::from_le_bytes(data[10..12].try_into().unwrap()) as usize;
        let payload_len = u64::from_le_bytes(data[12..20].try_into().unwrap());
        let sum = u64::from_le_bytes(data[20..28].try_into().unwrap());
        // `payload_len` is attacker-free but not trust-free (bit-rot in the HEADER lands here):
        // compute the expected size with checked arithmetic so a garbage length is a clean miss
        // rather than a wrapped index.
        let want = usize::try_from(payload_len)
            .ok()
            .and_then(|p| HEADER_LEN.checked_add(key_len)?.checked_add(p))?;
        if ver != FORMAT_VERSION || data.len() != want {
            return None;
        }
        if data[HEADER_LEN..HEADER_LEN + key_len] != self.key[..] {
            return None;
        }
        let payload = &data[HEADER_LEN + key_len..];
        if fnv1a(payload) != sum {
            // Damaged file: never hand it to the driver (invalid cache data is explicitly
            // undefined behavior, and on a GPU that reads as a hung ring rather than an error).
            // Drop it and let this launch rebuild.
            eprintln!(
                "[infr] kernel cache {} failed its checksum — discarding and rebuilding",
                path.display()
            );
            let _ = std::fs::remove_file(path);
            return None;
        }
        Some(payload.to_vec())
    }
}

/// A UNIQUE temp path for one [`KernelCache::store`]: `<path>.tmp.<pid>.<seq>`. The per-process
/// [`SAVE_SEQ`] makes two concurrent saves in one process write DISTINCT files, so neither clobbers
/// the other's in-flight bytes before its own rename. A free fn so the tests can assert distinct
/// names without an instance.
fn tmp_path(path: &Path) -> PathBuf {
    path.with_extension(format!(
        "tmp.{}.{}",
        std::process::id(),
        SAVE_SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

/// `fs::write` + `fsync`: the bytes are on the platter before the caller renames over the target.
fn write_durable(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(bytes)?;
    f.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAGIC: [u8; 8] = *b"INFRTST1";

    /// A cache rooted in a caller-chosen directory — the real [`KernelCache::open`] resolves
    /// `$XDG_CACHE_HOME`/`$HOME`, and a test must never write to the user's real cache (nor mutate
    /// process env, which races every other test in the binary).
    fn cache_in(dir: &Path, key: &[u8]) -> KernelCache {
        std::fs::create_dir_all(dir).unwrap();
        KernelCache {
            path: Some(dir.join("artifact.bin")),
            magic: MAGIC,
            key: key.to_vec(),
            nonce: INSTANCE_NONCE.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// A private temp dir for one test (removed at the end).
    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("infr-kcache-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A pid that is guaranteed DEAD (spawned and reaped). Reusing a just-reaped pid is
    /// theoretically possible but the kernel hands out pids sequentially, so it will not happen
    /// within one test.
    fn spawn_and_reap() -> u32 {
        let mut c = std::process::Command::new("true")
            .spawn()
            .expect("spawn /bin/true");
        let pid = c.id();
        c.wait().expect("reap");
        assert!(!pid_alive(pid as i32), "the reaped child must be gone");
        pid
    }

    /// The envelope round-trips a payload of every shape, and every way a file can be damaged or
    /// stale is REJECTED rather than handed to a driver — where invalid cache data is undefined
    /// behavior, i.e. a hung GPU.
    #[test]
    fn envelope_round_trips_and_rejects_damage() {
        let dir = tmp_dir("envelope");
        let c = cache_in(&dir, b"key-v1");
        let path = c.path().unwrap().to_path_buf();

        // Round-trip, including the degenerate empty payload.
        for len in [0usize, 1, 4096] {
            let payload: Vec<u8> = (0u8..=255).cycle().take(len).collect();
            c.store(&payload).unwrap();
            assert_eq!(c.load().as_deref(), Some(&payload[..]), "round-trip @{len}");
        }
        let payload: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        c.store(&payload).unwrap();
        let good = std::fs::read(&path).unwrap();
        assert_eq!(
            good.len(),
            HEADER_LEN + 6 + payload.len(),
            "header ++ key ++ payload, nothing else"
        );

        // A single flipped PAYLOAD byte must fail the checksum — and the file is deleted, not kept.
        let mut rot = good.clone();
        rot[HEADER_LEN + 6 + 100] ^= 0x01;
        std::fs::write(&path, &rot).unwrap();
        assert!(c.load().is_none(), "bit-rotted payload must be rejected");
        assert!(!path.exists(), "a damaged file is deleted, not kept");

        // Truncation (the tail never reached disk), and a header-only stub.
        std::fs::write(&path, &good[..good.len() - 64]).unwrap();
        assert!(c.load().is_none(), "truncated blob must be rejected");
        std::fs::write(&path, &good[..HEADER_LEN - 1]).unwrap();
        assert!(c.load().is_none(), "a sub-header stub must be rejected");

        // A garbage payload_len must be a clean miss, not a panic (checked arithmetic).
        let mut huge = good.clone();
        huge[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, &huge).unwrap();
        assert!(c.load().is_none(), "absurd payload_len must be rejected");

        // Wrong producer magic (another backend's file, or an older layout of ours).
        let mut foreign = good.clone();
        foreign[..8].copy_from_slice(b"INFRXXX9");
        std::fs::write(&path, &foreign).unwrap();
        assert!(c.load().is_none(), "foreign magic must be rejected");

        // Wrong envelope format version.
        let mut oldver = good.clone();
        oldver[8..10].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        std::fs::write(&path, &oldver).unwrap();
        assert!(
            c.load().is_none(),
            "a future/past envelope must be rejected"
        );

        // Wrong KEY: same file, a cache whose key moved on (a driver bump, a source change).
        std::fs::write(&path, &good).unwrap();
        let moved = cache_in(&dir, b"key-v2");
        assert!(
            moved.load().is_none(),
            "a key mismatch must discard the artifact"
        );
        // Same LENGTH, different bytes — the compare is verbatim, not a length check.
        let same_len = cache_in(&dir, b"key-v9");
        assert!(same_len.load().is_none(), "key compare must be byte-exact");
        // ...and the original key still loads it (a rejection must not delete a valid file).
        assert_eq!(
            c.load().as_deref(),
            Some(&payload[..]),
            "still valid for us"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// THE TRIPWIRE. A blob that hangs the GPU is perfectly well-formed, so no envelope check can
    /// see it — the only evidence is that the run which SEEDED from it never came back.
    #[test]
    fn tripwire_discards_a_blob_whose_seeded_run_died() {
        let dir = tmp_dir("tripwire");
        let c = cache_in(&dir, b"k");
        let path = c.path().unwrap().to_path_buf();
        let payload: Vec<u8> = (0u8..=255).cycle().take(2048).collect();
        let marker_for = |pid: u32| path.with_extension(format!("{MARKER_EXT}.{pid}-0"));

        // 1. No marker: an ordinary run loads the blob — and ARMS its own marker before handing
        //    the payload to the driver, so a hang from here on is attributable.
        c.store(&payload).unwrap();
        assert_eq!(c.load().as_deref(), Some(&payload[..]), "clean blob loads");
        assert!(c.marker(&path).exists(), "loading must arm the tripwire");

        // 2. A marker from a DEAD process: that run seeded from this blob and never exited
        //    cleanly. The blob is the suspect — discard it, and the load reports a cache miss.
        let dead = spawn_and_reap();
        std::fs::write(marker_for(dead), b"").unwrap();
        c.store(&payload).unwrap();
        assert!(
            c.load().is_none(),
            "a blob whose seeded run died must NOT be handed back to the driver"
        );
        assert!(!path.exists(), "the suspect blob is deleted");
        assert!(!marker_for(dead).exists(), "the corpse's marker is swept");

        // 3. A marker from a LIVE process (a concurrent `infr` — a serve alongside a CLI run) is
        //    NOT a corpse. It must not cost the other process its cache.
        c.store(&payload).unwrap();
        let live = marker_for(std::process::id());
        std::fs::write(&live, b"").unwrap();
        assert_eq!(
            c.load().as_deref(),
            Some(&payload[..]),
            "a concurrent live run's marker must not discard the cache"
        );
        assert!(path.exists(), "the blob survives a live marker");
        std::fs::remove_file(&live).unwrap();

        // 4. The driver rejected the artifact (or the device was lost): the blob goes, and so does
        //    this instance's marker — it accuses a file that no longer exists.
        c.store(&payload).unwrap();
        assert!(c.load().is_some(), "armed");
        c.invalidate();
        assert!(!path.exists(), "invalidate deletes the blob");
        assert!(!c.marker(&path).exists(), "and disarms its own marker");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Sibling caches in ONE process (distinct nonces on the same path) own SEPARATE markers — so
    /// one instance's clean `disarm()` never deletes a live sibling's, and a still-armed sibling
    /// that later dies uncleanly is still caught.
    #[test]
    fn sibling_markers_are_independent() {
        let dir = tmp_dir("siblings");
        let c1 = cache_in(&dir, b"k");
        let c2 = cache_in(&dir, b"k");
        let path = c1.path().unwrap().to_path_buf();
        assert_ne!(
            c1.marker(&path),
            c2.marker(&path),
            "sibling caches must not share a marker"
        );

        // Arm both, then disarm ONLY c1: c2's marker must survive (the bug was c1's disarm
        // deleting the shared marker out from under a live c2).
        std::fs::File::create(c1.marker(&path)).unwrap();
        std::fs::File::create(c2.marker(&path)).unwrap();
        c1.disarm();
        assert!(!c1.marker(&path).exists(), "c1 disarms its OWN marker");
        assert!(
            c2.marker(&path).exists(),
            "c2's marker survives c1's disarm"
        );

        // A dead process's `<pid>-<nonce>` marker is still swept by the tripwire, and the live
        // sibling's is left alone.
        let dead = spawn_and_reap();
        let dead_marker = path.with_extension(format!("{MARKER_EXT}.{dead}-3"));
        std::fs::write(&dead_marker, b"").unwrap();
        c2.discard_if_a_seeded_run_died();
        assert!(!dead_marker.exists(), "a dead instance's marker is swept");
        assert!(
            c2.marker(&path).exists(),
            "the live sibling's marker is left alone by the sweep"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two saves in one process must use DISTINCT temp files, or a concurrent pair (a `serve`
    /// compiling different kernels) torn-writes / failed-renames over each other.
    #[test]
    fn save_temp_paths_are_unique() {
        let path = std::env::temp_dir().join("infr-kcache-tmp-uniq/artifact.bin");
        let a = tmp_path(&path);
        let b = tmp_path(&path);
        assert_ne!(a, b, "two save temp paths in one process must differ");
        for t in [&a, &b] {
            let name = t.file_name().unwrap().to_str().unwrap();
            assert!(
                name.starts_with("artifact.tmp."),
                "temp name {name} keeps the <path>.tmp.<pid>.<seq> shape"
            );
        }
    }

    /// `store` leaves no litter: the temp file is renamed away, never left beside the target.
    #[test]
    fn store_leaves_no_temp_files() {
        let dir = tmp_dir("notemp");
        let c = cache_in(&dir, b"k");
        c.store(&[1u8; 128]).unwrap();
        c.store(&[2u8; 128]).unwrap();
        let names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["artifact.bin"], "only the target file remains");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A DISABLED cache is a total no-op: it never creates its directory and every method answers
    /// immediately. (`kernels.vulkan.pipeline_cache_disk = false`, `kernels.rocm.module_cache =
    /// false` must not leave a trace on a box that opted out.)
    #[test]
    fn a_disabled_cache_never_touches_the_filesystem() {
        let dir = tmp_dir("disabled");
        let probe = dir.join("witness");
        let c = KernelCache::open("must-not-exist.bin", MAGIC, b"k".to_vec(), false);
        assert!(c.path().is_none(), "a disabled cache has no path");
        assert!(c.load().is_none(), "disabled load is a miss");
        c.store(&[7u8; 64])
            .expect("disabled store is Ok and a no-op");
        c.disarm();
        c.invalidate();
        assert!(!probe.exists());
        // Nothing named after the file we asked for may exist anywhere we would have put it.
        let would_be = cache_dir()
            .unwrap_or_else(|| std::env::temp_dir().join("infr"))
            .join("must-not-exist.bin");
        assert!(!would_be.exists(), "a disabled cache wrote {would_be:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `open` resolves `$XDG_CACHE_HOME`/`$HOME` and creates the directory only when ENABLED.
    /// (Read-only assertions about the resolver — the real cache dir is not written to.)
    #[test]
    fn open_resolves_the_xdg_cache_dir() {
        if let Some(d) = cache_dir() {
            assert!(d.ends_with("infr"), "the cache dir is <base>/infr: {d:?}");
        }
    }
}
