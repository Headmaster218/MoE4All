//! On-disk `vkPipelineCache` persistence. Pipeline creation (`vkCreateComputePipelines`) is the
//! driver's GPU-specific codegen and was measured at ~5s across a cold DG forward's kernel set —
//! all one-time work that previously re-ran every process launch (we passed
//! `vk::PipelineCache::null()` everywhere and relied incidentally on Mesa's own shader cache,
//! which softens but does not eliminate it). This module seeds ONE `vk::PipelineCache` from a
//! per-device file at backend init and writes it back (debounced, and finally on drop), so every
//! launch after the first creates pipelines from cached binaries.
//!
//! Invalidation is three-layer:
//! - The DRIVER validates its own header inside the blob (vendor/device/driverVersion/cacheUUID)
//!   and silently ignores data it can't use — so a driver upgrade never corrupts, at worst it
//!   recompiles once and the next save replaces the file.
//! - OUR envelope carries the build-time SHADER_SET_FINGERPRINT (FNV-1a over every compiled
//!   SPIR-V blob — see build.rs): any shader change discards the old file WHOLESALE instead of
//!   letting entries for retired pipeline variants accumulate in the blob forever.
//! - OUR envelope also carries an FNV-1a CHECKSUM of the payload, verified on load. What lands in
//!   this file is driver-authored machine code that we hand straight back to the driver, and
//!   `vkCreatePipelineCache`'s contract is explicit that invalid data is allowed to produce
//!   UNDEFINED BEHAVIOR — on a GPU that means a hung ring, not a clean error. A truncated or
//!   bit-rotted file must therefore die HERE, at a cheap one-time recompile, rather than reach
//!   the driver. (Mesa/RADV happens to hash its own cache objects and drop the ones that don't
//!   validate — measured: a blob with 25% of its payload bytes scrambled still ran correctly —
//!   so this layer is defense-in-depth against a less careful driver, not a load-bearing fix for
//!   any failure observed on RADV.)
//!
//! Writes are atomic AND durable: the payload is written to a unique per-save temp file
//! (`<cache>.tmp.<pid>.<seq>`, so two concurrent saves in one process never share a path), `fsync`ed,
//! then `rename`d over the target, and the directory entry is `fsync`ed too. `rename` alone is only
//! atomic with respect to a concurrent READER — on a crash/power-loss it can leave the new name
//! pointing at an inode whose data blocks were never flushed (ext4 delayed allocation), i.e. a
//! valid-looking file over garbage. The checksum above would catch that; the fsync keeps it from
//! happening at all.
//!
//! ── THE TRIPWIRE ──────────────────────────────────────────────────────────────────────────────
//!
//! None of the above can catch the failure that actually happened: a blob that is perfectly
//! WELL-FORMED but contains a shader binary that HANGS THE GPU. (Real incident: a hung
//! `native_idm_q5k_sg2` sat in this file and was re-seeded on every launch, so a 35B MoE returned
//! all-zero logits or a device-lost — surviving reboots, reproducing at CI-green commits, and
//! ignoring every code knob, because the poison was on DISK, not in the tree.) A checksum sees
//! only the bytes we wrote, and they are exactly the bytes we wrote. RADV already hashes its own
//! cache entries and drops damaged ones, so corruption was never the mechanism.
//!
//! So instead of validating the CONTENT, watch what HAPPENS after we hand it to the driver — the
//! same dirty-bit trick a filesystem uses:
//!
//! 1. When a run seeds the driver from a loaded blob, drop a marker file next to it.
//! 2. On a clean exit (device NOT lost), delete the marker.
//! 3. If a marker from a DEAD process is found at startup, that run seeded from this blob and then
//!    died without a clean exit. The blob is not trustworthy: delete it and recompile.
//!
//! That closes the loop the incident was stuck in — a poisoned blob hangs the GPU, the run dies
//! with a lost device, the marker survives, and the NEXT launch throws the blob away. One slow
//! start instead of a hang that reproduces forever.
//!
//! The false positive is deliberate and cheap: any unclean death (a crash, a power cut, SIGKILL)
//! discards a perfectly good cache and costs one cold pipeline build. The asymmetry is total —
//! the other mistake is an undebuggable GPU hang, so we take the recompile every time.
//!
//! Markers are keyed PER-BACKEND-INSTANCE (`<cache>.seeded.<pid>-<nonce>`) and a marker is only
//! stale if its PROCESS is GONE, so neither a second `infr` running concurrently (a `serve`
//! alongside a CLI run) NOR a sibling backend in the same process (an MTP bench loop) is mistaken
//! for a crashed one — and one backend's clean drop only disarms ITS OWN marker, never a live
//! sibling's.
//!
//! `kernels.vulkan.pipeline_cache_disk = false` (`INFR_NO_PIPELINE_CACHE=1`) disables persistence
//! (the in-process cache handle still works).

use ash::vk;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

include!(concat!(env!("OUT_DIR"), "/shader_fingerprint.rs"));

/// Monotonic per-process sequence for save temp files (`<cache>.tmp.<pid>.<seq>`). Two threads
/// saving concurrently (a `serve` compiling different kernels) must never write+rename the SAME
/// temp path, or one's write/fsync races the other's rename → a torn write or a failed rename.
static SAVE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Monotonic per-process sequence handed to each [`PcachePersist`] as its tripwire-marker nonce.
/// Two backends built on the SAME device in ONE process (an `infr bench` MTP loop, a CPU/Vulkan
/// parity run) must get DISTINCT markers: keying the marker by PID alone let the first backend's
/// clean `disarm()` delete the shared marker while a sibling was still live, so a later unclean
/// death then left a poisoned blob uncaught — the exact case the tripwire exists for.
static PCACHE_NONCE: AtomicU64 = AtomicU64::new(0);

/// Envelope version. Bumped from `INFRVPC1` when the payload checksum was added — an old file has
/// no checksum field, and its `1` magic makes `load` reject it outright (one free recompile).
const MAGIC: &[u8; 8] = b"INFRVPC2";
/// MAGIC(8) + fingerprint(8) + driver_version(4) + pipelineCacheUUID(16) + payload_len(8) +
/// payload_hash(8).
const HEADER_LEN: usize = 52;
/// Debounce for mid-run saves: pipeline creation comes in bursts (warmup, a new arch's first
/// forward) — one save per burst-second is plenty, and the final Drop save catches the tail.
const SAVE_DEBOUNCE_SECS: u64 = 1;

/// FNV-1a over the blob — the same hash build.rs uses for `SHADER_SET_FINGERPRINT`. Not a
/// cryptographic checksum and not meant to be one: it guards against truncation/bit-rot on a file
/// only this process family writes, not against an adversary who can already write to `$HOME`.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Handle for one device's persisted pipeline cache: where it lives on disk and when it was last
/// written. `None`-able at the call sites (env-disabled or no writable cache dir).
pub(crate) struct PcachePersist {
    path: PathBuf,
    /// Driver version folded into the envelope alongside the shader-set fingerprint: the driver
    /// already ignores stale blobs itself, but a version flip also means retired entries would
    /// sit in the file forever — treat it like a shader-set change and start fresh.
    driver_version: u32,
    /// `VkPhysicalDeviceProperties::pipelineCacheUUID` — the driver's OWN identity for "binaries
    /// I can consume". The file is already keyed per (vendor_id, device_id) by NAME, so one GPU
    /// never reads another's blob on a multi-GPU box, and the driver re-checks this same UUID in
    /// the header it embeds inside the payload. Carrying it in OUR envelope too closes the one
    /// gap those leave: a driver REBUILD (or a distro rebuild) can keep `driver_version` while
    /// changing the cache UUID, and the reward for guessing wrong is a driver handed binaries it
    /// considers valid-ish — undefined behavior, i.e. a hung ring, not an error. Cheap to check,
    /// so check it.
    cache_uuid: [u8; 16],
    /// Per-backend-instance nonce (from [`PCACHE_NONCE`]) folded into this instance's tripwire
    /// marker name so sibling backends in one process own DISTINCT markers — see the module doc.
    nonce: u64,
    last_save: Mutex<Instant>,
}

/// Suffix for a tripwire marker: `<cache file>.seeded.<pid>-<nonce>`. See the module doc.
const MARKER_EXT: &str = "seeded";

/// Is `pid` still running? `kill(pid, 0)` delivers no signal and only asks the kernel whether the
/// process exists: `Ok` = alive, `EPERM` = alive but not ours (a foreign process reusing the pid —
/// treat as alive, i.e. do NOT discard the cache on it), anything else (`ESRCH`) = gone.
fn pid_alive(pid: i32) -> bool {
    // SAFETY: `kill` with signal 0 performs only an existence/permission check.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl PcachePersist {
    /// `~/.cache/infr/vk-pipeline-cache-{vendor:08x}-{device:08x}.bin` (XDG-aware) — keyed per
    /// device so a multi-GPU box never clobbers one GPU's cache with another's.
    ///
    /// `disk` is `kernels.vulkan.pipeline_cache_disk` off the backend's `Config`
    /// (`INFR_NO_PIPELINE_CACHE` inverted, S5a): `false` ⇒ no persistence, which is `None` here.
    /// The in-process cache handle is unaffected either way.
    pub(crate) fn new(props: &vk::PhysicalDeviceProperties, disk: bool) -> Option<Self> {
        if !disk {
            return None;
        }
        let base = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
        let dir = base.join("infr");
        std::fs::create_dir_all(&dir).ok()?;
        Some(Self {
            path: dir.join(format!(
                "vk-pipeline-cache-{:08x}-{:08x}.bin",
                props.vendor_id, props.device_id
            )),
            driver_version: props.driver_version,
            cache_uuid: props.pipeline_cache_uuid,
            nonce: PCACHE_NONCE.fetch_add(1, Ordering::Relaxed),
            last_save: Mutex::new(Instant::now()),
        })
    }

    /// This backend instance's tripwire marker: `<cache file>.seeded.<pid>-<nonce>`. The nonce
    /// keeps sibling backends in one process from sharing (and prematurely disarming) one marker.
    fn marker(&self) -> PathBuf {
        self.path.with_extension(format!(
            "{MARKER_EXT}.{}-{}",
            std::process::id(),
            self.nonce
        ))
    }

    /// A UNIQUE temp path for one `save`: `<cache>.tmp.<pid>.<seq>`. The per-process [`SAVE_SEQ`]
    /// makes two concurrent saves in one process write DISTINCT files, so neither clobbers the
    /// other's in-flight bytes before its own rename. An associated fn (not a method) so the unit
    /// test can assert distinct names without a full instance.
    fn tmp_path(cache_path: &std::path::Path) -> PathBuf {
        cache_path.with_extension(format!(
            "tmp.{}.{}",
            std::process::id(),
            SAVE_SEQ.fetch_add(1, Ordering::Relaxed)
        ))
    }

    /// TRIPWIRE, step 3 (see the module doc): a marker left behind by a process that is GONE means
    /// that run seeded from this blob and then died without a clean exit — a hung GPU is exactly
    /// how that happens, and the blob is the prime suspect. Discard it and recompile.
    ///
    /// Markers whose process is still ALIVE belong to a concurrently-running `infr` and are left
    /// alone. Returns true when the cache was discarded (for the test; the caller doesn't care).
    fn discard_if_a_seeded_run_died(&self) -> bool {
        let Some(dir) = self.path.parent() else {
            return false;
        };
        let Some(stem) = self.path.file_name().and_then(|s| s.to_str()) else {
            return false;
        };
        // `foo.bin` -> markers are `foo.seeded.<pid>-<nonce>` (with_extension replaces `.bin`).
        let prefix = format!(
            "{}.{MARKER_EXT}.",
            stem.strip_suffix(".bin").unwrap_or(stem)
        );
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
            // sibling backends of the same pid. (`split` also accepts a legacy bare `<pid>`.)
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
                "[infr] a previous run seeded the GPU pipeline cache and then died without a clean \
                 exit (a hung GPU does exactly that, and a cached shader binary is the prime \
                 suspect) — discarding {} and recompiling. This costs one slow start.",
                self.path.display()
            );
            let _ = std::fs::remove_file(&self.path);
        }
        dirty
    }

    /// Read + validate the persisted blob. Any mismatch (magic, fingerprint, driver version,
    /// truncation, or a payload that fails its checksum) returns `None` — the stale/damaged file
    /// is simply replaced by the next save, at the cost of one cold pipeline build.
    ///
    /// TRIPWIRE, steps 1+3: sweeps dead processes' markers first (which may discard the blob), and
    /// ARMS this process's marker if it does end up seeding the driver from the file.
    pub(crate) fn load(&self) -> Option<Vec<u8>> {
        self.discard_if_a_seeded_run_died();
        let payload = self.load_validated()?;
        // Armed BEFORE the payload reaches `vkCreatePipelineCache` — if that call is what hangs
        // the GPU, the marker has to already be on disk for the next launch to find.
        let _ = std::fs::File::create(self.marker());
        Some(payload)
    }

    /// The envelope checks alone (no tripwire) — split out so the unit tests can exercise them
    /// without a live process's marker bookkeeping.
    fn load_validated(&self) -> Option<Vec<u8>> {
        let data = std::fs::read(&self.path).ok()?;
        if data.len() < HEADER_LEN || &data[..8] != MAGIC {
            return None;
        }
        let fp = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let drv = u32::from_le_bytes(data[16..20].try_into().unwrap());
        let uuid: [u8; 16] = data[20..36].try_into().unwrap();
        let len = u64::from_le_bytes(data[36..44].try_into().unwrap()) as usize;
        let sum = u64::from_le_bytes(data[44..52].try_into().unwrap());
        if fp != SHADER_SET_FINGERPRINT
            || drv != self.driver_version
            || uuid != self.cache_uuid
            || data.len() != HEADER_LEN + len
        {
            return None;
        }
        let payload = &data[HEADER_LEN..];
        if fnv1a(payload) != sum {
            // Damaged file: never hand it to `vkCreatePipelineCache` (invalid cache data is
            // explicitly undefined behavior, and on a GPU that reads as a hung ring rather than
            // an error). Drop it and let this launch rebuild.
            eprintln!(
                "[infr] pipeline cache {} failed its checksum — discarding and rebuilding",
                self.path.display()
            );
            let _ = std::fs::remove_file(&self.path);
            return None;
        }
        Some(payload.to_vec())
    }

    /// Serialize the live cache to disk atomically AND durably: write the temp file, `fsync` it,
    /// `rename` it over the target, then `fsync` the directory entry. See the module doc for why
    /// the plain `write` + `rename` this replaces was not enough (rename is atomic for a reader,
    /// but on an unclean shutdown it can publish a name over unflushed data blocks).
    pub(crate) fn save(&self, device: &ash::Device, cache: vk::PipelineCache) {
        if cache == vk::PipelineCache::null() {
            return;
        }
        let Ok(blob) = (unsafe { device.get_pipeline_cache_data(cache) }) else {
            return;
        };
        if blob.is_empty() {
            return;
        }
        let mut out = Vec::with_capacity(HEADER_LEN + blob.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&SHADER_SET_FINGERPRINT.to_le_bytes());
        out.extend_from_slice(&self.driver_version.to_le_bytes());
        out.extend_from_slice(&self.cache_uuid);
        out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        out.extend_from_slice(&fnv1a(&blob).to_le_bytes());
        out.extend_from_slice(&blob);
        let tmp = Self::tmp_path(&self.path);
        if write_durable(&tmp, &out).is_ok() && std::fs::rename(&tmp, &self.path).is_ok() {
            // The rename itself is a directory metadata change: sync the directory so the new
            // entry survives a crash too (the payload it points at is already on disk).
            if let Some(dir) = self.path.parent() {
                if let Ok(d) = std::fs::File::open(dir) {
                    let _ = d.sync_all();
                }
            }
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
        *self.last_save.lock().unwrap() = Instant::now();
    }

    /// TRIPWIRE, step 2 (see the module doc): the run is over. `device_lost` is the verdict.
    ///
    /// * **Device NOT lost** — a clean exit. Save the cache and disarm the marker.
    /// * **Device LOST** — this run hung the GPU. Do NOT save (the live cache holds whatever
    ///   binary just hung it, and persisting it is how the poison propagates to the next launch),
    ///   and DELETE the on-disk blob outright, because if we seeded from it, it is the suspect. The
    ///   marker goes too: the file it accused is already gone.
    ///
    /// Note this fires on a lost device whether or not we seeded from disk this run — a hang from
    /// a freshly-compiled pipeline is just as unsafe to persist.
    pub(crate) fn finish(&self, device: &ash::Device, cache: vk::PipelineCache, device_lost: bool) {
        if device_lost {
            eprintln!(
                "[infr] the GPU device was lost during this run — discarding the pipeline cache {} \
                 rather than persisting a shader binary that may be what hung it.",
                self.path.display()
            );
            self.discard();
        } else {
            self.save(device, cache);
        }
        self.disarm();
    }

    /// Delete the on-disk blob (not the in-process cache handle, which the driver still owns).
    fn discard(&self) {
        let _ = std::fs::remove_file(&self.path);
    }

    /// Clear this process's tripwire marker — "I seeded from that blob and I came back alive".
    fn disarm(&self) {
        let _ = std::fs::remove_file(self.marker());
    }

    /// Debounced save for mid-run persistence (called after each NEW pipeline lands) — covers
    /// long-lived processes that never Drop cleanly (serve under SIGKILL).
    pub(crate) fn maybe_save(&self, device: &ash::Device, cache: vk::PipelineCache) {
        {
            let last = self.last_save.lock().unwrap();
            if last.elapsed().as_secs() < SAVE_DEBOUNCE_SECS {
                return;
            }
        }
        self.save(device, cache);
    }
}

/// `fs::write` + `fsync`: the bytes are on the platter before the caller renames over the target.
fn write_durable(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(bytes)?;
    f.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The envelope round-trips, and every way a file can be damaged (bad magic, a shader-set /
    /// driver flip, truncation, a flipped payload byte) is REJECTED rather than handed to
    /// `vkCreatePipelineCache` — where invalid data is undefined behavior, i.e. a hung GPU.
    #[test]
    fn envelope_rejects_damage() {
        let dir = std::env::temp_dir().join(format!("infr-pcache-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache.bin");
        const UUID: [u8; 16] = [9u8; 16];
        let p = PcachePersist {
            path: path.clone(),
            driver_version: 7,
            cache_uuid: UUID,
            nonce: 0,
            last_save: Mutex::new(Instant::now()),
        };
        let payload: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let envelope = |fp: u64, drv: u32, uuid: [u8; 16], sum: u64, body: &[u8]| {
            let mut out = Vec::new();
            out.extend_from_slice(MAGIC);
            out.extend_from_slice(&fp.to_le_bytes());
            out.extend_from_slice(&drv.to_le_bytes());
            out.extend_from_slice(&uuid);
            out.extend_from_slice(&(body.len() as u64).to_le_bytes());
            out.extend_from_slice(&sum.to_le_bytes());
            out.extend_from_slice(body);
            out
        };
        let good = envelope(SHADER_SET_FINGERPRINT, 7, UUID, fnv1a(&payload), &payload);

        std::fs::write(&path, &good).unwrap();
        assert_eq!(p.load().as_deref(), Some(&payload[..]), "good blob loads");

        // A single flipped payload byte must fail the checksum (and the file is removed).
        let mut rot = good.clone();
        rot[HEADER_LEN + 100] ^= 0x01;
        std::fs::write(&path, &rot).unwrap();
        assert!(p.load().is_none(), "bit-rotted payload must be rejected");
        assert!(!path.exists(), "a damaged cache file is deleted, not kept");

        // Truncation (the tail never reached disk).
        std::fs::write(&path, &good[..good.len() - 64]).unwrap();
        assert!(p.load().is_none(), "truncated blob must be rejected");

        // Wrong shader set / wrong driver.
        std::fs::write(
            &path,
            envelope(
                SHADER_SET_FINGERPRINT ^ 1,
                7,
                UUID,
                fnv1a(&payload),
                &payload,
            ),
        )
        .unwrap();
        assert!(p.load().is_none(), "stale shader set must be rejected");
        std::fs::write(
            &path,
            envelope(SHADER_SET_FINGERPRINT, 8, UUID, fnv1a(&payload), &payload),
        )
        .unwrap();
        assert!(p.load().is_none(), "driver-version flip must be rejected");

        // A blob from a driver that reports the SAME version but a different cache UUID (a driver
        // rebuild) — and, by the same check, any blob whose binaries this driver did not author.
        std::fs::write(
            &path,
            envelope(
                SHADER_SET_FINGERPRINT,
                7,
                [1u8; 16],
                fnv1a(&payload),
                &payload,
            ),
        )
        .unwrap();
        assert!(
            p.load().is_none(),
            "foreign pipelineCacheUUID must be rejected"
        );

        // A v1 (checksum-less) file from an older build.
        let mut v1 = good.clone();
        v1[..8].copy_from_slice(b"INFRVPC1");
        std::fs::write(&path, &v1).unwrap();
        assert!(p.load().is_none(), "old envelope version must be rejected");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// THE TRIPWIRE. A blob that hangs the GPU is perfectly well-formed, so no envelope check can
    /// see it — the only evidence is that the run which SEEDED from it never came back. These are
    /// the three states that distinguishes.
    #[test]
    fn tripwire_discards_a_blob_whose_seeded_run_died() {
        let dir = std::env::temp_dir().join(format!("infr-pcache-trip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache.bin");
        const UUID: [u8; 16] = [3u8; 16];
        let p = PcachePersist {
            path: path.clone(),
            driver_version: 5,
            cache_uuid: UUID,
            nonce: 0,
            last_save: Mutex::new(Instant::now()),
        };
        let payload: Vec<u8> = (0u8..=255).cycle().take(2048).collect();
        let write_good = || {
            let mut out = Vec::new();
            out.extend_from_slice(MAGIC);
            out.extend_from_slice(&SHADER_SET_FINGERPRINT.to_le_bytes());
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&UUID);
            out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            out.extend_from_slice(&fnv1a(&payload).to_le_bytes());
            out.extend_from_slice(&payload);
            std::fs::write(&path, &out).unwrap();
        };
        let marker_for = |pid: u32| path.with_extension(format!("{MARKER_EXT}.{pid}"));

        // 1. No marker: an ordinary run loads the blob — and ARMS its own marker before handing
        //    the payload to the driver, so a hang from here on is attributable.
        write_good();
        assert_eq!(p.load().as_deref(), Some(&payload[..]), "clean blob loads");
        assert!(p.marker().exists(), "loading must arm the tripwire");

        // 2. A marker from a DEAD process: that run seeded from this blob and never exited
        //    cleanly. The blob is the suspect — discard it, and the load reports a cache miss.
        let dead = spawn_and_reap();
        std::fs::write(marker_for(dead), b"").unwrap();
        write_good();
        assert!(
            p.load().is_none(),
            "a blob whose seeded run died must NOT be handed back to the driver"
        );
        assert!(!path.exists(), "the suspect blob is deleted");
        assert!(!marker_for(dead).exists(), "the corpse's marker is swept");

        // 3. A marker from a LIVE process (a concurrent `infr` — a serve alongside a CLI run) is
        //    NOT a corpse. It must not cost the other process its cache.
        write_good();
        let live = marker_for(std::process::id());
        std::fs::write(&live, b"").unwrap();
        assert_eq!(
            p.load().as_deref(),
            Some(&payload[..]),
            "a concurrent live run's marker must not discard the cache"
        );
        assert!(path.exists(), "the blob survives a live marker");

        // 4. The device-lost arm of `finish`: discard the blob rather than persist whatever binary
        //    just hung the GPU, then disarm. (These are the two calls `finish` makes on that path;
        //    `finish` itself needs a live `vk::Device`, which a unit test has no business creating.)
        write_good();
        p.discard();
        p.disarm();
        assert!(
            !path.exists(),
            "a run that lost the device discards the blob"
        );
        assert!(!p.marker().exists(), "and disarms its marker");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Finding #6: two saves in one process must use DISTINCT temp files, or a concurrent pair
    /// (a `serve` compiling different kernels) torn-writes/failed-renames over each other.
    #[test]
    fn save_temp_paths_are_unique() {
        let path = std::env::temp_dir().join("infr-pcache-tmp-uniq/cache.bin");
        let a = PcachePersist::tmp_path(&path);
        let b = PcachePersist::tmp_path(&path);
        assert_ne!(a, b, "two save temp paths in one process must differ");
        // Both still land beside the target and carry the pid (so a cross-process collision is
        // impossible too) and the unique sequence.
        for t in [&a, &b] {
            let name = t.file_name().unwrap().to_str().unwrap();
            assert!(
                name.starts_with("cache.tmp."),
                "temp name {name} keeps the <cache>.tmp.<pid>.<seq> shape"
            );
        }
    }

    /// Finding #7: sibling backends in ONE process (distinct nonces on the same cache path) own
    /// SEPARATE tripwire markers — so one backend's clean `disarm()` never deletes a live
    /// sibling's marker, and a still-armed sibling that later dies uncleanly is still caught.
    #[test]
    fn sibling_markers_are_independent() {
        let dir = std::env::temp_dir().join(format!("infr-pcache-sib-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache.bin");
        let mk = |nonce: u64| PcachePersist {
            path: path.clone(),
            driver_version: 1,
            cache_uuid: [0u8; 16],
            nonce,
            last_save: Mutex::new(Instant::now()),
        };
        let p1 = mk(0);
        let p2 = mk(1);
        assert_ne!(
            p1.marker(),
            p2.marker(),
            "sibling backends must not share a marker"
        );

        // Arm both, then disarm ONLY p1: p2's marker must survive (the bug was p1's disarm
        // deleting the shared marker out from under a live p2).
        std::fs::File::create(p1.marker()).unwrap();
        std::fs::File::create(p2.marker()).unwrap();
        p1.disarm();
        assert!(!p1.marker().exists(), "p1 disarms its OWN marker");
        assert!(p2.marker().exists(), "p2's marker must survive p1's disarm");

        // A dead process's new-format `<pid>-<nonce>` marker is still swept by the tripwire.
        let dead = spawn_and_reap();
        let dead_marker = path.with_extension(format!("{MARKER_EXT}.{dead}-3"));
        std::fs::write(&dead_marker, b"").unwrap();
        p2.discard_if_a_seeded_run_died();
        assert!(
            !dead_marker.exists(),
            "a dead backend's <pid>-<nonce> marker is swept"
        );
        assert!(
            p2.marker().exists(),
            "the live sibling's marker is left alone by the sweep"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A pid that is guaranteed DEAD (spawned and reaped), for the corpse case above. Reusing a
    /// just-reaped pid is theoretically possible but the kernel hands out pids sequentially, so it
    /// will not happen within this test.
    fn spawn_and_reap() -> u32 {
        let mut c = std::process::Command::new("true")
            .spawn()
            .expect("spawn /bin/true");
        let pid = c.id();
        c.wait().expect("reap");
        assert!(!pid_alive(pid as i32), "the reaped child must be gone");
        pid
    }
}
