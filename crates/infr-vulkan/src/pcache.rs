//! On-disk `vkPipelineCache` persistence — the Vulkan-specific SHELL around
//! [`infr_core::kernel_cache::KernelCache`].
//!
//! Pipeline creation (`vkCreateComputePipelines`) is the driver's GPU-specific codegen and was
//! measured at ~5s across a cold DG forward's kernel set — all one-time work that previously
//! re-ran every process launch (we passed `vk::PipelineCache::null()` everywhere and relied
//! incidentally on Mesa's own shader cache, which softens but does not eliminate it). This module
//! seeds ONE `vk::PipelineCache` from a per-device file at backend init and writes it back
//! (debounced, and finally on drop), so every launch after the first creates pipelines from cached
//! binaries.
//!
//! **Everything that is not Vulkan lives in `infr-core`'s [`KernelCache`]** — the envelope, the
//! payload checksum, the atomic+durable write, and THE TRIPWIRE that caught a hung shader binary
//! sitting in this very file (read that module's doc; it is the design rationale). What stays
//! here is what only Vulkan knows:
//!
//! - **The KEY** (compared verbatim by `KernelCache`): `driverVersion ++ pipelineCacheUUID ++
//!   SHADER_SET_FINGERPRINT`.
//!   - The DRIVER validates its own header inside the blob (vendor/device/driverVersion/cacheUUID)
//!     and silently ignores data it can't use — so a driver upgrade never corrupts, at worst it
//!     recompiles once. We fold `driverVersion` in anyway because a version flip also means
//!     retired entries would sit in the file forever; treat it like a shader-set change and start
//!     fresh.
//!   - `pipelineCacheUUID` is the driver's OWN identity for "binaries I can consume". The file is
//!     already keyed per (vendor_id, device_id) by NAME, and the driver re-checks this UUID in the
//!     header it embeds inside the payload. Carrying it in OUR key too closes the one gap those
//!     leave: a driver REBUILD (or a distro rebuild) can keep `driverVersion` while changing the
//!     cache UUID, and the reward for guessing wrong is a driver handed binaries it considers
//!     valid-ish — undefined behavior, i.e. a hung ring, not an error. Cheap to check, so check it.
//!   - `SHADER_SET_FINGERPRINT` (FNV-1a over every compiled SPIR-V blob — see build.rs): any
//!     shader change discards the old file WHOLESALE instead of letting entries for retired
//!     pipeline variants accumulate in the blob forever.
//! - **The API calls**: `vkGetPipelineCacheData` to produce the payload, `vkCreatePipelineCache`
//!   to consume it (in `lib.rs`), and the device-lost verdict on drop.
//! - **The DEBOUNCED mid-run saves.** Unique to Vulkan: the driver's cache GROWS during a run as
//!   new pipelines are created, so the file must be rewritten periodically for a long-lived
//!   process that never drops cleanly (a `serve` under SIGKILL). Metal's binary archive, by
//!   contrast, is produced once, whole, at init — it has nothing to re-save.
//!
//! `kernels.vulkan.pipeline_cache_disk = false` disables persistence (the in-process cache handle
//! still works).

use ash::vk;
use infr_core::kernel_cache::KernelCache;
use std::sync::Mutex;
use std::time::Instant;

include!(concat!(env!("OUT_DIR"), "/shader_fingerprint.rs"));

/// Producer magic + on-disk layout version. `INFRVPC1` → `INFRVPC2` added the payload checksum;
/// `INFRVPC2` → `INFRVPC3` moved the envelope onto the shared [`KernelCache`] layout (the
/// fingerprint/driver/UUID fields became one opaque KEY blob, and the header grew a format
/// version). An old file's magic makes `load` reject it outright — one free recompile — rather
/// than letting its fields be MISREAD at the new offsets.
const MAGIC: [u8; 8] = *b"INFRVPC3";

/// Debounce for mid-run saves: pipeline creation comes in bursts (warmup, a new arch's first
/// forward) — one save per burst-second is plenty, and the final Drop save catches the tail.
const SAVE_DEBOUNCE_SECS: u64 = 1;

/// The verbatim-compared cache key: `driverVersion(4) ++ pipelineCacheUUID(16) ++
/// SHADER_SET_FINGERPRINT(8)`. See the module doc for why each of the three is in here; a change in
/// ANY of them must discard the blob, which [`KernelCache`] does by byte-comparing this.
fn cache_key(props: &vk::PhysicalDeviceProperties) -> Vec<u8> {
    let mut key = Vec::with_capacity(28);
    key.extend_from_slice(&props.driver_version.to_le_bytes());
    key.extend_from_slice(&props.pipeline_cache_uuid);
    key.extend_from_slice(&SHADER_SET_FINGERPRINT.to_le_bytes());
    key
}

/// Handle for one device's persisted pipeline cache: the shared on-disk cache plus Vulkan's
/// mid-run save debounce. `None`-able at the call sites (config-disabled or no writable cache dir).
pub(crate) struct PcachePersist {
    cache: KernelCache,
    last_save: Mutex<Instant>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl PcachePersist {
    /// `~/.cache/infr/vk-pipeline-cache-{vendor:08x}-{device:08x}.bin` (XDG-aware) — keyed per
    /// device so a multi-GPU box never clobbers one GPU's cache with another's.
    ///
    /// `disk` is `kernels.vulkan.pipeline_cache_disk` off the backend's `Config` (S5a): `false` ⇒
    /// no persistence, which is `None` here. `None` too when there is no XDG/HOME cache dir at all
    /// — Vulkan reports "no persistence available" rather than falling back to a temp dir, since
    /// its blob is re-saved throughout the run. The in-process cache handle is unaffected either
    /// way.
    pub(crate) fn new(props: &vk::PhysicalDeviceProperties, disk: bool) -> Option<Self> {
        if !disk {
            return None;
        }
        infr_core::kernel_cache::cache_dir()?;
        let file = format!(
            "vk-pipeline-cache-{:08x}-{:08x}.bin",
            props.vendor_id, props.device_id
        );
        Some(Self {
            cache: KernelCache::open(&file, MAGIC, cache_key(props), true),
            last_save: Mutex::new(Instant::now()),
        })
    }

    /// Read + validate the persisted blob, arming the tripwire if it is handed back (see
    /// [`KernelCache::load`]). `None` = miss; the stale/damaged file is simply replaced by the next
    /// save, at the cost of one cold pipeline build.
    pub(crate) fn load(&self) -> Option<Vec<u8>> {
        self.cache.load()
    }

    /// Serialize the live cache to disk (atomically and durably — see [`KernelCache::store`]).
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
        let _ = self.cache.store(&blob);
        *self.last_save.lock().unwrap() = Instant::now();
    }

    /// TRIPWIRE step 2 (see [`infr_core::kernel_cache`]): the run is over. `device_lost` is the
    /// verdict.
    ///
    /// * **Device NOT lost** — a clean exit. Save the cache and disarm the marker.
    /// * **Device LOST** — this run hung the GPU. Do NOT save (the live cache holds whatever
    ///   binary just hung it, and persisting it is how the poison propagates to the next launch),
    ///   and DELETE the on-disk blob outright, because if we seeded from it, it is the suspect.
    ///
    /// Note this fires on a lost device whether or not we seeded from disk this run — a hang from
    /// a freshly-compiled pipeline is just as unsafe to persist.
    pub(crate) fn finish(&self, device: &ash::Device, cache: vk::PipelineCache, device_lost: bool) {
        if device_lost {
            tracing::warn!(
                "[infr] the GPU device was lost during this run — discarding the pipeline cache \
                 rather than persisting a shader binary that may be what hung it."
            );
            // Deletes the blob AND this instance's marker (which would otherwise accuse a file
            // that is already gone).
            self.cache.invalidate();
        } else {
            self.save(device, cache);
            self.cache.disarm();
        }
    }

    /// Is a mid-run save due? Vulkan-only bookkeeping — see the module doc for why Metal has no
    /// equivalent. Split out so a unit test can exercise the debounce without a live device.
    fn save_due(&self) -> bool {
        self.last_save.lock().unwrap().elapsed().as_secs() >= SAVE_DEBOUNCE_SECS
    }

    /// Debounced save for mid-run persistence (called after each NEW pipeline lands) — covers
    /// long-lived processes that never Drop cleanly (serve under SIGKILL).
    pub(crate) fn maybe_save(&self, device: &ash::Device, cache: vk::PipelineCache) {
        if !self.save_due() {
            return;
        }
        self.save(device, cache);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ENVELOPE, the checksum, the durable write and the tripwire are
    /// `infr_core::kernel_cache`'s, and tested there. What this crate owns is the KEY — and the key
    /// is only load-bearing if every input actually reaches it, since `KernelCache` compares the
    /// bytes verbatim and a field that fell out of the composition would silently stop
    /// invalidating. So flip each one and require the key to move.
    #[test]
    fn the_key_covers_driver_version_uuid_and_shader_set() {
        let base = vk::PhysicalDeviceProperties {
            driver_version: 7,
            pipeline_cache_uuid: [9u8; 16],
            ..Default::default()
        };
        let k = cache_key(&base);
        assert_eq!(k.len(), 28, "driverVersion(4) + UUID(16) + fingerprint(8)");
        assert!(
            k.ends_with(&SHADER_SET_FINGERPRINT.to_le_bytes()),
            "the build-time shader-set fingerprint must be in the key"
        );

        let drv = vk::PhysicalDeviceProperties {
            driver_version: 8,
            ..base
        };
        assert_ne!(
            cache_key(&drv),
            k,
            "a driver-version flip must move the key"
        );

        // A driver REBUILD keeps driverVersion and changes only the cache UUID — the case the
        // driver's own header would not catch for us.
        let uuid = vk::PhysicalDeviceProperties {
            pipeline_cache_uuid: [1u8; 16],
            ..base
        };
        assert_ne!(cache_key(&uuid), k, "a pipelineCacheUUID flip must move it");

        // Fields NOT in the key: the file NAME carries vendor/device, so they must not also
        // perturb the key (that would gratuitously invalidate on unrelated property churn).
        let other = vk::PhysicalDeviceProperties {
            vendor_id: 0x1002,
            device_id: 0x744c,
            ..base
        };
        assert_eq!(cache_key(&other), k, "vendor/device are keyed by FILE NAME");
    }

    /// The on-disk layout changed when this moved onto `KernelCache`, so the magic HAD to move
    /// with it: an `INFRVPC2` file's fields sit at different offsets and must be rejected outright
    /// rather than misread as a key + payload. (`KernelCache` rejects a foreign magic — tested
    /// there; what is asserted here is that we actually bumped ours.)
    #[test]
    fn the_magic_was_bumped_off_the_old_layout() {
        assert_eq!(&MAGIC, b"INFRVPC3");
        assert_ne!(
            &MAGIC, b"INFRVPC2",
            "an old-layout file must not be misread"
        );
        assert_ne!(&MAGIC, b"INFRVPC1");
    }

    /// The mid-run save debounce — the one piece of this that stayed Vulkan-specific (the driver's
    /// cache grows during a run; Metal's archive is produced whole at init). A freshly-constructed
    /// handle is NOT due, so a burst of pipeline creations writes the file once, not once per
    /// pipeline.
    #[test]
    fn mid_run_saves_are_debounced() {
        let p = PcachePersist {
            // No filesystem is touched: a disabled cache is a total no-op, and the debounce
            // predicate never reaches it.
            cache: KernelCache::open("unused.bin", MAGIC, Vec::new(), false),
            last_save: Mutex::new(Instant::now()),
        };
        assert!(!p.save_due(), "a just-saved cache must not save again");
        *p.last_save.lock().unwrap() =
            Instant::now() - std::time::Duration::from_secs(SAVE_DEBOUNCE_SECS + 1);
        assert!(p.save_due(), "past the debounce window, a save is due");
    }
}
