//! The device-free half of the Metal pipeline cache: what the payload LOOKS like, what the cache
//! KEY is made of, and how a device name becomes a file name.
//!
//! Split out from [`pcache`](crate::pcache) for one reason: this crate is `#![cfg(target_os =
//! "macos")]`, so nothing in it is built — let alone tested — by the Linux CI jobs or on the dev
//! box. A module that touches no Metal type can still be compiled and run standalone
//! (`rustc --test crates/infr-metal/src/pcache_blob.rs`), which is how the logic below was
//! exercised off-Mac. [`idcache`](crate::idcache) is the precedent; keep this file free of `metal`
//! and of anything that pulls in the crate graph (that is also why the FNV of the MSL source
//! arrives as a `u64` rather than being hashed here — the hash itself is
//! `infr_core::kernel_cache::fnv1a`, shared with every other backend).
//!
//! # The payload is not just the archive
//!
//! `MTLBinaryArchive` is write-only from our side: nothing on it reports which functions it
//! already contains. That matters, because the whole point of the cache is to STOP re-running the
//! driver's back end — and `addComputePipelineFunctionsWithDescriptor:` compiles the pipeline into
//! the archive. If every launch re-added every kernel, the archive would be re-serialized and the
//! compile paid all over again: a cache that costs exactly what it saves.
//!
//! So the payload carries a MANIFEST of the function names in the archive, ahead of the archive
//! bytes:
//!
//! ```text
//! name_count : u32 le
//! name_count × { name_len : u16 le, name : utf-8 bytes }
//! archive    : the MTLBinaryArchive file's bytes (to the end)
//! ```
//!
//! With it, `pcache` knows on the FIRST `get()` of a kernel whether the archive already holds it
//! (skip the add) or not (add it and mark the blob dirty), and a run that creates no new pipeline
//! writes nothing at all.
//!
//! This framing lives INSIDE the shared seam's payload — `infr_core::kernel_cache` still owns the
//! magic, the format version, the key compare, the FNV checksum, the atomic+durable write and the
//! poisoned-blob tripwire. It is a sub-layout of one producer's bytes, not a second cache format:
//! a manifest that fails to parse is simply a MISS, exactly like a failed checksum.

/// Cap on a decoded manifest, so a corrupt (but checksum-valid, i.e. faithfully-stored garbage
/// from a bug on OUR side) header cannot make us allocate wildly. The real kernel set is ~300
/// names; four thousand is unreachable slack.
const MAX_NAMES: usize = 4096;

/// Longest MSL function name we will store. Real names are ~40 chars.
const MAX_NAME_LEN: usize = 256;

/// `name_count ++ (name_len ++ name)* ++ archive`. `names` should be sorted+deduped by the caller
/// so the same archive produces the same bytes (a stable payload means an unchanged run rewrites
/// an identical file rather than churning the disk).
pub(crate) fn encode(names: &[String], archive: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + names.len() * 24 + archive.len());
    out.extend_from_slice(&(names.len() as u32).to_le_bytes());
    for n in names {
        out.extend_from_slice(&(n.len() as u16).to_le_bytes());
        out.extend_from_slice(n.as_bytes());
    }
    out.extend_from_slice(archive);
    out
}

/// Inverse of [`encode`]. `None` on anything malformed — a truncated header, a length that runs
/// past the end, a non-UTF-8 name, an implausible count/length. The caller treats that exactly
/// like a checksum failure: discard the blob and rebuild.
///
/// The archive bytes are NOT validated here (they cannot be — only Metal can say whether it will
/// load them). `pcache` hands them to `newBinaryArchiveWithDescriptor:` and treats a rejection as
/// a recovery, not an error.
pub(crate) fn decode(payload: &[u8]) -> Option<(Vec<String>, Vec<u8>)> {
    let count = u32::from_le_bytes(payload.get(..4)?.try_into().ok()?) as usize;
    if count > MAX_NAMES {
        return None;
    }
    let mut off = 4usize;
    let mut names = Vec::with_capacity(count);
    for _ in 0..count {
        let len = u16::from_le_bytes(payload.get(off..off + 2)?.try_into().ok()?) as usize;
        if len == 0 || len > MAX_NAME_LEN {
            return None;
        }
        off += 2;
        let raw = payload.get(off..off.checked_add(len)?)?;
        names.push(std::str::from_utf8(raw).ok()?.to_string());
        off += len;
    }
    let archive = payload.get(off..)?.to_vec();
    // An archive of zero bytes is not something Metal ever wrote; treat it as damage rather than
    // handing an empty file to `newBinaryArchiveWithDescriptor:`.
    if archive.is_empty() {
        return None;
    }
    Some((names, archive))
}

/// Everything that makes a previously-serialized archive WRONG for this launch, in the order it is
/// laid into the key. Plain data so this module stays device-free — `pcache` fills it in from the
/// live `MTLDevice`.
pub(crate) struct KeyInputs<'a> {
    /// `infr_core::kernel_cache::fnv1a(msl_source())`. `msl_source()` is assembled at RUN time (it
    /// embeds the IQ2/IQ3 grids emitted from the host tables), so hashing the actual string covers
    /// every kernel edit by construction and no build-script fingerprint can drift from it.
    pub(crate) src_hash: u64,
    /// Length of the same string — one extra cheap field against an FNV collision.
    pub(crate) src_len: u64,
    /// `MTLDevice.name` ("Apple M2 Max"). Also in the FILE NAME; here too because the name is a
    /// convention and the key is a check.
    pub(crate) device: &'a str,
    /// `MTLDevice.architecture.name` ("applegpu_g14p", macOS 14+) — the actual codegen target, and
    /// the field that separates two Apple GPUs that share a marketing name. Empty on an OS that
    /// does not answer the selector, which is fine: it just contributes nothing.
    pub(crate) architecture: &'a str,
    /// `NSProcessInfo.operatingSystemVersionString`. A Metal/driver update ships with the OS, and
    /// a compiled pipeline is only guaranteed loadable by the stack that produced it. (Metal
    /// validates its own archive header too and merely MISSES on a mismatch — we invalidate
    /// wholesale so retired entries do not sit in the file forever.)
    pub(crate) os: &'a str,
    /// The `MTLCompileOptions` that affect codegen. Fast-math is OFF and that is load-bearing for
    /// CPU numeric parity (see `Pipelines::build`) — if it ever flips, every archive built under
    /// the other setting must be discarded, so it is in the key.
    pub(crate) fast_math: bool,
}

impl KeyInputs<'_> {
    /// The verbatim-compared key handed to `KernelCache::open`. NUL-separated so two adjacent
    /// fields cannot be re-cut into the same byte string ("ab" ++ "c" vs "a" ++ "bc").
    pub(crate) fn compose(&self) -> Vec<u8> {
        let mut key = Vec::with_capacity(64 + self.device.len() + self.architecture.len());
        key.extend_from_slice(&self.src_hash.to_le_bytes());
        key.extend_from_slice(&self.src_len.to_le_bytes());
        for field in [self.device, self.architecture, self.os] {
            key.extend_from_slice(field.as_bytes());
            key.push(0);
        }
        key.push(self.fast_math as u8);
        key
    }
}

/// `MTLDevice.name` → a file-name token, so a Mac with two GPUs (an Intel Mac's iGPU + dGPU) never
/// hands one device the other's compiled pipelines.
///
/// The escape is INJECTIVE, not merely "safe" — the same lesson `infr-rocm`'s `sanitize_arch`
/// learned the hard way, where folding every non-alphanumeric to `_` made `gfx90a:xnack+` and
/// `gfx90a:xnack-` name ONE file. Alphanumerics pass through lowercased; everything else
/// (including `_` itself, so no input can forge another's escape) becomes `_<hex>`.
///
/// An empty result means "no usable device name" and the caller declines to cache rather than
/// sharing one `metal-pipelines-.bin` between devices.
pub(crate) fn device_token(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for b in raw.bytes() {
        if b.is_ascii_alphanumeric() {
            out.push(b.to_ascii_lowercase() as char);
        } else {
            out.push_str(&format!("_{b:02x}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// The manifest is what lets a launch tell "the archive already has this kernel" from "add it",
    /// so it has to survive the round trip exactly — including the empty set (a first run that
    /// serialized before creating anything) and long/odd names.
    #[test]
    fn the_payload_round_trips_names_and_archive_bytes() {
        let archive: Vec<u8> = (0u8..=255).cycle().take(9000).collect();
        for set in [
            vec![],
            names(&["rmsnorm"]),
            names(&["a", "linear_quik_q4k", "moe_down_routed_iq3s", "z"]),
        ] {
            let payload = encode(&set, &archive);
            let (got_names, got_archive) = decode(&payload).expect("round-trip");
            assert_eq!(got_names, set);
            assert_eq!(got_archive, archive);
        }

        // The archive is taken to the END of the payload, so its size is not stored twice and
        // cannot disagree with itself.
        let one = names(&["k"]);
        assert_eq!(encode(&one, &archive).len(), 4 + 2 + 1 + archive.len());
    }

    /// Every way the framing can be damaged must be a clean MISS (rebuild), never a panic and never
    /// a half-read manifest — an over-long name would otherwise make us skip adding a kernel the
    /// archive does not actually contain, and the pipeline would be re-created cold forever with
    /// the cache reporting a hit.
    #[test]
    fn malformed_payloads_are_rejected_not_guessed() {
        let archive = vec![7u8; 512];
        let good = encode(&names(&["alpha", "beta"]), &archive);
        assert!(decode(&good).is_some(), "the fixture must be valid");

        // Truncated anywhere in the header/manifest.
        for cut in 0..(4 + 2 + 5 + 2 + 4) {
            assert!(
                decode(&good[..cut]).is_none(),
                "a payload cut at {cut} must be rejected"
            );
        }

        // A count that promises more names than there are bytes.
        let mut lying = good.clone();
        lying[..4].copy_from_slice(&9u32.to_le_bytes());
        assert!(decode(&lying).is_none(), "over-long name count");

        // An absurd count must not preallocate — it is rejected on sight.
        let mut absurd = good.clone();
        absurd[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode(&absurd).is_none(), "absurd name count");

        // A name length that runs past the end, and a zero-length name.
        let mut long_name = good.clone();
        long_name[4..6].copy_from_slice(&60000u16.to_le_bytes());
        assert!(decode(&long_name).is_none(), "name length past the end");
        let mut empty_name = good.clone();
        empty_name[4..6].copy_from_slice(&0u16.to_le_bytes());
        assert!(decode(&empty_name).is_none(), "zero-length name");

        // A non-UTF-8 name.
        let mut bad_utf8 = good.clone();
        bad_utf8[6] = 0xff;
        assert!(decode(&bad_utf8).is_none(), "non-utf8 name");

        // No archive bytes at all: Metal never wrote a zero-byte archive, so this is damage.
        assert!(decode(&encode(&names(&["a"]), &[])).is_none(), "no archive");
    }

    /// `KernelCache` compares the key BYTE-FOR-BYTE, so a field that fell out of the composition
    /// would silently stop invalidating — a stale archive handed to a driver that no longer
    /// produced it. Flip each input in turn and require the key to move.
    #[test]
    fn every_key_input_actually_reaches_the_key() {
        let base = KeyInputs {
            src_hash: 0x0123_4567_89ab_cdef,
            src_len: 348_112,
            device: "Apple M2 Max",
            architecture: "applegpu_g14p",
            os: "Version 15.3 (Build 24D60)",
            fast_math: false,
        };
        let k = base.compose();

        let flips: [(&str, Vec<u8>); 6] = [
            (
                "a kernel-source edit",
                KeyInputs {
                    src_hash: 1,
                    ..base
                }
                .compose(),
            ),
            (
                "a source-length change",
                KeyInputs {
                    src_len: 348_113,
                    ..base
                }
                .compose(),
            ),
            (
                "a different GPU",
                KeyInputs {
                    device: "Apple M3 Max",
                    ..base
                }
                .compose(),
            ),
            (
                "a different codegen target",
                KeyInputs {
                    architecture: "applegpu_g15p",
                    ..base
                }
                .compose(),
            ),
            (
                "an OS/Metal update",
                KeyInputs {
                    os: "Version 15.4 (Build 24E5206s)",
                    ..base
                }
                .compose(),
            ),
            (
                "fast-math flipped on",
                KeyInputs {
                    fast_math: true,
                    ..base
                }
                .compose(),
            ),
        ];
        for (what, moved) in &flips {
            assert_ne!(&k, moved, "{what} must move the cache key");
        }

        // The same inputs must produce the same key, or the cache misses on every launch.
        assert_eq!(k, base.compose(), "key composition must be deterministic");

        // Adjacent string fields must not be re-cuttable into one another: with a NUL separator,
        // ("ab","c") and ("a","bc") are distinct keys.
        let ab_c = KeyInputs {
            device: "ab",
            architecture: "c",
            ..base
        }
        .compose();
        let a_bc = KeyInputs {
            device: "a",
            architecture: "bc",
            ..base
        }
        .compose();
        assert_ne!(ab_c, a_bc, "field boundaries must be unambiguous");
    }

    /// The blob file name is what keeps two GPUs in one Mac off each other's compiled pipelines,
    /// so the device token must be INJECTIVE as well as file-name safe.
    #[test]
    fn the_device_token_is_filename_safe_and_injective() {
        assert_eq!(device_token("AppleM2"), "applem2");
        assert_eq!(device_token("Apple M2 Max"), "apple_20m2_20max");

        let raw = [
            "Apple M2",
            "Apple M2 Max",
            "Apple M2 Pro",
            "AppleM2",
            "apple_20m2",
            "AMD Radeon Pro 5500M",
            "Intel(R) UHD Graphics 630",
        ];
        let toks: Vec<String> = raw.iter().map(|d| device_token(d)).collect();
        let mut uniq = toks.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), toks.len(), "device tokens collided: {toks:?}");
        for t in &toks {
            assert!(
                t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "the token names a file: {t}"
            );
        }
        // No device name ⇒ an empty token, which `pcache` reads as "do not cache" rather than
        // letting every device share one file.
        assert_eq!(device_token(""), "");
    }
}
