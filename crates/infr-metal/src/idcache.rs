//! Identity-keyed caches for per-buffer derived state.
//!
//! The Metal backend memoizes work derived from a bound weight buffer's *contents* — the f32
//! dequant ([`MetalBackend::weight_cache`](crate::MetalBackend)) and the compact factored form
//! ([`qui_cache`](crate::MetalBackend)). Both used to be keyed on
//! `MTLBuffer::contents()`, a raw address.
//!
//! **An address is not an identity.** Releasing an `MTLBuffer` returns its pages to Metal's
//! allocator and a later `newBuffer` hands the same `contents()` pointer back for entirely
//! different bytes, so an address-keyed entry is served for a weight it was never computed from —
//! silently wrong weights, no error anywhere. (This backend has already been bitten by exactly that
//! recycling: see the `Backend::compile` note about the decode-replay tape's
//! bound-buffer-address fingerprint colliding across the MTP head's per-draft-step recompiles.)
//!
//! So every allocation is stamped with a [`next_buffer_uid`] serial that is process-unique and
//! never reused, and a cache hit is served only when the stored uid matches the bound buffer's.
//! The (address, byte length) pair stays the map key so a recycled address *replaces* its stale
//! entry (freeing the dead derived buffer) instead of accumulating one entry per allocation ever
//! made — the map stays bounded by the live buffers.
//!
//! This module is deliberately free of any Metal type so its decision logic is unit-testable
//! without a device (see the tests at the bottom).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic source of [`BufId::uid`]. Process-wide (not per-backend) so a uid is unique across
/// every backend alive in the process — two models, `infr serve`, an MTP trunk + head pair.
/// `u64` and never reused: at one allocation per nanosecond it wraps in ~584 years.
static NEXT_BUFFER_UID: AtomicU64 = AtomicU64::new(1);

/// Stamp the next allocation identity. See [`BufId::uid`].
pub(crate) fn next_buffer_uid() -> u64 {
    NEXT_BUFFER_UID.fetch_add(1, Ordering::Relaxed)
}

/// What a [`crate::MetalBuffer`] looks like to the derived-state caches: where it lives (`addr`,
/// `len` — the *slot*, which Metal recycles) plus *which* buffer it actually is (`uid` — the
/// identity, which nothing recycles).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BufId {
    /// `MTLBuffer::contents()` of the underlying buffer. Stable for the buffer's lifetime, and
    /// re-handed-out to an unrelated buffer after it.
    pub(crate) addr: usize,
    /// Byte length of the allocation.
    pub(crate) len: usize,
    /// Process-unique, never-reused allocation serial ([`next_buffer_uid`]).
    pub(crate) uid: u64,
}

/// A `HashMap` from a buffer to state derived from that buffer's bytes, which only ever hands back
/// state derived from *that* buffer. See the module docs.
pub(crate) struct IdCache<V> {
    map: HashMap<(usize, usize), (u64, V)>,
}

impl<V> Default for IdCache<V> {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}

impl<V> IdCache<V> {
    /// The cached value for `id`, or `None` if there is none **or** the slot has since been
    /// recycled for a different buffer (uid mismatch) — in which case the stale entry must not be
    /// served and the caller recomputes.
    pub(crate) fn get(&self, id: BufId) -> Option<&V> {
        match self.map.get(&(id.addr, id.len)) {
            Some((uid, v)) if *uid == id.uid => Some(v),
            _ => None,
        }
    }

    /// Record `v` as derived from `id`. Inserting over an occupied slot is the recycled-address
    /// case: the replaced entry drops here, freeing the derived buffer computed from the dead
    /// allocation and keeping the map bounded by live buffers.
    pub(crate) fn insert(&mut self, id: BufId, v: V) {
        self.map.insert((id.addr, id.len), (id.uid, v));
    }

    /// Number of live entries (slots), for tests.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The regression this module exists for: a freed buffer's address (and length) handed back to
    /// a DIFFERENT buffer must MISS, not serve the previous buffer's derived state. Pre-fix — a
    /// bare `HashMap<usize, V>` on `contents()` — this returned `"weights of A"` for B.
    #[test]
    fn recycled_address_does_not_serve_the_previous_buffers_value() {
        let mut c: IdCache<&str> = IdCache::default();
        // Buffer A, dequantized and cached.
        let a = BufId {
            addr: 0xdead_0000,
            len: 4096,
            uid: next_buffer_uid(),
        };
        c.insert(a, "dequant of A");
        assert_eq!(c.get(a), Some(&"dequant of A"));

        // A is freed; the very next same-sized allocation gets the SAME address back.
        let b = BufId {
            addr: a.addr,
            len: a.len,
            uid: next_buffer_uid(),
        };
        assert_ne!(a.uid, b.uid, "a fresh allocation must get a fresh uid");
        assert_eq!(c.get(b), None, "stale value served for a recycled address");

        // Caching B replaces A's entry rather than growing the map.
        c.insert(b, "dequant of B");
        assert_eq!(c.get(b), Some(&"dequant of B"));
        assert_eq!(c.len(), 1, "recycled slot must be replaced, not duplicated");
        // ...and A, if it somehow came back, still cannot read B's value.
        assert_eq!(c.get(a), None);
    }

    /// The Metal cache's key carried NO length, so a recycled address collided even at a different
    /// size. Length is back in the key, but it is not what makes the cache correct — a
    /// same-address, same-length recycle (the common case: per-layer banks of a uniform model are
    /// byte-identical in size) must still miss.
    #[test]
    fn length_alone_is_not_enough() {
        let mut c: IdCache<u32> = IdCache::default();
        let a = BufId {
            addr: 0x1000,
            len: 1 << 20,
            uid: next_buffer_uid(),
        };
        c.insert(a, 1);
        // Same address, DIFFERENT length (what the old length-less Metal key got wrong).
        let diff_len = BufId {
            addr: a.addr,
            len: 1 << 21,
            uid: next_buffer_uid(),
        };
        assert_eq!(c.get(diff_len), None);
        // Same address, SAME length (what a length in the key would still get wrong).
        let same_len = BufId {
            addr: a.addr,
            len: a.len,
            uid: next_buffer_uid(),
        };
        assert_eq!(c.get(same_len), None);
    }

    /// Distinct live buffers keep distinct entries, and a hit is repeatable — the cache must still
    /// actually cache (a weight is dequantized once per generation, not once per forward).
    #[test]
    fn distinct_live_buffers_each_keep_their_own_value() {
        let mut c: IdCache<u32> = IdCache::default();
        let ids: Vec<BufId> = (0..4)
            .map(|i| BufId {
                addr: 0x2000 + i * 0x1000,
                len: 512,
                uid: next_buffer_uid(),
            })
            .collect();
        for (i, id) in ids.iter().enumerate() {
            c.insert(*id, i as u32);
        }
        assert_eq!(c.len(), 4);
        for _ in 0..3 {
            for (i, id) in ids.iter().enumerate() {
                assert_eq!(c.get(*id), Some(&(i as u32)));
            }
        }
    }

    /// uids are unique and never reused, which is the whole guarantee the cache leans on.
    #[test]
    fn uids_are_unique_and_monotonic() {
        let a = next_buffer_uid();
        let b = next_buffer_uid();
        assert!(b > a);
        let n = 1000;
        let uids: std::collections::HashSet<u64> = (0..n).map(|_| next_buffer_uid()).collect();
        assert_eq!(uids.len(), n, "uid source handed out a duplicate");
        assert!(!uids.contains(&a) && !uids.contains(&b));
    }
}
