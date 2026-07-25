//! Race-free `INFR_*` environment access for TESTS.
//!
//! The process environment is global mutable state, but `cargo test` runs a binary's tests on a
//! thread pool: two tests that `set_var` at the same time interleave, and a test that sets a knob
//! and never restores it leaks that knob into every test that runs after it. Both failure modes are
//! ORDER-DEPENDENT — they reproduce on one machine's core count and vanish on another's, which is
//! exactly the kind of flake that gets re-run until it passes instead of fixed. (The live example
//! this module was written for: `infr-vulkan`'s two `attn_flash_*_dequant_parity` tests both drive
//! `INFR_FLASH_SPLITS`, and whichever finished first `remove_var`-ed it out from under the other —
//! they passed individually and failed whenever they overlapped.)
//!
//! [`EnvGuard`] fixes both at once:
//!
//! - **serialization** — constructing one takes a process-wide lock, so at most one test is inside
//!   an env-mutating section at a time. The lock is poison-tolerant: a panicking test must not
//!   wedge every later one.
//! - **restoration** — the guard remembers each key's value at the moment it first touched it and
//!   puts it back on drop, so a test's knobs never outlive it, panic or not.
//!
//! ```no_run
//! # use infr_core::test_env::EnvGuard;
//! let _env = EnvGuard::with([("INFR_TEMP", "0"), ("INFR_NO_THINK", "1")]);
//! // ... the knobs are set, no other guarded test is running, and both are restored on drop.
//! ```
//!
//! **This is a serializer, not a substitute for plumbing.** A knob that its reader memoizes (a
//! `OnceLock`, a `static`) latches the FIRST value any test observed, and no guard can undo that —
//! such a test silently measures nothing. Prefer testing the parameterized function directly (the
//! way `tier::EnvRows::resolve` / `budget::flag_from` take the raw string, leaving only a one-line
//! `std::env::var` wrapper untested) and reach for a guard only when the knob has to cross a real
//! API boundary.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// The process-wide env lock. Every [`EnvGuard`] holds it for its lifetime.
fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// An exclusive, self-restoring handle on the process environment — see the [module docs](self).
///
/// Hold it for as long as the knobs must stay set (bind it to a named local: `let _env = ...`, not
/// `let _ = ...`, which drops it immediately and restores the environment before the test body
/// even starts).
///
/// **One per test.** The lock is a plain `Mutex`, so it is NOT re-entrant: constructing a second
/// guard while the first is alive deadlocks that test. Add keys to the guard you already hold
/// ([`set`](Self::set) / [`unset`](Self::unset)) instead.
pub struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    /// Every key this guard has touched → its value before the first touch (`None` = was unset).
    prior: HashMap<String, Option<String>>,
}

impl EnvGuard {
    /// Take the env lock without changing anything yet — for a test that only needs the
    /// serialization (e.g. it reads a knob, or calls code that does).
    pub fn new() -> Self {
        Self {
            _lock: env_lock().lock().unwrap_or_else(|e| e.into_inner()),
            prior: HashMap::new(),
        }
    }

    /// Take the lock and set the given `(key, value)` pairs.
    pub fn with<K, V, I>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut g = Self::new();
        for (k, v) in vars {
            g.set(k.as_ref(), v.as_ref());
        }
        g
    }

    /// Set one variable, remembering its prior value the first time this guard touches the key.
    pub fn set(&mut self, key: &str, value: &str) -> &mut Self {
        self.remember(key);
        std::env::set_var(key, value);
        self
    }

    /// Remove one variable (so the code under test sees it unset), remembering its prior value.
    pub fn unset(&mut self, key: &str) -> &mut Self {
        self.remember(key);
        std::env::remove_var(key);
        self
    }

    fn remember(&mut self, key: &str) {
        self.prior
            .entry(key.to_string())
            .or_insert_with(|| std::env::var(key).ok());
    }
}

impl Default for EnvGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // Restore BEFORE the lock is released (field order: `prior` is dropped after this body,
        // `_lock` after that), so the next guard never observes this one's values.
        for (key, prior) in self.prior.drain() {
            match prior {
                Some(v) => std::env::set_var(&key, v),
                None => std::env::remove_var(&key),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_and_restores_a_previously_unset_key() {
        const K: &str = "INFR_TEST_ENV_GUARD_UNSET";
        assert!(std::env::var(K).is_err(), "test key must start unset");
        {
            let _g = EnvGuard::with([(K, "42")]);
            assert_eq!(std::env::var(K).unwrap(), "42");
        }
        assert!(std::env::var(K).is_err(), "unset key must be REMOVED again");
    }

    /// A key that was ALREADY set before the guard existed must come back with its old value —
    /// the case that makes a guarded test invisible to whatever runs next. (The pre-set happens
    /// outside a guard on a key nothing else uses; the guard cannot be nested, it deadlocks.)
    #[test]
    fn restores_a_previously_set_key_to_its_old_value() {
        const K: &str = "INFR_TEST_ENV_GUARD_SET";
        std::env::set_var(K, "before");
        {
            let mut g = EnvGuard::new();
            g.set(K, "during");
            assert_eq!(std::env::var(K).unwrap(), "during");
            // A second write to the same key keeps the ORIGINAL prior, not the intermediate one.
            g.set(K, "during2");
        }
        assert_eq!(
            std::env::var(K).unwrap(),
            "before",
            "must restore old value"
        );
        std::env::remove_var(K);
    }

    #[test]
    fn unset_hides_a_set_key_then_restores_it() {
        const K: &str = "INFR_TEST_ENV_GUARD_HIDE";
        let mut g = EnvGuard::new();
        g.set(K, "visible");
        assert_eq!(std::env::var(K).unwrap(), "visible");
        g.unset(K);
        assert!(std::env::var(K).is_err(), "unset must hide the key");
        drop(g);
        assert!(std::env::var(K).is_err(), "guard-created key must not leak");
    }
}
