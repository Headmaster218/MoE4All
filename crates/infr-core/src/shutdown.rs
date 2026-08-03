//! The process-wide graceful-shutdown latch (SIGINT / SIGTERM).
//!
//! **Why this exists.** A GPU submit is not cancellable. Once a command buffer is on the queue the
//! only legal thing a host can do is WAIT for its fence — and if the process dies first, the kernel
//! is left holding a fence nobody will ever reap. On amdgpu that is not theoretical: a `SIGTERM`
//! delivered while an `infr` prefill submit was in flight on an integrated part left the task
//! unkillable in `D` state inside `drm_suballoc_insert` (the queued `SIGKILL` is never delivered
//! because the thread is blocked in the kernel on a GPU fence), and its buffer objects were never
//! released — the whole 2 GiB carveout plus 7.2 GiB of GTT stayed pinned by a dead process, so
//! every subsequent client got `ENOMEM` opening the render node and the GPU vanished from Vulkan
//! enumeration until a reboot. Two `amdgpu_gpu_recover` cycles did not free it.
//!
//! There are driver bugs under that, but the loaded gun is ours: `infr` installed no signal
//! handlers at all, so the default disposition (terminate immediately) fired mid-ioctl. The fix is
//! this latch:
//!
//! * The CLI's signal handler does exactly ONE thing — [`request_shutdown`], a lock-free atomic
//!   compare-exchange. Nothing else in the handler is async-signal-safe, so nothing else is in it.
//! * Everything that can issue GPU work polls [`shutdown_requested`] at the boundary where it is
//!   about to submit MORE work, and stops there. Work already submitted is always drained
//!   (`vkQueueWaitIdle` / fence wait) — never abandoned.
//! * Nobody calls `std::process::exit` on that path. The stack unwinds, the backend's `Drop` runs,
//!   `vkDestroyDevice` happens, and the kernel gets its buffer objects back.
//!
//! The latch is process-wide and monotonic (it never un-sets): the only thing it means is "this
//! process is on its way out, stop starting things". It is deliberately NOT the per-request abort
//! (`infr_llama::sampling::RequestCtx::abort`, which one `serve` request latches when its stop
//! sequence hits) — but that request-level poll ORs this one in, so every existing poll site
//! inherits process shutdown for free.
//!
//! `Relaxed` ordering throughout: there is no other data to publish (the flag IS the message), and
//! a poll loop that observes it one iteration late is fine — the guarantee is "stops promptly", not
//! "stops instantly".

use std::sync::atomic::{AtomicI32, Ordering};

/// The signal number that latched shutdown (`0` means no request). Publishing the request and its
/// signal as one atomic transition prevents observers from seeing shutdown without an exit status.
static SIGNO: AtomicI32 = AtomicI32::new(0);

/// Latch the shutdown request. Returns `true` if THIS call was the one that latched it, `false` if
/// it was already set (i.e. this is a second signal — the caller may then decide the user has given
/// up waiting and force-exit).
///
/// **`signo` must not be 0**, which is this latch's "no request" state rather than a signal —
/// POSIX reserves 0 as the null signal for `kill(pid, 0)` existence probes, so no handler is ever
/// invoked with it. Passing it would compare-exchange 0 for 0: the call reports itself as the one
/// that latched, while [`shutdown_requested`] keeps saying no. Debug-asserted; the CLI's handler
/// is the only caller and always passes the signal it was invoked with.
///
/// **Async-signal-safe**: one `compare_exchange` on a lock-free atomic, nothing else. No allocation,
/// no locking, no I/O. It is called directly from the CLI's `SIGINT`/`SIGTERM` handler. (The
/// `debug_assert` is compiled out of release builds, so the handler's release path stays exactly
/// one atomic instruction.)
pub fn request_shutdown(signo: i32) -> bool {
    debug_assert_ne!(signo, 0, "signal 0 is the unset state, not a signal");
    SIGNO
        .compare_exchange(0, signo, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
}

/// Has a shutdown been requested? One relaxed atomic load — cheap enough for a per-op poll inside
/// the Vulkan recording loop (it is dwarfed by the descriptor writes of a single dispatch).
#[inline]
pub fn shutdown_requested() -> bool {
    SIGNO.load(Ordering::Relaxed) != 0
}

/// The signal that latched the shutdown, or `None` if none has. The CLI turns this into the
/// conventional exit status (130 for `SIGINT`, 143 for `SIGTERM`) once the stack has unwound and
/// the GPU device has been destroyed.
pub fn shutdown_signal() -> Option<i32> {
    match SIGNO.load(Ordering::Relaxed) {
        0 => None,
        s => Some(s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The latch is monotonic and reports the FIRST signal only — the second one is the caller's
    /// cue to force-exit, and must not overwrite the exit status the first one chose.
    #[test]
    fn latch_is_monotonic_and_reports_first_signal() {
        assert!(!shutdown_requested());
        assert_eq!(shutdown_signal(), None);
        assert!(request_shutdown(15), "first signal latches");
        assert!(shutdown_requested());
        assert_eq!(shutdown_signal(), Some(15));
        assert!(!request_shutdown(2), "second signal does NOT latch again");
        assert_eq!(shutdown_signal(), Some(15), "exit status stays the first's");
        assert!(shutdown_requested(), "the latch never un-sets");
    }

    /// Signal 0 is the unset state, so latching it would report success while leaving
    /// [`shutdown_requested`] false. The assert fires BEFORE the compare-exchange, so this test
    /// cannot disturb the global latch the test above drives.
    #[test]
    #[should_panic(expected = "signal 0 is the unset state")]
    #[cfg(debug_assertions)]
    fn signal_zero_is_rejected() {
        let _ = request_shutdown(0);
    }
}
