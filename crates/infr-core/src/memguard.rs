//! The commit-charge circuit breaker (2026-10-19 memory-bomb conviction, `moe4all-main\agents.md`).
//!
//! A gray-zone prefill (every chunk below the layer-ring threshold) turned the expert pager into a
//! miss storm whose host-visible staging grew at ~7 GiB/s across thousands of small allocations —
//! no downstream budget caught it, and the box hit ~145 GiB committed until the display driver AV'd.
//! The breaker therefore watches the process↔system boundary directly: Windows commit headroom
//! (`GlobalMemoryStatusEx::ullAvailPageFile`, which WDDM also charges device memory against). When
//! headroom falls under the configured water line the process exits immediately with [`EXIT_CODE`]
//! — a deliberate, self-reported hard stop, because the alternative is a paging storm or a driver
//! crash that takes the desktop with it.
//!
//! Two tripwires: a background poller (bounds slow leaks) and inline [`check`] calls at the known
//! floodgates (host staging sites), which catch a GiB-per-second ramp between polls. Platforms
//! without a commit probe answer `None` from [`crate::hostmem::commit_available_bytes`] and the
//! breaker stays inert, like every other host-memory policy here.
//!
//! **Commit headroom alone is blind on big-budget boxes.** Measured 2026-10-20: a perfectly
//! healthy boot on the 96 GiB-RAM conviction machine bottoms out at 47.9 GiB commit headroom
//! (the engine commits >100 GiB legitimately — mmap'd clean sources, the 50 GiB host store, WDDM
//! device charge), while the 145 GiB bomb died at only ~3.5 GiB less. What actually kills such a
//! box is PHYSICAL RAM: the active working set overruns RAM and the machine enters a paging
//! storm. [`configure`] therefore takes a second, usually far more sensitive water line on
//! available physical RAM ([`crate::hostmem::available_bytes`]); the steady-state resident budget
//! keeps ~15 GiB of it on this box, and a miss storm drives it to zero.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

/// Exit code of a breaker trip — distinct from signal deaths (130/143) and panics (101), so test
/// scripts can tell "the guard saved the box" from any other failure.
pub const EXIT_CODE: i32 = 87;

/// Water line when the knob is unset: guards hard commit exhaustion only. Miss-storm testing wants
/// a much taller line (e.g. `50%` on a big-RAM box) so the trip lands before RAM itself runs out.
pub const DEFAULT_HEADROOM_BYTES: u64 = 8u64 << 30;

static HEADROOM: AtomicU64 = AtomicU64::new(0);
static RAM_HEADROOM: AtomicU64 = AtomicU64::new(0);
static POLLER: AtomicBool = AtomicBool::new(false);

/// Arm the breaker (or move its water lines). Both lines disable independently at `0`;
/// `ram_headroom_bytes` watches available PHYSICAL RAM (the sensitive axis on big-budget boxes).
pub fn configure(headroom_bytes: u64, ram_headroom_bytes: u64, poll_ms: u64) {
    HEADROOM.store(headroom_bytes, Ordering::Release);
    RAM_HEADROOM.store(ram_headroom_bytes, Ordering::Release);
    if headroom_bytes == 0 && ram_headroom_bytes == 0 {
        tracing::info!("[mem-guard] commit circuit breaker DISABLED (both water lines at 0)");
        return;
    }
    tracing::info!(
        headroom_bytes,
        ram_headroom_bytes,
        poll_ms,
        "[mem-guard] circuit breaker armed: below-water-line commit or physical-RAM headroom exits the process"
    );
    if POLLER.swap(true, Ordering::AcqRel) {
        return; // one poller per process; it re-reads the water line on every tick
    }
    let poll = Duration::from_millis(poll_ms.max(50));
    let _ = std::thread::Builder::new()
        .name("mem-guard".to_owned())
        .spawn(move || loop {
            std::thread::sleep(poll);
            check();
        });
}

/// The current water line, for diagnostics.
pub fn headroom() -> u64 {
    HEADROOM.load(Ordering::Relaxed)
}

/// Inline tripwire for allocation floodgates: one `GlobalMemoryStatusEx`, cheap enough for every
/// staging allocation. Trips the process when either headroom is under its water line.
pub fn check() {
    let headroom = HEADROOM.load(Ordering::Relaxed);
    if headroom != 0 {
        if let Some(avail) = crate::hostmem::commit_available_bytes() {
            if avail < headroom {
                trip("commit", avail, headroom);
            }
        }
    }
    let ram_headroom = RAM_HEADROOM.load(Ordering::Relaxed);
    if ram_headroom != 0 {
        if let Some(avail) = crate::hostmem::available_bytes() {
            if avail < ram_headroom {
                trip("physical-RAM", avail, ram_headroom);
            }
        }
    }
}

fn trip(axis: &str, avail: u64, water_line: u64) -> ! {
    tracing::error!(
        available_bytes = avail,
        water_line_bytes = water_line,
        "[mem-guard] TRIP: {axis} headroom below the water line - exiting to protect the box"
    );
    // print-ok: last-words-on-hard-exit — `process::exit` skips Drop/flush, so the tracing
    // error above may never reach a sink; this stderr line is the only guaranteed witness of
    // WHY exit 87 fired.
    eprintln!(
        "[mem-guard] {axis} headroom {} MiB < {} MiB water line; aborting (exit {EXIT_CODE})",
        avail >> 20,
        water_line >> 20
    );
    // `process::exit` skips Drop guards, so the pager profile (whose counters are exactly the
    // evidence of WHICH allocation funnel drove the ramp) must be flushed by hand here.
    // 2026-10-21: the first post-fix repros TRIPped with no profile at all, losing the forensics.
    crate::pager_profile::print_summary_if_enabled();
    // Deliberate hard exit with no unwinding: a half-fed DMA pipeline outlives the ramp this
    // is meant to stop, and WDDM releases every allocation on process exit anyway.
    std::process::exit(EXIT_CODE);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_breaker_never_trips() {
        configure(0, 0, 60_000);
        assert_eq!(headroom(), 0);
        // Both lines at 0 short-circuit before the platform probe: safe on any OS, even with the
        // real environment nearly committed.
        check();
    }

    #[test]
    fn headroom_round_trips_through_configure() {
        configure(1 << 40, 1 << 30, 60_000);
        assert_eq!(headroom(), 1 << 40);
        configure(0, 0, 60_000);
    }
}