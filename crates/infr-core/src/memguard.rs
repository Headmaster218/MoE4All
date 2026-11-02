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
//!
//! **The commit axis guards the plan's overrun, not the plan.** Measured 2026-11-02 on a 96 GiB
//! machine with a 6 GiB pagefile (commit ceiling 102 GiB): a Flash-Next load whose authorized host
//! store alone is 47.17 GiB, plus 31.35 GiB of host-DMA import and the WDDM device charge, settles at
//! ~4.06 GiB of remaining commit — under the plain 8 GiB line, so the breaker killed a legitimate
//! startup twice and the model could not be loaded at all. Consumption the planner already decided is
//! therefore declared up front with [`set_authorized_commit_bytes`] and subtracted from the line
//! ([`commit_line_for`]), floored at [`HARD_FLOOR_BYTES`]. What still trips is exactly what this
//! module is for: allocation BEYOND the plan — a staging floodgate opening at GiB/s — and hard
//! commit exhaustion.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

/// Exit code of a breaker trip — distinct from signal deaths (130/143) and panics (101), so test
/// scripts can tell "the guard saved the box" from any other failure.
pub const EXIT_CODE: i32 = 87;

/// Water line when the knob is unset: guards hard commit exhaustion only. Miss-storm testing wants
/// a much taller line (e.g. `50%` on a big-RAM box) so the trip lands before RAM itself runs out.
///
/// The line is a margin OVER the engine's own authorized host budget (see
/// [`set_authorized_commit_bytes`]), not an absolute claim about the machine: a model that legitimately
/// needs 47 GiB of commit on a 102 GiB-commit box used to be unable to start at all.
pub const DEFAULT_HEADROOM_BYTES: u64 = 8u64 << 30;

/// The absolute floor the commit axis never relaxes below, however large the authorized plan is:
/// at this level of remaining commit the next allocation is the one that wedges the driver.
pub const HARD_FLOOR_BYTES: u64 = 512u64 << 20;

static HEADROOM: AtomicU64 = AtomicU64::new(0);
static RAM_HEADROOM: AtomicU64 = AtomicU64::new(0);
/// Commit the engine has ALREADY decided to spend — the frozen process RAM budget. Consumption up to
/// this figure is the plan, not the anomaly the breaker exists to catch, so it is subtracted from the
/// water line (down to [`HARD_FLOOR_BYTES`]) instead of tripping it.
static AUTHORIZED_COMMIT: AtomicU64 = AtomicU64::new(0);
static POLLER: AtomicBool = AtomicBool::new(false);

/// The commit water line in force for `line` bytes of configured margin against `authorized` bytes of
/// planned consumption. Pure, so the policy is testable without touching the process globals.
pub fn commit_line_for(line: u64, authorized: u64) -> u64 {
    if line == 0 {
        return 0; // axis disabled
    }
    line.saturating_sub(authorized).max(HARD_FLOOR_BYTES)
}

/// Tell the breaker how much commit the engine has authorized for itself (the startup-frozen process
/// RAM budget). Called once from the CLI after configuration resolves, before any allocation.
/// Repeated calls take the latest value; `0` restores the plain absolute line.
pub fn set_authorized_commit_bytes(bytes: u64) {
    AUTHORIZED_COMMIT.store(bytes, Ordering::Release);
    tracing::info!(
        authorized_commit_bytes = bytes,
        "[mem-guard] commit water line will measure growth BEYOND the authorized host budget"
    );
}

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
    let commit_line = commit_line_for(headroom, AUTHORIZED_COMMIT.load(Ordering::Relaxed));
    if commit_line != 0 {
        if let Some(avail) = crate::hostmem::commit_available_bytes() {
            if avail < commit_line {
                trip("commit", avail, commit_line);
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

    /// The commit line measures growth BEYOND what the engine already authorized for itself, so a
    /// model whose planned store legitimately fills the commit space can still start.
    #[test]
    fn commit_line_measures_growth_beyond_the_authorized_plan() {
        let gib = 1u64 << 30;
        // Nothing announced → the configured absolute line, unchanged (old behaviour).
        assert_eq!(commit_line_for(8 * gib, 0), 8 * gib);
        // A partial plan keeps its margin on top.
        assert_eq!(commit_line_for(8 * gib, 6 * gib), 2 * gib);
        // A plan at or above the line — the 47 GiB store on a 96 GiB box — leaves only the
        // exhaustion floor, which is the whole point: 4 GiB of remaining commit is not a leak.
        assert_eq!(commit_line_for(8 * gib, 8 * gib), HARD_FLOOR_BYTES);
        assert_eq!(commit_line_for(8 * gib, 78 * gib), HARD_FLOOR_BYTES);
        // A disabled axis stays disabled whatever the plan says.
        assert_eq!(commit_line_for(0, 40 * gib), 0);
    }

    /// No authorized number, however absurd, relaxes the guard past hard exhaustion — and the
    /// saturating arithmetic must not wrap.
    #[test]
    fn authorized_plan_never_lowers_the_line_below_the_hard_floor() {
        assert_eq!(commit_line_for(1 << 30, u64::MAX), HARD_FLOOR_BYTES);
        assert_eq!(HARD_FLOOR_BYTES, 512 << 20);
    }

    /// The setter is observable through `check()`'s arithmetic without racing the poller: set,
    /// read the composed line, restore.
    #[test]
    fn authorized_commit_bytes_are_sticky_until_replaced() {
        set_authorized_commit_bytes(3 << 30);
        assert_eq!(
            commit_line_for(8 << 30, AUTHORIZED_COMMIT.load(Ordering::Relaxed)),
            5 << 30
        );
        set_authorized_commit_bytes(0);
        assert_eq!(
            commit_line_for(8 << 30, AUTHORIZED_COMMIT.load(Ordering::Relaxed)),
            8 << 30
        );
    }
}
