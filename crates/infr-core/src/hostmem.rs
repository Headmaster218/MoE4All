//! How much host memory a weight arena may take, and how much there is to take
//! (`docs/disk-streaming-plan.md` §7 question 3).
//!
//! The DRAM tier's arena is ANONYMOUS, non-evictable memory — the expensive kind backlog B30
//! measured — so its size cannot be a guess. Two separate concerns live here:
//!
//! - [`available_bytes`], a PLATFORM probe of what could be committed right now. It is deliberately
//!   allowed to answer "I do not know" rather than estimate, because an over-estimate here is an
//!   out-of-memory kill or a swap storm mid-generation, not a slow run.
//! - [`auto_arena_bytes`], the PURE arithmetic that turns that answer into a budget. Separate so
//!   the policy can be tested without a machine that happens to have the right amount of RAM free.

/// Host memory that could be committed right now, or `None` where this platform has no probe.
///
/// `None` is a real answer and callers must treat it as one: it means "do not auto-size", not
/// "assume zero" and not "assume plenty". The tier then stays off unless the user names a budget,
/// which is the conservative failure — a model that would have streamed simply does not, and says
/// so, instead of the process being killed part-way through a generation.
///
/// **Linux** reads `MemAvailable` from `/proc/meminfo`, which is the kernel's own estimate of what
/// a new allocation can have without swapping — it already accounts for reclaimable page cache, so
/// it is exactly the figure this tier wants and not something derivable from `MemTotal`.
///
/// **Windows** reads `ullAvailPhys` from `GlobalMemoryStatusEx`. Every other platform answers
/// `None` today; macOS would need `host_statistics64`'s free/inactive/purgeable split.
///
/// **A cgroup memory limit overrides it.** `/proc/meminfo` is host-wide and knows nothing about the
/// limit a container or a `systemd-run --scope -p MemoryMax=` puts on this process — measured on
/// this box, an 8 GB scope still reports 54.6 GB available. Sizing an anonymous arena from that
/// figure is an OOM kill, so the smaller of the two wins.
pub fn available_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let host = parse_mem_available(&text)?;
        Some(match cgroup_headroom() {
            Some(limited) => host.min(limited),
            None => host,
        })
    }
    #[cfg(windows)]
    {
        Some(windows_memory_status()?.ullAvailPhys)
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

#[cfg(windows)]
fn windows_memory_status() -> Option<windows::Win32::System::SystemInformation::MEMORYSTATUSEX> {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = MEMORYSTATUSEX::default();
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    unsafe { GlobalMemoryStatusEx(&mut status).ok()? };
    Some(status)
}

/// Memory this process may still commit before its cgroup kills it, or `None` when no ancestor
/// limits it.
///
/// Walks from the process's own cgroup up to the root, because the binding limit is the TIGHTEST
/// of the ancestors and not necessarily the leaf's — a container's leaf is often unlimited while
/// the pod slice above it is capped. Both hierarchy versions are read: v2's `memory.max` /
/// `memory.current`, and v1's `memory.limit_in_bytes` / `memory.usage_in_bytes`, whose "no limit"
/// is a sentinel near `u64::MAX` rather than a word.
#[cfg(target_os = "linux")]
fn cgroup_headroom() -> Option<u64> {
    let own = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mut tightest: Option<u64> = None;
    for line in own.lines() {
        // v2: `0::/a/b`. v1: `N:memory:/a/b` (other controllers are not ours to read).
        let mut parts = line.splitn(3, ':');
        let hier = parts.next()?;
        let ctrl = parts.next()?;
        let path = parts.next()?;
        let (root, max_file, cur_file) = if hier == "0" && ctrl.is_empty() {
            ("/sys/fs/cgroup", "memory.max", "memory.current")
        } else if ctrl.split(',').any(|c| c == "memory") {
            (
                "/sys/fs/cgroup/memory",
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
            )
        } else {
            continue;
        };
        // From the leaf upward: `/a/b`, `/a`, `/`.
        let mut at = std::path::PathBuf::from(root);
        at.push(path.trim_start_matches('/'));
        loop {
            let max = read_u64(&at.join(max_file));
            let cur = read_u64(&at.join(cur_file));
            if let (Some(max), Some(cur)) = (max, cur) {
                // v1 spells "unlimited" as a huge number; treat anything past the host's plausible
                // range as no limit rather than as headroom nobody has.
                if max < u64::MAX / 2 {
                    let free = max.saturating_sub(cur);
                    tightest = Some(tightest.map_or(free, |t: u64| t.min(free)));
                }
            }
            if at.as_os_str().len() <= root.len() || !at.pop() {
                break;
            }
        }
    }
    tightest
}

/// One cgroup value file: a decimal, or `None` for the `max` sentinel, a missing file, or junk.
#[cfg(target_os = "linux")]
fn read_u64(path: &std::path::Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Pull `MemAvailable` (in kB, as `/proc/meminfo` always reports it) out of the file's text.
///
/// Split from the read so the parse is testable against a literal — the one machine this runs on
/// cannot produce a file with the field missing, which is the case worth checking.
#[cfg(any(target_os = "linux", test))]
fn parse_mem_available(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    // `MemAvailable:   12345678 kB`
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Never take the last of the machine's memory: the larger of this and [`HEADROOM_FRACTION`] of
/// what is available is left alone.
///
/// The arena is not the only thing the run needs host memory for — the pinned staging ring, the
/// CPU backend's activations, the tokenizer, and whatever else shares the box. A fixed floor
/// matters on small hosts where a fraction rounds to nothing.
const HEADROOM_MIN: u64 = 1 << 30;

/// The share of available memory left unclaimed on a large host, where a 1 GiB floor would be
/// negligible. Reciprocal — `available / HEADROOM_FRACTION`.
const HEADROOM_FRACTION: u64 = 8;

/// Below this an arena is not worth building: the tier costs a copy per streamed block, and a
/// budget this small holds so little of a model that the hit rate cannot pay for it.
const MIN_USEFUL: u64 = 256 << 20;

/// The arena budget to take, given what is available and what the run has already spoken for.
///
/// - `available` — from [`available_bytes`].
/// - `committed` — host bytes this run will already hold that `available` does not know about. On a
///   UNIFIED-memory device (iGPU, APU, Metal) the "VRAM" budget is carved out of this same physical
///   RAM, so passing it here is what stops the two tiers from spending the same bytes twice. Zero
///   on a discrete GPU, whose VRAM is a separate pool.
/// - `pageable` — total bytes of the weights that could be paged. Budgeting past this buys nothing:
///   every block would already be resident.
///
/// Returns `0` when nothing worth having is left, which callers treat as "stay on the mmap path".
pub fn auto_arena_bytes(available: u64, committed: u64, pageable: u64) -> u64 {
    let headroom = HEADROOM_MIN.max(available / HEADROOM_FRACTION);
    let usable = available
        .saturating_sub(committed)
        .saturating_sub(headroom)
        .min(pageable);
    if usable < MIN_USEFUL {
        return 0;
    }
    usable
}

/// What a caller should do about a host weight arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaPlan {
    /// Build an arena of this many bytes.
    Take(u64),
    /// Build a tier that CACHES NOTHING — see [`crate::hostpager::HostPager::stream_only`]. The
    /// blocks still come from explicit positioned reads rather than the GGUF mapping, which is the
    /// whole point on a unified-memory device: the arena above is already GPU-accessible RAM, so
    /// the only thing missing beneath it is a reader that does not go through a page cache
    /// evicting by recency.
    StreamOnly,
    /// Keep the zero-copy mmap path, for this reason. Every reason is something the caller should
    /// SAY — a run that quietly did not page when it needed to is the confusing case.
    Skip(Skip),
}

/// Why a host arena was not built. Distinguished rather than collapsed to `None` because the
/// caller's message differs per case, and "we cannot tell" must never read as "it fits".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
    /// The weights fit the memory available — mmap is zero-copy and strictly better.
    Fits,
    /// No host-memory probe on this platform, so nothing can be sized. See [`available_bytes`].
    NoProbe,
    /// Streaming is needed but too little memory is free to seat a useful arena.
    TooLittle,
    /// The user turned it off by name (`paging.dram = 0`).
    Disabled,
}

/// What the user's `paging.dram` asked for. A budget of ZERO is not "no budget" — it is the
/// explicit OFF switch, and it has to be distinguishable from unset now that unset means "size it
/// yourself". Without it there is no way to A/B the tier against the mmap path it replaces, and no
/// way for a user to refuse an arena on a machine where the automatic answer is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requested {
    /// Nothing set — size from what the host can spare.
    Auto,
    /// `paging.dram = 0` — keep the mmap path whatever the fit test says.
    Off,
    /// A budget named by the user, which wins over every automatic rung.
    Fixed(u64),
    /// `paging.dram_bypass` — no host cache at all: blocks are read from disk straight into the
    /// arena above. A size cannot express this, which is why it is its own state.
    Bypass,
}

impl Requested {
    /// Read a resolved `paging.dram` / `paging.dram_bypass` pair: bypass wins (it says something a
    /// size cannot), then `None` unset, `Some(0)` off, `Some(n)` fixed.
    pub fn from_config(v: Option<u64>, bypass: bool) -> Self {
        if bypass {
            return Self::Bypass;
        }
        match v {
            None => Self::Auto,
            Some(0) => Self::Off,
            Some(n) => Self::Fixed(n),
        }
    }
}

/// The arena plan for a run that has ALREADY been decided to stream — the Vulkan dense and MoE
/// tiers, whose callers only reach them once residency was rejected.
///
/// `explicit` (the user's `paging.dram`) always wins, including on unified memory: forcing this
/// path on hardware that would not choose it is how it gets tested at all.
pub fn streaming_arena_plan(
    explicit: Requested,
    available: Option<u64>,
    unified: bool,
    pageable: u64,
) -> ArenaPlan {
    match explicit {
        // `Bypass` outranks a size: it is the one that says "no host cache at all", which a
        // number cannot express. It exists so the unified-memory shape can be exercised on a
        // discrete GPU, which is the only hardware this is developed on.
        Requested::Bypass => return ArenaPlan::StreamOnly,
        Requested::Fixed(b) => return ArenaPlan::Take(b),
        Requested::Off => return ArenaPlan::Skip(Skip::Disabled),
        Requested::Auto => {}
    }
    if unified {
        return ArenaPlan::StreamOnly;
    }
    let Some(available) = available else {
        return ArenaPlan::Skip(Skip::NoProbe);
    };
    match auto_arena_bytes(available, 0, pageable) {
        0 => ArenaPlan::Skip(Skip::TooLittle),
        n => ArenaPlan::Take(n),
    }
}

/// The arena plan for a backend with no VRAM ladder to decide for it — the CPU one, which must ask
/// whether the weights fit host memory itself.
///
/// The extra rung over [`streaming_arena_plan`] is [`Skip::Fits`]: when the weights fit, the mmap
/// path is zero-copy and an arena could only add copies, so paging a model that fits would be a
/// regression. `explicit` still wins over that test in both directions.
pub fn cpu_arena_plan(explicit: Requested, available: Option<u64>, pageable: u64) -> ArenaPlan {
    match explicit {
        Requested::Fixed(b) => return ArenaPlan::Take(b),
        Requested::Off => return ArenaPlan::Skip(Skip::Disabled),
        // Bypassing the host cache means "read straight into the tier above", and for the CPU
        // backend there IS no tier above — this arena is the only one. Reading through to nothing
        // would be a pure regression on the mapping, so the flag simply keeps the mmap path.
        Requested::Bypass => return ArenaPlan::Skip(Skip::Disabled),
        Requested::Auto => {}
    }
    let Some(available) = available else {
        return ArenaPlan::Skip(Skip::NoProbe);
    };
    if pageable <= available {
        return ArenaPlan::Skip(Skip::Fits);
    }
    match auto_arena_bytes(available, 0, pageable) {
        0 => ArenaPlan::Skip(Skip::TooLittle),
        n => ArenaPlan::Take(n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn parses_mem_available_in_kb() {
        let text = "MemTotal:       65780000 kB\nMemFree:         2000000 kB\n\
                    MemAvailable:   43000000 kB\nBuffers:          100000 kB\n";
        assert_eq!(parse_mem_available(text), Some(43_000_000 * 1024));
    }

    /// A kernel too old to report `MemAvailable` must produce `None`, not a figure derived from
    /// `MemTotal` — auto-sizing against total memory would commit the page cache's share too.
    #[test]
    fn a_file_without_the_field_is_unknown() {
        let text = "MemTotal:       65780000 kB\nMemFree:         2000000 kB\n";
        assert_eq!(parse_mem_available(text), None);
    }

    #[test]
    fn a_malformed_field_is_unknown() {
        assert_eq!(parse_mem_available("MemAvailable:   plenty kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable:\n"), None);
    }

    /// The probe must agree with itself on the machine running the tests: a plausible, non-zero
    /// figure no larger than what the same file reports as total.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_live_probe_is_plausible() {
        let avail = available_bytes().expect("linux always has /proc/meminfo");
        let text = std::fs::read_to_string("/proc/meminfo").expect("meminfo");
        let total: u64 = text
            .lines()
            .find(|l| l.starts_with("MemTotal:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .expect("MemTotal")
            * 1024;
        assert!(avail > 0, "available must be non-zero");
        assert!(avail <= total, "available {avail} exceeds total {total}");
    }

    #[cfg(windows)]
    #[test]
    fn the_windows_live_probe_is_plausible() {
        let avail = available_bytes().expect("windows GlobalMemoryStatusEx should answer");
        let status = windows_memory_status().expect("GlobalMemoryStatusEx");
        assert!(
            status.ullTotalPhys > 0,
            "total physical memory must be non-zero"
        );
        assert!(
            avail <= status.ullTotalPhys,
            "available {avail} exceeds total {}",
            status.ullTotalPhys
        );
    }

    /// Headroom is the point: the budget never equals what is available, however much there is.
    #[test]
    fn headroom_is_always_left() {
        for &avail in &[2 * GIB, 8 * GIB, 64 * GIB, 512 * GIB] {
            let got = auto_arena_bytes(avail, 0, u64::MAX);
            assert!(got < avail, "took all {avail} bytes");
            assert!(
                avail - got >= HEADROOM_MIN,
                "left less than the floor at {avail}: took {got}"
            );
        }
    }

    /// On a large host the fraction is what binds, not the floor.
    #[test]
    fn a_large_host_leaves_the_fraction() {
        let avail = 64 * GIB;
        assert_eq!(auto_arena_bytes(avail, 0, u64::MAX), avail - avail / 8);
    }

    /// Unified memory: the VRAM budget comes out of the same RAM, so it must reduce the arena
    /// one-for-one. Without this the two tiers each plan to use the same bytes.
    #[test]
    fn committed_bytes_reduce_the_budget_one_for_one() {
        let avail = 32 * GIB;
        let free = auto_arena_bytes(avail, 0, u64::MAX);
        let with = auto_arena_bytes(avail, 4 * GIB, u64::MAX);
        assert_eq!(
            free - with,
            4 * GIB,
            "committed bytes must come straight off"
        );
    }

    /// Never budget past what could actually be paged.
    #[test]
    fn the_pageable_total_is_a_ceiling() {
        assert_eq!(auto_arena_bytes(64 * GIB, 0, 3 * GIB), 3 * GIB);
    }

    /// **The flags must keep working.** A machine big enough to hold the model resident is exactly
    /// the machine streaming has to be tested on, so an explicit budget wins over every automatic
    /// rung — including the fits-in-RAM test, the no-probe case and unified memory.
    #[test]
    fn an_explicit_budget_always_wins() {
        let forced = Requested::Fixed(3 * GIB);
        // CPU: a model that comfortably fits is still paged when asked.
        assert_eq!(
            cpu_arena_plan(forced, Some(64 * GIB), GIB),
            ArenaPlan::Take(3 * GIB)
        );
        // ...and with no probe at all.
        assert_eq!(cpu_arena_plan(forced, None, GIB), ArenaPlan::Take(3 * GIB));
        // Streaming tiers: honoured on unified memory, which auto-sizing declines.
        assert_eq!(
            streaming_arena_plan(forced, Some(64 * GIB), true, 40 * GIB),
            ArenaPlan::Take(3 * GIB)
        );
        assert_eq!(
            streaming_arena_plan(forced, None, false, 40 * GIB),
            ArenaPlan::Take(3 * GIB)
        );
    }

    /// A budget of ZERO turns the tier off by name, on both paths and whatever the automatic rungs
    /// would have decided. Without this there is no way to A/B the tier against the mmap path it
    /// replaces once auto-sizing turns it on by itself, and `0` would otherwise read as "unset".
    #[test]
    fn a_zero_budget_is_the_off_switch() {
        assert_eq!(Requested::from_config(Some(0), false), Requested::Off);
        assert_eq!(
            cpu_arena_plan(Requested::Off, Some(GIB), 200 * GIB),
            ArenaPlan::Skip(Skip::Disabled)
        );
        assert_eq!(
            streaming_arena_plan(Requested::Off, Some(64 * GIB), false, 200 * GIB),
            ArenaPlan::Skip(Skip::Disabled)
        );
        // ...and it is distinct from unset, which on that same host DOES build one.
        assert!(matches!(
            streaming_arena_plan(Requested::Auto, Some(64 * GIB), false, 200 * GIB),
            ArenaPlan::Take(_)
        ));
    }

    #[test]
    fn from_config_maps_unset_and_sizes() {
        assert_eq!(Requested::from_config(None, false), Requested::Auto);
        assert_eq!(
            Requested::from_config(Some(42), false),
            Requested::Fixed(42)
        );
    }

    /// Requirement one: if it fits, everything stays resident. Paging a model that fits would add
    /// a copy per block over the zero-copy mapping and buy nothing.
    #[test]
    fn a_model_that_fits_is_not_paged() {
        assert_eq!(
            cpu_arena_plan(Requested::Auto, Some(64 * GIB), 8 * GIB),
            ArenaPlan::Skip(Skip::Fits)
        );
        // Exactly fitting still counts as fitting.
        assert_eq!(
            cpu_arena_plan(Requested::Auto, Some(8 * GIB), 8 * GIB),
            ArenaPlan::Skip(Skip::Fits)
        );
    }

    /// Requirement two: a model that does NOT fit streams, without being asked to.
    #[test]
    fn a_model_that_does_not_fit_streams() {
        match cpu_arena_plan(Requested::Auto, Some(32 * GIB), 200 * GIB) {
            ArenaPlan::Take(n) => assert!(n > 0 && n < 32 * GIB, "implausible budget {n}"),
            other => panic!("an over-sized model must stream, got {other:?}"),
        }
    }

    /// Unified memory reads DISK → GPU-accessible RAM with no host cache: the arena above is
    /// already in the one pool of RAM, so caching beneath it would hold a second copy the device
    /// cannot read in place — but the reads themselves must still be block-granular rather than
    /// left to the mapping, which is what `StreamOnly` expresses.
    #[test]
    fn unified_memory_streams_without_caching() {
        assert_eq!(
            streaming_arena_plan(Requested::Auto, Some(64 * GIB), true, 40 * GIB),
            ArenaPlan::StreamOnly
        );
        // It must NOT collapse to "keep the mmap path" — that is the thing it replaces.
        assert_ne!(
            streaming_arena_plan(Requested::Auto, Some(64 * GIB), true, 40 * GIB),
            ArenaPlan::Skip(Skip::Fits)
        );
        // A host with no memory to spare still streams on unified memory, because the decision
        // does not depend on having any to give.
        assert_eq!(
            streaming_arena_plan(Requested::Auto, Some(GIB), true, 40 * GIB),
            ArenaPlan::StreamOnly
        );
        // The same host WITHOUT unified memory caches instead — otherwise this test would pass
        // for a version that simply never auto-sizes.
        assert!(matches!(
            streaming_arena_plan(Requested::Auto, Some(64 * GIB), false, 40 * GIB),
            ArenaPlan::Take(_)
        ));
    }

    /// "Cannot tell" must never be reported as "fits": the two lead to opposite advice.
    #[test]
    fn no_probe_is_distinct_from_fitting() {
        assert_eq!(
            cpu_arena_plan(Requested::Auto, None, 200 * GIB),
            ArenaPlan::Skip(Skip::NoProbe)
        );
        assert_eq!(
            streaming_arena_plan(Requested::Auto, None, false, 200 * GIB),
            ArenaPlan::Skip(Skip::NoProbe)
        );
    }

    /// A host with nothing to spare declines rather than building a useless arena.
    #[test]
    fn a_squeezed_host_declines() {
        assert_eq!(auto_arena_bytes(GIB, 0, u64::MAX), 0);
        assert_eq!(auto_arena_bytes(64 * GIB, 63 * GIB, u64::MAX), 0);
        // Just under the useful floor, with headroom accounted for.
        assert_eq!(auto_arena_bytes(2 * GIB, 0, MIN_USEFUL - 1), 0);
    }
}
