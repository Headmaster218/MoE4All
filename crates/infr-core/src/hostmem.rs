//! How much host memory a weight arena may take, and how much there is to take
//! (`docs/disk-streaming-plan.md` §7 question 3).
//!
//! The DRAM tier's arena is ANONYMOUS, non-evictable memory — the expensive kind backlog B30
//! measured — so its size cannot be a guess. Two separate concerns live here:
//!
//! - [`available_bytes`], a PLATFORM probe of what could be committed right now. It is deliberately
//!   allowed to answer "I do not know" rather than estimate, because an over-estimate here is an
//!   out-of-memory kill or a swap storm mid-generation, not a slow run.
//! - [`auto_cache_bytes`], the PURE arithmetic that turns that answer into a budget. Separate so
//!   the policy can be tested without a machine that happens to have the right amount of RAM free.
//! - [`process_resident_bytes`], the current process working set used to resolve an explicit
//!   total-process RAM budget into the part that can actually become a weight cache.

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
/// **Windows** reads `GlobalMemoryStatusEx` and uses the smaller of `ullAvailPhys` and
/// `ullAvailPageFile`. The arena needs both reusable physical pages and commit charge;
/// `VirtualAlloc(MEM_COMMIT)` can fail even with free RAM when the process/system commit limit is
/// tighter. Every other platform answers `None` today; macOS would need `host_statistics64`'s
/// free/inactive/purgeable split.
///
/// **A cgroup memory limit overrides it.** `/proc/meminfo` is host-wide and knows nothing about the
/// limit a container or a `systemd-run --scope -p MemoryMax=` puts on this process — measured on
/// this box, an 8 GB scope still reports 54.6 GB available. Sizing an anonymous arena from that
/// figure is an OOM kill, so the smaller of the two wins.
pub fn available_bytes() -> Option<u64> {
    let observed = platform_available_bytes()?;
    match crate::test_resource::active() {
        None => Some(observed),
        Some(profile) => {
            let total = platform_total_bytes()?;
            Some(profile.cap_ram(total, observed).1)
        }
    }
}

fn platform_available_bytes() -> Option<u64> {
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
        let status = windows_memory_status()?;
        Some(windows_available_bytes(
            status.ullAvailPhys,
            status.ullAvailPageFile,
        ))
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

/// Total physical host RAM, used only as the base for percentage-valued total-process budgets.
///
/// This is deliberately separate from [`available_bytes`]: `device.ram_budget=80%` means 80% of
/// the machine's physical RAM as a process-wide target, while automatic cache sizing must continue
/// to use memory available right now and retain its existing headroom policy.
pub fn total_bytes() -> Option<u64> {
    let observed = platform_total_bytes()?;
    Some(match crate::test_resource::active() {
        None => observed,
        Some(profile) => profile.cap_ram(observed, observed).0,
    })
}

fn platform_total_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        parse_mem_total(&text)
    }
    #[cfg(windows)]
    {
        Some(windows_memory_status()?.ullTotalPhys)
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

#[cfg(any(windows, test))]
fn windows_available_bytes(available_phys: u64, available_commit: u64) -> u64 {
    available_phys.min(available_commit)
}

#[cfg(windows)]
fn windows_memory_status() -> Option<windows::Win32::System::SystemInformation::MEMORYSTATUSEX> {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    unsafe { GlobalMemoryStatusEx(&mut status).ok()? };
    Some(status)
}

/// Physical RAM currently resident in this process, including file-backed mappings.
///
/// This is intentionally a working-set measurement, not committed/private virtual memory. An
/// explicit `device.ram_budget` is a total physical-RAM target: cold pages may leave the working set,
/// while a page-file-backed reservation that is not resident must not consume the target twice.
/// Failure is reported as `None`; callers retain the historical fixed-cache fallback on platforms
/// without a probe rather than panicking during model load.
pub fn process_resident_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/status").ok()?;
        parse_process_resident(&text)
    }
    #[cfg(windows)]
    {
        windows_process_resident_bytes()
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

#[cfg(windows)]
fn windows_process_resident_bytes() -> Option<u64> {
    use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows::Win32::System::Threading::GetCurrentProcess;

    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    unsafe {
        GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb).ok()?;
    }
    Some(counters.WorkingSetSize as u64)
}

#[cfg(any(target_os = "linux", test))]
fn parse_process_resident(text: &str) -> Option<u64> {
    let line = text.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
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

#[cfg(any(target_os = "linux", test))]
fn parse_mem_total(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Never take the last of the machine's memory: the larger of this and the bounded
/// [`HEADROOM_FRACTION`] of what is available is left alone.
///
/// The arena is not the only thing the run needs host memory for — the pinned staging ring, the
/// CPU backend's activations, the tokenizer, and whatever else shares the box. A fixed floor
/// matters on small hosts where a fraction rounds to nothing.
const HEADROOM_MIN: u64 = 10 << 30;

/// The share of available memory left unclaimed on a large host, where the fixed floor would be
/// too aggressive. Reciprocal — `available / HEADROOM_FRACTION`.
const HEADROOM_FRACTION: u64 = 4;

/// Keep the proportional reserve useful without permanently stranding excessive RAM on very
/// large hosts.
const HEADROOM_FRACTION_MAX: u64 = 32 << 30;

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
pub fn auto_cache_bytes(available: u64, committed: u64, pageable: u64) -> u64 {
    let proportional = (available / HEADROOM_FRACTION).min(HEADROOM_FRACTION_MAX);
    let headroom = HEADROOM_MIN.max(proportional);
    let usable = available
        .saturating_sub(committed)
        .saturating_sub(headroom)
        .min(pageable);
    if usable < MIN_USEFUL {
        return 0;
    }
    usable
}

/// Convert an explicit total-process RAM target into bytes available to this new host arena.
///
/// `resident` is sampled immediately before the arena is planned. It includes existing model
/// mappings and earlier arenas, so repeated model/session loads share one process-wide ceiling.
/// Where the platform probe fails, zero preserves the historical interpretation as a best-effort
/// fallback; Windows and Linux have live probes and therefore enforce the total-budget meaning.
pub fn cache_bytes_for_total_budget(total: u64, resident: Option<u64>, pageable: u64) -> u64 {
    total.saturating_sub(resident.unwrap_or(0)).min(pageable)
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
    /// The user supplied zero through either explicit host-RAM compatibility spelling.
    Disabled,
}

/// How host RAM should be assigned. A budget of ZERO is not "no budget" — it is the explicit OFF
/// switch, and it has to be distinguishable from unset now that unset means automatic sizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamRequest {
    /// Nothing set — size from what the host can spare.
    Auto,
    /// Canonical `device.ram_budget`: a total-process resident-RAM target.
    TotalProcessBudget(u64),
    /// Compatibility-only `paging.dram`: the old exact host-cache allocation.
    LegacyCacheBudget(u64),
    /// `paging.dram_bypass` — no host cache at all: blocks are read from disk straight into the
    /// arena above. A size cannot express this, which is why it is its own state.
    Bypass,
}

impl RamRequest {
    /// Resolve the canonical total-process budget and the legacy raw-cache override. Bypass wins,
    /// then `device.ram_budget`, then compatibility-only `paging.dram`; zero remains the explicit
    /// off switch in either spelling while retaining which spelling supplied it.
    pub fn from_config(
        total_process_budget: Option<u64>,
        legacy_cache_budget: Option<u64>,
        bypass: bool,
    ) -> Self {
        if bypass {
            return Self::Bypass;
        }
        match total_process_budget {
            Some(n) => Self::TotalProcessBudget(n),
            None => match legacy_cache_budget {
                None => Self::Auto,
                Some(n) => Self::LegacyCacheBudget(n),
            },
        }
    }
}

/// The arena plan for a run that has ALREADY been decided to stream — the Vulkan dense and MoE
/// tiers, whose callers only reach them once residency was rejected.
///
/// A user request always wins, including on unified memory. `TotalProcessBudget` subtracts
/// `process_resident`; `LegacyCacheBudget` preserves the old exact-cache benchmark control. The
/// canonical total budget deliberately bypasses automatic headroom, so it may make the OS page out
/// unrelated cold memory.
pub fn streaming_arena_plan(
    request: RamRequest,
    available: Option<u64>,
    process_resident: Option<u64>,
    unified: bool,
    pageable: u64,
) -> ArenaPlan {
    match request {
        // `Bypass` outranks a size: it is the one that says "no host cache at all", which a
        // number cannot express. It exists so the unified-memory shape can be exercised on a
        // discrete GPU, which is the only hardware this is developed on.
        RamRequest::Bypass => return ArenaPlan::StreamOnly,
        RamRequest::TotalProcessBudget(0) => return ArenaPlan::Skip(Skip::Disabled),
        RamRequest::TotalProcessBudget(total) => {
            let bytes = cache_bytes_for_total_budget(total, process_resident, pageable);
            return if bytes == 0 {
                ArenaPlan::Skip(Skip::TooLittle)
            } else {
                ArenaPlan::Take(bytes)
            };
        }
        RamRequest::LegacyCacheBudget(0) => return ArenaPlan::Skip(Skip::Disabled),
        RamRequest::LegacyCacheBudget(bytes) => return ArenaPlan::Take(bytes),
        RamRequest::Auto => {}
    }
    if unified {
        return ArenaPlan::StreamOnly;
    }
    let Some(available) = available else {
        return ArenaPlan::Skip(Skip::NoProbe);
    };
    match auto_cache_bytes(available, 0, pageable) {
        0 => ArenaPlan::Skip(Skip::TooLittle),
        n => ArenaPlan::Take(n),
    }
}

/// The arena plan for a backend with no VRAM ladder to decide for it — the CPU one, which must ask
/// whether the weights fit host memory itself.
///
/// The extra rung over [`streaming_arena_plan`] is [`Skip::Fits`]: when the weights fit, the mmap
/// path is zero-copy and an arena could only add copies, so paging a model that fits would be a
/// regression. An explicit request still wins over that test in both directions.
pub fn cpu_arena_plan(
    request: RamRequest,
    available: Option<u64>,
    process_resident: Option<u64>,
    pageable: u64,
) -> ArenaPlan {
    match request {
        RamRequest::TotalProcessBudget(0) => return ArenaPlan::Skip(Skip::Disabled),
        RamRequest::TotalProcessBudget(total) => {
            let bytes = cache_bytes_for_total_budget(total, process_resident, pageable);
            return if bytes == 0 {
                ArenaPlan::Skip(Skip::TooLittle)
            } else {
                ArenaPlan::Take(bytes)
            };
        }
        RamRequest::LegacyCacheBudget(0) => return ArenaPlan::Skip(Skip::Disabled),
        RamRequest::LegacyCacheBudget(bytes) => return ArenaPlan::Take(bytes),
        // Bypassing the host cache means "read straight into the tier above", and for the CPU
        // backend there IS no tier above — this arena is the only one. Reading through to nothing
        // would be a pure regression on the mapping, so the flag simply keeps the mmap path.
        RamRequest::Bypass => return ArenaPlan::Skip(Skip::Disabled),
        RamRequest::Auto => {}
    }
    let Some(available) = available else {
        return ArenaPlan::Skip(Skip::NoProbe);
    };
    if pageable <= available {
        return ArenaPlan::Skip(Skip::Fits);
    }
    match auto_cache_bytes(available, 0, pageable) {
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
        assert_eq!(parse_mem_total(text), Some(65_780_000 * 1024));
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
        assert_eq!(parse_mem_total("MemTotal:   plenty kB\n"), None);
    }

    #[test]
    fn parses_process_resident_in_kb() {
        let text = "Name:\tinfr\nVmSize:\t100000 kB\nVmRSS:\t12345 kB\nRssAnon:\t8000 kB\n";
        assert_eq!(parse_process_resident(text), Some(12_345 * 1024));
        assert_eq!(parse_process_resident("Name:\tinfr\n"), None);
    }

    /// The probe must agree with itself on the machine running the tests: a plausible, non-zero
    /// figure no larger than what the same file reports as total.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_live_probe_is_plausible() {
        let avail = available_bytes().expect("linux always has /proc/meminfo");
        let total = total_bytes().expect("linux always has MemTotal");
        assert!(avail > 0, "available must be non-zero");
        assert!(avail <= total, "available {avail} exceeds total {total}");
        let resident = process_resident_bytes().expect("linux always has /proc/self/status");
        assert!(
            resident > 0,
            "the running test process must have resident pages"
        );
        assert!(
            resident <= total,
            "process RSS {resident} exceeds RAM {total}"
        );
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
        assert_eq!(total_bytes(), Some(status.ullTotalPhys));
        assert!(
            avail <= status.ullAvailPageFile,
            "available {avail} exceeds commit headroom {}",
            status.ullAvailPageFile
        );
        let resident = process_resident_bytes().expect("GetProcessMemoryInfo should answer");
        assert!(
            resident > 0,
            "the running test process must have resident pages"
        );
        assert!(
            resident <= status.ullTotalPhys,
            "process working set {resident} exceeds physical RAM {}",
            status.ullTotalPhys
        );
    }

    #[test]
    fn windows_probe_is_bounded_by_physical_and_commit_headroom() {
        assert_eq!(windows_available_bytes(48 * GIB, 20 * GIB), 20 * GIB);
        assert_eq!(windows_available_bytes(12 * GIB, 40 * GIB), 12 * GIB);
    }

    /// Headroom is the point: the budget never equals what is available, however much there is.
    #[test]
    fn headroom_is_always_left() {
        for &avail in &[2 * GIB, 8 * GIB, 64 * GIB, 512 * GIB] {
            let got = auto_cache_bytes(avail, 0, u64::MAX);
            assert!(got < avail, "took all {avail} bytes");
            assert!(
                avail < HEADROOM_MIN || avail - got >= HEADROOM_MIN,
                "left less than the floor at {avail}: took {got}"
            );
        }
    }

    /// On a large host the fraction is what binds, not the floor.
    #[test]
    fn a_large_host_leaves_the_fraction() {
        let avail = 64 * GIB;
        assert_eq!(auto_cache_bytes(avail, 0, u64::MAX), 48 * GIB);
    }

    #[test]
    fn a_workstation_keeps_a_quarter_available() {
        assert_eq!(auto_cache_bytes(52 * GIB, 0, u64::MAX), 39 * GIB);
    }

    #[test]
    fn a_huge_host_caps_the_proportional_reserve() {
        assert_eq!(auto_cache_bytes(512 * GIB, 0, u64::MAX), 480 * GIB);
    }

    /// Unified memory: the VRAM budget comes out of the same RAM, so it must reduce the arena
    /// one-for-one. Without this the two tiers each plan to use the same bytes.
    #[test]
    fn committed_bytes_reduce_the_budget_one_for_one() {
        let avail = 32 * GIB;
        let free = auto_cache_bytes(avail, 0, u64::MAX);
        let with = auto_cache_bytes(avail, 4 * GIB, u64::MAX);
        assert_eq!(
            free - with,
            4 * GIB,
            "committed bytes must come straight off"
        );
    }

    /// Never budget past what could actually be paged.
    #[test]
    fn the_pageable_total_is_a_ceiling() {
        assert_eq!(auto_cache_bytes(64 * GIB, 0, 3 * GIB), 3 * GIB);
        assert_eq!(
            cache_bytes_for_total_budget(50 * GIB, Some(2 * GIB), 3 * GIB),
            3 * GIB
        );
    }

    /// **The flags must keep working.** A machine big enough to hold the model resident is exactly
    /// the machine streaming has to be tested on, so an explicit budget wins over every automatic
    /// rung — including the fits-in-RAM test, the no-probe case and unified memory.
    #[test]
    fn a_legacy_exact_cache_budget_always_wins() {
        let forced = RamRequest::LegacyCacheBudget(3 * GIB);
        // Compatibility mode does not reinterpret historical cache-size experiments as process
        // totals, even when a process working-set measurement is available.
        assert_eq!(
            cpu_arena_plan(forced, Some(64 * GIB), Some(GIB), 8 * GIB),
            ArenaPlan::Take(3 * GIB)
        );
        assert_eq!(
            cpu_arena_plan(forced, None, None, 8 * GIB),
            ArenaPlan::Take(3 * GIB)
        );
        // Streaming tiers: honoured on unified memory, which auto-sizing declines.
        assert_eq!(
            streaming_arena_plan(forced, Some(64 * GIB), Some(GIB), true, 40 * GIB),
            ArenaPlan::Take(3 * GIB)
        );
        assert_eq!(
            streaming_arena_plan(forced, None, None, false, 40 * GIB),
            ArenaPlan::Take(3 * GIB)
        );
    }

    #[test]
    fn an_oversized_explicit_budget_bypasses_automatic_headroom() {
        let forced = RamRequest::TotalProcessBudget(50 * GIB);
        assert_eq!(
            streaming_arena_plan(forced, Some(48 * GIB), Some(2 * GIB), false, 80 * GIB),
            ArenaPlan::Take(48 * GIB),
            "50 GiB is the process target, so its existing 2 GiB leaves 48 GiB for cache"
        );
        assert_eq!(
            streaming_arena_plan(forced, Some(48 * GIB), Some(50 * GIB), false, 80 * GIB),
            ArenaPlan::Skip(Skip::TooLittle),
            "a process already at its explicit total budget must not allocate another arena"
        );
    }

    /// A budget of ZERO turns the tier off by name, on both paths and whatever the automatic rungs
    /// would have decided. Without this there is no way to A/B the tier against the mmap path it
    /// replaces once auto-sizing turns it on by itself, and `0` would otherwise read as "unset".
    #[test]
    fn a_zero_budget_is_the_off_switch() {
        assert_eq!(
            RamRequest::from_config(Some(0), None, false),
            RamRequest::TotalProcessBudget(0)
        );
        assert_eq!(
            cpu_arena_plan(
                RamRequest::TotalProcessBudget(0),
                Some(GIB),
                Some(0),
                200 * GIB
            ),
            ArenaPlan::Skip(Skip::Disabled)
        );
        assert_eq!(
            streaming_arena_plan(
                RamRequest::LegacyCacheBudget(0),
                Some(64 * GIB),
                Some(0),
                false,
                200 * GIB
            ),
            ArenaPlan::Skip(Skip::Disabled)
        );
        // ...and it is distinct from unset, which on that same host DOES build one.
        assert!(matches!(
            streaming_arena_plan(RamRequest::Auto, Some(64 * GIB), Some(0), false, 200 * GIB),
            ArenaPlan::Take(_)
        ));
    }

    #[test]
    fn from_config_maps_unset_and_sizes() {
        assert_eq!(RamRequest::from_config(None, None, false), RamRequest::Auto);
        assert_eq!(
            RamRequest::from_config(Some(42), None, false),
            RamRequest::TotalProcessBudget(42)
        );
        assert_eq!(
            RamRequest::from_config(None, Some(42), false),
            RamRequest::LegacyCacheBudget(42)
        );
        assert_eq!(
            RamRequest::from_config(Some(42), Some(7), false),
            RamRequest::TotalProcessBudget(42),
            "the canonical total-process parameter wins over the legacy cache override"
        );
    }

    /// Requirement one: if it fits, everything stays resident. Paging a model that fits would add
    /// a copy per block over the zero-copy mapping and buy nothing.
    #[test]
    fn a_model_that_fits_is_not_paged() {
        assert_eq!(
            cpu_arena_plan(RamRequest::Auto, Some(64 * GIB), Some(0), 8 * GIB),
            ArenaPlan::Skip(Skip::Fits)
        );
        // Exactly fitting still counts as fitting.
        assert_eq!(
            cpu_arena_plan(RamRequest::Auto, Some(8 * GIB), Some(0), 8 * GIB),
            ArenaPlan::Skip(Skip::Fits)
        );
    }

    /// Requirement two: a model that does NOT fit streams, without being asked to.
    #[test]
    fn a_model_that_does_not_fit_streams() {
        match cpu_arena_plan(RamRequest::Auto, Some(32 * GIB), Some(0), 200 * GIB) {
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
            streaming_arena_plan(RamRequest::Auto, Some(64 * GIB), Some(0), true, 40 * GIB),
            ArenaPlan::StreamOnly
        );
        // It must NOT collapse to "keep the mmap path" — that is the thing it replaces.
        assert_ne!(
            streaming_arena_plan(RamRequest::Auto, Some(64 * GIB), Some(0), true, 40 * GIB),
            ArenaPlan::Skip(Skip::Fits)
        );
        // A host with no memory to spare still streams on unified memory, because the decision
        // does not depend on having any to give.
        assert_eq!(
            streaming_arena_plan(RamRequest::Auto, Some(GIB), Some(0), true, 40 * GIB),
            ArenaPlan::StreamOnly
        );
        // The same host WITHOUT unified memory caches instead — otherwise this test would pass
        // for a version that simply never auto-sizes.
        assert!(matches!(
            streaming_arena_plan(RamRequest::Auto, Some(64 * GIB), Some(0), false, 40 * GIB),
            ArenaPlan::Take(_)
        ));
    }

    /// "Cannot tell" must never be reported as "fits": the two lead to opposite advice.
    #[test]
    fn no_probe_is_distinct_from_fitting() {
        assert_eq!(
            cpu_arena_plan(RamRequest::Auto, None, Some(0), 200 * GIB),
            ArenaPlan::Skip(Skip::NoProbe)
        );
        assert_eq!(
            streaming_arena_plan(RamRequest::Auto, None, Some(0), false, 200 * GIB),
            ArenaPlan::Skip(Skip::NoProbe)
        );
    }

    /// A host with nothing to spare declines rather than building a useless arena.
    #[test]
    fn a_squeezed_host_declines() {
        assert_eq!(auto_cache_bytes(GIB, 0, u64::MAX), 0);
        assert_eq!(auto_cache_bytes(64 * GIB, 63 * GIB, u64::MAX), 0);
        // Just under the useful floor, with headroom accounted for.
        assert_eq!(auto_cache_bytes(2 * GIB, 0, MIN_USEFUL - 1), 0);
    }
}
