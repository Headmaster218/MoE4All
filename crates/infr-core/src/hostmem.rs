//! How much host memory a weight arena may take, and how much there is to take
//! (`docs/disk-streaming-plan.md` §7 question 3).
//!
//! The DRAM tier's arena is ANONYMOUS, non-evictable memory — the expensive kind backlog B30
//! measured — so its size cannot be a guess. Two separate concerns live here:
//!
//! - [`available_bytes`], a PLATFORM probe of physical RAM available right now. It is deliberately
//!   allowed to answer "I do not know" rather than estimate, because an over-estimate here is an
//!   out-of-memory kill or a swap storm mid-generation, not a slow run.
//! - [`auto_cache_bytes`], the PURE arithmetic that turns that answer into a budget. Separate so
//!   the policy can be tested without a machine that happens to have the right amount of RAM free.
//! - [`process_resident_bytes`], the current process working set used to resolve an explicit
//!   total-process RAM budget into the part that can actually become a weight cache.

/// Physical host memory available right now, or `None` where this platform has no probe.
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
/// **Windows** reads `GlobalMemoryStatusEx::ullAvailPhys`. Commit headroom is deliberately exposed
/// separately through [`commit_available_bytes`]: the automatic RAM policy is frozen before any
/// Vulkan allocation and has the same semantics as an explicit total-process RAM budget. Folding
/// WDDM's later device-memory charge into this probe would silently shrink that startup decision.
/// Every other platform answers `None` today; macOS would need `host_statistics64`'s
/// free/inactive/purgeable split.
///
/// **A cgroup memory limit overrides it.** `/proc/meminfo` is host-wide and knows nothing about the
/// limit a container or a `systemd-run --scope -p MemoryMax=` puts on this process — measured on
/// this box, an 8 GiB scope still reports 54.6 GiB available. Sizing an anonymous arena from that
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
        Some(status.ullAvailPhys)
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

/// Total physical host RAM, used as the base for percentage-valued budgets and aggressive
/// automatic host sizing.
///
/// This is deliberately separate from [`available_bytes`]: `device.ram_budget=80%` means 80% of
/// the machine's physical RAM as a process-wide target. Ordinary automatic sizing uses live
/// availability; aggressive automatic sizing targets total RAM minus its fixed reserve.
pub fn total_bytes() -> Option<u64> {
    let observed = platform_total_bytes()?;
    Some(match crate::test_resource::active() {
        None => observed,
        Some(profile) => profile.cap_ram(observed, observed).0,
    })
}

/// Bytes the current process may still commit before the system commit limit is reached.
///
/// Windows/WDDM charges Vulkan device-memory allocations against this limit even when the bytes
/// live in dedicated VRAM. Callers that allocate a large host arena after their Vulkan heaps are
/// committed therefore need the raw commit headroom, independently from [`available_bytes`]'s
/// physical-RAM minimum. Other platforms do not need this WDDM-specific second ceiling.
pub fn commit_available_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        Some(windows_memory_status()?.ullAvailPageFile)
    }
    #[cfg(not(windows))]
    {
        None
    }
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

/// Ask Windows to evict clean pages from this process's working set at a load-phase boundary.
///
/// The Vulkan loader touches GGUF source pages while uploading fixed weights. Those pages are no
/// longer needed once the upload queue is drained, but Windows may otherwise keep them resident
/// while the anonymous MoE host arena is filled. This is deliberately an explicit one-shot hook:
/// calling it after the host arena exists would evict useful expert pages too. Other platforms do
/// nothing and preserve their existing VM behaviour.
pub fn trim_reclaimable_working_set() -> bool {
    #[cfg(windows)]
    {
        use windows::Win32::System::ProcessStatus::EmptyWorkingSet;
        use windows::Win32::System::Threading::GetCurrentProcess;

        unsafe { EmptyWorkingSet(GetCurrentProcess()).is_ok() }
    }
    #[cfg(not(windows))]
    {
        false
    }
}

#[cfg(any(target_os = "linux", test))]
fn parse_process_resident(text: &str) -> Option<u64> {
    let line = text.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
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

/// Pull `MemAvailable` out of `/proc/meminfo`. Linux labels the binary KiB value as `kB`.
///
/// Split from the read so the parse is testable against a literal — the one machine this runs on
/// cannot produce a file with the field missing, which is the case worth checking.
#[cfg(any(target_os = "linux", test))]
fn parse_mem_available(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    // `MemAvailable:   12345678 kB`
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

#[cfg(any(target_os = "linux", test))]
fn parse_mem_total(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

/// Ordinary automatic sizing spends currently available memory while retaining this fixed tail
/// for the OS, staging, activations, the tokenizer, and other processes.
const AUTO_AVAILABLE_RESERVE: u64 = 3 << 30;

/// Aggressive automatic sizing treats this as a process-wide reserve below total physical RAM.
/// Unlike the ordinary profile, already-used RAM does not permanently shrink the target: the
/// current process working set is subtracted from `total - reserve` to obtain the new arena size.
const AGGRESSIVE_TOTAL_RESERVE: u64 = 14 << 30;

/// One startup-time automatic RAM decision after it has been frozen into
/// `device.ram_budget`. From this point on it deliberately follows the explicit total-process
/// budget path, including its process-resident subtraction and allocation behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrozenAutomaticRamBudget {
    pub profile: crate::config::AutoProfile,
    pub budget_bytes: u64,
    pub startup_available_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
}

/// Resolve an unset automatic RAM policy once, before model or backend construction.
///
/// Explicit `device.ram_budget`, legacy `paging.dram`, and `paging.dram_bypass` retain their
/// existing precedence. A missing platform probe leaves the config unset so the backend's
/// best-effort automatic fallback remains available.
pub fn freeze_automatic_ram_budget(
    config: &mut crate::config::Config,
) -> Option<FrozenAutomaticRamBudget> {
    let available = available_bytes();
    let total = total_bytes();
    freeze_automatic_ram_budget_for_snapshot(config, available, total)
}

fn freeze_automatic_ram_budget_for_snapshot(
    config: &mut crate::config::Config,
    available: Option<u64>,
    total: Option<u64>,
) -> Option<FrozenAutomaticRamBudget> {
    if config.device.ram_budget.is_some()
        || config.paging.dram.is_some()
        || config.paging.dram_bypass
    {
        return None;
    }
    let profile = config.device.auto_profile;
    let budget_bytes = match profile {
        crate::config::AutoProfile::Conservative => {
            available?.saturating_sub(AUTO_AVAILABLE_RESERVE)
        }
        crate::config::AutoProfile::Aggressive => total?.saturating_sub(AGGRESSIVE_TOTAL_RESERVE),
    };
    config.device.ram_budget = Some(crate::SizeSpec::Bytes(budget_bytes));
    Some(FrozenAutomaticRamBudget {
        profile,
        budget_bytes,
        startup_available_bytes: available,
        total_bytes: total,
    })
}

/// Below this an arena is not worth building: the tier costs a copy per streamed block, and a
/// budget this small holds so little of a model that the hit rate cannot pay for it.
const MIN_USEFUL: u64 = 256 << 20;

/// The arena budget to take, given what is available and what the run has already spoken for.
///
/// - `available` — from [`available_bytes`].
/// - `committed` — host bytes this run will already hold that the selected target does not know
///   about. On a
///   UNIFIED-memory device (iGPU, APU, Metal) the "VRAM" budget is carved out of this same physical
///   RAM, so passing it here is what stops the two tiers from spending the same bytes twice. Zero
///   on a discrete GPU, whose VRAM is a separate pool.
/// - `pageable` — total bytes of the weights that could be paged. Budgeting past this buys nothing:
///   every block would already be resident.
///
/// Returns `0` when nothing worth having is left, which callers treat as "stay on the mmap path".
pub fn auto_cache_bytes(available: u64, committed: u64, pageable: u64) -> u64 {
    auto_cache_bytes_for_profile(
        crate::config::AutoProfile::Conservative,
        available,
        committed,
        pageable,
    )
}

/// Profile-aware form of [`auto_cache_bytes`]. Explicit total-process/cache budgets do not call
/// this function and therefore remain exact regardless of the automatic profile.
pub fn auto_cache_bytes_for_profile(
    profile: crate::config::AutoProfile,
    available: u64,
    committed: u64,
    pageable: u64,
) -> u64 {
    auto_cache_bytes_for_snapshot(profile, available, None, None, committed, pageable)
}

/// Automatic cache sizing with the complete host-memory snapshot. The aggressive profile needs
/// both total physical RAM and the current process working set to enforce its process-wide target;
/// when either is unavailable it falls back to the ordinary live-availability policy.
pub fn auto_cache_bytes_for_snapshot(
    profile: crate::config::AutoProfile,
    available: u64,
    total: Option<u64>,
    process_resident: Option<u64>,
    committed: u64,
    pageable: u64,
) -> u64 {
    let target = match (profile, total, process_resident) {
        (crate::config::AutoProfile::Aggressive, Some(total), Some(resident)) => total
            .saturating_sub(AGGRESSIVE_TOTAL_RESERVE)
            .saturating_sub(resident),
        // A total without a matching live process measurement is not enough to enforce a
        // process-wide target. Fall back to the ordinary live-availability policy.
        _ => available.saturating_sub(AUTO_AVAILABLE_RESERVE),
    };
    let usable = target.saturating_sub(committed).min(pageable);
    if usable < MIN_USEFUL {
        return 0;
    }
    usable
}

/// Convert an explicit total-process RAM target into bytes available to this new host arena.
///
/// `resident` is sampled immediately before the arena is planned. It includes existing model
/// mappings and earlier arenas, so repeated model/session loads share one process-wide ceiling.
/// Planning happens before pager registrations, initial KV segments and the server runtime are
/// fully built, so reserve their small persistent tail rather than handing literally every
/// unoccupied byte to the cache. This is accounting for not-yet-created process memory, not the
/// automatic policy's multi-GiB safety headroom.
/// Where the platform probe fails, zero preserves the historical interpretation as a best-effort
/// fallback; Windows and Linux have live probes and therefore enforce the total-budget meaning.
const TOTAL_BUDGET_FUTURE_RESERVE: u64 = 512 << 20;

pub fn cache_bytes_for_total_budget(total: u64, resident: Option<u64>, pageable: u64) -> u64 {
    total
        .saturating_sub(resident.unwrap_or(0))
        .saturating_sub(TOTAL_BUDGET_FUTURE_RESERVE)
        .min(pageable)
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
    streaming_arena_plan_for_profile(
        crate::config::AutoProfile::Conservative,
        request,
        available,
        process_resident,
        unified,
        pageable,
    )
}

/// Profile-aware form of [`streaming_arena_plan`]. The profile is consulted only for
/// [`RamRequest::Auto`]; every explicit request retains its existing exact semantics.
pub fn streaming_arena_plan_for_profile(
    profile: crate::config::AutoProfile,
    request: RamRequest,
    available: Option<u64>,
    process_resident: Option<u64>,
    unified: bool,
    pageable: u64,
) -> ArenaPlan {
    streaming_arena_plan_for_snapshot(
        profile,
        request,
        available,
        None,
        process_resident,
        unified,
        pageable,
    )
}

/// The rule is **`arena + in-use + fill-churn <= 90% of MemTotal`**:
///
/// * `in-use` is the kernel's own estimate `MemTotal - MemAvailable`, so reclaimable page cache
///   is not counted twice against us;
/// * `fill-churn` is the page cache a FILL will rebuild. Filling the arena streams the source
///   blocks back through the page cache, so the machine transiently needs room for both — and on
///   a fast fill the kernel cannot reclaim one fast enough to make space for the other. That term
///   is why a 47 GiB fall-through still OOM'd on a 58 GiB-available box, cap and cgroup and all.
///
/// This is deliberately DYNAMIC: free memory admits a larger arena, a busy machine clamps even a
/// modest one, and a request well inside the ceiling passes through untouched. On a quiet 60 GiB
/// box (`used` ~3 GiB) it settles near `(54 - 3) / 2` ~ 25 GiB.
pub fn clamp_arena_to_spare(
    bytes: u64,
    available: Option<u64>,
    total: Option<u64>,
    fill_churn_bytes: u64,
) -> u64 {
    let Some(total) = total else {
        // No physical-RAM probe (unusual outside Linux/Windows): nothing to bound against.
        return bytes;
    };
    let used = available.map_or(0, |available| total.saturating_sub(available));
    let budget = (total / 10 * 9).saturating_sub(used);
    // Never more churn than we take, and never a ceiling that double-counts it: solving
    // Solve `arena + min(churn, arena) <= budget` for the largest admissible arena: at most half
    // the budget, tightened further to `budget - churn` once the churn is the smaller term.
    let mut ceiling = bytes.min(budget / 2);
    if fill_churn_bytes < ceiling {
        ceiling = ceiling.min(budget.saturating_sub(fill_churn_bytes));
    }
    ceiling
}

/// Profile-aware arena planning with the complete host-memory snapshot.
pub fn streaming_arena_plan_for_snapshot(
    profile: crate::config::AutoProfile,
    request: RamRequest,
    available: Option<u64>,
    total: Option<u64>,
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
            // An explicit total-process budget keeps its documented meaning: the user is stating
            // what the PROCESS may use, so `cache_bytes_for_total_budget` already nets out the
            // resident set and its own future reserve. Do not second-guess it here —
            // `an_oversized_explicit_budget_bypasses_automatic_headroom` pins that contract.
            let bytes = cache_bytes_for_total_budget(total, process_resident, pageable);
            return if bytes == 0 {
                ArenaPlan::Skip(Skip::TooLittle)
            } else {
                ArenaPlan::Take(bytes)
            };
        }
        RamRequest::LegacyCacheBudget(0) => return ArenaPlan::Skip(Skip::Disabled),
        RamRequest::LegacyCacheBudget(bytes) => {
            return ArenaPlan::Take(clamp_arena_to_spare(bytes, available, total, pageable));
        }
        RamRequest::Auto => {}
    }
    if unified {
        return ArenaPlan::StreamOnly;
    }
    let Some(available) = available else {
        return ArenaPlan::Skip(Skip::NoProbe);
    };
    match auto_cache_bytes_for_snapshot(profile, available, total, process_resident, 0, pageable) {
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
    cpu_arena_plan_for_profile(
        crate::config::AutoProfile::Conservative,
        request,
        available,
        process_resident,
        pageable,
    )
}

/// Profile-aware form of [`cpu_arena_plan`]. Explicit requests bypass automatic headroom exactly
/// as they do in the conservative compatibility wrapper.
pub fn cpu_arena_plan_for_profile(
    profile: crate::config::AutoProfile,
    request: RamRequest,
    available: Option<u64>,
    process_resident: Option<u64>,
    pageable: u64,
) -> ArenaPlan {
    cpu_arena_plan_for_snapshot(
        profile,
        request,
        available,
        None,
        process_resident,
        pageable,
    )
}

/// Profile-aware CPU arena planning with the complete host-memory snapshot.
pub fn cpu_arena_plan_for_snapshot(
    profile: crate::config::AutoProfile,
    request: RamRequest,
    available: Option<u64>,
    total: Option<u64>,
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
    match auto_cache_bytes_for_snapshot(profile, available, total, process_resident, 0, pageable) {
        0 => ArenaPlan::Skip(Skip::TooLittle),
        n => ArenaPlan::Take(n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn parses_mem_available_from_linux_kb_field() {
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
    fn parses_process_resident_from_linux_kb_field() {
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
        let total = total_bytes().expect("GlobalMemoryStatusEx should report total RAM");
        assert!(total > 0, "total physical memory must be non-zero");
        assert!(avail <= total, "available {avail} exceeds total {}", total);
        assert!(commit_available_bytes().is_some());
        let resident = process_resident_bytes().expect("GetProcessMemoryInfo should answer");
        assert!(
            resident > 0,
            "the running test process must have resident pages"
        );
        assert!(
            resident <= total,
            "process working set {resident} exceeds physical RAM {total}",
        );
    }

    /// Ordinary automatic sizing retains exactly 3 GiB of current availability.
    #[test]
    fn headroom_is_always_left() {
        for &avail in &[2 * GIB, 8 * GIB, 64 * GIB, 512 * GIB] {
            let got = auto_cache_bytes(avail, 0, u64::MAX);
            assert!(got < avail, "took all {avail} bytes");
            assert_eq!(got, avail.saturating_sub(AUTO_AVAILABLE_RESERVE));
        }
    }

    #[test]
    fn a_large_host_leaves_three_gib() {
        let avail = 64 * GIB;
        assert_eq!(auto_cache_bytes(avail, 0, u64::MAX), 61 * GIB);
    }

    #[test]
    fn a_workstation_keeps_three_gib_available() {
        assert_eq!(auto_cache_bytes(52 * GIB, 0, u64::MAX), 49 * GIB);
    }

    #[test]
    fn aggressive_profile_targets_total_ram_minus_fourteen_gib() {
        let available = 40 * GIB;
        let conservative = auto_cache_bytes(available, 0, u64::MAX);
        let aggressive = auto_cache_bytes_for_snapshot(
            crate::config::AutoProfile::Aggressive,
            available,
            Some(64 * GIB),
            Some(2 * GIB),
            0,
            u64::MAX,
        );
        assert!(aggressive > conservative);
        assert_eq!(aggressive, 48 * GIB);
    }

    #[test]
    fn automatic_ram_budget_is_frozen_like_an_explicit_startup_value() {
        let mut conservative = crate::config::Config::default();
        let frozen = freeze_automatic_ram_budget_for_snapshot(
            &mut conservative,
            Some(48 * GIB),
            Some(64 * GIB),
        )
        .expect("conservative startup budget");
        assert_eq!(frozen.budget_bytes, 45 * GIB);
        assert_eq!(
            conservative.device.ram_budget,
            Some(crate::SizeSpec::Bytes(45 * GIB))
        );

        let mut aggressive = crate::config::Config::default();
        aggressive.device.auto_profile = crate::config::AutoProfile::Aggressive;
        let frozen = freeze_automatic_ram_budget_for_snapshot(
            &mut aggressive,
            Some(48 * GIB),
            Some(64 * GIB),
        )
        .expect("aggressive startup budget");
        assert_eq!(frozen.budget_bytes, 50 * GIB);
        assert_eq!(
            aggressive.device.ram_budget,
            Some(crate::SizeSpec::Bytes(50 * GIB))
        );
    }

    #[test]
    fn automatic_ram_freeze_never_overrides_an_explicit_host_policy() {
        let mut explicit = crate::config::Config::default();
        explicit.device.ram_budget = Some(crate::SizeSpec::Bytes(32 * GIB));
        assert_eq!(
            freeze_automatic_ram_budget_for_snapshot(&mut explicit, Some(48 * GIB), Some(64 * GIB)),
            None
        );
        assert_eq!(
            explicit.device.ram_budget,
            Some(crate::SizeSpec::Bytes(32 * GIB))
        );

        let mut legacy = crate::config::Config::default();
        legacy.paging.dram = Some(crate::SizeSpec::Bytes(7 * GIB));
        assert_eq!(
            freeze_automatic_ram_budget_for_snapshot(&mut legacy, Some(48 * GIB), Some(64 * GIB)),
            None
        );

        let mut bypass = crate::config::Config::default();
        bypass.paging.dram_bypass = true;
        assert_eq!(
            freeze_automatic_ram_budget_for_snapshot(&mut bypass, Some(48 * GIB), Some(64 * GIB)),
            None
        );
    }

    #[test]
    fn aggressive_profile_without_a_complete_snapshot_falls_back_to_live_availability() {
        assert_eq!(
            auto_cache_bytes_for_snapshot(
                crate::config::AutoProfile::Aggressive,
                40 * GIB,
                Some(64 * GIB),
                None,
                0,
                u64::MAX,
            ),
            37 * GIB
        );
    }

    #[test]
    fn explicit_ram_budget_is_profile_independent() {
        let request = RamRequest::TotalProcessBudget(40 * GIB);
        let conservative = streaming_arena_plan_for_profile(
            crate::config::AutoProfile::Conservative,
            request,
            Some(52 * GIB),
            Some(2 * GIB),
            false,
            u64::MAX,
        );
        let aggressive = streaming_arena_plan_for_profile(
            crate::config::AutoProfile::Aggressive,
            request,
            Some(52 * GIB),
            Some(2 * GIB),
            false,
            u64::MAX,
        );
        assert_eq!(aggressive, conservative);
    }

    #[test]
    fn a_huge_host_still_leaves_three_gib() {
        assert_eq!(auto_cache_bytes(512 * GIB, 0, u64::MAX), 509 * GIB);
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

    #[test]
    fn total_budget_keeps_room_for_post_plan_process_objects() {
        assert_eq!(
            cache_bytes_for_total_budget(22 * GIB, Some(2 * GIB), u64::MAX),
            20 * GIB - TOTAL_BUDGET_FUTURE_RESERVE
        );
        assert_eq!(
            cache_bytes_for_total_budget(
                2 * GIB + TOTAL_BUDGET_FUTURE_RESERVE - 1,
                Some(2 * GIB),
                u64::MAX,
            ),
            0,
            "a tiny remainder must saturate instead of escaping the total-process target"
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
            ArenaPlan::Take(48 * GIB - TOTAL_BUDGET_FUTURE_RESERVE),
            "the total-process target must also cover objects created after cache planning"
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
