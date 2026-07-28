//! Runtime for the `INFR_PROFILE=1` build-time instrumentation (see `infr-prof` and
//! docs/perf.md § "Build-time auto-instrumentation").
//!
//! `#[infr_prof::instrument]` rewrites every `fn` it covers to open a [`Site`]-keyed span at
//! entry via [`enter`]; the returned RAII [`Guard`] closes the span on any exit path (`?`,
//! `return`, panic unwind). Accounting is strictly thread-local on the hot path — a
//! `thread_local!` span stack plus a per-thread fixed-size accumulator table — so there is no
//! shared-state contention. Accumulator slots are `AtomicU64`s written with plain
//! load-then-store `Relaxed` ops (single writer: the owning thread); the atomics exist only so
//! the exit-time reporter may read tables of threads that are still alive (rayon workers never
//! unwind their TLS). The merged report prints to stderr at process exit (`atexit`), sorted by
//! self time, and is also written as JSON to `$INFR_PROF_OUT` if set.
//!
//! Metrics per site: call count, inclusive total (recursion-aware: only the outermost frame of
//! a site on a given thread adds to total), self time (inclusive minus instrumented children),
//! and average self per call.
//!
//! This crate compiles to dead code in default builds — nothing calls it unless a crate was
//! built with the `infr_profile` cfg (emitted by build.rs when `INFR_PROFILE=1`).
//!
//! It ALSO hosts the process-wide **per-op device aggregate** ([`gpu_add`] / [`gpu_reset`]): every
//! backend's per-op profiler folds its per-forward/per-submit rows in here through the shared
//! reporter ([`infr_core::prof::OpProf`]), and the exit report prints them as a separate device
//! section (and a `"gpu"` array in the `INFR_PROF_OUT` JSON). This path is runtime-gated
//! (`INFR_PROF_OPS` / `INFR_PROF_OPS`, via `prof.per_op()`), not build-gated — in a default build with
//! the knob set, the exit report contains only the device section; in an `INFR_PROFILE=1` build it
//! merges with the host function table in one report.
//!
//! Vulkan fed this alone until U3. rocm, metal and the cpu backend each accumulated and printed
//! their own totals and appeared nowhere in the exit report or the JSON — so "correlate the host
//! function table against device op time" was accidentally a vulkan-only capability.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Hard cap on distinct instrumented call sites. Sites past the cap are counted but not timed
/// (reported as `[dropped]`). 8192 * 24 B = 192 KiB per thread — cheap for a profiling build.
const MAX_SITES: usize = 8192;

/// Enter count for span sites whose id landed at or past [`MAX_SITES`] (never timed, no
/// per-thread slot). Surfaced by [`collect`] as one count-only `[dropped]` aggregate row so
/// over-cap activity is visible instead of vanishing silently. A real `fetch_add` (not the
/// single-writer store used for slots) since any thread may hit it — but it is the cold path
/// (only reached once a build has >8192 distinct instrumented sites), so the cost is irrelevant.
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// One static per instrumented `fn`, created by the `#[infr_prof::instrument]` expansion.
/// The id is assigned lazily on first call (relaxed fast path, mutex slow path).
pub struct Site {
    module: &'static str,
    name: &'static str,
    /// 0 = unassigned; otherwise site index + 1.
    id: AtomicU32,
}

impl Site {
    pub const fn new(module: &'static str, name: &'static str) -> Self {
        Site {
            module,
            name,
            id: AtomicU32::new(0),
        }
    }

    #[inline]
    fn id(&'static self) -> u32 {
        let id = self.id.load(Relaxed);
        if id != 0 {
            return id - 1;
        }
        self.register()
    }

    #[cold]
    fn register(&'static self) -> u32 {
        let g = global();
        let mut names = g.names.lock().unwrap();
        // Double-check under the lock (another thread may have registered this site).
        let id = self.id.load(Relaxed);
        if id != 0 {
            return id - 1;
        }
        let idx = names.len() as u32;
        names.push((self.module, self.name));
        self.id.store(idx + 1, Relaxed);
        idx
    }
}

struct Slot {
    count: AtomicU64,
    total_ns: AtomicU64,
    self_ns: AtomicU64,
}

/// Per-thread accumulator table, shared with the global registry so the exit reporter can read
/// it while worker threads (rayon pools, spin pools) are still parked.
struct AccumTable {
    slots: Box<[Slot]>,
}

impl AccumTable {
    fn new() -> Self {
        let mut v = Vec::with_capacity(MAX_SITES);
        for _ in 0..MAX_SITES {
            v.push(Slot {
                count: AtomicU64::new(0),
                total_ns: AtomicU64::new(0),
                self_ns: AtomicU64::new(0),
            });
        }
        AccumTable {
            slots: v.into_boxed_slice(),
        }
    }
}

/// Single-writer add: plain load+store (a `mov`, not `lock xadd`) — the owning thread is the
/// only writer; the reporter only reads.
#[inline]
fn add(a: &AtomicU64, v: u64) {
    a.store(a.load(Relaxed).wrapping_add(v), Relaxed);
}

struct Frame {
    id: u32,
    start: Instant,
    /// Nanoseconds spent in instrumented callees of this frame (for self-time).
    child_ns: u64,
}

struct ThreadProf {
    table: Arc<AccumTable>,
    stack: Vec<Frame>,
    /// Per-site recursion depth on this thread (inclusive total only counts the outermost).
    depth: Vec<u32>,
}

impl ThreadProf {
    fn new() -> Self {
        // Register this thread's 192 KiB table in the global vec and NEVER deregister it. This
        // is intentional: rayon/spin workers park (they never unwind their TLS), so the exit
        // reporter must still be able to read a parked worker's table — a `Weak` that dropped
        // when the thread parked would lose that data. The `tables` vec therefore grows by one
        // Arc per thread that ever runs instrumented code and is retained for the process
        // lifetime. That is bounded here: this engine uses fixed-size rayon pools plus the main
        // thread, so the count is small and constant. CAVEAT: a workload that spawns unbounded
        // *transient* threads would grow this vec without bound (and lengthen the reporter's
        // per-thread scan). If that ever becomes a real pattern, switch `tables` to hold
        // `Weak<AccumTable>` and prune upgrades that fail in `collect` — accepting that a
        // transient thread's data is lost once it fully exits.
        let table = Arc::new(AccumTable::new());
        global().tables.lock().unwrap().push(table.clone());
        ThreadProf {
            table,
            stack: Vec::with_capacity(64),
            depth: Vec::new(),
        }
    }
}

thread_local! {
    static TP: RefCell<ThreadProf> = RefCell::new(ThreadProf::new());
}

struct Global {
    names: Mutex<Vec<(&'static str, &'static str)>>,
    tables: Mutex<Vec<Arc<AccumTable>>>,
    start: Instant,
}

static GLOBAL: OnceLock<Global> = OnceLock::new();

fn global() -> &'static Global {
    GLOBAL.get_or_init(|| {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        extern "C" fn report_at_exit() {
            report();
        }
        unsafe {
            atexit(report_at_exit);
        }
        Global {
            names: Mutex::new(Vec::new()),
            tables: Mutex::new(Vec::new()),
            start: Instant::now(),
        }
    })
}

/// Process-wide per-op device aggregate: label → (total µs, dispatch count). Fed by every
/// backend's profiler through the shared reporter; printed as the device section of [`report`].
/// A Mutex is fine — this is touched once per submit/forward (not per dispatch), profiling only.
static GPU: Mutex<Vec<(String, f64, u64)>> = Mutex::new(Vec::new());

/// Accumulate `us` device-microseconds / `count` dispatches under `label`. Registers the exit
/// report (idempotent) so a default build with only the runtime knob set still prints the
/// aggregate.
///
/// Call this through [`infr_core::prof::OpProf::flush`], not directly — that is what keeps every
/// backend's label grammar, sort order and units consistent.
pub fn gpu_add(label: &str, us: f64, count: u64) {
    let _ = global(); // ensure the atexit report hook exists
    let mut g = GPU.lock().unwrap();
    match g.iter_mut().find(|(l, _, _)| l == label) {
        Some(row) => {
            row.1 += us;
            row.2 += count;
        }
        None => g.push((label.to_string(), us, count)),
    }
}

/// Clear the GPU aggregate. Benches call this after their untimed warmup so warmup dispatches
/// (pipeline compiles' first submits, cache-warming turns) don't pollute the exit report.
pub fn gpu_reset() {
    GPU.lock().unwrap().clear();
}

/// Runtime "suppress per-op GPU profiling" flag — the non-env replacement for the old
/// `INFR_PROF_OPS` env unset/restore dance (which mutated a process-global table around a
/// rayon-parallel forward: a data race, and `set_var` is `unsafe` under edition 2024). Set true
/// around untimed warmup / cache-warming work whose dispatches must not pollute the profile; the
/// Vulkan recorder ANDs [`profiling_suppressed`] into its `INFR_PROF_OPS` construction check, so those
/// submits record no timestamps. A plain `AtomicBool`: set/cleared on the thread that spawns and
/// joins the parallel region, so its writes happen-before the workers' reads.
static PROFILING_SUPPRESSED: AtomicBool = AtomicBool::new(false);

/// Set the [`PROFILING_SUPPRESSED`] flag, returning the PRIOR value so the caller can restore it
/// (supports nesting). See [`profiling_suppressed`].
pub fn set_profiling_suppressed(v: bool) -> bool {
    PROFILING_SUPPRESSED.swap(v, Relaxed)
}

/// Whether per-op GPU profiling is currently suppressed (see [`set_profiling_suppressed`]). The Vulkan
/// recorder reads this at construction alongside `INFR_PROF_OPS`.
pub fn profiling_suppressed() -> bool {
    PROFILING_SUPPRESSED.load(Relaxed)
}

/// Where [`report`] writes its JSON report — the resolved `prof.out` (`INFR_PROF_OUT`),
/// PUSHED here at startup by whoever owns the `Config` (S7; `infr-cli`'s `main`, beside
/// `publish_thread_count`). `None`/empty = no JSON, stderr report only.
///
/// It is report state on this crate's existing process-global reporter, not a `Config` bridge
/// (R4): the report runs from a C `atexit` hook — `extern "C" fn report_at_exit()` — which has no
/// receiver, no arguments and no config to borrow, and `infr-prof-rt` cannot depend on `infr-core`
/// to fetch one (the dependency runs the other way). It sits next to [`PROFILING_SUPPRESSED`], which
/// exists for the same reason.
static PROFILE_OUT: Mutex<Option<String>> = Mutex::new(None);

/// Install the resolved `prof.out`. Call once at startup, before the process can exit.
/// An empty path is stored as-is and treated as "no JSON" by [`report`], matching the old
/// `INFR_PROF_OUT=` grammar exactly.
pub fn set_profile_out(path: Option<&str>) {
    *PROFILE_OUT.lock().unwrap() = path.map(str::to_owned);
}

/// The installed [`PROFILE_OUT`], cloned out from under the lock (the reporter must not hold it
/// while it writes a file).
fn profile_out() -> Option<String> {
    PROFILE_OUT.lock().unwrap().clone()
}

/// Open a span for `site` on the current thread. Returns an RAII guard; span closes when the
/// guard drops (any exit path). Must be strictly LIFO per thread — guaranteed because the guard
/// is a local of the instrumented fn.
#[inline]
pub fn enter(site: &'static Site) -> Guard {
    let id = site.id();
    // try_with: during thread teardown the TLS is gone — record nothing rather than panic.
    let active = TP
        .try_with(|tp| {
            let mut tp = tp.borrow_mut();
            let idx = id as usize;
            if tp.depth.len() <= idx {
                tp.depth.resize(idx + 1, 0);
            }
            tp.depth[idx] += 1;
            tp.stack.push(Frame {
                id,
                start: Instant::now(),
                child_ns: 0,
            });
        })
        .is_ok();
    Guard { active }
}

pub struct Guard {
    active: bool,
}

impl Drop for Guard {
    #[inline]
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let end = Instant::now();
        let _ = TP.try_with(|tp| {
            let mut tp = tp.borrow_mut();
            let Some(frame) = tp.stack.pop() else { return };
            let elapsed = end.duration_since(frame.start).as_nanos() as u64;
            let self_ns = elapsed.saturating_sub(frame.child_ns);
            let idx = frame.id as usize;
            tp.depth[idx] -= 1;
            let outermost = tp.depth[idx] == 0;
            if let Some(parent) = tp.stack.last_mut() {
                parent.child_ns += elapsed;
            }
            if idx < MAX_SITES {
                let slot = &tp.table.slots[idx];
                add(&slot.count, 1);
                add(&slot.self_ns, self_ns);
                if outermost {
                    add(&slot.total_ns, elapsed);
                }
            } else {
                // Over the site cap: no slot exists to time this into. Count it so the exit
                // report can surface a single `[dropped]` aggregate instead of losing it.
                DROPPED.fetch_add(1, Relaxed);
            }
        });
    }
}

struct Row {
    name: String,
    count: u64,
    total_ns: u64,
    self_ns: u64,
}

fn collect() -> (Vec<Row>, usize, u64) {
    let g = global();
    let names = g.names.lock().unwrap().clone();
    let tables = g.tables.lock().unwrap();
    let n_threads = tables.len();
    let wall_ns = g.start.elapsed().as_nanos() as u64;
    let mut rows: Vec<Row> = names
        .iter()
        .map(|(m, n)| Row {
            name: format!("{m}::{n}"),
            count: 0,
            total_ns: 0,
            self_ns: 0,
        })
        .collect();
    for t in tables.iter() {
        for (i, row) in rows.iter_mut().enumerate() {
            if i >= MAX_SITES {
                break;
            }
            let s = &t.slots[i];
            row.count += s.count.load(Relaxed);
            row.total_ns += s.total_ns.load(Relaxed);
            row.self_ns += s.self_ns.load(Relaxed);
        }
    }
    rows.retain(|r| r.count > 0);
    rows.sort_by_key(|r| std::cmp::Reverse(r.self_ns));
    // Surface over-cap sites (idx >= MAX_SITES) as one count-only aggregate. self_ns == 0 keeps
    // it last in the self-time ordering; it is always present in the JSON.
    let dropped = DROPPED.load(Relaxed);
    if dropped > 0 {
        rows.push(Row {
            name: "[dropped]".to_string(),
            count: dropped,
            total_ns: 0,
            self_ns: 0,
        });
    }
    (rows, n_threads, wall_ns)
}

fn fmt_ns(ns: u64) -> String {
    if ns >= 10_000_000_000 {
        format!("{:.1}s", ns as f64 / 1e9)
    } else if ns >= 1_000_000_000 {
        format!("{:.2}s", ns as f64 / 1e9)
    } else if ns >= 1_000_000 {
        format!("{:.1}ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.1}us", ns as f64 / 1e3)
    } else {
        format!("{ns}ns")
    }
}

/// Print the merged profile to stderr and, if `INFR_PROF_OUT` is set, write the full table
/// as JSON to that path. Two sections, each printed only when it has data: the host function
/// table (build-time instrumentation, `INFR_PROFILE=1` builds) and the GPU op aggregate
/// (runtime `INFR_PROF_OPS` timestamps fed via [`gpu_add`]). Runs automatically at process exit;
/// may also be called on demand.
pub fn report() {
    // atexit may fire in odd states; never run twice.
    static DONE: AtomicBool = AtomicBool::new(false);
    if DONE.swap(true, Relaxed) {
        return;
    }
    if GLOBAL.get().is_none() {
        return;
    }
    let (rows, n_threads, wall_ns) = collect();
    let mut gpu: Vec<(String, f64, u64)> = GPU.lock().unwrap().clone();
    // total_cmp, not partial_cmp().unwrap(): a bad timestamp delta can yield a NaN `us`, and
    // panicking here would abort the whole process from inside the atexit reporter. total_cmp
    // is a total order (NaN sorts deterministically) so the report always prints.
    gpu.sort_by(|a, b| b.1.total_cmp(&a.1));
    if rows.is_empty() && gpu.is_empty() {
        return;
    }
    if !rows.is_empty() {
        let accounted: u64 = rows.iter().map(|r| r.self_ns).sum();
        eprintln!();
        eprintln!(
            "== INFR_PROFILE report: {} sites, {} threads, wall {} (since first instrumented call), accounted self {} ==",
            rows.len(),
            n_threads,
            fmt_ns(wall_ns),
            fmt_ns(accounted),
        );
        eprintln!(
            "{:>12} {:>7} {:>12} {:>12} {:>10}  function",
            "self", "self%", "total", "calls", "avg(self)"
        );
        const TOP: usize = 50;
        for r in rows.iter().take(TOP) {
            eprintln!(
                "{:>12} {:>6.2}% {:>12} {:>12} {:>10}  {}",
                fmt_ns(r.self_ns),
                100.0 * r.self_ns as f64 / wall_ns.max(1) as f64,
                fmt_ns(r.total_ns),
                r.count,
                fmt_ns(r.self_ns / r.count.max(1)),
                r.name
            );
        }
        if rows.len() > TOP {
            eprintln!(
                "  ... {} more sites (set INFR_PROF_OUT=path for the full table as JSON)",
                rows.len() - TOP
            );
        }
    }
    if !gpu.is_empty() {
        let total_us: f64 = gpu.iter().map(|(_, us, _)| us).sum();
        let total_n: u64 = gpu.iter().map(|(_, _, n)| n).sum();
        eprintln!();
        eprintln!(
            "== per-op device report: {} ops, {total_n} dispatches, device total {} (every backend, all submits; warmup excluded where the bench resets) ==",
            gpu.len(),
            fmt_ns((total_us * 1e3) as u64),
        );
        eprintln!(
            "{:>12} {:>7} {:>12} {:>10}  gpu op",
            "total", "gpu%", "dispatches", "avg"
        );
        for (label, us, count) in &gpu {
            eprintln!(
                "{:>12} {:>6.2}% {:>12} {:>10}  {label}",
                fmt_ns((us * 1e3) as u64),
                100.0 * us / total_us.max(f64::MIN_POSITIVE),
                count,
                fmt_ns((us * 1e3) as u64 / count.max(&1)),
            );
        }
    }
    if let Some(path) = profile_out() {
        if !path.is_empty() {
            match write_json(&path, &rows, &gpu, n_threads, wall_ns) {
                Ok(()) => eprintln!("profile JSON written to {path}"),
                Err(e) => eprintln!("failed to write INFR_PROF_OUT={path}: {e}"),
            }
        }
    }
}

fn write_json(
    path: &str,
    rows: &[Row],
    gpu: &[(String, f64, u64)],
    n_threads: usize,
    wall_ns: u64,
) -> std::io::Result<()> {
    use std::io::Write;
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "{{")?;
    writeln!(f, "  \"wall_ns\": {wall_ns},")?;
    writeln!(f, "  \"threads\": {n_threads},")?;
    writeln!(f, "  \"sites\": [")?;
    for (i, r) in rows.iter().enumerate() {
        let name = esc(&r.name);
        let comma = if i + 1 < rows.len() { "," } else { "" };
        writeln!(
            f,
            "    {{\"name\": \"{name}\", \"calls\": {}, \"total_ns\": {}, \"self_ns\": {}}}{comma}",
            r.count, r.total_ns, r.self_ns
        )?;
    }
    writeln!(f, "  ],")?;
    writeln!(f, "  \"gpu\": [")?;
    for (i, (label, us, count)) in gpu.iter().enumerate() {
        let label = esc(label);
        let comma = if i + 1 < gpu.len() { "," } else { "" };
        writeln!(
            f,
            "    {{\"label\": \"{label}\", \"dispatches\": {count}, \"total_us\": {us:.1}}}{comma}",
        )?;
    }
    writeln!(f, "  ]")?;
    writeln!(f, "}}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tests that mutate the process-global site registry (`global().names`), so one
    /// test padding `names` up to `MAX_SITES` can never make another test's sites register past
    /// the cap. Held across each such test's registration + `collect`.
    static REG_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn nesting_and_recursion() {
        let _reg = REG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        static OUTER: Site = Site::new("m", "outer");
        static INNER: Site = Site::new("m", "inner");
        static REC: Site = Site::new("m", "rec");

        fn rec(n: u32) {
            let _g = enter(&REC);
            std::thread::sleep(std::time::Duration::from_millis(2));
            if n > 0 {
                rec(n - 1);
            }
        }
        {
            let _o = enter(&OUTER);
            {
                let _i = enter(&INNER);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        rec(2);

        let (rows, _, _) = collect();
        let get = |n: &str| rows.iter().find(|r| r.name == format!("m::{n}")).unwrap();
        let outer = get("outer");
        let inner = get("inner");
        let rec = get("rec");
        assert_eq!(outer.count, 1);
        assert_eq!(inner.count, 1);
        assert_eq!(rec.count, 3);
        // outer's self excludes inner's time
        assert!(outer.self_ns < outer.total_ns);
        assert!(inner.total_ns >= 5_000_000);
        assert!(outer.total_ns >= inner.total_ns);
        // recursion: inclusive total counted once at the outermost frame (~6ms, not ~12ms
        // double-counted); self sums each level's own ~2ms.
        assert!(rec.total_ns >= 6_000_000 && rec.total_ns < 11_000_000);
        assert!(rec.self_ns >= 6_000_000 && rec.self_ns <= rec.total_ns);
    }

    #[test]
    fn gpu_sort_total_cmp_no_nan_panic() {
        // Same comparator report() uses. A NaN `us` (bad timestamp delta) must NOT panic —
        // partial_cmp().unwrap() would abort the atexit reporter; total_cmp is a total order.
        let mut gpu: Vec<(String, f64, u64)> = vec![
            ("a".to_string(), 1.0, 1),
            ("nan".to_string(), f64::NAN, 1),
            ("b".to_string(), 2.0, 1),
        ];
        gpu.sort_by(|a, b| b.1.total_cmp(&a.1));
        assert_eq!(gpu.len(), 3);
    }

    #[test]
    fn dropped_bucket_counts_over_cap() {
        let _reg = REG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Pad the registry to the cap so the next site's id lands at MAX_SITES (over-cap), then
        // restore it so serialized sibling tests still register below the cap.
        let saved_len = {
            let mut names = global().names.lock().unwrap();
            let saved = names.len();
            if names.len() < MAX_SITES {
                names.resize(MAX_SITES, ("", ""));
            }
            saved
        };
        static OVER: Site = Site::new("m", "over_cap");
        let before = DROPPED.load(Relaxed);
        {
            let _g = enter(&OVER);
        }
        assert_eq!(
            DROPPED.load(Relaxed),
            before + 1,
            "one over-cap enter must bump the dropped counter"
        );
        // collect() surfaces the aggregate as a single count-only `[dropped]` row.
        let (rows, _, _) = collect();
        let d = rows
            .iter()
            .find(|r| r.name == "[dropped]")
            .expect("[dropped] row present");
        assert!(d.count >= 1 && d.total_ns == 0 && d.self_ns == 0);
        // Restore the registry (drops the padding and OVER's over-cap registration).
        global().names.lock().unwrap().truncate(saved_len);
    }

    /// `prof.out` reaches the reporter by being PUSHED at startup (S7) — the report itself
    /// never touches the environment. Round-trips the installed value, including the "unset =
    /// stderr only" state; a non-empty value is what makes [`report`] write its JSON.
    #[test]
    fn profile_out_is_pushed_not_read_from_the_environment() {
        assert_eq!(profile_out(), None, "nothing installed by default");
        set_profile_out(Some("/tmp/infr-profile.json"));
        assert_eq!(profile_out().as_deref(), Some("/tmp/infr-profile.json"));
        // The `INFR_PROF_OUT=` grammar: an empty path is stored, and `report` treats it as off.
        set_profile_out(Some(""));
        assert_eq!(profile_out().as_deref(), Some(""));
        set_profile_out(None);
        assert_eq!(profile_out(), None);
    }
}
