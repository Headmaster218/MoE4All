//! Persistent spin-pool for the CPU op interpreter (threadpool restructure phase 2).
//!
//! Rayon's fork-join costs a wake/steal/sleep cycle per parallel op; between the ~400 ops of one
//! DiffusionGemma denoise graph that latency (and the deque/epoch plumbing measured at ~6% of
//! thread-time) is pure overhead. This pool keeps `N-1` workers alive across the whole graph:
//! a job is handed off by bumping a generation counter the workers spin on (~1µs handoff while
//! hot), tasks are claimed dynamically with a `fetch_add` cursor (straggler-proof), and workers
//! park after a short spin budget so an idle pool costs nothing (the host self-conditioning gap
//! between denoise steps, rayon-side MoeFfn work, and plain idle all put them to sleep).
//!
//! Scheduling only — every converted call site runs the exact same per-row math in the same
//! order as its rayon predecessor, so outputs are bit-identical.
//!
//! Clearing `kernels.cpu.spinpool` routes `run` through rayon instead
//! (A/B + escape hatch).

use infr_core::config::CpuCfg;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

/// One in-flight job: an index-space of `n_tasks`, claimed dynamically by all threads.
type Job = &'static (dyn Fn(usize) + Sync);

struct Shared {
    /// The current job's closure, valid from the `seq` bump until every worker has checked in
    /// (`done == workers`) — `run` does not return (and the borrowed closure cannot die) before
    /// that, so a worker can never observe a stale/torn slot: it only reads `job` after seeing a
    /// NEW `seq`, and no new `seq` can be published while any worker is still on the old job.
    job: UnsafeCell<Option<Job>>,
    /// Generation counter: bumped once per job; workers spin on it changing.
    seq: AtomicUsize,
    /// Dynamic task cursor (`fetch_add` claim), reset per job.
    cursor: AtomicUsize,
    n_tasks: AtomicUsize,
    /// Workers that finished the current job (drained the cursor).
    done: AtomicUsize,
    /// A task panicked (caught per-task so `done` still advances; `run` re-panics).
    panicked: AtomicBool,
    shutdown: AtomicBool,
    /// Ceiling for the adaptive spin budget (constant `SpinPool::spin_limit` since phase 3 removed
    /// the last rayon section from the interpreter; kept a field for future per-graph tuning).
    budget_cap: AtomicU32,
    /// Per-worker "I am parked" flags — see the park handshake in `worker_loop`.
    sleeping: Vec<AtomicBool>,
}

// SAFETY: `job` is only written by `run` while no worker is between jobs' check-ins (see the
// field doc), and only read by workers after an Acquire load of `seq` that the write
// happens-before (SeqCst bump).
unsafe impl Sync for Shared {}

pub(crate) struct SpinPool {
    shared: Arc<Shared>,
    handles: Vec<std::thread::JoinHandle<()>>,
    /// Worker thread count (callers participate too, so parallelism = workers + 1).
    workers: usize,
    /// `!kernels.cpu.spinpool`: run jobs through rayon instead.
    rayon_fallback: bool,
    /// CEILING for the adaptive per-worker spin budget, and the caller-side wait loop's
    /// yield threshold — `kernels.cpu.spin`, read ONCE at construction.
    ///
    /// Two measured failure modes bound the budget: too LONG and the spinning workers' SMT
    /// siblings throttle the op loop's SERIAL bookkeeping between pool ops (per-op profile:
    /// RmsNorm/Add more than DOUBLED; DG exec 2.87 → 3.18s at a fixed 32k) — rayon's MoeFfn itself
    /// was unharmed once a pause mechanism parked waiters for it (both since removed with the rayon
    /// sections themselves — phase 3 staged MoeFfn onto this pool). Too SHORT and dense prefill pays
    /// a worker wake per op (qwen3 pp512 404 → 356 t/s at a fixed 1k). The adaptive budget collapses
    /// after a park and regrows on jobs arriving mid-spin, so the ceiling can be generous.
    ///
    /// A FIELD, not a `OnceLock` memo (`docs/config-plan.md` §10.6): the value comes off the
    /// backend's `Config`, so a per-call read would be a `getenv`-shaped cost on the wait loop and
    /// a memo would pin the first pool's value process-wide.
    spin_limit: u32,
    /// Serializes `run` — the pool holds ONE job; concurrent dispatch is a caller bug
    /// (converted call sites are all reached from the single-threaded `execute` op loop).
    in_run: AtomicBool,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn worker_loop(me: usize, shared: Arc<Shared>) {
    // Baseline generation is the CONSTRUCTION-time value (0), not a fresh load: a worker whose
    // OS thread starts late — after the first `run()` already bumped `seq` — must still join
    // that in-flight job (its check-in is what `run` is blocked on). Loading `seq` here instead
    // would make the late worker treat the live job as already-seen and deadlock the caller.
    let mut seen = 0usize;
    // ADAPTIVE spin budget, per worker: spinning is only worth its SMT-sibling tax (it throttles
    // the op loop's serial sections running on the paired hyperthread) when the next job arrives
    // before the budget runs out. Jobs arriving mid-spin → inter-op gaps are short (dense
    // prefill) → grow toward `SpinPool::spin_limit`. Having to park → gaps are long (decode's serial
    // stretches, MoE's rayon section) → collapse to a near-immediate park. Measured: fixed
    // budgets force a 3-way tradeoff (qwen3 pp 416 vs tg 44 vs DG 2.87s — each best at a
    // DIFFERENT value); the gap history picks the right regime per phase automatically.
    const MIN_SPIN: u32 = 256;
    let mut budget = MIN_SPIN;
    loop {
        // ── Wait for a new generation ────────────────────────────────────────────────
        let mut spins = 0u32;
        let mut parked = false;
        loop {
            let s = shared.seq.load(Ordering::Acquire);
            if s != seen {
                seen = s;
                break;
            }
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            spins += 1;
            if spins < budget {
                std::hint::spin_loop();
            } else {
                // Park handshake: publish the flag, RE-CHECK seq/shutdown (SeqCst on both sides
                // orders flag-publish vs. the dispatcher's seq-bump-then-read-flag), then park.
                // A wake that slips in between the re-check and `park()` is absorbed by park's
                // token semantics (`unpark` before `park` makes `park` return immediately).
                shared.sleeping[me].store(true, Ordering::SeqCst);
                if shared.seq.load(Ordering::SeqCst) != seen
                    || shared.shutdown.load(Ordering::SeqCst)
                {
                    shared.sleeping[me].store(false, Ordering::SeqCst);
                    continue;
                }
                std::thread::park();
                shared.sleeping[me].store(false, Ordering::SeqCst);
                spins = 0;
                parked = true;
            }
        }
        budget = if parked {
            MIN_SPIN
        } else {
            (budget * 4).min(shared.budget_cap.load(Ordering::Relaxed))
        };
        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }
        // ── Drain the task cursor ────────────────────────────────────────────────────
        // SAFETY: `job` was written before the observed `seq` bump and stays alive until this
        // worker (and all others) increment `done` — see `Shared::job`'s doc.
        let job = unsafe { (*shared.job.get()).expect("spin-pool: seq bumped without a job") };
        let n = shared.n_tasks.load(Ordering::Acquire);
        loop {
            let t = shared.cursor.fetch_add(1, Ordering::Relaxed);
            if t >= n {
                break;
            }
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(t))).is_err() {
                shared.panicked.store(true, Ordering::Release);
            }
        }
        shared.done.fetch_add(1, Ordering::Release);
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl SpinPool {
    /// Thread count follows rayon's (`RAYON_NUM_THREADS` / available parallelism) so `-t` pins
    /// both pools identically. Both knobs (`spin`, `spinpool`) are read out of `cfg` HERE and
    /// never again — see [`SpinPool::spin_limit`].
    pub(crate) fn new(cfg: &CpuCfg) -> Self {
        let n_threads = rayon::current_num_threads().max(1);
        let workers = n_threads - 1;
        let rayon_fallback = !cfg.spinpool;
        let spin_limit = cfg.spin;
        let shared = Arc::new(Shared {
            job: UnsafeCell::new(None),
            seq: AtomicUsize::new(0),
            cursor: AtomicUsize::new(0),
            n_tasks: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            budget_cap: AtomicU32::new(spin_limit),
            sleeping: (0..workers).map(|_| AtomicBool::new(false)).collect(),
        });
        let handles = (0..workers)
            .map(|me| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name(format!("infr-spin-{me}"))
                    .spawn(move || worker_loop(me, sh))
                    .expect("spin-pool: spawn failed")
            })
            .collect();
        SpinPool {
            shared,
            handles,
            workers,
            rayon_fallback,
            spin_limit,
            in_run: AtomicBool::new(false),
        }
    }

    /// Run `f(0..n_tasks)` across the pool (caller participates). Dynamic task claim; returns
    /// once every task ran AND every worker checked in (the check-in is what makes the borrowed
    /// closure's lifetime sound — see `Shared::job`).
    pub(crate) fn run(&self, n_tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        if n_tasks == 0 {
            return;
        }
        // Single-task jobs and worker-less pools short-circuit: no handoff, no wake.
        if n_tasks == 1 || self.workers == 0 {
            for t in 0..n_tasks {
                f(t);
            }
            return;
        }
        // Busy pool (a second graph executing concurrently on this backend — e.g. parallel
        // serve sessions) or explicit fallback: route through rayon, which handles concurrent
        // callers natively. The pool itself holds ONE job at a time.
        if self.rayon_fallback
            || self
                .in_run
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
        {
            use rayon::prelude::*;
            (0..n_tasks).into_par_iter().for_each(f);
            return;
        }
        let sh = &self.shared;
        // SAFETY: lifetime erasure of `f` — sound because this function does not return until
        // every worker has incremented `done`, after which no worker touches the slot again.
        unsafe {
            *sh.job.get() = Some(std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f));
        }
        sh.cursor.store(0, Ordering::Relaxed);
        sh.n_tasks.store(n_tasks, Ordering::Release);
        sh.done.store(0, Ordering::Relaxed);
        sh.seq.fetch_add(1, Ordering::SeqCst);
        for (i, flag) in sh.sleeping.iter().enumerate() {
            if flag.load(Ordering::SeqCst) {
                self.handles[i].thread().unpark();
            }
        }
        // Participate.
        loop {
            let t = sh.cursor.fetch_add(1, Ordering::Relaxed);
            if t >= n_tasks {
                break;
            }
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(t))).is_err() {
                sh.panicked.store(true, Ordering::Release);
            }
        }
        // Wait for every worker's check-in (they may still be draining the cursor's tail).
        // Periodic yield: under thread oversubscription (tests spawning several pools, a busy
        // rayon pool alongside) a pure spin here can starve the very workers it waits on.
        let mut spins = 0u32;
        while sh.done.load(Ordering::Acquire) < self.workers {
            spins += 1;
            if spins < self.spin_limit {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        unsafe {
            *sh.job.get() = None;
        }
        self.in_run.store(false, Ordering::Release);
        if sh.panicked.swap(false, Ordering::AcqRel) {
            panic!("spin-pool: a task panicked (caught per-task; state may be incomplete)");
        }
    }

    /// Chunk `data` into `chunk`-sized pieces and run `f(chunk_index, piece)` across the pool,
    /// `grain` consecutive chunks per claimed task (coarsening for huge chunk counts). The
    /// last piece may be shorter (`data.len() % chunk`). Bit-identity: pure scheduling — each
    /// piece is processed by exactly one thread with unchanged math.
    pub(crate) fn for_chunks_mut<T: Send>(
        &self,
        data: &mut [T],
        chunk: usize,
        grain: usize,
        f: &(dyn Fn(usize, &mut [T]) + Sync),
    ) {
        let len = data.len();
        if len == 0 {
            return;
        }
        let n_chunks = len.div_ceil(chunk);
        let grain = grain.max(1);
        let n_tasks = n_chunks.div_ceil(grain);
        let base = SendPtr(data.as_mut_ptr());
        self.run(n_tasks, &move |task| {
            let c0 = task * grain;
            let c1 = (c0 + grain).min(n_chunks);
            for c in c0..c1 {
                let start = c * chunk;
                let end = (start + chunk).min(len);
                // SAFETY: chunk ranges are disjoint across tasks and in-bounds by construction.
                let piece =
                    unsafe { std::slice::from_raw_parts_mut(base.get().add(start), end - start) };
                f(c, piece);
            }
        });
    }

    /// `(0..n).map(f).collect()` across the pool, order-preserving.
    pub(crate) fn collect<T: Send>(&self, n: usize, f: &(dyn Fn(usize) -> T + Sync)) -> Vec<T> {
        let mut out: Vec<std::mem::MaybeUninit<T>> = Vec::with_capacity(n);
        // SAFETY: every index 0..n is written exactly once below before assume-init.
        unsafe { out.set_len(n) };
        let base = SendPtr(out.as_mut_ptr());
        self.run(n, &move |i| {
            // SAFETY: each task writes only its own slot.
            unsafe { base.get().add(i).write(std::mem::MaybeUninit::new(f(i))) };
        });
        // SAFETY: all n slots are initialized. `run` returns only after every task completed, and
        // if any task panicked `run` re-panics before we get here, so the rebuild below is only
        // reached on the all-written path.
        //
        // LEAK ON PANIC (not UB, but not free either): `run` catches each task's panic
        // individually and only re-panics once ALL tasks have finished — so by the time that panic
        // unwinds through here, the non-panicking tasks have already written their slots. `out` is
        // still typed `Vec<MaybeUninit<T>>`, and dropping a `MaybeUninit<T>` is a no-op, so every
        // `T` that WAS constructed leaks its destructor. That is not free at the call sites this
        // has: `T` is `Q8`/`Q8x32`, each holding several heap `Vec`s, so an aborted quantize of an
        // m-row activation leaks up to m of them. It is a real leak, not merely a lost value —
        // fixing it would mean tracking which indices were written and dropping those in place.
        // The old comment claimed only that a panic "propagates out of `run` before we get here",
        // which is true but says nothing about what the already-written slots cost on that path.
        //
        // The rebuild is done via `from_raw_parts` rather than
        // `transmute::<Vec<MaybeUninit<T>>, Vec<T>>`: `Vec` is not `repr(C)`, so transmuting
        // between two instantiations of it is not a language guarantee even though
        // `MaybeUninit<T>` and `T` have identical layout — the compiler is free to lay out
        // `Vec<A>` and `Vec<B>` differently. Taking the pointer/len/capacity apart and handing
        // them back to `Vec::from_raw_parts` is the sanctioned form and relies only on the
        // documented `MaybeUninit<T>`-to-`T` layout equivalence. (`Vec::into_raw_parts` is still
        // unstable, hence the manual `forget`.)
        let (ptr, len, cap) = (out.as_mut_ptr(), out.len(), out.capacity());
        std::mem::forget(out);
        // SAFETY: `ptr`/`len`/`cap` come straight from a `Vec<MaybeUninit<T>>` that we have
        // `forget`ten (so it will not also free the allocation), `MaybeUninit<T>` has the same
        // size and alignment as `T`, and all `len` elements are initialized per the argument
        // above — so the allocation is valid to reinterpret as a `Vec<T>` of the same capacity.
        unsafe { Vec::from_raw_parts(ptr as *mut T, len, cap) }
    }
}

/// Raw base pointer that may cross thread boundaries; safety is argued at each use site
/// (disjoint index ranges per task). Accessed via [`SendPtr::get`], NOT the field — edition-2021
/// closures capture individual FIELDS, and a captured bare `*mut T` loses these unsafe impls.
pub(crate) struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
// not instrumented: per-task pointer accessors in the dispatch hot loop, too small to probe
impl<T> SendPtr<T> {
    pub(crate) fn new(p: *mut T) -> Self {
        SendPtr(p)
    }
    pub(crate) fn get(&self) -> *mut T {
        self.0
    }
}
// not instrumented: trivial Copy-clone in hot loops; a span here would be pure overhead
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.seq.fetch_add(1, Ordering::SeqCst);
        for h in &self.handles {
            h.thread().unpark();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn spin_pool_runs_every_task_once() {
        let pool = SpinPool::new(&CpuCfg::default());
        for &n in &[1usize, 2, 7, 64, 1000, 10007] {
            let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
            pool.run(n, &|t| {
                hits[t].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
                "n={n}: some task ran zero or multiple times"
            );
        }
    }

    #[test]
    fn spin_pool_chunks_and_collect() {
        let pool = SpinPool::new(&CpuCfg::default());
        let mut v = vec![0u32; 1000];
        pool.for_chunks_mut(&mut v, 16, 3, &|c, piece| {
            for (i, x) in piece.iter_mut().enumerate() {
                *x = (c * 16 + i) as u32;
            }
        });
        assert!(v.iter().enumerate().all(|(i, &x)| x == i as u32));
        let got = pool.collect(257, &|i| i * 2);
        assert!(got.iter().enumerate().all(|(i, &x)| x == i * 2));
    }

    #[test]
    fn spin_pool_reusable_across_many_jobs() {
        let pool = SpinPool::new(&CpuCfg::default());
        let acc = AtomicU64::new(0);
        for _ in 0..200 {
            pool.run(32, &|_| {
                acc.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(acc.load(Ordering::Relaxed), 200 * 32);
    }

    /// `kernels.cpu.spin` reaches the two places it is used — the workers' adaptive-budget ceiling
    /// and the caller-side wait loop's yield threshold — from the `CpuCfg` the pool was built with,
    /// and the SHIPPED default is still `1 << 15`. Before S3 this was a process-wide `OnceLock`
    /// memo of the knob's env spelling, so a second pool could not see a different value at all
    /// (§10.6); two pools with different budgets in one test is what could not be written before.
    #[test]
    fn spin_budget_comes_off_the_config_per_pool() {
        assert_eq!(CpuCfg::default().spin, 1 << 15);
        let dflt = SpinPool::new(&CpuCfg::default());
        assert_eq!(dflt.spin_limit, 1 << 15);
        assert_eq!(dflt.shared.budget_cap.load(Ordering::Relaxed), 1 << 15);

        let tight = SpinPool::new(&CpuCfg {
            spin: 4,
            ..CpuCfg::default()
        });
        assert_eq!(tight.spin_limit, 4);
        assert_eq!(tight.shared.budget_cap.load(Ordering::Relaxed), 4);
        // Scheduling-only knob: a near-zero budget parks workers immediately but must still run
        // every task exactly once.
        let hits: Vec<AtomicU64> = (0..777).map(|_| AtomicU64::new(0)).collect();
        tight.run(777, &|t| {
            hits[t].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    /// `kernels.cpu.spinpool` cleared routes `run` through rayon —
    /// scheduling only, so the observable contract is unchanged: every task still runs once.
    #[test]
    fn cleared_spinpool_falls_back_to_rayon() {
        assert!(CpuCfg::default().spinpool);
        assert!(!SpinPool::new(&CpuCfg::default()).rayon_fallback);

        let pool = SpinPool::new(&CpuCfg {
            spinpool: false,
            ..CpuCfg::default()
        });
        assert!(pool.rayon_fallback);
        let hits: Vec<AtomicU64> = (0..1000).map(|_| AtomicU64::new(0)).collect();
        pool.run(1000, &|t| {
            hits[t].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }
}
