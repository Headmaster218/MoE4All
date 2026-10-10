//! Fixed-size caller-participating CPU task pool. No global Rayon pool reconfiguration.
use crate::pool::SpinPool;

pub struct CpuTaskPool(SpinPool, usize);

impl CpuTaskPool {
    /// Nonblocking end of a burst created with `with_idle_parking`; no empty job is dispatched.
    pub fn park_idle_workers(&self) {
        self.0.park_idle_workers();
    }
    pub fn new(threads: usize) -> Self {
        Self::with_min_spin(threads, 256)
    }

    /// Keep short multi-phase tasks awake across their phase barrier, without busy-waiting idle
    /// sessions. This affects only this private pool, never the CPU interpreter's pool.
    pub fn with_min_spin(threads: usize, min_spin: u32) -> Self {
        let mut cfg = infr_core::config::CpuCfg::default();
        cfg.spin = cfg.spin.max(min_spin);
        Self(
            SpinPool::new_sized_min_spin(&cfg, threads, min_spin),
            threads.max(1),
        )
    }

    /// Opt-in idle checks are absent from the normal pool's compile-time worker specialization.
    pub fn with_idle_parking(threads: usize, min_spin: u32) -> Self {
        let mut cfg = infr_core::config::CpuCfg::default();
        cfg.spin = cfg.spin.max(min_spin);
        Self(
            SpinPool::new_private_min_spin(&cfg, threads, min_spin),
            threads.max(1),
        )
    }

    pub fn parallelism(&self) -> usize {
        self.1
    }

    pub fn run(&self, tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        self.0.run(tasks, f);
    }

    pub fn finish_burst(&self) {
        self.0.park_workers();
    }

    pub fn for_chunks_mut<T: Send>(
        &self,
        data: &mut [T],
        chunk: usize,
        f: &(dyn Fn(usize, &mut [T]) + Sync),
    ) {
        self.0.for_chunks_mut(data, chunk, 1, f);
    }
}
