//! Optional, calling-thread counters for the native CPU wrapper A/B experiment.
//! Dispatch includes packing, SGEMM, pool handoff and existing output transpose.
//! Summed SGEMM task wall time overlaps across workers and is not wall latency.
use std::{cell::Cell, time::Instant};

#[derive(Default, Clone, Copy, Debug)]
pub struct Counters {
    pub linear_calls: u64,
    pub linear_wall_ns: u64,
    pub dispatch_calls: u64,
    pub dispatch_wall_ns: u64,
    pub parallel_sgemm_task_wall_ns_sum: u64,
}
thread_local! {static COUNTERS:Cell<Counters>=const{Cell::new(Counters{linear_calls:0,linear_wall_ns:0,dispatch_calls:0,dispatch_wall_ns:0,parallel_sgemm_task_wall_ns_sum:0})};}
pub fn take() -> Counters {
    COUNTERS.with(|v| v.replace(Counters::default()))
}
pub struct LinearTimer(Instant);
impl LinearTimer {
    pub fn start() -> Self {
        Self(Instant::now())
    }
}
impl Drop for LinearTimer {
    fn drop(&mut self) {
        COUNTERS.with(|v| {
            let mut c = v.get();
            c.linear_calls += 1;
            c.linear_wall_ns += self.0.elapsed().as_nanos() as u64;
            v.set(c);
        });
    }
}
pub(super) struct DispatchTimer(Instant);
impl DispatchTimer {
    pub fn start() -> Self {
        Self(Instant::now())
    }
}
impl Drop for DispatchTimer {
    fn drop(&mut self) {
        COUNTERS.with(|v| {
            let mut c = v.get();
            c.dispatch_calls += 1;
            c.dispatch_wall_ns += self.0.elapsed().as_nanos() as u64;
            v.set(c);
        });
    }
}
pub(super) fn add_sgemm_task_ns(ns: u64) {
    COUNTERS.with(|v| {
        let mut c = v.get();
        c.parallel_sgemm_task_wall_ns_sum += ns;
        v.set(c);
    });
}
