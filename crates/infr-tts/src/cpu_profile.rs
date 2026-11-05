//! Optional calling-thread wall counters. Categories can nest; do not sum them.
use std::{cell::Cell, time::Instant};

#[derive(Clone, Copy)]
#[allow(dead_code)] // A/B routes deliberately leave some counters at zero.
pub(crate) enum Phase {
    Conv,
    ConvPack,
    ConvLinear,
    ConvCopy,
    Transpose,
    TransposeLinear,
    TransposeScatter,
    Bias,
    Add,
    Leaky,
    ConvEpilogue,
}
thread_local! {static NS:Cell<[u64;11]>=const{Cell::new([0;11])};}
pub fn take() -> [f64; 11] {
    NS.with(|cell| cell.replace([0; 11]).map(|n| n as f64 / 1e9))
}
pub(crate) struct Timer(Phase, Instant);
impl Timer {
    pub(crate) fn start(phase: Phase) -> Self {
        Self(phase, Instant::now())
    }
}
impl Drop for Timer {
    fn drop(&mut self) {
        NS.with(|cell| {
            let mut ns = cell.get();
            ns[self.0 as usize] += self.1.elapsed().as_nanos() as u64;
            cell.set(ns);
        });
    }
}
