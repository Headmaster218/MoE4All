//! Opt-in execution profiler (`prof.ops` / `INFR_PROF_OPS`). Aggregates, across the whole run, the
//! per-op wall time and the commit+wait ("dispatch") wall vs actual GPU-active time, then prints a
//! summary on drop. This is the evidence for *where* the reference backend spends its time — the
//! per-op command-buffer barrier, not the arithmetic.

use std::collections::HashMap;
use std::time::Duration;

#[derive(Default)]
pub(crate) struct Profile {
    /// op name → (call count, total wall time spent in `run_op` for that op)
    per_op: HashMap<&'static str, (u64, Duration)>,
    /// op name → total GPU wall (commit+wait) attributed to that op — only populated under
    /// `prof.metal_device_time=flush`, where the batch is flushed after each op so its GPU time is
    /// isolable. Costs the batching, so it's for analysis, not the fast path.
    per_op_gpu: HashMap<&'static str, Duration>,
    /// total wall time inside `dispatch()` (commit + GPU schedule + wait), summed over all ops
    dispatch_wall: Duration,
    dispatch_count: u64,
    forwards: u64,
}

impl Profile {
    pub fn add_op(&mut self, name: &'static str, d: Duration) {
        let e = self.per_op.entry(name).or_default();
        e.0 += 1;
        e.1 += d;
    }

    pub fn add_op_gpu(&mut self, name: &'static str, d: Duration) {
        *self.per_op_gpu.entry(name).or_default() += d;
    }

    pub fn add_dispatch(&mut self, wall: Duration) {
        self.dispatch_wall += wall;
        self.dispatch_count += 1;
    }

    pub fn add_forward(&mut self) {
        self.forwards += 1;
    }

    /// Print through the shared reporter ([`infr_core::prof::OpProf`]) so this backend's table has
    /// the same columns, the same sort and the same `[prof:<backend>]` tag as vulkan's and
    /// the cpu backend's — and so its DEVICE rows land in the process-wide exit aggregate and the
    /// `INFR_PROF_OUT` JSON, which they never did before.
    ///
    /// Two tables, not one, because Metal measures two different quantities and only one of them is
    /// device time: the always-available per-op number is host ENCODE wall (ops batch into one
    /// command buffer, so nothing else is free), while GPU wall per op exists only in the modes
    /// that pay for it (`prof.metal_device_time=flush` flushes after each op; `=counters` samples
    /// stage-boundary counters). Summing them would be meaningless, so they print as separate units
    /// and only the device one feeds the aggregate.
    pub fn print_summary(&self) {
        if self.forwards == 0 {
            return;
        }
        let total: Duration = self.per_op.values().map(|(_, d)| *d).sum();
        let total_s = total.as_secs_f64().max(1e-9);

        // print-ok: the `prof.stages` profiling REPORT — same category as `infr_core::prof::OpProf`'s
        // table (which the next lines feed): column-aligned output the user asked for, not a
        // diagnostic, and a per-line `tracing` prefix would destroy the columns.
        eprintln!("\n── infr-metal profile ({} forwards) ──", self.forwards);
        if !self.per_op_gpu.is_empty() {
            let mut p = infr_core::prof::OpProf::new("metal", infr_core::prof::Unit::Device);
            for (name, d) in &self.per_op_gpu {
                let calls = self.per_op.get(name).map(|(c, _)| *c).unwrap_or(1);
                p.add_n(*name, d.as_secs_f64() * 1e6, calls);
            }
            p.flush();
        }
        let mut p = infr_core::prof::OpProf::new("metal", infr_core::prof::Unit::HostEncode);
        for (name, (calls, d)) in &self.per_op {
            p.add_n(*name, d.as_secs_f64() * 1e6, *calls);
        }
        p.flush();

        // The per-op wall above is CPU-side *encode* time (each op appends to the batch). The GPU
        // actually runs at flush (commit + wait), which the batch defers — so report the two
        // separately rather than as fractions of each other.
        let dwall = self.dispatch_wall.as_secs_f64();
        let f = self.forwards as f64;
        // print-ok: report line, see above.
        eprintln!(
            "── CPU encode: {:.1} ms total ({:.2} ms/forward)",
            total_s * 1e3,
            total_s * 1e3 / f
        );
        // print-ok: report line, see above.
        eprintln!(
            "── GPU (commit+wait): {:.1} ms total ({:.2} ms/forward) over {} command buffers ({:.2}/forward)",
            dwall * 1e3,
            dwall * 1e3 / f,
            self.dispatch_count,
            self.dispatch_count as f64 / f
        );
    }
}
