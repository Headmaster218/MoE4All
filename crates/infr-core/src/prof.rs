//! The per-op profiling seam: ONE collector, ONE label grammar, ONE report, for every backend.
//!
//! ## Why this exists
//!
//! Per-op profiling was re-invented four times. Every backend could answer "where did the forward
//! go", and every backend answered it in its own dialect: vulkan stamped timestamp queries and
//! printed `[prof2]`, metal accumulated host
//! encode wall and printed `── infr-metal profile`, cpu summed `Instant`s and printed `[prof-ops]`.
//! Four knobs (`INFR_PROF2`, `INFR_METAL_PROFILE`, `INFR_PROF_OPS`), four
//! label grammars, four sort orders, four notions of what the total is a percentage OF. All four
//! are now [`ProfCfg::ops`](crate::config::ProfCfg::ops) — one knob, `INFR_PROF_OPS`.
//!
//! That is worse than untidy. It cost real work:
//!
//! * Metal's table counted the **untimed warmup forward**. Only vulkan honored the
//!   suppression flag, so on the other the published share was diluted by work the bench
//!   itself excluded.
//! * Only vulkan fed [`infr_prof_rt`], so only vulkan appeared in the exit aggregate and in the
//!   `INFR_PROF_OUT` JSON. Correlating a host function table against device op time was a
//!   vulkan-only capability by accident, not by design.
//!
//! ## What is shared and what deliberately is not
//!
//! Shared here: the ACCOUNTING and the OUTPUT — [`OpProf`] (label → count + µs, folded into the
//! process aggregate and printed in one format), [`op_label`] (what two dispatches must agree on to
//! share a row), and [`enabled`] (the one predicate that decides whether to profile at all,
//! including the warmup-suppression AND).
//!
//! NOT shared, on purpose: **how a backend obtains a duration.** Those differ for hardware reasons,
//! not drift, and flattening them would silently degrade three of the four:
//!
//! * vulkan writes `BOTTOM_OF_PIPE` timestamps into a query pool and reads the whole pool back
//!   after the submit completes.
//! * metal has no free per-op device timing at all: its ops batch into one command buffer, so
//!   isolating an op's GPU wall means flushing after it (`prof.metal_device_time=flush`, which
//!   costs the batching) or sampling stage-boundary counters (`=counters`). Its cheap default mode
//!   measures host ENCODE time, which is a different quantity and is labeled as such.
//! * cpu measures host wall directly, which for a host backend IS the device time.
//!
//! So each backend keeps its own timing acquisition and calls [`OpProf::add`] with the result. The
//! unit is microseconds everywhere — the conversion belongs at the acquisition site, where the
//! clock's native unit is known (vulkan ticks × `timestamp_period`,
//! host `Duration`s).

use crate::config::ProfCfg;
use crate::graph::{AttnMask, Graph, Op};
use std::collections::BTreeMap;

/// Should this backend profile per-op right now?
///
/// The AND with [`infr_prof_rt::profiling_suppressed`] is the part that used to be vulkan-only. Benches
/// and warmup paths set that flag around untimed work (see `infr_llama::with_profiling_suppressed`), so
/// a backend that skips this predicate reports a table covering forwards the bench deliberately did
/// not time.
///
/// Read this ONCE per forward (or per recorder construction), not per op — every backend's op walk
/// is hot enough that a config indirection per op would show.
#[inline]
pub fn enabled(cfg: &ProfCfg) -> bool {
    cfg.ops && !suppressed()
}

/// Is per-op profiling currently suppressed by a warmup / untimed-work scope?
///
/// Re-exported through this module so a backend gating on it needs only the `infr-core` dependency
/// it already has, and so the whole profiling seam reads from ONE import.
#[inline]
pub fn suppressed() -> bool {
    infr_prof_rt::profiling_suppressed()
}

/// One profiling session's accumulated per-op time: label → (dispatches, total µs).
///
/// Scope is the backend's choice — vulkan folds one per submit and metal one per forward, cpu
/// one per `execute`. Whatever the scope, [`flush`](Self::flush) pushes the rows into the
/// process-wide aggregate, so the exit report is over the whole run regardless.
pub struct OpProf {
    /// Which backend produced these rows — printed in the header so a multi-backend run (paged MoE
    /// with a cpu fallback, a multi-GPU split) stays readable.
    backend: &'static str,
    /// What [`add`](Self::add) measured. Named, because it is NOT the same quantity on every
    /// backend and a table that doesn't say so invites comparing metal's encode µs against vulkan's
    /// device µs.
    unit: Unit,
    rows: BTreeMap<String, (u64, f64)>,
}

/// What a profiled duration actually measures.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unit {
    /// Time the DEVICE spent executing the op, from the device's own clocks (vulkan timestamp
    /// queries, metal's flush/counter modes). The number to optimize against.
    Device,
    /// Time the HOST spent encoding/submitting the op. Metal's cheap default mode and the cpu
    /// backend both report this — for cpu it IS the execution time; for metal it is emphatically
    /// not, and a large encode share is itself the finding (per-op command-buffer overhead).
    HostEncode,
}

impl Unit {
    fn label(self) -> &'static str {
        match self {
            Unit::Device => "device",
            Unit::HostEncode => "host-encode",
        }
    }
}

impl OpProf {
    pub fn new(backend: &'static str, unit: Unit) -> Self {
        OpProf {
            backend,
            unit,
            rows: BTreeMap::new(),
        }
    }

    /// Accumulate one dispatch of `label` costing `us` microseconds.
    #[inline]
    pub fn add(&mut self, label: impl Into<String>, us: f64) {
        self.add_n(label, us, 1);
    }

    /// Accumulate `count` dispatches of `label` costing `us` microseconds IN TOTAL — for backends
    /// that have already aggregated (vulkan folds its query pool by label before reporting).
    pub fn add_n(&mut self, label: impl Into<String>, us: f64, count: u64) {
        let e = self.rows.entry(label.into()).or_insert((0, 0.0));
        e.0 += count;
        e.1 += us;
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Total µs across every row.
    pub fn total_us(&self) -> f64 {
        self.rows.values().map(|(_, us)| us).sum()
    }

    /// Rows sorted by descending total time — the order every report prints, and the order a test
    /// asserts against.
    pub fn ranked(&self) -> Vec<(&str, u64, f64)> {
        let mut rows: Vec<(&str, u64, f64)> = self
            .rows
            .iter()
            .map(|(l, (n, us))| (l.as_str(), *n, *us))
            .collect();
        // Descending by total, then by label so equal-time rows have a stable order across runs
        // (BTreeMap iteration is already label-ordered, so this only matters for the ties).
        rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        rows
    }

    /// Push every row into the process-wide aggregate ([`infr_prof_rt::gpu_add`]) and print the
    /// table. Consumes the collector: a session reports once.
    ///
    /// Device-unit rows feed the aggregate; host-encode rows do NOT — mixing metal's encode time
    /// into the same exit table as vulkan's device time would produce a sum that means nothing.
    /// They still print.
    pub fn flush(self) {
        if self.rows.is_empty() {
            return;
        }
        let total = self.total_us();
        eprintln!(
            "[prof:{}] per-op {} time — {:.1} ms over {} dispatches",
            self.backend,
            self.unit.label(),
            total / 1000.0,
            self.rows.values().map(|(n, _)| n).sum::<u64>(),
        );
        eprintln!(
            "[prof:{}] {:>10} {:>7} {:>8} {:>9}  op",
            self.backend, "ms", "share", "count", "us/ea"
        );
        for (label, n, us) in self.ranked() {
            if self.unit == Unit::Device {
                infr_prof_rt::gpu_add(label, us, n);
            }
            let share = if total > 0.0 { us / total * 100.0 } else { 0.0 };
            let each = if n > 0 { us / n as f64 } else { 0.0 };
            eprintln!(
                "[prof:{}] {:>10.2} {share:>6.1}% {n:>8} {each:>9.1}  {label}",
                self.backend,
                us / 1000.0,
            );
        }
    }
}

/// The profile key for one op: its kind, plus the shape that makes its cost.
///
/// Two dispatches share a row exactly when they do the same work — so a 28-layer model folds into a
/// handful of rows instead of 28 copies of each, and a row's `us/ea` is meaningful (it is one
/// shape's cost, not an average over shapes that differ by 100x).
///
/// The three itemized variants are the ones where the shape IS the cost, and they are the three
/// that have been the answer in every profiled perf slice so far: `Linear`'s `m` separates decode's
/// GEMV from prefill's GEMM on the same weight, `Attention`'s `kv_len` is the whole at-depth story,
/// and `MoeFfn`'s expert geometry is what a 30B-A3B forward is made of. Everything else falls back
/// to the bare op kind — a norm is a norm.
///
/// Vulkan is the deliberate exception: it labels by KERNEL name, not op kind, because its walk is a
/// recorder that lowers one op into several dispatches and the kernel is the finer, truer unit
/// there. Its shape itemizer (`prof.prof2_shapes`) is that backend's equivalent of this function.
pub fn op_label(op: &Op, g: &Graph) -> String {
    match *op {
        Op::Linear {
            weight,
            m,
            in_f,
            out_f,
            ..
        } => format!("Linear m={m} {in_f}x{out_f} {:?}", g.desc(weight).dtype),
        Op::Attention {
            rows,
            kv_len,
            n_head,
            n_kv,
            head_dim,
            mask,
            ..
        } => format!(
            "Attention rows={rows} kv={kv_len} h={n_head}/{n_kv} d={head_dim} {}",
            match mask {
                AttnMask::Causal => "causal",
                AttnMask::SlidingWindow(_) => "swa",
                AttnMask::Canvas { .. } => "canvas",
            }
        ),
        Op::MoeFfn {
            gate_exps,
            down_exps,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            ..
        } => format!(
            "MoeFfn ne={ne} nff={n_ff_exp} {n_used}/{n_expert} {:?}/{:?}",
            g.desc(gate_exps).dtype,
            g.desc(down_exps).dtype
        ),
        _ => op.kind().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::graph::{Graph, Op};
    use crate::tensor::{DType, TensorDesc};

    #[test]
    fn enabled_requires_the_knob_and_clears_under_suppression() {
        let mut cfg = Config::default();
        assert!(!enabled(&cfg.prof), "off by default");
        cfg.prof.ops = true;
        assert!(enabled(&cfg.prof));

        // The warmup path's flag must win over the knob — this AND is what metal was
        // missing, and it is the difference between a table of the timed reps and a table
        // polluted by the untimed warmup forward.
        let prev = infr_prof_rt::set_profiling_suppressed(true);
        assert!(!enabled(&cfg.prof), "suppression must beat the knob");
        infr_prof_rt::set_profiling_suppressed(prev);
        assert!(enabled(&cfg.prof), "and restore must un-suppress");
    }

    #[test]
    fn ranked_is_descending_by_total_not_by_count() {
        let mut p = OpProf::new("test", Unit::Device);
        p.add("cheap", 10.0);
        p.add("cheap", 10.0);
        p.add("cheap", 10.0);
        p.add("dear", 100.0);
        assert_eq!(p.total_us(), 130.0);
        let rows = p.ranked();
        assert_eq!(rows[0].0, "dear", "sorted by time, not by dispatch count");
        assert_eq!(rows[0].1, 1);
        assert_eq!(rows[1], ("cheap", 3, 30.0));
    }

    #[test]
    fn add_n_folds_a_pre_aggregated_batch() {
        let mut p = OpProf::new("test", Unit::Device);
        p.add_n("k", 90.0, 3);
        p.add("k", 10.0);
        assert_eq!(p.ranked(), vec![("k", 4, 100.0)]);
    }

    /// A tiny graph with the three itemized ops plus one that falls back to its kind.
    fn label_graph() -> (Graph, Vec<Op>) {
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![4], DType::F32));
        let w = g.weight(TensorDesc::new(vec![16], DType::Q4K));
        let dst = g.internal(TensorDesc::new(vec![4], DType::F32));
        let kc = g.input(TensorDesc::new(vec![64], DType::F16));
        let vc = g.input(TensorDesc::new(vec![64], DType::F16));
        let ops = vec![
            Op::Linear {
                x,
                weight: w,
                dst,
                m: 1,
                in_f: 8,
                out_f: 2,
                w_off: 0,
            },
            Op::Attention {
                q: x,
                k_cache: kc,
                v_cache: vc,
                dst,
                rows: 1,
                kv_len: 512,
                n_head: 8,
                n_kv: 2,
                head_dim: 64,
                scale: 1.0,
                mask: AttnMask::Causal,
                pos: 0,
            },
            Op::RmsNorm {
                x,
                weight: w,
                dst,
                rows: 1,
                dim: 4,
                eps: 1e-6,
            },
        ];
        (g, ops)
    }

    /// The itemized labels carry the cost-making shape; everything else is the bare kind.
    #[test]
    fn op_label_itemizes_the_shape_that_makes_the_cost() {
        let (g, ops) = label_graph();
        assert_eq!(op_label(&ops[0], &g), "Linear m=1 8x2 Q4K");
        assert_eq!(
            op_label(&ops[1], &g),
            "Attention rows=1 kv=512 h=8/2 d=64 causal"
        );
        assert_eq!(op_label(&ops[2], &g), "RmsNorm", "unitemized ops = kind");
    }

    /// **Anti-drift tripwire.** Every backend must report through [`OpProf::flush`] — which is the
    /// single place that folds into the process-wide aggregate, so it is also the single place that
    /// decides the label grammar, the sort order, the columns and the units.
    ///
    /// Four backends previously each called their own accounting and their own `eprintln!`, and the
    /// cost was not aesthetic: three of the four silently profiled the bench's untimed warmup, and
    /// three of the four never appeared in the exit report or the `INFR_PROF_OUT` JSON at all.
    /// A fifth backend (cuda and sycl both have plans in `docs/`) would have made it five. This
    /// fails the moment a new one reaches past the seam.
    #[test]
    fn no_backend_feeds_the_aggregate_behind_the_shared_reporter() {
        let Some(crates) = repo_crates_dir() else {
            return; // packaged/vendored build — no `crates/` to scan
        };
        let mut offenders = Vec::new();
        for entry in std::fs::read_dir(&crates).expect("read crates/") {
            let dir = entry.expect("crates/ entry").path();
            // The seam itself and the aggregate's own crate are where `gpu_add` is DEFINED and
            // called on everyone's behalf; every other crate must go through `OpProf`.
            let name = dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            if name == "infr-prof-rt" {
                continue;
            }
            scan_for_gpu_add(&dir.join("src"), &mut offenders);
        }
        assert!(
            offenders.is_empty(),
            "these sites push into the per-op device aggregate directly instead of through \
             `infr_core::prof::OpProf` — route them through the shared reporter so the label \
             grammar, warmup suppression and report format stay one implementation: {offenders:#?}"
        );
    }

    fn repo_crates_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")) // …/crates/infr-core
            .parent()?
            .to_path_buf();
        dir.is_dir().then_some(dir)
    }

    /// Every non-comment `gpu_add(` call under `dir`, except the one inside this module's
    /// [`OpProf::flush`] (identified by path, so moving the call still trips the check).
    fn scan_for_gpu_add(dir: &std::path::Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan_for_gpu_add(&path, out);
                continue;
            }
            if path.extension().is_some_and(|e| e == "rs") && !path.ends_with("prof.rs") {
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                for (n, line) in text.lines().enumerate() {
                    let code = line.trim_start();
                    if !code.starts_with("//") && code.contains("gpu_add(") {
                        out.push(format!("{}:{}", path.display(), n + 1));
                    }
                }
            }
        }
    }

    /// The point of itemizing: the SAME op kind at two different shapes must not share a row, or
    /// prefill's GEMM hides inside decode's GEMV average.
    #[test]
    fn op_label_separates_the_same_kind_at_different_shapes() {
        let (g, ops) = label_graph();
        let Op::Attention { .. } = ops[1] else {
            unreachable!()
        };
        let mut deep = ops[1].clone();
        if let Op::Attention { kv_len, .. } = &mut deep {
            *kv_len = 2048;
        }
        assert_ne!(op_label(&ops[1], &g), op_label(&deep, &g));

        let mut prefill = ops[0].clone();
        if let Op::Linear { m, .. } = &mut prefill {
            *m = 512;
        }
        assert_ne!(
            op_label(&ops[0], &g),
            op_label(&prefill, &g),
            "decode GEMV and prefill GEMM on one weight must not share a row"
        );
    }
}
