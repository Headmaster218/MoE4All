//! Single-op microbench — isolate WHY the instrumented decode trace shows Vulkan0 running the
//! same shader at the same geometry 5-15x slower than Vulkan2.
//!
//! The op trace (`OP_TRACE=1`) timestamps every dispatch, which localized the damage to a handful
//! of kernels (`silu_mul`, `qwen_hc_mix`, `qwen_hc_inject`, `qwen_hc_norm`, `expert_gateup`) but
//! cannot say whether the cost is GPU execution, per-submit latency, or host-side recording. This
//! probe replays those kernels in isolation against the SAME device-local buffer shape the model
//! uses, in two regimes:
//!
//!   * `batched` — `REPS` dispatches recorded into ONE command buffer, one submit+wait. There is no
//!     CPU/GPU bubble between dispatches, so this is the device's steady-state kernel throughput.
//!   * `spread`  — `REPS` recorders, each ONE dispatch + its own submit+wait. This adds the full
//!     submit+sync round trip, i.e. what a non-replayed decode step pays per op.
//!
//! `spread - batched` is the per-dispatch submit cost; `batched` alone is the GPU. A kernel that is
//! slow in BOTH regimes is a GPU/driver-state problem (desktop contention, post-reset clocks); a
//! kernel slow only in `spread` is host-side submit overhead, which the replay/overlap work fixes.
//!
//! `grid=1` (`n=64`) is the floor: one 64-thread workgroup, trivially fast on any device, so it
//! measures pure per-dispatch cost with no work — the cleanest read on "is device0 just slow to
//! dispatch?".
//!
//! Run (one device at a time, nothing else on the GPU):
//!   INFR_DEV=Vulkan0 cargo test -p infr-vulkan --release --test op_microbench -- --ignored --nocapture
//!   INFR_DEV=Vulkan2 cargo test -p infr-vulkan --release --test op_microbench -- --ignored --nocapture
//! (the cargo wrapper swallows test stdout — run `target/release/deps/op_microbench-*` directly
//! for the numbers).
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_vulkan::{Recorder, VulkanBackend};
use std::time::Instant;

/// Qwen3.8-Flash decode shape, recovered from the op trace's workgroup counts:
/// `qwen_hc_mix grid=160` → `rows * n_embd = 10240` → 2 rows x 5120; `qwen_hc_inject grid=640`
/// → `rows * hc * n_embd = 40960` → hc = 4. `silu_mul grid=640` is `n = 40960`.
const N_EMBD: u32 = 5120;
const HC: u32 = 4;
const ROWS: u32 = 2;

struct Case {
    name: String,
    grid: u32,
    reps: usize,
    bufs: Vec<Box<dyn Buffer>>,
    emit: fn(&Recorder, &[Box<dyn Buffer>]),
}

/// Deterministic non-zero f32 payload (all-zeros can let a driver skip work, and NaNs would make a
/// future correctness cross-check vacuous).
fn f32_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n * 4);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let v = ((s >> 40) as f32 / 16_777_216.0) - 0.5;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn alloc_floats(be: &VulkanBackend, floats: usize, seed: u64) -> Box<dyn Buffer> {
    let b = be
        .alloc(floats * 4, BufferUsage::Activations)
        .expect("alloc activation buffer");
    be.upload(b.as_ref(), &f32_bytes(floats, seed)).expect("upload");
    b
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn print_vram(be: &VulkanBackend, label: &str) {
    let v = be.vram();
    let used = v.total.saturating_sub(v.available);
    eprintln!(
        "[microbench] {label}: used {:.2} GiB / {:.2} GiB, live={}",
        used as f64 / (1u64 << 30) as f64,
        v.total as f64 / (1u64 << 30) as f64,
        v.live,
    );
}

/// Fill the resident-BDA arena with the same kind of allocation the model uses. The buffers are
/// intentionally leaked until the test returns and ARE NOT freed between cases, so every measured
/// dispatch runs with the device's live VRAM occupancy near the production run. `alloc_uninit`
/// skips the load-time zero fill; VRAM is physically committed at allocation time on this dGPU, so
/// the occupancy is established even though the benchmark never reads these pages.
fn install_pressure(be: &VulkanBackend, gib: usize) -> Vec<Box<dyn Buffer>> {
    let mut keepalive = Vec::new();
    if gib == 0 {
        return keepalive;
    }
    let chunk_mib = env_usize("OPMB_PRESSURE_CHUNK_MIB", 256).max(1);
    let chunk = chunk_mib * 1024 * 1024;
    let want = gib.saturating_mul(1024 * 1024 * 1024);
    let mut done = 0usize;
    while done < want {
        let bytes = chunk.min(want - done);
        match be.alloc_uninit(bytes, BufferUsage::Weights) {
            Ok(buf) => {
                keepalive.push(buf);
                done += bytes;
            }
            Err(e) => {
                eprintln!(
                    "[microbench] pressure stopped at {:.2} GiB: {e}",
                    done as f64 / (1u64 << 30) as f64
                );
                break;
            }
        }
    }
    eprintln!(
        "[microbench] installed {:.2} GiB pressure in {} x {} MiB BDA blocks",
        done as f64 / (1u64 << 30) as f64,
        keepalive.len(),
        chunk_mib,
    );
    keepalive
}

/// A non-capturing decoder per element count — `silu_mul` takes `n` as a plain argument, and a fn
/// pointer cannot carry it, so the sizes the probe sweeps are reified here.
fn silu_emit(n: usize) -> fn(&Recorder, &[Box<dyn Buffer>]) {
    match n {
        64 => |rec, b| rec.silu_mul(b[0].as_ref(), b[1].as_ref(), b[2].as_ref(), 64, None),
        640 => |rec, b| rec.silu_mul(b[0].as_ref(), b[1].as_ref(), b[2].as_ref(), 640, None),
        40_960 => |rec, b| rec.silu_mul(b[0].as_ref(), b[1].as_ref(), b[2].as_ref(), 40_960, None),
        _ => |rec, b| {
            rec.silu_mul(b[0].as_ref(), b[1].as_ref(), b[2].as_ref(), 1_310_720, None)
        },
    }
}

fn cases(be: &VulkanBackend) -> Vec<Case> {
    let mut out = Vec::new();

    // --- silu_mul: y[i] = silu(gate[i]) * up[i] ---
    for (label, n, reps) in [
        ("silu_mul n=64     (grid 1, dispatch floor)", 64usize, 3000usize),
        ("silu_mul n=640    (grid 10)", 640, 3000),
        ("silu_mul n=40960  (grid 640, trace scale)", 40_960, 1000),
        ("silu_mul n=1310720(grid 20480, bw-bound)", 1_310_720, 200),
    ] {
        let gate = alloc_floats(be, n, 0x5111);
        let up = alloc_floats(be, n, 0x7222);
        let y = be.alloc(n * 4, BufferUsage::Activations).unwrap();
        out.push(Case {
            name: label.to_string(),
            grid: n.div_ceil(64) as u32,
            reps,
            bufs: vec![gate, up, y],
            emit: silu_emit(n),
        });
    }

    // --- qwen_hc_mix: dst[i] = sum_h x[(r*hc+h)*n_embd+d] * sigmoid(gate[...]) / hc ---
    {
        let total = ROWS * N_EMBD;
        let x = alloc_floats(be, (total * HC) as usize, 0xA001);
        let gate = alloc_floats(be, (total * HC) as usize, 0xA002);
        let dst = be.alloc(total as usize * 4, BufferUsage::Activations).unwrap();
        out.push(Case {
            name: format!("qwen_hc_mix rows={ROWS} n_embd={N_EMBD} (grid {})", total.div_ceil(64)),
            grid: total.div_ceil(64),
            reps: 2000,
            bufs: vec![x, gate, dst],
            emit: move |rec, b| {
                rec.qwen_hc_mix(b[0].as_ref(), b[1].as_ref(), b[2].as_ref(), ROWS, HC, N_EMBD)
            },
        });
    }

    // --- qwen_hc_inject: dst[i] = residual[i] + block[r*n_embd+d] * gate_scale ---
    {
        let total = ROWS * HC * N_EMBD;
        let residual = alloc_floats(be, total as usize, 0xB001);
        let block = alloc_floats(be, (ROWS * N_EMBD) as usize, 0xB002);
        let gate = alloc_floats(be, (ROWS * HC) as usize, 0xB003);
        let dst = be.alloc(total as usize * 4, BufferUsage::Activations).unwrap();
        out.push(Case {
            name: format!("qwen_hc_inject rows={ROWS} hc={HC} n_embd={N_EMBD} (grid {})", total.div_ceil(64)),
            grid: total.div_ceil(64),
            reps: 2000,
            bufs: vec![residual, block, gate, dst],
            emit: move |rec, b| {
                rec.qwen_hc_inject(b[0].as_ref(), b[1].as_ref(), b[2].as_ref(), b[3].as_ref(), ROWS, HC, N_EMBD)
            },
        });
    }

    // --- qwen_hc_norm: one subgroup (32 lanes) per (row, hc) stream ---
    {
        const NORM_ROWS: u32 = 4;
        let x = alloc_floats(be, (NORM_ROWS * HC * N_EMBD) as usize, 0xC001);
        let norm = alloc_floats(be, N_EMBD as usize, 0xC002);
        let dst = be
            .alloc((NORM_ROWS * HC * N_EMBD) as usize * 4, BufferUsage::Activations)
            .unwrap();
        out.push(Case {
            name: format!("qwen_hc_norm rows={NORM_ROWS} hc={HC} (grid {})", NORM_ROWS * HC),
            grid: NORM_ROWS * HC,
            reps: 2000,
            bufs: vec![x, norm, dst],
            emit: |rec, b| {
                rec.qwen_hc_norm(
                    b[0].as_ref(), b[1].as_ref(), b[2].as_ref(), NORM_ROWS, HC, N_EMBD, 1e-6,
                )
            },
        });
    }

    // --- qwen_hc_inject_norm: the fused pair the trace flagged as 202 ms total ---
    {
        let total = ROWS * HC * N_EMBD;
        let residual = alloc_floats(be, total as usize, 0xD001);
        let block = alloc_floats(be, (ROWS * N_EMBD) as usize, 0xD002);
        let gate = alloc_floats(be, (ROWS * HC) as usize, 0xD003);
        let norm = alloc_floats(be, N_EMBD as usize, 0xD004);
        let r_dst = be.alloc(total as usize * 4, BufferUsage::Activations).unwrap();
        let n_dst = be.alloc(total as usize * 4, BufferUsage::Activations).unwrap();
        out.push(Case {
            name: format!("qwen_hc_inject_norm rows={ROWS} hc={HC} (grid {})", ROWS * HC),
            grid: ROWS * HC,
            reps: 2000,
            bufs: vec![residual, block, gate, norm, r_dst, n_dst],
            emit: |rec, b| {
                rec.qwen_hc_inject_norm(
                    b[0].as_ref(), b[1].as_ref(), b[2].as_ref(), b[3].as_ref(), b[4].as_ref(),
                    b[5].as_ref(), ROWS, HC, N_EMBD, 1e-6,
                )
            },
        });
    }

    out
}

/// `(batched us, spread us, engine-shape us)` — see `bench_case`.
fn bench_case(be: &VulkanBackend, c: &Case, ops_per_submit: usize, submits: usize) -> (f64, f64, f64) {
    // Warm-up (pipeline compile + clock spin-up) — discarded.
    for _ in 0..2 {
        let rec = be.recorder().unwrap();
        (c.emit)(&rec, &c.bufs);
        rec.finish().unwrap();
    }

    // batched: REPS dispatches, one submit+wait.
    let rec = be.recorder().unwrap();
    (c.emit)(&rec, &c.bufs);
    rec.finish().unwrap();
    let t0 = Instant::now();
    let rec = be.recorder().unwrap();
    for _ in 0..c.reps {
        (c.emit)(&rec, &c.bufs);
    }
    rec.finish().unwrap();
    let batched = t0.elapsed().as_secs_f64() * 1e6 / c.reps as f64;

    // spread: one dispatch per submit+wait.
    let t0 = Instant::now();
    for _ in 0..c.reps.min(300) {
        let rec = be.recorder().unwrap();
        (c.emit)(&rec, &c.bufs);
        rec.finish().unwrap();
    }
    let n = c.reps.min(300) as f64;
    let spread = t0.elapsed().as_secs_f64() * 1e6 / n;

    // engine-shape: `ops_per_submit` dispatches per submit+wait, `submits` times. The instrumented
    // decode records 64-73 ops per segment and submits each segment separately, so this is the
    // regime the op trace actually measures. If dev0 is 40x slower HERE but not above, the cost is
    // per-SUBMIT (queue wake / contest with the display engine), not per-dispatch.
    for _ in 0..2 {
        let rec = be.recorder().unwrap();
        for _ in 0..ops_per_submit {
            (c.emit)(&rec, &c.bufs);
        }
        rec.finish().unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..submits {
        let rec = be.recorder().unwrap();
        for _ in 0..ops_per_submit {
            (c.emit)(&rec, &c.bufs);
        }
        rec.finish().unwrap();
    }
    let engine = t0.elapsed().as_secs_f64() * 1e6 / (submits * ops_per_submit) as f64;

    (batched, spread, engine)
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench); run alone, nothing else on the GPU"]
fn op_microbench() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    if let Ok(devs) = VulkanBackend::enumerate_devices_from_env() {
        let sel = devs.iter().find(|d| d.is_default_pick);
        if let Some(d) = sel {
            eprintln!(
                "[microbench] bound Vulkan{} = {} ({}), {:.1} GiB device-local",
                d.index,
                d.name,
                d.device_type,
                d.vram_bytes as f64 / (1u64 << 30) as f64
            );
        }
    }

    let pressure_gib = env_usize("OPMB_PRESSURE_GIB", 0);
    print_vram(&be, "before pressure");
    let _pressure = install_pressure(&be, pressure_gib);
    print_vram(&be, "after pressure");

    let cases = cases(&be);
    let ops_per_submit = env_usize("OPMB_OPS_PER_SUBMIT", 64);
    let submits = env_usize("OPMB_SUBMITS", 120);
    let filter = std::env::var("OPMB_FILTER").unwrap_or_default();
    eprintln!(
        "\n{:<48} {:>7} {:>12} {:>12} {:>14} {:>12}",
        "case", "grid", "batched us", "spread us", "engine-shape", "spread-batch"
    );
    for c in &cases {
        if !filter.is_empty() && !c.name.contains(&filter) {
            continue;
        }
        let (batched, spread, engine) = bench_case(&be, c, ops_per_submit, submits);
        eprintln!(
            "{:<48} {:>7} {:>12.2} {:>12.2} {:>14.2} {:>12.2}",
            c.name,
            c.grid,
            batched,
            spread,
            engine,
            spread - batched
        );
    }
}
