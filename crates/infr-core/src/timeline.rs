//! Opt-in wall-clock traces. Queries retain their original context until the normal fence retires.
use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::Serialize;

static ENABLED: AtomicBool = AtomicBool::new(false);
static FLIGHT_ENABLED: OnceLock<bool> = OnceLock::new();
static NEXT_STEP: AtomicU64 = AtomicU64::new(0);
static NEXT_SUBMIT: AtomicU64 = AtomicU64::new(1);
static STATE: OnceLock<Mutex<State>> = OnceLock::new();
const MAX_EVENTS: usize = 2_000_000;

thread_local! {
    static CONTEXT: Cell<Option<Context>> = const { Cell::new(None) };
    static EVENTS: RefCell<Vec<HostEvent>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Context {
    pub step: u64,
    pub lanes: usize,
    pub layer: i32,
    /// Pipeline device the traced work executes on, or -1 when the backend does not
    /// expose a per-segment device identity.
    pub device: i32,
    /// Decode phase the traced work belongs to ("draft"/"verify"/…), or "" outside one. The
    /// flight recorder uses it to bill a submit to the phase that issued it; the timestamp
    /// timeline ignores it.
    pub phase: &'static str,
}

#[derive(Serialize)]
struct HostEvent {
    name: &'static str,
    context: Context,
    start_ns: u64,
    end_ns: u64,
    submit: u64,
}

#[derive(Serialize)]
struct Step {
    id: u64,
    lanes: usize,
    positions: Vec<usize>,
    start_ns: u64,
    end_ns: u64,
}

#[derive(Clone, Copy, Serialize)]
pub struct ClockSample {
    pub host_ns: u64,
    pub device_tick: u64,
    pub period_ns: f32,
    pub deviation_ns: u64,
}

#[derive(Serialize)]
pub struct DeviceEvent {
    pub name: &'static str,
    pub queue: &'static str,
    pub context: Context,
    pub submit: u64,
    pub host_submit_ns: u64,
    pub start_tick: u64,
    pub end_tick: u64,
    pub valid_bits: u32,
    pub period_ns: f32,
    pub transfer_wait: u64,
    pub transfer_signal: u64,
    pub bytes: u64,
    pub dispatches: usize,
}

struct State {
    path: PathBuf,
    origin_ns: u64,
    skip: usize,
    count: usize,
    stride: usize,
    windows: usize,
    lanes: usize,
    events: Vec<HostEvent>,
    steps: Vec<Step>,
    device: Vec<DeviceEvent>,
    clocks: Vec<ClockSample>,
    dropped: usize,
}

#[cfg(windows)]
pub fn clock_ns() -> u64 {
    use windows::Win32::System::Performance::QueryPerformanceCounter;
    let mut tick = 0;
    unsafe { QueryPerformanceCounter(&mut tick).expect("QPC") };
    calibrated_host_ns(tick as u64)
}

#[cfg(windows)]
pub fn calibrated_host_ns(tick: u64) -> u64 {
    use windows::Win32::System::Performance::QueryPerformanceFrequency;
    static FREQUENCY: OnceLock<u64> = OnceLock::new();
    let frequency = *FREQUENCY.get_or_init(|| {
        let mut value = 0;
        unsafe { QueryPerformanceFrequency(&mut value).expect("QPC frequency") };
        value as u64
    });
    ((tick as u128 * 1_000_000_000) / frequency as u128) as u64
}

#[cfg(not(windows))]
pub fn calibrated_host_ns(tick: u64) -> u64 {
    tick
}

#[cfg(not(windows))]
pub fn clock_ns() -> u64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
    time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64
}

pub fn configure(cfg: &crate::config::ProfCfg) -> Result<(), String> {
    let Some(path) = &cfg.timeline_path else {
        return Ok(());
    };
    if cfg.ops || cfg.stages {
        return Err("timeline requires prof.ops=false and prof.stages=false to preserve asynchronous submits".into());
    }
    if cfg.timeline_steps == 0
        || cfg.timeline_stride < cfg.timeline_steps
        || cfg.timeline_windows == 0
    {
        return Err("timeline requires positive steps/windows and stride >= steps".into());
    }
    STATE
        .set(Mutex::new(State {
            path: path.clone(),
            origin_ns: clock_ns(),
            skip: cfg.timeline_skip_steps,
            count: cfg.timeline_steps,
            stride: cfg.timeline_stride,
            windows: cfg.timeline_windows,
            lanes: cfg.timeline_lanes,
            events: Vec::with_capacity(100_000),
            steps: Vec::new(),
            device: Vec::with_capacity(20_000),
            clocks: Vec::new(),
            dropped: 0,
        }))
        .map_err(|_| "timeline already configured".to_string())?;
    ENABLED.store(true, Ordering::Release);
    Ok(())
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

#[inline]
pub fn context() -> Option<Context> {
    if !enabled() {
        return None;
    }
    CONTEXT.with(Cell::get)
}

fn selected(index: usize, skip: usize, count: usize, stride: usize, windows: usize) -> bool {
    index
        .checked_sub(skip)
        .is_some_and(|offset| offset / stride < windows && offset % stride < count)
}

pub struct StepGuard {
    previous: Option<Context>,
    step: Step,
    record_timeline: bool,
}

pub fn begin_step(positions: &[usize]) -> Option<StepGuard> {
    let trace = enabled();
    let flight = flight_enabled();
    if (!trace && !flight) || crate::prof::suppressed() {
        return None;
    }
    let mut id = None;
    let mut record_timeline = false;
    if trace {
        if let Some(state) = STATE.get() {
            let state = state.lock().unwrap();
            if state.lanes == 0 || state.lanes == positions.len() {
                let candidate = NEXT_STEP.fetch_add(1, Ordering::Relaxed);
                record_timeline = selected(
                    candidate as usize,
                    state.skip,
                    state.count,
                    state.stride,
                    state.windows,
                );
                id = Some(candidate);
            }
        }
    }
    let id = match id {
        Some(id) => id,
        None if flight => NEXT_STEP.fetch_add(1, Ordering::Relaxed),
        None => return None,
    };
    if !record_timeline && !flight {
        return None;
    }
    if record_timeline {
        EVENTS.with(|events| {
            let mut events = events.borrow_mut();
            if events.capacity() < 16_384 {
                events.reserve(16_384);
            }
        });
    }
    let ctx = Context {
        step: id,
        lanes: positions.len(),
        layer: -1,
        device: -1,
        phase: "",
    };
    let previous = CONTEXT.with(|current| current.replace(Some(ctx)));
    Some(StepGuard {
        previous,
        step: Step {
            id,
            lanes: positions.len(),
            positions: positions.to_vec(),
            start_ns: clock_ns(),
            end_ns: 0,
        },
        record_timeline,
    })
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        CONTEXT.with(|current| current.set(self.previous));
        if !self.record_timeline {
            return;
        }
        self.step.end_ns = clock_ns();
        let mut state = STATE.get().unwrap().lock().unwrap();
        EVENTS.with(|events| {
            let mut events = events.borrow_mut();
            let room = MAX_EVENTS.saturating_sub(state.events.len());
            if events.len() > room {
                state.dropped += events.len() - room;
                events.truncate(room);
            }
            state.events.extend(events.drain(..));
        });
        state.steps.push(Step {
            id: self.step.id,
            lanes: self.step.lanes,
            positions: std::mem::take(&mut self.step.positions),
            start_ns: self.step.start_ns,
            end_ns: self.step.end_ns,
        });
    }
}

pub struct Span {
    name: &'static str,
    ctx: Context,
    start: u64,
    submit: u64,
}

#[inline]
pub fn span(name: &'static str) -> Option<Span> {
    span_with_submit(name, 0)
}

pub fn span_with_submit(name: &'static str, submit: u64) -> Option<Span> {
    context().map(|ctx| Span {
        name,
        ctx,
        start: clock_ns(),
        submit,
    })
}

impl Drop for Span {
    fn drop(&mut self) {
        let end_ns = clock_ns();
        EVENTS.with(|events| {
            events.borrow_mut().push(HostEvent {
                name: self.name,
                context: self.ctx,
                start_ns: self.start,
                end_ns,
                submit: self.submit,
            })
        });
    }
}

pub struct LayerGuard(Option<Context>);
pub fn layer(layer: usize) -> LayerGuard {
    let previous = attribution_context();
    if let Some(mut ctx) = previous {
        ctx.layer = layer as i32;
        CONTEXT.with(|current| current.set(Some(ctx)));
    }
    LayerGuard(previous)
}
impl Drop for LayerGuard {
    fn drop(&mut self) {
        if enabled() || flight_enabled() {
            CONTEXT.with(|current| current.set(self.0));
        }
    }
}

/// Attribute spans and device submissions to pipeline segment `device` until the guard drops.
/// A no-op outside an active [`begin_step`] context (e.g. background threads).
pub struct DeviceGuard(Option<Context>);
/// Attribute submissions to a named decode phase until the guard drops. A no-op when neither the
/// timestamp timeline nor the flight recorder is active, so production pays one relaxed load.
pub struct PhaseGuard(Option<Context>);
pub fn phase(name: &'static str) -> PhaseGuard {
    let previous = attribution_context();
    if let Some(mut ctx) = previous {
        ctx.phase = name;
        CONTEXT.with(|current| current.set(Some(ctx)));
    }
    PhaseGuard(previous)
}
impl Drop for PhaseGuard {
    fn drop(&mut self) {
        if enabled() || flight_enabled() {
            CONTEXT.with(|current| current.set(self.0));
        }
    }
}

pub fn device(device: usize) -> DeviceGuard {
    let previous = attribution_context();
    if let Some(mut ctx) = previous {
        ctx.device = device as i32;
        CONTEXT.with(|current| current.set(Some(ctx)));
    }
    DeviceGuard(previous)
}
impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if enabled() || flight_enabled() {
            CONTEXT.with(|current| current.set(self.0));
        }
    }
}

/// The calling thread's current attribution context, if a step is active — the counterpart of
/// [`adopt`], for work about to be handed to a worker thread.
#[inline]
pub fn current_context() -> Option<Context> {
    CONTEXT.with(Cell::get)
}

/// Adopt `ctx` as this thread's attribution context until the guard drops. The context is
/// thread-local, so a spawned worker sees none of the step/phase bookkeeping its parent set up;
/// the pipeline's device-1 half uses this to bill its submits to the same decode step and phase as
/// the device-0 half that spawned it.
pub struct AdoptGuard(Option<Context>);
pub fn adopt(ctx: Option<Context>) -> AdoptGuard {
    let previous = CONTEXT.with(Cell::get);
    if enabled() || flight_enabled() {
        CONTEXT.with(|current| current.set(ctx));
    }
    AdoptGuard(previous)
}
impl Drop for AdoptGuard {
    fn drop(&mut self) {
        if enabled() || flight_enabled() {
            CONTEXT.with(|current| current.set(self.0));
        }
    }
}

/// Whether a lightweight trace requested decode-step attribution. The submit flight recorder and
/// the per-dispatch op trace both need the same thread-local context, even when the heavyweight
/// device timeline is off.
#[inline]
pub fn flight_enabled() -> bool {
    *FLIGHT_ENABLED.get_or_init(|| {
        [
            std::env::var_os("INFR_SUBMIT_TRACE"),
            std::env::var_os("INFR_PROF_OP_TRACE"),
        ]
        .iter()
        .flatten()
        .any(|value| !value.is_empty())
    })
}

/// Current decode context for the submit flight recorder. Unlike [`context`], this does not
/// require the device-timestamp timeline to be active.
#[inline]
pub fn flight_context() -> Option<Context> {
    if !flight_enabled() {
        return None;
    }
    CONTEXT.with(Cell::get)
}

#[inline]
fn attribution_context() -> Option<Context> {
    if enabled() {
        context()
    } else {
        flight_context()
    }
}

pub fn next_submit() -> u64 {
    NEXT_SUBMIT.fetch_add(1, Ordering::Relaxed)
}
pub fn record_device(event: DeviceEvent) {
    if let Some(state) = STATE.get() {
        state.lock().unwrap().device.push(event);
    }
}

/// A background CPU producer retains the submitting decode step's context rather than using its
/// own (empty) thread-local graph context. The interval is wall time, including worker handoffs.
pub fn record_cpu_miss(name: &'static str, context: Context, start_ns: u64, end_ns: u64) {
    if let Some(state) = STATE.get() {
        let mut state = state.lock().unwrap();
        if state.events.len() < MAX_EVENTS {
            state.events.push(HostEvent {
                name,
                context,
                start_ns,
                end_ns,
                submit: 0,
            });
        } else {
            state.dropped += 1;
        }
    }
}
pub fn record_clock(sample: ClockSample) {
    if let Some(state) = STATE.get() {
        state.lock().unwrap().clocks.push(sample);
    }
}

pub fn flush() -> std::io::Result<()> {
    let Some(state) = STATE.get() else {
        return Ok(());
    };
    let state = state.lock().unwrap();
    let mut trace = Vec::with_capacity(state.events.len() + state.device.len() + state.steps.len());
    let origin = state.origin_ns;
    for step in &state.steps {
        trace.push(
            serde_json::json!({"name":"decode_step", "ph":"X", "pid":1, "tid":1,
            "ts":step.start_ns.saturating_sub(origin) as f64 / 1000.0,
            "dur":step.end_ns.saturating_sub(step.start_ns) as f64 / 1000.0,
            "args":{"step":step.id,"lanes":step.lanes,"positions":step.positions}}),
        );
    }
    for event in &state.events {
        trace.push(serde_json::json!({"name":event.name,"ph":"X","pid":1,"tid":if event.name == "recorder_lifetime" {2} else if matches!(event.name, "cpu_miss_ffn_one" | "cpu_miss_ffn_two" | "cpu_miss_ffn_three" | "cpu_miss_gu" | "cpu_miss_down") {3} else {1},
            "ts":event.start_ns.saturating_sub(origin) as f64 / 1000.0,
            "dur":event.end_ns.saturating_sub(event.start_ns) as f64 / 1000.0,
            "args":{"step":event.context.step,"layer":event.context.layer,"device":event.context.device,"submit":event.submit}}));
    }
    for event in &state.device {
        if let Some(sample) = state
            .clocks
            .iter()
            .min_by_key(|sample| sample.host_ns.abs_diff(event.host_submit_ns))
        {
            let start = map_tick(event.start_tick, event.valid_bits, *sample);
            let duration = tick_duration(
                event.start_tick,
                event.end_tick,
                event.valid_bits,
                event.period_ns,
            );
            trace.push(serde_json::json!({"name":event.name,"ph":"X","pid":2,
                "tid":if event.queue != "main" {2} else if event.name == "queue_envelope" {1} else {3},
                "ts":start.saturating_sub(origin) as f64 / 1000.0,"dur":duration as f64 / 1000.0,
                "args":{"step":event.context.step,"layer":event.context.layer,"device":event.context.device,"submit":event.submit,
                    "wait":event.transfer_wait,"signal":event.transfer_signal,"bytes":event.bytes,
                    "dispatches":event.dispatches,"calibration_deviation_ns":sample.deviation_ns}}));
        }
    }
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":1,"tid":1,"args":{"name":"CPU inference"}}));
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":1,"tid":2,"args":{"name":"Recorder lifetimes"}}));
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":1,"tid":3,"args":{"name":"CPU miss worker phase envelopes"}}));
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":2,"tid":1,"args":{"name":"GPU main queue envelopes"}}));
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":2,"tid":2,"args":{"name":"GPU dedicated DMA"}}));
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":2,"tid":3,"args":{"name":"GPU completion milestones (not isolated kernel times)"}}));
    let document = serde_json::json!({"traceEvents":trace,"displayTimeUnit":"ms",
        "origin_ns":origin,"steps":state.steps,"host_events":state.events,"device_events":state.device,
        "clock_samples":state.clocks,"dropped_events":state.dropped,
        "gpu_interval_semantics":"TOP/BOTTOM queue envelope; includes dependencies and stalls, not ALU busy time"});
    if let Some(parent) = state.path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    serde_json::to_writer(
        std::io::BufWriter::new(std::fs::File::create(&state.path)?),
        &document,
    )?;
    tracing::info!(path=%state.path.display(), steps=state.steps.len(), device_events=state.device.len(), dropped=state.dropped, "Decode timeline saved");
    Ok(())
}

pub fn tick_duration(start: u64, end: u64, bits: u32, period: f32) -> u64 {
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    ((end.wrapping_sub(start) & mask) as f64 * period as f64) as u64
}

fn map_tick(tick: u64, bits: u32, sample: ClockSample) -> u64 {
    let delta = if bits >= 64 {
        tick.wrapping_sub(sample.device_tick) as i64 as i128
    } else {
        let modulus = 1i128 << bits;
        let half = modulus / 2;
        (tick as i128 - sample.device_tick as i128 + half).rem_euclid(modulus) - half
    };
    (sample.host_ns as i128 + (delta as f64 * sample.period_ns as f64) as i128).max(0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn windows_are_bounded_and_skip_warmup() {
        let ids = (0..100)
            .filter(|&i| selected(i, 7, 3, 10, 2))
            .collect::<Vec<_>>();
        assert_eq!(ids, [7, 8, 9, 17, 18, 19]);
    }
    #[test]
    fn timestamp_wrap_and_both_sides_of_calibration() {
        let sample = ClockSample {
            host_ns: 10000,
            device_tick: 250,
            period_ns: 2.0,
            deviation_ns: 10,
        };
        assert_eq!(map_tick(5, 8, sample), 10022);
        assert_eq!(map_tick(240, 8, sample), 9980);
        assert_eq!(tick_duration(250, 5, 8, 2.0), 22);
        assert_eq!(
            map_tick(
                99,
                64,
                ClockSample {
                    device_tick: 100,
                    ..sample
                }
            ),
            9998
        );
    }
    #[test]
    fn blocking_profilers_and_invalid_windows_are_rejected() {
        let mut cfg = crate::config::ProfCfg {
            timeline_path: Some("unused.json".into()),
            ops: true,
            ..Default::default()
        };
        assert!(configure(&cfg).unwrap_err().contains("asynchronous"));
        cfg.ops = false;
        cfg.timeline_stride = 1;
        assert!(configure(&cfg).is_err());
    }
}
