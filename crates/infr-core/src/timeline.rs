//! Opt-in wall-clock traces. Queries retain their original context until the normal fence retires.
use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::Serialize;

static ENABLED: AtomicBool = AtomicBool::new(false);
static DEVICE_TIMESTAMPS: AtomicBool = AtomicBool::new(false);
static HOST_SPANS: AtomicBool = AtomicBool::new(false);
static NEXT_STEP: AtomicU64 = AtomicU64::new(0);
static NEXT_SUBMIT: AtomicU64 = AtomicU64::new(1);
static STATE: OnceLock<Mutex<State>> = OnceLock::new();
static CAPTURE: OnceLock<CaptureFilter> = OnceLock::new();
const MAX_EVENTS: usize = 2_000_000;

#[derive(Default, Serialize)]
struct CaptureFilter {
    device: Option<Vec<String>>,
    layer: Option<usize>,
    host: Option<Vec<String>>,
}

fn label_patterns(value: Option<&str>) -> Result<Option<Vec<String>>, String> {
    value.map(|value| {
        let patterns: Vec<_> = value.split(',').map(str::trim).map(str::to_owned).collect();
        if patterns.iter().any(String::is_empty) {
            return Err("timeline label filters require non-empty comma-separated substrings".into());
        }
        Ok(patterns)
    }).transpose()
}

fn matches_label(patterns: Option<&[String]>, name: &str) -> bool {
    patterns.is_none_or(|patterns| patterns.iter().any(|pattern| name.contains(pattern)))
}

impl CaptureFilter {
    fn device_matches(&self, name: &str, layer: i32) -> bool {
        self.layer.is_none_or(|wanted| layer >= 0 && wanted == layer as usize)
            && matches_label(self.device.as_deref(), name)
    }
}

pub fn device_label_selected(name: &str, layer: i32) -> bool {
    CAPTURE.get().is_none_or(|filter| filter.device_matches(name, layer))
}

pub fn device_labels_selective() -> bool {
    CAPTURE.get().is_some_and(|filter| filter.device.is_some() || filter.layer.is_some())
}

pub fn host_label_selected(name: &str) -> bool {
    CAPTURE.get().is_none_or(|filter| matches_label(filter.host.as_deref(), name))
}

thread_local! {
    static CONTEXT: Cell<Option<Context>> = const { Cell::new(None) };
    static EVENTS: RefCell<Vec<HostEvent>> = const { RefCell::new(Vec::new()) };
    static DEVICE_PHASE: Cell<Option<&'static str>> = const { Cell::new(None) };
    static LOCAL_HOST_SPANS: Cell<bool> = const { Cell::new(true) };
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Context {
    pub step: u64,
    pub lanes: usize,
    pub layer: i32,
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

#[derive(Serialize)]
pub struct ResidentNotice {
    pub context: Context,
    pub ordinal: usize,
    pub status: u32,
    pub hit_masks: [u32; crate::backend::MAX_DECODE_ROWS],
    pub count: usize,
    pub routed_ids: Vec<u32>,
    /// GPU-only retirement samples carry status, not a reconstructed route or hit mask.
    pub route_payload_present: bool,
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
    resident_notices: Vec<ResidentNotice>,
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
    if cfg.timeline_cpu_only && cfg.timeline_gpu_only {
        return Err("timeline_cpu_only and timeline_gpu_only are mutually exclusive".into());
    }
    if cfg.timeline_steps == 0
        || cfg.timeline_stride < cfg.timeline_steps
        || cfg.timeline_windows == 0
    {
        return Err("timeline requires positive steps/windows and stride >= steps".into());
    }
    let capture = CaptureFilter {
        device: label_patterns(cfg.timeline_device_select.as_deref())?,
        layer: cfg.timeline_device_layer,
        host: label_patterns(cfg.timeline_host_select.as_deref())?,
    };
    if capture.layer.is_some_and(|layer| layer > i32::MAX as usize) {
        return Err("timeline device layer exceeds the trace context range".into());
    }
    if STATE.get().is_some() {
        return Err("timeline already configured".into());
    }
    CAPTURE.set(capture).map_err(|_| "timeline filters already configured".to_string())?;
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
            resident_notices: Vec::new(),
            dropped: 0,
        }))
        .map_err(|_| "timeline already configured".to_string())?;
    DEVICE_TIMESTAMPS.store(wants_device_timestamps(cfg), Ordering::Relaxed);
    HOST_SPANS.store(!cfg.timeline_gpu_only, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Release);
    Ok(())
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

fn wants_device_timestamps(cfg: &crate::config::ProfCfg) -> bool {
    cfg.timeline_path.is_some() && !cfg.timeline_cpu_only
}

#[inline]
pub fn device_enabled() -> bool {
    ENABLED.load(Ordering::Acquire) && DEVICE_TIMESTAMPS.load(Ordering::Relaxed)
}

#[inline]
pub fn host_enabled() -> bool {
    enabled() && HOST_SPANS.load(Ordering::Relaxed)
}

/// Propagate GPU context without misclassifying uploader spans as main-thread CPU work.
pub struct ContextGuard(Option<Context>, bool);
pub fn scoped_context(context: Option<Context>) -> ContextGuard {
    ContextGuard(
        CONTEXT.with(|current| current.replace(context)),
        LOCAL_HOST_SPANS.with(|current| current.replace(false)),
    )
}
impl Drop for ContextGuard {
    fn drop(&mut self) {
        CONTEXT.with(|current| current.set(self.0));
        LOCAL_HOST_SPANS.with(|current| current.set(self.1));
    }
}

pub struct DevicePhaseGuard(Option<&'static str>);
pub fn device_phase_scope(phase: &'static str) -> DevicePhaseGuard {
    DevicePhaseGuard(DEVICE_PHASE.with(|current| current.replace(Some(phase))))
}
pub fn device_phase() -> Option<&'static str> {
    DEVICE_PHASE.with(Cell::get)
}
impl Drop for DevicePhaseGuard {
    fn drop(&mut self) {
        DEVICE_PHASE.with(|current| current.set(self.0));
    }
}

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
}

pub fn begin_step(positions: &[usize]) -> Option<StepGuard> {
    if !enabled() || crate::prof::suppressed() {
        return None;
    }
    let state = STATE.get()?.lock().unwrap();
    if state.lanes != 0 && state.lanes != positions.len() {
        return None;
    }
    let id = NEXT_STEP.fetch_add(1, Ordering::Relaxed);
    if !selected(
        id as usize,
        state.skip,
        state.count,
        state.stride,
        state.windows,
    ) {
        return None;
    }
    drop(state);
    EVENTS.with(|events| {
        let mut events = events.borrow_mut();
        if events.capacity() < 16_384 {
            events.reserve(16_384);
        }
    });
    let ctx = Context {
        step: id,
        lanes: positions.len(),
        layer: -1,
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
    })
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        self.step.end_ns = clock_ns();
        CONTEXT.with(|current| current.set(self.previous));
        let flush_start = host_enabled().then(clock_ns);
        let mut state = STATE.get().unwrap().lock().unwrap();
        let flush_locked = flush_start.map(|_| clock_ns());
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
        if let (Some(start), Some(locked)) = (flush_start, flush_locked) {
            // Flushing happens outside the recorded step. Account for the profiler itself
            // before interpreting the interval between steps as inference overhead.
            let end = clock_ns();
            let context = Context {
                step: self.step.id,
                lanes: self.step.lanes,
                layer: -1,
            };
            for (name, start_ns, end_ns) in [
                ("trace_step_flush_lock", start, locked),
                ("trace_step_flush_copy", locked, end),
            ] {
                if !host_label_selected(name) {
                    continue;
                }
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
    if !host_enabled() || !LOCAL_HOST_SPANS.with(Cell::get) || !host_label_selected(name) {
        return None;
    }
    context().map(|ctx| Span {
        name,
        ctx,
        start: clock_ns(),
        submit,
    })
}

fn background_context() -> Option<Context> {
    if LOCAL_HOST_SPANS.with(Cell::get) {
        None
    } else {
        CONTEXT.with(Cell::get)
    }
}

/// Explicit worker phases use the global recorder; normal uploader spans remain suppressed.
pub struct BackgroundSpan {
    name: &'static str,
    context: Context,
    start_ns: u64,
}

pub fn background_span(name: &'static str) -> Option<BackgroundSpan> {
    if !host_enabled() || !host_label_selected(name) {
        return None;
    }
    background_context().map(|context| BackgroundSpan {
        name,
        context,
        start_ns: clock_ns(),
    })
}

impl Drop for BackgroundSpan {
    fn drop(&mut self) {
        record_cpu_miss(self.name, self.context, self.start_ns, clock_ns());
    }
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
    let previous = context();
    if let Some(mut ctx) = previous {
        ctx.layer = layer as i32;
        CONTEXT.with(|current| current.set(Some(ctx)));
    }
    LayerGuard(previous)
}
impl Drop for LayerGuard {
    fn drop(&mut self) {
        if enabled() {
            CONTEXT.with(|current| current.set(self.0));
        }
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
    if !host_enabled() || !host_label_selected(name) {
        return;
    }
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
pub fn record_resident_notice(notice: ResidentNotice) {
    if device_enabled() {
        if let Some(state) = STATE.get() {
            state.lock().unwrap().resident_notices.push(notice);
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
        trace.push(serde_json::json!({"name":event.name,"ph":"X","pid":1,"tid":if event.name == "recorder_lifetime" {2} else if event.name.starts_with("resident_producer_") {4} else if matches!(event.name, "cpu_miss_ffn_one" | "cpu_miss_ffn_two" | "cpu_miss_ffn_three" | "cpu_miss_gu" | "cpu_miss_down") {3} else {1},
            "ts":event.start_ns.saturating_sub(origin) as f64 / 1000.0,
            "dur":event.end_ns.saturating_sub(event.start_ns) as f64 / 1000.0,
            "args":{"step":event.context.step,"layer":event.context.layer,"submit":event.submit}}));
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
                "args":{"step":event.context.step,"layer":event.context.layer,"submit":event.submit,
                    "wait":event.transfer_wait,"signal":event.transfer_signal,"bytes":event.bytes,
                    "dispatches":event.dispatches,"calibration_deviation_ns":sample.deviation_ns}}));
        }
    }
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":1,"tid":1,"args":{"name":"CPU inference"}}));
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":1,"tid":2,"args":{"name":"Recorder lifetimes"}}));
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":1,"tid":3,"args":{"name":"CPU miss worker phase envelopes"}}));
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":1,"tid":4,"args":{"name":"Resident notice producer phase envelopes (not GPU DMA)"}}));
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":2,"tid":1,"args":{"name":"GPU main queue envelopes"}}));
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":2,"tid":2,"args":{"name":"GPU dedicated DMA"}}));
    trace.push(serde_json::json!({"name":"thread_name","ph":"M","pid":2,"tid":3,"args":{"name":"GPU completion milestones (not isolated kernel times)"}}));
    let document = serde_json::json!({"traceEvents":trace,"displayTimeUnit":"ms",
        "origin_ns":origin,"steps":state.steps,"host_events":state.events,"device_events":state.device,
        "clock_samples":state.clocks,"dropped_events":state.dropped,
        "device_timestamps":DEVICE_TIMESTAMPS.load(Ordering::Relaxed),
        "host_spans":HOST_SPANS.load(Ordering::Relaxed),
        "capture_filters":CAPTURE.get(),
        "resident_notices":state.resident_notices,
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
    fn step_flush_events_are_outside_inference_and_respect_capture_mode() {
        const CHILD: &str = "MOE4ALL_TIMELINE_FLUSH_TEST_CHILD";
        let Ok(mode) = std::env::var(CHILD) else {
            // Each capture owns process-global OnceLocks. Keep this test isolated from
            // other profiler tests without resetting production globals or writing files.
            for mode in ["host", "gpu_only", "filtered"] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "timeline::tests::step_flush_events_are_outside_inference_and_respect_capture_mode",
                        "--test-threads=1",
                    ])
                    .env(CHILD, mode)
                    .output()
                    .unwrap();
                assert!(output.status.success(), "{mode}: {} {}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr));
            }
            return;
        };
        let cfg = crate::config::ProfCfg {
            timeline_path: Some("unused-no-flush.json".into()),
            timeline_steps: 1,
            timeline_skip_steps: 0,
            timeline_stride: 1,
            timeline_windows: 1,
            timeline_lanes: 2,
            timeline_gpu_only: mode == "gpu_only",
            timeline_host_select: (mode == "filtered").then(|| "no_matching_event".into()),
            ..Default::default()
        };
        configure(&cfg).unwrap();
        drop(begin_step(&[32, 64]).expect("selected step"));
        assert!(CONTEXT.with(Cell::get).is_none());
        let state = STATE.get().unwrap().lock().unwrap();
        assert_eq!(state.steps.len(), 1);
        if mode != "host" {
            assert!(state.events.is_empty());
            return;
        }
        assert_eq!(state.events.len(), 2);
        assert_eq!(state.events[0].name, "trace_step_flush_lock");
        assert_eq!(state.events[1].name, "trace_step_flush_copy");
        for event in &state.events {
            assert_eq!(event.context.step, state.steps[0].id);
            assert_eq!(event.context.lanes, 2);
            assert_eq!(event.context.layer, -1);
            assert!(event.start_ns >= state.steps[0].end_ns);
            assert!(event.end_ns >= event.start_ns);
        }
    }
    #[test]
    fn selective_capture_filters_are_explicit_and_do_not_accept_empty_terms() {
        assert!(label_patterns(None).unwrap().is_none());
        for input in ["", " ", ",owner", "owner,", "owner,,hit"] {
            assert!(label_patterns(Some(input)).is_err(), "{input}");
        }
        let filter = CaptureFilter {
            device: label_patterns(Some(" topk , FFN ")).unwrap(),
            layer: Some(3),
            host: label_patterns(Some("queue_submit,PLE_")).unwrap(),
        };
        assert!(filter.device_matches("resident_owned_topk_prepare", 3));
        assert!(filter.device_matches("resident_hit_FFN", 3));
        assert!(!filter.device_matches("resident_hit_FFN", -1));
        assert!(!filter.device_matches("resident_hit_FFN", 2));
        assert!(!filter.device_matches("native_q8", 3));
        assert!(matches_label(filter.host.as_deref(), "PLE_wait"));
        assert!(!matches_label(filter.host.as_deref(), "command_encode"));
        assert!(CaptureFilter::default().device_matches("any_kernel", -1));
    }
    #[test]
    fn background_context_restores_tls_without_cpu_spans() {
        assert!(LOCAL_HOST_SPANS.with(Cell::get));
        assert!(background_context().is_none());
        {
            let _scope = scoped_context(Some(Context {
                step: 91,
                lanes: 2,
                layer: 7,
            }));
            assert_eq!(CONTEXT.with(Cell::get).unwrap().step, 91);
            assert!(!LOCAL_HOST_SPANS.with(Cell::get));
            assert_eq!(background_context().unwrap().layer, 7);
            {
                let _nested = scoped_context(None);
                assert!(CONTEXT.with(Cell::get).is_none());
                assert!(background_context().is_none());
            }
            assert_eq!(CONTEXT.with(Cell::get).unwrap().layer, 7);
        }
        assert!(CONTEXT.with(Cell::get).is_none());
        assert!(LOCAL_HOST_SPANS.with(Cell::get));
        assert!(background_context().is_none());
    }
    #[test]
    fn device_phase_scopes_restore_nested_state() {
        assert_eq!(device_phase(), None);
        {
            let _hit = device_phase_scope("hit");
            assert_eq!(device_phase(), Some("hit"));
            {
                let _miss = device_phase_scope("miss");
                assert_eq!(device_phase(), Some("miss"));
            }
            assert_eq!(device_phase(), Some("hit"));
        }
        assert_eq!(device_phase(), None);
        let mut cfg = crate::config::ProfCfg {
            timeline_path: Some("unused.json".into()),
            timeline_gpu_only: true,
            ..Default::default()
        };
        assert!(wants_device_timestamps(&cfg));
        cfg.timeline_cpu_only = true;
        assert!(configure(&cfg).unwrap_err().contains("mutually exclusive"));
    }
    #[test]
    fn cpu_only_tracing_never_requests_device_queries() {
        for path in [None, Some(PathBuf::from("unused.json"))] {
            for cpu_only in [false, true] {
                let cfg = crate::config::ProfCfg {
                    timeline_path: path.clone(),
                    timeline_cpu_only: cpu_only,
                    ..Default::default()
                };
                assert_eq!(wants_device_timestamps(&cfg), path.is_some() && !cpu_only);
            }
        }
    }
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
