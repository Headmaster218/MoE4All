//! v65 hardware admission probe only: host latency, not a GPU-end or model-speed measurement.
use super::*;
use std::time::{Duration, Instant};

const BUFFER_BYTES: usize = 16 * 1024 * 1024;
const INITIAL_ROUND_TRIPS: usize = 32;
const MAX_ROUND_TRIPS: usize = 128;
const PAIRED_ROUNDS: usize = 8;
const POLL_BUDGET: Duration = Duration::from_millis(32);
const TARGET_HOST_LATENCY: Duration = Duration::from_millis(4);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Wait,
    PollThenWait,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Self::Wait => "A_wait",
            Self::PollThenWait => "B_poll_then_wait",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PollStop {
    Complete,
    Budget,
}

#[derive(Clone, Copy, Debug)]
struct PollOutcome {
    stop: PollStop,
    probes: usize,
    elapsed: Duration,
}

fn bounded_poll<E>(
    budget: Duration,
    mut probe: impl FnMut() -> std::result::Result<bool, E>,
    mut elapsed: impl FnMut() -> Duration,
) -> std::result::Result<PollOutcome, E> {
    let mut probes = 0;
    loop {
        let spent = elapsed();
        if spent >= budget {
            return Ok(PollOutcome {
                stop: PollStop::Budget,
                probes,
                elapsed: spent,
            });
        }
        probes += 1;
        if probe()? {
            return Ok(PollOutcome {
                stop: PollStop::Complete,
                probes,
                elapsed: elapsed(),
            });
        }
        std::hint::spin_loop();
    }
}

fn after_drain<T, E>(
    polled: std::result::Result<T, E>,
    drain: impl FnOnce() -> std::result::Result<(), E>,
) -> std::result::Result<T, E> {
    let drained = drain();
    // Always retire Pending, preserving the first probe error if retirement also fails.
    match polled {
        Err(error) => Err(error),
        Ok(value) => drained.map(|()| value),
    }
}

struct Sample {
    total: Duration,
    submit: Duration,
    finish: Duration,
    poll: Option<PollOutcome>,
}

struct Payload {
    source: Box<dyn Buffer>,
    destination: Box<dyn Buffer>,
    expected: Vec<u8>,
    readback: Vec<u8>,
}

impl Payload {
    fn new(backend: &VulkanBackend) -> Result<Self> {
        // Require device-local coherent VRAM; host upload/download use bulk copies, not MMIO loops.
        let source = backend.alloc_mapped_arena_bda(BUFFER_BYTES)?.0;
        let destination = backend.alloc_mapped_arena_bda(BUFFER_BYTES)?.0;
        let expected = (0..BUFFER_BYTES)
            .map(|index| ((index.wrapping_mul(17) + (index >> 9)) % 255 + 1) as u8)
            .collect();
        Ok(Self {
            source,
            destination,
            expected,
            readback: vec![0; BUFFER_BYTES],
        })
    }

    fn reset(&mut self, backend: &VulkanBackend) -> Result<()> {
        backend.upload(self.source.as_ref(), &self.expected)?;
        // Every expected byte is nonzero, so stale or partially copied destination bytes fail.
        self.readback.fill(0);
        backend.upload(self.destination.as_ref(), &self.readback)
    }

    fn verify(&mut self, backend: &VulkanBackend) -> Result<()> {
        for (name, buffer) in [
            ("source", self.source.as_ref()),
            ("destination", self.destination.as_ref()),
        ] {
            backend.download(buffer, &mut self.readback)?;
            if let Some(index) = self
                .readback
                .iter()
                .zip(&self.expected)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(be(format!(
                    "v65 {name} byte {index}: actual={} expected={}",
                    self.readback[index], self.expected[index]
                )));
            }
        }
        Ok(())
    }
}

fn run(
    backend: &VulkanBackend,
    payload: &mut Payload,
    round_trips: usize,
    mode: Mode,
) -> Result<Sample> {
    assert!((1..=MAX_ROUND_TRIPS).contains(&round_trips));
    payload.reset(backend)?;
    let recorder = backend.recorder()?;
    // Explicitly reject query-bearing or blocking-profile recorders before any trial submit.
    assert!(!recorder.prof_ops && !recorder.stages);
    assert_eq!(recorder.query_pool, vk::QueryPool::null());
    assert_eq!(recorder.submit_query_pool.get(), vk::QueryPool::null());
    assert!(recorder.timeline_query.borrow().is_none());
    recorder.seed_barrier();
    for _ in 0..round_trips {
        recorder.copy(
            payload.source.as_ref(),
            0,
            payload.destination.as_ref(),
            0,
            BUFFER_BYTES,
        );
        recorder.copy(
            payload.destination.as_ref(),
            0,
            payload.source.as_ref(),
            0,
            BUFFER_BYTES,
        );
    }
    recorder.publish_host_reads();

    // Reset, recording and byte verification are outside both timed paths.
    let start = Instant::now();
    let pending = recorder.finish_nowait()?;
    let submitted = Instant::now();
    let poll = match mode {
        Mode::Wait => {
            pending.wait()?;
            None
        }
        Mode::PollThenWait => {
            let polling = Instant::now();
            let outcome = bounded_poll(POLL_BUDGET, || pending.is_complete(), || polling.elapsed());
            // Complete, timeout and error all drain through the unchanged production wait.
            Some(after_drain(outcome, || pending.wait())?)
        }
    };
    let finished = Instant::now();
    let sample = Sample {
        total: finished.duration_since(start),
        submit: submitted.duration_since(start),
        finish: finished.duration_since(submitted),
        poll,
    };
    payload.verify(backend)?;
    Ok(sample)
}

fn us(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

fn distribution(values: &[f64]) -> (f64, f64) {
    assert!(!values.is_empty());
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    };
    let p90_index = (values.len() * 9).div_ceil(10) - 1;
    (median, values[p90_index])
}

fn report(mode: Mode, samples: &[Sample]) {
    let metric = |read: fn(&Sample) -> Duration| {
        distribution(
            &samples
                .iter()
                .map(|sample| us(read(sample)))
                .collect::<Vec<_>>(),
        )
    };
    let (total_median, total_p90) = metric(|sample| sample.total);
    let (submit_median, submit_p90) = metric(|sample| sample.submit);
    let (finish_median, finish_p90) = metric(|sample| sample.finish);
    let polls: Vec<_> = samples.iter().filter_map(|sample| sample.poll).collect();
    let fallbacks = polls
        .iter()
        .filter(|poll| poll.stop == PollStop::Budget)
        .count();
    let poll_median = if polls.is_empty() {
        0.0
    } else {
        distribution(
            &polls
                .iter()
                .map(|poll| us(poll.elapsed))
                .collect::<Vec<_>>(),
        )
        .0
    };
    println!(
        "FENCE_HANDOFF_PROBE_V65 mode={} n={} total_us_median={total_median:.3} total_us_p90={total_p90:.3} submit_us_median={submit_median:.3} submit_us_p90={submit_p90:.3} finish_us_median={finish_median:.3} finish_us_p90={finish_p90:.3} poll_us_median={poll_median:.3} budget_fallbacks={fallbacks}",
        mode.label(),
        samples.len()
    );
}

#[test]
#[ignore = "v65 parent serial GPU runner only; explicitly set MOE4ALL_TEST_DEVICE=Vulkan1"]
fn v65_pending_wait_vs_bounded_poll_then_wait_rotating_pairs() -> Result<()> {
    let selected = std::env::var("MOE4ALL_TEST_DEVICE")
        .expect("parent must explicitly select MOE4ALL_TEST_DEVICE=Vulkan1 on an idle GPU");
    assert_eq!(selected, "Vulkan1");
    assert!(!infr_core::timeline::device_enabled());
    assert!(!pager_profile::active());
    let mut config = infr_core::config::Config::default();
    config.device.dev = Some(selected);
    // An explicit cap disables automatic submission timestamp calibration too.
    config.device.submit_dispatches = Some(0);
    config.prof.ops = false;
    config.prof.stages = false;
    config.prof.pager_profile = false;
    config.prof.timeline_path = None;
    let backend = VulkanBackend::new_with(Arc::new(config))?;
    let mut payload = Payload::new(&backend)?;
    println!(
        "FENCE_HANDOFF_PROBE_V65 device=Vulkan1 gpu_buffers=2 bytes_per_buffer={BUFFER_BYTES} warmups=2 paired_rounds={PAIRED_ROUNDS} poll_budget_ms=32 gpu_queries=false timing=host_finish_nowait_entry_to_pending_wait_return validation=bulk_exact_both_buffers_outside_timing"
    );
    println!(
        "FENCE_HANDOFF_PROBE_V65 submit includes command end/fence checkout/queue submit; finish includes GPU execution plus host wait/drain; neither identifies GPU-end or OS wake loss; spin budget does not bound a stalled driver call or the fallback wait; no model-speed claim"
    );

    let first = run(&backend, &mut payload, INITIAL_ROUND_TRIPS, Mode::Wait)?;
    // One host-only calibration, not a measurement of GPU duration. Keep the measured workload fixed.
    let round_trips = ((INITIAL_ROUND_TRIPS as f64 * TARGET_HOST_LATENCY.as_secs_f64()
        / first.total.as_secs_f64().max(1e-6))
    .round() as usize)
        .clamp(1, MAX_ROUND_TRIPS);
    let second = run(&backend, &mut payload, round_trips, Mode::PollThenWait)?;
    println!(
        "FENCE_HANDOFF_PROBE_V65 warmup_A_us={:.3} initial_round_trips={INITIAL_ROUND_TRIPS} warmup_B_us={:.3} measured_round_trips={round_trips} copied_bytes_per_trial={}",
        us(first.total),
        us(second.total),
        round_trips * 2 * BUFFER_BYTES
    );

    let mut a = Vec::with_capacity(PAIRED_ROUNDS);
    let mut b = Vec::with_capacity(PAIRED_ROUNDS);
    let mut deltas = Vec::with_capacity(PAIRED_ROUNDS);
    for round in 0..PAIRED_ROUNDS {
        let modes = if round % 2 == 0 {
            [Mode::Wait, Mode::PollThenWait]
        } else {
            [Mode::PollThenWait, Mode::Wait]
        };
        for mode in modes {
            let sample = run(&backend, &mut payload, round_trips, mode)?;
            println!(
                "FENCE_HANDOFF_PROBE_V65 round={round} mode={} total_us={:.3} submit_us={:.3} finish_us={:.3} polls={} fallback={}",
                mode.label(),
                us(sample.total),
                us(sample.submit),
                us(sample.finish),
                sample.poll.map_or(0, |poll| poll.probes),
                sample
                    .poll
                    .is_some_and(|poll| poll.stop == PollStop::Budget)
            );
            match mode {
                Mode::Wait => a.push(sample),
                Mode::PollThenWait => b.push(sample),
            }
        }
        deltas.push(us(a[round].total) - us(b[round].total));
    }
    report(Mode::Wait, &a);
    report(Mode::PollThenWait, &b);
    let (delta_median, delta_p90) = distribution(&deltas);
    let wait_median = distribution(&a.iter().map(|sample| us(sample.total)).collect::<Vec<_>>()).0;
    let host_window_admitted = (1000.0..=8000.0).contains(&wait_median);
    println!(
        "FENCE_HANDOFF_PROBE_V65 paired_total_A_minus_B_us_median={delta_median:.3} paired_total_A_minus_B_us_p90={delta_p90:.3} host_workload_1_to_8ms={host_window_admitted} percentile=nearest_rank_p90_n8 no_performance_threshold_or_wakeup_attribution"
    );
    Ok(())
}

#[test]
fn v65_poll_helper_completion_and_budget_are_bounded() {
    use std::cell::Cell;
    let ticks = Cell::new(0);
    let timeout = bounded_poll(
        Duration::from_micros(4),
        || -> std::result::Result<bool, ()> {
            ticks.set(ticks.get() + 1);
            Ok(false)
        },
        || Duration::from_micros(ticks.get()),
    )
    .unwrap();
    assert_eq!(timeout.stop, PollStop::Budget);
    assert_eq!(timeout.probes, 4);
    assert_eq!(timeout.elapsed, Duration::from_micros(4));

    ticks.set(0);
    let complete = bounded_poll(
        Duration::from_micros(4),
        || -> std::result::Result<bool, ()> {
            ticks.set(ticks.get() + 1);
            Ok(ticks.get() == 2)
        },
        || Duration::from_micros(ticks.get()),
    )
    .unwrap();
    assert_eq!(complete.stop, PollStop::Complete);
    assert_eq!(complete.probes, 2);
}

#[test]
fn v65_poll_helper_always_drains_and_preserves_first_error() {
    let polled = bounded_poll(POLL_BUDGET, || Err::<bool, _>("probe"), || Duration::ZERO);
    let mut drained = false;
    let outcome = after_drain(polled, || {
        drained = true;
        Err("drain")
    });
    assert!(drained);
    assert_eq!(outcome.unwrap_err(), "probe");
    assert_eq!(
        after_drain(Ok::<_, &str>(()), || Err("drain")),
        Err("drain")
    );
    assert_eq!(
        distribution(&[8.0, 1.0, 5.0, 3.0, 2.0, 7.0, 4.0, 6.0]),
        (4.5, 8.0)
    );
}
