//! Opt-in scalar Decode offload. The pager still admits every miss into the GPU cache.
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

use infr_core::{error::Result, hostpager::AlignedHostBuffer, DType};
use infr_cpu::task_pool::CpuTaskPool;

use crate::{be, transfer::DeviceTransferTarget};

mod topology;

#[cfg(target_arch = "x86_64")]
#[path = "cpu_miss/kernels.rs"]
mod kernels;

pub(crate) fn eligible(threads: usize, avx2: bool, fma: bool) -> bool {
    threads != 0 && avx2 && fma
}

pub(crate) fn within_limit(misses: usize, maximum: usize) -> bool {
    (1..=3).contains(&maximum) && (1..=maximum).contains(&misses)
}

pub(crate) fn supported(threads: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        threads <= topology::worker_limit()
            && eligible(
                threads,
                is_x86_feature_detected!("avx2"),
                is_x86_feature_detected!("fma"),
            )
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = threads;
        false
    }
}

pub(crate) fn geometry(
    rows: usize,
    misses: usize,
    k: usize,
    n: usize,
    gate: DType,
    up: DType,
    down: DType,
) -> bool {
    rows == 1
        && (1..=3).contains(&misses)
        && k == 2560
        && n == 640
        && matches!(gate, DType::Iq2S | DType::Iq3S)
        && up == gate
        && down == DType::Iq4Nl
}

pub(crate) fn decode_layer(layer: u32) -> bool {
    layer != 0
}

/// Immutable, permanent full-RAM source. Bounded/SSD cache blocks are deliberately ineligible.
pub(crate) struct Weight {
    owner: Arc<AlignedHostBuffer>,
    offset: usize,
    len: usize,
    target: Option<DeviceTransferTarget>,
}

impl Weight {
    pub(crate) fn new(owner: Arc<AlignedHostBuffer>, offset: usize, len: usize) -> Option<Self> {
        (offset.checked_add(len)? <= owner.len()).then_some(Self {
            owner,
            offset,
            len,
            target: None,
        })
    }
    pub(crate) fn set_target(&mut self, target: DeviceTransferTarget) -> Result<()> {
        if target.len() != self.len || target.mapped_ptr().is_none() {
            return Err(be("CPU miss push requires a complete mapped expert target"));
        }
        self.target = Some(target);
        Ok(())
    }
    fn bytes(&self) -> &[u8] {
        // The permanent host store is initialized before execution, and never rewritten.
        unsafe { self.owner.slice(self.offset, self.len) }
    }
    fn row<'a>(&'a self, offset: usize, len: usize, local: &'a mut [u8]) -> &'a [u8] {
        let src = &self.bytes()[offset..offset + len];
        if let Some(target) = &self.target {
            let local = &mut local[..len];
            local.copy_from_slice(src);
            // Rows are disjoint tasks; the open pager epoch protects every live GPU hit.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    local.as_ptr(),
                    target.mapped_ptr().unwrap().add(offset),
                    len,
                );
            }
            local
        } else {
            src
        }
    }
}

pub(crate) struct Expert {
    pub(crate) slot: usize,
    pub(crate) weights: [Weight; 3],
}

pub(crate) struct Output {
    pub(crate) slots: Vec<usize>,
    pub(crate) values: Vec<f32>,
    pub(crate) tasks: Vec<(&'static str, u64, u64)>,
    pub(crate) experts: Vec<Expert>,
}

pub(crate) struct PromotionJob(Option<mpsc::Receiver<Result<()>>>);

impl PromotionJob {
    pub(crate) fn wait(mut self) -> Result<()> {
        self.0
            .take()
            .unwrap()
            .recv()
            .map_err(|_| be("CPU promotion worker disconnected"))?
    }
}

impl Drop for PromotionJob {
    fn drop(&mut self) {
        if let Some(receiver) = self.0.take() {
            let _ = receiver.recv();
        }
    }
}

pub(crate) struct Job {
    receiver: Option<mpsc::Receiver<Result<Output>>>,
}
impl Job {
    #[cfg(test)]
    pub(crate) fn test_receiver(receiver: mpsc::Receiver<Result<Output>>) -> Self {
        Self {
            receiver: Some(receiver),
        }
    }

    pub(crate) fn wait(mut self) -> Result<Output> {
        let receiver = self.receiver.take().unwrap();
        let started = Instant::now();
        // Hit work was submitted first; the remaining FFN tail is usually shorter than an
        // OS sleep/wake round trip. Bound polling so a slow or failed worker still parks.
        loop {
            match receiver.try_recv() {
                Ok(output) => return output,
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(be("CPU miss worker disconnected"));
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
            if started.elapsed() >= std::time::Duration::from_micros(150) {
                break;
            }
            for _ in 0..64 {
                std::hint::spin_loop();
            }
        }
        receiver
            .recv()
            .map_err(|_| be("CPU miss worker disconnected"))?
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        // Cancellation/errors must not close the epoch while a worker still writes its targets.
        if let Some(receiver) = self.receiver.take() {
            let _ = receiver.recv();
        }
    }
}

#[derive(Default)]
struct Stats {
    one: usize,
    two: usize,
    three: usize,
    samples: Vec<(f64, f64, f64)>,
}
struct FixtureCapture {
    directory: std::path::PathBuf,
    captured: std::sync::atomic::AtomicBool,
}
struct Command {
    experts: Vec<Expert>,
    kind: DType,
    x: Vec<f32>,
    queued: Instant,
    context: Option<infr_core::timeline::Context>,
    sender: mpsc::SyncSender<Result<Output>>,
    split_acc: bool,
    flush_denormals: bool,
    grouped_dot: bool,
    fixture: Option<Arc<FixtureCapture>>,
}
enum Work {
    Compute(Command),
    FinishBurst(mpsc::SyncSender<()>),
    Promote(
        Vec<Expert>,
        Option<infr_core::timeline::Context>,
        mpsc::SyncSender<Result<()>>,
    ),
}
pub(crate) struct Worker {
    sender: mpsc::SyncSender<Option<Work>>,
    thread: Option<std::thread::JoinHandle<()>>,
    stats: Arc<Mutex<Stats>>,
    threads: usize,
    push: bool,
    controller_mask: Option<topology::Affinity>,
    split_acc: bool,
    flush_denormals: bool,
    grouped_dot: bool,
    fixture: Option<Arc<FixtureCapture>>,
}

impl Worker {
    pub(crate) fn new(threads: usize, push: bool) -> Option<Self> {
        // Keep GU/Down workers warm at their short phase boundary. Reserve core zero for the
        // controller/driver when the machine has a spare physical core; idle jobs still park.
        Self::new_tuned(threads, push, 16384, 1)
    }

    pub(crate) fn new_options(
        threads: usize,
        push: bool,
        split_acc: bool,
        flush_denormals: bool,
        min_spin: u32,
        grouped_dot: bool,
        idle_park: bool,
        coordinator_poll: u32,
        fixture_dir: Option<std::path::PathBuf>,
    ) -> Option<Self> {
        let mut worker =
            Self::new_tuned_with_poll(threads, push, min_spin, 1, idle_park, coordinator_poll)?;
        worker.split_acc = split_acc;
        worker.flush_denormals = flush_denormals;
        worker.grouped_dot = grouped_dot;
        worker.fixture = fixture_dir.map(|directory| {
            Arc::new(FixtureCapture {
                directory,
                captured: std::sync::atomic::AtomicBool::new(false),
            })
        });
        Some(worker)
    }

    fn new_tuned(threads: usize, push: bool, min_spin: u32, core_offset: usize) -> Option<Self> {
        Self::new_tuned_with_idle(threads, push, min_spin, core_offset, false)
    }

    fn new_tuned_with_idle(
        threads: usize,
        push: bool,
        min_spin: u32,
        core_offset: usize,
        idle_park: bool,
    ) -> Option<Self> {
        Self::new_tuned_with_poll(threads, push, min_spin, core_offset, idle_park, 0)
    }

    fn new_tuned_with_poll(
        threads: usize,
        push: bool,
        min_spin: u32,
        core_offset: usize,
        idle_park: bool,
        coordinator_poll: u32,
    ) -> Option<Self> {
        if !supported(threads) {
            return None;
        }
        let min_spin = min_spin.clamp(1, 1 << 20);
        #[cfg(target_arch = "x86_64")]
        let _ = kernels::tables();
        let (masks, controller_mask) = topology::affinity_plan(threads, core_offset);
        let stats = Arc::new(Mutex::new(Stats::default()));
        let worker_stats = Arc::clone(&stats);
        let (sender, jobs) = mpsc::sync_channel::<Option<Work>>(1);
        let thread = std::thread::Builder::new()
            .name("infr-cpu-miss".into())
            .spawn(move || {
                let pool = if idle_park {
                    CpuTaskPool::with_idle_parking(threads, min_spin)
                } else {
                    CpuTaskPool::with_min_spin(threads, min_spin)
                };
                let cursor = std::sync::atomic::AtomicUsize::new(0);
                let ready = std::sync::Barrier::new(threads);
                pool.run(threads, &|_| {
                    let i = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if let Some(affinity) = masks.get(i) {
                        let _ = topology::set_affinity(*affinity);
                    }
                    ready.wait();
                });
                let mut poll = 0;
                while let Ok(Some(work)) = receive_work(&jobs, poll) {
                    poll = coordinator_poll.min(65536);
                    match work {
                        Work::Compute(command) => run_command(&pool, &worker_stats, command),
                        Work::FinishBurst(sender) => {
                            poll = 0;
                            pool.finish_burst();
                            let _ = sender.send(());
                        }
                        Work::Promote(experts, context, sender) => {
                            let start = context.map(|_| infr_core::timeline::clock_ns());
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    promote(&pool, &experts);
                                }));
                            if let (Some(ctx), Some(start)) = (context, start) {
                                infr_core::timeline::record_cpu_miss(
                                    "cpu_miss_promotion",
                                    ctx,
                                    start,
                                    infr_core::timeline::clock_ns(),
                                );
                            }
                            if result.is_err() {
                                for expert in &experts {
                                    for weight in &expert.weights {
                                        if let Some(target) = &weight.target {
                                            crate::copy_to_mapped(
                                                weight.bytes(),
                                                target.mapped_ptr().unwrap(),
                                            );
                                        }
                                    }
                                }
                            }
                            let _ = sender.send(result.map_err(|_| be("CPU promotion panicked")));
                        }
                    }
                    if idle_park {
                        pool.park_idle_workers();
                    }
                }
            })
            .map_err(|e| tracing::warn!("CPU miss worker unavailable: {e}; using GPU"))
            .ok()?;
        tracing::info!(
            threads,
            push,
            "experimental AVX2+FMA CPU miss worker ready (up to 3 misses, full RAM only)"
        );
        Some(Self {
            sender,
            thread: Some(thread),
            stats,
            threads,
            push,
            controller_mask,
            split_acc: false,
            flush_denormals: false,
            grouped_dot: false,
            fixture: None,
        })
    }

    pub(crate) fn controller_guard(&self) -> Option<AffinityGuard> {
        let previous = topology::set_affinity(self.controller_mask?)?;
        Some(AffinityGuard {
            previous,
            _thread: std::marker::PhantomData,
        })
    }
    pub(crate) fn submit(&self, experts: Vec<Expert>, kind: DType, x: Vec<f32>) -> Job {
        let (sender, receiver) = mpsc::sync_channel(1);
        let command = Command {
            experts,
            kind,
            x,
            queued: Instant::now(),
            context: infr_core::timeline::context(),
            sender,
            split_acc: self.split_acc,
            flush_denormals: self.flush_denormals,
            grouped_dot: self.grouped_dot,
            fixture: self.fixture.clone(),
        };
        let _ = self.sender.send(Some(Work::Compute(command)));
        Job {
            receiver: Some(receiver),
        }
    }

    pub(crate) fn promote(&self, experts: Vec<Expert>) -> PromotionJob {
        let (sender, receiver) = mpsc::sync_channel(1);
        let _ = self.sender.send(Some(Work::Promote(
            experts,
            infr_core::timeline::context(),
            sender,
        )));
        PromotionJob(Some(receiver))
    }

    pub(crate) fn finish_burst(&self) -> Result<()> {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.sender
            .send(Some(Work::FinishBurst(sender)))
            .map_err(|_| be("CPU burst worker disconnected"))?;
        receiver
            .recv()
            .map_err(|_| be("CPU burst completion disconnected"))
    }
}

fn receive_work<T>(
    receiver: &mpsc::Receiver<Option<T>>,
    polls: u32,
) -> std::result::Result<Option<T>, mpsc::RecvError> {
    for _ in 0..polls {
        match receiver.try_recv() {
            Ok(work) => return Ok(work),
            Err(mpsc::TryRecvError::Disconnected) => return Err(mpsc::RecvError),
            Err(mpsc::TryRecvError::Empty) => std::hint::spin_loop(),
        }
    }
    receiver.recv()
}

fn promote(pool: &CpuTaskPool, experts: &[Expert]) {
    let mut copies = Vec::new();
    for expert in experts {
        for weight in &expert.weights {
            if let Some(target) = &weight.target {
                let src = weight.bytes();
                for offset in (0..src.len()).step_by(64 * 1024) {
                    copies.push((
                        src.as_ptr() as usize + offset,
                        target.mapped_ptr().unwrap() as usize + offset,
                        (src.len() - offset).min(64 * 1024),
                    ));
                }
            }
        }
    }
    // Every range is disjoint, and the borrowed owners/targets live until all workers check in.
    promote_copies(pool, &copies);
}

fn promote_copies(pool: &CpuTaskPool, copies: &[(usize, usize, usize)]) {
    let groups = pool.parallelism().min(copies.len());
    pool.run(groups, &|task| {
        for &(src, dst, len) in
            &copies[task * copies.len() / groups..(task + 1) * copies.len() / groups]
        {
            unsafe {
                std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len);
            }
        }
        // Drain this task's WC writes once before publishing completion, not once per 64 KiB.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_sfence();
        }
    });
}

pub(crate) struct AffinityGuard {
    previous: topology::Affinity,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for AffinityGuard {
    fn drop(&mut self) {
        let _ = topology::set_affinity(self.previous);
    }
}

fn run_command(pool: &CpuTaskPool, stats: &Mutex<Stats>, command: Command) {
    let Command {
        experts,
        kind,
        x,
        queued,
        context,
        sender,
        split_acc,
        flush_denormals,
        grouped_dot,
        fixture,
    } = command;
    capture_fixture(fixture.as_deref(), &experts, kind, &x, context);
    let started = Instant::now();
    let clock_start = context.map(|_| infr_core::timeline::clock_ns());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compute_options(
            pool,
            &experts,
            kind,
            &x,
            context,
            split_acc,
            flush_denormals,
            grouped_dot,
        )
    }));
    let clock_end = context.map(|_| infr_core::timeline::clock_ns());
    let mut tasks = Vec::new();
    let result = match result {
        Ok(mut output) => {
            tasks = std::mem::take(&mut output.tasks);
            Ok(output)
        }
        Err(_) => {
            // A failed compute must not leave a half-written expert marked resident.
            for expert in &experts {
                for w in &expert.weights {
                    if let Some(target) = &w.target {
                        crate::copy_to_mapped(w.bytes(), target.mapped_ptr().unwrap());
                    }
                }
            }
            Err(be("CPU miss computation panicked"))
        }
    };
    let count = experts.len();
    let sample = (
        count as f64,
        started.duration_since(queued).as_secs_f64() * 1e6,
        started.elapsed().as_secs_f64() * 1e6,
    );
    let result = result.map(|mut output| {
        output.experts = experts;
        output
    });
    let _ = sender.send(result);
    if let (Some(ctx), Some(start), Some(end)) = (context, clock_start, clock_end) {
        let label = match count {
            1 => "cpu_miss_ffn_one",
            2 => "cpu_miss_ffn_two",
            _ => "cpu_miss_ffn_three",
        };
        infr_core::timeline::record_cpu_miss(label, ctx, start, end);
    }
    // Retain the private pool's bounded spin budget across result delivery and promotion.
    // An explicit empty parking job here adds another handoff on every layer; idle workers
    // still park naturally, and the normal CPU interpreter's pool is unchanged.
    {
        let mut stats = stats.lock().unwrap();
        match count {
            1 => stats.one += 1,
            2 => stats.two += 1,
            _ => stats.three += 1,
        }
        if stats.samples.len() < 4096 {
            stats.samples.push(sample);
        }
    }
    if let Some(ctx) = context {
        for (name, start, end) in tasks {
            infr_core::timeline::record_cpu_miss(name, ctx, start, end);
        }
    }
}

fn capture_fixture(
    fixture: Option<&FixtureCapture>,
    experts: &[Expert],
    kind: DType,
    x: &[f32],
    context: Option<infr_core::timeline::Context>,
) {
    let Some(fixture) = fixture else {
        return;
    };
    if experts.len() != 1
        || kind != DType::Iq2S
        || !context.is_some_and(|ctx| ctx.layer == 8)
        || fixture
            .captured
            .swap(true, std::sync::atomic::Ordering::Relaxed)
    {
        return;
    }
    let directory = &fixture.directory;
    let save = || -> std::io::Result<()> {
        std::fs::create_dir_all(directory)?;
        std::fs::write(directory.join("input.f32"), bytemuck::cast_slice(x))?;
        for (role, weight) in ["gate", "up", "down"].iter().zip(&experts[0].weights) {
            std::fs::write(directory.join(format!("{role}.bin")), weight.bytes())?;
        }
        Ok(())
    };
    if let Err(error) = save() {
        tracing::warn!(%error, "CPU miss diagnostic fixture capture failed");
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.sender.send(None);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let stats = self.stats.lock().unwrap();
        for misses in [1, 2, 3] {
            let mut compute: Vec<_> = stats
                .samples
                .iter()
                .filter(|s| s.0 == misses as f64)
                .map(|s| s.2)
                .collect();
            let mut queue: Vec<_> = stats
                .samples
                .iter()
                .filter(|s| s.0 == misses as f64)
                .map(|s| s.1)
                .collect();
            compute.sort_by(f64::total_cmp);
            queue.sort_by(f64::total_cmp);
            if !compute.is_empty() {
                tracing::info!(
                    threads = self.threads,
                    push = self.push,
                    misses,
                    samples = compute.len(),
                    jobs = match misses {
                        1 => stats.one,
                        2 => stats.two,
                        _ => stats.three,
                    },
                    compute_p50_us = compute[compute.len() / 2],
                    compute_p90_us = compute[compute.len() * 9 / 10],
                    queue_p50_us = queue[queue.len() / 2],
                    "CPU miss experiment summary (bounded first 4096 samples)"
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn compute(
    pool: &CpuTaskPool,
    experts: &[Expert],
    kind: DType,
    x: &[f32],
    context: Option<infr_core::timeline::Context>,
) -> Output {
    if kind == DType::Iq2S || experts.len() == 2 {
        compute_tiled::<false, true>(pool, experts, kind, x, context)
    } else {
        compute_tiled::<false, false>(pool, experts, kind, x, context)
    }
}

#[cfg(target_arch = "x86_64")]
fn compute_tiled<const GU_PAIRED: bool, const DOWN_PAIRED: bool>(
    pool: &CpuTaskPool,
    experts: &[Expert],
    kind: DType,
    x: &[f32],
    context: Option<infr_core::timeline::Context>,
) -> Output {
    compute_chunked::<GU_PAIRED, DOWN_PAIRED, 32, 128>(pool, experts, kind, x, context)
}

#[cfg(target_arch = "x86_64")]
fn compute_single_dispatch(pool: &CpuTaskPool, expert: &Expert, kind: DType, x: &[f32]) -> Output {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let mut intermediate = vec![0.0f32; 640];
    let mut values = vec![0.0f32; 2560];
    let intermediate_ptr = intermediate.as_mut_ptr() as usize;
    let output_ptr = values.as_mut_ptr() as usize;
    let count = pool.parallelism();
    let arrived = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    // One task per participating thread: disjoint GU ranges, then an acquire/release
    // barrier before disjoint Down ranges. Always arrive even if GU panics.
    pool.run(count, &|worker| {
        let start = worker * 640 / count;
        let end = (worker + 1) * 640 / count;
        let gu = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            let out = std::slice::from_raw_parts_mut(
                (intermediate_ptr as *mut f32).add(start),
                end - start,
            );
            if kind == DType::Iq2S {
                gate_rows::<2, false, false, false, false>(expert, start, x, out);
            } else {
                gate_rows::<3, false, false, false, false>(expert, start, x, out);
            }
        }));
        if gu.is_err() {
            failed.store(true, Ordering::Release);
        }
        arrived.fetch_add(1, Ordering::AcqRel);
        let mut spins = 0usize;
        while arrived.load(Ordering::Acquire) != count {
            if spins % 16384 == 16383 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
            spins += 1;
        }
        if failed.load(Ordering::Acquire) {
            return;
        }
        let start = worker * 2560 / count;
        let end = (worker + 1) * 2560 / count;
        unsafe {
            let a = std::slice::from_raw_parts(intermediate_ptr as *const f32, 640);
            let out =
                std::slice::from_raw_parts_mut((output_ptr as *mut f32).add(start), end - start);
            // Odd partition lengths at 5/6 cores use the scalar-row tile without losing a tail.
            if kind == DType::Iq2S && out.len().is_multiple_of(2) {
                down_rows::<true, false, false>(expert, start, a, out);
            } else {
                down_rows::<false, false, false>(expert, start, a, out);
            }
        }
    });
    assert!(!failed.load(Ordering::Acquire), "CPU GU phase failed");
    Output {
        slots: vec![expert.slot],
        values,
        tasks: Vec::new(),
        experts: Vec::new(),
    }
}

#[cfg(target_arch = "x86_64")]
fn compute_options(
    pool: &CpuTaskPool,
    experts: &[Expert],
    kind: DType,
    x: &[f32],
    context: Option<infr_core::timeline::Context>,
    split: bool,
    flush: bool,
    grouped: bool,
) -> Output {
    if grouped && !split && !flush {
        return if kind == DType::Iq2S {
            compute_chunked_impl::<false, true, 32, 128, false, false, true, false>(
                pool, experts, kind, x, context,
            )
        } else {
            compute_chunked_impl::<false, false, 32, 128, false, false, true, false>(
                pool, experts, kind, x, context,
            )
        };
    }
    match (split, flush, kind == DType::Iq2S || experts.len() == 2) {
        (false, false, _) => compute(pool, experts, kind, x, context),
        (true, false, true) => {
            compute_chunked_impl::<false, true, 32, 128, true, false, false, false>(
                pool, experts, kind, x, context,
            )
        }
        (true, false, false) => {
            compute_chunked_impl::<false, false, 32, 128, true, false, false, false>(
                pool, experts, kind, x, context,
            )
        }
        (false, true, true) => {
            compute_chunked_impl::<false, true, 32, 128, false, true, false, false>(
                pool, experts, kind, x, context,
            )
        }
        (false, true, false) => {
            compute_chunked_impl::<false, false, 32, 128, false, true, false, false>(
                pool, experts, kind, x, context,
            )
        }
        (true, true, true) => {
            compute_chunked_impl::<false, true, 32, 128, true, true, false, false>(
                pool, experts, kind, x, context,
            )
        }
        (true, true, false) => {
            compute_chunked_impl::<false, false, 32, 128, true, true, false, false>(
                pool, experts, kind, x, context,
            )
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn compute_options(
    pool: &CpuTaskPool,
    experts: &[Expert],
    kind: DType,
    x: &[f32],
    context: Option<infr_core::timeline::Context>,
    _: bool,
    _: bool,
    _: bool,
) -> Output {
    compute(pool, experts, kind, x, context)
}

#[cfg(target_arch = "x86_64")]
fn compute_chunked<
    const GU_PAIRED: bool,
    const DOWN_PAIRED: bool,
    const GU_CHUNK: usize,
    const DOWN_CHUNK: usize,
>(
    pool: &CpuTaskPool,
    experts: &[Expert],
    kind: DType,
    x: &[f32],
    context: Option<infr_core::timeline::Context>,
) -> Output {
    compute_chunked_impl::<GU_PAIRED, DOWN_PAIRED, GU_CHUNK, DOWN_CHUNK, false, false, false, false>(
        pool, experts, kind, x, context,
    )
}

#[cfg(target_arch = "x86_64")]
struct FlushDenormals(u32);
#[cfg(target_arch = "x86_64")]
impl FlushDenormals {
    #[allow(deprecated)]
    fn new() -> Self {
        unsafe {
            let previous = std::arch::x86_64::_mm_getcsr();
            std::arch::x86_64::_mm_setcsr(previous | (1 << 15) | (1 << 6));
            Self(previous)
        }
    }
}
#[cfg(target_arch = "x86_64")]
impl Drop for FlushDenormals {
    #[allow(deprecated)]
    fn drop(&mut self) {
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.0);
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn compute_chunked_impl<
    const GU_PAIRED: bool,
    const DOWN_PAIRED: bool,
    const GU_CHUNK: usize,
    const DOWN_CHUNK: usize,
    const SPLIT: bool,
    const FLUSH: bool,
    const GROUPED: bool,
    const DUAL: bool,
>(
    pool: &CpuTaskPool,
    experts: &[Expert],
    kind: DType,
    x: &[f32],
    context: Option<infr_core::timeline::Context>,
) -> Output {
    compute_chunked_down_tile::<
        GU_PAIRED,
        DOWN_PAIRED,
        GU_CHUNK,
        DOWN_CHUNK,
        SPLIT,
        FLUSH,
        GROUPED,
        DUAL,
        2,
    >(pool, experts, kind, x, context)
}

#[cfg(target_arch = "x86_64")]
fn compute_chunked_down_tile<
    const GU_PAIRED: bool,
    const DOWN_PAIRED: bool,
    const GU_CHUNK: usize,
    const DOWN_CHUNK: usize,
    const SPLIT: bool,
    const FLUSH: bool,
    const GROUPED: bool,
    const DUAL: bool,
    const DOWN_TILE: usize,
>(
    pool: &CpuTaskPool,
    experts: &[Expert],
    kind: DType,
    x: &[f32],
    context: Option<infr_core::timeline::Context>,
) -> Output {
    const {
        assert!(640 % GU_CHUNK == 0 && GU_CHUNK.is_multiple_of(2));
    }
    const {
        assert!(2560 % DOWN_CHUNK == 0 && DOWN_CHUNK.is_multiple_of(2));
    }
    let gu_count = 640 / GU_CHUNK;
    let down_count = 2560 / DOWN_CHUNK;
    let task_times = |count| -> Vec<(std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64)> {
        (0..experts.len() * count)
            .map(|_| (0.into(), 0.into()))
            .collect()
    };
    let gu_tasks = context.map(|_| task_times(gu_count));
    let down_tasks = context.map(|_| task_times(down_count));
    let mut intermediate = vec![0.0f32; experts.len() * 640];
    let mut values = vec![0.0f32; experts.len() * 2560];
    let gu_start = context.map(|_| infr_core::timeline::clock_ns());
    pool.for_chunks_mut(&mut intermediate, GU_CHUNK, &|chunk, out| {
        let _float_mode = FLUSH.then(FlushDenormals::new);
        if let Some(tasks) = &gu_tasks {
            tasks[chunk].0.store(
                infr_core::timeline::clock_ns(),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let expert = &experts[chunk / gu_count];
        let start = chunk % gu_count * GU_CHUNK;
        unsafe {
            if kind == DType::Iq2S {
                gate_rows::<2, GU_PAIRED, SPLIT, GROUPED, DUAL>(expert, start, x, out);
            } else {
                gate_rows::<3, GU_PAIRED, SPLIT, GROUPED, DUAL>(expert, start, x, out);
            }
        }
        if let Some(tasks) = &gu_tasks {
            tasks[chunk].1.store(
                infr_core::timeline::clock_ns(),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    });
    let down_start = context.map(|_| infr_core::timeline::clock_ns());
    pool.for_chunks_mut(&mut values, DOWN_CHUNK, &|chunk, out| {
        let _float_mode = FLUSH.then(FlushDenormals::new);
        if let Some(tasks) = &down_tasks {
            tasks[chunk].0.store(
                infr_core::timeline::clock_ns(),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let e = chunk / down_count;
        let start = chunk % down_count * DOWN_CHUNK;
        let a = &intermediate[e * 640..(e + 1) * 640];
        unsafe {
            if DOWN_TILE == 4 || DOWN_TILE == 8 {
                down_rows_tile::<DOWN_TILE>(&experts[e], start, a, out);
            } else {
                down_rows::<DOWN_PAIRED, GROUPED, DUAL>(&experts[e], start, a, out);
            }
        }
        if let Some(tasks) = &down_tasks {
            tasks[chunk].1.store(
                infr_core::timeline::clock_ns(),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    });
    let down_end = context.map(|_| infr_core::timeline::clock_ns());
    Output {
        slots: experts.iter().map(|e| e.slot).collect(),
        values,
        experts: Vec::new(),
        tasks: [
            ("cpu_miss_task_gu", gu_tasks),
            ("cpu_miss_task_down", down_tasks),
        ]
        .into_iter()
        .flat_map(|(name, tasks)| {
            tasks
                .into_iter()
                .flatten()
                .map(move |(start, end)| (name, start.into_inner(), end.into_inner()))
        })
        .chain(context.into_iter().flat_map(|_| {
            [
                ("cpu_miss_gu", gu_start.unwrap(), down_start.unwrap()),
                ("cpu_miss_down", down_start.unwrap(), down_end.unwrap()),
            ]
        }))
        .collect(),
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn compute(
    _: &CpuTaskPool,
    _: &[Expert],
    _: DType,
    _: &[f32],
    _: Option<infr_core::timeline::Context>,
) -> Output {
    unreachable!("CPU miss ISA gate")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gate_rows<
    const KIND: u8,
    const PAIRED: bool,
    const SPLIT: bool,
    const GROUPED: bool,
    const DUAL: bool,
>(
    expert: &Expert,
    start: usize,
    x: &[f32],
    out: &mut [f32],
) {
    let stride = if KIND == 2 { 820 } else { 1100 };
    let tables = kernels::tables();
    if expert.weights[..2].iter().all(|w| w.target.is_none()) {
        let gate = expert.weights[0].bytes();
        let up = expert.weights[1].bytes();
        assert!((start + out.len()) * stride <= gate.len());
        assert!((start + out.len()) * stride <= up.len());
        if PAIRED {
            for (row, pair_out) in out.chunks_exact_mut(2).enumerate() {
                let offset = (start + row * 2) * stride;
                let rows = [
                    gate.as_ptr().add(offset),
                    up.as_ptr().add(offset),
                    gate.as_ptr().add(offset + stride),
                    up.as_ptr().add(offset + stride),
                ];
                let v = if DUAL && !GROUPED {
                    kernels::tile_stream_dual::<KIND, 4>(rows, x, tables)
                } else if GROUPED && SPLIT {
                    kernels::tile_stream_grouped_lut::<KIND, 4>(rows, x, tables)
                } else if GROUPED {
                    kernels::tile_stream_grouped::<KIND, 4>(rows, x, tables)
                } else if SPLIT {
                    kernels::tile_stream_split::<KIND, 4, true>(rows, x, tables)
                } else if KIND == 2 {
                    kernels::tile_stream::<2, 4, true>(rows, x, tables)
                } else {
                    kernels::tile_lut::<3, 4>(rows, x, tables)
                };
                pair_out[0] = v[0] / (1.0 + (-v[0]).exp()) * v[1];
                pair_out[1] = v[2] / (1.0 + (-v[2]).exp()) * v[3];
            }
            return;
        }
        for (row, value) in out.iter_mut().enumerate() {
            let offset = (start + row) * stride;
            let rows = [gate.as_ptr().add(offset), up.as_ptr().add(offset)];
            let pair = if DUAL && !GROUPED {
                kernels::tile_stream_dual::<KIND, 2>(rows, x, tables)
            } else if GROUPED && SPLIT {
                kernels::tile_stream_grouped_lut::<KIND, 2>(rows, x, tables)
            } else if GROUPED {
                kernels::tile_stream_grouped::<KIND, 2>(rows, x, tables)
            } else if SPLIT {
                kernels::tile_stream_split::<KIND, 2, true>(rows, x, tables)
            } else if KIND == 2 {
                kernels::tile_stream::<2, 2, true>(rows, x, tables)
            } else {
                kernels::tile_lut::<3, 2>(rows, x, tables)
            };
            *value = pair[0] / (1.0 + (-pair[0]).exp()) * pair[1];
        }
        return;
    }
    let mut gate_local = [0u8; 1100];
    let mut up_local = [0u8; 1100];
    if PAIRED {
        let mut gate_next = [0u8; 1100];
        let mut up_next = [0u8; 1100];
        for (row, pair_out) in out.chunks_exact_mut(2).enumerate() {
            let offset = (start + row * 2) * stride;
            let gate = expert.weights[0].row(offset, stride, &mut gate_local);
            let up = expert.weights[1].row(offset, stride, &mut up_local);
            let next_gate = expert.weights[0].row(offset + stride, stride, &mut gate_next);
            let next_up = expert.weights[1].row(offset + stride, stride, &mut up_next);
            let rows = [
                gate.as_ptr(),
                up.as_ptr(),
                next_gate.as_ptr(),
                next_up.as_ptr(),
            ];
            let v = if DUAL && !GROUPED {
                kernels::tile_stream_dual::<KIND, 4>(rows, x, tables)
            } else if GROUPED && SPLIT {
                kernels::tile_stream_grouped_lut::<KIND, 4>(rows, x, tables)
            } else if GROUPED {
                kernels::tile_stream_grouped::<KIND, 4>(rows, x, tables)
            } else if SPLIT {
                kernels::tile_stream_split::<KIND, 4, true>(rows, x, tables)
            } else if KIND == 2 {
                kernels::tile_stream::<2, 4, true>(rows, x, tables)
            } else {
                kernels::tile_lut::<3, 4>(rows, x, tables)
            };
            pair_out[0] = v[0] / (1.0 + (-v[0]).exp()) * v[1];
            pair_out[1] = v[2] / (1.0 + (-v[2]).exp()) * v[3];
        }
        if expert.weights.iter().any(|w| w.target.is_some()) {
            std::arch::x86_64::_mm_sfence();
        }
        return;
    }
    for (row, value) in out.iter_mut().enumerate() {
        let offset = (start + row) * stride;
        let gate = expert.weights[0].row(offset, stride, &mut gate_local);
        let up = expert.weights[1].row(offset, stride, &mut up_local);
        let pair = if DUAL && !GROUPED {
            kernels::tile_stream_dual::<KIND, 2>([gate.as_ptr(), up.as_ptr()], x, tables)
        } else if GROUPED && SPLIT {
            kernels::tile_stream_grouped_lut::<KIND, 2>([gate.as_ptr(), up.as_ptr()], x, tables)
        } else if GROUPED {
            kernels::tile_stream_grouped::<KIND, 2>([gate.as_ptr(), up.as_ptr()], x, tables)
        } else if SPLIT {
            kernels::tile_stream_split::<KIND, 2, true>([gate.as_ptr(), up.as_ptr()], x, tables)
        } else if KIND == 2 {
            kernels::tile_stream::<2, 2, true>([gate.as_ptr(), up.as_ptr()], x, tables)
        } else {
            kernels::tile_lut::<3, 2>([gate.as_ptr(), up.as_ptr()], x, tables)
        };
        *value = pair[0] / (1.0 + (-pair[0]).exp()) * pair[1];
    }
    if expert.weights.iter().any(|w| w.target.is_some()) {
        std::arch::x86_64::_mm_sfence();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn down_rows<const PAIRED: bool, const GROUPED: bool, const DUAL: bool>(
    expert: &Expert,
    start: usize,
    a: &[f32],
    out: &mut [f32],
) {
    if expert.weights[2].target.is_none() {
        let bytes = expert.weights[2].bytes();
        assert!((start + out.len()) * 360 <= bytes.len());
        if PAIRED {
            for (row, pair_out) in out.chunks_exact_mut(2).enumerate() {
                let ptr = bytes.as_ptr().add((start + row * 2) * 360);
                let rows = [ptr, ptr.add(360)];
                pair_out.copy_from_slice(&if GROUPED && DUAL {
                    kernels::tile_iq4_grouped_dual::<2>(rows, a)
                } else if GROUPED {
                    kernels::tile_iq4_grouped::<2>(rows, a)
                } else {
                    kernels::tile::<4, 2>(rows, a)
                });
            }
        } else {
            for (row, value) in out.iter_mut().enumerate() {
                let rows = [bytes.as_ptr().add((start + row) * 360)];
                *value = if GROUPED && DUAL {
                    kernels::tile_iq4_grouped_dual::<1>(rows, a)[0]
                } else if GROUPED {
                    kernels::tile_iq4_grouped::<1>(rows, a)[0]
                } else {
                    kernels::tile::<4, 1>(rows, a)[0]
                };
            }
        }
        return;
    }
    let mut local = [0u8; 360];
    if PAIRED {
        let mut next_local = [0u8; 360];
        for (row, pair_out) in out.chunks_exact_mut(2).enumerate() {
            let offset = (start + row * 2) * 360;
            let w = expert.weights[2].row(offset, 360, &mut local);
            let next = expert.weights[2].row(offset + 360, 360, &mut next_local);
            let rows = [w.as_ptr(), next.as_ptr()];
            pair_out.copy_from_slice(&if GROUPED && DUAL {
                kernels::tile_iq4_grouped_dual::<2>(rows, a)
            } else if GROUPED {
                kernels::tile_iq4_grouped::<2>(rows, a)
            } else {
                kernels::tile::<4, 2>(rows, a)
            });
        }
        if expert.weights[2].target.is_some() {
            std::arch::x86_64::_mm_sfence();
        }
        return;
    }
    for (row, value) in out.iter_mut().enumerate() {
        let w = expert.weights[2].row((start + row) * 360, 360, &mut local);
        *value = if GROUPED && DUAL {
            kernels::tile_iq4_grouped_dual::<1>([w.as_ptr()], a)[0]
        } else if GROUPED {
            kernels::tile_iq4_grouped::<1>([w.as_ptr()], a)[0]
        } else {
            kernels::tile::<4, 1>([w.as_ptr()], a)[0]
        };
    }
    if expert.weights[2].target.is_some() {
        std::arch::x86_64::_mm_sfence();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn down_rows_tile<const NR: usize>(
    expert: &Expert,
    start: usize,
    a: &[f32],
    out: &mut [f32],
) {
    const {
        assert!(NR >= 1 && NR <= 8);
    }
    // Selected runtime math still uses the original path until this candidate wins A/B.
    if expert.weights[2].target.is_some() {
        down_rows::<false, false, false>(expert, start, a, out);
        return;
    }
    let bytes = expert.weights[2].bytes();
    assert!((start + out.len()) * 360 <= bytes.len());
    let output_len = out.len();
    let mut chunks = out.chunks_exact_mut(NR);
    for (row, values) in chunks.by_ref().enumerate() {
        let ptr = bytes.as_ptr().add((start + row * NR) * 360);
        values.copy_from_slice(&kernels::tile::<4, NR>(
            std::array::from_fn(|i| ptr.add(i * 360)),
            a,
        ));
    }
    let remainder = chunks.into_remainder();
    let tail = output_len - remainder.len();
    for (row, value) in remainder.iter_mut().enumerate() {
        *value = kernels::tile::<4, 1>([bytes.as_ptr().add((start + tail + row) * 360)], a)[0];
    }
}

#[cfg(all(test, windows))]
fn physical_masks() -> Vec<usize> {
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Info {
        mask: usize,
        relationship: u32,
        pad: u32,
        reserved: [u64; 2],
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetLogicalProcessorInformation(buffer: *mut Info, length: *mut u32) -> i32;
    }
    let mut bytes = 0;
    unsafe {
        GetLogicalProcessorInformation(std::ptr::null_mut(), &mut bytes);
    }
    if bytes == 0 || bytes as usize % std::mem::size_of::<Info>() != 0 {
        return Vec::new();
    }
    let mut info = vec![Info::default(); bytes as usize / std::mem::size_of::<Info>()];
    if unsafe { GetLogicalProcessorInformation(info.as_mut_ptr(), &mut bytes) } == 0 {
        return Vec::new();
    }
    info.iter()
        .filter(|i| i.relationship == 0 && i.mask != 0)
        .map(|i| 1usize << i.mask.trailing_zeros())
        .collect()
}
#[cfg(all(test, not(windows)))]
fn physical_masks() -> Vec<usize> {
    Vec::new()
}

#[cfg(all(test, windows))]
fn set_affinity(mask: usize) -> Option<usize> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn SetThreadAffinityMask(thread: *mut std::ffi::c_void, mask: usize) -> usize;
    }
    let previous = unsafe { SetThreadAffinityMask(GetCurrentThread(), mask) };
    (previous != 0).then_some(previous)
}
#[cfg(all(test, not(windows)))]
fn set_affinity(_: usize) -> Option<usize> {
    None
}

#[cfg(test)]
fn pin(mask: Option<usize>) {
    if let Some(mask) = mask {
        let _ = set_affinity(mask);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinator_poll_preserves_pending_work_shutdown_and_blocking_fallback() {
        for polls in [0, 32, 32768] {
            let (tx, rx) = mpsc::channel();
            tx.send(Some(7)).unwrap();
            assert_eq!(receive_work(&rx, polls).unwrap(), Some(7));
            tx.send(None).unwrap();
            assert_eq!(receive_work(&rx, polls).unwrap(), None);
            drop(tx);
            assert!(receive_work(&rx, polls).is_err());
            let (tx, rx) = mpsc::channel();
            let sender = std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(1));
                tx.send(Some(11)).unwrap();
            });
            assert_eq!(receive_work(&rx, polls).unwrap(), Some(11));
            sender.join().unwrap();
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn wider_down_tiles_preserve_every_f32_bit_and_tail() {
        if !supported(4) {
            return;
        }
        let expert = fixture(DType::Iq2S, 3);
        for scale in [0.0, 0.003, 0.2, f32::MIN_POSITIVE] {
            let a: Vec<_> = (0..640).map(|i| ((i % 31) as f32 - 15.0) * scale).collect();
            for start in [0, 3] {
                for len in [0, 1, 3, 4, 7, 8, 9, 31, 128, 2557] {
                    let mut reference = vec![0.0; len];
                    unsafe {
                        down_rows::<false, false, false>(&expert, start, &a, &mut reference);
                    }
                    let mut four = vec![0.0; len];
                    let mut eight = vec![0.0; len];
                    unsafe {
                        down_rows_tile::<4>(&expert, start, &a, &mut four);
                        down_rows_tile::<8>(&expert, start, &a, &mut eight);
                    }
                    for actual in [four, eight] {
                        assert_eq!(
                            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "start={start} len={len} scale={scale}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn alternate_fma_and_denormal_modes_preserve_finite_expert_outputs() {
        if !supported(4) {
            return;
        }
        let pool = CpuTaskPool::new(4);
        for kind in [DType::Iq2S, DType::Iq3S] {
            let experts = [fixture(kind, 3)];
            for scale in [0.0, 0.003, 0.2, f32::MIN_POSITIVE] {
                let x: Vec<_> = (0..2560)
                    .map(|i| ((i % 31) as f32 - 15.0) * scale)
                    .collect();
                let reference = compute(&pool, &experts, kind, &x, None);
                for (split, flush) in [(true, false), (false, true), (true, true)] {
                    let output =
                        compute_options(&pool, &experts, kind, &x, None, split, flush, false);
                    for (&actual, &expected) in output.values.iter().zip(&reference.values) {
                        assert!(actual.is_finite() && expected.is_finite());
                        assert!(
                            (actual - expected).abs() <= 0.003 + expected.abs() * 0.002,
                            "{kind:?} split={split} flush={flush} {actual} != {expected}"
                        );
                    }
                }
            }
        }
        #[allow(deprecated)]
        unsafe {
            let before = std::arch::x86_64::_mm_getcsr();
            {
                let _guard = FlushDenormals::new();
                assert_eq!(std::arch::x86_64::_mm_getcsr() & 0x8040, 0x8040);
            }
            assert_eq!(std::arch::x86_64::_mm_getcsr(), before);
            let _ = std::panic::catch_unwind(|| {
                let _guard = FlushDenormals::new();
                panic!("MXCSR restore probe");
            });
            assert_eq!(std::arch::x86_64::_mm_getcsr(), before);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn grouped_scales_preserve_f32_expert_outputs() {
        if !supported(4) {
            return;
        }
        for threads in [4, 5, 6] {
            let pool = CpuTaskPool::new(threads);
            for kind in [DType::Iq2S, DType::Iq3S] {
                let experts = [fixture(kind, 3)];
                for scale in [0.0, 0.003, 0.2, f32::MIN_POSITIVE] {
                    let x: Vec<_> = (0..2560)
                        .map(|i| ((i % 31) as f32 - 15.0) * scale)
                        .collect();
                    let reference = compute(&pool, &experts, kind, &x, None);
                    let compact =
                        compute_chunked_impl::<false, true, 32, 128, false, false, true, false>(
                            &pool, &experts, kind, &x, None,
                        );
                    let grid16 =
                        compute_chunked_impl::<false, true, 32, 128, true, false, true, false>(
                            &pool, &experts, kind, &x, None,
                        );
                    assert_eq!(
                        compact
                            .values
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        grid16
                            .values
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                    for output in [
                        compute_chunked_impl::<false, true, 32, 128, false, false, true, false>(
                            &pool, &experts, kind, &x, None,
                        ),
                        compute_chunked_impl::<false, true, 32, 128, false, false, false, true>(
                            &pool, &experts, kind, &x, None,
                        ),
                        compute_chunked_impl::<false, true, 32, 128, false, false, true, true>(
                            &pool, &experts, kind, &x, None,
                        ),
                        compute_chunked_impl::<false, true, 32, 128, true, false, true, false>(
                            &pool, &experts, kind, &x, None,
                        ),
                        compute_chunked_impl::<true, true, 32, 128, false, false, true, false>(
                            &pool, &experts, kind, &x, None,
                        ),
                        compute_chunked_impl::<false, true, 32, 128, true, false, true, true>(
                            &pool, &experts, kind, &x, None,
                        ),
                    ] {
                        for (&actual, &expected) in output.values.iter().zip(&reference.values) {
                            assert!(actual.is_finite() && expected.is_finite());
                            assert!(
                                (actual - expected).abs() <= 0.003 + expected.abs() * 0.002,
                                "{kind:?} grouped/dual {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn promotion_groups_copy_every_range_and_preserve_guards() {
        if !supported(4) {
            return;
        }
        for threads in [4, 5, 6] {
            let pool = CpuTaskPool::new(threads);
            for count in [0, 1, 2, 3, 4, 5, 31, 32, 33] {
                const STRIDE: usize = 65536 + 64;
                let source: Vec<u8> = (0..count * STRIDE)
                    .map(|i| (i.wrapping_mul(17) % 251) as u8)
                    .collect();
                let mut output = vec![0xaau8; source.len()];
                let mut expected = output.clone();
                let mut copies = Vec::new();
                for i in 0..count {
                    let offset = i * STRIDE;
                    let size = if i + 1 == count { 257 } else { 65536 };
                    expected[offset..offset + size].copy_from_slice(&source[offset..offset + size]);
                    copies.push((
                        source.as_ptr() as usize + offset,
                        output.as_mut_ptr() as usize + offset,
                        size,
                    ));
                }
                promote_copies(&pool, &copies);
                assert_eq!(output, expected, "threads={threads} ranges={count}");
            }
        }
    }

    #[test]
    fn result_receive_handles_ready_delayed_error_disconnect_and_cancel() {
        let output = || Output {
            slots: vec![3],
            values: vec![1.25],
            tasks: Vec::new(),
            experts: Vec::new(),
        };
        let (sender, receiver) = mpsc::channel();
        sender.send(Ok(output())).unwrap();
        assert_eq!(Job::test_receiver(receiver).wait().unwrap().values, [1.25]);

        let (sender, receiver) = mpsc::channel();
        let producer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(5));
            sender.send(Ok(output())).unwrap();
        });
        assert_eq!(Job::test_receiver(receiver).wait().unwrap().slots, [3]);
        producer.join().unwrap();

        let (sender, receiver) = mpsc::channel();
        sender.send(Err(be("fixture failure"))).unwrap();
        assert!(Job::test_receiver(receiver).wait().is_err());
        let (sender, receiver) = mpsc::channel();
        drop(sender);
        assert!(Job::test_receiver(receiver).wait().is_err());

        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_done = Arc::clone(&done);
        let (sender, receiver) = mpsc::channel();
        let producer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(5));
            worker_done.store(true, std::sync::atomic::Ordering::Release);
            sender.send(Ok(output())).unwrap();
        });
        drop(Job::test_receiver(receiver));
        assert!(done.load(std::sync::atomic::Ordering::Acquire));
        producer.join().unwrap();
    }

    #[cfg(all(windows, target_arch = "x86_64"))]
    #[test]
    fn controller_affinity_restores_on_normal_and_error_exit() {
        #[repr(C)]
        #[derive(Default)]
        struct GroupAffinity {
            mask: usize,
            group: u16,
            reserved: [u16; 3],
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn GetThreadGroupAffinity(
                thread: *mut std::ffi::c_void,
                affinity: *mut GroupAffinity,
            ) -> i32;
        }
        let current = || {
            let mut affinity = GroupAffinity::default();
            assert_ne!(
                unsafe { GetThreadGroupAffinity(GetCurrentThread(), &mut affinity) },
                0
            );
            affinity.mask
        };
        if !supported(4) || physical_masks().len() <= 4 {
            return;
        }
        let worker = Worker::new(4, false).unwrap();
        let original = current();
        {
            let _guard = worker.controller_guard().unwrap();
            assert_eq!(current(), worker.controller_mask.unwrap().mask);
        }
        assert_eq!(current(), original);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = worker.controller_guard().unwrap();
            panic!("exercise affinity restoration during an error");
        }));
        assert!(result.is_err());
        assert_eq!(current(), original);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "worker scheduling probe; requires CPU_MISS_FIXTURE"]
    fn worker_scheduling_probe() {
        let directory = std::path::PathBuf::from(std::env::var_os("CPU_MISS_FIXTURE").unwrap());
        let bytes = std::fs::read(directory.join("input.f32")).unwrap();
        let real: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let banks: Vec<_> = ["gate", "up", "down"]
            .iter()
            .map(|role| {
                let bytes = std::fs::read(directory.join(format!("{role}.bin"))).unwrap();
                let owner = AlignedHostBuffer::new(bytes.len()).unwrap();
                unsafe {
                    owner.copy_from_slice(0, &bytes);
                }
                owner
            })
            .collect();
        let compact = std::env::var_os("CPU_MISS_SCHED_COMPACT").is_some();
        let thread_counts = if compact { vec![4] } else { vec![4, 5, 6] };
        let spins = if compact {
            vec![256, 4096, 16384, 65536, 262144]
        } else {
            vec![256, 1024, 4096, 16384]
        };
        let offsets = if compact { vec![1] } else { vec![0, 1] };
        let controllers = if compact { vec![150] } else { vec![0, 150] };
        let affinities = if compact {
            vec![true]
        } else {
            vec![false, true]
        };
        for threads in thread_counts {
            for &min_spin in &spins {
                for &offset in &offsets {
                    for &controller_us in &controllers {
                        for &controller_pinned in &affinities {
                            let worker =
                                Worker::new_tuned(threads, false, min_spin, offset).unwrap();
                            let _guard = controller_pinned.then(|| worker.controller_guard());
                            let mut time = Vec::new();
                            for i in 0..288 {
                                std::thread::sleep(std::time::Duration::from_micros(600));
                                let experts = vec![Expert {
                                    slot: 0,
                                    weights: std::array::from_fn(|r| {
                                        Weight::new(Arc::clone(&banks[r]), 0, banks[r].len())
                                            .unwrap()
                                    }),
                                }];
                                let start = Instant::now();
                                let job = worker.submit(experts, DType::Iq2S, real.clone());
                                while start.elapsed().as_micros() < controller_us {
                                    std::hint::spin_loop();
                                }
                                let output = job.wait().unwrap();
                                std::hint::black_box(output);
                                if i >= 32 {
                                    time.push(start.elapsed().as_secs_f64() * 1e6);
                                }
                            }
                            time.sort_by(f64::total_cmp);
                            let stats = worker.stats.lock().unwrap();
                            let mut kernel: Vec<_> =
                                stats.samples.iter().skip(32).map(|s| s.2).collect();
                            kernel.sort_by(f64::total_cmp);
                            println!("SCHED threads={threads} min_spin={min_spin} offset={offset} controller_us={controller_us} controller_pinned={controller_pinned} p50_us={:.2} p90_us={:.2} kernel_p50_us={:.2} kernel_p90_us={:.2}", time[time.len()/2], time[time.len()*9/10], kernel[kernel.len()/2], kernel[kernel.len()*9/10]);
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "captured production activation probe; requires CPU_MISS_FIXTURE"]
    fn activation_isolation_probe() {
        let directory = std::path::PathBuf::from(std::env::var_os("CPU_MISS_FIXTURE").unwrap());
        let real_bytes = std::fs::read(directory.join("input.f32")).unwrap();
        let real: Vec<f32> = real_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(real.len(), 2560);
        let synthetic: Vec<f32> = (0..2560)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let banks: Vec<_> = ["gate", "up", "down"]
            .iter()
            .map(|role| {
                let bytes = std::fs::read(directory.join(format!("{role}.bin"))).unwrap();
                let owner = AlignedHostBuffer::new(bytes.len()).unwrap();
                unsafe {
                    owner.copy_from_slice(0, &bytes);
                }
                owner
            })
            .collect();
        let fixture = || {
            vec![Expert {
                slot: 0,
                weights: std::array::from_fn(|r| {
                    Weight::new(Arc::clone(&banks[r]), 0, banks[r].len()).unwrap()
                }),
            }]
        };
        for threads in 4..=6 {
            let worker = Worker::new(threads, false).unwrap();
            for (name, x) in [("synthetic", &synthetic), ("production", &real)] {
                let mut time = Vec::new();
                for i in 0..544 {
                    let start = Instant::now();
                    let output = worker
                        .submit(fixture(), DType::Iq2S, x.clone())
                        .wait()
                        .unwrap();
                    std::hint::black_box(output);
                    if i >= 32 {
                        time.push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
                time.sort_by(f64::total_cmp);
                println!("ACTIVATION threads={threads} input={name} p50_us={:.2} p90_us={:.2} subnormals={}", time[time.len()/2], time[time.len()*9/10], x.iter().filter(|v| v.is_subnormal()).count());
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "real-weight CPU isolation probe; requires CPU_MISS_MODEL"]
    fn kernel_isolation_probe() {
        use infr_core::loader::WeightSource;
        let path = std::env::var("CPU_MISS_MODEL").expect("CPU_MISS_MODEL");
        let model = infr_gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
        let samples: usize = std::env::var("CPU_MISS_SAMPLES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);
        let threads: usize = std::env::var("CPU_MISS_CORES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        assert!(supported(threads));
        let percentile = |values: &mut Vec<f64>, p: usize| {
            values.sort_by(f64::total_cmp);
            values[(values.len() - 1) * p / 100]
        };
        let x: Vec<f32> = (0..2560)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect();
        for layer in [4, 0] {
            let banks: Vec<_> = ["gate", "up", "down"]
                .iter()
                .map(|role| {
                    let name = format!("blk.{layer}.ffn_{role}_exps.weight");
                    let info = model.tensors().iter().find(|t| t.name == name).unwrap();
                    let stride = info.nbytes / info.shape[2];
                    let bytes = model.tensor_bytes(&name).unwrap();
                    let owner = AlignedHostBuffer::new(bytes.len()).unwrap();
                    unsafe {
                        owner.copy_from_slice(0, bytes);
                    }
                    (owner, stride, info.dtype)
                })
                .collect();
            let kind = banks[0].2;
            let experts = |index: usize, misses: usize, working: usize| {
                (0..misses)
                    .map(|slot| Expert {
                        slot,
                        weights: std::array::from_fn(|role| {
                            let (owner, stride, _) = &banks[role];
                            Weight::new(
                                Arc::clone(owner),
                                ((index + slot) % working) * stride,
                                *stride,
                            )
                            .unwrap()
                        }),
                    })
                    .collect::<Vec<_>>()
            };
            let pool = CpuTaskPool::new(threads);
            let masks = physical_masks();
            let cursor = std::sync::atomic::AtomicUsize::new(0);
            let ready = std::sync::Barrier::new(threads);
            pool.run(threads, &|_| {
                let i = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                pin(masks.get(i).copied());
                ready.wait();
            });
            if std::env::var_os("CPU_MISS_FOUR_TARGETS").is_some() {
                let fixture_input = std::env::var_os("CPU_MISS_FIXTURE")
                    .map(|dir| {
                        std::fs::read(std::path::PathBuf::from(dir).join("input.f32"))
                            .unwrap()
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_else(|| x.clone());
                let mode_count = if std::env::var_os("CPU_MISS_DOWN_TILE").is_some() {
                    3
                } else {
                    11
                };
                let mut timings: Vec<Vec<f64>> = (0..mode_count).map(|_| Vec::new()).collect();
                for batch in 0..12 {
                    for position in 0..mode_count {
                        let mode = (position + batch) % mode_count;
                        for i in 0..64 {
                            let input = experts(i, 1, 64);
                            let start = Instant::now();
                            let output = if mode_count == 3 && mode == 1 {
                                compute_chunked_down_tile::<
                                    false,
                                    true,
                                    32,
                                    128,
                                    false,
                                    false,
                                    false,
                                    false,
                                    4,
                                >(
                                    &pool, &input, kind, &fixture_input, None
                                )
                            } else if mode_count == 3 && mode == 2 {
                                compute_chunked_down_tile::<
                                    false,
                                    true,
                                    32,
                                    128,
                                    false,
                                    false,
                                    false,
                                    false,
                                    8,
                                >(
                                    &pool, &input, kind, &fixture_input, None
                                )
                            } else if mode == 10 {
                                compute_chunked_impl::<true, true, 32, 128, true, false, true, false>(
                                    &pool,
                                    &input,
                                    kind,
                                    &fixture_input,
                                    None,
                                )
                            } else if mode == 9 {
                                compute_chunked_impl::<true, true, 32, 128, false, false, true, false>(
                                    &pool,
                                    &input,
                                    kind,
                                    &fixture_input,
                                    None,
                                )
                            } else if mode == 8 {
                                compute_chunked_impl::<false, true, 32, 128, true, false, true, false>(
                                    &pool,
                                    &input,
                                    kind,
                                    &fixture_input,
                                    None,
                                )
                            } else if mode == 7 {
                                compute_chunked_impl::<false, true, 32, 128, false, false, true, true>(
                                    &pool,
                                    &input,
                                    kind,
                                    &fixture_input,
                                    None,
                                )
                            } else if mode == 6 {
                                compute_chunked_impl::<
                                    false,
                                    true,
                                    32,
                                    128,
                                    false,
                                    false,
                                    false,
                                    true,
                                >(
                                    &pool, &input, kind, &fixture_input, None
                                )
                            } else if mode == 5 {
                                compute_chunked_impl::<
                                    false,
                                    true,
                                    32,
                                    128,
                                    false,
                                    false,
                                    true,
                                    false,
                                >(
                                    &pool, &input, kind, &fixture_input, None
                                )
                            } else if mode == 4 {
                                compute_single_dispatch(&pool, &input[0], kind, &fixture_input)
                            } else {
                                compute_options(
                                    &pool,
                                    &input,
                                    kind,
                                    &fixture_input,
                                    None,
                                    mode & 1 != 0,
                                    mode & 2 != 0,
                                    false,
                                )
                            };
                            std::hint::black_box(output);
                            if batch > 1 {
                                timings[mode].push(start.elapsed().as_secs_f64() * 1e6);
                            }
                        }
                    }
                }
                for (mode, time) in timings.iter_mut().enumerate() {
                    if mode_count == 3 {
                        println!("DOWN_TILE layer={layer} threads={threads} rows={} samples={} p50_us={:.2} p90_us={:.2}",
                            if mode == 0 { if kind == DType::Iq2S { 2 } else { 1 } } else if mode == 1 { 4 } else { 8 },
                            time.len(), percentile(time, 50), percentile(time, 90));
                    } else {
                        println!("FOUR_TARGETS layer={layer} threads={threads} mode={mode} split={} flush={} single_dispatch={} grouped={} dual={} full_lut={} gu_paired={} samples={} p50_us={:.2} p90_us={:.2}",
                            mode < 4 && mode & 1 != 0, mode < 4 && mode & 2 != 0, mode == 4, mode == 5 || mode >= 7, mode == 6 || mode == 7, mode == 8 || mode == 10, mode == 9 || mode == 10, time.len(), percentile(time, 50), percentile(time, 90));
                    }
                }
                continue;
            }
            if std::env::var_os("CPU_MISS_CHUNKS").is_some() {
                let mut timings: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
                for batch in 0..12 {
                    for position in 0..5 {
                        let mode = (position + batch) % 5;
                        for i in 0..64 {
                            let input = experts(i, 1, 64);
                            let start = Instant::now();
                            let output = match mode {
                                1 => compute_chunked::<false, true, 64, 128>(
                                    &pool, &input, kind, &x, None,
                                ),
                                2 => compute_chunked::<false, true, 64, 256>(
                                    &pool, &input, kind, &x, None,
                                ),
                                3 => compute_chunked::<false, true, 80, 320>(
                                    &pool, &input, kind, &x, None,
                                ),
                                4 => compute_chunked::<false, true, 160, 640>(
                                    &pool, &input, kind, &x, None,
                                ),
                                _ => compute_chunked::<false, true, 32, 128>(
                                    &pool, &input, kind, &x, None,
                                ),
                            };
                            if batch >= 2 {
                                timings[mode].push(start.elapsed().as_secs_f64() * 1e6);
                            }
                            std::hint::black_box(output);
                        }
                    }
                }
                for (mode, &(gu, down)) in [(32, 128), (64, 128), (64, 256), (80, 320), (160, 640)]
                    .iter()
                    .enumerate()
                {
                    println!("CHUNKS layer={layer} threads={threads} gu={gu} down={down} samples={} p50_us={:.2} p90_us={:.2}",
                        timings[mode].len(), percentile(&mut timings[mode], 50), percentile(&mut timings[mode], 90));
                }
                pin(Some(masks.iter().copied().fold(0, |a, b| a | b)));
                continue;
            }
            if std::env::var_os("CPU_MISS_PAIRING").is_some() {
                for misses in [1, 2] {
                    if std::env::var_os("CPU_MISS_PAIRING_BALANCED").is_some() {
                        let modes = [(false, false), (false, true), (true, false), (true, true)];
                        let mut timing: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
                        for batch in 0..18 {
                            for position in 0..4 {
                                let mode = (position + batch) % 4;
                                for i in 0..64 {
                                    let input = experts(i, misses, 64);
                                    let start = Instant::now();
                                    let output = match modes[mode] {
                                        (true, true) => compute_tiled::<true, true>(
                                            &pool, &input, kind, &x, None,
                                        ),
                                        (false, true) => compute_tiled::<false, true>(
                                            &pool, &input, kind, &x, None,
                                        ),
                                        (true, false) => compute_tiled::<true, false>(
                                            &pool, &input, kind, &x, None,
                                        ),
                                        _ => compute_tiled::<false, false>(
                                            &pool, &input, kind, &x, None,
                                        ),
                                    };
                                    if batch >= 2 {
                                        timing[mode].push(start.elapsed().as_secs_f64() * 1e6);
                                    }
                                    std::hint::black_box(output);
                                }
                            }
                        }
                        for (mode, &(gu_paired, down_paired)) in modes.iter().enumerate() {
                            println!("BALANCED layer={layer} threads={threads} misses={misses} gu_paired={gu_paired} down_paired={down_paired} samples={} p50_us={:.2} p90_us={:.2}", timing[mode].len(), percentile(&mut timing[mode], 50), percentile(&mut timing[mode], 90));
                        }
                        continue;
                    }
                    for (gu_paired, down_paired) in
                        [(false, false), (true, true), (false, true), (true, false)]
                    {
                        let mut time = Vec::new();
                        for i in 0..samples + 64 {
                            let input = experts(i, misses, 64);
                            let start = Instant::now();
                            let output = match (gu_paired, down_paired) {
                                (true, true) => {
                                    compute_tiled::<true, true>(&pool, &input, kind, &x, None)
                                }
                                (false, true) => {
                                    compute_tiled::<false, true>(&pool, &input, kind, &x, None)
                                }
                                (true, false) => {
                                    compute_tiled::<true, false>(&pool, &input, kind, &x, None)
                                }
                                _ => compute_tiled::<false, false>(&pool, &input, kind, &x, None),
                            };
                            if i >= 64 {
                                time.push(start.elapsed().as_secs_f64() * 1e6);
                            }
                            std::hint::black_box(output);
                        }
                        println!("PAIR layer={layer} threads={threads} misses={misses} gu_paired={gu_paired} down_paired={down_paired} p50_us={:.2} p90_us={:.2}", percentile(&mut time, 50), percentile(&mut time, 90));
                    }
                }
                pin(Some(masks.iter().copied().fold(0, |a, b| a | b)));
                continue;
            }
            for working in [64, 512] {
                for misses in [1, 2] {
                    for gap_us in [0, 600] {
                        for asynchronous in [false, true] {
                            let worker = asynchronous.then(|| Worker::new(threads, false).unwrap());
                            let mut total = Vec::new();
                            for i in 0..samples + 32 {
                                if gap_us != 0 {
                                    let gap = Instant::now();
                                    while gap.elapsed().as_micros() < gap_us {
                                        std::hint::spin_loop();
                                    }
                                }
                                let input = experts(i, misses, working);
                                let start = Instant::now();
                                let output = if let Some(worker) = &worker {
                                    worker.submit(input, kind, x.clone()).wait().unwrap()
                                } else {
                                    compute(&pool, &input, kind, &x, None)
                                };
                                if i >= 32 {
                                    total.push(start.elapsed().as_secs_f64() * 1e6);
                                }
                                std::hint::black_box(output);
                            }
                            let mut queue = Vec::new();
                            let mut kernel = Vec::new();
                            if let Some(worker) = &worker {
                                let stats = worker.stats.lock().unwrap();
                                for s in stats.samples.iter().skip(32) {
                                    queue.push(s.1);
                                    kernel.push(s.2);
                                }
                            }
                            let q = if queue.is_empty() {
                                0.0
                            } else {
                                percentile(&mut queue, 50)
                            };
                            let k = if kernel.is_empty() {
                                0.0
                            } else {
                                percentile(&mut kernel, 50)
                            };
                            println!("PROBE layer={layer} threads={threads} working={working} misses={misses} gap_us={gap_us} async={asynchronous} p50_us={:.2} p90_us={:.2} queue_p50_us={q:.2} kernel_p50_us={k:.2}", percentile(&mut total, 50), percentile(&mut total, 90));
                        }
                    }
                }
            }
            // The test harness caller must not retain a restricted affinity for later tests.
            pin(Some(masks.iter().copied().fold(0, |a, b| a | b)));
        }
    }
    #[test]
    fn policy_falls_back_outside_the_validated_envelope() {
        for threads in 0..=8 {
            for avx in [false, true] {
                for fma in [false, true] {
                    assert_eq!(eligible(threads, avx, fma), threads != 0 && avx && fma);
                }
            }
        }
        for rows in 1..=3 {
            for misses in 0..=10 {
                assert_eq!(
                    geometry(
                        rows,
                        misses,
                        2560,
                        640,
                        DType::Iq2S,
                        DType::Iq2S,
                        DType::Iq4Nl
                    ),
                    rows == 1 && (1..=3).contains(&misses)
                );
            }
        }
        assert!(!decode_layer(0));
        for maximum in 0..=4 {
            for misses in 0..=10 {
                assert_eq!(
                    within_limit(misses, maximum),
                    (1..=3).contains(&maximum) && (1..=maximum).contains(&misses)
                );
            }
        }
        assert!(decode_layer(1));
        assert!(decode_layer(47));
        assert!(!geometry(
            1,
            1,
            2560,
            640,
            DType::Iq3S,
            DType::Iq2S,
            DType::Iq4Nl
        ));
        assert!(!geometry(
            1,
            1,
            2560,
            640,
            DType::Iq2S,
            DType::Iq2S,
            DType::Q4K
        ));
        assert!(!geometry(
            1,
            1,
            256,
            640,
            DType::Iq2S,
            DType::Iq2S,
            DType::Iq4Nl
        ));
    }

    #[cfg(target_arch = "x86_64")]
    fn fixture(kind: DType, slot: usize) -> Expert {
        let mut seed = 91537u32 + slot as u32;
        let weights = std::array::from_fn(|role| {
            let (blocks, size) = if role == 2 {
                (20 * 2560, 18)
            } else {
                (10 * 640, if kind == DType::Iq2S { 82 } else { 110 })
            };
            let mut bytes = vec![0; blocks * size];
            for block in bytes.chunks_mut(size) {
                for v in block.iter_mut() {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    *v = seed as u8;
                }
                block[..2].copy_from_slice(&0x1800u16.to_le_bytes());
            }
            let owner = AlignedHostBuffer::new(bytes.len() + 3).unwrap();
            unsafe {
                owner.copy_from_slice(3, &bytes);
            }
            Weight::new(owner, 3, bytes.len()).unwrap()
        });
        Expert { slot, weights }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn one_to_three_misses_match_independent_row_decode() {
        if !supported(4) {
            return;
        }
        for kind in [DType::Iq2S, DType::Iq3S] {
            let x: Vec<_> = (0..2560)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.003)
                .collect();
            let fixtures = [fixture(kind, 2), fixture(kind, 7), fixture(kind, 9)];
            let mut expected = Vec::new();
            for e in &fixtures {
                let stride = if kind == DType::Iq2S { 820 } else { 1100 };
                let a: Vec<_> = (0..640)
                    .map(|row| unsafe {
                        let g = e.weights[0].bytes().as_ptr().add(row * stride);
                        let u = e.weights[1].bytes().as_ptr().add(row * stride);
                        let v = if kind == DType::Iq2S {
                            kernels::tile::<2, 2>([g, u], &x)
                        } else {
                            kernels::tile::<3, 2>([g, u], &x)
                        };
                        v[0] / (1.0 + (-v[0]).exp()) * v[1]
                    })
                    .collect();
                expected.extend((0..2560).map(|row| unsafe {
                    kernels::tile::<4, 1>([e.weights[2].bytes().as_ptr().add(row * 360)], &a)[0]
                }));
            }
            for threads in [1, 2, 4, 5, 6] {
                if !supported(threads) {
                    continue;
                }
                let worker = Worker::new(threads, false).unwrap();
                for misses in 1..=3 {
                    let experts = [2, 7, 9]
                        .into_iter()
                        .take(misses)
                        .map(|slot| fixture(kind, slot))
                        .collect();
                    let output = worker.submit(experts, kind, x.clone()).wait().unwrap();
                    assert_eq!(output.slots, [2, 7, 9][..misses]);
                    assert_eq!(
                        output
                            .values
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        expected[..misses * 2560]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                    let experts: Vec<_> = [2, 7, 9]
                        .into_iter()
                        .take(misses)
                        .map(|slot| fixture(kind, slot))
                        .collect();
                    if misses == 1 {
                        let fused = compute_single_dispatch(
                            &CpuTaskPool::new(threads),
                            &experts[0],
                            kind,
                            &x,
                        );
                        assert_eq!(
                            fused.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            expected[..2560]
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>()
                        );
                    }
                    let paired = compute_tiled::<true, true>(
                        &CpuTaskPool::new(threads),
                        &experts,
                        kind,
                        &x,
                        None,
                    );
                    assert_eq!(
                        paired
                            .values
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        expected[..misses * 2560]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                }
                // Dropping an unfinished result joins its target writes; the pool remains reusable.
                drop(worker.submit(vec![fixture(kind, 2)], kind, x.clone()));
                assert_eq!(
                    worker
                        .submit(vec![fixture(kind, 2)], kind, vec![0.0; 2560])
                        .wait()
                        .unwrap()
                        .values,
                    vec![0.0; 2560]
                );
            }
        }
    }

    #[test]
    fn host_ranges_reject_overflow_and_out_of_bounds() {
        let owner = AlignedHostBuffer::new(16).unwrap();
        assert!(Weight::new(Arc::clone(&owner), 8, 8).is_some());
        assert!(Weight::new(Arc::clone(&owner), 9, 8).is_none());
        assert!(Weight::new(owner, usize::MAX, 1).is_none());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn token_burst_parking_preserves_compute_and_reuses_worker() {
        if !supported(4) {
            return;
        }
        for (threads, polls) in (4..=6)
            .filter(|&threads| supported(threads))
            .flat_map(|threads| [0, 32768].map(|polls| (threads, polls)))
        {
            let worker =
                Worker::new_tuned_with_poll(threads, false, 262144, 1, false, polls).unwrap();
            worker.finish_burst().unwrap();
            let x: Vec<_> = (0..2560).map(|i| (i % 31) as f32 * 0.001).collect();
            let first = worker
                .submit(vec![fixture(DType::Iq2S, 2)], DType::Iq2S, x.clone())
                .wait()
                .unwrap();
            let expected: Vec<_> = first.values.iter().map(|v| v.to_bits()).collect();
            for _ in 0..16 {
                worker.finish_burst().unwrap();
                let output = worker
                    .submit(vec![fixture(DType::Iq2S, 2)], DType::Iq2S, x.clone())
                    .wait()
                    .unwrap();
                assert_eq!(
                    output
                        .values
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                );
            }
            worker.finish_burst().unwrap();
        }
    }

    #[test]
    fn deferred_cpu_miss_promotion_drop_waits_for_the_producer() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_done = Arc::clone(&done);
        let producer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(5));
            worker_done.store(true, std::sync::atomic::Ordering::Release);
            sender.send(Ok(())).unwrap();
        });
        drop(PromotionJob(Some(receiver)));
        assert!(done.load(std::sync::atomic::Ordering::Acquire));
        producer.join().unwrap();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn one_miss_chunk_candidates_preserve_outputs() {
        if !supported(4) {
            return;
        }
        let pool = CpuTaskPool::new(4);
        let x: Vec<_> = (0..2560).map(|i| (i % 41) as f32 * 0.001 - 0.02).collect();
        for kind in [DType::Iq2S, DType::Iq3S] {
            let experts = [fixture(kind, 3)];
            let expected = compute_tiled::<false, true>(&pool, &experts, kind, &x, None).values;
            let actual = [
                compute_chunked::<false, true, 64, 128>(&pool, &experts, kind, &x, None),
                compute_chunked::<false, true, 64, 256>(&pool, &experts, kind, &x, None),
                compute_chunked::<false, true, 80, 320>(&pool, &experts, kind, &x, None),
                compute_chunked::<false, true, 160, 640>(&pool, &experts, kind, &x, None),
            ];
            for output in actual {
                assert_eq!(output.values, expected);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "requires a mapped Vulkan GPU; run explicitly"]
    fn deferred_cpu_miss_promotion_copies_after_compute_and_survives_drop() {
        use infr_core::backend::{Backend, Buffer};
        if !supported(4) {
            return;
        }
        let be_ = crate::VulkanBackend::new().unwrap();
        let worker = Worker::new(4, true).unwrap();
        for kind in [DType::Iq2S, DType::Iq3S] {
            let mut output = worker
                .submit(vec![fixture(kind, 3)], kind, vec![0.1; 2560])
                .wait()
                .unwrap();
            let targets: Vec<Arc<dyn Buffer>> = output.experts[0]
                .weights
                .iter_mut()
                .map(|weight| {
                    let (buffer, _) = be_.alloc_mapped_arena_bda(weight.len).unwrap();
                    let buffer: Arc<dyn Buffer> = Arc::from(buffer);
                    be_.upload(buffer.as_ref(), &vec![0; weight.len]).unwrap();
                    weight
                        .set_target(
                            DeviceTransferTarget::new(Arc::clone(&buffer), 0, weight.len).unwrap(),
                        )
                        .unwrap();
                    buffer
                })
                .collect();
            for target in &targets {
                let mut bytes = vec![1; target.len_bytes()];
                be_.download(target.as_ref(), &mut bytes).unwrap();
                assert!(bytes.iter().all(|&v| v == 0));
            }
            // Cancellation still drains the copy before the cache slot can be reused.
            drop(worker.promote(output.experts));
            let source = fixture(kind, 3);
            for (role, target) in targets.iter().enumerate() {
                let mut bytes = vec![0; target.len_bytes()];
                be_.download(target.as_ref(), &mut bytes).unwrap();
                assert_eq!(bytes, source.weights[role].bytes());
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "requires a mapped Vulkan GPU; run explicitly"]
    fn cpu_push_promotes_exact_bytes_and_gpu_ffn_matches_cpu() {
        use infr_core::backend::{Backend, Buffer, BufferUsage};
        if !supported(4) {
            return;
        }
        let be_ = crate::VulkanBackend::new().expect("Vulkan GPU");
        for kind in [DType::Iq2S, DType::Iq3S] {
            for threads in (4..=6).filter(|&threads| supported(threads)) {
                for misses in 1..=3 {
                    let worker = Worker::new(threads, true).unwrap();
                    let mut experts: Vec<_> = [2, 5, 7]
                        .into_iter()
                        .take(misses)
                        .map(|s| fixture(kind, s))
                        .collect();
                    let targets: Vec<[Arc<dyn Buffer>; 3]> = experts
                        .iter_mut()
                        .map(|expert| {
                            std::array::from_fn(|r| {
                                let w = &mut expert.weights[r];
                                let (buffer, _) = be_.alloc_mapped_arena_bda(w.len).unwrap();
                                let buffer: Arc<dyn Buffer> = Arc::from(buffer);
                                w.set_target(
                                    DeviceTransferTarget::new(Arc::clone(&buffer), 0, w.len)
                                        .unwrap(),
                                )
                                .unwrap();
                                buffer
                            })
                        })
                        .collect();
                    let x: Vec<_> = (0..2560)
                        .map(|i| ((i % 31) as f32 - 15.0) * 0.003)
                        .collect();
                    let output = worker.submit(experts, kind, x.clone()).wait().unwrap();
                    worker.promote(output.experts).wait().unwrap();
                    for (e, buffers) in targets.iter().enumerate() {
                        let source = fixture(kind, [2, 5, 7][e]);
                        for (r, buffer) in buffers.iter().enumerate() {
                            let mut bytes = vec![0u8; buffer.len_bytes()];
                            be_.download(buffer.as_ref(), &mut bytes).unwrap();
                            assert_eq!(
                                bytes,
                                source.weights[r].bytes(),
                                "push byte parity threads={threads} miss={e} role={r}"
                            );
                        }
                        let xb = be_.alloc(2560 * 4, BufferUsage::Activations).unwrap();
                        let g = be_.alloc(640 * 4, BufferUsage::Activations).unwrap();
                        let u = be_.alloc(640 * 4, BufferUsage::Activations).unwrap();
                        let a = be_.alloc(640 * 4, BufferUsage::Activations).unwrap();
                        let y = be_.alloc(2560 * 4, BufferUsage::Activations).unwrap();
                        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
                        let rc = be_.recorder().unwrap();
                        rc.seed_barrier();
                        rc.arena_stream_barrier();
                        rc.linear_native(
                            kind,
                            buffers[0].as_ref(),
                            0,
                            xb.as_ref(),
                            g.as_ref(),
                            1,
                            2560,
                            640,
                        );
                        rc.linear_native(
                            kind,
                            buffers[1].as_ref(),
                            0,
                            xb.as_ref(),
                            u.as_ref(),
                            1,
                            2560,
                            640,
                        );
                        rc.silu_mul(g.as_ref(), u.as_ref(), a.as_ref(), 640, None);
                        rc.linear_native(
                            DType::Iq4Nl,
                            buffers[2].as_ref(),
                            0,
                            a.as_ref(),
                            y.as_ref(),
                            1,
                            640,
                            2560,
                        );
                        rc.finish().unwrap();
                        let mut actual = vec![0f32; 2560];
                        be_.download(y.as_ref(), bytemuck::cast_slice_mut(&mut actual))
                            .unwrap();
                        for (&gpu, &cpu) in
                            actual.iter().zip(&output.values[e * 2560..(e + 1) * 2560])
                        {
                            assert!(
                                gpu.is_finite()
                                    && cpu.is_finite()
                                    && (gpu - cpu).abs() <= 1e-5 + gpu.abs() * 1e-4,
                                "FFN parity {kind:?} {threads} threads cpu={cpu} gpu={gpu}"
                            );
                        }
                    }
                }
            }
        }
    }
}
