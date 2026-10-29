//! Ignored module experiment, never part of the production queue scheduler.
//! Child of recorder::owned_indirect_tests::fused::splitk_hit_v2.
//! No external SPIR-V, event, fused FFN, ownership transfer, or runtime integration.
//!
//! lib.rs requests queue 1 only under cfg(test) and explicit opt-in.
//! Explicit opt-in: MOE4ALL_TEST_AUX_COMPUTE_QUEUE=1 and MOE4ALL_TEST_DEVICE.
//! Run only this ignored lib test with --test-threads=1 and a serialized idle GPU.
//! MOE4ALL_DUALQUEUE_REPEATS=16 amortizes TWO host submits across 16 complete
//! stages. Per-stage Ready/H/Copy timeline edges remain included. Detail queries
//! require a single stage. Neither mode proves a production end-to-end benefit.
//!
//! Fixture minimum imported banks=3, rows=1/2/4, Q5+Q8shared or Flash IQ2/IQ3+IQ4NL.
//! Optional filters: MOE4ALL_DUALQUEUE_FORMAT=q5k|iq2s|iq3s|all,
//! MOE4ALL_DUALQUEUE_ROWS=1|2|4|all, MOE4ALL_DUALQUEUE_MODE=allhit|one_miss|all.
//! Caller must route any future build/kernel cache and logs per disk_usage_rule.md.
//!
//! main queue 0: [prefix -> Ready] [H]
//!               [wait Copy -> seed barrier joins H/Copy -> M] -> main fence
//! aux  queue 1: [wait Ready -> WHOLE copy -> Copy] -> auxiliary fence
//! All four batches are submitted in TWO vkQueueSubmit calls, main first.
//! Ready and Copy use separate timelines. A binary future
//! wait would be invalid here. Prefix makes initialized host jobs/source visible
//! before both queues; no host operation waits between H and copy submissions.
//!
//! Core H and copy are two retained Recorder recordings, each ONE segment.
//! Prefix and M/join are separate retained recordings, also single segment.
//! All command buffers use the existing MAIN FAMILY pool, valid on queues 0/1.
//! All buffer/import/ownership paths remain unchanged: queue FAMILY is identical.
//! In particular aux reads journal/jobs + imported BDA source, writes the same
//! mapped weights allocation; H only reads disjoint resident expert ranges.
//!
//! GPU module delta is main-queue begin->end, INCLUDING Ready + join + M.
//! aux timestamps are used ONLY for aux copy duration, never subtracted from
//! main timestamps. wall includes both submit calls and both blocking fence waits.
//! Startup recording/pipeline allocation, host resets/checks/query reads are
//! excluded and reported separately. Per-fixture frames are not token runtime.
//!
//! Raw records are evidence; speedup does not prove physical hardware overlap.
//! Spec: https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueSubmit.html
//! Timing: https://docs.vulkan.org/samples/latest/samples/api/timestamp_queries/README.html
use super::*;

use std::sync::Mutex;
use std::time::Instant;

type ProbeResult<T> = std::result::Result<T, String>;
const WARMUP_MS: u64 = 120;
const SAMPLES: usize = 8;
const QUERY_COUNT: u32 = 7;
// Every use of queue 1 in this module must hold this guard, including draining.
static AUX_COMPUTE_SUBMIT: Mutex<()> = Mutex::new(());

struct Resources<'a> {
    device: &'a ash::Device,
    ready: vk::Semaphore,
    copy_done: vk::Semaphore,
    main_fence: vk::Fence,
    aux_fence: vk::Fence,
    query: vk::QueryPool,
    bits: u32,
    period_ns: f64,
    detail: bool,
    repeats: usize,
    aux_first: bool,
}

impl<'a> Resources<'a> {
    fn new(backend: &'a VulkanBackend) -> ProbeResult<Self> {
        let device = &backend.shared.device;
        let mut r = Self {
            device,
            ready: vk::Semaphore::null(),
            copy_done: vk::Semaphore::null(),
            main_fence: vk::Fence::null(),
            aux_fence: vk::Fence::null(),
            query: vk::QueryPool::null(),
            bits: backend.shared.submit_timestamp_valid_bits,
            period_ns: backend.shared.submit_timestamp_period_ns as f64,
            detail: std::env::var("MOE4ALL_DUALQUEUE_DETAIL").is_ok_and(|v| v == "1"),
            repeats: std::env::var("MOE4ALL_DUALQUEUE_REPEATS")
                .map_or(1, |v| v.parse::<usize>().expect("positive repeat count")),
            aux_first: std::env::var("MOE4ALL_DUALQUEUE_AUX_FIRST").is_ok_and(|v| v == "1"),
        };
        assert!((1..=64).contains(&r.repeats));
        assert!(
            !r.detail || r.repeats == 1,
            "detail timestamps require one stage"
        );
        if r.bits == 0 || !r.period_ns.is_finite() || r.period_ns <= 0.0 {
            return Err("DECLINE: main family has no usable timestamp support".into());
        }
        let timeline = || {
            let mut kind = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            unsafe {
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                    None,
                )
            }
            .map_err(|e| format!("create timeline: {e}"))
        };
        r.ready = timeline()?;
        r.copy_done = timeline()?;
        r.main_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| format!("create main fence: {e}"))?;
        r.aux_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| format!("create aux fence: {e}"))?;
        r.query = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(QUERY_COUNT),
                None,
            )
        }
        .map_err(|e| format!("create query pool: {e}"))?;
        Ok(r)
    }

    fn begin(&self, rec: &Recorder<'_>) {
        unsafe {
            self.device
                .cmd_reset_query_pool(rec.cmd, self.query, 0, QUERY_COUNT)
        };
        self.stamp(rec, 0, vk::PipelineStageFlags::TOP_OF_PIPE);
    }

    fn stamp(&self, rec: &Recorder<'_>, index: u32, stage: vk::PipelineStageFlags) {
        if !self.detail && !matches!(index, 0 | 4) {
            return;
        }
        unsafe {
            self.device
                .cmd_write_timestamp(rec.cmd, stage, self.query, index)
        };
    }

    fn tick(&self, index: u32) -> ProbeResult<u64> {
        let mut tick = [0u64];
        unsafe {
            self.device.get_query_pool_results(
                self.query,
                index,
                &mut tick,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .map_err(|e| format!("query {index}: {e}"))?;
        Ok(tick[0])
    }

    fn delta(&self, start: u64, end: u64) -> f64 {
        let mask = if self.bits >= 64 {
            u64::MAX
        } else {
            (1u64 << self.bits) - 1
        };
        (end.wrapping_sub(start) & mask) as f64 * self.period_ns / 1000.0
    }
}

impl Drop for Resources<'_> {
    fn drop(&mut self) {
        // Pending guard drains BOTH queues before any owner can reach this drop.
        unsafe {
            if self.query != vk::QueryPool::null() {
                self.device.destroy_query_pool(self.query, None);
            }
            for fence in [self.main_fence, self.aux_fence] {
                if fence != vk::Fence::null() {
                    self.device.destroy_fence(fence, None);
                }
            }
            for semaphore in [self.ready, self.copy_done] {
                if semaphore != vk::Semaphore::null() {
                    self.device.destroy_semaphore(semaphore, None);
                }
            }
        }
    }
}

// Owns no recording: borrows the frame so recordings, fixtures, imported host
// owner, semaphores, query pool and fences cannot drop while either queue is live.
struct Pending<'a, 'd> {
    r: &'a Resources<'d>,
    main_live: bool,
    aux_live: bool,
    unblock_copy: Option<u64>,
    unblock_ready: Option<u64>,
}

impl Pending<'_, '_> {
    fn wait(&mut self) -> ProbeResult<()> {
        // No polling, get_fence_status loop, spin, or queue_wait_idle in this path.
        if self.aux_live {
            unsafe {
                self.r
                    .device
                    .wait_for_fences(&[self.r.aux_fence], true, u64::MAX)
            }
            .map_err(|e| format!("aux fence: {e}"))?;
            self.aux_live = false;
        }
        if self.main_live {
            unsafe {
                self.r
                    .device
                    .wait_for_fences(&[self.r.main_fence], true, u64::MAX)
            }
            .map_err(|e| format!("main fence: {e}"))?;
            self.main_live = false;
        }
        Ok(())
    }

    fn terminal(result: std::result::Result<(), vk::Result>, context: &str) {
        match result {
            Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => {}
            Err(e) => {
                // Cannot release pending Vulkan objects after an unproven drain.
                eprintln!("DUALQUEUE_FATAL context={context} error={e}; refusing pending drop");
                std::process::abort();
            }
        }
    }
}

impl Drop for Pending<'_, '_> {
    fn drop(&mut self) {
        if self.aux_live && !self.main_live {
            if let Some(value) = self.unblock_ready.take() {
                // Main submit failed without enqueue. Release ONLY for draining
                // the already queued auxiliary jobs; this sample is rejected.
                Self::terminal(
                    unsafe {
                        self.r.device.signal_semaphore(
                            &vk::SemaphoreSignalInfo::default()
                                .semaphore(self.r.ready)
                                .value(value),
                        )
                    },
                    "cancel missing main submission",
                );
            }
        }
        if self.main_live && !self.aux_live {
            if let Some(value) = self.unblock_copy.take() {
                // Aux submit failed before enqueue (OOM is state preserving).
                // Release main's future wait for cleanup ONLY; reject the sample.
                // Never host-signal a value while an aux signal is still pending.
                Self::terminal(
                    unsafe {
                        self.r.device.signal_semaphore(
                            &vk::SemaphoreSignalInfo::default()
                                .semaphore(self.r.copy_done)
                                .value(value),
                        )
                    },
                    "cancel missing auxiliary submission",
                );
            }
        }
        if self.aux_live {
            Self::terminal(
                unsafe {
                    self.r
                        .device
                        .wait_for_fences(&[self.r.aux_fence], true, u64::MAX)
                },
                "drop auxiliary fence",
            );
        }
        if self.main_live {
            Self::terminal(
                unsafe {
                    self.r
                        .device
                        .wait_for_fences(&[self.r.main_fence], true, u64::MAX)
                },
                "drop main fence",
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    gpu_module_us: f64,
    wall_us: f64,
    main_submit_us: f64,
    aux_submit_us: f64,
    // These are observed intervals, not isolated costs to subtract.
    ready_startup_us: Option<f64>,
    h_us: Option<f64>,
    post_h_join_wait_us: Option<f64>,
    m_and_end_us: Option<f64>,
    aux_copy_us: Option<f64>,
}

struct Frame<'a> {
    backend: &'a VulkanBackend,
    // Explicit fixture + source borrows retain every referenced buffer and host owner.
    _fixture: &'a SplitFixture,
    _source: &'a Sources,
    prefix: RecordedCmd,
    hit: RecordedCmd,
    copy: RecordedCmd,
    tail: RecordedCmd,
    baseline: RecordedCmd,
    begin: RecordedCmd,
    end: RecordedCmd,
    aux: vk::Queue,
    serial: u64,
    // Destroy query/fences/semaphores AFTER retained recordings have dropped.
    r: Resources<'a>,
}

fn recording(
    backend: &VulkanBackend,
    body: impl FnOnce(&Recorder<'_>),
) -> ProbeResult<RecordedCmd> {
    let rec = backend.recorder_persistent().map_err(|e| e.to_string())?;
    if rec.no_barrier || rec.full_barrier || rec.prof_ops {
        return Err("DECLINE: diagnostic barrier/profiler override is active".into());
    }
    body(&rec);
    let command = rec.finish_record().map_err(|e| e.to_string())?;
    if command.segments.len() != 1 {
        return Err("DECLINE: frame requires one retained segment per recording".into());
    }
    Ok(command)
}

fn handle(recording: &RecordedCmd) -> vk::CommandBuffer {
    let [segment] = &recording.segments[..] else {
        unreachable!("checked single segment")
    };
    segment.cmd
}

fn native_hit(ff: &FusedFixture, rec: &Recorder<'_>) {
    ff.projections(rec, 1, 0);
    ff.activation(rec);
    ff.down(rec, 1, 0);
}

fn native_miss(ff: &FusedFixture, rec: &Recorder<'_>) {
    if ff.inner.job_count != 0 {
        ff.projections(rec, 2, 0);
        ff.activation(rec);
        ff.down(rec, 2, 0);
    }
}

impl<'a> Frame<'a> {
    fn new(
        backend: &'a VulkanBackend,
        fixture: &'a SplitFixture,
        source: &'a Sources,
    ) -> ProbeResult<Self> {
        if std::env::var("MOE4ALL_TEST_AUX_COMPUTE_QUEUE")
            .ok()
            .as_deref()
            != Some("1")
        {
            return Err("DECLINE: device-init auxiliary queue opt-in is absent".into());
        }
        let family = backend.shared.queue_family_index;
        let families = unsafe {
            backend
                .shared
                .instance
                .get_physical_device_queue_family_properties(backend.shared.physical_device)
        };
        let properties = &families[family as usize];
        if properties.queue_count < 2 || !properties.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            return Err("DECLINE: main family does not advertise two compute queues".into());
        }
        // The env is never changed by this module. cfg(test) init guarantees index 1
        // was requested at device creation; never retrieve an uncreated queue.
        let aux = unsafe { backend.shared.device.get_device_queue(family, 1) };
        if aux == vk::Queue::null() || aux == backend.shared.queue {
            return Err("DECLINE: auxiliary queue is null or aliases queue 0".into());
        }
        eprintln!(
            "DUALQUEUE_CAP main_family={family} main_index=0 aux_family={family} aux_index=1 advertised={} flags={:?} same_family=true ownership_transfer=false sharing_import_edits=false hardware_overlap_verified=false",
            properties.queue_count, properties.queue_flags
        );
        let r = Resources::new(backend)?;
        let ff = &fixture.ff;
        let prefix = recording(backend, |rec| {
            // Real source-ready producer, included in the measured module.
            rec.seed_barrier();
        })?;
        let begin = recording(backend, |rec| {
            rec.seed_barrier();
            r.begin(rec);
        })?;
        let end = recording(backend, |rec| {
            rec.publish_host_reads();
            r.stamp(rec, 4, vk::PipelineStageFlags::BOTTOM_OF_PIPE);
        })?;
        let hit = recording(backend, |rec| {
            r.stamp(rec, 1, vk::PipelineStageFlags::TOP_OF_PIPE);
            rec.seed_barrier();
            native_hit(ff, rec);
            r.stamp(rec, 2, vk::PipelineStageFlags::BOTTOM_OF_PIPE);
        })?;
        let copy = recording(backend, |rec| {
            r.stamp(rec, 5, vk::PipelineStageFlags::TOP_OF_PIPE);
            rec.seed_barrier();
            ff.inner.cu_copy(rec); // Builtin whole-payload local1024/groups8.
            r.stamp(rec, 6, vk::PipelineStageFlags::BOTTOM_OF_PIPE);
        })?;
        let tail = recording(backend, |rec| {
            // Copy's semaphore plus this same-queue H dependency join both arms.
            rec.seed_barrier();
            r.stamp(rec, 3, vk::PipelineStageFlags::COMPUTE_SHADER);
            native_miss(ff, rec);
        })?;
        let baseline = recording(backend, |rec| {
            rec.seed_barrier();
            r.begin(rec);
            // Match existing native baseline: whole copy independent of H GU,
            // normal join, SiLU/Down, then ordinary native M.
            // Allhit control retains the zero-job copy schedule in BOTH arms.
            for _ in 0..r.repeats {
                let independent = rec.independent_dispatch_scope();
                ff.inner.cu_copy(rec);
                ff.projections(rec, 1, 0);
                drop(independent);
                ff.activation(rec);
                ff.down(rec, 1, 0);
                if ff.inner.job_count != 0 {
                    rec.global_barrier();
                    native_miss(ff, rec);
                }
                rec.global_barrier();
            }
            rec.publish_host_reads();
            r.stamp(rec, 4, vk::PipelineStageFlags::BOTTOM_OF_PIPE);
        })?;
        Ok(Self {
            backend,
            _fixture: fixture,
            _source: source,
            prefix,
            hit,
            copy,
            tail,
            baseline,
            begin,
            end,
            aux,
            serial: 0,
            r,
        })
    }

    fn run(&mut self, dual: bool) -> ProbeResult<Sample> {
        // External queue synchronization extends through failure cleanup. The
        // main gate also excludes the backend's optional VRAM keepalive submit.
        let _main = self.backend.shared.queue_access.lock().unwrap();
        let _aux = AUX_COMPUTE_SUBMIT.lock().unwrap();
        let device = self.r.device;
        unsafe { device.reset_fences(&[self.r.main_fence, self.r.aux_fence]) }
            .map_err(|e| format!("reset fences: {e}"))?;
        let first_value = self
            .serial
            .checked_add(1)
            .ok_or("timeline serial overflow")?;
        self.serial = self
            .serial
            .checked_add(self.r.repeats as u64)
            .ok_or("timeline serial overflow")?;
        let copy_value = self.serial;
        let mut pending = Pending {
            r: &self.r,
            main_live: false,
            aux_live: false,
            unblock_copy: None,
            unblock_ready: None,
        };
        let wall;
        let main_submit_us;
        let mut aux_submit_us = 0.0;
        if dual {
            let prefix_cmd = [handle(&self.prefix)];
            let hit_cmd = [handle(&self.hit)];
            let copy_cmd = [handle(&self.copy)];
            let tail_cmd = [handle(&self.tail)];
            let ready_sem = [self.r.ready];
            let copy_sem = [self.r.copy_done];
            let one_stage = [vk::PipelineStageFlags::ALL_COMMANDS];
            let values: Vec<_> = (first_value..=self.serial).map(|v| [v]).collect();
            let mut prefix_times: Vec<_> = values
                .iter()
                .map(|v| vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(v))
                .collect();
            let mut tail_times: Vec<_> = values
                .iter()
                .map(|v| vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(v))
                .collect();
            let mut copy_times: Vec<_> = values
                .iter()
                .map(|v| {
                    vk::TimelineSemaphoreSubmitInfo::default()
                        .wait_semaphore_values(v)
                        .signal_semaphore_values(v)
                })
                .collect();
            let begin_cmd = [handle(&self.begin)];
            let end_cmd = [handle(&self.end)];
            let mut main_infos = Vec::with_capacity(3 * self.r.repeats + 2);
            let mut aux_infos = Vec::with_capacity(self.r.repeats);
            main_infos.push(vk::SubmitInfo::default().command_buffers(&begin_cmd));
            for ((prefix_time, tail_time), copy_time) in prefix_times
                .iter_mut()
                .zip(tail_times.iter_mut())
                .zip(copy_times.iter_mut())
            {
                main_infos.push(
                    vk::SubmitInfo::default()
                        .command_buffers(&prefix_cmd)
                        .signal_semaphores(&ready_sem)
                        .push_next(prefix_time),
                );
                main_infos.push(vk::SubmitInfo::default().command_buffers(&hit_cmd));
                main_infos.push(
                    vk::SubmitInfo::default()
                        .command_buffers(&tail_cmd)
                        .wait_semaphores(&copy_sem)
                        .wait_dst_stage_mask(&one_stage)
                        .push_next(tail_time),
                );
                aux_infos.push(
                    vk::SubmitInfo::default()
                        .command_buffers(&copy_cmd)
                        .wait_semaphores(&ready_sem)
                        .wait_dst_stage_mask(&one_stage)
                        .signal_semaphores(&copy_sem)
                        .push_next(copy_time),
                );
            }
            main_infos.push(vk::SubmitInfo::default().command_buffers(&end_cmd));
            wall = Instant::now();
            if self.r.aux_first {
                let submit = Instant::now();
                unsafe { device.queue_submit(self.aux, &aux_infos, self.r.aux_fence) }
                    .map_err(|e| format!("aux-first submit: {e}"))?;
                aux_submit_us = submit.elapsed().as_secs_f64() * 1e6;
                pending.aux_live = true;
                pending.unblock_ready = Some(self.serial);
            }
            let submit = Instant::now();
            // No recovering helper: retry/drain of main here would deadlock on
            // the not-yet-submitted Copy. Direct calls are guarded, fail closed.
            unsafe {
                device.queue_submit(self.backend.shared.queue, &main_infos, self.r.main_fence)
            }
            .map_err(|e| format!("main multi-batch submit: {e}"))?;
            main_submit_us = submit.elapsed().as_secs_f64() * 1e6;
            pending.main_live = true;
            pending.unblock_ready = None;
            if !self.r.aux_first {
                pending.unblock_copy = Some(copy_value);
                let submit = Instant::now();
                unsafe { device.queue_submit(self.aux, &aux_infos, self.r.aux_fence) }
                    .map_err(|e| format!("aux submit (sample rejected; main will drain): {e}"))?;
                aux_submit_us = submit.elapsed().as_secs_f64() * 1e6;
                pending.aux_live = true;
                pending.unblock_copy = None;
            }
        } else {
            let commands = [handle(&self.baseline)];
            let infos = [vk::SubmitInfo::default().command_buffers(&commands)];
            wall = Instant::now();
            let submit = Instant::now();
            unsafe { device.queue_submit(self.backend.shared.queue, &infos, self.r.main_fence) }
                .map_err(|e| format!("baseline submit: {e}"))?;
            main_submit_us = submit.elapsed().as_secs_f64() * 1e6;
            pending.main_live = true;
        }
        pending.wait()?;
        let wall_us = wall.elapsed().as_secs_f64() * 1e6;
        // Only written query indices are fetched. Baseline leaves 1/2/3/5/6 reset.
        let start = self.r.tick(0)?;
        let end = self.r.tick(4)?;
        let mut sample = Sample {
            gpu_module_us: self.r.delta(start, end) / self.r.repeats as f64,
            wall_us: wall_us / self.r.repeats as f64,
            main_submit_us,
            aux_submit_us,
            ready_startup_us: None,
            h_us: None,
            post_h_join_wait_us: None,
            m_and_end_us: None,
            aux_copy_us: None,
        };
        if dual && self.r.detail {
            let h0 = self.r.tick(1)?;
            let h1 = self.r.tick(2)?;
            let tail = self.r.tick(3)?;
            sample.ready_startup_us = Some(self.r.delta(start, h0));
            sample.h_us = Some(self.r.delta(h0, h1));
            sample.post_h_join_wait_us = Some(self.r.delta(h1, tail));
            sample.m_and_end_us = Some(self.r.delta(tail, end));
            // Same aux queue for both endpoints; never cross-queue subtraction.
            sample.aux_copy_us = Some(self.r.delta(self.r.tick(5)?, self.r.tick(6)?));
        }
        Ok(sample)
    }
}

fn buffers(f: &SplitFixture) -> Vec<&dyn Buffer> {
    let i = &f.ff.inner;
    let mut b: Vec<&dyn Buffer> = vec![
        i.weights.as_ref(),
        i.tape.as_ref(),
        i.ids.as_ref(),
        i.input.as_ref(),
        i.gate.as_ref(),
        i.up.as_ref(),
        i.activation.as_ref(),
        i.output.as_ref(),
        i.journal.as_ref(),
        i.jobs.as_ref(),
        i.commands.as_ref(),
        i.commands_late.as_ref(),
        i.dummy.as_ref(),
        f.partial.as_ref(),
    ];
    if let Some(shared) = &f.ff.shared {
        b.extend(shared.matrices.iter().map(|v| v.as_ref()));
    }
    b
}

fn audit_jobs(f: &SplitFixture, source: &Sources, bank: usize, mixed: bool) {
    let i = &f.ff.inner;
    // Existing coherent import is mandatory; never silently replace PCIe source.
    assert!(matches!(
        &as_vk_buf(source.imported.as_ref()).unwrap().backing,
        Backing::ImportedHost { .. }
    ));
    let src_base = source
        .imported
        .device_addr()
        .expect("DECLINE: imported storage BDA required");
    let dst_base = i.weights.device_addr().unwrap();
    assert_eq!(i.job_count, if mixed { 3 } else { 0 });
    for role in 0..i.job_count {
        let range = &source.ranges[role][MISS_ID];
        let at = (ORDINAL * CAPACITY + role) * 8;
        let job = &i.job_words[at..at + 8];
        let u64_at = |j: usize| job[j] as u64 | ((job[j + 1] as u64) << 32);
        assert_eq!(
            u64_at(0),
            src_base + (bank * source.span + range.start) as u64
        );
        assert_eq!(u64_at(2), dst_base + range.start as u64);
        assert_eq!(
            job[4] as usize,
            range.len(),
            "whole role payload, not a tile"
        );
        assert_eq!(job[5] as usize, role * i.p.used + MISS_ID);
        // Every copied expert must be excluded from H, across ALL rows.
        for (row, ids) in i.routed.chunks_exact(i.p.used).enumerate() {
            for (slot, &expert) in ids.iter().enumerate() {
                assert_eq!(
                    i.hits[row] & (1 << slot) != 0,
                    !mixed || expert as usize != MISS_ID,
                    "H/copy disjointness"
                );
            }
        }
    }
    // All allocations, including imported source and Q8 shared weights, belong
    // to THIS backend; the sole queue-family identity makes sharing mode moot.
    for b in buffers(f)
        .into_iter()
        .chain(std::iter::once(source.imported.as_ref()))
    {
        let v = as_vk_buf(b).unwrap();
        assert!(Arc::ptr_eq(
            &v.shared,
            &as_vk_buf(i.weights.as_ref()).unwrap().shared
        ));
    }
}

fn verify_exact(f: &SplitFixture, source: &Sources, bank: usize, expected: &[u8], ro: &[Vec<u8>]) {
    assert_eq!(
        f.readonly(),
        ro,
        "immutable IDs/tape/input/jobs/journal/shared"
    );
    assert_eq!(source.bytes(bank), f.ff.inner.expected_weights);
    f.ff.verify(expected); // Payload bytes, scratch guards, Q8 shared guards.
    let actual = read(f.ff.inner.output.as_ref());
    assert_eq!(
        actual, expected,
        "native H/M must match native oracle BIT FOR BIT"
    );
    for word in actual[..actual.len() - GUARD].chunks_exact(4) {
        assert!(f32::from_le_bytes(word.try_into().unwrap()).is_finite());
    }
}

fn selected(value: &str, env: &str) -> bool {
    std::env::var(env).map_or(true, |s| s == "all" || s == value)
}

fn middle(mut data: Vec<f64>) -> f64 {
    data.sort_by(f64::total_cmp);
    (data[(data.len() - 1) / 2] + data[data.len() / 2]) * 0.5
}

#[test]
#[ignore = "explicit idle GPU; test-only same-family queue1; serial native exact H/M matrix"]
fn gpu_owned_native_same_family_dualqueue_module() {
    assert_eq!(
        std::env::var("MOE4ALL_TEST_AUX_COMPUTE_QUEUE")
            .ok()
            .as_deref(),
        Some("1"),
        "opt in before backend construction"
    );
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev =
        Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select idle GPU explicitly"));
    cfg.kernels.vulkan.decode_gpu_resident_copy = Some(true);
    let backend = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    let limits = unsafe {
        backend
            .shared
            .instance
            .get_physical_device_properties(backend.shared.physical_device)
    }
    .limits;
    assert!(
        limits.max_compute_work_group_invocations >= 1024
            && limits.max_compute_work_group_size[0] >= 1024,
        "DECLINE: native whole-copy local1024 unavailable"
    );
    let [q5, iq2] = native_profiles();
    let profiles = [
        q5,
        iq2,
        Profile {
            name: "iq3s_iq4nl_2560_640",
            gate_up: DType::Iq3S,
            ..iq2
        },
    ];
    for p in profiles {
        if !selected(format_name(p), "MOE4ALL_DUALQUEUE_FORMAT") {
            continue;
        }
        // Three immutable imported banks, bounded; no disk/cache/artifact writes.
        let source = Sources::new(&backend, p, 0);
        assert_eq!(source.banks, 3);
        let source_before: Vec<_> = (0..source.banks)
            .map(|b| source.bytes(b).to_vec())
            .collect();
        for rows in [1usize, 2, 4] {
            if !selected(&rows.to_string(), "MOE4ALL_DUALQUEUE_ROWS") {
                continue;
            }
            for mixed in [false, true] {
                let mode = if mixed { "one_miss" } else { "allhit" };
                if !selected(mode, "MOE4ALL_DUALQUEUE_MODE") {
                    continue;
                }
                for bank in 0..source.banks {
                    let mut fixture =
                        SplitFixture::new(&backend, &source, p, rows, p.gate_up == DType::Q5K, 32);
                    let input: Vec<f32> = (0..rows * p.ne)
                        .map(|i| ((i % 17) as f32 - 8.0) * 4.0)
                        .collect();
                    write(
                        fixture.ff.inner.input.as_ref(),
                        bytemuck::cast_slice(&input),
                    );
                    let expected = fixture.ff.reference(&backend, &source, bank);
                    assert!(expected[..expected.len() - GUARD]
                        .chunks_exact(4)
                        .any(|w| f32::from_le_bytes(w.try_into().unwrap()).abs() > 1e-3));
                    fixture.prepare(&source, bank, mixed, false);
                    audit_jobs(&fixture, &source, bank, mixed);
                    let readonly = fixture.readonly();
                    // Restore poisoned M/scratch for qualification and warmup arms,
                    // never between timed pairs. Retained IDs stay unchanged.
                    let saved: Vec<_> = buffers(&fixture)
                        .into_iter()
                        .map(|b| (b, read(b)))
                        .collect();
                    let restore = || {
                        for (b, bytes) in &saved {
                            write(*b, bytes);
                        }
                    };
                    let startup = Instant::now();
                    let mut frame = Frame::new(&backend, &fixture, &source).unwrap();
                    let startup_record_us = startup.elapsed().as_secs_f64() * 1e6;
                    let mut warmup_runs = [0usize; 2];
                    let mut warmup_wall_ms = [0.0f64; 2];
                    for (arm, use_dual) in [false, true].into_iter().enumerate() {
                        restore();
                        let warm = Instant::now();
                        while warm.elapsed() < std::time::Duration::from_millis(WARMUP_MS) {
                            // Each run blocks on fences: this is GPU warmup, not CPU polling.
                            frame.run(use_dual).unwrap();
                            warmup_runs[arm] += 1;
                        }
                        warmup_wall_ms[arm] = warm.elapsed().as_secs_f64() * 1000.0;
                        verify_exact(&fixture, &source, bank, &expected, &readonly);
                    }
                    let mut base = Vec::new();
                    let mut dual = Vec::new();
                    for pair in 0..SAMPLES {
                        // Alternate AB/BA; equal sample count, no duration-based spin.
                        for use_dual in if pair % 2 == 0 {
                            [false, true]
                        } else {
                            [true, false]
                        } {
                            let sample = frame.run(use_dual).unwrap();
                            println!(
                                "DUALQUEUE_SAMPLE format={} profile={} rows={rows} mode={mode} bank={bank} pair={pair} arm={} runtime_schedule=true pure_floor=false max_abs=0 max_rel=0 sample={sample:?}",
                                format_name(p),
                                p.name,
                                if use_dual { "two_queue" } else { "baseline" }
                            );
                            if use_dual {
                                dual.push(sample);
                            } else {
                                base.push(sample);
                            }
                        }
                    }
                    // No BAR uploads/readbacks between timed pairs: they perturb clocks
                    // and cache state. Prove a fresh candidate-only run after measurement.
                    restore();
                    frame.run(true).unwrap();
                    verify_exact(&fixture, &source, bank, &expected, &readonly);
                    println!(
                        "DUALQUEUE_MODULE format={} profile={} rows={rows} mode={mode} bank={bank} base_gpu_us={} dual_gpu_us={} base_wall_us={} dual_wall_us={} startup_record_us={startup_record_us} base_submit_calls=1 dual_submit_calls=2 repeats={} aux_first={} samples={SAMPLES} warmup_min_ms={WARMUP_MS} warmup_runs={warmup_runs:?} warmup_wall_ms={warmup_wall_ms:?} warmup_timed=false runtime_schedule=true pure_floor=false whole_payload=true native_exact=true max_abs=0 max_rel=0 scope=complete_H_copy_join_M_ready_and_join_included per_fixture=true per_token_runtime=false hardware_overlap_verified=false",
                        format_name(p),
                        p.name,
                        middle(base.iter().map(|s| s.gpu_module_us).collect()),
                        middle(dual.iter().map(|s| s.gpu_module_us).collect()),
                        middle(base.iter().map(|s| s.wall_us).collect()),
                        middle(dual.iter().map(|s| s.wall_us).collect()),
                        frame.r.repeats,
                        frame.r.aux_first,
                    );
                    // Explicitly drain guard is already gone before frame and its
                    // RecordedCmd owners drop. No aux commands reach a main-only drop.
                    drop(frame);
                    for (b, before) in source_before.iter().enumerate() {
                        assert_eq!(
                            source.bytes(b),
                            before,
                            "all imported banks remain readonly"
                        );
                    }
                }
            }
        }
    }
}
