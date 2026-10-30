#![cfg(test)]

//! Real FFN module probes for the test-only indirect payload helper. No owner-policy fixture.
use super::*;
use crate::AlignedHostBuffer;
use infr_core::DType;
use std::{collections::HashMap, ops::Range};

const CAPACITY: usize = 384;
const ORDINAL: usize = 1;
const LAYERS: usize = ORDINAL + 1;
const GUARD: usize = 256;
const POISON: u8 = 0xa5;
const PAIRS: usize = 8;
const SOURCE_RING_BYTES: usize = 384 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
struct Profile {
    name: &'static str,
    ne: usize,
    nff: usize,
    used: usize,
    gate_up: DType,
    down: DType,
}

fn profiles(native: bool) -> [Profile; 2] {
    [
        Profile {
            name: "synthetic_approx_q4k_used8",
            ne: if native { 2048 } else { 256 },
            nff: if native { 512 } else { 256 },
            used: 8,
            gate_up: DType::Q4K,
            down: DType::Q4K,
        },
        Profile {
            name: "synthetic_approx_iq2s_iq4nl_used10",
            ne: if native { 2560 } else { 512 },
            nff: if native { 768 } else { 256 },
            used: 10,
            gate_up: DType::Iq2S,
            down: DType::Iq4Nl,
        },
    ]
}

#[derive(Clone, Copy, Debug)]
enum Variant {
    CuSerial,
    CuPartial,
    KhrFull,
    KhrSerial,
    KhrPipeline,
}

const VARIANTS: [Variant; 5] = [
    Variant::CuSerial,
    Variant::CuPartial,
    Variant::KhrFull,
    Variant::KhrSerial,
    Variant::KhrPipeline,
];

struct CopyReadyEvent {
    shared: Arc<crate::VulkanShared>,
    event: vk::Event,
}

impl CopyReadyEvent {
    fn new(be: &VulkanBackend) -> Self {
        let event = unsafe {
            be.shared
                .device
                .create_event(&vk::EventCreateInfo::default(), None)
        }
        .unwrap();
        Self {
            shared: Arc::clone(&be.shared),
            event,
        }
    }
}

impl Drop for CopyReadyEvent {
    fn drop(&mut self) {
        // Fixture submissions are drained before the retained event is released.
        unsafe { self.shared.device.destroy_event(self.event, None) };
    }
}

fn backend() -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev =
        Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select an idle GPU explicitly"));
    cfg.kernels.vulkan.decode_gpu_resident_copy = Some(true);
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    assert!(
        be.shared.indirect_copy.is_some(),
        "KHR indirect copy required"
    );
    let properties = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    };
    assert!(properties.limits.max_compute_work_group_invocations >= 1024);
    assert!(properties.limits.max_compute_work_group_size[0] >= 1024);
    be
}

fn mapped(be: &VulkanBackend, bytes: usize) -> Box<dyn Buffer> {
    be.alloc_mapped_arena_bda(bytes).unwrap().0
}

fn write(buffer: &dyn Buffer, bytes: &[u8]) {
    assert!(bytes.len() <= buffer.len_bytes());
    let ptr = as_vk_buf(buffer)
        .unwrap()
        .mapped_ptr()
        .expect("mapped fixture");
    // Every caller has drained the previous submission before changing fixture memory.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
}

fn read(buffer: &dyn Buffer) -> Vec<u8> {
    let ptr = as_vk_buf(buffer)
        .unwrap()
        .mapped_ptr()
        .expect("mapped fixture");
    unsafe { std::slice::from_raw_parts(ptr, buffer.len_bytes()).to_vec() }
}

fn missing(role: usize, expert: usize) -> bool {
    match expert % 5 {
        0 => false,
        1 => true,
        2 => role == 2,
        3 => role == 0,
        _ => role == 1,
    }
}

struct Sources {
    host: Arc<AlignedHostBuffer>,
    imported: Arc<dyn Buffer>,
    ranges: [Vec<Range<usize>>; 3],
    span: usize,
    banks: usize,
    miss_bytes: usize,
}

impl Sources {
    fn new(be: &VulkanBackend, p: Profile, minimum_bytes: usize) -> Self {
        let mut cursor = GUARD;
        let ranges = std::array::from_fn(|role| {
            let dtype = if role == 2 { p.down } else { p.gate_up };
            let bytes = infr_testkit::synth_weight(dtype, p.ne * p.nff, 7).len();
            assert_eq!(bytes % 16, 0);
            (0..p.used)
                .map(|_| {
                    let start = cursor;
                    cursor += bytes + GUARD;
                    start..cursor - GUARD
                })
                .collect::<Vec<_>>()
        });
        let span = cursor;
        let banks = minimum_bytes.div_ceil(span).max(3);
        let host = AlignedHostBuffer::new(span * banks).unwrap();
        let bytes = unsafe { std::slice::from_raw_parts_mut(host.as_ptr(), span * banks) };
        bytes.fill(POISON);
        for bank in 0..banks {
            for (role, role_ranges) in ranges.iter().enumerate() {
                let dtype = if role == 2 { p.down } else { p.gate_up };
                for (expert, range) in role_ranges.iter().enumerate() {
                    let seed = 17 + (bank * 127 + role * 31 + expert) as u32;
                    let weight = infr_testkit::synth_weight(dtype, p.ne * p.nff, seed);
                    bytes[bank * span + range.start..bank * span + range.end]
                        .copy_from_slice(&weight);
                }
            }
        }
        let imported = be
            .try_import_host_shard(
                be.shared
                    .external_memory_host
                    .as_ref()
                    .expect("host import required"),
                Arc::clone(&host),
                0,
                host.allocated_len(),
            )
            .unwrap()
            .buffer;
        assert!(matches!(
            &as_vk_buf(imported.as_ref()).unwrap().backing,
            Backing::ImportedHost { .. }
        ));
        assert!(
            imported.device_addr().is_some(),
            "no VRAM or DMA source substitution"
        );
        let miss_bytes = ranges
            .iter()
            .enumerate()
            .flat_map(|(role, ranges)| ranges.iter().enumerate().map(move |(e, r)| (role, e, r)))
            .filter(|(role, e, _)| missing(*role, *e))
            .map(|(_, _, r)| r.len())
            .sum();
        Self {
            host,
            imported,
            ranges,
            span,
            banks,
            miss_bytes,
        }
    }

    fn bytes(&self, bank: usize) -> &[u8] {
        assert!(bank < self.banks);
        unsafe { std::slice::from_raw_parts(self.host.as_ptr().add(bank * self.span), self.span) }
    }
}

struct Fixture {
    p: Profile,
    rows: usize,
    weights: Box<dyn Buffer>,
    tape: Box<dyn Buffer>,
    ids: Box<dyn Buffer>,
    input: Box<dyn Buffer>,
    gate: Box<dyn Buffer>,
    up: Box<dyn Buffer>,
    activation: Box<dyn Buffer>,
    output: Box<dyn Buffer>,
    journal: Box<dyn Buffer>,
    jobs: Box<dyn Buffer>,
    commands: Box<dyn Buffer>,
    commands_late: Box<dyn Buffer>,
    early_ready: CopyReadyEvent,
    dummy: Box<dyn Buffer>,
    routed: Vec<u32>,
    hits: Vec<u32>,
    expected_weights: Vec<u8>,
    job_words: Vec<u32>,
    job_count: usize,
}

impl Fixture {
    fn new(be: &VulkanBackend, source: &Sources, p: Profile, rows: usize) -> Self {
        assert!((1..=8).contains(&rows));
        let routed: Vec<_> = (0..rows)
            .flat_map(|row| (0..p.used).map(move |slot| ((slot + row * 3) % p.used) as u32))
            .collect();
        let hits = routed
            .chunks_exact(p.used)
            .map(|ids| {
                ids.iter().enumerate().fold(0, |mask, (slot, &expert)| {
                    mask | (u32::from((0..3).all(|role| !missing(role, expert as usize))) << slot)
                })
            })
            .collect::<Vec<_>>();
        let mut ids = routed.clone();
        ids.extend_from_slice(&hits);
        ids.extend(hits.iter().map(|mask| ((1u32 << p.used) - 1) ^ mask));
        let fixture = Self {
            p,
            rows,
            weights: mapped(be, source.span),
            tape: mapped(be, 3 * p.used * 8),
            ids: mapped(be, ids.len() * 4),
            input: mapped(be, rows * p.ne * 4),
            gate: mapped(be, rows * p.used * p.nff * 4 + GUARD),
            up: mapped(be, rows * p.used * p.nff * 4 + GUARD),
            activation: mapped(be, rows * p.used * p.nff * 4 + GUARD),
            output: mapped(be, rows * p.used * p.ne * 4 + GUARD),
            journal: mapped(be, LAYERS * RESIDENT_JOURNAL_BYTES),
            jobs: mapped(be, LAYERS * CAPACITY * 32),
            commands: mapped(be, LAYERS * CAPACITY * 32 + GUARD),
            commands_late: mapped(be, LAYERS * CAPACITY * 32 + GUARD),
            early_ready: CopyReadyEvent::new(be),
            dummy: mapped(be, 16 + LAYERS * CAPACITY * 16 + GUARD),
            routed,
            hits,
            expected_weights: Vec::new(),
            job_words: Vec::new(),
            job_count: 0,
        };
        write(fixture.ids.as_ref(), bytemuck::cast_slice(&ids));
        let input = (0..rows * p.ne)
            .map(|i| ((i as f32 * 0.037).sin() + (i as f32 * 0.013).cos()) * 0.005)
            .collect::<Vec<_>>();
        write(fixture.input.as_ref(), bytemuck::cast_slice(&input));
        let base = fixture.weights.device_addr().unwrap();
        let tape: Vec<u32> = source
            .ranges
            .iter()
            .flatten()
            .flat_map(|range| {
                let address = base + range.start as u64;
                [address as u32, (address >> 32) as u32]
            })
            .collect();
        write(fixture.tape.as_ref(), bytemuck::cast_slice(&tape));
        fixture
    }

    fn prepare(&mut self, source: &Sources, bank: usize, resident: bool) {
        self.hits = self
            .routed
            .chunks_exact(self.p.used)
            .map(|ids| {
                ids.iter().enumerate().fold(0, |mask, (slot, &expert)| {
                    mask | (u32::from((0..3).all(|role| !missing(role, expert as usize))) << slot)
                })
            })
            .collect();
        self.write_masks();
        self.expected_weights = source.bytes(bank).to_vec();
        let mut weights = self.expected_weights.clone();
        self.job_words = vec![0u32; LAYERS * CAPACITY * 8];
        self.job_count = 0;
        for (role, ranges) in source.ranges.iter().enumerate() {
            for (expert, range) in ranges.iter().enumerate() {
                if !missing(role, expert) {
                    continue;
                }
                if !resident {
                    weights[range.clone()].fill(POISON);
                }
                let src = source.imported.device_addr().unwrap()
                    + (bank * source.span + range.start) as u64;
                let dst = self.weights.device_addr().unwrap() + range.start as u64;
                let at = (ORDINAL * CAPACITY + self.job_count) * 8;
                self.job_words[at..at + 8].copy_from_slice(&[
                    src as u32,
                    (src >> 32) as u32,
                    dst as u32,
                    (dst >> 32) as u32,
                    range.len() as u32,
                    (role * self.p.used + expert) as u32,
                    self.job_count as u32,
                    (role * self.p.used + expert) as u32,
                ]);
                self.job_count += 1;
            }
        }
        assert!(self.job_count > 0 && self.job_count < CAPACITY);
        write(self.weights.as_ref(), &weights);
        write(self.jobs.as_ref(), bytemuck::cast_slice(&self.job_words));
        self.set_status(4, self.job_count);
        write(
            self.commands.as_ref(),
            &vec![POISON; self.commands.len_bytes()],
        );
        write(
            self.commands_late.as_ref(),
            &vec![POISON; self.commands_late.len_bytes()],
        );
        let mut dummy = vec![POISON; self.dummy.len_bytes()];
        dummy[..4].copy_from_slice(&[0x12, 0x34, 0x56, 0x78]);
        write(self.dummy.as_ref(), &dummy);
        for buffer in [&self.gate, &self.up, &self.activation, &self.output] {
            let mut bytes = vec![POISON; buffer.len_bytes()];
            let body = bytes.len() - GUARD;
            for word in bytes[..body].chunks_exact_mut(4) {
                word.copy_from_slice(&0.125f32.to_le_bytes());
            }
            write(buffer.as_ref(), &bytes);
        }
    }

    fn write_masks(&self) {
        let mut ids = self.routed.clone();
        ids.extend_from_slice(&self.hits);
        ids.extend(
            self.hits
                .iter()
                .map(|mask| ((1u32 << self.p.used) - 1) ^ mask),
        );
        write(self.ids.as_ref(), bytemuck::cast_slice(&ids));
    }

    fn prepare_all_hit(&mut self, source: &Sources, bank: usize) {
        self.prepare(source, bank, true);
        self.job_count = 0;
        self.set_status(1, 0);
        self.hits.fill((1u32 << self.p.used) - 1);
        self.write_masks();
    }

    fn set_status(&self, status: u32, count: usize) {
        let mut journal = vec![0u32; LAYERS * RESIDENT_JOURNAL_WORDS];
        let base = ORDINAL * RESIDENT_JOURNAL_WORDS;
        journal[base + RESIDENT_JOURNAL_STATUS_WORD] = status;
        journal[base + RESIDENT_JOURNAL_JOB_COUNT_WORD] = count as u32;
        journal[base + RESIDENT_JOURNAL_JOB_COUNT_WORD + 1] = 19;
        write(self.journal.as_ref(), bytemuck::cast_slice(&journal));
    }

    fn projections(&self, rec: &Recorder<'_>, bank: u32, mask: u32) {
        for (role, output) in [(0, self.gate.as_ref()), (1, self.up.as_ref())] {
            rec.linear_native_id_multi_paged(
                self.p.gate_up,
                self.weights.device_addr().unwrap(),
                0,
                self.tape.as_ref(),
                self.ids.as_ref(),
                self.p.used,
                role * self.p.used,
                self.input.as_ref(),
                false,
                output,
                self.p.ne,
                self.p.nff,
                self.rows,
                mask,
                bank,
            );
        }
    }

    fn tail(&self, rec: &Recorder<'_>, bank: u32, mask: u32) {
        rec.silu_mul(
            self.gate.as_ref(),
            self.up.as_ref(),
            self.activation.as_ref(),
            self.rows * self.p.used * self.p.nff,
            None,
        );
        rec.linear_native_id_multi_paged(
            self.p.down,
            self.weights.device_addr().unwrap(),
            0,
            self.tape.as_ref(),
            self.ids.as_ref(),
            self.p.used,
            2 * self.p.used,
            self.activation.as_ref(),
            true,
            self.output.as_ref(),
            self.p.nff,
            self.p.ne,
            self.rows,
            mask,
            bank,
        );
    }

    fn cu_copy(&self, rec: &Recorder<'_>) {
        let kernel = rec.be.kernel(
            "gpu_resident_copy_payload1024",
            crate::gemm::gpu_resident_copy_payload_spv(true),
            2,
            4,
        );
        rec.dispatch(
            kernel,
            &[
                Recorder::vkb(self.journal.as_ref()),
                Recorder::vkb(self.jobs.as_ref()),
            ],
            0,
            bytemuck::bytes_of(&(ORDINAL as u32)),
            8,
        );
    }

    fn khr_copy(&self, rec: &Recorder<'_>) {
        rec.begin_owned_indirect_copy(
            self.journal.as_ref(),
            self.jobs.as_ref(),
            self.commands.as_ref(),
            self.dummy.as_ref(),
            ORDINAL as u32,
            (3 * self.rows * self.p.used) as u32,
        )
        .unwrap();
    }

    fn record(&self, rec: &Recorder<'_>, variant: Variant, hit_only: bool) {
        if matches!(variant, Variant::KhrPipeline) {
            rec.begin_owned_indirect_copy_phase(
                self.journal.as_ref(),
                self.jobs.as_ref(),
                self.commands.as_ref(),
                self.dummy.as_ref(),
                ORDINAL as u32,
                (3 * self.rows * self.p.used) as u32,
                [1, 0, self.p.used as u32, self.p.used as u32],
            )
            .unwrap();
            rec.mark_owned_indirect_copy_ready(self.early_ready.event);
            rec.wait_owned_indirect_copy_ready(self.early_ready.event, true);
            rec.begin_owned_indirect_copy_phase(
                self.journal.as_ref(),
                self.jobs.as_ref(),
                self.commands_late.as_ref(),
                self.dummy.as_ref(),
                ORDINAL as u32,
                (3 * self.rows * self.p.used) as u32,
                [2, 0, self.p.used as u32, self.p.used as u32],
            )
            .unwrap();
            self.projections(rec, 1, 0);
            self.tail(rec, 1, 0);
            rec.wait_owned_indirect_copy_ready(self.early_ready.event, false);
            if !hit_only {
                self.projections(rec, 2, 0);
                rec.silu_mul(
                    self.gate.as_ref(),
                    self.up.as_ref(),
                    self.activation.as_ref(),
                    self.rows * self.p.used * self.p.nff,
                    None,
                );
            }
            rec.global_barrier();
            if !hit_only {
                rec.linear_native_id_multi_paged(
                    self.p.down,
                    self.weights.device_addr().unwrap(),
                    0,
                    self.tape.as_ref(),
                    self.ids.as_ref(),
                    self.p.used,
                    2 * self.p.used,
                    self.activation.as_ref(),
                    true,
                    self.output.as_ref(),
                    self.p.nff,
                    self.p.ne,
                    self.rows,
                    0,
                    2,
                );
            }
            return;
        }
        match variant {
            Variant::CuSerial => {
                self.cu_copy(rec);
                rec.global_barrier();
            }
            Variant::CuPartial => {
                let independent = rec.independent_dispatch_scope();
                self.cu_copy(rec);
                self.projections(rec, 1, 0);
                drop(independent);
            }
            Variant::KhrFull => self.khr_copy(rec),
            Variant::KhrSerial => {
                self.khr_copy(rec);
                rec.global_barrier();
            }
            Variant::KhrPipeline => unreachable!(),
        }
        if !matches!(variant, Variant::CuPartial) {
            self.projections(rec, 1, 0);
        }
        self.tail(rec, 1, 0);
        rec.global_barrier();
        if !hit_only && self.job_count != 0 {
            self.projections(rec, 2, 0);
            self.tail(rec, 2, 0);
        }
    }

    fn verify(&self, expected: &[u8], hit_only: bool) {
        assert_eq!(
            read(self.weights.as_ref()),
            self.expected_weights,
            "payload and every weight guard"
        );
        let actual = read(self.output.as_ref());
        let width = self.p.ne * 4;
        for (assignment, &expert) in self.routed.iter().enumerate() {
            let range = assignment * width..(assignment + 1) * width;
            let hit = self.hits[assignment / self.p.used] & (1 << (assignment % self.p.used)) != 0;
            if hit_only && !hit {
                assert!(
                    actual[range]
                        .chunks_exact(4)
                        .all(|v| v == 0.125f32.to_le_bytes().as_slice()),
                    "masked Down overwrote an inactive assignment: expert={expert}"
                );
            } else {
                assert_eq!(
                    &actual[range.clone()],
                    &expected[range],
                    "FFN assignment={assignment} expert={expert}"
                );
            }
        }
        for buffer in [&self.gate, &self.up, &self.activation, &self.output] {
            let bytes = read(buffer.as_ref());
            assert!(
                bytes[bytes.len() - GUARD..].iter().all(|&v| v == POISON),
                "FFN output guard"
            );
        }
        for buffer in [&self.commands, &self.commands_late, &self.dummy] {
            let bytes = read(buffer.as_ref());
            assert!(
                bytes[bytes.len() - GUARD..].iter().all(|&v| v == POISON),
                "indirect sidecar guard"
            );
        }
    }

    fn reference(&mut self, be: &VulkanBackend, source: &Sources, bank: usize) -> Vec<u8> {
        self.prepare(source, bank, true);
        let rec = be.recorder().unwrap();
        rec.seed_barrier();
        self.projections(&rec, 0, (1 << self.p.used) - 1);
        self.tail(&rec, 0, (1 << self.p.used) - 1);
        rec.publish_host_reads();
        rec.finish().unwrap();
        let result = read(self.output.as_ref());
        let values: &[f32] = bytemuck::cast_slice(&result[..result.len() - GUARD]);
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(
            values.iter().any(|v| v.abs() > 1e-7),
            "reference must exercise nonzero FFN output"
        );
        result
    }
}

struct Timestamps<'a> {
    device: &'a ash::Device,
    pool: vk::QueryPool,
}

impl<'a> Timestamps<'a> {
    fn new(be: &'a VulkanBackend) -> Self {
        assert!(be.shared.submit_timestamp_valid_bits > 0);
        let device = &be.shared.device;
        let pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .unwrap();
        Self { device, pool }
    }

    fn begin(&self, rec: &Recorder<'_>) {
        unsafe {
            self.device.cmd_reset_query_pool(rec.cmd, self.pool, 0, 2);
            self.device.cmd_write_timestamp(
                rec.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.pool,
                0,
            );
        }
    }

    fn end(&self, rec: &Recorder<'_>) {
        unsafe {
            self.device.cmd_write_timestamp(
                rec.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                1,
            )
        };
    }

    fn elapsed(&self, be: &VulkanBackend) -> f64 {
        let mut ticks = [0u64; 2];
        unsafe {
            self.device.get_query_pool_results(
                self.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .unwrap();
        let bits = be.shared.submit_timestamp_valid_bits;
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let us = (ticks[1].wrapping_sub(ticks[0]) & mask) as f64
            * be.shared.submit_timestamp_period_ns as f64
            / 1000.0;
        assert!(us.is_finite() && us > 0.0);
        us
    }
}

impl Drop for Timestamps<'_> {
    fn drop(&mut self) {
        unsafe { self.device.destroy_query_pool(self.pool, None) };
    }
}

fn sample(
    be: &VulkanBackend,
    fixture: &Fixture,
    variant: Variant,
    hit_only: bool,
    query: Option<&Timestamps<'_>>,
) -> Option<f64> {
    unsafe { be.shared.device.reset_event(fixture.early_ready.event) }.unwrap();
    let rec = be.recorder().unwrap();
    assert!(
        !rec.prof_ops && rec.timeline_query.borrow().is_none(),
        "disable per-op/timeline tracing"
    );
    assert!(!rec.no_barrier && !rec.full_barrier && !rec.stages);
    rec.seed_barrier();
    if let Some(query) = query {
        query.begin(&rec);
    }
    fixture.record(&rec, variant, hit_only);
    if let Some(query) = query {
        query.end(&rec);
    }
    rec.publish_host_reads();
    rec.finish().unwrap();
    query.map(|query| query.elapsed(be))
}

#[test]
#[ignore = "explicit MOE4ALL_TEST_DEVICE; imported RAM BDA, real quantized FFN, rows 1 through 8"]
fn gpu_owned_indirect_real_ffn_correctness_rows_1_to_8() {
    let be = backend();
    for p in profiles(false) {
        let source = Sources::new(&be, p, 0);
        for rows in 1..=8 {
            let mut fixture = Fixture::new(&be, &source, p, rows);
            let bank = rows % source.banks;
            let expected = fixture.reference(&be, &source, bank);
            for variant in VARIANTS {
                for hit_only in [true, false] {
                    fixture.prepare(&source, bank, false);
                    sample(&be, &fixture, variant, hit_only, None);
                    fixture.verify(&expected, hit_only);
                }
                fixture.prepare_all_hit(&source, bank);
                sample(&be, &fixture, variant, false, None);
                fixture.verify(&expected, false);
            }
        }
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) * 0.5
}

fn paired_profiles(rows_set: &[usize]) {
    let be = backend();
    let query = Timestamps::new(&be);
    for p in profiles(true) {
        let source = Sources::new(&be, p, SOURCE_RING_BYTES);
        for &rows in rows_set {
            let mut fixture = Fixture::new(&be, &source, p, rows);
            let mut references = HashMap::new();
            let mut visited = HashSet::new();
            let mut cursor = 0;
            for all_hit in [false, true] {
                let candidates: &[Variant] = if all_hit {
                    &[Variant::KhrFull]
                } else {
                    &[
                        Variant::CuPartial,
                        Variant::KhrFull,
                        Variant::KhrSerial,
                        Variant::KhrPipeline,
                    ]
                };
                for &candidate in candidates {
                    let mut base_times = Vec::new();
                    let mut candidate_times = Vec::new();
                    let mut ratios = Vec::new();
                    let mut banks = Vec::new();
                    for pair in 0..=PAIRS {
                        let reverse = pair % 2 != 0;
                        let order = if reverse {
                            [candidate, Variant::CuSerial]
                        } else {
                            [Variant::CuSerial, candidate]
                        };
                        let mut times = [0.0; 2];
                        let mut pair_banks = [0; 2];
                        for (arm, variant) in order.into_iter().enumerate() {
                            let bank = cursor % source.banks;
                            cursor += 1;
                            if !all_hit {
                                visited.insert(bank);
                            }
                            pair_banks[arm] = bank;
                            let expected = references
                                .entry(bank)
                                .or_insert_with(|| fixture.reference(&be, &source, bank));
                            if all_hit {
                                fixture.prepare_all_hit(&source, bank);
                            } else {
                                fixture.prepare(&source, bank, false);
                            }
                            times[arm] =
                                sample(&be, &fixture, variant, false, Some(&query)).unwrap();
                            fixture.verify(expected, false);
                        }
                        if pair != 0 {
                            let (base, compared) = if reverse {
                                (times[1], times[0])
                            } else {
                                (times[0], times[1])
                            };
                            base_times.push(base);
                            candidate_times.push(compared);
                            ratios.push(base / compared);
                            banks.push(pair_banks);
                        }
                    }
                    println!("OWNED_INDIRECT_REAL_FFN profile={} synthetic_approximation=true ne={} nff={} used={} gate_up={:?} down={:?} rows={rows} all_hit={all_hit} jobs={} baseline=CuSerial candidate={candidate:?} cu_local=1024 cu_groups=8 khr_record_count={} pairs={PAIRS} order=AB_BA timestamps=outer_only source=ImportedHostBDA source_banks={banks:?} baseline_us={base_times:?} candidate_us={candidate_times:?} baseline_median_us={} candidate_median_us={} median_paired_speedup={} verified=true",
                    p.name, p.ne, p.nff, p.used, p.gate_up, p.down,
                    fixture.job_count,
                    3 * rows * p.used,
                    median(&base_times), median(&candidate_times), median(&ratios));
                }
            }
            let visited_bytes = visited.len() * source.miss_bytes;
            assert!(
                visited_bytes > 128 * 1024 * 1024,
                "actual copied source working set must rotate beyond cache"
            );
            println!("OWNED_INDIRECT_REAL_FFN_WORKING_SET profile={} rows={rows} banks={} visited_banks={} visited_miss_bytes={visited_bytes} allocated_source_bytes={}",
                p.name, source.banks, visited.len(), source.span * source.banks);
        }
    }
}

#[test]
#[ignore = "explicit MOE4ALL_TEST_DEVICE; paired native-size synthetic profiles, outer timestamps, rows 1/2"]
fn gpu_owned_indirect_real_ffn_paired_native_rows_1_2() {
    paired_profiles(&[1, 2]);
}

#[test]
#[ignore = "explicit MOE4ALL_TEST_DEVICE; parent opts in after initial positive result"]
fn gpu_owned_indirect_real_ffn_paired_native_rows_4() {
    paired_profiles(&[4]);
}

#[test]
#[ignore = "explicit MOE4ALL_TEST_DEVICE; opt-in native row-8 capability, no automatic fallback"]
fn gpu_owned_indirect_real_ffn_paired_native_rows_8() {
    paired_profiles(&[8]);
}

#[test]
#[ignore = "parent serial GPU; continuous cold-working-set frames, imported RAM and quantized FFN"]
fn gpu_owned_indirect_continuous_frames() {
    let be = backend();
    let query = Timestamps::new(&be);
    for p in profiles(true) {
        let source = Sources::new(&be, p, SOURCE_RING_BYTES);
        for rows in [1, 2] {
            let mut fixtures = Vec::new();
            let mut expected = Vec::new();
            for bank in 0..source.banks {
                let mut fixture = Fixture::new(&be, &source, p, rows);
                expected.push(fixture.reference(&be, &source, bank));
                fixture.prepare(&source, bank, false);
                fixtures.push(fixture);
            }
            assert!(source.banks * source.miss_bytes > 128 * 1024 * 1024);
            let variants = [
                Variant::CuPartial,
                Variant::CuSerial,
                Variant::KhrFull,
                Variant::KhrPipeline,
            ];
            let recordings: Vec<_> = variants
                .iter()
                .map(|&variant| {
                    let rec = be.recorder_persistent().unwrap();
                    rec.seed_barrier();
                    rec.global_barrier();
                    query.begin(&rec);
                    for fixture in &fixtures {
                        fixture.record(&rec, variant, false);
                        rec.global_barrier();
                    }
                    query.end(&rec);
                    rec.publish_host_reads();
                    rec.finish_record().unwrap()
                })
                .collect();
            let reset_events = || {
                for fixture in &fixtures {
                    unsafe { be.shared.device.reset_event(fixture.early_ready.event) }.unwrap();
                }
            };
            // Warm continuous GPU work, not a repeated single expert bank. Each frame
            // crosses >128 MiB of unique source payload and an independent VRAM bank per step.
            for recording in &recordings {
                reset_events();
                recording.replay().unwrap();
                reset_events();
                recording.replay().unwrap();
            }
            for (candidate, _) in recordings.iter().enumerate().skip(1) {
                let mut baseline_us = Vec::new();
                let mut candidate_us = Vec::new();
                for pair in 0..6 {
                    let order = if pair % 2 == 0 {
                        [0, candidate]
                    } else {
                        [candidate, 0]
                    };
                    for index in order {
                        reset_events();
                        recordings[index].replay().unwrap();
                        let us = query.elapsed(&be) / fixtures.len() as f64;
                        if index == 0 {
                            baseline_us.push(us);
                        } else {
                            candidate_us.push(us);
                        }
                    }
                }
                // Payload byte-for-byte correctness is covered separately for all variants.
                // Here verify every row/expert output after sustained frames, without putting
                // a full BAR readback between the timed arms and idling the GPU for milliseconds.
                for (fixture, reference) in fixtures.iter().zip(&expected) {
                    assert_eq!(
                        read(fixture.output.as_ref()),
                        *reference,
                        "continuous output profile={} rows={rows} candidate={:?}",
                        p.name,
                        variants[candidate]
                    );
                }
                println!("OWNED_INDIRECT_CONTINUOUS profile={} rows={rows} layers={} source_bytes={} payload_bytes_per_frame={} baseline=CuPartial candidate={:?} baseline_us_per_layer={baseline_us:?} candidate_us_per_layer={candidate_us:?} baseline_median_us={} candidate_median_us={} verified=true",
                    p.name, fixtures.len(), source.span * source.banks,
                    source.banks * source.miss_bytes, variants[candidate],
                    median(&baseline_us), median(&candidate_us));
            }
        }
    }
}

#[test]
#[ignore = "explicit MOE4ALL_TEST_DEVICE; unconditional helper SKIP/invalid/CPU/padding GPU cases"]
fn gpu_owned_indirect_helper_skip_invalid_cpu_and_stale_padding() {
    let be = backend();
    let p = profiles(false)[0];
    let source = Sources::new(&be, p, 0);
    let mut fixture = Fixture::new(&be, &source, p, 2);
    let stale_target = mapped(&be, CAPACITY * 16 + GUARD);
    for case in [
        "skip",
        "all_hit",
        "invalid_status",
        "zero_jobs",
        "count_over_capacity",
        "count_over_recorded",
        "cpu_only",
        "cpu_mixed",
        "zero_src_gpu",
        "bad_size",
        "bad_destination",
        "stale_padding",
    ] {
        fixture.prepare(&source, 0, false);
        let count = fixture.job_count;
        let status = match case {
            "skip" => 0,
            "all_hit" => 1,
            "invalid_status" => 3,
            "cpu_only" | "cpu_mixed" => 5,
            _ => 4,
        };
        let journal_count = match case {
            "all_hit" | "zero_jobs" => 0,
            "count_over_capacity" => CAPACITY + 1,
            "count_over_recorded" => 3 * fixture.rows * p.used + 1,
            _ => count,
        };
        fixture.set_status(status, journal_count);
        let job_base = ORDINAL * CAPACITY * 8;
        match case {
            "cpu_only" => {
                for j in 0..count {
                    fixture.job_words[job_base + j * 8..job_base + j * 8 + 2].fill(0);
                }
            }
            "cpu_mixed" | "zero_src_gpu" => fixture.job_words[job_base..job_base + 2].fill(0),
            "bad_size" => fixture.job_words[job_base + (count - 1) * 8 + 4] = 0,
            "bad_destination" => fixture.job_words
                [job_base + (count - 1) * 8 + 2..job_base + (count - 1) * 8 + 4]
                .fill(0),
            _ => (),
        }
        write(
            fixture.jobs.as_ref(),
            bytemuck::cast_slice(&fixture.job_words),
        );
        let before_weights = read(fixture.weights.as_ref());
        let before_jobs = read(fixture.jobs.as_ref());
        let before_journal = read(fixture.journal.as_ref());
        write(
            stale_target.as_ref(),
            &vec![POISON; stale_target.len_bytes()],
        );
        let mut stale = vec![POISON; fixture.commands.len_bytes()];
        // Stale entries are themselves legal, disjoint copies into an isolated canary arena.
        // A missed converter rewrite is observable without exposing invalid addresses to KHR.
        let src = source.imported.device_addr().unwrap() + source.ranges[0][0].start as u64;
        for j in 0..CAPACITY {
            let dst = stale_target.device_addr().unwrap() + (j * 16) as u64;
            let command = [
                src as u32,
                (src >> 32) as u32,
                dst as u32,
                (dst >> 32) as u32,
                16,
                0,
                0,
                0,
            ];
            let offset = (ORDINAL * CAPACITY + j) * 32;
            stale[offset..offset + 32].copy_from_slice(bytemuck::cast_slice(&command));
        }
        write(fixture.commands.as_ref(), &stale);
        let recorded_count = if case == "count_over_capacity" {
            CAPACITY
        } else {
            3 * fixture.rows * p.used
        };
        let rec = be.recorder().unwrap();
        rec.seed_barrier();
        rec.begin_owned_indirect_copy(
            fixture.journal.as_ref(),
            fixture.jobs.as_ref(),
            fixture.commands.as_ref(),
            fixture.dummy.as_ref(),
            ORDINAL as u32,
            recorded_count as u32,
        )
        .unwrap();
        rec.global_barrier();
        rec.publish_host_reads();
        rec.finish().unwrap();
        let valid_batch = matches!(case, "stale_padding" | "cpu_only" | "cpu_mixed");
        let mut expected_weights = before_weights;
        let mut expected_commands = stale;
        let mut expected_dummy = vec![POISON; fixture.dummy.len_bytes()];
        expected_dummy[..4].copy_from_slice(&[0x12, 0x34, 0x56, 0x78]);
        for j in 0..recorded_count {
            let job = &fixture.job_words[job_base + j * 8..job_base + j * 8 + 8];
            let real = valid_batch && j < count && (job[0] != 0 || job[1] != 0);
            let command = if real {
                let block = job[5] as usize;
                let range = &source.ranges[block / p.used][block % p.used];
                expected_weights[range.clone()].copy_from_slice(&source.bytes(0)[range.clone()]);
                [job[0], job[1], job[2], job[3], job[4], 0, 0, 0]
            } else {
                let offset = 16 + (ORDINAL * CAPACITY + j) * 16;
                expected_dummy[offset..offset + 4].copy_from_slice(&[0x12, 0x34, 0x56, 0x78]);
                let src = fixture.dummy.device_addr().unwrap();
                let dst = src + offset as u64;
                [
                    src as u32,
                    (src >> 32) as u32,
                    dst as u32,
                    (dst >> 32) as u32,
                    4,
                    0,
                    0,
                    0,
                ]
            };
            let offset = (ORDINAL * CAPACITY + j) * 32;
            expected_commands[offset..offset + 32].copy_from_slice(bytemuck::cast_slice(&command));
        }
        assert_eq!(
            read(fixture.weights.as_ref()),
            expected_weights,
            "case={case} payload/guards"
        );
        assert_eq!(
            read(fixture.commands.as_ref()),
            expected_commands,
            "case={case} command/ordinal guards"
        );
        assert_eq!(
            read(fixture.dummy.as_ref()),
            expected_dummy,
            "case={case} unique dummy destinations"
        );
        assert_eq!(
            read(stale_target.as_ref()),
            vec![POISON; stale_target.len_bytes()],
            "case={case} stale commands executed"
        );
        assert_eq!(
            read(fixture.jobs.as_ref()),
            before_jobs,
            "case={case} immutable jobs"
        );
        assert_eq!(
            read(fixture.journal.as_ref()),
            before_journal,
            "case={case} immutable journal"
        );
    }
}

#[path = "recorder_owned_fused_tests.rs"]
mod fused;

#[path = "recorder_owned_indirect_hint_tests.rs"]
mod hint;
