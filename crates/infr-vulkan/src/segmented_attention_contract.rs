#![cfg(test)]

//! Portable regression contracts for the accepted segmented Q8 split-K path.
//!
//! Mount as a child of recorder.rs. CPU contracts run normally; the three GPU
//! contracts are explicitly ignored and require a serial, correctly selected GPU.
//! A01 route/Canvas tests require Windows discrete RDNA3. Resource failures fail
//! loudly. No external shader files, report output, timing, or new dependencies.
//! 108 same-specialization pairs + 120 real Recorder pairs + 18 Canvas/wrapper pairs.
//! Generic F32 and optimized QK_F16/PV_F16 are not asserted bit-identical.
use super::*;
use infr_core::backend::BufferUsage;
use std::result::Result;

#[path = "segmented_attention_canvas_contract.rs"]
mod canvas;
#[path = "segmented_attention_host_contract.rs"]
mod host;

const HD: usize = 256;
const NKV: usize = 2;
const SEG_ROWS: usize = 32 * 1024;
const VALUES_PER_SEGMENT: usize = SEG_ROWS * NKV * HD;
const SEG_SHIFT: u32 = VALUES_PER_SEGMENT.trailing_zeros();
const POISON: u32 = 0x7fc0_1234;
const CANARY: u32 = 0xc5f3_5555;

fn download(be: &VulkanBackend, buffer: &dyn Buffer) -> Vec<f32> {
    let mut values = vec![0.0; buffer.len_bytes() / 4];
    be.download(buffer, bytemuck::cast_slice_mut(&mut values))
        .unwrap();
    values
}

fn device_probe(be: &VulkanBackend) {
    let limits = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    }
    .limits;
    assert!(be.caps().f16, "contract requires shaderFloat16");
    assert!(
        limits.max_compute_shared_memory_size >= 32768,
        "contract requires 32 KiB LDS; no silent skip"
    );
    assert!(limits.max_compute_work_group_invocations >= 256);
    assert!(limits.max_compute_work_group_size[0] >= 256);
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    len: usize,
    nh: usize,
    chunk: usize,
    replay: bool,
}

impl Shape {
    fn live(self) -> usize {
        self.len.div_ceil(self.chunk)
    }
    fn stride(self) -> usize {
        self.live() + if self.replay { 3 } else { 0 }
    }
}

struct Guard {
    buffer: Box<dyn Buffer>,
    prefix: usize,
    words: usize,
}

impl Guard {
    fn new(be: &VulkanBackend, words: usize, alignment: usize) -> Result<Self, String> {
        let prefix = 256usize.next_multiple_of(alignment.max(4)) / 4;
        let buffer = be
            .alloc((prefix + words + 64) * 4, BufferUsage::Activations)
            .map_err(|e| format!("guard allocation: {e}"))?;
        let result = Self {
            buffer,
            prefix,
            words,
        };
        result.reset(be);
        Ok(result)
    }
    fn reset(&self, be: &VulkanBackend) {
        let mut words = vec![CANARY; self.buffer.len_bytes() / 4];
        words[self.prefix..self.prefix + self.words].fill(POISON);
        be.upload(self.buffer.as_ref(), bytemuck::cast_slice(&words))
            .unwrap();
    }
    fn binding(&self) -> vk::DescriptorBufferInfo {
        let mut info = Recorder::vkb(self.buffer.as_ref());
        info.offset += (self.prefix * 4) as u64;
        info.range = (self.words * 4) as u64;
        info
    }
    fn read(&self, be: &VulkanBackend) -> Vec<f32> {
        let all = download(be, self.buffer.as_ref());
        assert!(
            all[..self.prefix].iter().all(|v| v.to_bits() == CANARY),
            "prefix overwrite"
        );
        assert!(
            all[self.prefix + self.words..]
                .iter()
                .all(|v| v.to_bits() == CANARY),
            "suffix overwrite"
        );
        all[self.prefix..self.prefix + self.words].to_vec()
    }
}

fn raw_buffer(
    be: &VulkanBackend,
    bytes: &[u8],
    usage: BufferUsage,
) -> Result<Box<dyn Buffer>, String> {
    let buffer = be
        .alloc(bytes.len(), usage)
        .map_err(|e| format!("input allocation: {e}"))?;
    be.upload(buffer.as_ref(), bytes)
        .map_err(|e| format!("input upload: {e}"))?;
    Ok(buffer)
}

struct Scratch {
    pm: Guard,
    pl: Guard,
    pa: Guard,
    out: Guard,
    args: Guard,
}

impl Scratch {
    fn new(be: &VulkanBackend, s: Shape, align: usize) -> Result<Self, String> {
        Ok(Self {
            pm: Guard::new(be, s.nh * s.stride(), align)?,
            pl: Guard::new(be, s.nh * s.stride(), align)?,
            pa: Guard::new(be, s.nh * s.stride() * HD, align)?,
            out: Guard::new(be, s.nh * HD, align)?,
            args: Guard::new(be, 4, align)?,
        })
    }
    fn reset(&self, be: &VulkanBackend) {
        for g in [&self.pm, &self.pl, &self.pa, &self.out, &self.args] {
            g.reset(be);
        }
    }
    fn capture(&self, be: &VulkanBackend, s: Shape, group: usize) -> Capture {
        let result = Capture {
            pm: self.pm.read(be),
            pl: self.pl.read(be),
            pa: self.pa.read(be),
            out: self.out.read(be),
        };
        if s.replay {
            let args: Vec<u32> = self.args.read(be).iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                args,
                [(s.nh / group * s.live()) as u32, 1, 1, s.live() as u32],
                "live prologue ABI"
            );
        } else {
            assert!(self.args.read(be).iter().all(|v| v.to_bits() == POISON));
        }
        result
    }
}

struct Capture {
    pm: Vec<f32>,
    pl: Vec<f32>,
    pa: Vec<f32>,
    out: Vec<f32>,
}

#[derive(Clone, Copy)]
struct Variant {
    group: usize,
    kernel: ComputeKernel,
}

// Separate acceptance contract: the LUT extension must be byte-exact to the SAME
// flat specialization, independently of its older FP32 generic-vs-half comparison.
struct DenseGuardedInput {
    buffer: Box<dyn Buffer>,
    address: u64,
    prefix: usize,
    payload: usize,
}

impl DenseGuardedInput {
    fn new(be: &VulkanBackend, payload: &[u8], prefix: usize) -> Result<Self, String> {
        let mut bytes = vec![0x3du8; prefix + payload.len() + 256];
        bytes[prefix..prefix + payload.len()].copy_from_slice(payload);
        bytes[prefix + payload.len()..].fill(0xa7);
        let buffer = raw_buffer(be, &bytes, BufferUsage::KvCache)?;
        let address = buffer
            .device_addr()
            .ok_or("dense BDA allocation missing address")?
            + prefix as u64;
        Ok(Self {
            buffer,
            address,
            prefix,
            payload: payload.len(),
        })
    }
    fn verify_guards(&self, be: &VulkanBackend) {
        let mut bytes = vec![0u8; self.buffer.len_bytes()];
        be.download(self.buffer.as_ref(), &mut bytes).unwrap();
        assert!(
            bytes[..self.prefix].iter().all(|v| *v == 0x3d),
            "input prefix written"
        );
        assert!(
            bytes[self.prefix + self.payload..]
                .iter()
                .all(|v| *v == 0xa7),
            "input suffix written"
        );
    }
}

fn random_word(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn dense_planar(elements: usize, seed: u32) -> Vec<u8> {
    assert_eq!(elements % 32, 0);
    let mut state = seed;
    assert_ne!(state, 0);
    let mut bytes = vec![0u8; elements + elements / 32 * 2];
    for word in bytes[..elements].chunks_exact_mut(4) {
        word.copy_from_slice(&random_word(&mut state).to_le_bytes());
    }
    for (block, scale) in bytes[elements..].chunks_exact_mut(2).enumerate() {
        let bits = random_word(&mut state);
        let sign = if bits & 1 == 0 { 0.0 } else { -1.0 };
        let magnitude = 0.0003 + ((bits >> 8) as f32 / 16777216.0) * 0.02;
        let mut value =
            half::f16::from_f32(if sign == 0.0 { magnitude } else { -magnitude }).to_bits();
        if block % 97 == 0 {
            value = 0;
        }
        if block % 101 == 0 {
            value = 0x8000;
        }
        if block % 113 == 0 {
            value = 0x001b | if bits & 1 == 0 { 0 } else { 0x8000 };
        }
        scale.copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn dense_segment_from_flat(flat: &[u8], len: usize, segment: usize) -> Vec<u8> {
    let flat_elements = len * NKV * HD;
    let start = segment * VALUES_PER_SEGMENT;
    let active = (flat_elements - start).min(VALUES_PER_SEGMENT);
    let mut bytes = vec![0x7fu8; VALUES_PER_SEGMENT + VALUES_PER_SEGMENT / 32 * 2];
    for scale in bytes[VALUES_PER_SEGMENT..].chunks_exact_mut(2) {
        scale.copy_from_slice(&0x7e00u16.to_le_bytes());
    }
    bytes[..active].copy_from_slice(&flat[start..start + active]);
    bytes[VALUES_PER_SEGMENT..VALUES_PER_SEGMENT + active / 32 * 2].copy_from_slice(
        &flat[flat_elements + start / 32 * 2..flat_elements + (start + active) / 32 * 2],
    );
    assert_eq!(&bytes[..active], &flat[start..start + active]);
    assert_eq!(
        &bytes[VALUES_PER_SEGMENT..VALUES_PER_SEGMENT + active / 32 * 2],
        &flat[flat_elements + start / 32 * 2..flat_elements + (start + active) / 32 * 2]
    );
    bytes
}

struct DenseContractInput {
    q: Box<dyn Buffer>,
    flat_k: DenseGuardedInput,
    flat_v: DenseGuardedInput,
    kt: Box<dyn Buffer>,
    vt: Box<dyn Buffer>,
    planes: Vec<DenseGuardedInput>,
}

impl DenseContractInput {
    fn new(be: &VulkanBackend, len: usize, nh: usize, lane: usize) -> Result<Self, String> {
        let seed = 0x4751_a017u32.wrapping_add(lane as u32 * 0x10203);
        let elements = len * NKV * HD;
        let fk = dense_planar(elements, seed);
        let fv = dense_planar(elements, seed ^ 0x5d7a_139b);
        let mut state = seed ^ 0x1234_5678;
        let query: Vec<u16> = (0..nh * HD)
            .map(|i| {
                let word = random_word(&mut state);
                let value = ((word >> 8) as f32 / 16777216.0 - 0.5) * 1.5;
                half::f16::from_f32(if i % 37 == 0 { 0.0 } else { value }).to_bits()
            })
            .collect();
        let q = raw_buffer(be, bytemuck::cast_slice(&query), BufferUsage::Activations)?;
        // Nonzero BDA offsets differ between layouts and lanes, avoiding a base-zero fixture.
        let flat_k = DenseGuardedInput::new(be, &fk, 64 + lane * 64)?;
        let flat_v = DenseGuardedInput::new(be, &fv, 128 + lane * 64)?;
        let mut planes = Vec::new();
        let mut ka = vec![0u64; len.div_ceil(SEG_ROWS) + 1];
        let mut va = ka.clone();
        // Allocate in reverse logical order. The LUT, not physical adjacency, owns order.
        for segment in (0..len.div_ceil(SEG_ROWS)).rev() {
            for value in [true, false] {
                let bytes = dense_segment_from_flat(if value { &fv } else { &fk }, len, segment);
                let plane = DenseGuardedInput::new(be, &bytes, 256 + (lane + segment) * 64)?;
                if value {
                    va[segment] = plane.address;
                } else {
                    ka[segment] = plane.address;
                }
                planes.push(plane);
            }
        }
        let kt = raw_buffer(be, bytemuck::cast_slice(&ka), BufferUsage::Activations)?;
        let vt = raw_buffer(be, bytemuck::cast_slice(&va), BufferUsage::Activations)?;
        eprintln!("A01CONTRACT INPUT len={len} nh={nh} lane={lane} seed={seed} dense_query=true signed_codes=true signed_nondyadic_scales=true exact_code_scale_concatenation=true physical_segments={} tail_rows={}", len.div_ceil(SEG_ROWS), len % SEG_ROWS);
        Ok(Self {
            q,
            flat_k,
            flat_v,
            kt,
            vt,
            planes,
        })
    }
    fn verify_guards(&self, be: &VulkanBackend) {
        self.flat_k.verify_guards(be);
        self.flat_v.verify_guards(be);
        for p in &self.planes {
            p.verify_guards(be);
        }
    }
}

fn contract_kernel(
    be: &VulkanBackend,
    group: usize,
    cap: usize,
    segmented: bool,
) -> Result<ComputeKernel, String> {
    assert!(matches!(group, 1 | 4) && matches!(cap, 512 | 1024));
    let (name, words) = if segmented {
        let (_, words) = crate::gemm::attn_decode_segmented_kernel(group == 4, cap == 1024);
        let name = match (group, cap) {
            (1, 512) => "contract_segmented_d8_c512",
            (1, 1024) => "contract_segmented_d8_c1024",
            (4, 512) => "contract_segmented_gqa4_c512",
            (4, 1024) => "contract_segmented_gqa4_c1024",
            _ => unreachable!(),
        };
        (name, words)
    } else {
        match (group, cap) {
            (1, 512) => (
                "contract_flat_d8_c512",
                crate::gemm::attn_decode_hd256_q8_c512_spv(),
            ),
            (1, 1024) => (
                "contract_flat_d8_c1024",
                crate::gemm::attn_decode_hd256_q8_spv(),
            ),
            // Static GQA4 has a fixed 1024-score slab. Q8_CHUNK1024 only changes
            // SELF_CHUNK replay (absent here); the c512/c1024 static SPVs were
            // byte-identical in the original acceptance evidence. This is NOT
            // substituting a different arithmetic or replay specialization.
            (4, 512) => (
                "contract_flat_gqa4_c512",
                crate::gemm::attn_decode_hd256_q8_gqa4_spv(),
            ),
            (4, 1024) => (
                "contract_flat_gqa4_c1024",
                crate::gemm::attn_decode_hd256_q8_gqa4_spv(),
            ),
            _ => unreachable!(),
        }
    };
    assert!(
        words.len() >= 5 && words[0] == 0x0723_0203,
        "invalid embedded SPIR-V {name}"
    );
    Ok(be.kernel_sg(name, words, 6, 60, 32))
}

#[test]
fn segmented_embedded_shader_tuple_contract() {
    // Exercise the shipped getters without constructing a Vulkan device. Missing
    // OUT_DIR artifacts are build errors, never runtime fallbacks to old probes.
    for (group, cap, expected) in [
        (1, 512, "attn_decode_hd256_q8_seg_d8"),
        (1, 1024, "attn_decode_hd256_q8_seg_d8_c1024"),
        (4, 512, "attn_decode_hd256_q8_seg_gqa4"),
        (4, 1024, "attn_decode_hd256_q8_seg_gqa4_c1024"),
    ] {
        let (name, words) = crate::gemm::attn_decode_segmented_kernel(group == 4, cap == 1024);
        assert_eq!(name, expected);
        assert!(words.len() >= 5);
        assert_eq!(words[0], 0x0723_0203);
        let mut at = 5;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            assert!(count > 0 && at + count <= words.len(), "malformed {name}");
            at += count;
        }
        assert_eq!(at, words.len());
    }
    // Pin the static-only alias used by flat GQA4 c512. A future shader change
    // making Q8_CHUNK1024 affect static GQA4 must first provide a separate flat
    // c512 artifact and update the reference, not silently weaken this contract.
    assert_eq!(
        crate::gemm::attn_decode_segmented_kernel(true, false).1,
        crate::gemm::attn_decode_segmented_kernel(true, true).1,
        "static GQA4 c512/c1024 must remain the same embedded specialization"
    );
}

fn record_contract(
    be: &VulkanBackend,
    r: &Recorder<'_>,
    s: Shape,
    input: &DenseContractInput,
    scratch: &Scratch,
    kernel: ComputeKernel,
    group: usize,
    segmented: bool,
) {
    let (ka, va) = if segmented {
        (0, 0)
    } else {
        (input.flat_k.address, input.flat_v.address)
    };
    let push = [
        s.len as u32,
        s.nh as u32,
        NKV as u32,
        HD as u32,
        s.chunk as u32,
        s.stride() as u32,
        0,
        (1.0f32 / 16.0).to_bits(),
        if segmented {
            SEG_SHIFT
        } else {
            (s.len * NKV * HD) as u32
        },
        s.len as u32 - 1,
        0,
        ka as u32,
        (ka >> 32) as u32,
        va as u32,
        (va >> 32) as u32,
    ];
    let kb = if segmented {
        input.kt.as_ref()
    } else {
        input.flat_k.buffer.as_ref()
    };
    let vb = if segmented {
        input.vt.as_ref()
    } else {
        input.flat_v.buffer.as_ref()
    };
    r.dispatch(
        kernel,
        &[
            Recorder::vkb(input.q.as_ref()),
            Recorder::vkb(kb),
            Recorder::vkb(vb),
            scratch.pm.binding(),
            scratch.pl.binding(),
            scratch.pa.binding(),
        ],
        3,
        bytemuck::cast_slice(&push),
        (s.nh / group * s.live()) as u32,
    );
    let combine = be.kernel_sg(
        "contract_a01_contract_combine_sg",
        crate::gemm::attn_combine_sg_spv(),
        4,
        16,
        32,
    );
    r.dispatch(
        combine,
        &[
            scratch.pm.binding(),
            scratch.pl.binding(),
            scratch.pa.binding(),
            scratch.out.binding(),
        ],
        1,
        bytemuck::cast_slice(&[s.nh as u32, HD as u32, s.stride() as u32, 16]),
        s.nh as u32 * 16,
    );
}

fn exact_contract(a: &Capture, b: &Capture) -> Result<(), String> {
    for (label, aa, bb) in [
        ("pm", &a.pm, &b.pm),
        ("pl", &a.pl, &b.pl),
        ("pacc", &a.pa, &b.pa),
        ("out", &a.out, &b.out),
    ] {
        assert_eq!(aa.len(), bb.len());
        let mut different = 0;
        let mut first = None;
        for (i, (&x, &y)) in aa.iter().zip(bb).enumerate() {
            if !x.is_finite() || !y.is_finite() {
                return Err(format!("{label}[{i}] nonfinite flat={x} seg={y}"));
            }
            if x.to_bits() != y.to_bits() {
                different += 1;
                first.get_or_insert((i, x.to_bits(), y.to_bits()));
            }
        }
        if different != 0 {
            return Err(format!("{label} differing_f32_values={different} first={first:?}; extension contract is BYTE-EXACT, not tolerance"));
        }
    }
    if !a.pl.iter().all(|v| *v > 0.0) {
        return Err("zero/negative softmax denominator".into());
    }
    Ok(())
}

#[test]
#[ignore = "requires serial Vulkan GPU: random dense SAME-specialization flat/seg bit contract"]
fn gqa_segmented_matches_flat_same_specialization_exact() {
    let be = VulkanBackend::new().unwrap();
    device_probe(&be);
    let limits = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    }
    .limits;
    assert!(limits.max_compute_shared_memory_size >= 32768);
    assert!(limits.max_compute_work_group_invocations >= 256);
    assert!(limits.max_compute_work_group_size[0] >= 256);
    eprintln!("A01CONTRACT SETUP precision=QK_F16+PV_F16 query_i8=false same_codes_and_scales=true acceptance=all_partial_and_combined_bytes_equal no_tolerance=true");
    let (mut compared, mut failed) = (0, 0);
    for len in [30_000usize, 32769, 150_000] {
        for nh in [16usize, 32] {
            let inputs = (0..3)
                .map(|lane| DenseContractInput::new(&be, len, nh, lane))
                .collect::<Result<Vec<_>, _>>();
            let inputs = inputs.expect(
                "complete regression matrix requires all fixture resources and embedded shaders",
            );
            for chunk in [511usize, 512, 1024] {
                let s = Shape {
                    len,
                    nh,
                    chunk,
                    replay: false,
                };
                let scratch = (0..3)
                    .map(|_| {
                        Scratch::new(&be, s, limits.min_storage_buffer_offset_alignment as usize)
                    })
                    .collect::<Result<Vec<_>, _>>();
                let scratch = scratch.expect("complete regression matrix requires all fixture resources and embedded shaders");
                for group in [1usize, 4] {
                    let cap = if chunk <= 512 { 512 } else { 1024 };
                    let kernels = contract_kernel(&be, group, cap, false).and_then(|flat| {
                        contract_kernel(&be, group, cap, true).map(|seg| [flat, seg])
                    });
                    let kernels = kernels.expect("complete regression matrix requires all fixture resources and embedded shaders");
                    let mut flat: Option<Vec<Capture>> = None;
                    for (segmented, kernel) in [(false, kernels[0]), (true, kernels[1])] {
                        for sc in &scratch {
                            sc.reset(&be);
                        }
                        let r = be.recorder().unwrap();
                        for lane in 0..3 {
                            record_contract(
                                &be,
                                &r,
                                s,
                                &inputs[lane],
                                &scratch[lane],
                                kernel,
                                group,
                                segmented,
                            );
                        }
                        r.finish().unwrap();
                        let actual = scratch
                            .iter()
                            .map(|sc| sc.capture(&be, s, group))
                            .collect::<Vec<_>>();
                        if !segmented {
                            flat = Some(actual);
                            continue;
                        }
                        let baseline = flat.as_ref().expect("flat captured before segmented");
                        for lane in 0..3 {
                            match exact_contract(&baseline[lane], &actual[lane]) {
                                Ok(()) => eprintln!("A01CONTRACT PASS len={len} nh={nh} chunk={chunk} group={group} lane={lane} partial_bytes_equal=true output_bytes_equal=true"),
                                Err(e) => { failed += 1; eprintln!("A01CONTRACT FAIL len={len} nh={nh} chunk={chunk} group={group} lane={lane} reason={e}"); }
                            }
                            compared += 1;
                        }
                    }
                }
            }
            for input in &inputs {
                input.verify_guards(&be);
            }
        }
    }
    eprintln!("A01CONTRACT COMPLETE compared={compared} failed={failed} expected=108");
    assert_eq!(compared, 108, "Incomplete same-specialization matrix");
    assert_eq!(
        failed, 0,
        "LUT extension changed SAME-specialization bytes; do not integrate"
    );
}

// Route coverage is intentionally separate from shader parity: exercise the real
// Recorder entry points, including descriptor offsets and precision opt-outs.
struct RouteIo {
    q: DenseGuardedInput,
    scratch: Scratch,
}

impl RouteIo {
    fn new(
        be: &VulkanBackend,
        input: &DenseContractInput,
        s: Shape,
        offset: bool,
        lane: usize,
        alignment: usize,
    ) -> Result<Self, String> {
        let mut query = vec![0u8; input.q.len_bytes()];
        be.download(input.q.as_ref(), &mut query)
            .map_err(|e| format!("query readback: {e}"))?;
        let prefix = if offset {
            (256 + lane * 256).next_multiple_of(alignment.max(4))
        } else {
            0
        };
        let q = DenseGuardedInput::new(be, &query, prefix)?;
        let guard = |words, prefix| {
            let buffer = be
                .alloc((prefix + words + 64) * 4, BufferUsage::Activations)
                .map_err(|e| format!("route scratch: {e}"))?;
            let g = Guard {
                buffer,
                prefix,
                words,
            };
            g.reset(be);
            Ok::<_, String>(g)
        };
        Ok(Self {
            q,
            scratch: Scratch {
                pm: guard(s.nh * s.stride(), 0)?,
                pl: guard(s.nh * s.stride(), 0)?,
                pa: guard(s.nh * s.stride() * HD, 0)?,
                out: guard(s.nh * HD, prefix / 4)?,
                args: guard(4, 0)?,
            },
        })
    }
}

fn record_route_control(
    be: &VulkanBackend,
    r: &Recorder<'_>,
    s: Shape,
    input: &DenseContractInput,
    io: &RouteIo,
    v: Variant,
) {
    let sc = &io.scratch;
    let mut qb = Recorder::vkb(io.q.buffer.as_ref());
    qb.offset += io.q.prefix as u64;
    qb.range = io.q.payload as u64;
    let push = [
        s.len as u32,
        s.nh as u32,
        NKV as u32,
        HD as u32,
        s.chunk as u32,
        s.stride() as u32,
        0,
        (1.0f32 / 16.0).to_bits(),
        SEG_SHIFT,
        s.len as u32 - 1,
        0,
        0,
        0,
        0,
        0,
    ];
    r.dispatch(
        v.kernel,
        &[
            qb,
            Recorder::vkb(input.kt.as_ref()),
            Recorder::vkb(input.vt.as_ref()),
            sc.pm.binding(),
            sc.pl.binding(),
            sc.pa.binding(),
        ],
        3,
        bytemuck::cast_slice(&push),
        (s.nh / v.group * s.live()) as u32,
    );
    let combine = be.kernel_sg(
        "contract_a01_route_control_combine_sg",
        crate::gemm::attn_combine_sg_spv(),
        4,
        16,
        32,
    );
    r.dispatch(
        combine,
        &[
            sc.pm.binding(),
            sc.pl.binding(),
            sc.pa.binding(),
            sc.out.binding(),
        ],
        1,
        bytemuck::cast_slice(&[s.nh as u32, HD as u32, s.stride() as u32, 16]),
        s.nh as u32 * 16,
    );
}

#[test]
#[ignore = "requires serial Vulkan GPU: real segmented Recorder route and precision fallback bytes"]
fn gqa_segmented_production_recorder_route_exact() {
    let base = VulkanBackend::cfg_from_env().unwrap();
    let mut compared = 0;
    for mode in [
        "d8",
        "gqa4",
        "disabled",
        "qk_f32",
        "pv_f32",
        "chunk1024_off",
    ] {
        let mut cfg = (*base).clone();
        let vk = &mut cfg.kernels.vulkan;
        vk.attn_decode = true;
        vk.no_attn_hd_spec = false;
        vk.q8_decode_segmented = mode != "disabled";
        vk.q8_decode_d8 = true;
        vk.q8_decode_ls256 = true;
        vk.q8_qk_f16 = mode != "qk_f32";
        vk.q8_pv_f16 = mode != "pv_f32";
        vk.q8_decode_gqa4 = mode != "d8";
        vk.q8_decode_chunk1024 = mode != "chunk1024_off";
        vk.q8_decode_combine_sg = true;
        let be = VulkanBackend::new_with(std::sync::Arc::new(cfg)).unwrap();
        device_probe(&be);
        assert!(
            be.prefers_segmented_q8_decode(),
            "route test requires the production Windows discrete RDNA3 gate"
        );
        let limits = unsafe {
            be.shared
                .instance
                .get_physical_device_properties(be.shared.physical_device)
        }
        .limits;
        assert!(limits.max_compute_shared_memory_size >= 32768);
        let fast = matches!(mode, "d8" | "gqa4");
        // Full actual-context matrix for positive routing; one crossing-boundary
        // shape suffices for each independently tested configuration rejection.
        let shapes = if fast {
            [30_000, 150_000]
                .into_iter()
                .flat_map(|len| {
                    [16, 32].into_iter().flat_map(move |nh| {
                        [512, 1024].into_iter().map(move |chunk| Shape {
                            len,
                            nh,
                            chunk,
                            replay: false,
                        })
                    })
                })
                .collect::<Vec<_>>()
        } else {
            vec![Shape {
                len: 32769,
                nh: 16,
                chunk: if mode == "chunk1024_off" { 1024 } else { 511 },
                replay: false,
            }]
        };
        for s in shapes {
            let group = if mode == "gqa4" { 4 } else { 1 };
            let reference = if fast {
                contract_kernel(&be, group, if s.chunk <= 512 { 512 } else { 1024 }, true)
                    .map(|kernel| Variant { group, kernel })
            } else {
                Ok(Variant {
                    group: 1,
                    kernel: be.kernel_sg(
                        "contract_segmented_generic_control",
                        crate::gemm::attn_partial_seg_q8_spv(),
                        6,
                        60,
                        32,
                    ),
                })
            };
            let reference = reference.expect("all production reference shaders must be embedded");
            let expected = if fast {
                crate::gemm::attn_decode_segmented_kernel(group == 4, s.chunk > 512).0
            } else {
                "attn_partial_seg_q8"
            };
            for lane in 0..3 {
                let input = DenseContractInput::new(&be, s.len, s.nh, lane)
                    .expect("route fixture allocation; a resource failure is not a pass");
                for offset in [false, true] {
                    let io = RouteIo::new(
                        &be,
                        &input,
                        s,
                        offset,
                        lane,
                        limits.min_storage_buffer_offset_alignment as usize,
                    )
                    .unwrap();
                    let sc = &io.scratch;
                    let r = be.recorder().unwrap();
                    if offset {
                        r.attention_kv_split_segmented_off(
                            io.q.buffer.as_ref(),
                            io.q.prefix,
                            input.kt.as_ref(),
                            input.vt.as_ref(),
                            sc.out.buffer.as_ref(),
                            sc.out.prefix * 4,
                            sc.pm.buffer.as_ref(),
                            sc.pl.buffer.as_ref(),
                            sc.pa.buffer.as_ref(),
                            1,
                            s.len - 1,
                            s.len,
                            s.nh,
                            NKV,
                            HD,
                            s.chunk,
                            s.stride(),
                            1.0 / 16.0,
                            0,
                            true,
                            true,
                            SEG_SHIFT,
                        );
                    } else {
                        r.attention_kv_split_segmented(
                            io.q.buffer.as_ref(),
                            input.kt.as_ref(),
                            input.vt.as_ref(),
                            sc.out.buffer.as_ref(),
                            sc.pm.buffer.as_ref(),
                            sc.pl.buffer.as_ref(),
                            sc.pa.buffer.as_ref(),
                            1,
                            s.len - 1,
                            s.len,
                            s.nh,
                            NKV,
                            HD,
                            s.chunk,
                            s.stride(),
                            1.0 / 16.0,
                            0,
                            true,
                            true,
                            SEG_SHIFT,
                        );
                    }
                    r.finish().unwrap();
                    let names = be.built_kernel_names();
                    assert!(
                        names.contains(&expected),
                        "mode={mode} expected={expected} actual={names:?}"
                    );
                    if !fast {
                        assert!(!names.iter().any(|n| n.starts_with("attn_decode_hd256_q8_seg_")),
                            "precision/config opt-out incorrectly selected a segmented half specialization");
                    }
                    let routed = sc.capture(&be, s, group);
                    sc.reset(&be);
                    let r = be.recorder().unwrap();
                    record_route_control(&be, &r, s, &input, &io, reference);
                    r.finish().unwrap();
                    let direct = sc.capture(&be, s, group);
                    exact_contract(&direct, &routed).unwrap_or_else(|e| {
                        panic!("A01ROUTE FAIL mode={mode} shape={s:?} lane={lane} offset={offset}: {e}")
                    });
                    io.q.verify_guards(&be);
                    compared += 1;
                    eprintln!("A01ROUTE PASS mode={mode} len={} nh={} chunk={} lane={lane} io_offset={offset} selected={expected} partial_bytes_equal=true output_bytes_equal=true", s.len, s.nh, s.chunk);
                }
                input.verify_guards(&be);
            }
        }
    }
    eprintln!("A01ROUTE COMPLETE compared={compared} expected=120");
    assert_eq!(compared, 120, "Incomplete route acceptance matrix");
}
