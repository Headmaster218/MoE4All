#![cfg(test)]

//! Self-contained Recorder regression, independent of the research campaign modules.
use super::*;
use infr_core::{backend::BufferUsage, DType};

const EXPERTS: usize = 5;
const WBASE: usize = 512;
const LUT_BASE: usize = 3;
const PREFIX_ROWS: usize = 3;
const PAD_ROWS: usize = 64;
const POISON: f32 = -13457.25;
const COUNTS: [[u32; EXPERTS]; 4] = [
    [0, 0, 0, 0, 0],
    [0, 1, 0, 3, 4],
    [0, 31, 1, 33, 0],
    [0, 32, 0, 65, 1],
];

#[derive(Clone, Copy, Debug)]
struct Profile {
    dtype: DType,
    k: usize,
    n: usize,
    paged: bool,
}

impl Profile {
    fn block_bytes(self) -> usize {
        if self.dtype == DType::Q3K {
            110
        } else {
            210
        }
    }

    fn names(self) -> (&'static str, &'static str) {
        match (self.dtype, self.paged) {
            (DType::Q3K, false) => (
                "native_gemm_mmq_q3_k_xp32",
                "native_gemm_mmq_q3_k_packed_xp32",
            ),
            (DType::Q3K, true) => (
                "native_gemm_mmq_q3_k_xpg32",
                "native_gemm_mmq_q3_k_packed_xpg32",
            ),
            (DType::Q6K, true) => (
                "native_gemm_mmq_q6k_xpg32",
                "native_gemm_mmq_q6k_packed_xpg32",
            ),
            (DType::Q6K, false) => (
                "native_gemm_mmq_q6k_xp32w",
                "native_gemm_mmq_q6k_packed_xp32w",
            ),
            _ => unreachable!(),
        }
    }
}

fn profiles() -> Vec<Profile> {
    let mut out = Vec::new();
    for (dtype, shapes) in [
        (DType::Q3K, [(3072, 1024), (1024, 3072)]),
        (DType::Q6K, [(2048, 512), (512, 2048)]),
    ] {
        for (k, n) in shapes {
            for paged in [false, true] {
                out.push(Profile { dtype, k, n, paged });
            }
        }
    }
    out
}

fn payload(p: Profile, expert: usize) -> Vec<u8> {
    let bytes = p.block_bytes();
    let mut data = vec![0; p.k * p.n / 256 * bytes];
    let mut rng = 0xc61e_0200u64 + expert as u64 * 7919;
    for (index, block) in data.chunks_exact_mut(bytes).enumerate() {
        for byte in block.iter_mut() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (rng >> 32) as u8;
        }
        let d = [1.0 / 512.0, -1.0 / 1024.0, 1.0 / 2048.0][(index + expert) % 3];
        block[bytes - 2..].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    }
    data
}

// Scalar GGUF element equations, independent of the shaders' packed-word decode.
fn decode(dtype: DType, block: &[u8], pos: usize) -> (i32, f32) {
    let half = pos / 128;
    let group = pos % 128 / 32;
    let lane = pos % 32;
    let (code, scale) = if dtype == DType::Q3K {
        let low = (block[32 + half * 32 + lane] >> (2 * group)) & 3;
        let high = (block[lane] >> (half * 4 + group)) & 1;
        let sub = pos / 16;
        let b = sub % 4;
        let (lo, shift) = match sub / 4 {
            0 => (block[96 + b] & 15, 0),
            1 => (block[100 + b] & 15, 2),
            2 => (block[96 + b] >> 4, 4),
            _ => (block[100 + b] >> 4, 6),
        };
        (
            i32::from(low | (high << 2)) - 4,
            i32::from(lo | (((block[104 + b] >> shift) & 3) << 4)) - 32,
        )
    } else {
        let low = (block[half * 64 + (group % 2) * 32 + lane] >> (group / 2 * 4)) & 15;
        let high = (block[128 + half * 32 + lane] >> (2 * group)) & 3;
        (
            i32::from(low | (high << 4)) - 32,
            i32::from(block[192 + half * 8 + group * 2 + lane / 16] as i8),
        )
    };
    let end = block.len();
    let d = half::f16::from_le_bytes([block[end - 2], block[end - 1]]).to_f32();
    (code, d * scale as f32)
}

fn upload(be: &VulkanBackend, bytes: &[u8], usage: BufferUsage) -> Box<dyn Buffer> {
    let b = be.alloc(bytes.len(), usage).unwrap();
    be.upload(b.as_ref(), bytes).unwrap();
    b
}

fn download(be: &VulkanBackend, b: &dyn Buffer) -> Vec<u8> {
    let mut bytes = vec![0; b.len_bytes()];
    be.download(b, &mut bytes).unwrap();
    bytes
}

struct Weights {
    raw: Vec<u8>,
    bank: Box<dyn Buffer>,
    stride: usize,
    prefix: usize,
}

impl Weights {
    fn new(be: &VulkanBackend, p: Profile) -> Self {
        let prefix = WBASE / 256 * p.block_bytes();
        let size = p.k * p.n / 256 * p.block_bytes();
        let stride = size + 2 * prefix;
        let mut raw = vec![0xa5; stride * EXPERTS];
        for expert in 0..EXPERTS {
            let slot = Self::slot(p, expert);
            raw[slot * stride + prefix..slot * stride + prefix + size]
                .copy_from_slice(&payload(p, expert));
        }
        let bank = upload(be, &raw, BufferUsage::Weights);
        Self {
            raw,
            bank,
            stride,
            prefix,
        }
    }

    fn slot(p: Profile, expert: usize) -> usize {
        if p.paged {
            [2, 4, 1, 0, 3][expert]
        } else {
            expert
        }
    }
}

fn offsets(counts: &[u32; EXPERTS]) -> [u32; EXPERTS] {
    let mut next = PREFIX_ROWS as u32;
    std::array::from_fn(|e| {
        let first = next;
        next += counts[e];
        first
    })
}

struct Output {
    bits: Vec<u32>,
    qa: Vec<u8>,
    da: Vec<u8>,
    sa: Vec<u8>,
}

fn oracle(p: Profile, w: &Weights, counts: &[u32; EXPERTS], off: &[u32; EXPERTS], out: &Output) {
    for expert in 0..EXPERTS {
        if counts[expert] == 0 {
            continue;
        }
        let first = off[expert] as usize;
        let last = first + counts[expert] as usize - 1;
        for row in [first, (first + last) / 2, last] {
            for col in [0, 31, 63, 64, 127, p.n / 2 + 1, p.n - 1] {
                let mut expected = 0.0f64;
                let mut magnitude = 0.0f64;
                for start in (0..p.k).step_by(16) {
                    let g = col * p.k + start;
                    let base =
                        Weights::slot(p, expert) * w.stride + w.prefix + g / 256 * p.block_bytes();
                    let block = &w.raw[base..base + p.block_bytes()];
                    let mut dot = 0;
                    for j in 0..16 {
                        dot += decode(p.dtype, block, g % 256 + j).0
                            * i32::from(out.qa[row * p.k + start + j] as i8);
                    }
                    let scale = decode(p.dtype, block, g % 256).1;
                    let a = 2 * (row * (p.k / 32) + start / 32);
                    let d = half::f16::from_le_bytes([out.da[a], out.da[a + 1]]).to_f32();
                    let term = f64::from(d) * f64::from(scale) * f64::from(dot);
                    expected += term;
                    magnitude += term.abs();
                }
                let actual = f64::from(f32::from_bits(out.bits[row * p.n + col]));
                let tolerance =
                    1e-4 + 4.0 * f64::from(f32::EPSILON) * (p.k / 16 + 4) as f64 * magnitude;
                assert!((actual - expected).abs() <= tolerance,
                    "baseline address/decode {p:?} expert={expert} row={row} col={col}: {actual} vs {expected}, tol={tolerance}");
            }
        }
    }
}

fn run(be: &VulkanBackend, p: Profile, w: &Weights, counts: [u32; EXPERTS]) -> Output {
    let used_rows = counts.iter().map(|&x| x as usize).sum::<usize>();
    let rows = used_rows.max(1);
    let quant_rows = PREFIX_ROWS + used_rows;
    let padded = quant_rows + PAD_ROWS;
    let off = offsets(&counts);
    let values: Vec<f32> = (0..quant_rows * p.k)
        .map(|i| ((i.wrapping_mul(73) + i / 64 * 31) % 521) as f32 / 521.0 - 0.5)
        .collect();
    let input_bytes = bytemuck::cast_slice(&values);
    let x = upload(be, input_bytes, BufferUsage::Activations);
    let qa = upload(be, &vec![0x5a; padded * p.k], BufferUsage::Activations);
    let halves = padded * (p.k / 32) * 2;
    let da = upload(be, &vec![0x22; halves], BufferUsage::Activations);
    let sa = upload(be, &vec![0x22; halves], BufferUsage::Activations);
    let cnt = upload(be, bytemuck::cast_slice(&counts), BufferUsage::Activations);
    let offs = upload(be, bytemuck::cast_slice(&off), BufferUsage::Activations);
    let poison = vec![POISON.to_bits(); padded * p.n];
    let c = upload(be, bytemuck::cast_slice(&poison), BufferUsage::Readback);
    // Current Q3/Q6 shader LUT is uvec2 absolute BDA, NOT the historical doc's slot index.
    let mut lut_words = vec![0u32; 2 * (LUT_BASE + EXPERTS + 2)];
    for e in 0..EXPERTS {
        if counts[e] != 0 {
            let addr =
                w.bank.device_addr().unwrap() + (Weights::slot(p, e) * w.stride + w.prefix) as u64;
            lut_words[2 * (LUT_BASE + e)] = addr as u32;
            lut_words[2 * (LUT_BASE + e) + 1] = (addr >> 32) as u32;
        }
    }
    let lut = upload(
        be,
        bytemuck::cast_slice(&lut_words),
        BufferUsage::Activations,
    );
    let rec = be.recorder().unwrap();
    rec.quant_q8(
        x.as_ref(),
        qa.as_ref(),
        da.as_ref(),
        sa.as_ref(),
        quant_rows,
        p.k,
    );
    rec.finish().unwrap();
    let quant = [
        download(be, qa.as_ref()),
        download(be, da.as_ref()),
        download(be, sa.as_ref()),
    ];
    assert!(quant[0][..quant_rows * p.k]
        .iter()
        .all(|&b| b as i8 != -128));
    for values in &quant[1..] {
        assert!(values
            .chunks_exact(2)
            .all(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32().is_finite()));
    }
    assert!(quant[0][quant_rows * p.k..].iter().all(|&b| b == 0x5a));
    assert!(quant[1][quant_rows * (p.k / 32) * 2..]
        .iter()
        .all(|&b| b == 0x22));
    assert!(quant[2][quant_rows * (p.k / 32) * 2..]
        .iter()
        .all(|&b| b == 0x22));
    let rec = be.recorder().unwrap();
    if p.paged {
        rec.matmul_mmq_experts_paged(
            p.dtype,
            "kquant_decode_contract",
            qa.as_ref(),
            da.as_ref(),
            None,
            w.bank.device_addr().unwrap(),
            w.stride as u32,
            lut.as_ref(),
            LUT_BASE,
            cnt.as_ref(),
            offs.as_ref(),
            c.as_ref(),
            rows,
            p.k,
            p.n,
            EXPERTS,
            1,
        );
    } else {
        rec.matmul_mmq_experts(
            p.dtype,
            "kquant_decode_contract",
            qa.as_ref(),
            da.as_ref(),
            None,
            w.bank.as_ref(),
            WBASE,
            p.k * p.n + 2 * WBASE,
            cnt.as_ref(),
            offs.as_ref(),
            c.as_ref(),
            rows,
            p.k,
            p.n,
            EXPERTS,
            1,
        );
    }
    rec.finish().unwrap();
    assert_eq!(download(be, x.as_ref()), input_bytes);
    assert_eq!(download(be, w.bank.as_ref()), w.raw);
    assert_eq!(
        download(be, cnt.as_ref()),
        bytemuck::cast_slice::<u32, u8>(&counts)
    );
    assert_eq!(
        download(be, offs.as_ref()),
        bytemuck::cast_slice::<u32, u8>(&off)
    );
    assert_eq!(
        download(be, lut.as_ref()),
        bytemuck::cast_slice::<u32, u8>(&lut_words)
    );
    for (buffer, before) in [qa.as_ref(), da.as_ref(), sa.as_ref()]
        .into_iter()
        .zip(&quant)
    {
        assert_eq!(
            &download(be, buffer),
            before,
            "MMQ changed quantized activations/padding"
        );
    }
    let bytes = download(be, c.as_ref());
    let bits: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
        .collect();
    for (i, &b) in bits.iter().enumerate() {
        if i < PREFIX_ROWS * p.n || i >= quant_rows * p.n {
            assert_eq!(b, POISON.to_bits(), "{p:?} counts={counts:?} canary at {i}");
        } else {
            assert!(
                f32::from_bits(b).is_finite() && b != POISON.to_bits(),
                "{p:?} unwritten/nonfinite C at {i}"
            );
        }
    }
    let [qa, da, sa] = quant;
    Output { bits, qa, da, sa }
}

#[test]
fn kquant_decode_layout_and_scalar_oracle_cpu() {
    for p in profiles() {
        assert_eq!(
            p.n % if p.dtype == DType::Q6K && !p.paged {
                128
            } else {
                64
            },
            0
        );
        let stride = p.k * p.n + 2 * WBASE;
        assert_eq!(
            expert_stride_bytes(p.dtype, stride) as usize,
            stride / 256 * p.block_bytes()
        );
        for counts in COUNTS {
            let rows = counts.iter().map(|&c| c as usize).sum::<usize>().max(1);
            assert!(rows / EXPERTS <= MOE_EXPERT_SMALL_TILE_AVG_ROWS);
            assert!(counts.iter().all(|&c| c as usize <= rows));
            let off = offsets(&counts);
            assert_eq!(off[0], PREFIX_ROWS as u32);
            assert_eq!(
                off[EXPERTS - 1] + counts[EXPERTS - 1],
                PREFIX_ROWS as u32 + counts.iter().sum::<u32>()
            );
        }
        let small = Profile { k: 256, n: 3, ..p };
        let raw = payload(small, 1);
        let factored = infr_gguf::dequant::dequant_factored(p.dtype, &raw).unwrap();
        let mut codes = std::collections::BTreeSet::new();
        for g in 0..768 {
            let bd = g / 256 * p.block_bytes();
            let (q, dl) = decode(p.dtype, &raw[bd..bd + p.block_bytes()], g % 256);
            codes.insert(q);
            let center = if p.dtype == DType::Q3K { 4 } else { 32 };
            assert_eq!(q, i32::from(factored.codes[g]) - center);
            assert_eq!(
                dl.to_bits(),
                (factored.dd[g / 256 * 2].to_f32() * f32::from(factored.scm[g / 16 * 2])).to_bits()
            );
        }
        assert_eq!(codes.len(), if p.dtype == DType::Q3K { 8 } else { 64 });
    }
}

#[test]
#[ignore = "requires a Windows/RDNA3 discrete Vulkan GPU: asserts real packed Recorder selection; no benchmark"]
fn kquant_decode_recorder_flags_exact() {
    for p in profiles() {
        let backend = |packed| {
            let mut cfg = (*VulkanBackend::cfg_from_env().unwrap()).clone();
            cfg.kernels.vulkan.kquant_packed_decode = packed;
            VulkanBackend::new_with(std::sync::Arc::new(cfg)).unwrap()
        };
        let off = backend(false);
        let on = backend(true);
        assert!(on.prefers_kquant_packed_decode(), "selected device is outside production packed gate; baseline/baseline parity is NOT coverage");
        let a = Weights::new(&off, p);
        let b = Weights::new(&on, p);
        for counts in COUNTS {
            let reference = run(&off, p, &a, counts);
            oracle(p, &a, &counts, &offsets(&counts), &reference);
            let actual = run(&on, p, &b, counts);
            assert_eq!(reference.qa, actual.qa, "same QA required");
            assert_eq!(reference.da, actual.da, "same DA required");
            assert_eq!(reference.sa, actual.sa, "same Sact required");
            assert_eq!(
                reference.bits, actual.bits,
                "all raw C exact-f32 including guards: {p:?} {counts:?}"
            );
        }
        let (baseline, packed) = p.names();
        let off_names = off.built_kernel_names();
        let on_names = on.built_kernel_names();
        assert!(off_names.contains(&baseline) && !off_names.contains(&packed));
        assert!(on_names.contains(&packed) && !on_names.contains(&baseline));
        eprintln!("KQUANT_CONTRACT {p:?} cases=4 flags_off_on=true all_raw_C_exact=true canaries=true oracle=true actual_variants=true");
    }
}
