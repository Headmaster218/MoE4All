//! GPU↔host parity for the NEWLY covered MoE id-GEMV dtypes (the dense-parity extension of the
//! expert-kernel floor): fp4 (MXFP4/NVFP4), ternary (TQ1_0/TQ2_0), the grid i-quants
//! (IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S), and the float banks (BF16/F16/F32). Each
//! dtype is proven on all FOUR kernel shapes — `linear_native_id` (single-slot),
//! `linear_native_id_multi` (all-slots-in-one-dispatch), and both `_paged` twins under eviction
//! churn (more distinct experts than pager slots, mirroring `pager_gemv_parity.rs`).
//!
//! Synthetic banks: pseudo-random block bytes with the SCALE fields patched to small, sane
//! values. That is a VALID encoding for every format here — codebook/grid index bits cover their
//! full table ranges, sign/ternary bit-math is total (the GPU shader and the host reference
//! implement the identical llama.cpp bit-mash) — so no per-format quantizer is needed. The host
//! reference is `infr_gguf::dequant::dequant_block` (the production host dequant, ported
//! arm-for-arm from ggml-quants.c), which keeps this test honest against the SAME decode the CPU
//! backend trusts rather than a re-implementation living next to the shader.
//!
//! The 12 previously covered affine/codebook formats keep their parity coverage in
//! `pager_gemv_parity.rs` / `pager_gemv_multi_parity.rs` / adapter.rs's batched MoE tests.
//!
//! Run: `cargo test -p infr-vulkan --test moe_id_gemv_new_formats_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::linear::pad_to_u32_align;
use infr_vulkan::pager::GpuPager;
use infr_vulkan::VulkanBackend;

/// Deterministic byte stream (SplitMix64-ish) so failures reproduce.
struct Rng(u64);
impl Rng {
    fn byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

/// (elements per block, bytes per block) for each dtype under test.
fn block_geom(dt: DType) -> (usize, usize) {
    match dt {
        DType::Mxfp4 => (32, 17),
        DType::Nvfp4 => (64, 36),
        DType::Tq1_0 => (256, 54),
        DType::Tq2_0 => (256, 66),
        DType::Q2_0 => (64, 18),
        DType::Iq2Xxs => (256, 66),
        DType::Iq2Xs => (256, 74),
        DType::Iq2S => (256, 82),
        DType::Iq3Xxs => (256, 98),
        DType::Iq3S => (256, 110),
        DType::Q5K => (256, 176),
        DType::Q6K => (256, 210),
        DType::Iq1S => (256, 50),
        DType::Iq1M => (256, 56),
        DType::Bf16 | DType::F16 => (1, 2),
        DType::F32 => (1, 4),
        other => panic!("no geometry for {other:?}"),
    }
}

/// One synthetic VALID block: random payload bytes, scale fields patched small (see module doc).
fn synth_block(dt: DType, rng: &mut Rng) -> Vec<u8> {
    let (_, bpb) = block_geom(dt);
    let mut b: Vec<u8> = (0..bpb).map(|_| rng.byte()).collect();
    let d16 = half::f16::from_f32(0.02).to_le_bytes();
    match dt {
        // e8m0 scale byte: 122..=125 → 2^-6..2^-3 (e8m0_half(x) = 2^(x-128) for x >= 2)
        DType::Mxfp4 => b[0] = 122 + (rng.byte() & 3),
        // ue4m3 scale bytes: keep in [0x18, 0x37] → small positive scales, never 0/0x7F
        DType::Nvfp4 => {
            for s in b.iter_mut().take(4) {
                *s = 0x18 + (rng.byte() & 0x1F);
            }
        }
        // trailing f16 d
        DType::Tq1_0 => b[52..54].copy_from_slice(&d16),
        DType::Tq2_0 => b[64..66].copy_from_slice(&d16),
        // leading f16 d (Bonsai ternary; 2-bit codes cover their full {0,1,2,3} range)
        DType::Q2_0 => b[0..2].copy_from_slice(&d16),
        // leading f16 d
        DType::Iq2Xxs | DType::Iq2Xs | DType::Iq2S | DType::Iq3Xxs | DType::Iq3S | DType::Iq1S => {
            b[0..2].copy_from_slice(&d16)
        }
        // K-quants used by the paged-SG regression below. Q5_K carries leading d/dmin; Q6_K's
        // single d follows ql[128] + qh[64] + scales[16]. Random payload is otherwise valid.
        DType::Q5K => {
            b[0..2].copy_from_slice(&d16);
            b[2..4].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        }
        DType::Q6K => b[208..210].copy_from_slice(&d16),
        // IQ1_M spreads its f16 d across the four scale-u16s' TOP NIBBLES (bytes 48..56); keep
        // the low 12 bits random (real 3-bit sub-scales) and plant d's nibbles on top.
        DType::Iq1M => {
            let d_bits = half::f16::from_f32(0.02).to_bits();
            for i in 0..4usize {
                let lo = u16::from_le_bytes([b[48 + 2 * i], b[49 + 2 * i]]) & 0x0FFF;
                let w = lo | (((d_bits >> (4 * i)) & 0xF) << 12);
                b[48 + 2 * i..50 + 2 * i].copy_from_slice(&w.to_le_bytes());
            }
        }
        // floats: small finite values built directly in the storage format (exact both sides)
        DType::Bf16 => {
            // sign(1) | exp 0x7B..0x7E (2^-4..2^-1) | mantissa(7)
            let bits: u16 = (((rng.byte() & 1) as u16) << 15)
                | (((0x7B + (rng.byte() & 3) as u16) & 0xFF) << 7)
                | (rng.byte() & 0x7F) as u16;
            b.copy_from_slice(&bits.to_le_bytes());
        }
        DType::F16 => {
            // sign(1) | exp 11..14 of 31 (2^-4..2^-1) | mantissa(10)
            let bits: u16 = (((rng.byte() & 1) as u16) << 15)
                | ((11 + (rng.byte() & 3) as u16) << 10)
                | ((rng.byte() as u16) << 2 | (rng.byte() & 3) as u16);
            b.copy_from_slice(&bits.to_le_bytes());
        }
        DType::F32 => {
            let v = ((rng.byte() as f32) - 127.5) * 0.004;
            b.copy_from_slice(&v.to_le_bytes());
        }
        other => panic!("no synth for {other:?}"),
    }
    b
}

/// A whole expert bank (`n_elems` elements, block-aligned) of valid synthetic blocks.
fn synth_bank(dt: DType, n_elems: usize, seed: u64) -> Vec<u8> {
    let (epb, bpb) = block_geom(dt);
    assert_eq!(n_elems % epb, 0, "bank must be block-aligned");
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n_elems / epb * bpb);
    for _ in 0..n_elems / epb {
        out.extend_from_slice(&synth_block(dt, &mut rng));
    }
    out
}

fn host_gemv(w_dequant: &[f32], x: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
    (0..out_f)
        .map(|o| {
            (0..in_f)
                .map(|i| w_dequant[o * in_f + i] * x[i])
                .sum::<f32>()
        })
        .collect()
}

fn assert_close(dt: DType, kind: &str, got: &[f32], want: &[f32]) {
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-3 + 1e-3 * w.abs(),
            "{dt:?} {kind} mismatch at {i}: got {g} want {w}"
        );
    }
}

const NEW_DTYPES: &[DType] = &[
    DType::Mxfp4,
    DType::Nvfp4,
    DType::Tq1_0,
    DType::Tq2_0,
    DType::Q2_0,
    DType::Iq2Xxs,
    DType::Iq2Xs,
    DType::Iq2S,
    DType::Iq3Xxs,
    DType::Iq3S,
    DType::Iq1S,
    DType::Iq1M,
    DType::Bf16,
    DType::F16,
    DType::F32,
];

/// Resident-bank floor: `linear_native_id` (every slot) + `linear_native_id_multi` (all slots in
/// one dispatch), per new dtype, vs the host dequant reference.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn new_dtype_id_gemv_matches_host() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // in_f = 256 satisfies every block size here (32/64/256/1); stride stays block-aligned.
    let (in_f, out_f, n_expert) = (256usize, 4usize, 3usize);
    let stride = in_f * out_f;

    for &dt in NEW_DTYPES {
        let bank = synth_bank(dt, n_expert * stride, 0x5eed ^ dt as u64);
        let host_w = infr_gguf::dequant::dequant_block(dt, &bank).unwrap();
        assert_eq!(host_w.len(), n_expert * stride);

        let wbuf = be
            .alloc(pad_to_u32_align(&bank).len(), BufferUsage::Weights)
            .unwrap();
        be.upload(wbuf.as_ref(), &pad_to_u32_align(&bank)).unwrap();

        let x: Vec<f32> = (0..in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
        be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

        // ids scrambled so a wrong-expert read is a detectable mismatch, not accidental equality.
        let ids: Vec<u32> = vec![2, 0, 1];
        let ids_buf = be.alloc(ids.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&ids))
            .unwrap();

        let y_slots: Vec<_> = (0..n_expert)
            .map(|_| be.alloc(out_f * 4, BufferUsage::Activations).unwrap())
            .collect();
        let y_multi = be
            .alloc(n_expert * out_f * 4, BufferUsage::Activations)
            .unwrap();

        let rec = be.recorder().unwrap();
        for (slot, yb) in y_slots.iter().enumerate() {
            rec.linear_native_id(
                dt,
                wbuf.as_ref(),
                ids_buf.as_ref(),
                slot,
                stride,
                x_buf.as_ref(),
                yb.as_ref(),
                1,
                in_f,
                out_f,
            );
        }
        rec.linear_native_id_multi(
            dt,
            wbuf.as_ref(),
            ids_buf.as_ref(),
            n_expert,
            stride,
            x_buf.as_ref(),
            false,
            y_multi.as_ref(),
            in_f,
            out_f,
            1,
        );
        rec.finish().unwrap();

        for (slot, yb) in y_slots.iter().enumerate() {
            let e = ids[slot] as usize;
            let mut out = vec![0u8; out_f * 4];
            be.download(yb.as_ref(), &mut out).unwrap();
            let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out).to_vec();
            let want = host_gemv(&host_w[e * stride..(e + 1) * stride], &x, in_f, out_f);
            assert_close(dt, &format!("id slot {slot}"), &got, &want);
        }
        let mut out = vec![0u8; n_expert * out_f * 4];
        be.download(y_multi.as_ref(), &mut out).unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out).to_vec();
        for slot in 0..n_expert {
            let e = ids[slot] as usize;
            let want = host_gemv(&host_w[e * stride..(e + 1) * stride], &x, in_f, out_f);
            assert_close(
                dt,
                &format!("idm slot {slot}"),
                &got[slot * out_f..(slot + 1) * out_f],
                &want,
            );
        }
        println!("{dt:?}: resident id + idm OK");
    }
}

/// Paged floor under eviction churn: `linear_native_id_paged` + `linear_native_id_multi_paged`
/// through a 2-slot `GpuPager` serving 4 experts (every step past the warm-up evicts the LRU
/// resident expert and reuses its slot — the coherent-but-wrong bug class the `NW()` word-base
/// doc in native_decode.glsl records).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn new_dtype_id_gemv_paged_matches_host_under_eviction_churn() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f, n_expert) = (256usize, 4usize, 4usize);
    let stride = in_f * out_f;

    for &dt in NEW_DTYPES {
        let (epb, bpb) = block_geom(dt);
        let stride_bytes = stride / epb * bpb;
        let banks: Vec<Vec<u8>> = (0..n_expert)
            .map(|e| synth_bank(dt, stride, 0xfeed ^ dt as u64 ^ (e as u64) << 32))
            .collect();
        let host_w: Vec<Vec<f32>> = banks
            .iter()
            .map(|b| infr_gguf::dequant::dequant_block(dt, b).unwrap())
            .collect();

        let x: Vec<f32> = (0..in_f).map(|i| ((i % 13) as f32 - 6.0) * 0.04).collect();
        let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
        be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

        // 2 slots for 4 experts: each pair below evicts the previous pair.
        let mut pager = GpuPager::new(&be, n_expert, 2, stride_bytes).unwrap();
        let staging = be.alloc_uninit(stride_bytes, BufferUsage::Staging).unwrap();
        let ids_buf = be.alloc(2 * 4, BufferUsage::Activations).unwrap();
        let y_id = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        let y_idm = be.alloc(2 * out_f * 4, BufferUsage::Activations).unwrap();

        let mut evicted = false;
        for pair in [[0u32, 1], [2, 3], [1, 2], [3, 0]] {
            for &eid in &pair {
                pager
                    .ensure_resident(&be, staging.as_ref(), eid, &banks[eid as usize])
                    .unwrap();
            }
            pager.flush_lut(&be).unwrap();
            be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&pair))
                .unwrap();

            let rec = be.recorder().unwrap();
            // slot 1 of the pair through the single-slot kernel, both through the multi kernel
            // (lut_base = 0: this synthetic single-layer bank's local ids ARE its LUT indices).
            rec.linear_native_id_paged(
                dt,
                pager.arena_addr(),
                pager.slot_bytes() as u32,
                pager.lut_buffer(),
                ids_buf.as_ref(),
                1,
                0,
                x_buf.as_ref(),
                y_id.as_ref(),
                1,
                in_f,
                out_f,
            );
            rec.linear_native_id_multi_paged(
                dt,
                pager.arena_addr(),
                pager.slot_bytes() as u32,
                pager.lut_buffer(),
                ids_buf.as_ref(),
                2,
                0,
                x_buf.as_ref(),
                false,
                y_idm.as_ref(),
                in_f,
                out_f,
                1,
            );
            rec.finish().unwrap();

            let mut out = vec![0u8; out_f * 4];
            be.download(y_id.as_ref(), &mut out).unwrap();
            let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out).to_vec();
            let want = host_gemv(&host_w[pair[1] as usize], &x, in_f, out_f);
            assert_close(dt, &format!("paged id expert {}", pair[1]), &got, &want);

            let mut out = vec![0u8; 2 * out_f * 4];
            be.download(y_idm.as_ref(), &mut out).unwrap();
            let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out).to_vec();
            for (slot, &eid) in pair.iter().enumerate() {
                let want = host_gemv(&host_w[eid as usize], &x, in_f, out_f);
                assert_close(
                    dt,
                    &format!("paged idm expert {eid}"),
                    &got[slot * out_f..(slot + 1) * out_f],
                    &want,
                );
            }
            evicted |= pager.stats().evictions > 0;
        }
        assert!(evicted, "{dt:?}: churn sequence must actually evict");
        println!("{dt:?}: paged id + idm under eviction churn OK");
    }
}

/// The paged decode route must use the same reassociation-tolerant subgroup+NR kernels as the
/// resident route for the three deliberately enrolled heavy formats. `out_f=2048` enters the
/// default SG band; `rows=1` is decode. Scrambled ids and a nontrivial LUT placement prove the
/// paged shader resolves `lut[lut_base + id]` rather than treating ids as physical slots.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn paged_sg_id_gemv_matches_host() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f, n_expert) = (256usize, 2048usize, 4usize);
    let stride = in_f * out_f;
    let ids = [3u32, 0, 2];
    let ids_buf = be.alloc(ids.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&ids))
        .unwrap();
    let x: Vec<f32> = (0..in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.015).collect();
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

    for &dt in &[DType::Q5K, DType::Q6K, DType::Iq3S] {
        let (epb, bpb) = block_geom(dt);
        let stride_bytes = stride / epb * bpb;
        let banks: Vec<Vec<u8>> = (0..n_expert)
            .map(|e| synth_bank(dt, stride, 0xa117 ^ dt as u64 ^ ((e as u64) << 32)))
            .collect();
        let host: Vec<Vec<f32>> = banks
            .iter()
            .map(|b| infr_gguf::dequant::dequant_block(dt, b).unwrap())
            .collect();
        let mut pager = GpuPager::new(&be, n_expert, ids.len(), stride_bytes).unwrap();
        let staging = be.alloc_uninit(stride_bytes, BufferUsage::Staging).unwrap();
        // Load in a different order from ids so logical expert id != physical slot.
        for &eid in &[0u32, 2, 3] {
            pager
                .ensure_resident(&be, staging.as_ref(), eid, &banks[eid as usize])
                .unwrap();
        }
        pager.flush_lut(&be).unwrap();
        let y = be
            .alloc(ids.len() * out_f * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native_id_multi_paged(
            dt,
            pager.arena_addr(),
            pager.slot_bytes() as u32,
            pager.lut_buffer(),
            ids_buf.as_ref(),
            ids.len(),
            0,
            x_buf.as_ref(),
            false,
            y.as_ref(),
            in_f,
            out_f,
            1,
        );
        rec.finish().unwrap();

        let mut out = vec![0u8; ids.len() * out_f * 4];
        be.download(y.as_ref(), &mut out).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&out);
        for (slot, &eid) in ids.iter().enumerate() {
            let want = host_gemv(&host[eid as usize], &x, in_f, out_f);
            assert_close(
                dt,
                &format!("paged SG expert {eid}"),
                &got[slot * out_f..(slot + 1) * out_f],
                &want,
            );
        }
        println!("{dt:?}: paged SG idm OK");
    }
}

/// Backlog B39: the native-block kernels walk K as `nsub = in_f / 32` whole 32-element sub-blocks,
/// so an expert bank narrower than 32 ran ZERO iterations — and since the GEMV writes its
/// accumulator unconditionally, `linear_native_id_multi` completed cleanly and stored an exact
/// `0.0` over every output (a `-7.0` sentinel in `y` came back as zeros against a host reference of
/// ~3.1). No driver error, no validation message, no seam error, and nothing a zero-init check
/// could tell from a real answer. The dense `Op::Linear` path had the identical hazard at
/// `in_f = 8`.
///
/// The floor is now asserted at dispatch rather than lifted. A partial sub-block is only decodable
/// for the three UNBLOCKED dtypes here (BF16/F16/F32); every blocked format's own GGUF layout
/// already forbids a row shorter than its block, so a masked-tail kernel would have to be written
/// into the whole `native_gemv*`/`native_mmv*`/`embed_gather` family to buy correctness in a case
/// no model can reach.
///
/// Both halves are pinned so the guard cannot pass by rejecting everything: BF16 at the smallest
/// LEGAL width (`in_f = 32`) still matches the host reference with a non-degenerate answer, and
/// the same bank at `in_f = 8` panics instead of quietly writing zeros. The recorder survives the
/// caught panic — the guard runs before any recording, so the segment below is finished normally
/// and no in-flight recorder leaks its descriptor pools (the validation layer reports those at
/// `vkDestroyDevice`). No panic-hook fiddling: `set_hook` is process-global and `cargo test` runs
/// tests in parallel, so silencing it would swallow a genuinely failing test's backtrace too;
/// libtest captures per-test stderr and prints it only on failure.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn id_gemv_multi_rejects_sub_block_in_f() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // BF16 is one of the three dtypes for which a sub-32 bank is even representable — a blocked
    // format cannot express in_f = 8 at all, which is why the floor is a guard and not a kernel.
    let dt = DType::Bf16;
    let (in_f, out_f, n_expert) = (32usize, 4usize, 2usize);
    let stride = in_f * out_f;
    let bank = synth_bank(dt, n_expert * stride, 0x5eed ^ 0xb39);
    let host_w = infr_gguf::dequant::dequant_block(dt, &bank).unwrap();
    let wbuf = be
        .alloc(pad_to_u32_align(&bank).len(), BufferUsage::Weights)
        .unwrap();
    be.upload(wbuf.as_ref(), &pad_to_u32_align(&bank)).unwrap();

    let x: Vec<f32> = (0..in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let ids: Vec<u32> = vec![1, 0];
    let ids_buf = be.alloc(ids.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&ids))
        .unwrap();
    let y = be
        .alloc(n_expert * out_f * 4, BufferUsage::Activations)
        .unwrap();

    let rec = be.recorder().unwrap();
    rec.linear_native_id_multi(
        dt,
        wbuf.as_ref(),
        ids_buf.as_ref(),
        n_expert,
        stride,
        x_buf.as_ref(),
        false,
        y.as_ref(),
        in_f,
        out_f,
        1,
    );
    // Same bank, same buffers, K one sub-block short of the floor.
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rec.linear_native_id_multi(
            dt,
            wbuf.as_ref(),
            ids_buf.as_ref(),
            n_expert,
            stride,
            x_buf.as_ref(),
            false,
            y.as_ref(),
            8,
            out_f,
            1,
        );
    }));
    let payload = caught.expect_err(
        "in_f = 8 must be refused at dispatch: the kernel's `nsub = in_f / 32` loop runs zero \
         times and writes dst all-zero, which no later check can distinguish from a real answer",
    );
    let msg = payload
        .downcast_ref::<String>()
        .expect("the K guard panics with a formatted message")
        .clone();
    assert!(
        msg.contains("in_f=8") && msg.contains("multiple of 32"),
        "refused for the wrong reason: {msg}"
    );
    rec.finish().unwrap();

    // The legal-width dispatch recorded before the refusal still ran, and still agrees with the
    // host dequant — so the guard rejects the off-grid K only, and "all zeros" would be a
    // detectable answer here rather than a coincidence.
    let mut out = vec![0u8; n_expert * out_f * 4];
    be.download(y.as_ref(), &mut out).unwrap();
    let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out).to_vec();
    for (slot, &eid) in ids.iter().enumerate() {
        let e = eid as usize;
        let want = host_gemv(&host_w[e * stride..(e + 1) * stride], &x, in_f, out_f);
        assert!(
            want.iter().any(|v| v.abs() > 1e-3),
            "reference for slot {slot} is ~zero — the fixture cannot tell a no-op dispatch apart"
        );
        assert_close(
            dt,
            &format!("idm slot {slot} at the in_f=32 floor"),
            &got[slot * out_f..(slot + 1) * out_f],
            &want,
        );
    }
}
