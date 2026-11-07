//! Proves the decode-GEMV / weight-read kernels correctly read their weight by 64-bit device
//! address (`native_weight_addr.glsl`) — i.e. that passing an explicit `arena_addr` base, at any
//! byte offset, decodes the right bytes. Two forms, depending on whether the family still has a
//! genuinely-dispatched bound-SSBO twin:
//!
//! * RESIDENT-VS-BDA (`run_case`/`run_mmv_case` + `assert_case`, most tests below): for families a
//!   resident (bound-SSBO) build is STILL compiled for (production dispatches it, or an eager/parity
//!   caller binds the weight directly — see build.rs's `mechanism_b_resident`, the `-DSTREAMED` dual
//!   sources), this proves the address-addressed build computes BIT-IDENTICALLY to it. The two
//!   builds differ only in where `NW(i)` sources its word from — the dequant math and the
//!   accumulation order in `native_decode.glsl` are the same code — so anything short of bitwise
//!   equality is a bug, not a tolerance question.
//! * OFFSET-INVARIANCE (`run_case_offset_invariant`/`run_mmv_case_offset_invariant` +
//!   `assert_offset_invariant`): for families whose resident build no longer exists (retired
//!   alongside the eager/parity-recorder callers that were its sole consumers — the address-
//!   addressed build was ALREADY the only thing production ever dispatched, so a resident compare
//!   would either not compile or just re-derive the same result from itself), this drops the
//!   resident leg and instead runs the SAME kernel twice — once with its weight at arena offset 0,
//!   once with the identical bytes parked behind a garbage prefix at a non-zero offset — asserting
//!   the two outputs match bit-for-bit.
//!
//! Either way, the second, load-bearing assertion is the NON-ZERO ARENA OFFSET: the resident-BDA
//! integration places every tensor at its own byte offset inside ONE multi-GiB arena and passes
//! `arena_addr + tensor_off` as the kernel's base. A build that only works at offset 0 would pass a
//! naive parity test and then read garbage for every tensor but the first, so each case is run
//! twice — once at offset 0, once at a non-zero offset — and both legs must agree (with the
//! resident result, or with each other, per the form above).
//!
//! WEIGHT BYTES: deliberately NOT a faithful quantization of any target tensor. Parity only needs
//! both paths to decode the SAME bytes; what float those bytes mean is irrelevant. Every byte is
//! drawn from `0x00..=0x3F`, which is what makes this safe for EVERY dtype without hand-rolling a
//! per-format quantizer: an f16 is NaN/Inf only when its 5 exponent bits are all ones, which needs
//! bit 14 (0x40) set in the high byte — unreachable when every byte is < 0x40. So all scales
//! decode finite and non-degenerate, no output is NaN (which would make a bitwise compare pass
//! vacuously, hiding the very mis-addressing this test exists to catch), and a wrong address still
//! yields visibly different finite floats.
//!
//! Run: `cargo test -p infr-vulkan --test weight_addr_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

/// Pseudo-random weight bytes in `0x00..=0x3F` — see the module header for why the range matters.
fn synth_weight_bytes(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7;
            (h % 0x40) as u8
        })
        .collect()
}

/// Activations kept small and mixed-sign so a sign/stride error shows up as a magnitude change.
fn synth_x(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect()
}

fn weight_bytes_for(dtype: DType, elems: usize) -> usize {
    let (blk_elems, blk_bytes) = infr_gguf::block_layout(dtype);
    assert_eq!(
        elems % blk_elems,
        0,
        "test shape must be a whole number of {dtype:?} blocks"
    );
    elems / blk_elems * blk_bytes
}

/// Byte offset a tensor sits at inside the shared arena for the non-zero-offset leg. Must be a
/// whole number of blocks (so `w_base`-relative decode still lands on a block boundary) and is
/// 256-byte aligned to satisfy any natural alignment the pointer reads assume.
fn nonzero_off(dtype: DType) -> usize {
    let (_, blk_bytes) = infr_gguf::block_layout(dtype);
    let mut off = blk_bytes * 8;
    off = off.div_ceil(256) * 256;
    // Re-round UP to a block multiple; 256-alignment alone can land mid-block for odd block sizes.
    off.div_ceil(blk_bytes) * blk_bytes
}

struct Case {
    /// Human name for the failure message.
    name: String,
    /// Bits of the resident dispatch's output.
    resident: Vec<u32>,
    /// Bits of the streamed dispatch's output, at arena offset 0 and at a non-zero offset.
    streamed_at0: Vec<u32>,
    streamed_atoff: Vec<u32>,
}

fn bits(v: &[u8]) -> Vec<u32> {
    bytemuck::cast_slice::<u8, u32>(v).to_vec()
}

type Buf = dyn infr_core::backend::Buffer;
/// Records the resident dispatch: `(recorder, weight_ssbo, x, y)`.
type DispatchRes<'f> = dyn Fn(&infr_vulkan::Recorder, &Buf, &Buf, &Buf) + 'f;
/// Records the streamed dispatch: `(recorder, weight_device_address, x, y)`.
type DispatchStr<'f> = dyn Fn(&infr_vulkan::Recorder, u64, &Buf, &Buf) + 'f;

/// Runs one kernel family through resident + both streamed legs. `dispatch_res` records the
/// resident dispatch; `dispatch_str` records the streamed one against a given device address.
#[allow(clippy::too_many_arguments)]
fn run_case(
    be: &VulkanBackend,
    name: String,
    dtype: DType,
    in_f: usize,
    out_f: usize,
    out_elems: usize,
    dispatch_res: &DispatchRes,
    dispatch_str: &DispatchStr,
) -> Case {
    let w_elems = in_f * out_f;
    let w_bytes = weight_bytes_for(dtype, w_elems);
    let w = synth_weight_bytes(w_bytes, in_f * 31 + out_f);

    let x = synth_x(in_f);
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let y_buf = be.alloc(out_elems * 4, BufferUsage::Activations).unwrap();

    // ── Resident: weight in its own bound SSBO ────────────────────────────────────────────────
    let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
    be.upload(w_buf.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_res(&rec, w_buf.as_ref(), x_buf.as_ref(), y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let resident = bits(&out);

    // ── Streamed leg 1: arena offset 0 ────────────────────────────────────────────────────────
    let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
    be.upload(arena0.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_str(&rec, addr0, x_buf.as_ref(), y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let streamed_at0 = bits(&out);

    // ── Streamed leg 2: the SAME weight parked at a non-zero offset in a bigger arena ─────────
    // Mirrors the resident-BDA layout (one arena, many tensors). The prefix is filled with
    // DIFFERENT bytes, so a twin that ignores the offset and reads from the arena base produces a
    // visibly wrong result instead of accidentally matching.
    let off = nonzero_off(dtype);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&w);
    let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena1.as_ref(), &backing).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_str(&rec, addr1 + off as u64, x_buf.as_ref(), y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let streamed_atoff = bits(&out);

    Case {
        name,
        resident,
        streamed_at0,
        streamed_atoff,
    }
}

fn assert_case(c: &Case) {
    assert!(
        c.resident.iter().any(|&b| b != 0),
        "{}: resident output is all zeros — the case is not exercising the kernel",
        c.name
    );
    for (i, (&r, &s)) in c.resident.iter().zip(c.streamed_at0.iter()).enumerate() {
        assert_eq!(
            r,
            s,
            "{}: streamed@0 differs from resident at out {i}: {} vs {} (bits {r:#010x} vs {s:#010x})",
            c.name,
            f32::from_bits(r),
            f32::from_bits(s)
        );
    }
    for (i, (&r, &s)) in c.resident.iter().zip(c.streamed_atoff.iter()).enumerate() {
        assert_eq!(
            r,
            s,
            "{}: streamed@nonzero-offset differs from resident at out {i}: {} vs {} — the twin is \
             ignoring its arena base offset, which breaks every tensor but the first in a shared arena",
            c.name,
            f32::from_bits(r),
            f32::from_bits(s)
        );
    }
}

/// The result of an offset-invariance case: the streamed kernel's output bits at arena offset 0
/// and at a non-zero offset (same weight bytes, different address) — no resident leg. Used for
/// kernel families whose bound-SSBO resident build no longer exists (retired alongside its sole
/// eager/parity-recorder caller — see build.rs's `mechanism_b_resident`), so the ONLY thing left
/// to prove is that the streamed kernel itself doesn't hardcode arena offset 0.
struct OffsetCase {
    /// Human name for the failure message.
    name: String,
    /// Bits of the streamed dispatch's output at arena offset 0.
    at0: Vec<u32>,
    /// Bits of the streamed dispatch's output at a non-zero offset, weight bytes UNCHANGED.
    atoff: Vec<u32>,
}

/// Runs one streamed kernel twice — once with its weight at arena offset 0, once with the SAME
/// weight bytes parked behind a garbage-filled prefix at a non-zero offset — and returns both
/// output-bit vectors. Mirrors [`run_case`]'s streamed legs exactly (same synth weight/x, same
/// non-zero-offset construction); it just drops the resident leg `run_case` also captures.
#[allow(clippy::too_many_arguments)]
fn run_case_offset_invariant(
    be: &VulkanBackend,
    name: String,
    dtype: DType,
    in_f: usize,
    out_f: usize,
    out_elems: usize,
    dispatch_str: &DispatchStr,
) -> OffsetCase {
    let w_elems = in_f * out_f;
    let w_bytes = weight_bytes_for(dtype, w_elems);
    let w = synth_weight_bytes(w_bytes, in_f * 31 + out_f);

    let x = synth_x(in_f);
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let y_buf = be.alloc(out_elems * 4, BufferUsage::Activations).unwrap();

    // ── Leg 1: arena offset 0 ──────────────────────────────────────────────────────────────────
    let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
    be.upload(arena0.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_str(&rec, addr0, x_buf.as_ref(), y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let at0 = bits(&out);

    // ── Leg 2: the SAME weight bytes parked at a non-zero offset in a bigger arena ────────────
    // The prefix is filled with DIFFERENT bytes, so a kernel that ignores the offset and reads
    // from the arena base produces a visibly wrong result instead of accidentally matching.
    let off = nonzero_off(dtype);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&w);
    let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena1.as_ref(), &backing).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_str(&rec, addr1 + off as u64, x_buf.as_ref(), y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let atoff = bits(&out);

    OffsetCase { name, at0, atoff }
}

fn assert_offset_invariant(c: &OffsetCase) {
    assert!(
        c.at0.iter().any(|&b| b != 0),
        "{}: streamed@0 output is all zeros — the case is not exercising the kernel",
        c.name
    );
    for (i, (&a, &b)) in c.at0.iter().zip(c.atoff.iter()).enumerate() {
        assert_eq!(
            a,
            b,
            "{}: streamed@nonzero-offset differs from streamed@0 at out {i}: {} vs {} (bits \
             {a:#010x} vs {b:#010x}) — the kernel is ignoring its arena base offset, which breaks \
             every tensor but the first in a shared arena",
            c.name,
            f32::from_bits(a),
            f32::from_bits(b)
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mrow_offset_invariant() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    for dtype in [DType::Q8_0, DType::Q4K, DType::Q6K] {
        for rows in [2usize, 4, 8] {
            let c = run_case_offset_invariant(
                &be,
                format!("mrow dtype={dtype:?} rows={rows}"),
                dtype,
                in_f,
                out_f,
                rows * out_f,
                &|rec, addr, x, y| {
                    rec.linear_native_mrow_at(dtype, addr, 0, x, y, rows, in_f, out_f)
                },
            );
            assert_offset_invariant(&c);
            println!("ok: {}", c.name);
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mrow_q8_two_rows_offset_invariant() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f, rows) = (256usize, 8usize, 2usize);
    let c = run_case_offset_invariant(
        &be,
        "mrow Q8_0 rows=2 specialization".into(),
        DType::Q8_0,
        in_f,
        out_f,
        rows * out_f,
        &|rec, addr, x, y| rec.linear_native_mrow_at(DType::Q8_0, addr, 0, x, y, rows, in_f, out_f),
    );
    assert_offset_invariant(&c);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn rm_offset_invariant() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    for dtype in [DType::Q4K, DType::Q6K] {
        for rm in [2u32, 4] {
            let c = run_case_offset_invariant(
                &be,
                format!("rm dtype={dtype:?} rm={rm}"),
                dtype,
                in_f,
                out_f,
                out_f,
                &|rec, addr, x, y| rec.linear_native_rm_at(dtype, addr, 0, x, y, in_f, out_f, rm),
            );
            assert_offset_invariant(&c);
            println!("ok: {}", c.name);
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn sg_offset_invariant() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    let dtype = DType::Q6K; // the only dtype with an SG build
    for nr in [2u32, 4, 8] {
        let c = run_case_offset_invariant(
            &be,
            format!("sg dtype={dtype:?} nr={nr}"),
            dtype,
            in_f,
            out_f,
            out_f,
            &|rec, addr, x, y| rec.linear_native_sg_at(dtype, addr, 0, x, y, in_f, out_f, nr),
        );
        assert_offset_invariant(&c);
        println!("ok: {}", c.name);
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn rm_v2_offset_invariant() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    // Variant/dtype pairs that have a build (see gemm::native_rm_variant_spv).
    for (variant, dtype) in [
        ("sg", DType::Q4K),
        ("sg", DType::Q6K),
        ("dbuf", DType::Q4K),
        ("wg128", DType::Q4K),
        ("reg", DType::Q4K),
    ] {
        let c = run_case_offset_invariant(
            &be,
            format!("rm_v2 variant={variant} dtype={dtype:?}"),
            dtype,
            in_f,
            out_f,
            out_f,
            &|rec, addr, x, y| {
                rec.linear_native_rm_v2_at(variant, dtype, addr, 0, x, y, in_f, out_f)
            },
        );
        assert_offset_invariant(&c);
        println!("ok: {}", c.name);
    }
}

/// The `w_base` within-tensor element offset must keep working under BDA — it is what lets one
/// stacked tensor (fused QKV slices, a stacked MoE expert bank) serve several logical weights, and
/// the resident-BDA layout composes it WITH the arena base offset (`arena_addr + tensor_off` as the
/// base, `w_base` as the slice within that tensor). Both must apply, so this checks a non-zero
/// `w_base` against a non-zero arena offset — same weight bytes, streamed twice (arena offset 0 vs
/// a non-zero offset behind a garbage prefix), `w_base` held constant across both legs.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn w_base_composes_with_arena_offset() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let dtype = DType::Q4K;
    let (in_f, out_f) = (256usize, 8usize);
    // Two stacked "experts" in one tensor; the second is selected via w_base.
    let stride_elems = in_f * out_f;
    let w_bytes = weight_bytes_for(dtype, stride_elems * 2);
    let w = synth_weight_bytes(w_bytes, 77);

    let x = synth_x(in_f);
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let y_buf = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();

    // ── Leg 1: arena offset 0, w_base selects the second stacked "expert" ─────────────────────
    let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
    be.upload(arena0.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    rec.linear_native_rm_at(
        dtype,
        addr0,
        stride_elems,
        x_buf.as_ref(),
        y_buf.as_ref(),
        in_f,
        out_f,
        2,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; out_f * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let at0 = bits(&out);
    assert!(
        at0.iter().any(|&b| b != 0),
        "w_base@offset-0 output is all zeros"
    );

    // ── Leg 2: the SAME weight bytes at a non-zero arena offset, SAME w_base ──────────────────
    let off = nonzero_off(dtype);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&w);
    let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena1.as_ref(), &backing).unwrap();
    let rec = be.recorder().unwrap();
    rec.linear_native_rm_at(
        dtype,
        addr1 + off as u64,
        stride_elems,
        x_buf.as_ref(),
        y_buf.as_ref(),
        in_f,
        out_f,
        2,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; out_f * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let atoff = bits(&out);

    for (i, (&a, &b)) in at0.iter().zip(atoff.iter()).enumerate() {
        assert_eq!(
            a,
            b,
            "w_base+arena_off: streamed@nonzero-offset differs from streamed@0 at out {i}: {} vs {}",
            f32::from_bits(a),
            f32::from_bits(b)
        );
    }
    println!("ok: w_base composes with a non-zero arena offset");
}

// ─── int8 dp4a (MMV) family — slice A2 ─────────────────────────────────────────────────────────
// Same resident-vs-streamed bitwise contract as the dequant GEMVs above, with one extra input
// stage: the activation is pre-quantized ONCE by quant_q8 (qa/dact/sact) and both legs read the
// SAME quantized buffers, so any difference is the weight path, never the activation path.

/// Quantized-activation buffers for `rows` rows of `in_f` f32s, produced on-GPU by `quant_q8`.
struct QAct {
    qa: Box<Buf2>,
    dact: Box<Buf2>,
    sact: Box<Buf2>,
}
type Buf2 = dyn infr_core::backend::Buffer;

fn quantize_x(be: &VulkanBackend, rows: usize, in_f: usize) -> QAct {
    let x: Vec<f32> = (0..rows * in_f)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05 + ((i / in_f) as f32) * 0.01)
        .collect();
    let x_buf = be.alloc(rows * in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let nblk = in_f / 32;
    let qa = be.alloc(rows * in_f, BufferUsage::Activations).unwrap();
    let dact = be.alloc(rows * nblk * 2, BufferUsage::Activations).unwrap();
    let sact = be.alloc(rows * nblk * 2, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.quant_q8(
        x_buf.as_ref(),
        qa.as_ref(),
        dact.as_ref(),
        sact.as_ref(),
        rows,
        in_f,
    );
    rec.finish().unwrap();
    QAct { qa, dact, sact }
}

/// Resident + both streamed legs for an int8-dp4a kernel; returns the three output-bit vectors.
#[allow(clippy::too_many_arguments)]
fn run_mmv_case(
    be: &VulkanBackend,
    name: String,
    dtype: DType,
    in_f: usize,
    out_f: usize,
    out_elems: usize,
    q: &QAct,
    dispatch_res: &dyn Fn(&infr_vulkan::Recorder, &Buf2, &Buf2),
    dispatch_str: &dyn Fn(&infr_vulkan::Recorder, u64, &Buf2),
) -> Case {
    let w_bytes = weight_bytes_for(dtype, in_f * out_f);
    let w = synth_weight_bytes(w_bytes, in_f * 31 + out_f);
    let y_buf = be.alloc(out_elems * 4, BufferUsage::Activations).unwrap();

    let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
    be.upload(w_buf.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_res(&rec, w_buf.as_ref(), y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let resident = bits(&out);

    let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
    be.upload(arena0.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_str(&rec, addr0, y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let streamed_at0 = bits(&out);

    let off = nonzero_off(dtype);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&w);
    let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena1.as_ref(), &backing).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_str(&rec, addr1 + off as u64, y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let streamed_atoff = bits(&out);

    let _ = q; // the qa/dact/sact buffers are shared by both legs via the dispatch closures
    Case {
        name,
        resident,
        streamed_at0,
        streamed_atoff,
    }
}

/// Offset-invariance twin of [`run_mmv_case`] for the int8-dp4a family: no resident leg, just the
/// streamed kernel run at arena offset 0 and at a non-zero offset (same weight bytes). Used by the
/// `linear_mmv`/`linear_mmv_mrow` cases — both are BDA wrappers whose "resident" arm was already
/// just a `device_addr()` fork onto the SAME streamed kernel (see those fns' doc), making the old
/// resident-vs-streamed compare vacuous; this asserts the thing that was never actually checked.
#[allow(clippy::too_many_arguments)]
fn run_mmv_case_offset_invariant(
    be: &VulkanBackend,
    name: String,
    dtype: DType,
    in_f: usize,
    out_f: usize,
    out_elems: usize,
    q: &QAct,
    dispatch_str: &dyn Fn(&infr_vulkan::Recorder, u64, &Buf2),
) -> OffsetCase {
    let w_bytes = weight_bytes_for(dtype, in_f * out_f);
    let w = synth_weight_bytes(w_bytes, in_f * 31 + out_f);
    let y_buf = be.alloc(out_elems * 4, BufferUsage::Activations).unwrap();

    let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
    be.upload(arena0.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_str(&rec, addr0, y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let at0 = bits(&out);

    let off = nonzero_off(dtype);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&w);
    let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena1.as_ref(), &backing).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_str(&rec, addr1 + off as u64, y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let atoff = bits(&out);

    let _ = q; // the qa/dact/sact buffers are shared by both legs via the dispatch closure
    OffsetCase { name, at0, atoff }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_offset_invariant() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    let q = quantize_x(&be, 1, in_f);
    for dtype in [DType::Q4K, DType::Q6K, DType::Iq4Xs] {
        let c = run_mmv_case_offset_invariant(
            &be,
            format!("mmv dtype={dtype:?}"),
            dtype,
            in_f,
            out_f,
            out_f,
            &q,
            &|rec, addr, y| {
                rec.linear_mmv_at(
                    dtype,
                    addr,
                    0,
                    q.qa.as_ref(),
                    q.dact.as_ref(),
                    q.sact.as_ref(),
                    y,
                    in_f,
                    out_f,
                )
            },
        );
        assert_offset_invariant(&c);
        println!("ok: {}", c.name);
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_mrow_offset_invariant() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // in_f=256 exercises the OUTS4 layout (in_f < 2048), in_f=2048 the base 2-output layout;
    // rows 1/4 hit the m4 tier, 8 the MR=8 tier, 12 the -DMRV=16 tier. Q3_K covers the
    // funnel-shift (2-aligned block) load path.
    for in_f in [256usize, 2048] {
        let out_f = 8usize;
        for dtype in [DType::Q4K, DType::Q6K, DType::Q3K, DType::Q8_0] {
            for rows in [1usize, 4, 8, 12] {
                let q = quantize_x(&be, rows, in_f);
                let c = run_mmv_case_offset_invariant(
                    &be,
                    format!("mmv_mrow dtype={dtype:?} rows={rows} in_f={in_f}"),
                    dtype,
                    in_f,
                    out_f,
                    rows * out_f,
                    &q,
                    &|rec, addr, y| {
                        rec.linear_mmv_mrow_at(
                            dtype,
                            addr,
                            0,
                            q.qa.as_ref(),
                            q.dact.as_ref(),
                            q.sact.as_ref(),
                            None,
                            y,
                            rows,
                            in_f,
                            out_f,
                        )
                    },
                );
                assert_offset_invariant(&c);
                println!("ok: {}", c.name);
            }
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_mw_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    let q = quantize_x(&be, 1, in_f);
    for dtype in [DType::Q4K, DType::Q6K, DType::Q2K, DType::Q3K, DType::Q5K] {
        for warps in [4u32, 8] {
            let c = run_mmv_case(
                &be,
                format!("mmv_mw dtype={dtype:?} warps={warps}"),
                dtype,
                in_f,
                out_f,
                out_f,
                &q,
                &|rec, w, y| {
                    rec.linear_mmv_mw(
                        dtype,
                        warps,
                        w,
                        0,
                        q.qa.as_ref(),
                        q.dact.as_ref(),
                        q.sact.as_ref(),
                        None,
                        y,
                        in_f,
                        out_f,
                    )
                },
                &|rec, addr, y| {
                    rec.linear_mmv_mw_at(
                        dtype,
                        warps,
                        addr,
                        0,
                        q.qa.as_ref(),
                        q.dact.as_ref(),
                        q.sact.as_ref(),
                        None,
                        y,
                        in_f,
                        out_f,
                    )
                },
            );
            assert_case(&c);
            println!("ok: {}", c.name);
        }
    }
}

// ─── tiled dp4a expert GEMM (MMQ) family — slice A3 ────────────────────────────────────────────
// Dense (non-EXPERT_GRID) parity per dtype. The `_xp*` EXPERT_GRID streamed twins are covered by
// composition: streamed-xp differs from resident-xp ONLY in the rb() weight source (proven here
// per dtype) and from streamed-dense ONLY in the EXPERT_GRID row/stride logic (proven by the
// existing resident `_xp` adapter tests — that logic is the SHARED #else arm of both compiles).
// The xp streamed dispatch lands with the expert router's `arena: Option<u64>` arm at integration.

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmq_dense_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // n%64, k%32 (and k%256 so every K-quant super-block is whole); m=8 pads the C tile to 64 rows.
    let (m, k, n) = (8usize, 256usize, 64usize);
    let q = quantize_x(&be, m, k);
    let c_rows = m.div_ceil(64) * 64;
    for dtype in [
        DType::Q4K,
        DType::Q6K,
        DType::Q8_0,
        DType::Q5_0,
        DType::Q5K,
        DType::Q5_1,
        DType::Q2K,
        DType::Q3K,
        DType::Q4_0,
        DType::Q4_1,
        DType::Iq4Nl,
        DType::Iq4Xs,
        DType::Iq2S,
        DType::Iq3S,
        DType::Q2_0,
    ] {
        let c = run_mmv_case(
            &be,
            format!("mmq dense dtype={dtype:?}"),
            dtype,
            k,
            n,
            c_rows * n,
            &q,
            &|rec, w, y| {
                rec.matmul_mmq(
                    dtype,
                    q.qa.as_ref(),
                    q.dact.as_ref(),
                    q.sact.as_ref(),
                    w,
                    0,
                    y,
                    m,
                    k,
                    n,
                )
            },
            &|rec, addr, y| {
                rec.matmul_mmq_at(
                    dtype,
                    q.qa.as_ref(),
                    q.dact.as_ref(),
                    q.sact.as_ref(),
                    addr,
                    0,
                    y,
                    m,
                    k,
                    n,
                )
            },
        );
        // Padded C rows (m..c_rows) hold garbage from staged-garbage activations in BOTH legs, and
        // the two legs' garbage can differ (the pad rows read whatever followed qa). Compare only
        // the real m rows.
        let real = m * n;
        let trimmed = Case {
            name: c.name,
            resident: c.resident[..real].to_vec(),
            streamed_at0: c.streamed_at0[..real].to_vec(),
            streamed_atoff: c.streamed_atoff[..real].to_vec(),
        };
        assert_case(&trimmed);
        println!("ok: {}", trimmed.name);
    }
}

// ─── id-indexed resident expert family — slice A4 ──────────────────────────────────────────────
// The load-bearing difference from the earlier sections: the STREAMED build REPURPOSES `stride`
// as the per-expert BYTE stride applied on the 64-bit pointer (`w_addr = arena + u64(ids[slot]) *
// u64(stride_bytes)`), replacing the resident u32 element-space multiply that wraps past 2^32
// elements. So every case here selects a NON-ZERO expert id — an id of 0 would pass even if the
// stride scaling were completely broken.

/// A stacked bank of `n_expert` experts (each `in_f`×`out_f` of `dtype`), the per-expert byte
/// stride, and an ids buffer on the GPU.
struct Bank {
    bytes: Vec<u8>,
    stride_bytes: usize,
    stride_elems: usize,
}

fn synth_bank(dtype: DType, in_f: usize, out_f: usize, n_expert: usize) -> Bank {
    let stride_elems = in_f * out_f;
    let stride_bytes = weight_bytes_for(dtype, stride_elems);
    let bytes = synth_weight_bytes(stride_bytes * n_expert, in_f * 7 + out_f);
    Bank {
        bytes,
        stride_bytes,
        stride_elems,
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn id_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f, n_expert) = (256usize, 8usize, 4usize);
    let x = synth_x(in_f);
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let ids_buf = be.alloc(4, BufferUsage::Activations).unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&[2u32]))
        .unwrap();

    // F16 exercises the float NW decode — floats never had a streamed build before this campaign.
    for dtype in [DType::Q4K, DType::Q6K, DType::Q8_0, DType::F16] {
        let bank = synth_bank(dtype, in_f, out_f, n_expert);
        let y_buf = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();

        let w_buf = be.alloc(bank.bytes.len(), BufferUsage::Weights).unwrap();
        be.upload(w_buf.as_ref(), &bank.bytes).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native_id(
            dtype,
            w_buf.as_ref(),
            ids_buf.as_ref(),
            0,
            bank.stride_elems,
            x_buf.as_ref(),
            y_buf.as_ref(),
            1,
            in_f,
            out_f,
        );
        rec.finish().unwrap();
        let mut out = vec![0u8; out_f * 4];
        be.download(y_buf.as_ref(), &mut out).unwrap();
        let resident = bits(&out);
        assert!(
            resident.iter().any(|&b| b != 0),
            "id dtype={dtype:?}: resident output is all zeros"
        );

        // Streamed at a non-zero arena offset (the offset-0 leg is subsumed: expert id 2 already
        // proves base + non-trivial byte offset addressing).
        let off = nonzero_off(dtype);
        let mut backing = synth_weight_bytes(off, 0xBAD);
        backing.extend_from_slice(&bank.bytes);
        let (arena, addr) = be.alloc_arena_bda(backing.len()).unwrap();
        be.upload(arena.as_ref(), &backing).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native_id_at(
            dtype,
            addr + off as u64,
            bank.stride_bytes as u32,
            ids_buf.as_ref(),
            0,
            x_buf.as_ref(),
            y_buf.as_ref(),
            1,
            in_f,
            out_f,
        );
        rec.finish().unwrap();
        let mut out = vec![0u8; out_f * 4];
        be.download(y_buf.as_ref(), &mut out).unwrap();
        let streamed = bits(&out);
        for (i, (&r, &s)) in resident.iter().zip(streamed.iter()).enumerate() {
            assert_eq!(
                r,
                s,
                "id dtype={dtype:?} out {i}: streamed {} != resident {} — expert-id byte-stride \
                 scaling or arena offset broken",
                f32::from_bits(s),
                f32::from_bits(r)
            );
        }
        println!("ok: id dtype={dtype:?}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn idm_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f, n_expert, n_used) = (256usize, 8usize, 4usize, 4usize);
    let x = synth_x(in_f);
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    // Permuted, all-distinct ids: a stride bug scrambles WHICH expert each slot reads, so outputs
    // land wrong even when every expert is resident.
    let ids_buf = be.alloc(n_used * 4, BufferUsage::Activations).unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&[2u32, 0, 3, 1]))
        .unwrap();

    // Q6K takes the SG route on both sides (native_id_sg_choice); Q4K/Q8_0/F16 the tree kernel.
    for dtype in [DType::Q4K, DType::Q6K, DType::Q8_0, DType::F16] {
        let bank = synth_bank(dtype, in_f, out_f, n_expert);
        let y_buf = be
            .alloc(n_used * out_f * 4, BufferUsage::Activations)
            .unwrap();

        let w_buf = be.alloc(bank.bytes.len(), BufferUsage::Weights).unwrap();
        be.upload(w_buf.as_ref(), &bank.bytes).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native_id_multi(
            dtype,
            w_buf.as_ref(),
            ids_buf.as_ref(),
            n_used,
            bank.stride_elems,
            x_buf.as_ref(),
            false,
            y_buf.as_ref(),
            in_f,
            out_f,
            1,
        );
        rec.finish().unwrap();
        let mut out = vec![0u8; n_used * out_f * 4];
        be.download(y_buf.as_ref(), &mut out).unwrap();
        let resident = bits(&out);
        assert!(
            resident.iter().any(|&b| b != 0),
            "idm dtype={dtype:?}: resident output is all zeros"
        );

        let off = nonzero_off(dtype);
        let mut backing = synth_weight_bytes(off, 0xBAD);
        backing.extend_from_slice(&bank.bytes);
        let (arena, addr) = be.alloc_arena_bda(backing.len()).unwrap();
        be.upload(arena.as_ref(), &backing).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native_id_multi_at(
            dtype,
            addr + off as u64,
            bank.stride_bytes as u32,
            ids_buf.as_ref(),
            n_used,
            x_buf.as_ref(),
            false,
            y_buf.as_ref(),
            in_f,
            out_f,
            1,
        );
        rec.finish().unwrap();
        let mut out = vec![0u8; n_used * out_f * 4];
        be.download(y_buf.as_ref(), &mut out).unwrap();
        let streamed = bits(&out);
        for (i, (&r, &s)) in resident.iter().zip(streamed.iter()).enumerate() {
            assert_eq!(
                r,
                s,
                "idm dtype={dtype:?} out {i} (slot {}): streamed {} != resident {}",
                i / out_f,
                f32::from_bits(s),
                f32::from_bits(r)
            );
        }
        println!("ok: idm dtype={dtype:?}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_id_q4k_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f, n_expert, n_used) = (256usize, 8usize, 4usize, 4usize);
    let dtype = DType::Q4K;
    let q = quantize_x(&be, 1, in_f);
    let ids_buf = be.alloc(n_used * 4, BufferUsage::Activations).unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&[2u32, 0, 3, 1]))
        .unwrap();
    let bank = synth_bank(dtype, in_f, out_f, n_expert);
    let y_buf = be
        .alloc(n_used * out_f * 4, BufferUsage::Activations)
        .unwrap();

    let w_buf = be.alloc(bank.bytes.len(), BufferUsage::Weights).unwrap();
    be.upload(w_buf.as_ref(), &bank.bytes).unwrap();
    let rec = be.recorder().unwrap();
    rec.linear_mmv_id_multi_q4k(
        w_buf.as_ref(),
        q.qa.as_ref(),
        q.dact.as_ref(),
        q.sact.as_ref(),
        ids_buf.as_ref(),
        n_used,
        bank.stride_elems,
        y_buf.as_ref(),
        in_f,
        out_f,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; n_used * out_f * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let resident = bits(&out);
    assert!(
        resident.iter().any(|&b| b != 0),
        "mmv_id_q4k: resident output is all zeros"
    );

    let off = nonzero_off(dtype);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&bank.bytes);
    let (arena, addr) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena.as_ref(), &backing).unwrap();
    let rec = be.recorder().unwrap();
    rec.linear_mmv_id_multi_q4k_at(
        addr + off as u64,
        bank.stride_bytes as u32,
        q.qa.as_ref(),
        q.dact.as_ref(),
        q.sact.as_ref(),
        ids_buf.as_ref(),
        n_used,
        y_buf.as_ref(),
        in_f,
        out_f,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; n_used * out_f * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let streamed = bits(&out);
    for (i, (&r, &s)) in resident.iter().zip(streamed.iter()).enumerate() {
        assert_eq!(
            r,
            s,
            "mmv_id_q4k out {i} (slot {}): streamed {} != resident {}",
            i / out_f,
            f32::from_bits(s),
            f32::from_bits(r)
        );
    }
    println!("ok: mmv_id_q4k");
}

// ─── float-weight linear family — slice A4 ─────────────────────────────────────────────────────
// These kernels read TYPED weight arrays (float16_t / float / vec4), so their STREAMED arms use
// typed buffer_references instead of the uint-word NW() seam — parity across each type proves the
// typed pointer reads decode identically to the bound-SSBO loads.

/// `linear`/`linear_bf16`/`linear_f32` are BDA wrappers — each is just a `device_addr()` fork onto
/// the SAME streamed kernel this test drives directly (see those `Recorder` fns' doc), so a
/// resident-vs-streamed compare here was always vacuous. Offset-invariance is the real assertion.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn float_linear_offset_invariant() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);

    // (name, dtype, rows) — rows>1 exercises linear_f32's mrow/vec4 routing.
    for rows in [1usize, 4] {
        let x = synth_x(rows * in_f);
        let x_buf = be.alloc(rows * in_f * 4, BufferUsage::Activations).unwrap();
        be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

        for dtype in [DType::F16, DType::Bf16, DType::F32] {
            let c = run_case_offset_invariant(
                &be,
                format!("float linear dtype={dtype:?} rows={rows}"),
                dtype,
                in_f,
                out_f,
                rows * out_f,
                &|rec, addr, _x, y| match dtype {
                    DType::F16 => rec.linear_at(addr, x_buf.as_ref(), y, rows, in_f, out_f),
                    DType::Bf16 => rec.linear_bf16_at(addr, x_buf.as_ref(), y, rows, in_f, out_f),
                    DType::F32 => rec.linear_f32_at(addr, x_buf.as_ref(), y, rows, in_f, out_f),
                    _ => unreachable!(),
                },
            );
            assert_offset_invariant(&c);
            println!("ok: {}", c.name);
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn f16_noext_and_res_match_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f, rows) = (256usize, 8usize, 1usize);
    let x = synth_x(in_f);
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let res: Vec<f32> = (0..out_f).map(|i| (i as f32) * 0.25 - 1.0).collect();
    let res_buf = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
    be.upload(res_buf.as_ref(), bytemuck::cast_slice(&res))
        .unwrap();

    let c = run_case(
        &be,
        "linear_f16_noext".to_string(),
        DType::F16,
        in_f,
        out_f,
        out_f,
        &|rec, w, _x, y| rec.linear_f16_noext(w, x_buf.as_ref(), y, rows, in_f, out_f),
        &|rec, addr, _x, y| rec.linear_f16_noext_at(addr, x_buf.as_ref(), y, rows, in_f, out_f),
    );
    assert_case(&c);
    println!("ok: {}", c.name);

    let c = run_case(
        &be,
        "linear_res (fused residual)".to_string(),
        DType::F16,
        in_f,
        out_f,
        out_f,
        &|rec, w, _x, y| rec.linear_add(w, x_buf.as_ref(), res_buf.as_ref(), y, rows, in_f, out_f),
        &|rec, addr, _x, y| {
            rec.linear_add_at(addr, x_buf.as_ref(), res_buf.as_ref(), y, rows, in_f, out_f)
        },
    );
    assert_case(&c);
    println!("ok: {}", c.name);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn fma_gemm_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // n%64, k%32; m=8 pads C to 64 rows (compare the real rows only, like the mmq test).
    let (m, k, n) = (8usize, 256usize, 64usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let a_buf = be.alloc(m * k * 4, BufferUsage::Activations).unwrap();
    be.upload(a_buf.as_ref(), bytemuck::cast_slice(&a)).unwrap();
    let c_rows = m.div_ceil(64) * 64;

    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let c = run_case(
            &be,
            format!("fma gemm dtype={dtype:?}"),
            dtype,
            k,
            n,
            c_rows * n,
            &|rec, w, _x, y| rec.matmul_fma(dtype, a_buf.as_ref(), w, 0, y, m, k, n),
            &|rec, addr, _x, y| rec.matmul_fma_at(dtype, a_buf.as_ref(), addr, 0, y, m, k, n),
        );
        let real = m * n;
        let trimmed = Case {
            name: c.name,
            resident: c.resident[..real].to_vec(),
            streamed_at0: c.streamed_at0[..real].to_vec(),
            streamed_atoff: c.streamed_atoff[..real].to_vec(),
        };
        assert_case(&trimmed);
        println!("ok: {}", trimmed.name);
    }
}

// ─── token-embedding gather family — slice A5 ──────────────────────────────────────────────────
// Op::EmbedGather's signature (table + ids -> dst, gathering NON-CONTIGUOUS rows rather than
// computing a dot product) doesn't fit run_case's weight/x/y shape, so this is a bespoke runner —
// same resident-vs-streamed-at-two-offsets double-leg shape as run_case, mirroring the id-family
// tests' bespoke style above. Ids are non-trivial (out of order, includes row 0 AND a high row
// index) so a broken row-stride multiply can't pass by accident the way an all-zero id would.

#[test]
#[ignore = "requires a Vulkan GPU"]
fn embed_gather_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (n_table_rows, ne, rows) = (16usize, 256usize, 4usize);
    let scale = 0.5f32;
    let ids: [i32; 4] = [3, 0, 7, 12];
    let ids_buf = be.alloc(rows * 4, BufferUsage::Activations).unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&ids))
        .unwrap();

    for dtype in [DType::Q4K, DType::Q6K, DType::Q8_0, DType::F16] {
        let w_bytes = weight_bytes_for(dtype, n_table_rows * ne);
        let table = synth_weight_bytes(w_bytes, ne * 13 + n_table_rows);
        let y_buf = be.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();

        // ── Resident: table in its own bound SSBO ─────────────────────────────────────────────
        let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
        be.upload(w_buf.as_ref(), &table).unwrap();
        let rec = be.recorder().unwrap();
        rec.embed_gather(
            dtype,
            w_buf.as_ref(),
            ids_buf.as_ref(),
            y_buf.as_ref(),
            rows,
            ne,
            scale,
        );
        rec.finish().unwrap();
        let mut out = vec![0u8; rows * ne * 4];
        be.download(y_buf.as_ref(), &mut out).unwrap();
        let resident = bits(&out);
        assert!(
            resident.iter().any(|&b| b != 0),
            "embed_gather dtype={dtype:?}: resident output is all zeros — the case is not \
             exercising the kernel"
        );

        // ── Streamed leg 1: arena offset 0 ────────────────────────────────────────────────────
        let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
        be.upload(arena0.as_ref(), &table).unwrap();
        let rec = be.recorder().unwrap();
        rec.embed_gather_at(
            dtype,
            addr0,
            ids_buf.as_ref(),
            y_buf.as_ref(),
            rows,
            ne,
            scale,
            0, // u32-fitting table → original element-offset gather path (issue #77)
        );
        rec.finish().unwrap();
        let mut out = vec![0u8; rows * ne * 4];
        be.download(y_buf.as_ref(), &mut out).unwrap();
        let streamed_at0 = bits(&out);
        for (i, (&r, &s)) in resident.iter().zip(streamed_at0.iter()).enumerate() {
            assert_eq!(
                r,
                s,
                "embed_gather dtype={dtype:?}: streamed@0 differs from resident at out {i}: {} \
                 vs {} (bits {r:#010x} vs {s:#010x})",
                f32::from_bits(r),
                f32::from_bits(s)
            );
        }

        // ── Streamed leg 2: the SAME table parked at a non-zero offset in a bigger arena ──────
        // Mirrors the resident-BDA layout (one arena, many tensors); the prefix is filled with
        // DIFFERENT bytes so a twin that ignores the offset reads visibly wrong data.
        let off = nonzero_off(dtype);
        let mut backing = synth_weight_bytes(off, 0xBAD);
        backing.extend_from_slice(&table);
        let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
        be.upload(arena1.as_ref(), &backing).unwrap();
        let rec = be.recorder().unwrap();
        rec.embed_gather_at(
            dtype,
            addr1 + off as u64,
            ids_buf.as_ref(),
            y_buf.as_ref(),
            rows,
            ne,
            scale,
            0, // u32-fitting table → original element-offset gather path (issue #77)
        );
        rec.finish().unwrap();
        let mut out = vec![0u8; rows * ne * 4];
        be.download(y_buf.as_ref(), &mut out).unwrap();
        let streamed_atoff = bits(&out);
        for (i, (&r, &s)) in resident.iter().zip(streamed_atoff.iter()).enumerate() {
            assert_eq!(
                r,
                s,
                "embed_gather dtype={dtype:?}: streamed@nonzero-offset differs from resident at \
                 out {i}: {} vs {} — the twin is ignoring its arena base offset, which breaks \
                 every tensor but the first in a shared arena",
                f32::from_bits(r),
                f32::from_bits(s)
            );
        }
        println!("ok: embed_gather dtype={dtype:?}");
    }
}

// ─── model-specific float-weight kernels — qwen35 conv1d + gemma4 E2B ─────────────────────────
// Same typed buffer_reference seam as the float-weight linear family (slice A4); these are
// per-model fused kernels rather than a generic GEMV shape (conv1d_silu needs a bespoke runner
// for its stateful third input; e2b_gate fits `run_case_offset_invariant`).

/// `conv1d_silu`/`conv1d_silu_batch` are BDA wrappers — each is just a `device_addr()` fork onto
/// the SAME streamed kernel this test drives directly (see those `Recorder` fns' doc), so a
/// resident-vs-streamed compare here was always vacuous once the port landed. Offset-invariance is
/// the real assertion. `conv1d_silu` (single-token) and `conv1d_silu_par` (BATCH pass 1, via
/// `conv1d_silu_batch`) both read the per-channel conv kernel `wconv` through the SAME typed seam.
/// Neither shape fits `run_case_offset_invariant` — there's a THIRD input, the per-channel history
/// `state`, that is read AND written in place — so `state` is re-uploaded to its initial value
/// before EACH leg (streamed@0, streamed@nonzero-offset) to keep the legs from drifting off each
/// other's mutated history.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn conv1d_silu_offset_invariant() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let dtype = DType::F32;
    let (cc, kconv) = (40usize, 4usize);
    let w_bytes = weight_bytes_for(dtype, cc * kconv);
    let w = synth_weight_bytes(w_bytes, cc * 31 + kconv);
    let state0 = synth_x((kconv - 1) * cc);
    let off = nonzero_off(dtype);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&w);

    // ── single-token path (conv1d_silu) ───────────────────────────────────────────────────────
    {
        let qkv = synth_x(cc);
        let qkv_buf = be.alloc(cc * 4, BufferUsage::Activations).unwrap();
        be.upload(qkv_buf.as_ref(), bytemuck::cast_slice(&qkv))
            .unwrap();
        let out_buf = be.alloc(cc * 4, BufferUsage::Activations).unwrap();
        let fresh_state = |be: &VulkanBackend| {
            let s = be
                .alloc((kconv - 1) * cc * 4, BufferUsage::Activations)
                .unwrap();
            be.upload(s.as_ref(), bytemuck::cast_slice(&state0))
                .unwrap();
            s
        };
        let download = |be: &VulkanBackend| {
            let mut out = vec![0u8; cc * 4];
            be.download(out_buf.as_ref(), &mut out).unwrap();
            bits(&out)
        };

        let state_buf = fresh_state(&be);
        let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
        be.upload(arena0.as_ref(), &w).unwrap();
        let rec = be.recorder().unwrap();
        rec.conv1d_silu_at(
            qkv_buf.as_ref(),
            addr0,
            state_buf.as_ref(),
            out_buf.as_ref(),
            1,
            cc,
            kconv,
        );
        rec.finish().unwrap();
        let at0 = download(&be);

        let state_buf = fresh_state(&be);
        let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
        be.upload(arena1.as_ref(), &backing).unwrap();
        let rec = be.recorder().unwrap();
        rec.conv1d_silu_at(
            qkv_buf.as_ref(),
            addr1 + off as u64,
            state_buf.as_ref(),
            out_buf.as_ref(),
            1,
            cc,
            kconv,
        );
        rec.finish().unwrap();
        let atoff = download(&be);

        let c = OffsetCase {
            name: "conv1d_silu (single-token)".to_string(),
            at0,
            atoff,
        };
        assert_offset_invariant(&c);
        println!("ok: {}", c.name);
    }

    // ── batch path (conv1d_silu_par + conv1d_shift, rows >= kconv-1) ─────────────────────────
    {
        let rows = 6usize;
        let qkv = synth_x(rows * cc);
        let qkv_buf = be.alloc(rows * cc * 4, BufferUsage::Activations).unwrap();
        be.upload(qkv_buf.as_ref(), bytemuck::cast_slice(&qkv))
            .unwrap();
        let out_buf = be.alloc(rows * cc * 4, BufferUsage::Activations).unwrap();
        let fresh_state = |be: &VulkanBackend| {
            let s = be
                .alloc((kconv - 1) * cc * 4, BufferUsage::Activations)
                .unwrap();
            be.upload(s.as_ref(), bytemuck::cast_slice(&state0))
                .unwrap();
            s
        };
        let download = |be: &VulkanBackend| {
            let mut out = vec![0u8; rows * cc * 4];
            be.download(out_buf.as_ref(), &mut out).unwrap();
            bits(&out)
        };

        let state_buf = fresh_state(&be);
        let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
        be.upload(arena0.as_ref(), &w).unwrap();
        let rec = be.recorder().unwrap();
        rec.conv1d_silu_batch_at(
            qkv_buf.as_ref(),
            addr0,
            state_buf.as_ref(),
            out_buf.as_ref(),
            rows,
            cc,
            kconv,
        );
        rec.finish().unwrap();
        let at0 = download(&be);

        let state_buf = fresh_state(&be);
        let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
        be.upload(arena1.as_ref(), &backing).unwrap();
        let rec = be.recorder().unwrap();
        rec.conv1d_silu_batch_at(
            qkv_buf.as_ref(),
            addr1 + off as u64,
            state_buf.as_ref(),
            out_buf.as_ref(),
            rows,
            cc,
            kconv,
        );
        rec.finish().unwrap();
        let atoff = download(&be);

        let c = OffsetCase {
            name: "conv1d_silu_par (batch)".to_string(),
            at0,
            atoff,
        };
        assert_offset_invariant(&c);
        println!("ok: {}", c.name);
    }
}

/// `e2b_gate` is a BDA wrapper — just a `device_addr()` fork onto the SAME streamed kernel this
/// test drives directly (see `Recorder::e2b_gate`'s doc), so a resident-vs-streamed compare here
/// was always vacuous once the port landed. Offset-invariance is the real assertion. Fits
/// `run_case_offset_invariant`'s weight/x/y shape (the extra `up` buffer is captured by the
/// dispatch closure, same trick the float-linear tests use for their externally-sized multi-row
/// `x`). `up_off` is non-zero and `up_stride` is wider than `out_f`, so a twin that drops either
/// would read the wrong slice of `up`.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn e2b_gate_offset_invariant() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let dtype = DType::F32;
    let (m, in_f, out_f) = (4usize, 64usize, 8usize);
    let (up_off, up_stride) = (3usize, 16usize);

    let x = synth_x(m * in_f);
    let x_buf = be.alloc(m * in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

    let up_elems = up_off + (m - 1) * up_stride + out_f;
    let up: Vec<f32> = (0..up_elems)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
        .collect();
    let up_buf = be.alloc(up_elems * 4, BufferUsage::Activations).unwrap();
    be.upload(up_buf.as_ref(), bytemuck::cast_slice(&up))
        .unwrap();

    let c = run_case_offset_invariant(
        &be,
        "e2b_gate".to_string(),
        dtype,
        in_f,
        out_f,
        m * out_f,
        &|rec, addr, _x, y| {
            rec.e2b_gate_at(
                addr,
                x_buf.as_ref(),
                up_buf.as_ref(),
                up_off,
                up_stride,
                y,
                m,
                in_f,
                out_f,
            )
        },
    );
    assert_offset_invariant(&c);
    println!("ok: {}", c.name);
}

// ─── coopmat projection GEMM (gemm_proj / gemm_proj_warp) — this slice ────────────────────────
// f16 weight parity: proves the resident/streamed seam for the (sole) production arm.

#[test]
#[ignore = "requires a Vulkan GPU"]
fn gemm_proj_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // n%64==0, k%32==0; m=8 < 768 keeps BOTH legs on the non-warp gemm_proj kernel (matmul_proj /
    // matmul_proj_at share the same `warp = m>=768 && n%256==0` gate).
    let (m, k, n) = (8usize, 256usize, 64usize);
    let mpad = m.div_ceil(64) * 64;

    let a = synth_x(m * k);
    let a_buf = be.alloc(m * k * 4, BufferUsage::Activations).unwrap();
    be.upload(a_buf.as_ref(), bytemuck::cast_slice(&a)).unwrap();
    let c_buf = be.alloc(mpad * n * 4, BufferUsage::Activations).unwrap();

    let w_bytes = weight_bytes_for(DType::F16, k * n);
    let w = synth_weight_bytes(w_bytes, k * 31 + n);

    // ── Resident: weight in its own bound SSBO ────────────────────────────────────────────────
    let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
    be.upload(w_buf.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    rec.matmul_proj(a_buf.as_ref(), w_buf.as_ref(), c_buf.as_ref(), m, k, n);
    rec.finish().unwrap();
    let mut out = vec![0u8; mpad * n * 4];
    be.download(c_buf.as_ref(), &mut out).unwrap();
    let resident = bits(&out);

    // ── Streamed leg 1: arena offset 0 ────────────────────────────────────────────────────────
    let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
    be.upload(arena0.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    rec.matmul_proj_at(a_buf.as_ref(), addr0, c_buf.as_ref(), m, k, n);
    rec.finish().unwrap();
    let mut out = vec![0u8; mpad * n * 4];
    be.download(c_buf.as_ref(), &mut out).unwrap();
    let streamed_at0 = bits(&out);

    // ── Streamed leg 2: the SAME weight parked at a non-zero offset in a bigger arena ─────────
    let off = nonzero_off(DType::F16);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&w);
    let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena1.as_ref(), &backing).unwrap();
    let rec = be.recorder().unwrap();
    rec.matmul_proj_at(a_buf.as_ref(), addr1 + off as u64, c_buf.as_ref(), m, k, n);
    rec.finish().unwrap();
    let mut out = vec![0u8; mpad * n * 4];
    be.download(c_buf.as_ref(), &mut out).unwrap();
    let streamed_atoff = bits(&out);

    // C is allocated ceil(m/64)*64 padded rows; compare only the m real rows (same trim pattern as
    // the mmq/fma parity cases — C is row-major so the first m*n elements are exactly rows 0..m).
    let real = m * n;
    let c = Case {
        name: "gemm_proj".to_string(),
        resident: resident[..real].to_vec(),
        streamed_at0: streamed_at0[..real].to_vec(),
        streamed_atoff: streamed_atoff[..real].to_vec(),
    };
    assert_case(&c);
    println!("ok: {}", c.name);
}

// ─── int8-coopmat Q8_0 prefill GEMM (native_gemm_i8cm_q8_0) — this slice ───────────────────────
// Measurement kernel, gated behind INFR_I8_COOPMAT=1 + `caps.i8_coopmat()` (16x16x16 int8 WMMA —
// DETECTION ONLY at the Capabilities level, per its doc). The env toggle alone is a no-op on
// hardware/driver that doesn't detect the config: the adapter never routes here, so dispatching the
// kernel by hand (as this test does) would just prove nothing on a box without the coopmat shape —
// self-skip instead of silently passing vacuously.

// Exercises `Recorder::matmul_i8cm_q8_0_at`, a parity-only entry point gated behind the crate's
// `parity` feature (no production caller) — so this test only compiles with `--features parity`.
#[cfg(feature = "parity")]
#[test]
#[ignore = "requires a Vulkan GPU"]
fn i8cm_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    if !be.capabilities().i8_coopmat() {
        eprintln!("skip: no 16x16x16 i8 coopmat on this device");
        return;
    }
    // n%16, k%32 (see the shader header); m=8 keeps everything inside one BM=128 workgroup row-tile
    // (gr < pc.m zero-guards the rest, so C only ever needs m*n reals — no pad-row garbage to trim).
    let (m, k, n) = (8usize, 256usize, 64usize);
    let q = quantize_x(&be, m, k);

    let w_bytes = weight_bytes_for(DType::Q8_0, k * n);
    let w = synth_weight_bytes(w_bytes, k * 31 + n);
    let c_buf = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();

    // ── Resident: weight in its own bound SSBO ────────────────────────────────────────────────
    let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
    be.upload(w_buf.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    rec.matmul_i8cm_q8_0(
        q.qa.as_ref(),
        q.dact.as_ref(),
        w_buf.as_ref(),
        0,
        c_buf.as_ref(),
        m,
        k,
        n,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; m * n * 4];
    be.download(c_buf.as_ref(), &mut out).unwrap();
    let resident = bits(&out);

    // ── Streamed leg 1: arena offset 0 ────────────────────────────────────────────────────────
    let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
    be.upload(arena0.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    rec.matmul_i8cm_q8_0_at(
        q.qa.as_ref(),
        q.dact.as_ref(),
        addr0,
        0,
        c_buf.as_ref(),
        m,
        k,
        n,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; m * n * 4];
    be.download(c_buf.as_ref(), &mut out).unwrap();
    let streamed_at0 = bits(&out);

    // ── Streamed leg 2: the SAME weight parked at a non-zero offset in a bigger arena ─────────
    let off = nonzero_off(DType::Q8_0);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&w);
    let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena1.as_ref(), &backing).unwrap();
    let rec = be.recorder().unwrap();
    rec.matmul_i8cm_q8_0_at(
        q.qa.as_ref(),
        q.dact.as_ref(),
        addr1 + off as u64,
        0,
        c_buf.as_ref(),
        m,
        k,
        n,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; m * n * 4];
    be.download(c_buf.as_ref(), &mut out).unwrap();
    let streamed_atoff = bits(&out);

    let c = Case {
        name: "i8cm_q8_0".to_string(),
        resident,
        streamed_at0,
        streamed_atoff,
    };
    assert_case(&c);
    println!("ok: {}", c.name);
}

// ─── load-time Q8_0->E4M3 repack (repack_q8_to_f8) — this slice ───────────────────────────────
// Not a GEMV/GEMM: the kernel's whole job IS the output, so parity here means the two OUTPUT byte
// buffers (resident-read repack vs streamed-read repack) must be byte-identical — there is no
// separate "real answer" to check either against, only that both legs decode the SAME Q8_0 source
// bytes into the SAME E4M3 bytes. Defensively gated on `caps.f8_coopmat()` even though the kernel
// itself does no coopmat math: `floate4m3_t` storage (GL_EXT_float_e4m3) needs the same fp8 device
// support the coopmat tier detects, and the shader's own header notes this path is
// "compile-checked only" on hardware without it.

// Exercises `Recorder::repack_q8_to_f8_at`, a parity-only entry point gated behind the crate's
// `parity` feature (no production caller) — so this test only compiles with `--features parity`.
#[cfg(feature = "parity")]
#[test]
#[ignore = "requires a Vulkan GPU"]
fn repack_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    if !be.capabilities().f8_coopmat() {
        eprintln!("skip: no f8 coopmat / float8 storage on this device");
        return;
    }
    let (n, k) = (64usize, 256usize); // k%32==0 (one 32-elem Q8_0 block per repack thread)
    let w_bytes = weight_bytes_for(DType::Q8_0, n * k);
    let w = synth_weight_bytes(w_bytes, k * 31 + n);
    let w8_bytes = n * k; // [n, k] E4M3, 1 byte/elem

    // ── Resident: Q8_0 source in its own bound SSBO ───────────────────────────────────────────
    let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
    be.upload(w_buf.as_ref(), &w).unwrap();
    let w8_buf = be.alloc(w8_bytes, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.repack_q8_to_f8(w_buf.as_ref(), 0, w8_buf.as_ref(), n, k);
    rec.finish().unwrap();
    let mut resident = vec![0u8; w8_bytes];
    be.download(w8_buf.as_ref(), &mut resident).unwrap();
    assert!(
        resident.iter().any(|&b| b != 0),
        "repack: resident w8 output is all zeros — the case is not exercising the kernel"
    );

    // ── Streamed leg 1: arena offset 0 ────────────────────────────────────────────────────────
    let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
    be.upload(arena0.as_ref(), &w).unwrap();
    let w8_buf0 = be.alloc(w8_bytes, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.repack_q8_to_f8_at(addr0, 0, w8_buf0.as_ref(), n, k);
    rec.finish().unwrap();
    let mut streamed_at0 = vec![0u8; w8_bytes];
    be.download(w8_buf0.as_ref(), &mut streamed_at0).unwrap();
    assert_eq!(
        resident, streamed_at0,
        "repack: streamed@0 w8 output differs from resident, byte-for-byte"
    );

    // ── Streamed leg 2: the SAME Q8_0 source parked at a non-zero offset, garbage prefix ──────
    let off = nonzero_off(DType::Q8_0);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&w);
    let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena1.as_ref(), &backing).unwrap();
    let w8_buf1 = be.alloc(w8_bytes, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.repack_q8_to_f8_at(addr1 + off as u64, 0, w8_buf1.as_ref(), n, k);
    rec.finish().unwrap();
    let mut streamed_atoff = vec![0u8; w8_bytes];
    be.download(w8_buf1.as_ref(), &mut streamed_atoff).unwrap();
    assert_eq!(
        resident, streamed_atoff,
        "repack: streamed@nonzero-offset w8 output differs from resident — the twin is ignoring \
         its arena base offset"
    );

    println!("ok: repack_q8_to_f8 streamed");
}

// ─── fused-residual family (this slice): native/mmv/mmv_mrow `-DUSE_RES` streamed twins ───────
// The decode Linear+Add fusion (`linear_add_native` / `linear_add_mmv` / `linear_mmv_mrow` with
// `Some(residual)`) dispatches resident weights today — the fused-add peephole filters streamed
// weights OUT before this path is ever reached (adapter.rs). This slice adds the matching
// `_res_streamed` SPIR-V + Rust plumbing so the resident-BDA endgame (deleting every bound-SSBO
// weight path) has a residual twin ready when that filter is eventually lifted. Same bitwise
// contract as the rest of this file (offset 0 AND a non-zero offset with a garbage prefix), with a
// non-trivial MIXED-SIGN residual buffer so a dropped or zeroed residual add shows up as a visibly
// wrong answer instead of an accidental pass.

/// Mixed-sign residual values — deliberately not all-positive/all-zero so a broken (or missing)
/// residual add can't pass by producing a coincidentally-plausible magnitude.
fn synth_residual(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect()
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn linear_add_native_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // out_f=2048, in_f%32==0: Q6_K lands on the SG route (`native_sg_choice`) and Q4_K on the
    // default "reg" RM-variant route (`native_rm_v2_streamed_build_spv`'s res arm) — see
    // recorder.rs's `linear_add_native`/`linear_add_native_at` routing precedence. Both are
    // exercised here rather than falling through to the plain-GEMV arm.
    let (in_f, out_f) = (256usize, 2048usize);
    let res = synth_residual(out_f);
    let res_buf = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
    be.upload(res_buf.as_ref(), bytemuck::cast_slice(&res))
        .unwrap();

    for dtype in [DType::Q4K, DType::Q6K] {
        let c = run_case(
            &be,
            format!("linear_add_native dtype={dtype:?}"),
            dtype,
            in_f,
            out_f,
            out_f,
            &|rec, w, x, y| rec.linear_add_native(dtype, w, x, res_buf.as_ref(), y, 1, in_f, out_f),
            &|rec, addr, x, y| {
                rec.linear_add_native_at(dtype, addr, x, res_buf.as_ref(), y, 1, in_f, out_f)
            },
        );
        assert_case(&c);
        println!("ok: {}", c.name);
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn linear_add_mmv_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    let dtype = DType::Q4K;
    let q = quantize_x(&be, 1, in_f);
    let res = synth_residual(out_f);
    let res_buf = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
    be.upload(res_buf.as_ref(), bytemuck::cast_slice(&res))
        .unwrap();

    let c = run_mmv_case(
        &be,
        format!("linear_add_mmv dtype={dtype:?}"),
        dtype,
        in_f,
        out_f,
        out_f,
        &q,
        &|rec, w, y| {
            rec.linear_add_mmv(
                dtype,
                w,
                q.qa.as_ref(),
                q.dact.as_ref(),
                q.sact.as_ref(),
                res_buf.as_ref(),
                y,
                in_f,
                out_f,
            )
        },
        &|rec, addr, y| {
            rec.linear_add_mmv_at(
                dtype,
                addr,
                q.qa.as_ref(),
                q.dact.as_ref(),
                q.sact.as_ref(),
                res_buf.as_ref(),
                y,
                in_f,
                out_f,
            )
        },
    );
    assert_case(&c);
    println!("ok: {}", c.name);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn linear_mmv_mrow_residual_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // rows=1: the only shape a fused residual is ever legal at (the decode Linear+Add fusion) —
    // see linear_mmv_mrow's doc.
    let (in_f, out_f, rows) = (256usize, 8usize, 1usize);
    let dtype = DType::Q4K;
    let q = quantize_x(&be, rows, in_f);
    let res = synth_residual(out_f);
    let res_buf = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
    be.upload(res_buf.as_ref(), bytemuck::cast_slice(&res))
        .unwrap();

    let c = run_mmv_case(
        &be,
        format!("linear_mmv_mrow residual dtype={dtype:?}"),
        dtype,
        in_f,
        out_f,
        rows * out_f,
        &q,
        &|rec, w, y| {
            rec.linear_mmv_mrow(
                dtype,
                w,
                0,
                q.qa.as_ref(),
                q.dact.as_ref(),
                q.sact.as_ref(),
                Some(res_buf.as_ref()),
                y,
                rows,
                in_f,
                out_f,
            )
        },
        &|rec, addr, y| {
            rec.linear_mmv_mrow_at(
                dtype,
                addr,
                0,
                q.qa.as_ref(),
                q.dact.as_ref(),
                q.sact.as_ref(),
                Some(res_buf.as_ref()),
                y,
                rows,
                in_f,
                out_f,
            )
        },
    );
    assert_case(&c);
    println!("ok: {}", c.name);
}

// ─── resident-BDA routing proof — dense model path integration ────────────────────────────────
// Every other test in this file drives a `_streamed` twin DIRECTLY with a raw `u64` address —
// proving the twin's math, not that anything routes to it. This test proves the wiring: it calls
// the RESIDENT entry point (`linear_native`/`linear_mmv`/`matmul_mmq`/`embed_gather`) with a
// weight buffer built via `bda_weight_alloc_for_test` (the same "construct the arena alloc
// directly, not via env" approach `resident_bda_weight_arena_roundtrip` uses in lib.rs's own
// tests) — a `device_addr()`-reporting sub-tensor, exactly what `INFR_RESIDENT_BDA=1` produces.
// Each resident fn's internal `if let Some(arena_addr) = w.device_addr() { ... }` fork must catch
// this and dispatch the `_streamed` kernel; if it forgot, `Self::vkb(w)` would either trip its
// debug_assert (debug build) or silently mis-bind the whole arena block as a WHOLE_SIZE
// descriptor (release build). Compared bitwise against the SAME resident fn called with an
// ordinary bound-SSBO weight holding identical bytes.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn resident_bda_linear_route_dispatches() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };

    // ── linear_native (Q4_K, m=1 decode GEMV) ─────────────────────────────────────────────
    {
        let dtype = DType::Q4K;
        let (in_f, out_f) = (256usize, 64usize);
        let w_bytes = weight_bytes_for(dtype, in_f * out_f);
        let w = synth_weight_bytes(w_bytes, 401);
        let x = synth_x(in_f);
        let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
        be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

        let resident = {
            let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
            be.upload(w_buf.as_ref(), &w).unwrap();
            let y_buf = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.linear_native(
                dtype,
                w_buf.as_ref(),
                0,
                x_buf.as_ref(),
                y_buf.as_ref(),
                1,
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            let mut out = vec![0u8; out_f * 4];
            be.download(y_buf.as_ref(), &mut out).unwrap();
            bits(&out)
        };
        assert!(
            resident.iter().any(|&b| b != 0),
            "linear_native: resident output is all zeros"
        );

        let routed = {
            let w_bda = be.bda_weight_alloc_for_test(w_bytes).unwrap();
            assert!(
                w_bda.device_addr().is_some(),
                "bda_weight_alloc_for_test must report Some(device_addr)"
            );
            be.upload(w_bda.as_ref(), &w).unwrap();
            let y_buf = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.linear_native(
                dtype,
                w_bda.as_ref(),
                0,
                x_buf.as_ref(),
                y_buf.as_ref(),
                1,
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            let mut out = vec![0u8; out_f * 4];
            be.download(y_buf.as_ref(), &mut out).unwrap();
            bits(&out)
        };
        assert_eq!(
            resident, routed,
            "linear_native: resident-BDA route diverged from the SSBO reference"
        );
        println!("ok: resident-BDA route linear_native dtype={dtype:?}");
    }

    // ── linear_mmv (Q4_K int8 dp4a decode GEMV) ───────────────────────────────────────────────
    {
        let dtype = DType::Q4K;
        let (in_f, out_f) = (256usize, 8usize);
        let w_bytes = weight_bytes_for(dtype, in_f * out_f);
        let w = synth_weight_bytes(w_bytes, 402);
        let q = quantize_x(&be, 1, in_f);

        let resident = {
            let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
            be.upload(w_buf.as_ref(), &w).unwrap();
            let y_buf = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.linear_mmv(
                dtype,
                w_buf.as_ref(),
                0,
                q.qa.as_ref(),
                q.dact.as_ref(),
                q.sact.as_ref(),
                y_buf.as_ref(),
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            let mut out = vec![0u8; out_f * 4];
            be.download(y_buf.as_ref(), &mut out).unwrap();
            bits(&out)
        };
        assert!(
            resident.iter().any(|&b| b != 0),
            "linear_mmv: resident output is all zeros"
        );

        let routed = {
            let w_bda = be.bda_weight_alloc_for_test(w_bytes).unwrap();
            assert!(w_bda.device_addr().is_some());
            be.upload(w_bda.as_ref(), &w).unwrap();
            let y_buf = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.linear_mmv(
                dtype,
                w_bda.as_ref(),
                0,
                q.qa.as_ref(),
                q.dact.as_ref(),
                q.sact.as_ref(),
                y_buf.as_ref(),
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            let mut out = vec![0u8; out_f * 4];
            be.download(y_buf.as_ref(), &mut out).unwrap();
            bits(&out)
        };
        assert_eq!(
            resident, routed,
            "linear_mmv: resident-BDA route diverged from the SSBO reference"
        );
        println!("ok: resident-BDA route linear_mmv dtype={dtype:?}");
    }

    // ── matmul_mmq (Q4_K tiled dp4a prefill GEMM) ─────────────────────────────────────────────
    {
        let dtype = DType::Q4K;
        let (m, k, n) = (8usize, 256usize, 64usize);
        let w_bytes = weight_bytes_for(dtype, k * n);
        let w = synth_weight_bytes(w_bytes, 403);
        let q = quantize_x(&be, m, k);
        let c_rows = m.div_ceil(64) * 64;

        let resident = {
            let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
            be.upload(w_buf.as_ref(), &w).unwrap();
            let c_buf = be.alloc(c_rows * n * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.matmul_mmq(
                dtype,
                q.qa.as_ref(),
                q.dact.as_ref(),
                q.sact.as_ref(),
                w_buf.as_ref(),
                0,
                c_buf.as_ref(),
                m,
                k,
                n,
            );
            rec.finish().unwrap();
            let mut out = vec![0u8; c_rows * n * 4];
            be.download(c_buf.as_ref(), &mut out).unwrap();
            bits(&out)
        };
        assert!(
            resident.iter().any(|&b| b != 0),
            "matmul_mmq: resident output is all zeros"
        );

        let routed = {
            let w_bda = be.bda_weight_alloc_for_test(w_bytes).unwrap();
            assert!(w_bda.device_addr().is_some());
            be.upload(w_bda.as_ref(), &w).unwrap();
            let c_buf = be.alloc(c_rows * n * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.matmul_mmq(
                dtype,
                q.qa.as_ref(),
                q.dact.as_ref(),
                q.sact.as_ref(),
                w_bda.as_ref(),
                0,
                c_buf.as_ref(),
                m,
                k,
                n,
            );
            rec.finish().unwrap();
            let mut out = vec![0u8; c_rows * n * 4];
            be.download(c_buf.as_ref(), &mut out).unwrap();
            bits(&out)
        };
        assert_eq!(
            resident, routed,
            "matmul_mmq: resident-BDA route diverged from the SSBO reference"
        );
        println!("ok: resident-BDA route matmul_mmq dtype={dtype:?}");
    }

    // ── embed_gather (Q4_K token-embedding gather) ────────────────────────────────────────────
    {
        let dtype = DType::Q4K;
        let (n_table_rows, ne, rows) = (16usize, 256usize, 4usize);
        let scale = 0.5f32;
        let ids: [i32; 4] = [3, 0, 7, 12];
        let ids_buf = be.alloc(rows * 4, BufferUsage::Activations).unwrap();
        be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&ids))
            .unwrap();
        let w_bytes = weight_bytes_for(dtype, n_table_rows * ne);
        let table = synth_weight_bytes(w_bytes, 404);

        let resident = {
            let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
            be.upload(w_buf.as_ref(), &table).unwrap();
            let y_buf = be.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.embed_gather(
                dtype,
                w_buf.as_ref(),
                ids_buf.as_ref(),
                y_buf.as_ref(),
                rows,
                ne,
                scale,
            );
            rec.finish().unwrap();
            let mut out = vec![0u8; rows * ne * 4];
            be.download(y_buf.as_ref(), &mut out).unwrap();
            bits(&out)
        };
        assert!(
            resident.iter().any(|&b| b != 0),
            "embed_gather: resident output is all zeros"
        );

        let routed = {
            let w_bda = be.bda_weight_alloc_for_test(w_bytes).unwrap();
            assert!(w_bda.device_addr().is_some());
            be.upload(w_bda.as_ref(), &table).unwrap();
            let y_buf = be.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.embed_gather(
                dtype,
                w_bda.as_ref(),
                ids_buf.as_ref(),
                y_buf.as_ref(),
                rows,
                ne,
                scale,
            );
            rec.finish().unwrap();
            let mut out = vec![0u8; rows * ne * 4];
            be.download(y_buf.as_ref(), &mut out).unwrap();
            bits(&out)
        };
        assert_eq!(
            resident, routed,
            "embed_gather: resident-BDA route diverged from the SSBO reference"
        );
        println!("ok: resident-BDA route embed_gather dtype={dtype:?}");
    }
}

// ─── resident-BDA sub-range descriptor bind ────────────────────────────────────────────────────
// `rmsnorm`'s gamma has no `-DSTREAMED` twin (it is a small resident weight, unlike the big matmul
// families above), so a resident-BDA gamma is BOUND as a descriptor via `Recorder::vkb`, never read
// through `device_addr()`. This allocates TWO tensors back-to-back from the same arena block, so the
// SECOND lands at a NON-ZERO byte offset: the first is garbage that must never be read by the second
// tensor's dispatch. If `vkb` ever regressed to binding `(0, WHOLE_SIZE)` — the whole shared block,
// not the tensor's own `(sub_offset, range)` — the kernel would read the garbage tensor's bytes
// (which sit at the block's byte 0) instead of the real gamma, and the output would diverge from an
// ordinary-buffer reference computed over the identical gamma bytes.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn resident_bda_bound_subrange_bind() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };

    let (rows, dim) = (2usize, 64usize);
    let eps = 1e-5f32;
    let x = synth_x(rows * dim);
    let x_buf = be.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

    // Known, finite gamma bytes — rmsnorm's gamma is consumed directly as f32 (no dequant), so
    // `synth_weight_bytes`'s low-byte-range NaN-avoidance trick doesn't apply here.
    let gamma: Vec<f32> = (0..dim).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
    let gamma_bytes: &[u8] = bytemuck::cast_slice(&gamma);

    // ── First BDA tensor: garbage occupying the block's front (sub_offset 0) — must never be read
    let garbage_bda = be.bda_weight_alloc_for_test(1024).unwrap();
    let garbage: Vec<u8> = (0..1024u32).map(|i| (i % 256) as u8).collect();
    be.upload(garbage_bda.as_ref(), &garbage).unwrap();

    // ── Second BDA tensor: the real gamma, forced to a non-zero offset by the alloc above ───────
    let gamma_bda = be.bda_weight_alloc_for_test(gamma_bytes.len()).unwrap();
    assert!(
        gamma_bda.device_addr().is_some(),
        "bda_weight_alloc_for_test must report Some(device_addr)"
    );
    be.upload(gamma_bda.as_ref(), gamma_bytes).unwrap();

    let bda_out = {
        let y_buf = be.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.rmsnorm(
            x_buf.as_ref(),
            gamma_bda.as_ref(),
            y_buf.as_ref(),
            rows,
            dim,
            eps,
        );
        rec.finish().unwrap();
        let mut out = vec![0u8; rows * dim * 4];
        be.download(y_buf.as_ref(), &mut out).unwrap();
        bits(&out)
    };
    assert!(
        bda_out.iter().any(|&b| b != 0),
        "rmsnorm: resident-BDA output is all zeros"
    );

    let ordinary_out = {
        let gamma_buf = be.alloc(gamma_bytes.len(), BufferUsage::Weights).unwrap();
        be.upload(gamma_buf.as_ref(), gamma_bytes).unwrap();
        let y_buf = be.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.rmsnorm(
            x_buf.as_ref(),
            gamma_buf.as_ref(),
            y_buf.as_ref(),
            rows,
            dim,
            eps,
        );
        rec.finish().unwrap();
        let mut out = vec![0u8; rows * dim * 4];
        be.download(y_buf.as_ref(), &mut out).unwrap();
        bits(&out)
    };

    assert_eq!(
        bda_out, ordinary_out,
        "rmsnorm: resident-BDA gamma bind diverged from the ordinary-buffer reference — vkb's \
         sub-range descriptor bind is broken (bound the whole arena block instead of the tensor's \
         own byte range)"
    );
    println!("ok: resident-BDA sub-range descriptor bind (rmsnorm gamma at non-zero offset)");
}

// ─── resident-BDA routing proof — MoE expert-grid path (slice A6b-3) ──────────────────────────
// Same intent as `resident_bda_linear_route_dispatches_streamed` above, extended to the MoE
// expert-grid families: proves `linear_native_id_multi`/`linear_mmv_id_multi_q4k`/
// `matmul_mmq_experts` fork to their `_streamed` twins when the weight is arena-allocated. Each
// arena leg is placed at a NON-ZERO offset by allocating a garbage tensor first (same trick as
// `resident_bda_bound_subrange_bind`) — id 0 / offset 0 would pass even if the byte-stride scaling
// this campaign added were completely broken (see the id-family module doc above).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn resident_bda_id_route_dispatches() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };

    // out_f=2048 clears `native_id_sg_choice`'s default MINOUT band so Q6K's rows==1 leg actually
    // takes the SG route (not just the tree kernel) — Q4K never qualifies for SG regardless of
    // out_f, so it stays on the tree kernel at every row count tested here.
    let (in_f, out_f, n_expert, n_used) = (256usize, 2048usize, 4usize, 3usize);

    for dtype in [DType::Q4K, DType::Q6K] {
        let bank = synth_bank(dtype, in_f, out_f, n_expert);

        // Garbage tensor first: every `bda_weight_alloc_for_test` call after this in the test's
        // arena block lands at a non-zero byte offset.
        let garbage = be.bda_weight_alloc_for_test(4096).unwrap();
        be.upload(garbage.as_ref(), &synth_weight_bytes(4096, 0xBAD))
            .unwrap();
        let w_bda = be.bda_weight_alloc_for_test(bank.bytes.len()).unwrap();
        assert!(
            w_bda.device_addr().is_some(),
            "bda_weight_alloc_for_test must report Some(device_addr)"
        );
        be.upload(w_bda.as_ref(), &bank.bytes).unwrap();

        let w_buf = be.alloc(bank.bytes.len(), BufferUsage::Weights).unwrap();
        be.upload(w_buf.as_ref(), &bank.bytes).unwrap();

        // ── linear_native_id_multi: rows=1 (Q6K → SG route, Q4K → tree) and rows=2 (tree, both) ──
        for rows in [1usize, 2usize] {
            // Permuted, distinct, non-zero ids per row (n_used of n_expert experts).
            let ids: Vec<u32> = if rows == 1 {
                vec![3, 1, 2]
            } else {
                vec![3, 1, 2, 2, 3, 1]
            };
            assert_eq!(ids.len(), rows * n_used);
            let ids_buf = be.alloc(ids.len() * 4, BufferUsage::Activations).unwrap();
            be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&ids))
                .unwrap();
            let x = synth_x(rows * in_f);
            let x_buf = be.alloc(rows * in_f * 4, BufferUsage::Activations).unwrap();
            be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
            let y_elems = rows * n_used * out_f;

            let resident = {
                let y_buf = be.alloc(y_elems * 4, BufferUsage::Activations).unwrap();
                let rec = be.recorder().unwrap();
                rec.linear_native_id_multi(
                    dtype,
                    w_buf.as_ref(),
                    ids_buf.as_ref(),
                    n_used,
                    bank.stride_elems,
                    x_buf.as_ref(),
                    false,
                    y_buf.as_ref(),
                    in_f,
                    out_f,
                    rows,
                );
                rec.finish().unwrap();
                let mut out = vec![0u8; y_elems * 4];
                be.download(y_buf.as_ref(), &mut out).unwrap();
                bits(&out)
            };
            assert!(
                resident.iter().any(|&b| b != 0),
                "linear_native_id_multi dtype={dtype:?} rows={rows}: resident output is all zeros"
            );

            let routed = {
                let y_buf = be.alloc(y_elems * 4, BufferUsage::Activations).unwrap();
                let rec = be.recorder().unwrap();
                rec.linear_native_id_multi(
                    dtype,
                    w_bda.as_ref(),
                    ids_buf.as_ref(),
                    n_used,
                    bank.stride_elems,
                    x_buf.as_ref(),
                    false,
                    y_buf.as_ref(),
                    in_f,
                    out_f,
                    rows,
                );
                rec.finish().unwrap();
                let mut out = vec![0u8; y_elems * 4];
                be.download(y_buf.as_ref(), &mut out).unwrap();
                bits(&out)
            };
            assert_eq!(
                resident, routed,
                "linear_native_id_multi dtype={dtype:?} rows={rows}: resident-BDA route diverged \
                 from the SSBO reference"
            );
            println!("ok: resident-BDA route linear_native_id_multi dtype={dtype:?} rows={rows}");
        }

        // ── linear_mmv_id_multi_q4k (Q4K only — the only dtype this kernel covers) ──────────────
        if dtype == DType::Q4K {
            let ids: [u32; 3] = [3, 1, 2];
            let ids_buf = be.alloc(ids.len() * 4, BufferUsage::Activations).unwrap();
            be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&ids))
                .unwrap();
            let q = quantize_x(&be, 1, in_f);
            let y_elems = n_used * out_f;

            let resident = {
                let y_buf = be.alloc(y_elems * 4, BufferUsage::Activations).unwrap();
                let rec = be.recorder().unwrap();
                rec.linear_mmv_id_multi_q4k(
                    w_buf.as_ref(),
                    q.qa.as_ref(),
                    q.dact.as_ref(),
                    q.sact.as_ref(),
                    ids_buf.as_ref(),
                    n_used,
                    bank.stride_elems,
                    y_buf.as_ref(),
                    in_f,
                    out_f,
                );
                rec.finish().unwrap();
                let mut out = vec![0u8; y_elems * 4];
                be.download(y_buf.as_ref(), &mut out).unwrap();
                bits(&out)
            };
            assert!(
                resident.iter().any(|&b| b != 0),
                "linear_mmv_id_multi_q4k: resident output is all zeros"
            );

            let routed = {
                let y_buf = be.alloc(y_elems * 4, BufferUsage::Activations).unwrap();
                let rec = be.recorder().unwrap();
                rec.linear_mmv_id_multi_q4k(
                    w_bda.as_ref(),
                    q.qa.as_ref(),
                    q.dact.as_ref(),
                    q.sact.as_ref(),
                    ids_buf.as_ref(),
                    n_used,
                    bank.stride_elems,
                    y_buf.as_ref(),
                    in_f,
                    out_f,
                );
                rec.finish().unwrap();
                let mut out = vec![0u8; y_elems * 4];
                be.download(y_buf.as_ref(), &mut out).unwrap();
                bits(&out)
            };
            assert_eq!(
                resident, routed,
                "linear_mmv_id_multi_q4k: resident-BDA route diverged from the SSBO reference"
            );
            println!("ok: resident-BDA route linear_mmv_id_multi_q4k dtype={dtype:?}");
        }
    }
}

/// One expert-grid tile-selection variant to exercise in
/// `resident_bda_mmq_experts_route_dispatches_streamed`: `n` picks BN (64 vs the 128-wide twin)
/// and `rows_param`/`n_used` (equal, so `avg_rows == rows_param` — see `matmul_mmq_experts`'s
/// `avg_rows` calc) picks BM via the `small_tile` threshold (`MOE_EXPERT_SMALL_TILE_AVG_ROWS`).
struct MmqVariant {
    label: &'static str,
    n: usize,
    rows_param: usize,
}

// ─── resident-BDA routing proof — batched MoE expert GEMM (slice A6b-3) ───────────────────────
// `matmul_mmq_experts`'s EXPERT_GRID tile selection picks one of four kernels
// (`_xp`/`_xp32`/`_xp128`/`_xp32w`); all four get exercised below (n_expert=4, n_used=n_expert so
// `avg_rows` reads directly as `rows_param`) to prove the resident→streamed fork holds across the
// whole selection, not just the default tile. Only the REAL packed C rows are compared — rows past
// the last expert's segment are never written by the EXPERT_GRID kernel (the per-row `rr >=
// rowEnd` clip in the shader), so they hold whatever was in the two independently-allocated output
// buffers before the dispatch, which is not guaranteed equal (an earlier slice hit exactly this).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn resident_bda_mmq_experts_route_dispatches() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };

    let k = 256usize;
    let n_expert = 4usize;
    // Bucket layout: 4 experts, a few packed rows each, contiguous (no gaps) — offsets[e] =
    // running sum of counts[..e]. Real packed row count R = 10.
    let counts: [u32; 4] = [3, 2, 4, 1];
    let offsets: [u32; 4] = [0, 3, 5, 9];
    let real_rows: usize = counts.iter().map(|&c| c as usize).sum();
    // Generous padding past the last expert's segment for the EXPERT_GRID kernels' bounded
    // overread (up to BM-1 rows past a segment end, discarded) — see `Op::MoeFfn`'s `npad` doc in
    // adapter.rs for the production analogue.
    let row_pad = real_rows + 128;

    let counts_buf = be.alloc(n_expert * 4, BufferUsage::Activations).unwrap();
    be.upload(counts_buf.as_ref(), bytemuck::cast_slice(&counts))
        .unwrap();
    let offsets_buf = be.alloc(n_expert * 4, BufferUsage::Activations).unwrap();
    be.upload(offsets_buf.as_ref(), bytemuck::cast_slice(&offsets))
        .unwrap();
    // Shared packed activations: k is the same for every dtype/variant below, so one quant pass
    // covers all of them.
    let q = quantize_x(&be, row_pad, k);

    // `_xp`/`_xp32` (BN=64) and `_xp128`/`_xp32w` (BN=128) each need their own weight bank (the
    // per-expert stride depends on n) — built once per `n` below and shared by the two `rows_param`
    // legs (small_tile true/false) that reuse it.
    let variants = [
        MmqVariant {
            label: "xp (default BM=64/BN=64)",
            n: 64,
            rows_param: 100,
        },
        MmqVariant {
            label: "xp32 (small-tile BM=32/BN=64)",
            n: 64,
            rows_param: 8,
        },
        MmqVariant {
            label: "xp128 (BM=64/BN=128 wide)",
            n: 128,
            rows_param: 100,
        },
        MmqVariant {
            label: "xp32w (small-tile BM=32/BN=128 wide)",
            n: 128,
            rows_param: 8,
        },
    ];

    for dtype in [DType::Q4K, DType::Q6K] {
        let needs_sact = infr_core::tensor::moe_mmq_needs_sact(dtype);
        let sact: Option<&Buf2> = needs_sact.then(|| q.sact.as_ref());

        // One bank per distinct `n` used below (64 and 128), each uploaded to an ordinary SSBO and
        // to a garbage-prefixed arena tensor.
        let mut banks: std::collections::HashMap<usize, (Bank, Box<Buf>, Box<Buf>)> =
            std::collections::HashMap::new();
        for n in [64usize, 128usize] {
            let bank = synth_bank(dtype, k, n, n_expert);
            let w_buf = be.alloc(bank.bytes.len(), BufferUsage::Weights).unwrap();
            be.upload(w_buf.as_ref(), &bank.bytes).unwrap();
            // Garbage-first so this bank's arena tensor lands at a non-zero byte offset.
            let garbage = be.bda_weight_alloc_for_test(4096).unwrap();
            be.upload(garbage.as_ref(), &synth_weight_bytes(4096, 0xBAD))
                .unwrap();
            let w_bda = be.bda_weight_alloc_for_test(bank.bytes.len()).unwrap();
            assert!(
                w_bda.device_addr().is_some(),
                "bda_weight_alloc_for_test must report Some(device_addr)"
            );
            be.upload(w_bda.as_ref(), &bank.bytes).unwrap();
            banks.insert(n, (bank, w_buf, w_bda));
        }

        for v in &variants {
            let (bank, w_buf, w_bda) = banks.get(&v.n).unwrap();
            let real = real_rows * v.n;

            let resident = {
                let c_buf = be
                    .alloc(row_pad * v.n * 4, BufferUsage::Activations)
                    .unwrap();
                let rec = be.recorder().unwrap();
                rec.matmul_mmq_experts(
                    dtype,
                    "test_expert_gateup",
                    q.qa.as_ref(),
                    q.dact.as_ref(),
                    sact,
                    w_buf.as_ref(),
                    0,
                    bank.stride_elems,
                    counts_buf.as_ref(),
                    offsets_buf.as_ref(),
                    c_buf.as_ref(),
                    v.rows_param,
                    k,
                    v.n,
                    n_expert,
                    n_expert, // n_used == n_expert so avg_rows reads as rows_param directly
                );
                rec.finish().unwrap();
                let mut out = vec![0u8; row_pad * v.n * 4];
                be.download(c_buf.as_ref(), &mut out).unwrap();
                bits(&out)[..real].to_vec()
            };
            assert!(
                resident.iter().any(|&b| b != 0),
                "matmul_mmq_experts dtype={dtype:?} variant={}: resident output is all zeros",
                v.label
            );

            let routed = {
                let c_buf = be
                    .alloc(row_pad * v.n * 4, BufferUsage::Activations)
                    .unwrap();
                let rec = be.recorder().unwrap();
                rec.matmul_mmq_experts(
                    dtype,
                    "test_expert_gateup",
                    q.qa.as_ref(),
                    q.dact.as_ref(),
                    sact,
                    w_bda.as_ref(),
                    0,
                    bank.stride_elems,
                    counts_buf.as_ref(),
                    offsets_buf.as_ref(),
                    c_buf.as_ref(),
                    v.rows_param,
                    k,
                    v.n,
                    n_expert,
                    n_expert,
                );
                rec.finish().unwrap();
                let mut out = vec![0u8; row_pad * v.n * 4];
                be.download(c_buf.as_ref(), &mut out).unwrap();
                bits(&out)[..real].to_vec()
            };
            assert_eq!(
                resident, routed,
                "matmul_mmq_experts dtype={dtype:?} variant={}: resident-BDA route diverged from \
                 the SSBO reference",
                v.label
            );
            println!(
                "ok: resident-BDA route matmul_mmq_experts dtype={dtype:?} variant={}",
                v.label
            );
        }
    }
}
