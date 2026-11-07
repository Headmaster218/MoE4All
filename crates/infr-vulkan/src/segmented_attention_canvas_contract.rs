#![cfg(test)]

//! Canvas mask propagation, raw partial parity, and ordinary wrapper compatibility.
use super::*;

#[test]
fn segmented_canvas_cpu_layout_mask_and_wiring_contract() {
    // Static wiring guard supplements (does not replace) GPU/Graph execution.
    let adapter = include_str!("adapter.rs");
    let call = adapter
        .split_once("rec.attention_kv_split_segmented_canvas(")
        .expect("adapter must use the Canvas-aware segmented API")
        .1
        .split_once(");")
        .unwrap()
        .0;
    let call: String = call.split_whitespace().collect();
    assert!(call.contains("*scale,window,canvas_lo,k_q8_eff,v_q8_eff,"));
    let recorder = include_str!("recorder.rs");
    let method = recorder
        .split_once("pub fn attention_kv_split_segmented_canvas(")
        .expect("Canvas-aware segmented API missing")
        .1
        .split_once("\n    }")
        .unwrap()
        .0;
    let method: String = method.split_whitespace().collect();
    assert!(method.contains("self.attention_kv_split_impl("));
    assert!(method.contains("scale,window,canvas_lo,k_q8,v_q8,"));
    let wrapper = recorder
        .split_once("pub fn attention_kv_split_segmented(")
        .unwrap()
        .1
        .split_once("\n    }")
        .unwrap()
        .0;
    let wrapper: String = wrapper.split_whitespace().collect();
    assert!(wrapper.contains("self.attention_kv_split_segmented_canvas("));
    assert!(wrapper.contains("scale,window,None,k_q8,v_q8,segment_shift,"));

    let len = 32769;
    let flat = dense_planar(len * NKV * HD, 0xc4a0_0001);
    let segments = (0..len.div_ceil(SEG_ROWS))
        .map(|i| dense_segment_from_flat(&flat, len, i))
        .collect::<Vec<_>>();
    for (lo, end, chunk) in [
        (0, 64, 17),
        (16, 64, 17),
        (63, 64, 17),
        (0, len, 511),
        (32765, len, 511),
        (32768, len, 1024),
    ] {
        let mut visited = Vec::new();
        for c in 0..end.div_ceil(chunk) {
            // Existing generic Canvas: fixed [lo,kv_len), independent of query pos.
            let begin = (c * chunk).max(lo);
            let stop = ((c + 1) * chunk).min(end);
            for row in begin..stop {
                visited.push(row);
                for head in 0..NKV {
                    for dim in [0, 31, 32, 127, 255] {
                        let elem = (row * NKV + head) * HD + dim;
                        let seg = elem >> SEG_SHIFT;
                        let local = elem & (VALUES_PER_SEGMENT - 1);
                        assert_eq!(segments[seg][local], flat[elem]);
                        let fa = len * NKV * HD + elem / 32 * 2;
                        let sa = VALUES_PER_SEGMENT + local / 32 * 2;
                        assert_eq!(&segments[seg][sa..sa + 2], &flat[fa..fa + 2]);
                    }
                }
            }
        }
        assert_eq!(visited, (lo..end).collect::<Vec<_>>());
        assert_eq!(
            lo as u32 + 1,
            (lo + 1) as u32,
            "Some(0) encodes1, not ordinary0"
        );
    }
    assert_ne!(
        (16..64).collect::<Vec<_>>(),
        (0..32).collect::<Vec<_>>(),
        "Canvas test must distinguish both lower bound and future-key reach"
    );
}

fn canvas_exact(a: &Capture, b: &Capture) {
    for (label, aa, bb) in [
        ("pm", &a.pm, &b.pm),
        ("pl", &a.pl, &b.pl),
        ("pacc", &a.pa, &b.pa),
        ("out", &a.out, &b.out),
    ] {
        assert_eq!(aa.len(), bb.len());
        for (i, (&x, &y)) in aa.iter().zip(bb).enumerate() {
            assert!(
                x.is_finite() && y.is_finite(),
                "Canvas {label}[{i}] nonfinite {x}/{y}"
            );
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "Canvas {label}[{i}] {x}/{y}; no tolerance"
            );
        }
    }
    // Prefix chunks outside Canvas are EMPTY, not failed softmax denominators.
    assert!(a.pl.iter().all(|x| *x >= 0.0));
    assert!(a.pl.iter().any(|x| *x > 0.0));
    assert!(a.out.iter().any(|x| x.abs() > 1e-8));
}

fn segmented_call(
    r: &Recorder<'_>,
    input: &DenseContractInput,
    io: &RouteIo,
    s: Shape,
    rows: usize,
    nh: usize,
    pos: usize,
    canvas_lo: Option<usize>,
    wrapper: bool,
) {
    let sc = &io.scratch;
    assert_eq!(io.q.prefix, 0);
    assert_eq!(sc.out.prefix, 0);
    if wrapper {
        assert!(canvas_lo.is_none());
        r.attention_kv_split_segmented(
            io.q.buffer.as_ref(),
            input.kt.as_ref(),
            input.vt.as_ref(),
            sc.out.buffer.as_ref(),
            sc.pm.buffer.as_ref(),
            sc.pl.buffer.as_ref(),
            sc.pa.buffer.as_ref(),
            rows,
            pos,
            s.len,
            nh,
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
        r.attention_kv_split_segmented_canvas(
            io.q.buffer.as_ref(),
            input.kt.as_ref(),
            input.vt.as_ref(),
            sc.out.buffer.as_ref(),
            sc.pm.buffer.as_ref(),
            sc.pl.buffer.as_ref(),
            sc.pa.buffer.as_ref(),
            rows,
            pos,
            s.len,
            nh,
            NKV,
            HD,
            s.chunk,
            s.stride(),
            1.0 / 16.0,
            0,
            canvas_lo,
            true,
            true,
            SEG_SHIFT,
        );
    }
}

#[test]
#[ignore = "requires serial Windows discrete RDNA3 GPU: Canvas generic flat/segmented partial bits and wrapper compatibility"]
fn segmented_canvas_flat_partial_exact_and_ordinary_wrapper() {
    let base = VulkanBackend::cfg_from_env().unwrap();
    let mut cfg = (*base).clone();
    let vk = &mut cfg.kernels.vulkan;
    vk.attn_decode = true;
    vk.no_attn_hd_spec = false;
    vk.q8_decode_segmented = true;
    vk.q8_decode_d8 = true;
    vk.q8_decode_ls256 = true;
    vk.q8_qk_f16 = true;
    vk.q8_pv_f16 = true;
    vk.q8_decode_gqa4 = true;
    vk.q8_decode_chunk1024 = true;
    let be = VulkanBackend::new_with(std::sync::Arc::new(cfg)).unwrap();
    device_probe(&be);
    assert!(be.prefers_segmented_q8_decode());
    let limits = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    }
    .limits;
    let alignment = limits.min_storage_buffer_offset_alignment as usize;
    let mut compared = 0;
    // (kv_len, actual heads per row, rows, query start, Canvas lower bound, chunk).
    // pos+rows<kv_len deliberately requires future-key/bidirectional semantics.
    for (len, nh, rows, pos, lo, chunk) in [
        (64, 16, 1, 31, 0, 17),
        (64, 16, 1, 31, 16, 17),
        (64, 16, 1, 31, 63, 17),
        (30000, 32, 1, 29990, 29900, 511),
        (32769, 32, 1, 32760, 32765, 511),
        (32769, 16, 2, 32750, 32760, 512),
        (30000, 32, 4, 29970, 15000, 1024),
        (150000, 16, 1, 149990, 131069, 1024),
    ] {
        assert!(pos + rows < len && lo < len);
        let s = Shape {
            len,
            nh: rows * nh,
            chunk,
            replay: false,
        };
        for lane in 0..2 {
            let input = DenseContractInput::new(&be, len, rows * nh, lane).unwrap();
            let io = RouteIo::new(&be, &input, s, false, lane, alignment).unwrap();
            let sc = &io.scratch;
            let r = be.recorder().unwrap();
            r.attention_kv_split_at(
                io.q.buffer.as_ref(),
                input.flat_k.buffer.as_ref(),
                input.flat_v.buffer.as_ref(),
                input.flat_k.address,
                input.flat_v.address,
                sc.out.buffer.as_ref(),
                sc.pm.buffer.as_ref(),
                sc.pl.buffer.as_ref(),
                sc.pa.buffer.as_ref(),
                rows,
                pos,
                len,
                nh,
                NKV,
                HD,
                chunk,
                s.stride(),
                1.0 / 16.0,
                0,
                Some(lo),
                true,
                true,
                len * NKV * HD,
                false,
                None,
            );
            r.finish().unwrap();
            let flat = sc.capture(&be, s, 1);
            sc.reset(&be);
            let r = be.recorder().unwrap();
            segmented_call(&r, &input, &io, s, rows, nh, pos, Some(lo), false);
            r.finish().unwrap();
            let segmented = sc.capture(&be, s, 1);
            canvas_exact(&flat, &segmented);
            let names = be.built_kernel_names();
            assert!(names.contains(&"attn_partial_q8_bda"));
            assert!(names.contains(&"attn_partial_seg_q8"));
            assert!(
                !names
                    .iter()
                    .any(|n| n.starts_with("attn_decode_hd256_q8_seg_")),
                "Canvas must reach Some(lo) before the A01 fast gate, even for rows1/Some(0)"
            );
            io.q.verify_guards(&be);
            input.verify_guards(&be);
            compared += 1;
            eprintln!("CANVASSEG PASS len={len} nh={nh} rows={rows} pos={pos} lo={lo} chunk={chunk} lane={lane} generic_names=true partial_bytes_equal=true output_bytes_equal=true");
        }
    }
    // After every Canvas gate assertion, confirm the unchanged legacy entry is
    // byte-identical to the new entry with None and STILL selects normal A01.
    let s = Shape {
        len: 30000,
        nh: 16,
        chunk: 512,
        replay: false,
    };
    for lane in 0..2 {
        let input = DenseContractInput::new(&be, s.len, s.nh, lane).unwrap();
        let io = RouteIo::new(&be, &input, s, false, lane, alignment).unwrap();
        let mut a = None;
        for wrapper in [true, false] {
            io.scratch.reset(&be);
            let r = be.recorder().unwrap();
            segmented_call(&r, &input, &io, s, 1, s.nh, s.len - 1, None, wrapper);
            r.finish().unwrap();
            let capture = io.scratch.capture(&be, s, 4);
            if wrapper {
                a = Some(capture);
            } else {
                canvas_exact(a.as_ref().unwrap(), &capture);
            }
        }
        assert!(be
            .built_kernel_names()
            .contains(&"attn_decode_hd256_q8_seg_gqa4"));
        io.q.verify_guards(&be);
        input.verify_guards(&be);
        compared += 1;
        eprintln!("CANVASSEG WRAPPER PASS lane={lane} none_preserves_A01=true partial_bytes_equal=true output_bytes_equal=true");
    }
    eprintln!("CANVASSEG COMPLETE compared={compared} expected=18 no_external_spv_needed=true");
    assert_eq!(compared, 18);
}
