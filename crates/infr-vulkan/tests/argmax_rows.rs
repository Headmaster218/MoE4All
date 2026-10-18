//! Regression for the multi-row greedy argmax used by MTP verify.
//!
//! The verify path reads token ids back through a `Readback` buffer. This test runs the exact
//! `Recorder::argmax` lowering with the same output placed in `Readback`, `Staging`, and
//! `Activations`, so a mismatch identifies the buffer class rather than the reduction itself.
use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::TensorDesc;
use infr_core::DType;
use infr_vulkan::VulkanBackend;

fn host_argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

fn check_usage(be: &VulkanBackend, usage: BufferUsage, label: &str, logits: &[f32], rows: usize, n: usize) {
    let logits_buf = be
        .alloc(logits.len() * std::mem::size_of::<f32>(), BufferUsage::Activations)
        .unwrap();
    let part_buf = be.alloc(512 * 4, BufferUsage::Activations).unwrap();
    let out_buf = be
        .alloc(rows * std::mem::size_of::<u32>(), usage)
        .unwrap();

    be.upload(logits_buf.as_ref(), bytemuck::cast_slice(logits))
        .unwrap();

    let rec = be.recorder().unwrap();
    rec.argmax(logits_buf.as_ref(), part_buf.as_ref(), out_buf.as_ref(), n, rows);
    rec.finish().unwrap();

    let mut bytes = vec![0u8; rows * std::mem::size_of::<u32>()];
    be.download(out_buf.as_ref(), &mut bytes).unwrap();
    let got: &[u32] = bytemuck::cast_slice(&bytes);
    for (row, &value) in got.iter().enumerate() {
        let want = host_argmax(&logits[row * n..(row + 1) * n]);
        assert_eq!(
            value, want,
            "{label}: row {row}/{rows} (vocab={n}) got {value}, want {want}"
        );
    }
}

#[test]
fn argmax_rows_matches_host_for_every_output_buffer_class() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };

    let (rows, n) = (4usize, 151_936usize);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut logits = Vec::with_capacity(rows * n);
    for row in 0..rows {
        for col in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut value = ((state >> 40) as f32 / (1u64 << 24) as f32) * 20.0 - 10.0;
            // Force a non-trivial winning index away from both ends of every row.
            if col == 12_345 + row {
                value += 100.0;
            }
            logits.push(value);
        }
    }

    check_usage(&be, BufferUsage::Readback, "Readback", &logits, rows, n);
    check_usage(&be, BufferUsage::Staging, "Staging", &logits, rows, n);
    check_usage(&be, BufferUsage::Activations, "Activations", &logits, rows, n);
    eprintln!("argmax multi-row output buffer classes OK");
}

#[test]
fn graph_argmax_rows_matches_host_through_adapter() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };

    let (rows, n) = (4usize, 151_936usize);
    let mut state = 0xD1B5_4A32_D192_ED03u64;
    let mut logits = Vec::with_capacity(rows * n);
    for row in 0..rows {
        for col in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut value = ((state >> 40) as f32 / (1u64 << 24) as f32) * 20.0 - 10.0;
            if col == 54_321 - row {
                value += 100.0;
            }
            logits.push(value);
        }
    }

    let mut graph = Graph::new();
    let x = graph.input(TensorDesc::new(vec![rows, n], DType::F32));
    let ids = graph.output(TensorDesc::new(vec![rows], DType::U32));
    graph.push(Op::Argmax {
        x,
        dst: ids,
        n: n as u32,
        rows: rows as u32,
    });

    let x_buf = be
        .alloc(logits.len() * std::mem::size_of::<f32>(), BufferUsage::Activations)
        .unwrap();
    let ids_buf = be
        .alloc(rows * std::mem::size_of::<u32>(), BufferUsage::Readback)
        .unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&logits)).unwrap();

    let plan = be.compile(&graph).unwrap();
    let mut bindings = Bindings::new();
    bindings.bind(x, x_buf.as_ref()).bind(ids, ids_buf.as_ref());
    be.execute(plan.as_ref(), &bindings).unwrap();

    let mut bytes = vec![0u8; rows * std::mem::size_of::<u32>()];
    be.download(ids_buf.as_ref(), &mut bytes).unwrap();
    let got: &[u32] = bytemuck::cast_slice(&bytes);
    for row in 0..rows {
        let want = host_argmax(&logits[row * n..(row + 1) * n]);
        assert_eq!(
            got[row], want,
            "Graph adapter: row {row}/{rows} (vocab={n}) got {}, want {want}",
            got[row]
        );
    }
    eprintln!("argmax multi-row Graph adapter path OK");
}
