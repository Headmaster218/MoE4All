use infr_core::{
    backend::{Backend, Bindings, BufferUsage, MAX_DECODE_ROWS},
    graph::{Graph, Op},
    DType, TensorDesc,
};
use infr_cpu::CpuBackend;
use infr_vulkan::VulkanBackend;

fn host_sample(logits: &[f32], uniform: f32, stochastic: bool) -> u32 {
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    if !stochastic {
        return indices[0] as u32;
    }
    indices.truncate(20);
    let mut probabilities = indices
        .iter()
        .map(|&i| ((logits[i] - logits[indices[0]]) / 0.8).exp())
        .collect::<Vec<_>>();
    let sum: f32 = probabilities.iter().sum();
    for p in &mut probabilities {
        *p /= sum;
    }
    let mut cumulative = 0.0;
    let cutoff = probabilities
        .iter()
        .position(|&p| {
            cumulative += p;
            cumulative >= 0.95
        })
        .map_or(probabilities.len(), |i| i + 1);
    let target = uniform * probabilities[..cutoff].iter().sum::<f32>();
    cumulative = 0.0;
    for i in 0..cutoff {
        cumulative += probabilities[i];
        if target <= cumulative {
            return indices[i] as u32;
        }
    }
    indices[cutoff - 1] as u32
}

fn check_penalties(backend: &dyn Backend, rows: usize, stochastic: bool) {
    let n = 1031;
    let mut graph = Graph::new();
    let x = graph.input(TensorDesc::new(vec![rows * n], DType::F32));
    let counts = graph.input(TensorDesc::new(vec![rows * n], DType::F32));
    let control = graph.input(TensorDesc::new(vec![rows * 4], DType::F32));
    let u = graph.input(TensorDesc::new(vec![rows], DType::F32));
    let ids = graph.output(TensorDesc::new(vec![rows], DType::F32));
    graph.push(Op::SamplingPenalty {
        x,
        counts,
        control,
        n: n as u32,
        rows: rows as u32,
    });
    if stochastic {
        graph.push(Op::Sample {
            x,
            u,
            dst: ids,
            n: n as u32,
            rows: rows as u32,
            top_k: 20,
            temp: 0.8,
            top_p: 0.95,
        });
    } else {
        graph.push(Op::Argmax {
            x,
            dst: ids,
            n: n as u32,
            rows: rows as u32,
        });
    }
    let plan = backend.compile(&graph).unwrap();
    let xb = backend
        .alloc(rows * n * 4, BufferUsage::Activations)
        .unwrap();
    let cb = backend
        .alloc(rows * n * 4, BufferUsage::Activations)
        .unwrap();
    let pb = backend.alloc(rows * 16, BufferUsage::Staging).unwrap();
    let ub = backend.alloc(rows * 4, BufferUsage::Staging).unwrap();
    let ob = backend.alloc(rows * 4, BufferUsage::Readback).unwrap();
    let uniforms: Vec<f32> = (0..rows)
        .map(|row| (row as f32 + 0.3) / rows as f32)
        .collect();
    backend
        .upload(ub.as_ref(), bytemuck::cast_slice(&uniforms))
        .unwrap();
    let mut bindings = Bindings::new();
    bindings
        .bind(x, xb.as_ref())
        .bind(counts, cb.as_ref())
        .bind(control, pb.as_ref())
        .bind(u, ub.as_ref())
        .bind(ids, ob.as_ref());
    // Repeat a whole request with the same buffers: history must be reset, not inherited.
    for request in 0..2 {
        let mut expected_counts = vec![0f32; rows * n];
        for row in 0..rows {
            expected_counts[row * n + 3] = 2.0;
        }
        backend
            .upload(cb.as_ref(), bytemuck::cast_slice(&expected_counts))
            .unwrap();
        for step in 0..6 {
            let mut logits: Vec<f32> = (0..rows * n)
                .map(|i| ((i * 37 % n) as f32 - 501.0) / 31.0)
                .collect();
            let controls: Vec<[u32; 4]> = (0..rows)
                .map(|row| {
                    let (presence, frequency): (f32, f32) = match row % 4 {
                        0 => (1.5, 0.0),
                        1 => (0.0, 0.5),
                        2 => (-0.5, -0.25),
                        _ => (0.0, 0.0),
                    };
                    // Reordered lanes, repeated id, no id, and out-of-vocab repair ids.
                    let previous = match step {
                        0 => u32::MAX,
                        1 | 2 => 3,
                        3 => n as u32 + 17,
                        _ => (row * 13 + 7) as u32,
                    };
                    [
                        presence.to_bits(),
                        frequency.to_bits(),
                        previous,
                        (rows - 1 - row) as u32,
                    ]
                })
                .collect();
            backend
                .upload(xb.as_ref(), bytemuck::cast_slice(&logits))
                .unwrap();
            backend
                .upload(pb.as_ref(), bytemuck::cast_slice(&controls))
                .unwrap();
            for row in 0..rows {
                let [presence, frequency, previous, count_row] = controls[row];
                let presence = f32::from_bits(presence);
                let frequency = f32::from_bits(frequency);
                let counts =
                    &mut expected_counts[count_row as usize * n..(count_row as usize + 1) * n];
                if let Some(count) = counts.get_mut(previous as usize) {
                    *count += 1.0;
                }
                for (logit, &count) in logits[row * n..(row + 1) * n].iter_mut().zip(counts.iter())
                {
                    if count > 0.0 {
                        *logit -= presence + frequency * count;
                    }
                }
            }
            backend.execute(plan.as_ref(), &bindings).unwrap();
            let mut actual = vec![0f32; rows * n];
            backend
                .download(xb.as_ref(), bytemuck::cast_slice_mut(&mut actual))
                .unwrap();
            assert_eq!(actual, logits, "request={request} step={step} rows={rows}");
            backend
                .download(cb.as_ref(), bytemuck::cast_slice_mut(&mut actual))
                .unwrap();
            assert_eq!(actual, expected_counts, "committed counts, not draft ids");
            let mut actual_ids = vec![0u32; rows];
            backend
                .download(ob.as_ref(), bytemuck::cast_slice_mut(&mut actual_ids))
                .unwrap();
            for row in 0..rows {
                let expected =
                    host_sample(&logits[row * n..(row + 1) * n], uniforms[row], stochastic);
                assert_eq!(actual_ids[row], expected, "sample after penalty, row={row}");
            }
        }
    }
}

#[test]
fn penalty_cpu_reference() {
    for rows in [1, 2, 4, MAX_DECODE_ROWS] {
        for stochastic in [false, true] {
            check_penalties(&CpuBackend::new(), rows, stochastic);
        }
    }
}

#[test]
#[ignore = "requires an idle Vulkan GPU"]
fn gpu_penalties_match_cpu_semantics() {
    let backend = VulkanBackend::new().unwrap();
    for rows in [1, 2, 4, MAX_DECODE_ROWS] {
        for stochastic in [false, true] {
            check_penalties(&backend, rows, stochastic);
        }
    }
}
