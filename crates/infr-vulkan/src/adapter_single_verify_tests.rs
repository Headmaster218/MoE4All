//! One-layer Verify admission, recovery and retained-controller reuse.
use super::*;

struct VerifyGraph {
    graph: Graph,
    frontend: TensorId,
    trailing: TensorId,
    buffers: Vec<(TensorId, Box<dyn Buffer>)>,
    expected: Vec<(TensorId, Vec<f32>)>,
}

impl VerifyGraph {
    fn new(fixture: &Fixture, rows: usize, layers: usize) -> Self {
        assert!((2..=4).contains(&rows));
        assert!((1..=LAYERS).contains(&layers));
        let mut graph = fixture.graph.clone();
        let frontend_op = graph.ops[0].clone();
        let Op::Add { b: one, .. } = &frontend_op else {
            unreachable!()
        };
        let one = *one;
        let Op::MoeFfn { x, router_x, .. } = &graph.ops[1] else {
            unreachable!()
        };
        let (input, routing) = (*x, *router_x);
        graph.ops.retain(|op| matches!(op, Op::MoeFfn { .. }));
        graph.ops.truncate(layers);
        graph.ops.insert(0, frontend_op);
        graph.mtp_verify = true;
        graph.independent_rows = false;
        graph.sequence_spans.clear();
        for tensor in &mut graph.tensors {
            if tensor.kind != infr_core::graph::TensorKind::Weight {
                tensor.desc = TensorDesc::new(vec![rows, NE], DType::F32);
            }
        }
        if let Op::Add { n, .. } = &mut graph.ops[0] {
            *n = (rows * NE) as u32;
        }
        let trailing = graph.input(TensorDesc::new(vec![rows, NE], DType::F32));
        graph.push(Op::Add {
            a: trailing,
            b: one,
            dst: trailing,
            n: (rows * NE) as u32,
        });
        let input_values: Vec<_> = (0..rows * NE)
            .map(|index| 4.0 + (index / NE) as f32 / 4.0 + (index % NE) as f32 / 64.0)
            .collect();
        let mut buffers = Vec::new();
        for (index, tensor) in graph.tensors.iter().enumerate() {
            if tensor.kind == infr_core::graph::TensorKind::Weight {
                continue;
            }
            let id = TensorId(index as u32);
            let values = if id == input {
                input_values.clone()
            } else if id == one || id == routing {
                vec![1.0f32; rows * NE]
            } else {
                vec![0.0f32; rows * NE]
            };
            let buffer = fixture
                .backend
                .alloc(rows * NE * 4, BufferUsage::Readback)
                .unwrap();
            fixture
                .backend
                .upload(buffer.as_ref(), bytemuck::cast_slice(&values))
                .unwrap();
            buffers.push((id, buffer));
        }
        let mut values = input_values;
        let mut expected = Vec::new();
        for (layer, &id) in fixture.outputs[..layers].iter().enumerate() {
            values = scalar_layer(&values, layer);
            expected.push((id, values.clone()));
        }
        Self {
            graph,
            frontend: fixture.frontend,
            trailing,
            buffers,
            expected,
        }
    }

    fn buffer(&self, id: TensorId) -> &dyn Buffer {
        self.buffers
            .iter()
            .find(|(key, _)| *key == id)
            .unwrap()
            .1
            .as_ref()
    }

    fn run(&self, fixture: &Fixture, missing: bool) {
        let rows = self.graph.desc(self.frontend).numel() / NE;
        let layouts = resident_graph_layouts(&fixture.backend, &self.graph, true)
            .expect("Verify must admit the resident controller, including one MoE");
        assert_eq!(layouts.len(), self.expected.len());
        assert!(layouts.iter().all(|layout| layout.rows == rows));
        for id in [self.frontend, self.trailing] {
            fixture
                .backend
                .upload(
                    self.buffer(id),
                    bytemuck::cast_slice(&vec![0.0f32; rows * NE]),
                )
                .unwrap();
        }
        fixture.preload();
        if missing {
            let mut guard = fixture.backend.moe_pager().lock().unwrap();
            let session = guard.as_mut().unwrap();
            session.discard_cpu_miss(fixture.roles[0], &[ROUTED[0]]);
            assert_eq!(
                session
                    .routed_roles_resident_mask(&fixture.roles[0], &ROUTED)
                    .unwrap(),
                HIT_MASK & !1
            );
        }
        let mut bindings = fixture.bindings();
        for (id, buffer) in &self.buffers {
            bindings.bind(*id, buffer.as_ref());
        }
        let plan = VkDecodePlan::new_cpu(&self.graph, false, true);
        execute_static(&fixture.backend, &plan, &bindings).unwrap();
        for id in [self.frontend, self.trailing] {
            assert_eq!(
                read_values(&fixture.backend, self.buffer(id), rows * NE),
                vec![1.0f32; rows * NE],
                "frontend/trailing must execute once, even after ordinal-zero recovery"
            );
        }
        for (id, expected) in &self.expected {
            let actual = read_values(&fixture.backend, self.buffer(*id), rows * NE);
            for (&actual, &expected) in actual.iter().zip(expected) {
                assert!(
                    actual.is_finite()
                        && (actual - expected).abs() <= 1e-10f32.max(expected.abs() * 1e-4),
                    "rows={rows}, layers={}, miss0={missing}: {actual} != {expected}",
                    self.expected.len()
                );
            }
        }
        let guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_ref().unwrap();
        for roles in &fixture.roles[..self.expected.len()] {
            assert_eq!(
                session.routed_roles_resident_mask(roles, &ROUTED).unwrap(),
                HIT_MASK,
                "hit and restored miss must both remain visible"
            );
        }
    }
}

fn read_values(backend: &VulkanBackend, buffer: &dyn Buffer, count: usize) -> Vec<f32> {
    let mut values = vec![0.0f32; count];
    backend
        .download(buffer, bytemuck::cast_slice_mut(&mut values))
        .unwrap();
    values
}

fn scalar_layer(input: &[f32], layer: usize) -> Vec<f32> {
    let first_weight = 1.0 / (1.0 + (-1.0f32).exp());
    input
        .iter()
        .map(|&x| {
            (0..TOP)
                .map(|expert| {
                    let scale =
                        |role| 0.125 + (layer * 3 + role) as f32 / 64.0 + expert as f32 / 32.0;
                    let gate = x * scale(0);
                    let up = x * scale(1);
                    let weight = if expert == 0 {
                        first_weight
                    } else {
                        1.0 - first_weight
                    };
                    weight * scale(2) * (gate / (1.0 + (-gate).exp())) * up
                })
                .sum()
        })
        .collect()
}

fn assert_drained(fixture: &Fixture, producer_enabled: bool) -> (Arc<dyn Buffer>, Arc<dyn Buffer>) {
    let (control, journal, notices, producer) = {
        let phase = fixture.backend.runtime_phase.lock().unwrap();
        let resident = phase
            .resident_buffers
            .as_ref()
            .expect("real execution must retain resident state");
        assert!(resident.gpu_copy.is_none());
        assert!(!resident.gpu_copy_init_attempted);
        assert_eq!(resident.producer.is_some(), producer_enabled);
        (
            Arc::clone(&resident.control),
            Arc::clone(&resident.journal),
            resident.notices.clone(),
            resident.producer.clone(),
        )
    };
    if let Some(producer) = producer {
        // FIFO acknowledgement rules out a false empty-queue pass while a job is still running.
        let (done, receiver) = mpsc::channel();
        producer
            .sender
            .as_ref()
            .unwrap()
            .send(ResidentNoticeJob {
                journal: Arc::clone(&journal),
                notices: notices.unwrap(),
                fence: None,
                entries: Vec::new(),
                mapped_serial_max_bytes: 0,
                done,
                trace_context: None,
                gpu_copy: None,
            })
            .unwrap();
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("notice worker must finish every prior job");
        assert!(
            producer.receive(false).unwrap().is_none(),
            "undrained notice result"
        );
        assert_eq!(producer.announced_miss.load(Ordering::Acquire), usize::MAX);
    }
    (control, journal)
}

#[test]
#[ignore = "requires an idle Vulkan GPU; model-free one-layer Verify and 4 MiB pool"]
fn gpu_single_verify_hit_miss_and_prefix_suffix_prefix() {
    for (submit_layers, producer) in [(1, false), (1, true), (2, false), (2, true)] {
        let fixture = Fixture::new_with_copy_policy(submit_layers, producer, Some(false));
        let mut retained: Option<(Arc<dyn Buffer>, Arc<dyn Buffer>)> = None;
        for rows in [2, 3, 4] {
            let prefix = VerifyGraph::new(&fixture, rows, 1);
            let suffix = VerifyGraph::new(&fixture, rows, 2);
            for missing in [false, true] {
                for graph in [&prefix, &suffix, &prefix] {
                    // A reused controller must publish fresh metadata, not pass using stale state.
                    if let Some((control, journal)) = &retained {
                        fixture
                            .backend
                            .upload(control.as_ref(), bytemuck::cast_slice(&[u32::MAX; 4]))
                            .unwrap();
                        fixture
                            .backend
                            .upload(
                                journal.as_ref(),
                                bytemuck::cast_slice(
                                    &[u32::MAX; crate::recorder::RESIDENT_JOURNAL_WORDS],
                                ),
                            )
                            .unwrap();
                    }
                    graph.run(&fixture, missing);
                    let current = assert_drained(&fixture, producer);
                    if let Some((control, journal)) = &retained {
                        assert!(Arc::ptr_eq(control, &current.0));
                        assert!(Arc::ptr_eq(journal, &current.1));
                    }
                    if graph.expected.len() == 1 {
                        let mut status = [0u32; 4];
                        fixture
                            .backend
                            .download(current.0.as_ref(), bytemuck::cast_slice_mut(&mut status))
                            .unwrap();
                        assert_eq!(
                            status,
                            // Non-producer recovery clears the chunk before replay; the
                            // trailing Add starts an empty chunk and resets only control.
                            // Early recovery includes that Add before retiring at graph end.
                            if missing && !producer {
                                [1, u32::MAX, 0, 0]
                            } else if missing {
                                [0, 0, 1, 0]
                            } else {
                                [1, u32::MAX, 1, 0]
                            }
                        );
                        let words = resident_notice_words(current.1.as_ref(), 0).unwrap();
                        assert_eq!(
                            words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD],
                            if missing { 2 } else { 1 }
                        );
                        for row in 0..rows {
                            assert_eq!(&words[row * TOP..(row + 1) * TOP], &ROUTED);
                            assert_eq!(
                                words[crate::recorder::RESIDENT_JOURNAL_MASK_WORD + row],
                                if missing { HIT_MASK & !1 } else { HIT_MASK }
                            );
                        }
                    }
                    retained = Some(current);
                }
            }
        }
    }
}
