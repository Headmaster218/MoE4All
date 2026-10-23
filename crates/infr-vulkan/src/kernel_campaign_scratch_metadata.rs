//! R02 CPU-only adapter child. Parent mounts with cpu-tests-mount.applypatch.
//! Uses the REAL Graph, scratch_layout, scratch_reset_indices and plan memoizer.
//! No Vulkan backend/device/buffer is constructed. The optional ignored benchmark
//! counts actual Rust heap allocations on its own thread, not synthetic token Vecs.
//! Its System allocator wrapper is TEST-ONLY; do not mount in a production build.

use super::*;
use infr_core::{DType, TensorDesc};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::time::Instant;

thread_local! {
    static ALLOCS: Cell<(bool, u64, u64)> = const { Cell::new((false, 0, 0)) };
}

struct CountSystem;

fn allocated(bytes: usize) {
    let _ = ALLOCS.try_with(|state| {
        let (on, calls, total) = state.get();
        if on {
            state.set((true, calls + 1, total + bytes as u64));
        }
    });
}

unsafe impl GlobalAlloc for CountSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, layout, size) };
        if !ptr.is_null() {
            allocated(size);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static R02_TEST_ALLOCATOR: CountSystem = CountSystem;

fn counted<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    struct Disable;
    impl Drop for Disable {
        fn drop(&mut self) {
            ALLOCS.with(|s| {
                let (_, calls, bytes) = s.get();
                s.set((false, calls, bytes));
            });
        }
    }
    ALLOCS.with(|s| {
        assert!(!s.get().0, "allocation counters cannot nest");
        s.set((true, 0, 0));
    });
    let disable = Disable;
    let value = f();
    drop(disable);
    let (_, calls, bytes) = ALLOCS.with(Cell::get);
    (value, calls, bytes)
}

fn plan(graph: &Graph) -> VkDecodePlan {
    VkDecodePlan {
        graph: graph.clone(),
        eligible: false,
        replay: Mutex::new(None),
        static_scratch: OnceLock::new(),
    }
}

fn fixture() -> (Graph, [TensorId; 3]) {
    let mut g = Graph::new();
    let internal = || TensorDesc::new(vec![8], DType::F32);
    let src = g.input(internal());
    let full = g.internal(internal());
    let partial = g.internal(internal());
    let before_write = g.internal(internal());
    let read_out = g.internal(internal());
    let part_out = g.internal(internal());
    let strided = g.internal(internal());
    let stride_out = g.internal(internal());
    g.push(Op::Scale {
        x: before_write,
        dst: read_out,
        s: 1.0,
        n: 8,
    });
    g.push(Op::Copy {
        src,
        src_off: 0,
        dst: full,
        dst_off: 0,
        n: 8,
    });
    g.push(Op::Copy {
        src: full,
        src_off: 0,
        dst: partial,
        dst_off: 0,
        n: 3,
    });
    g.push(Op::Scale {
        x: partial,
        dst: part_out,
        s: 2.0,
        n: 8,
    });
    g.push(Op::CopyStrided {
        src,
        src_off: 0,
        src_stride: 8,
        dst: strided,
        dst_off: 0,
        dst_stride: 8,
        rows: 1,
        n: 4,
    });
    g.push(Op::Scale {
        x: strided,
        dst: stride_out,
        s: 0.5,
        n: 8,
    });
    (g, [partial, before_write, strided])
}

#[test]
fn r02_read_before_write_and_partial_producers_match_original_analysis() {
    let (g, reset) = fixture();
    let p = plan(&g);
    assert!(p.static_scratch.get().is_none());
    let m = p.static_scratch_metadata().unwrap();
    assert_eq!(m.layout, scratch_layout(&p.graph).unwrap());
    assert_eq!(m.reset_indices, scratch_reset_indices(&p.graph));
    assert_eq!(m.reset_indices, reset.map(|id| id.0 as usize));
    for _ in 0..128 {
        assert!(std::ptr::eq(m, p.static_scratch_metadata().unwrap()));
    }
}

#[test]
fn r02_row_dtype_two_plans_and_caller_graph_clone_are_isolated() {
    for (rows, dtype, bytes) in [
        (0, DType::F32, 4),
        (1, DType::F32, 2048),
        (65, DType::F16, 2048),
        (4096, DType::Bf16, 65536),
    ] {
        let mut g = Graph::new();
        let id = g.internal(TensorDesc::new(vec![rows, 8], dtype));
        let a = plan(&g);
        g.tensors[id.0 as usize].desc = TensorDesc::new(vec![129, 8], DType::F32);
        let b = plan(&g);
        let am = a.static_scratch_metadata().unwrap();
        let bm = b.static_scratch_metadata().unwrap();
        assert_eq!(am.layout[id.0 as usize], Some(bytes));
        assert_eq!(bm.layout[id.0 as usize], Some(6144));
        assert!(!std::ptr::eq(am, bm));
        assert_eq!(a.graph.desc(id).shape, vec![rows, 8]);
    }
}

#[test]
fn r02_invalid_layout_retries_without_publishing_or_moving_compile_error() {
    let mut g = Graph::new();
    let id = g.internal(TensorDesc::new(vec![256], DType::Q8_0));
    let invalid = plan(&g); // Construction deliberately remains infallible.
    for _ in 0..2 {
        let error = invalid.static_scratch_metadata().err().unwrap();
        assert!(error
            .to_string()
            .contains("internal tensor must be a dense dtype"));
        assert!(invalid.static_scratch.get().is_none());
    }
    g.tensors[id.0 as usize].desc.dtype = DType::F32;
    let corrected = plan(&g);
    assert!(corrected.static_scratch_metadata().is_ok());
    assert!(invalid.static_scratch.get().is_none());
}

// Logical-output CPU probe only: actual lowering/phase cleanup stays in the adapter.
fn copy_scale_probe(
    g: &Graph,
    m: &StaticScratchMetadata,
    data: &mut [Vec<f32>],
    fail_at: Option<usize>,
) -> Result<()> {
    for &i in &m.reset_indices {
        data[i].fill(0.0);
    }
    for (index, op) in g.ops.iter().enumerate() {
        if fail_at == Some(index) {
            return Err(be("R02 injected post-analysis lowering failure"));
        }
        match *op {
            Op::Copy {
                src,
                src_off,
                dst,
                dst_off,
                n,
            } => {
                for i in 0..n as usize {
                    data[dst.0 as usize][dst_off as usize + i] =
                        data[src.0 as usize][src_off as usize + i];
                }
            }
            Op::CopyStrided {
                src,
                src_off,
                src_stride,
                dst,
                dst_off,
                dst_stride,
                rows,
                n,
            } => {
                for row in 0..rows as usize {
                    for i in 0..n as usize {
                        data[dst.0 as usize][dst_off as usize + row * dst_stride as usize + i] =
                            data[src.0 as usize][src_off as usize + row * src_stride as usize + i];
                    }
                }
            }
            Op::Scale { x, dst, s, n } => {
                for i in 0..n as usize {
                    data[dst.0 as usize][i] = data[x.0 as usize][i] * s;
                }
            }
            _ => panic!("CPU probe covers only the Copy/Scale fixture, not Vulkan lowering"),
        }
    }
    Ok(())
}

#[test]
fn r02_post_analysis_failure_and_repeated_output_bytes_match_uncached() {
    let (g, _) = fixture();
    let p = plan(&g);
    let m = p.static_scratch_metadata().unwrap();
    for fail_at in 0..g.ops.len() {
        let mut a: Vec<_> = g
            .tensors
            .iter()
            .map(|t| vec![91.0; t.desc.numel()])
            .collect();
        let mut b = a.clone();
        for iteration in 0..8 {
            let values: Vec<_> = (0..8)
                .map(|i| (iteration * 8 + i) as f32 * -0.125)
                .collect();
            a[0].clone_from(&values);
            b[0].clone_from(&values);
            assert!(copy_scale_probe(&g, m, &mut a, Some(fail_at)).is_err());
            assert!(copy_scale_probe(
                &g,
                &StaticScratchMetadata::analyze(&g).unwrap(),
                &mut b,
                Some(fail_at)
            )
            .is_err());
            copy_scale_probe(&g, p.static_scratch_metadata().unwrap(), &mut a, None).unwrap();
            copy_scale_probe(
                &g,
                &StaticScratchMetadata::analyze(&g).unwrap(),
                &mut b,
                None,
            )
            .unwrap();
            assert!(std::ptr::eq(m, p.static_scratch_metadata().unwrap()));
            for (a, b) in a.iter().zip(&b) {
                assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }
}

#[test]
fn r02_warm_metadata_borrow_has_zero_heap_allocations() {
    let (g, _) = fixture();
    let p = plan(&g);
    p.static_scratch_metadata().unwrap();
    let (_, calls, bytes) = counted(|| {
        for _ in 0..128 {
            let m = black_box(&p).static_scratch_metadata().unwrap();
            black_box((&m.layout, &m.reset_indices));
        }
    });
    assert_eq!((calls, bytes), (0, 0));
}

#[test]
#[ignore = "parent-only CPU microbenchmark; actual scratch analysis/OnceLock, no Vulkan device"]
fn r02_metadata_cpu_alloc_and_timing() {
    for nodes in [16usize, 128] {
        let mut graph = Graph::new();
        let mut src = graph.input(TensorDesc::new(vec![4096, 2560], DType::F32));
        for node in 0..nodes {
            let dst = graph.internal(TensorDesc::new(vec![4096, 2560], DType::F32));
            graph.push(Op::Copy {
                src,
                src_off: 0,
                dst,
                dst_off: 0,
                n: if node % 8 == 0 { 5120 } else { 4096 * 2560 },
            });
            src = dst;
        }
        let p = plan(&graph);
        let (_, cold_calls, cold_bytes) = counted(|| {
            black_box(p.static_scratch_metadata().unwrap());
        });
        let metadata = p.static_scratch_metadata().unwrap();
        let retained = metadata.layout.capacity() * std::mem::size_of::<Option<usize>>()
            + metadata.reset_indices.capacity() * std::mem::size_of::<usize>();
        println!("R02_CPU setup ops={nodes} cold_alloc_calls={cold_calls} cold_requested_bytes={cold_bytes} retained_metadata_capacity_bytes={retained} inline_cache_bytes={} graph_clone_excluded=true buffers_allocated=false", std::mem::size_of::<OnceLock<StaticScratchMetadata>>());
        let reps = 1000;
        for round in 0..5 {
            for (slot, cached) in [false, true, true, false].into_iter().enumerate() {
                let analyze = || {
                    for _ in 0..reps {
                        if cached {
                            let m = black_box(&p).static_scratch_metadata().unwrap();
                            black_box((&m.layout, &m.reset_indices));
                        } else {
                            let m = StaticScratchMetadata::analyze(black_box(&p.graph)).unwrap();
                            black_box((&m.layout, &m.reset_indices));
                        }
                    }
                };
                let (_, calls, bytes) = counted(|| analyze());
                if cached {
                    assert_eq!((calls, bytes), (0, 0));
                } else {
                    assert!(calls > 0);
                }
                let started = Instant::now();
                analyze();
                let elapsed = started.elapsed();
                println!("R02_CPU ops={nodes} round={round} slot={slot} side={} reps={reps} elapsed_ns_per_call={:.3} alloc_calls={calls} requested_bytes={bytes} timing_counter_disabled=true actual_analysis=true gpu=false", if cached { "B" } else { "A" }, elapsed.as_nanos() as f64 / reps as f64);
            }
        }
    }
}
