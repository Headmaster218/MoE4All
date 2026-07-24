//! Slice 26 probe — isolated throughput of a library f16 GEMM (rocBLAS `rocblas_gemm_ex`,
//! fp16 in/out, fp32 accumulate) on representative prefill shapes, to judge whether the
//! dequant→f16→library-GEMM route beats the hand-written int8 WMMA kernel (Slice 25 plateau
//! ~11-14 TFLOP/s). Self-contained: declares its own HIP + rocBLAS FFI, allocates raw device
//! buffers, and times `rocblas_gemm_ex` in a tight loop. No integration — a pure measurement.
//!
//! Measured (gfx1100): 25.8 / 77.0 / 58.2 / 74.0 TFLOP/s vs the hand kernel's 7.1 / 13.1 / 11.1 /
//! 14.2 → 3.6-5.9× on the GEMM ITSELF. BUT integrating it (INFR_ROCM_BLAS=1) LOSES end-to-end:
//! ~0.88× pp512 on 0.6B-8B (the per-forward dequant→f16 tax outweighs the GEMM win) and the
//! transient f16 buffers OOM at 8B. So the shipping prefill path stays on int8 WMMA; the right
//! next step is a pipelined int8 MMQ (llama.cpp's actual fast quant path), not this f16 route.
//!
//! Build: cargo build --release --features rocm -p infr-rocm --example blas_probe
//! Run:   LD_LIBRARY_PATH=/opt/rocm/lib ./target/release/examples/blas_probe

#![allow(non_camel_case_types)]

use half::f16;
use std::ffi::c_void;
use std::time::Instant;

// ── HIP FFI ──────────────────────────────────────────────────────────────────
type hipStream_t = *mut c_void;
#[link(name = "amdhip64")]
extern "C" {
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn hipStreamCreate(stream: *mut hipStream_t) -> i32;
    fn hipStreamSynchronize(stream: hipStream_t) -> i32;
    fn hipDeviceSynchronize() -> i32;
}
const H2D: i32 = 1;

// ── rocBLAS FFI ──────────────────────────────────────────────────────────────
type rocblas_handle = *mut c_void;
const OP_NONE: i32 = 111;
const DT_F16_R: i32 = 150;
const DT_F32_R: i32 = 151;
const ALGO_STANDARD: i32 = 0;
const ROCBLAS_SUCCESS: i32 = 0;

#[link(name = "rocblas")]
extern "C" {
    fn rocblas_create_handle(handle: *mut rocblas_handle) -> i32;
    fn rocblas_destroy_handle(handle: rocblas_handle) -> i32;
    fn rocblas_set_stream(handle: rocblas_handle, stream: hipStream_t) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn rocblas_gemm_ex(
        handle: rocblas_handle,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        a_type: i32,
        lda: i32,
        b: *const c_void,
        b_type: i32,
        ldb: i32,
        beta: *const c_void,
        c: *const c_void,
        c_type: i32,
        ldc: i32,
        d: *mut c_void,
        d_type: i32,
        ldd: i32,
        compute_type: i32,
        algo: i32,
        solution_index: i32,
        flags: u32,
    ) -> i32;
}

fn dev_f16(n: usize) -> *mut c_void {
    let host: Vec<f16> = (0..n)
        .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) * 0.05))
        .collect();
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let rc = unsafe { hipMalloc(&mut ptr, n * 2) };
    assert_eq!(rc, 0, "hipMalloc");
    let rc = unsafe { hipMemcpy(ptr, host.as_ptr() as *const c_void, n * 2, H2D) };
    assert_eq!(rc, 0, "hipMemcpy");
    ptr
}

/// Time an f16 GEMM D[m,n] = A[m,k] * B[k,n] (2*m*n*k flops) via rocblas_gemm_ex, GFLOP/s.
/// Column-major library: pass the row-major problem as its column-major transpose
/// (compute D^T[n,m] = B^T * A^T), which for throughput measurement executes the identical
/// m*n*k FMA volume.
fn bench(
    h: rocblas_handle,
    stream: hipStream_t,
    m: usize,
    k: usize,
    n: usize,
    iters: usize,
) -> f64 {
    let a = dev_f16(m * k);
    let b = dev_f16(k * n);
    let mut d: *mut c_void = std::ptr::null_mut();
    assert_eq!(unsafe { hipMalloc(&mut d, m * n * 2) }, 0);
    let alpha = f32::to_bits(1.0);
    let beta = f32::to_bits(0.0);
    // col-major: D_cm[n x m] = B_cm[n x k] * A_cm[k x m], no transpose.
    // B row-major [k,n] == col-major [n,k]; A row-major [m,k] == col-major [k,m].
    let call = |_i: usize| unsafe {
        rocblas_gemm_ex(
            h,
            OP_NONE,
            OP_NONE,
            n as i32,
            m as i32,
            k as i32,
            &alpha as *const u32 as *const c_void,
            b,
            DT_F16_R,
            n as i32,
            a,
            DT_F16_R,
            k as i32,
            &beta as *const u32 as *const c_void,
            d,
            DT_F16_R,
            n as i32,
            d,
            DT_F16_R,
            n as i32,
            DT_F32_R,
            ALGO_STANDARD,
            0,
            0,
        )
    };
    for i in 0..5 {
        let rc = call(i);
        assert_eq!(rc, ROCBLAS_SUCCESS, "rocblas_gemm_ex warmup rc={rc}");
    }
    unsafe { hipDeviceSynchronize() };
    let t = Instant::now();
    for i in 0..iters {
        call(i);
    }
    unsafe {
        hipStreamSynchronize(stream);
        hipDeviceSynchronize();
    }
    let secs = t.elapsed().as_secs_f64();
    let flops = 2.0 * m as f64 * n as f64 * k as f64 * iters as f64;
    flops / secs / 1e9
}

fn main() {
    let mut stream: hipStream_t = std::ptr::null_mut();
    assert_eq!(unsafe { hipStreamCreate(&mut stream) }, 0);
    let mut h: rocblas_handle = std::ptr::null_mut();
    assert_eq!(
        unsafe { rocblas_create_handle(&mut h) },
        ROCBLAS_SUCCESS,
        "create_handle"
    );
    assert_eq!(unsafe { rocblas_set_stream(h, stream) }, ROCBLAS_SUCCESS);

    println!("rocblas_gemm_ex  f16 in/out, f32 accumulate  (gfx1100)");
    let shapes = [
        (512usize, 1024usize, 1024usize, "qkv/o   512x1024x1024"),
        (512, 1024, 3072, "up/gate 512x1024x3072"),
        (512, 3072, 1024, "down    512x3072x1024"),
        (512, 1024, 4096, "wide    512x1024x4096"),
    ];
    for (m, k, n, label) in shapes {
        let g = bench(h, stream, m, k, n, 200);
        println!("  {label}: {:9.1} GFLOP/s  ({:.1} TFLOP/s)", g, g / 1000.0);
    }
    unsafe { rocblas_destroy_handle(h) };
}
