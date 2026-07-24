//! Slice-31 probe: does HIP-graph replay cut the ~10 µs ROCm per-dispatch launch
//! floor on gfx1100? Self-contained HIP FFI (mirrors `smoke_test`) — throwaway
//! bring-up tool. Launches a representative repeated sequence (100 tiny back-to-back
//! kernels = one "token") N times, four ways, and reports effective per-op µs:
//!
//!   A. individual launches, sync per token   (realistic decode: host reads logits)
//!   B. individual launches, one final sync    (pure CPU enqueue throughput)
//!   C. graph replay,        sync per token    (realistic decode with graph)
//!   D. graph replay,        one final sync     (graph enqueue throughput)
//!
//! Decision: compare A vs C. If C's per-op is materially below A's (< ~5 µs vs
//! ~10 µs), HIP graphs help → integrate. If they're ~equal, the floor is GPU
//! command-processor overhead the graph can't remove → STOP, recommend op-fusion.
//! B vs A is the diagnostic: if B ≪ A the floor is GPU-serialization latency
//! (per-token sync), which a graph cannot fix.
//!
//! ── MEASURED FINDING (gfx1100, ROCm 7.2, 2026-07) — NEGATIVE, do NOT integrate ──
//! effective per-op latency (µs):        grid=1        grid=304
//!   A individual + sync/token          3.073         3.418   <- realistic baseline
//!   B individual + 1 final sync        2.911         3.235   <- CPU enqueue
//!   C graph      + sync/token          2.768         2.905   <- realistic w/ graph
//!   D graph      + 1 final sync        2.606         2.791   <- graph enqueue
//!   A/C per-op speedup                 1.11x         1.18x
//! The individual per-op launch floor is only ~3 µs here (NOT the ~10 µs the real
//! decode profile attributes to "per-dispatch"), and B ≈ A (enqueue ≈ sync-per-
//! token) => the launch path is a small, CPU-bound slice, not GPU-serialization.
//! HIP-graph replay shaves only ~0.3–0.6 µs/op (12–18%). At 479 launches/token
//! that is ~0.25–0.30 ms saved on a ~9.7 ms token ≈ 2.5–3.1% ceiling — and that
//! ceiling assumes real kernels cost no more than this trivial one (they cost
//! more, shrinking the graph fraction further). The real ~10 µs/dispatch decode
//! floor is per-kernel GPU work + memory/dependency latency, which HIP graphs do
//! NOT touch (graphs amortize CPU-side launch enqueue, already cheap on ROCm).
//! => STOP. The launch floor is not the lever; op-fusion (fewer, bigger kernels
//!    that cut the ~479 real kernel launches + their memory round-trips) is.
//!
//! Build: `cargo build --release --features rocm -p infr-rocm --example graph_probe`
//! Run:   `LD_LIBRARY_PATH=/opt/rocm/lib ./target/release/examples/graph_probe`

// Throwaway probe with its own hand-rolled HIP FFI: keep the C type names and the
// full binding surface; don't style-lint the ad-hoc launch code.
#![allow(
    non_camel_case_types,
    dead_code,
    clippy::needless_range_loop,
    clippy::manual_div_ceil
)]

use std::ffi::{c_char, c_int, c_void, CString};
use std::time::Instant;

type hipModule_t = *mut c_void;
type hipFunction_t = *mut c_void;
type hiprtcProgram = *mut c_void;
type hipStream_t = *mut c_void;
type hipGraph_t = *mut c_void;
type hipGraphExec_t = *mut c_void;
type hipGraphNode_t = *mut c_void;

#[link(name = "amdhip64")]
extern "C" {
    fn hipSetDevice(device: c_int) -> c_int;
    fn hipGetDeviceCount(count: *mut c_int) -> c_int;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn hipFree(ptr: *mut c_void) -> c_int;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn hipStreamCreate(stream: *mut hipStream_t) -> c_int;
    fn hipStreamSynchronize(stream: hipStream_t) -> c_int;
    fn hipStreamDestroy(stream: hipStream_t) -> c_int;
    fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> c_int;
    fn hipModuleGetFunction(
        func: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> c_int;
    fn hipModuleLaunchKernel(
        f: hipFunction_t,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shm: u32,
        stream: hipStream_t,
        kp: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> c_int;
    // ── HIP graph API ──
    fn hipStreamBeginCapture(stream: hipStream_t, mode: c_int) -> c_int;
    fn hipStreamEndCapture(stream: hipStream_t, graph: *mut hipGraph_t) -> c_int;
    fn hipGraphInstantiate(
        exec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        err_node: *mut hipGraphNode_t,
        log: *mut c_char,
        log_size: usize,
    ) -> c_int;
    fn hipGraphLaunch(exec: hipGraphExec_t, stream: hipStream_t) -> c_int;
    fn hipGraphExecDestroy(exec: hipGraphExec_t) -> c_int;
    fn hipGraphDestroy(graph: hipGraph_t) -> c_int;
}
#[link(name = "hiprtc")]
extern "C" {
    fn hiprtcCreateProgram(
        p: *mut hiprtcProgram,
        src: *const c_char,
        name: *const c_char,
        nh: c_int,
        hdrs: *const *const c_char,
        incl: *const *const c_char,
    ) -> c_int;
    fn hiprtcCompileProgram(p: hiprtcProgram, no: c_int, opts: *const *const c_char) -> c_int;
    fn hiprtcGetCodeSize(p: hiprtcProgram, s: *mut usize) -> c_int;
    fn hiprtcGetCode(p: hiprtcProgram, code: *mut c_char) -> c_int;
    fn hiprtcDestroyProgram(p: *mut hiprtcProgram) -> c_int;
}

const HIP_SUCCESS: c_int = 0;
const H2D: c_int = 1;
// hipStreamCaptureModeGlobal
const CAPTURE_GLOBAL: c_int = 0;

fn check(rc: c_int, msg: &str) {
    if rc != HIP_SUCCESS {
        panic!("{msg}: rc={rc}");
    }
}

// One "token" = this many back-to-back kernel launches.
const OPS_PER_TOKEN: usize = 100;
// Number of tokens to time.
const TOKENS: usize = 300;
// Warmup tokens (compile/JIT/first-touch out of the timed path).
const WARMUP: usize = 30;

fn main() {
    let mut count: c_int = 0;
    check(
        unsafe { hipGetDeviceCount(&mut count) },
        "hipGetDeviceCount",
    );
    assert!(count > 0, "no HIP devices");
    check(unsafe { hipSetDevice(0) }, "hipSetDevice");

    // Non-default stream (capture is illegal on the null stream).
    let mut stream: hipStream_t = std::ptr::null_mut();
    check(unsafe { hipStreamCreate(&mut stream) }, "hipStreamCreate");

    // Trivial elementwise kernel — grid 1 / block 64, minimal compute, so the
    // per-dispatch launch/overhead floor dominates over actual GPU work.
    let kernel_src = r#"
extern "C" __global__ void bump(float* data, int n) {
    int i = threadIdx.x;
    if (i < n) data[i] += 1.0f;
}
"#;
    let csrc = CString::new(kernel_src).unwrap();
    let cname = CString::new("probe").unwrap();
    let mut prog: hiprtcProgram = std::ptr::null_mut();
    check(
        unsafe {
            hiprtcCreateProgram(
                &mut prog,
                csrc.as_ptr(),
                cname.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        },
        "hiprtcCreateProgram",
    );
    let std_flag = CString::new("-std=c++17").unwrap();
    let opts: [*const c_char; 1] = [std_flag.as_ptr()];
    check(
        unsafe { hiprtcCompileProgram(prog, 1, opts.as_ptr()) },
        "hiprtcCompileProgram",
    );
    let mut code_size: usize = 0;
    check(
        unsafe { hiprtcGetCodeSize(prog, &mut code_size) },
        "hiprtcGetCodeSize",
    );
    let mut code: Vec<u8> = vec![0; code_size];
    check(
        unsafe { hiprtcGetCode(prog, code.as_mut_ptr() as *mut c_char) },
        "hiprtcGetCode",
    );
    unsafe { hiprtcDestroyProgram(&mut prog) };

    let mut module: hipModule_t = std::ptr::null_mut();
    check(
        unsafe { hipModuleLoadData(&mut module, code.as_ptr() as *const c_void) },
        "hipModuleLoadData",
    );
    let cfn = CString::new("bump").unwrap();
    let mut func: hipFunction_t = std::ptr::null_mut();
    check(
        unsafe { hipModuleGetFunction(&mut func, module, cfn.as_ptr()) },
        "hipModuleGetFunction",
    );

    // Big enough that a realistic grid (blocks across all CUs) has data to touch.
    let n: i32 = 304 * 256;
    let mut dptr: *mut c_void = std::ptr::null_mut();
    check(
        unsafe { hipMalloc(&mut dptr, (n as usize) * 4) },
        "hipMalloc",
    );
    let host: Vec<f32> = vec![0.0; n as usize];
    check(
        unsafe { hipMemcpy(dptr, host.as_ptr() as *const c_void, host.len() * 4, H2D) },
        "hipMemcpy H2D",
    );

    let n_arg = n;
    let mut args: [*mut c_void; 2] = [
        &mut dptr as *mut _ as *mut c_void,
        &n_arg as *const i32 as *mut c_void,
    ];

    let launch = |grid: u32, args: &mut [*mut c_void; 2]| {
        check(
            unsafe {
                hipModuleLaunchKernel(
                    func,
                    grid,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "hipModuleLaunchKernel",
        );
    };

    // Sweep representative grids: 1 block (trivial, pure launch floor), and 304
    // blocks (one wave across all gfx1100 CUs, closer to a real decode kernel).
    for &grid in &[1u32, 304u32] {
        // ── warmup (JIT / first-touch / allocator) ──
        for _ in 0..WARMUP {
            for _ in 0..OPS_PER_TOKEN {
                launch(grid, &mut args);
            }
            check(unsafe { hipStreamSynchronize(stream) }, "warmup sync");
        }

        let total_ops = (TOKENS * OPS_PER_TOKEN) as f64;

        // ── A. individual launches, sync per token (realistic decode) ──
        let t = Instant::now();
        for _ in 0..TOKENS {
            for _ in 0..OPS_PER_TOKEN {
                launch(grid, &mut args);
            }
            check(unsafe { hipStreamSynchronize(stream) }, "A sync");
        }
        let a_us = t.elapsed().as_secs_f64() * 1e6 / total_ops;

        // ── B. individual launches, one final sync (pure enqueue throughput) ──
        let t = Instant::now();
        for _ in 0..TOKENS {
            for _ in 0..OPS_PER_TOKEN {
                launch(grid, &mut args);
            }
        }
        check(unsafe { hipStreamSynchronize(stream) }, "B sync");
        let b_us = t.elapsed().as_secs_f64() * 1e6 / total_ops;

        // ── capture one "token" (OPS_PER_TOKEN launches) into a graph, once ──
        check(
            unsafe { hipStreamBeginCapture(stream, CAPTURE_GLOBAL) },
            "hipStreamBeginCapture",
        );
        for _ in 0..OPS_PER_TOKEN {
            launch(grid, &mut args);
        }
        let mut graph: hipGraph_t = std::ptr::null_mut();
        check(
            unsafe { hipStreamEndCapture(stream, &mut graph) },
            "hipStreamEndCapture",
        );
        let mut exec: hipGraphExec_t = std::ptr::null_mut();
        check(
            unsafe {
                hipGraphInstantiate(
                    &mut exec,
                    graph,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                )
            },
            "hipGraphInstantiate",
        );

        // graph warmup
        for _ in 0..WARMUP {
            check(
                unsafe { hipGraphLaunch(exec, stream) },
                "graph warmup launch",
            );
            check(unsafe { hipStreamSynchronize(stream) }, "graph warmup sync");
        }

        // ── C. graph replay, sync per token (realistic decode with graph) ──
        let t = Instant::now();
        for _ in 0..TOKENS {
            check(unsafe { hipGraphLaunch(exec, stream) }, "C launch");
            check(unsafe { hipStreamSynchronize(stream) }, "C sync");
        }
        let c_us = t.elapsed().as_secs_f64() * 1e6 / total_ops;

        // ── D. graph replay, one final sync (graph enqueue throughput) ──
        let t = Instant::now();
        for _ in 0..TOKENS {
            check(unsafe { hipGraphLaunch(exec, stream) }, "D launch");
        }
        check(unsafe { hipStreamSynchronize(stream) }, "D sync");
        let d_us = t.elapsed().as_secs_f64() * 1e6 / total_ops;

        println!("\n=== Slice-31 HIP-graph probe (gfx1100), grid={grid} blocks x256 ===");
        println!("config: {OPS_PER_TOKEN} kernels/token, {TOKENS} tokens, {WARMUP} warmup");
        println!("effective per-op latency (µs):");
        println!("  A individual + sync/token   : {a_us:8.3}  <- realistic decode baseline");
        println!("  B individual + 1 final sync : {b_us:8.3}  <- CPU enqueue throughput");
        println!("  C graph      + sync/token   : {c_us:8.3}  <- realistic decode w/ graph");
        println!("  D graph      + 1 final sync : {d_us:8.3}  <- graph enqueue throughput");
        let speedup = a_us / c_us;
        println!(
            "  per-token wall  A: {:8.2} µs   C: {:8.2} µs   A/C speedup: {speedup:.2}x",
            a_us * OPS_PER_TOKEN as f64,
            c_us * OPS_PER_TOKEN as f64
        );
        if c_us < 5.0 && speedup > 1.3 {
            println!("  VERDICT: graph cuts the per-op floor -> INTEGRATE");
        } else {
            println!(
                "  VERDICT: graph does NOT materially cut the per-op floor -> STOP (op-fusion)"
            );
        }

        unsafe {
            hipGraphExecDestroy(exec);
            hipGraphDestroy(graph);
        }
    }

    unsafe {
        hipFree(dptr);
        hipStreamDestroy(stream);
    }
}
