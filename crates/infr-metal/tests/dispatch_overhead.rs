//! Probe: per-kernel overhead inside one command buffer, serialized (hazard-tracked chain on one
//! buffer) vs independent (round-robin over many buffers). Temporary evidence for the op-fusion
//! work — run with `cargo test -p infr-metal --test dispatch_overhead -- --ignored --nocapture`.
#![cfg(target_os = "macos")]

use metal::{Device, MTLResourceOptions, MTLSize};

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void tick(device float* buf [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0) buf[0] += 1.0f;
}
"#;

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn dispatch_overhead() {
    let dev = Device::system_default().unwrap();
    let queue = dev.new_command_queue();
    let lib = dev
        .new_library_with_source(MSL, &metal::CompileOptions::new())
        .unwrap();
    let f = lib.get_function("tick", None).unwrap();
    let pso = dev.new_compute_pipeline_state_with_function(&f).unwrap();

    let n = 10_000usize;
    let bufs: Vec<_> = (0..64)
        .map(|_| dev.new_buffer(4096, MTLResourceOptions::StorageModeShared))
        .collect();

    for (label, nbuf) in [
        ("serialized (1 buf)", 1usize),
        ("independent (64 bufs)", 64),
    ] {
        // warmup + 3 reps
        for rep in 0..4 {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            for i in 0..n {
                enc.set_buffer(0, Some(&bufs[i % nbuf]), 0);
                enc.dispatch_threads(MTLSize::new(32, 1, 1), MTLSize::new(32, 1, 1));
            }
            enc.end_encoding();
            let t0 = std::time::Instant::now();
            cb.commit();
            cb.wait_until_completed();
            let dt = t0.elapsed();
            if rep > 0 {
                println!(
                    "{label}: {n} kernels in {:.2} ms -> {:.2} us/kernel",
                    dt.as_secs_f64() * 1e3,
                    dt.as_secs_f64() * 1e6 / n as f64
                );
            }
        }
    }
}

// Serial vs concurrent dispatch types on a DEPENDENT chain: with MTLDispatchTypeSerial every
// dispatch implicitly orders after the previous one; with Concurrent, ordering exists only at
// explicit memory barriers, so runs of independent dispatches (decode's q/k/v GEMVs, the two
// KV writes) can overlap. This measures what barrier GRANULARITY is worth on a decode-shaped
// dispatch count.
#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn dispatch_concurrency() {
    use metal::MTLDispatchType;
    let dev = Device::system_default().unwrap();
    let queue = dev.new_command_queue();
    let lib = dev
        .new_library_with_source(MSL, &metal::CompileOptions::new())
        .unwrap();
    let f = lib.get_function("tick", None).unwrap();
    let pso = dev.new_compute_pipeline_state_with_function(&f).unwrap();

    let n = 4096usize; // ~decode-token dispatch count x 10 for signal
    let bufs: Vec<_> = (0..8)
        .map(|_| dev.new_buffer(4096, MTLResourceOptions::StorageModeShared))
        .collect();

    // group = how many round-robin dispatches run between barriers (1 = fully serialized).
    for (label, group) in [
        ("serial encoder", 0usize),
        ("concurrent, barrier each", 1),
        ("concurrent, barrier per 4", 4),
        ("concurrent, no barriers", n),
    ] {
        for rep in 0..4 {
            let cb = queue.new_command_buffer();
            let enc = if group == 0 {
                cb.new_compute_command_encoder().to_owned()
            } else {
                cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent)
                    .to_owned()
            };
            enc.set_compute_pipeline_state(&pso);
            for i in 0..n {
                enc.set_buffer(0, Some(&bufs[i % 8]), 0);
                enc.dispatch_threads(MTLSize::new(32, 1, 1), MTLSize::new(32, 1, 1));
                if group > 0 && group < n && (i + 1) % group == 0 {
                    let refs: Vec<&metal::ResourceRef> = bufs
                        .iter()
                        .map(|b| b.as_ref() as &metal::ResourceRef)
                        .collect();
                    enc.memory_barrier_with_resources(&refs);
                }
            }
            enc.end_encoding();
            let t0 = std::time::Instant::now();
            cb.commit();
            cb.wait_until_completed();
            let dt = t0.elapsed();
            if rep > 0 {
                println!(
                    "{label}: {n} kernels in {:.2} ms -> {:.2} us/kernel",
                    dt.as_secs_f64() * 1e3,
                    dt.as_secs_f64() * 1e6 / n as f64
                );
            }
        }
    }
}

const MSL_BW: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void sumr(device const uint4* src [[buffer(0)]],
                 device float*       out [[buffer(1)]],
                 constant uint&      n4  [[buffer(2)]],
                 uint gid  [[thread_position_in_grid]],
                 uint gsz  [[threads_per_grid]]) {
    uint4 acc = uint4(0);
    for (uint i = gid; i < n4; i += gsz) acc += src[i];
    uint s = acc.x + acc.y + acc.z + acc.w;
    if (s == 0xFFFFFFFFu) out[gid & 1023u] = 1.0f;   // keep the loads alive
}
"#;

#[test]
#[ignore = "requires a Metal GPU; evidence probe, not a correctness test"]
fn read_bandwidth_ceiling() {
    let dev = Device::system_default().unwrap();
    let queue = dev.new_command_queue();
    let lib = dev
        .new_library_with_source(MSL_BW, &metal::CompileOptions::new())
        .unwrap();
    let f = lib.get_function("sumr", None).unwrap();
    let pso = dev.new_compute_pipeline_state_with_function(&f).unwrap();

    let mib = 512usize;
    let bytes = mib * 1024 * 1024;
    let src = dev.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    let out = dev.new_buffer(4096, MTLResourceOptions::StorageModeShared);
    let n4 = (bytes / 16) as u32;

    for threads in [32768u64, 131072, 524288] {
        for rep in 0..3 {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&src), 0);
            enc.set_buffer(1, Some(&out), 0);
            enc.set_bytes(2, 4, &n4 as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            let t0 = std::time::Instant::now();
            cb.commit();
            cb.wait_until_completed();
            let dt = t0.elapsed().as_secs_f64();
            if rep > 0 {
                println!(
                    "read ceiling ({threads} threads): {mib} MiB in {:.2} ms -> {:.1} GiB/s",
                    dt * 1e3,
                    bytes as f64 / dt / (1u64 << 30) as f64
                );
            }
        }
    }
}
