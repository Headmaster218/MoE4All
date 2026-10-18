//! 最小验证：引擎自己的缓冲（p2p 导出的 dma-buf + BDA）能不能直接交给 rocBLAS 算，
//! 结果对不对、快多少。左边是引擎的 `matmul_native(Bf16)`，右边是 rocBLAS GemmEx，
//! 两边读写的是**同一块显存**（零拷贝）。
//!
//! 跑法: cargo test -p infr-vulkan --test rocm_gemm_xcheck -- --ignored --nocapture
//!
//! 这不是生产路径：没有分配器改动、没有分流开关。它只回答"值不值得做生产集成"。

use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::{P2pHandleType, VulkanBackend};
use std::sync::Arc;

const CLI: &str = "/home/qiankun/.hermes/workspace/dual/hipblaslt-gate/rocm_gemm_cli";

fn fill_f32(v: &mut Vec<u8>, count: usize, seed: u32) {
    let mut s = seed;
    for _ in 0..count {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = ((s >> 8) as f32 / 16777216.0) - 0.5;
        v.extend_from_slice(&f.to_le_bytes());
    }
}

/// 与引擎的 bf16 权重一致：f32 直接截断成 bf16（"bf16-truncated f32, an exact decode"）。
fn fill_bf16(v: &mut Vec<u8>, count: usize, seed: u32) {
    let mut s = seed;
    for _ in 0..count {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = ((s >> 8) as f32 / 16777216.0) - 0.5;
        v.extend_from_slice(&((f.to_bits() >> 16) as u16).to_le_bytes());
    }
}

fn read_f32(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])
}

#[test]
#[ignore = "需要 Vulkan GPU + rocm_gemm_cli 助手"]
fn rocm_gemm_xcheck() {
    let (m, k, n) = (1280usize, 10240usize, 320usize);
    let reps = 20usize;

    let be = VulkanBackend::new_with(Arc::new(infr_core::config::Config::default())).unwrap();
    assert!(
        be.p2p_supported(P2pHandleType::DmaBuf),
        "这台设备不给 dma-buf 导出能力"
    );

    // 一块内存，两边用：引擎当自己的缓冲跑 kernel，rocBLAS 通过 fd 读同一批字节。
    let a = be
        .p2p_export_with(m * k * 4, P2pHandleType::DmaBuf, true)
        .unwrap();
    let w = be
        .p2p_export_with(n * k * 2, P2pHandleType::DmaBuf, true)
        .unwrap();
    let c = be
        .p2p_export_with(m * n * 4, P2pHandleType::DmaBuf, true)
        .unwrap();
    assert!(
        w.buffer().device_addr().is_some(),
        "权重没拿到 64 位设备地址(BDA)，引擎的 GEMM 读不了"
    );

    let mut ha = Vec::with_capacity(m * k * 4);
    fill_f32(&mut ha, m * k, 12345);
    let mut hw = Vec::with_capacity(n * k * 2);
    fill_bf16(&mut hw, n * k, 999);
    be.upload(a.buffer(), &ha).unwrap();
    be.upload(w.buffer(), &hw).unwrap();
    be.sync().unwrap();

    // ── 引擎自己算（一次预热 + 计时循环） ──────────────────────────────────────────────
    let rec = be.recorder().unwrap();
    rec.matmul_native(
        infr_core::DType::Bf16,
        a.buffer(),
        w.buffer(),
        c.buffer(),
        m,
        k,
        n,
    );
    rec.finish().unwrap();

    let t0 = std::time::Instant::now();
    let rec = be.recorder().unwrap();
    for _ in 0..reps {
        rec.matmul_native(
            infr_core::DType::Bf16,
            a.buffer(),
            w.buffer(),
            c.buffer(),
            m,
            k,
            n,
        );
    }
    rec.finish().unwrap();
    let us = t0.elapsed().as_micros() as f64 / reps as f64;
    let gflops = 2.0 * m as f64 * n as f64 * k as f64 / (us * 1e3);
    println!("[engine native_gemm_bf16] {us:8.1} us/GEMM  {gflops:6.1} GFLOP/s");

    let mut cref = vec![0u8; m * n * 4];
    be.download(c.buffer(), &mut cref).unwrap();

    // ── 同一个 C，交给 rocBLAS 再算一遍 ────────────────────────────────────────────────
    // 注意：驱动给导出的 fd 打了 FD_CLOEXEC，不 clear 的话子进程看不到它。
    for e in [&a, &w, &c] {
        let rc = unsafe { libc::fcntl(e.fd(), libc::F_SETFD, 0) };
        assert!(rc != -1, "清 FD_CLOEXEC 失败");
    }
    let out = std::process::Command::new(CLI)
        .args([
            a.fd().to_string(),
            w.fd().to_string(),
            c.fd().to_string(),
            m.to_string(),
            k.to_string(),
            n.to_string(),
            a.mem_size().to_string(),
            w.mem_size().to_string(),
            c.mem_size().to_string(),
            reps.to_string(),
        ])
        .output()
        .expect("跑 rocm_gemm_cli");
    print!("{}", String::from_utf8_lossy(&out.stdout));
    eprint!("{}", String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "rocm_gemm_cli 失败");

    let mut croc = vec![0u8; m * n * 4];
    be.download(c.buffer(), &mut croc).unwrap();

    // ── 对拍 ─────────────────────────────────────────────────────────────────────────
    // 引擎把 A 转 f16、rocBLAS 转 bf16，舍入路径不同：**逐元素相对误差在近零元素上必然爆掉**
    // （|x|→0 时 1e-2 的绝对差就是 100%），所以判据取"绝对差 ≤ 2% 的 C 的均方根量级"。
    let (mut sumsq, mut maxabs) = (0f64, 0f64);
    let (mut worst_i, mut worst_x, mut worst_y) = (0usize, 0f32, 0f32);
    let (mut sum_e, mut sum_r) = (0f64, 0f64);
    for i in 0..m * n {
        let x = read_f32(&cref, i) as f64;
        let y = read_f32(&croc, i) as f64;
        sumsq += x * x;
        sum_e += x;
        sum_r += y;
        let d = (x - y).abs();
        if d > maxabs {
            maxabs = d;
            worst_i = i;
            worst_x = x as f32;
            worst_y = y as f32;
        }
    }
    let rms = (sumsq / (m * n) as f64).sqrt();
    let tol = 0.02 * rms;
    let mut bad = 0usize;
    for i in 0..m * n {
        let d = (read_f32(&cref, i) as f64 - read_f32(&croc, i) as f64).abs();
        if d > tol {
            bad += 1;
        }
    }
    println!("[对拍] rms={rms:.4} maxabs={maxabs:.4e} 容忍(2%rms)={tol:.4} 超差={bad}/{}", m * n);
    println!("       最差 idx={worst_i} 引擎={worst_x:.6} rocBLAS={worst_y:.6}");
    println!("[校验和] 引擎={sum_e:.6e}  rocBLAS={sum_r:.6e}");
    assert!(bad == 0, "两边结果不一致：{bad} 个元素超出 2% rms");
    println!("\n结论: PASS —— 引擎自己的缓冲直接喂 rocBLAS，结果一致、零拷贝");
}
