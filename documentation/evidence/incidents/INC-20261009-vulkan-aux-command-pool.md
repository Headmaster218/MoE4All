---
kind: incident
status: historical
scope: vulkan-command-pool
recorded: 2026-10-11
source_checkpoint: release-0.10.0
---

# 辅助请求与 Vulkan command pool 生命周期冲突

## 症状与证据

发布前普通 Decode+Vision/Embedding overlap 在修复前版本出现退出码3221225477（0xC0000005）。局部速度已输出，但组测试没有完成，不能算 PASS。native crash 现场涉及 DecodeRecorder 创建/同步 stream 与图片 position upload 的 Backend one-shot upload，两个线程进入 AMD command buffer 路径。

## 不变量与根因

同一个 Vulkan command pool 的 host 访问不仅 allocation/free，recording 生命周期也不能由多个线程无保护交错。只给 allocation 加锁不足；同时辅助 workspace 租约必须覆盖 GPU 消费阶段，不能在异步工作结束前归还给 arena。

`53afa86f3` 保留辅助 workspace，并让一次性 state transfer/staging recorder 使用共享、可重入 execution gate。已持 gate 的同线程嵌套 copy 可重入，其他线程在主 Recorder 持有期间等待。这样约束的是 shared pool 的合法生命周期，不是靠扩大 VRAM 掩盖问题。

## 验证与限制

历史报告记录225项 unit和6项GPU测试；GPU回归覆盖主 Recorder 持 gate、第二线程 unmapped upload 等待、同线程 nested copy 与数据一致。修复候选的辅助 overlap 完成 PASS：文本42.8、Embedding overlap41.6、Vision overlap29.1、combined30.7 tok/s。单组完成不代表任意辅助模型或无限期运行已验证。

完整失败/成功对照见[发布矩阵](../benchmarks/2026-10-09-release-0.10.0-matrix.md)。本轮只核对源码和历史报告，没有使用 GPU。
