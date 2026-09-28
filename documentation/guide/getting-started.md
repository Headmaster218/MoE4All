---
kind: guide
status: current
scope: windows-release
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# Windows 快速开始

## 准备

- 64 位 Windows 和可用的 Vulkan 显卡驱动。
- 从项目 Release 下载并完整解压 `MoE4All-Windows-x86_64-v*.zip`。
- 准备受支持的 GGUF；分片模型的所有分片必须在同一目录。
- 大型 MoE 需要足够的 SSD，超出显存的部分还会使用 RAM 或按需读取 SSD。

运行发布版不需要安装 Rust、Visual Studio 或 Vulkan SDK。

## 第一次启动

双击：

```text
Start-INFR-Wizard.cmd
```

建议第一次选择：

1. 实时终端对话。
2. 一个已经验证过的小模型或发布说明推荐模型。
3. 自动配置：保守。
4. 上下文和高级选项先保留默认。

向导会显示最终命令。确认 GPU、上下文、RAM/VRAM 预算和模型路径后再启动。

## 三种入口

- `infr run`：终端多轮对话。
- `infr serve`：OpenAI-compatible Chat Completions、Vision 和可选 Embedding。
- `infr bench`：Prefill/Decode 性能测试。

API 默认使用 loopback。对局域网开放时应设置 Bearer API key，不要把无鉴权服务暴露到公网。

## 自动策略

- **保守**：启动时以系统当前可用 RAM 减 3 GiB 作为总进程 RAM 预算；VRAM 按当前可用量保留合计约 1 GiB（含 Vulkan 分配器 256 MiB guard）。离散 GPU 默认 Prefill ubatch 从 2048 行起选。
- **激进**：启动时以总物理 RAM 减 14 GiB 为总进程 RAM 预算；GPU 总显存减 2 GiB 为进程上限，但仍受设备实时剩余量约束。离散 GPU 默认从 4096 行起选。
- **手动**：用于固定实验条件；显式 RAM、VRAM、Ubatch 和 submit 设置优先于自动策略。

RAM 自动预算启动时冻结；显式值优先。ubatch 可按实际放置能力下调；iGPU 有单独的较小默认值，并发 ubatch 未显式指定时继承该次选出的值。这些预算不是最终专家缓存大小，固定分配后的真实显存余量仍会决定 arena。

## 继续阅读

- [配置参考](../reference/configuration.md)
- [API 使用](serving/api-quickstart.md)
- [模型能力矩阵](../reference/model-capabilities.md)
- [系统总览](../architecture/system-overview.md)
- [并发调度](../architecture/runtime/parallel-scheduler.md)
- [运行时资源生命周期](../architecture/memory/runtime-resource-lifecycle.md)
