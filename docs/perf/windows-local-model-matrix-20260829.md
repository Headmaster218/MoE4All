# Windows 本地大模型代表性性能记录

[返回主 README](../../README.md)

本文汇总 MoE4All 在原生 Windows 11、AMD Radeon RX 7900 XTX 上已经完成的历史稳定
测试。2026-08-29 的资源压力矩阵曾把进程 RAM 设得过于接近物理内存上限，Windows
进入系统分页，得到的低速结果不代表正常性能，因此不再保留在产品 benchmark 表中。

## 测试平台

| 项目 | 配置 |
|---|---|
| 操作系统 | 原生 Windows 11 |
| GPU | AMD Radeon RX 7900 XTX 24 GiB，Vulkan0 |
| CPU | AMD Ryzen 5 5600X，6 核 12 线程 |
| 内存 | 64 GiB DDR4 |
| 存储 | 本地 NVMe SSD；不同模型位于 D: 或 G: |

## 数据口径

- 表中只保留未观察到系统分页、运行稳定且测试条件有记录的历史成绩。
- 不同模型来自不同优化阶段，并非同一天完成的统一矩阵，不能用小于约 5% 的差异做
  跨模型或跨提交结论。
- `synthetic depth` 会构造真实 KV allocation、访问长度和 allocator 状态，但 KV 内容
  没有语言意义，适合测试长上下文执行成本。
- Qwen3.8 表为三次平均；Qwen3.6 长 decode 使用 1,000 token，减少冷专家预热对结果
  的影响。

## Qwen3.6-35B-A3B

模型：APEX-I-Balanced，Q8 K/V。该组来自统一显存验收，使用 8 GiB 专家缓存。

| Context depth | Decode | Prefill | 负载 |
|---:|---:|---:|---|
| 0 | **84.4 tok/s** | **2,855.6 tok/s** | decode 1,000；prefill 4,096 |
| 250K synthetic | **41.2 tok/s** | **477.9 tok/s** | decode 1,000；prefill 4,096 |

完整前后对照见[统一显存验收记录](../unified-vram-acceptance-20260822.md)。

## Qwen3.5-122B-A10B

模型：APEX-I-Quality。历史 depth 0 decode 在 F16 K/V、45 GiB bounded RAM、三次重复下
达到 **23.2 tok/s**。后续资源压力测试发生系统分页，其低速数据不再引用。

## Ling-3.0-Flash

模型：Q5_K_M，F16 MLA KV。在专家缓存完成预热、系统未分页的 depth 0 decode 验收中，
该模型达到约 **36 tok/s**。这项成绩用于证明 83 GiB 级模型可在 24 GiB 显存设备上
正常运行；当时未形成可复现的完整长上下文矩阵，因此不补写缺失行。

## Qwen3.8-Flash-Next Q2_K_XL

条件：Q8 K/V、40 GiB bounded RAM、`tg128`、`pp1024`、ubatch 1024，三次平均。

| Context depth | Decode | Prefill |
|---:|---:|---:|
| 0 | **29.45 tok/s** | **155.16 tok/s** |
| 128K synthetic | **26.23 tok/s** | **170.44 tok/s** |
| 250K synthetic | **22.82 tok/s** | **152.27 tok/s** |

## Qwen3.8-Flash-Next IQ4_XS

条件：Q8 K/V、40 GiB bounded RAM、`tg128`、`pp1024`、ubatch 1024，三次平均。

| Context depth | Decode | Prefill |
|---:|---:|---:|
| 0 | **16.85 tok/s** | **244.68 tok/s** |
| 128K synthetic | **14.15 tok/s** | **250.55 tok/s** |
| 250K synthetic | **15.26 tok/s** | **239.00 tok/s** |

Qwen3.8 的 Q2_K_XL 与 IQ4_XS 均通过三轮真实 API 对话，能够保持校验码、完成跨轮
算术并总结先前内容。这里记录的是历史稳定能力，不是对所有 Radeon GPU 的性能承诺。
