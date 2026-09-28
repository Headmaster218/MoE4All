---
kind: reference
status: current
scope: quantized-linear-kernels
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 跨后端快速内核覆盖情况

本页汇总**各后端中哪些权重量化格式拥有原生快速线性内核**。“原生”指专用的着色器内 / int8 点积解码路径，而不是 `dequant → f16/f32` 回退路径（将整个块解码为浮点数，再运行通用浮点 GEMV/GEMM）。回退路径是正确的，但会传输更多字节，并在热点路径上浪费该格式的压缩优势。

`DType` 在 `crates/infr-vulkan/src/linear.rs` 中以规范形式枚举（`ALL_DTYPES`，通过穷尽匹配防止偏差）。覆盖测试包括：`infr-cpu` 中的 CPU 测试（SIMD↔标量位一致性，以及与 `dequant_block` 的容差一致性）、Vulkan 算子一致性测试套件，以及 Metal 的 `crates/infr-metal/tests/parity.rs`（在 CI 的 `test-macos` 作业中以真实 Metal 硬件进行数值验证）。

## 覆盖情况 — 全部 24 种权重量化格式在三个后端均为原生支持

| 格式    | CPU | Vulkan | Metal | 类型                       |
| --------- | :-: | :----: | :---: | -------------------------- |
| `Q4_0`    | ✅  |   ✅   |  ✅   | 仿射 4 位               |
| `Q4_1`    | ✅  |   ✅   |  ✅   | 仿射 4 位 + 最小值         |
| `Q5_0`    | ✅  |   ✅   |  ✅   | 仿射 5 位               |
| `Q5_1`    | ✅  |   ✅   |  ✅   | 仿射 5 位 + 最小值         |
| `Q8_0`    | ✅  |   ✅   |  ✅   | 8 位，32 元素块            |
| `Q2_K`    | ✅  |   ✅   |  ✅   | K 量化 2 位              |
| `Q3_K`    | ✅  |   ✅   |  ✅   | K 量化 3 位              |
| `Q4_K`    | ✅  |   ✅   |  ✅   | K 量化 4 位              |
| `Q5_K`    | ✅  |   ✅   |  ✅   | K 量化 5 位              |
| `Q6_K`    | ✅  |   ✅   |  ✅   | K 量化 6 位              |
| `IQ4_NL`  | ✅  |   ✅   |  ✅   | 非线性码本，平坦布局  |
| `IQ4_XS`  | ✅  |   ✅   |  ✅   | 非线性，超块    |
| `IQ2_XXS` | ✅  |   ✅   |  ✅   | 网格，KSIGNS 查找        |
| `IQ2_XS`  | ✅  |   ✅   |  ✅   | 网格，9 位索引          |
| `IQ2_S`   | ✅  |   ✅   |  ✅   | 网格码本              |
| `IQ3_XXS` | ✅  |   ✅   |  ✅   | 网格                       |
| `IQ3_S`   | ✅  |   ✅   |  ✅   | 网格，每 32 个元素一个缩放值         |
| `IQ1_S`   | ✅  |   ✅   |  ✅   | 1 位网格 + 增量         |
| `IQ1_M`   | ✅  |   ✅   |  ✅   | 1 位，d-in-scales         |
| `TQ1_0`   | ✅  |   ✅   |  ✅   | 三元，base-3 打包     |
| `TQ2_0`   | ✅  |   ✅   |  ✅   | 三元，2 位             |
| `Q2_0`    | ✅  |   ✅   |  ✅   | Bonsai 三元，64 元素块   |
| `MXFP4`   | ✅  |   ✅   |  ✅   | fp4 + E8M0 缩放值（gpt-oss） |
| `NVFP4`   | ✅  |   ✅   |  ✅   | fp4 + 每 16 个元素一个 UE4M3 缩放值   |

**每个后端均原生支持 24/24 种格式，所有后端中均无权重量化回退至 dequant→float。**浮点格式（`F16` / `F32` / `Bf16`）也在所有位置获得原生支持。

## 非权重线性内核（已正确排除）

- **`I2S`**（BitNet `i2_s`）— 在运行器的 `wload` 中由主机转换为 `f16`，因此绝不会以 `I2S` 形式到达后端；按设计没有原生内核（所有 Vulkan `*_kernel_name`/spv 门控对此返回 `None`）。
- **`Turbo2` / `Turbo3` / `Turbo4`** — KV 缓存量化格式（TurboQuant），不是权重格式；它们不参与线性内核。

## 各后端解码策略

- **CPU**（`infr-cpu`）— int8 量化激活点积：将激活行量化为一次 int8，然后针对原生权重码执行逐块整数点积（标量 → AVX2 → AVX-512BW → AVX-512-VNNI `dpbusd`），最多使用 8 行的缓存分块。网格/码本格式将网格行一次展开为有符号 i8，随后复用每子块缩放值 × 整数点积。三元格式将 `(digit−1)` 折叠为有符号 i8 + 单缩放值整数点积。各格式的合入历史和实测加速见 `cpu.md`。
- **Vulkan**（`infr-vulkan`）— 两个家族：用于预填充（宽 `m`）的 `dqblk` 解码 f16 coopmat GEMM，以及用于解码（`m=1`）的 dp4a `mmq` 整数 GEMV。int8 `mmq` 路径（每个线程拥有自己的累加器 → 可无额外代价地后置缩放）是有原则的整数路线；fp8/int8/bf16 _coopmat_ 操作数替换为何经测量后被拒绝、转而采用 f16 coopmat GEMM，见 `playbook.md`。
- **Metal**（`infr-metal`）— `DEC16_<DT>` 解码宏为每个 16 元素块索引将连续 16 个权重元素写入 `wk[16]`，并在 GEMV / 行分块 / coopmat-GEMM 内核家族中实例化。使用逐字节解码（对奇数块步长的对齐安全），而不是打包的 `ushort` 加载。见 `documentation/architecture/backends/metal.md`。

## 与性能文档的关系

本文档跟踪**覆盖情况**（原生内核是否存在）。关于**吞吐量**工作——相对 llama.cpp 的比值、优化方法和瓶颈分类——请参阅 `playbook.md`（GPU / 通用）和 `cpu.md`（CPU 后端）。
