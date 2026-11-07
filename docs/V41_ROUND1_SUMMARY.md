# Round 1 小结（2026-11-08 01:03）

目标仍未达成：decode 6.5 t/s vs poorseek 27.7，prefill 6-7 vs 1063。差距在。

## 本轮做的两件事，都是否定结论

### 1. 批量预填充：能开吗？→ **不能，会丢设备**

上游 `batched_prefill_ok` 里 `!is_dsv4_family(c)` 关着。我一度判断它保守（批量走整图，不触发部分 span 的 assert），
用临时环境变量把两道门都打开实测：

| prompt | 模式 | 结果 |
|---|---|---|
| 41 tok | 逐 token | 7 t/s |
| 41 tok | 批量 | **8 t/s** |
| 2032 tok | 逐 token | 6 t/s ✅ |
| 2032 tok | **批量** | ❌ **`The logical device has been lost`** |

批量跑一个 3072 行 chunk、约 5 分钟后设备丢失。**驱动没 wedge**（无 `xe ... wedged`，Vulkan1 仍可枚举，
逐 token 路径立刻恢复到 6.4 t/s）。所以是批量 V4.1 图本身的可复现挂起。

上游注释那句「no batch > 1 V4 graph has ever been EXECUTED」**是在陈述事实**，不是保守。
开关已撤回，注释改为记录实测。

**顺带排除一个假设**：预填充一次一个 token **不是**与 poorseek 差距 150x 的原因 —— 41-token prompt 上批量只买到 7→8 t/s。

### 2. 专家 MMQ 逐 tensor 化：能提速吗？→ **在这个文件上几乎为零**

有人建议把 `moe_batched_ok` 改成逐 tensor 判定。我先数了 dtype：

| 位置 | IQ1_M | IQ2_XXS | IQ2_XS |
|---|---:|---:|---:|
| gate/up | 16 | **17** | 7 |
| down | **28** | 8 | 4 |

而 `MOE_MMQ_DTYPES` 里 **`Iq1M` 和 `Iq2Xxs` 都不在**（只有 `Iq2S`/`Iq2Xs`），
`shaders/` 也**没有 `iq2_xxs` / `iq1_m` 的 MMQ kernel**。

→ `down` **40/40 层全部不合格**，只有 7 层的 gate/up 能走 MMQ。
**这类模型的上限卡在 down 的 dtype，不在判据形状上。** 判据保持原样，注释里写清为什么别再改。

## 真正要做的下一步（已定位，未做）

**写一个 IQ1_M 的 MMQ kernel。** 这是唯一能动的量级 —— 因为 28/40 层的 down 是 IQ1_M。

难点（已查清）：IQ1_M 把子块内幅度的恢复编码成 ±0.125，符号存在独立的符号表里，
所以 `Σ(±0.125)·x` **不能**塌缩成 `scale · sact` 一个标量。
需要每个 8 元素子块维护**两路**激活和（按符号位分开），无法复用 `MOE_MMQ_SACT_DTYPES` 的单标量方案。

这是独立的、有实质工作量的项目，不是一个参数改动。

## 状态

- PR #183：**21 commits**，远端 tree 与本地逐字节一致
- 测试：`synthetic_deepseek2` **38 passed / 0 failed**，`infr-gguf` lib **65 passed**
- 工作树干净
- 真实模型：`"What is the capital of France?"` → `"We need answer. Need identify capital of France. Answer: Paris."`
  `prefill 38 tok @ 6 t/s | decode 24 tok @ 6.5 t/s`
