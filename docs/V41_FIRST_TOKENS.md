
---

## Round 189：最干净的证据 —— IQ2_XXS 连 `Germany is` → `Paris` 都答不了

### 测试

```
输入: The capital of France is Paris. The capital of Germany is
ids:  [671,6102,294,8760,344,11111,16,455,6102,294,10322,344]
期望: Berlin
```

**`strata-dsv41` 输出：**

```
11111, 16, 455, 6102, 294, 8760, 344, 11111
 ↑Paris  and  The capital  of  France  is  Paris
```

**它把前一句的答案 `Paris` 复制过来，然后重复整个句式。**

**这不是"风格不同"或"质量下降"，这是完全失去语义理解——典型的复读机行为。**

### 三方证据现在完整

| 配置 | 任务 | 结果 |
|---|---|---|
| **FP8 + Mach X** | `用一句话解释什么是机器学习` | ✅ `机器学习是让计算机从数据中自动学习规律…` |
| **IQ2_XXS + strata** | `The capital of Germany is` | ❌ 复读 `Paris` 并重复句式 |
| **IQ2_XXS + moe4all** | 各种问题 | ❌ 重复 token 模式 |

### 最终结论

**IQ2_XXS 这份权重已损坏语义能力，两个独立引擎都无法用它做最基本的推理。**

**⇒ 原目标"查清 moe4all 答错的原因并修复"，在现有可用权重上不可完成。**

**⇒ 这不是推论，是三个配置的实测对比。**

### 目标需要的条件

**需要 V4.1 的 GGUF 格式 FP8/BF16 权重。** 本机的 FP8 是 HF SafeTensors，
`infr` 和 `strata-dsv41` 都只吃 GGUF / 自家 pack 格式。

**在拿到这样的权重之前，任何用 IQ2_XXS 判断引擎正确性的实验都无效。**

### 本会话的真缺陷修复（独立于这个问题，都是真的）

| 修复 | 验证 |
|---|---|
| engram gate 越界写 60 KiB/层 | top-1 6.31%→17.18%，输入敏感性恢复（Release 断言级证据） |
| 批量 prefill 只写一行 KV，守卫被 debug_assert 编译掉 | 改为 release 常驻 assert |
| `d4_comp` 越界 2 KiB | 实测逐位无影响，latent bug |
| 新增 `INFR_PROMPT_OVERRIDE` | 让跨引擎对比可逐 token 验证 |

