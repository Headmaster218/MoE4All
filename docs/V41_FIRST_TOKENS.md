

---

## Round 117：用官方检查脚本取基线 —— engram 的 A/B 已固化进脚本

### 发现现成的验证工具

`scripts/check-v41-answers.sh` 早就存在，而且它的设计思路与我 Round 116 的结论**完全一致**：

> *"The point of this script is the control model. Reading V4.1's output and deciding whether it
> 'looks wrong' is not a test — it depends on the reader knowing the answer, and it gives no way to
> tell a broken model path from a broken engine."*

它还带 `flock` 保护（GPU 是单例资源），符合"一次一个 GPU 作业"的硬规则。

### 基线数据（本机无对照模型，只测 V4.1）

| engram | 模式 | 输出 |
|---|---|---|
| **有** | default | `** 0.0.0` |
| **无** | default | `We need answer. The question is "` |
| 有 | batched | `We need answer. We need to reason` |
| 无 | batched | `The user wants me to translate the analysis` |

### 这批数据的意义

**① engram 的 A/B 差异是稳定可复现的**
有 engram 出 `** 0.0.0`（含数字），无 engram 出纯散文。**两次独立运行都是这个模式。**

**② 两条路径都答不对**，但错的**方式**完全不同：
- default：`** 0.0.0` —— 出了 markdown 和数字
- batched：`We need answer...` —— 纯英文思考流

**③ batched 路径的注释早就说明了**：

> *"runner.rs documents that no batch>1 V4 graph has executed correctly, so a batched PASS
> would be surprising and a batched FAIL alone says nothing about the default path."*

### 脚本改进

把 engram A/B 固化进 `check-v41-answers.sh`——设置了 `INFR_ENGRAM_TABLE` 时，
自动跑 `engram-with` / `engram-without` 两组，**任何一侧退化都能看到**。

### 现状判断

| 问题 | 状态 |
|---|---|
| engram 是否参与计算 | ✅ 已证明（A/B 稳定可复现）|
| engram 数学实现 | ✅ 逐项核对正确 |
| engram 数据 | ✅ 数值健康 |
| **整模型能否答对** | **❌ 两条路径都不行** |

**⇒ engram 这条线已经做完了能做的部分。** 剩下的是模型级的问题，
本 goal 里列出的 compressor / APE 那条线很可能就是主因。

### 下一步

既然本机没有对照模型，**先确认 batched 与 default 之外的第三条线索**：
`** 0.0.0` 这个输出形态很特殊——markdown 加粗 + 数字，像是**在输出格式模板**而不是在算数。
下一轮看它是不是 chat template 的问题（prompt 未按对话格式构造）。

