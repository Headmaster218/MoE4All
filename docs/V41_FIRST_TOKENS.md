
---

## Round 174-175：注意力在工作，但语义映射错位

### 长度敏感性测试（同一末 token，不同长度前缀）

| 输入 | top-1 | 熵 |
|---|---|---|
| `France is` | 294: **9.73%** | 6.198 |
| `The capital of France is` | 294: **16.65%** | 5.619 |
| `Q: What is the capital of France? A: The capital of France is` | 294: **15.41%** | 5.907 |

**前缀越长，置信度越高（9.73% → 16.65%）。**
**⇒ 注意力确实在整合上下文，不是失效的。**

**但 top-1 始终是 `294` —— 三种完全不同的上下文给出同一个答案，而且不是 Paris。**

### 这不是"信息没传到"

Round 170 修完 engram 越界后，输入敏感性恢复了（Hi / Ho / France 各自不同的 top-1）。
现在进一步确认：**上下文长度也能影响置信度。**

**⇒ 残差流里有信息，注意力在工作，但模型无法把信息映射到正确的输出 token。**

**问题在输出侧或最后几层的表示质量。**

### 已检查但无问题的项

| 项 | 结论 |
|---|---|
| `attn_kv.weight [n_embd, head_dim]` | ✅ 单 MQA 头直接输出 512 维，无需上投影 |
| K = V 同向量 | ✅ 与参考实现一致 |
| `d4_kv` 原地 RMSNorm | ✅ 尺寸匹配 |
| `lm_head_sampling_input` | ✅ `logits_rows == batch` 时直接透传 |
| `d4_comp` 越界 2 KiB | 已修，但实测逐位无影响（Round 173） |

### 逐层观测特性：三轮尝试后确认需要架构改动

想让每层残差流可读，必须让 `generate_dense_backend_inner` 多带一个 `Option<&mut Vec<f32>>`
（照 `h_out` 的模式），这会改动它的签名和**所有调用点**。

三次尝试都卡在结构问题：
1. sink 是构建期局部变量，执行期看不见
2. `thread_local!` 被代码注释明确禁止（"only sound while one generation owned one thread"）
3. 复用 `h_out` 不行，它由 `h_tap` 构建参数控制，默认关闭

**这是要设计的 API 改动，不是顺手能加的探针。** 已记录为下一步的独立任务。

