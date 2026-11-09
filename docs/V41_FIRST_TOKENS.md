

---

## Round 123：开关已默认打开 —— prefill 4.9 倍，decode 不变

### 改动

`runner.rs`：`INFR_DSV41_BATCHED_PREFILL` 从 **默认关闭** 改为 **默认开启**。
`=0` 仍可强制回退到旧的逐 token 路径（用于二分数值差异）。

### 验证默认确实生效

| 命令 | pp512 |
|---|---|
| 无环境变量（新默认） | **29.5 t/s** |
| `INFR_DSV41_BATCHED_PREFILL=0` | **6.0 t/s** |

### 当前阶段性能

| 指标 | 改动前 | 改动后 |
|---|---|---|
| **prefill pp512** | 6.0 t/s | **29.5 t/s**（4.9×）|
| **prefill pp32** | 6.4 t/s | **15.7 t/s**（2.5×）|
| **decode** | 5.9 t/s | **5.8 t/s**（不变，符合预期）|

**decode 不受影响** —— 批量路径只优化 prefill。

### 对照 poorseek

| 指标 | poorseek（RTX 4090） | 本机（Arc BMG G31） |
|---|---|---|
| prefill | — | **29.5 t/s** |
| **单 token decode** | **27.7 t/s** | **5.8 t/s** |
| MTP decode | 35.9 t/s | 未测 |

**prefill 已达到可对比的水平；decode 仍有 4.8 倍差距。**

### decode 为什么没跟上

decode 是 batch=1，**批量路径对它不起作用**（无 batch 可批）。
它的 6 t/s 仍来自 Round 120 发现的那个问题：**每 token 885 次专家 GEMV**，
在 batch=1 时无法摊薄。

**⇒ decode 的提速需要另一套手段**（MTP/投机解码、或减少每 token 的专家访问），
不是这个开关能解决的。

### 正确性仍未解决

默认路径输出 `We need answer. Need to determine actual expected behavior. The user wants me to`
—— 连贯但没答对 `2+2=`。

**这是 goal 原定的 compressor/APE 那条线，尚未处理。**

