

---

## Round 58（续）：A/B 否定 —— submit 数不是瓶颈

### 假设

Round 58 定位到「空档在 submit 之间」，于是减少 submit 数应该有效。
`AUTO_SUBMIT_INITIAL_CAP`（默认 16）决定每次 forward 被切成多少段。

### A/B（改常量 + 重编译）

| | cap=16（默认） | cap=256 |
|---|---|---|
| `main_intervals` | 1149 | **949** |
| `main_busy` | 2.86s | 2.88s |
| `submits` | 1684 | **1484** |
| `cap_end` | split/16 | split/256 |
| **decode** | **6.2 tok/s** | **6.2 tok/s** |

**submit 数降了 12%，性能一格没动。**

### 为什么又错了

空档确实在 submit 之间，但**减少 submit 数并没有减少总空档**——
大概率是：submit 少了，但每次 submit 之间的空档变长了，总账不变。

这正是 Round 55 那个教训的第二次出现：
**「时间花在 X 上」不等于「减少 X 就能省时间」。**

### 累计 7 个被 A/B 否掉的假设

统一缩放、慢 copy、`main_busy` 当上限、带宽、`queue_wait_idle`、
staging ring slots、**submit 数**。

### 已回退

cap 改回 16。

### 现状与判断

decode 6.2 tok/s。七个假设全灭，而每一个都是"看起来最像的那个"。

**下一步该换的是问题本身**：不要再问"空闲时间怎么省"，而是量一下
**每次 submit 之间那 1.35ms 到底在 CPU 侧做什么**——直接抓那段 host 时间。

现在的指标体系只能告诉我们"GPU 闲着"，量不到"CPU 在干嘛"。

