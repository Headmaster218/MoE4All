

---

## Round 60：第 9 个否证 —— inflight 深度

### 假设

`decode_gpu_resident_inflight: usize = 1`（默认），注释说它被 clamp 在 1..2，
2 表示「允许额外一个投机段」。**如果只有 1 个在途段，CPU 就无法与下一段重叠。**

这与 Round 59 的结论吻合（CPU 在 `sync_wait` 里等，`overlap_lower_bound=0`）。

### A/B

| | inflight=1（默认） | inflight=2 |
|---|---|---|
| **decode** | **6.2 tok/s** | **6.2 tok/s** |

**没有变化。**

### 这是第 9 个

| # | 假设 | 结论 |
|---|---|---|
| 1 | 统一缩放 | 否 |
| 2 | 慢 copy | 否 |
| 3 | `main_busy` 当上限 | 否 |
| 4 | 带宽不足 | 否 |
| 5 | `queue_wait_idle` | 否 |
| 6 | staging ring slots | 否 |
| 7 | submit 数 | 否 |
| 8 | 隐藏 CPU 开销 | **不存在（已 100% 归因）** |
| 9 | inflight 深度 | 否 |

### 我现在的判断

**decode 没有"可以摘掉的浪费"。** Round 59 的对账已经证明 host 侧 <1% 未归因，
现在方向 2（重叠）也被否掉。

剩下的只有 **方向 1：减少 GPU 的实际工作量**。

而 decode 的 GPU 工作量主要由**每 token 要过多少专家权重**决定：
- `gpu cache hit_rate = 59.9%`
- 40 层 × 每层 top-6 专家 = 每 token 240 个专家实例

**这不是一个"调参"问题，是算法/调度问题。** 可选项：
- 更好的专家亲和性（让连续 token 复用同一批专家）
- 更激进的路由近似（poorseek 用了 `--route-bias`，明确牺牲精度换速度）
- 减少层数（V4.1 的 34 个复用层是否真的都需要跑满）

### 诚实的进度判断

**prefill 已从 12 → 25 tok/s（2.1 倍，已确认）。**
**decode 卡在 6.2，九个方向全部用尽，没有任何一个带来改善。**

对标 poorseek 的 27.7 tok/s 需要的是**架构级的改变**，不是继续找瓶颈。

