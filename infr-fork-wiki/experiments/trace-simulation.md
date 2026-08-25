# Ordered Route Trace 与离线模拟

[首页](../README.md) / [实验](README.md) / Trace & simulation

## 为什么从“继续实测”转向“先模拟”

缓存 policy 的搜索空间包含 VRAM size、RAM size、inclusive/exclusive、admission、shadow、
不同量化 bytes、预取和 route pattern。每个组合都重新加载 70～150 GB 模型不仅慢，还会
混入 SSD cache、GPU state 和 warmup 差异。

只要记录按真实执行顺序的 block access，就能对 LRU/capacity/policy 做 exact replay；真正
涉及 kernel、PCIe、queue overlap 的候选再上硬件。

## Trace 记录内容

每个 access 至少包含：

```text
call_id, sequence, phase, pool, layer, role, expert, block_id, bytes,
gpu_hit, ram_hit, evicted
```

ordered 的关键不是 CSV 行很多，而是可以重建：

- 同一 token 的 48 layers × 8 experts × 3 roles；
- batch epoch 内的访问顺序；
- 每个 size pool 的独立 LRU；
- GPU miss 后 RAM/SSD 的 conditional path；
- cold startup、request Prefill、Decode 三个 phase。

## 122B 完整性验收

2K cold trace：

| 项目 | 数值 |
|---|---:|
| Total accesses | 2,441,088 |
| Pager calls | 98,448 |
| Warmup | 2,304 accesses / 96 calls |
| Request prefill route | 79,488 accesses / 48 calls |
| Decode | 2,359,296 accesses / 98,304 calls |
| Identity check | `2048*48*8*3 = 2,359,296` |
| Sequence discontinuities | 0 |
| Backward call ids | 0 |

这说明 trace 能用于精确 replay，而不是 profiler sampling。

## Replay 模型

### GPU

每个 size pool 建独立 exact LRU，容量用 slot 数而非 GiB 浮点。访问命中则 promote；miss
选择合法 victim，并记录 bytes。

### RAM

必须重现真实 startup preload 与 inclusive shadow：

- 按 pool/layer 比例初始化 resident set；
- GPU block 的 RAM shadow 在 capacity 内保留；
- RAM miss 才计 SSD bytes；
- 不凭空假设 GPU eviction 自动回写 RAM。

### 时间模型

总时间通常写成：

```text
T_token = T_compute_and_fixed
        + GPU_miss_bytes / measured_H2D_bw
        + RAM_miss_bytes / measured_SSD_bw
        - validated_overlap
```

固定项必须用同一 binary/shape 的实测校准；overlap 不能简单把两个时间取 max，除非 timeline
实验证明确实并行。

## DeepSeek V4 示例

在 12.67 GiB GPU + 40 GiB RAM、4.3 tok/s 校准：

```text
token_ms = 72.57
         + 0.2396 * MXFP4_GPU_miss_blocks
         + 1.0170 * MXFP4_RAM_miss_blocks
```

模拟显示该 128-token trace 的 distinct working set 约 59.0 GiB，因此 60 GiB RAM 后平台；
不是因为完整 147.17 GB payload 被装下。75/100/110 GiB 在同一短 trace 上不再提升，不能
推导新对话也不提升。

## 预取是否值得

预取必须比较：

- precision：预取 block 后是否真的很快访问；
- pollution：是否挤掉更热 resident；
- lead time：从已知 router 到需要 compute 有多少层/多少 µs；
- SSD/H2D 并发队列是否有空隙；
- 多 conversation 路由是否仍可预测。

当前 trace 只能评估基于过去 route 的 policy；无法证明模型内部 router 未来 token 的
语义预测。默认不随机填充 RAM/VRAM，也不因有空带宽就盲目预取。

## 模拟能回答/不能回答

可以：

- 90%/95% hit 需要多少 slots；
- VRAM/RAM 二维容量甜点；
- inclusive shadow 的容量成本；
- 不同 block bytes 的 size-effect；
- policy 的 miss/eviction 数是否更好。

不能直接回答：

- 新 kernel 真实执行时间；
- Windows driver import ceiling 是否变化；
- compute/copy 是否真正 overlap；
- SSD thermal/cache/queue 的长期带宽；
- 量化改变后输出路由是否仍是同一 trace。

最后一点很重要：用 MXFP4 route trace 模拟 IQ3_M 只表示“保持同样访问序列、block bytes
缩至 87.3%”的容量效果，不代表 IQ3_M 改变 logits 后仍访问同一 experts。

---

[实验索引](README.md) · [122B trace](../reference/qwen122-trace.md) ·
[DeepSeek data](../reference/deepseek-v4-data.md)
