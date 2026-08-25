# MoE U/G/D、Shared 与 Miss 调度

[首页](../README.md) / [架构](README.md) / MoE scheduling

## 计算对象

一个 routed Expert 的 FFN：

```text
u = Up(x)
g = activation(Gate(x))
h = u ⊙ g
y = Down(h)
```

文档中的 `UGD` 表示完整 Gate+Up+activation+Down；`UG` 表示只算到中间激活。每个 token
每层通常有 8 routed Experts，Qwen 另有 1 shared expert，因此完整计算规模是 9。

## Shared 为什么可以融合

Shared expert 数学上也是一套 U/G/D，不同点主要是：

- 永远参与，不由 router top-k 决定；
- 权重可能是 Q8，而 routed experts 是 IQ4/Q5/Q6；
- 旧图把它作为独立 dense branch，再做独立 accumulate。

新增 mixed descriptor 后，一个 batched kernel 可以为每个 item 选择自己的 dtype/block
decoder。Q8 不会预先膨胀成 F16 再与 Q5/Q6 混算；各自仍在 shader 中按原量化格式直接
dequant/accumulate。

融合结果：35B 约 85.7/86.5 → 90.0/90.6 tok/s；122B 约 17.0/17.3 → 19.2/19.2。

## 最终 scheduler

```text
router results
      ↓
classify resident / miss and protect all required blocks
      ↓
┌───────────────────────────────┬──────────────────────────────┐
│ GPU: shared + resident UGD    │ CPU/I/O: promote all miss UGD│
└───────────────────────────────┴──────────────────────────────┘
      ↓ dependency satisfied
GPU: all promoted miss UGD as one batch
      ↓
accumulate routed weights + shared output
```

Gate/Up/Down 在 metadata 上仍是独立 blocks，便于不同 size class 和 full-RAM compatibility；
在一次 promotion/compute request 中则尽量合批，减少固定开销。

## 为何不按 D 所在 tier 分支

讨论过的复杂策略：

- RAM miss：先搬 UG，计算 UG 时搬 D；
- SSD miss：固定成本大，一次搬完整 UGD；
- miss 少：先算已有；miss 多：等一半或全等；
- 根据 RAM/SSD miss 比例动态找阈值。

实测后否决，因为时间尺度不匹配。

### 计算时间

122B 完整 UGD，2→9 experts：

- IQ4_XS：34.5 → 120.0 µs；
- Q5_K：26.2 → 68.9 µs；
- Q6_K：29.5 → 87.9 µs。

### 搬运时间

1 expert RAM UGD 已是 353/442/523 µs，远长于完整 9-expert compute。SSD 1 expert 是
1.689～2.753 ms。

### 分段固定成本

- first submit：约 25～50 µs；
- record/submit hand-off：约 20～50 µs；
- resident UGD 拆成 UG + D：比单 submit 多 54～118 µs；
- 122B routed UG compute 仅 14～70 µs；RAM D promotion 109～1348 µs。

UG 计算窗连一个 D promotion 都覆盖不了，反而多一个 CPU/GPU hand-off。模拟中 split-Down
通常比 all-UGD 慢 0.1～0.75 ms（RAM）或 0.6～3.1 ms（SSD）。

## “CPU/GPU 同步”的准确含义

不必在 CPU 上阻塞 `pending_ug.wait()` 才能记录 D；同 queue ordering/fence 可以让 D kernel
在 UG 后执行。但 D copy 和第二段 submit 必须在 UG 完成前到达队列，否则 GPU 仍会 bubble。

因此同步成本包括：

- CPU 得知/保证 D 可用的 dependency management；
- record + second submit；
- queue 在依赖未满足时的 idle；
- 可能的 fence/timeline 操作。

它不是“GPU 会神奇地自动等 RAM copy 然后继续”，也不等于必须做一次 CPU blocking wait。

## 为什么仍先算 resident

虽然 compute 比 transfer 短，resident-first 仍可在 miss 较多时藏住一部分 promotion 固定
成本，并融合 shared 的必算工作。122B 微模型对多 miss 的 hit-first UGD 通常比 wait-all
少数百微秒；35B 某些 1～2 miss 点受噪声/submit 成本影响可略负，但 shared fusion 的端到端
ABBA 为正。

因此策略保持两段，不再加第三段“等一半 miss”。

## Copy 优先级

CPU push 与 GPU compute 可以并行，但没有一个跨设备通用的简单开关保证“PCIe push 优先级
高于 shader”。CPU direct ReBAR write、host DRAM、PCIe、GPU memory controller 和 queue
各自参与。可控手段是：

- 合并足够大的 copy batch；
- 提前提交 promotion；
- 使用 imported-host DMA；
- 减少小 submit；
- 合理使用独立 queue/timeline（仍待生产验证）。

---

[架构索引](README.md) · [微基准](../reference/moe-schedule-microbench.md) ·
[Qwen 122B](../models/qwen35-122b.md)
