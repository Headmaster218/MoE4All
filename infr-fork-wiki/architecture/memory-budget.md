# 统一内存预算与模型驱动规划

[首页](../README.md) / [架构](README.md) / Memory budget

## 问题从哪里来

早期参数只表示 Expert Cache。用户填 7/10/14 GiB 时，KV、fixed weights、recurrent state、
activation scratch 和驱动余量都在参数之外。这样会出现两个相反问题：

- Prefill 峰值把总显存顶穿；
- Decode 时 Prefill reserve 空着，但 Expert 不能利用，20～22 GiB budget 最终只占约 17 GiB。

类似地，Q8 KV 曾按粗略常数估成 8 GiB，而真实 200～250K 场景只有约 2～3 GiB；固定
双槽/层数/矩阵大小也会把 35B 假设泄漏到新模型。

## 最终预算输入

服务启动时的核心用户输入是：

- 要同时加载的模型（LLM、Embedding；Vision/Draft 为未来）；
- 总 VRAM budget；
- 总 RAM budget；
- context 与 KV dtype；
- 必要的安全 reserve/显式 override。

其余容量由 GGUF metadata、tensor dtype、shape、实际 alignment 和后端 allocation rule
计算，而不是根据模型名写死。

## 规划对象

### 固定 VRAM

- dense/shared fixed weights（未进入 paged Expert 的部分）；
- backend/device 必须常驻对象；
- KV cache 与 recurrent persistent state。

这些对象生命周期覆盖整个 session，不适合与每请求 Expert/Embedding 互相淘汰。动态 KV
可以继续研究，但不在当前实现范围。

### 弹性 VRAM

- paged routed Experts；
- LLM activation/runtime scratch；
- Embedding weights/runtime；
- 预留的 Vision weights/runtime；
- 预留的 Draft weights/runtime。

runtime reserve 是容量约束，不再是一块从启动到结束都空占的独立 allocation。对象不在
生命周期内时，空间可全部成为 Expert slot。

### RAM

- 模型 loader/metadata 与不可避免的小对象；
- full-RAM Host Store，若 routed-expert payload 能装下；或
- bounded inclusive RAM expert cache；
- Embedding/Vision 等从 VRAM 淘汰后的 host source（由各 engine 生命周期决定）；
- aligned/importable host allocations；
- I/O 和 promotion 的有限工作 buffer。

RAM budget 是所有模型的共同上限，不应只计 LLM experts，再让 Embedding/Vision 在预算外
额外 mmap 一份。

## Memory plan 的计算顺序

```text
解析 GGUF metadata / tensor directory
        ↓
得到每个 tensor 的真实 dtype、shape、byte size、alignment
        ↓
分类：fixed / persistent state / paged expert / elastic runtime
        ↓
计算 context 对 KV 与 recurrent state 的真实需求
        ↓
总 VRAM - fixed - state - margin = elastic arena
        ↓
按各 expert block size 建 pool 几何
        ↓
RAM budget 能否覆盖完整 routed payload？
    ├─ 能：full-RAM Host Store
    └─ 否：bounded inclusive RAM/SSD
```

实际 allocation 与 planner 共用同一套 sizing helper。原因是“预估不准”通常不是算术错，
而是 estimator 与 allocator 在 alignment、dtype、graph shape、phase peak 上各自维护一份
规则后漂移。

## Context 的双预算修正

统一 arena 已经在启动时被 Expert slots 填满。若 auto-context 只看“当前尚未 committed 的
显存”，会错误报告 0；若又允许 persistent KV 直接挤 Expert，则破坏生命周期边界。

最终分成：

- KV/recurrent persistent budget：必须放进 device 真正未承诺给 elastic arena 的空间；
- activation scratch budget：可以借用/复用已经 committed 的 elastic arena。

这样 full Expert cache 不会让 auto-context 错判 0，同时 KV 也不会偷偷成为可淘汰对象。

## RAM 策略自动选择

设 routed expert payload 为 `E`，可用于专家的 RAM budget 为 `R`：

- `R >= E`：完整 Host Store，运行期不触发 SSD demand read；
- `R < E`：按 pool/layer 均匀预加载 `R`，其余由 SSD on-demand fill；
- 小模型若 fixed+state+experts 全进 VRAM，Pager 自然退化为近全命中。

“不走 SSD”不是用户再选一个模式，而是 planner 发现完整 payload 已被 RAM budget 覆盖后
自动选择 full-RAM path。

## 安全 margin 与浪费的区别

合理的 margin 包括：

- Vulkan allocation alignment；
- 驱动/descriptor/command 需要的少量 headroom；
- 模型 load 阶段暂时比稳态更高的 allocation；
- 物理 shard 上不能变成完整 expert slot 的尾部。

不可接受的浪费包括：

- 为 Prefill/Decode 各留一整份 cache；
- Embedding 不使用时仍永久驻留；
- 每种 role 单独切固定容量导致其他池有空槽也不能用；
- runtime reserve 在对象不存在时仍物理占用。

弹性 VRAM 验收中 18.79 GB arena 的初始 tail 只有 720,896 bytes（0.00384%），证明按
slot rounding 留下的空间已很小。

---

[架构索引](README.md) · [统一 VRAM](unified-vram.md) · [RAM/SSD](ram-ssd-cache.md)
