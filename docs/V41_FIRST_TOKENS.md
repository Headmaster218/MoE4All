

---

## Round 78：拿参考实现对照 `hc_*` 接线

### 参考实现里的关键结构

`deepseek41.cpp`（948 行，本机 llama.cpp fork）：

```cpp
const int64_t hc_mult    = hparams.dsv4_hc_mult;      // 4
const int64_t hc_dim     = hc_mult * n_embd;
const int64_t hc_mix_dim = (2 + hc_mult) * hc_mult;   // 24
...
layer.hc_attn_fn    = {hc_dim, hc_mix_dim}   // [4*n_embd, 24]
layer.hc_attn_base  = {hc_mix_dim}           // [24]
layer.hc_attn_scale = {3}
layer.hc_ffn_fn     = {hc_dim, hc_mix_dim}
layer.hc_ffn_base   = {hc_mix_dim}
layer.hc_ffn_scale  = {3}
```

以及 `dsv41_hc_mean`：**对 `hc_mult` 个副本求平均**（求和后 `scale 1/hc`），
只用在 `t_layer_inp`（层输入嵌入）上，不用于主干。

### 本项目的接线是对的

`runner.rs:2463-2472`：

```rust
for name in [
    "hc_attn_fn.weight",  "hc_attn_base.weight",  "hc_attn_scale.weight",
    "hc_ffn_fn.weight",   "hc_ffn_base.weight",   "hc_ffn_scale.weight",
] { wload(&[&p(name)])?; }
```

**六个全都在，顺序也对。** 模型里 `hc_*` 共 240 个 = 40 层 × 6，一个不少。

（我一开始只 grep 到 `hc_attn_scale`/`hc_ffn_scale` 两处，差点又误判成"少加载 4 个"，
是 `for name in [...]` 的数组把另外四个藏住了。）

### 但参考实现多了一个本项目没有的概念

`hc_mix_dim = (2 + hc_mult) * hc_mult = 24`：

- `2` 应是 attn 与 ffn 两个子层的混合系数
- `hc_mult = 4` 是四路并行残差流

**即每层的 hc 混合矩阵产出 24 个系数**，而本项目用 `hc_ones`（全 1 向量）+ RMSNorm
来做归一化。

**这两种做法不等价**：参考实现是学习出的混合（`hc_*_fn` 投影 + `base` 偏置 + `scale` 缩放），
本项目是固定的 1/hc 平均。

**这正是 `hc_mult` 残差处理的核心差异**，也是接下来要逐行对照的地方。

### 下一步

读参考实现的 `build_arch` 主循环里 `hc_attn_fn`/`hc_attn_base`/`hc_attn_scale`
的实际用法（不是只加载，而是**怎么算**），与 `runner.rs` 的对应片段比较。

