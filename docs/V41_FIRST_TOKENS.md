

---

## Round 78（续）：`hc_mult` 也被排除了 —— 移植是完整的

### 参考实现的实际算法（`deepseek4.cpp:352` `build_hc_mixes`）

V4.1 复用 V4 的这套：

```cpp
flat      = reshape(x, [hc_dim, nt])              // hc_dim = 4*n_embd
flat_norm = rms_norm(flat)
mixes     = hc_fn @ flat_norm                     // 24 个系数 = (2+4)*4

pre  = sigmoid(affine(mixes[0..4],   scale[0], base[0..4])  + eps)
post = 2 * sigmoid(affine(mixes[4..8], scale[1], base[4..8]))
comb = sinkhorn(mixes[8..24], scale[2], base[8..24])        // 4x4 混合矩阵
```

### 本项目的对应实现（`runner.rs:5809`）

```rust
Op::RmsNorm { x: res, weight: ones, dim: hcw, ... }     // ones = hc_ones，全 1 向量
Op::Linear  { weight: t.w_fn, out_f: (2 + hc_mult) * hc_mult, ... }
Op::HyperConnectMix { mixes, scale: t.scale, base: t.base,
                      pre: mix_pre, gates: Some(HyperGates { post, comb }),
                      hc, eps, n_iter: hc_sinkhorn_iters }
```

**逐项对上**：
- flatten + RMSNorm 用的是 `hc_ones`（weight 为全 1，等价于纯归一化）—— 这是**故意的权重力学技巧**，
  `runner.rs:3587-3591` 的注释解释过 `x * s * 1.0 == x * s`
- `out_f = (2+hc)*hc = 24` —— 与参考实现一致
- `sigmoid` / `2*sigmoid` / `Sinkhorn` 全部在 `Op::HyperConnectMix` 内
- `n_iter = hc_sinkhorn_iters` —— 与 `dsv4_hc_sinkhorn_iters` 一致

### 而且注释还处理了 V4.1 独有的时序差异

`runner.rs:5805-5808`：

> *"DeepSeek V4.1 consumes a mix one sublayer BELOW the one that computes it:
> a sublayer runs the matmul but collapses `res` with its PREDECESSOR's weights
> (`deepseek41.cpp` 843-886). V4 does both in the same sublayer."*

**这个滞后语义也移植了。** V4.1 的 HEAD 复用最后一层 FFN mix（`deepseek41.cpp` 919-923）也在注释里。

### 结论：`hc_mult` 候选排除

| 候选 | 状态 |
|---|---|
| 34 复用层 / compressor 属主 | ❌ 排除（Round 75） |
| 角色语义 | ❌ 排除（Round 75） |
| **hc_mult 残差** | ❌ **排除（本轮）** |
| **IQ1_M 专家 dequant** | ⬅ **唯一剩下的** |

### 剩下这一个为什么重要

`grid_mmq_parity` 声称 IQ1_M 与 host 一致，但**两边是实现同一套解码**。
如果那套解码本身错了（比如 IQ1_M 的 `d0` 位序、grid 索引移位），两边会一起错，测试照样通过。

**而 IQ1_M 占专家权重的 79%（30/40 层的 exps 张量）。** 它错了，整个模型就废了——
这与观测到的"全局性套话输出"完全吻合。

### 下一步

逐字段比对 IQ1_M 解码：
- `dequant.rs` 里的 IQ1_M 实现
- 参考实现（llama.cpp 的 `ggml-quants.c` 或这个 fork）的 IQ1_M
- 重点看 `d` 的位打包、四个 grid 索引的移位、delta 符号

**这是唯一还没被排除的候选，而且它影响面最大。**

