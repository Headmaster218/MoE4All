

---

## Round 116：逐项核对 engram 数学与数据 —— 全部正确，问题不在这些地方

### 核对清单（对照 `deepseek41.cpp` 的 `build_engram`）

| 项 | 参考 | 本项目 | 结论 |
|---|---|---|---|
| key half 偏移 | `[0, hc_dim)` | `w_off: 0`, `out_f: hcw` | ✅ |
| value half 偏移 | `[hc_dim, hc_dim+n_embd)` | `w_off: hcw`, `out_f: ne` | ✅ |
| key 归一化 | `grouped_norm(key, engram_k)` | `QwenHcNorm{x: engram_kv, norm: ew.k}` | ✅ |
| query 归一化 | `grouped_norm(x, engram_q)` | `QwenHcNorm{x: res, norm: ew.q}` | ✅ |
| gate | `sigmoid(sgn(s)·sqrt(|s| clamp 1e-6))` | `QwenPleGate` 同式 | ✅ |
| value 广播 | `ggml_repeat_4d(v, n_embd, hc, nt, 1)` | `row[d] = vs[r*ne + d] * scale`（`r = rh/hc`）| ✅ |
| 残差合并 | `x + v*gate` | `Op::Add{a: res, b: hn, dst: res}` | ✅ |
| eps | `norm_rms_eps` | `c.rms_eps` | ✅ |

### 数值探针：数据完全健康

```
ENGRAM_ROWS layer=1  n=6144 first=-0.117188 max_abs=3.750000 nonfinite=false
ENGRAM_ROWS layer=14 n=6144 first=-0.281250 max_abs=3.250000 nonfinite=false
```

- `n = 6144 = 24 × 256`（`n_cols × key_length`）✓
- 量级 O(1)（`max_abs` 3–4），**无 NaN/Inf** ✓
- 两层数值**不同**（层 1 `-0.117` vs 层 14 `-0.281`）✓ ⇒ 按层 hash 生效

### 结论

**engram 的公式、偏移、广播、eps、数据解量化——全部正确。**

上一轮我怀疑的布局问题（Round 114）和这一轮怀疑的偏移问题，**都不存在**。

### 那 `2+2=` 输出 `0` 是为什么

**可能不是 engram 的问题。** 需要对照：**关掉 engram 时 `2+2=` 输出什么？**

Round 114 的 A/B 里，无 engram 输出 `We need answer. The question is "2+2". So 2+`——
**那个输出同样不是 `4`**。

**⇒ 不带 engram 的基线本来就不对。** V4.1 在 moe4all 上还有其他未修的问题
（如本轮程序 `goal` 里列出的 compressor/APE 那条线），engram 只是其中一环。

### 下一步

**先确认基线**：关掉 engram 时模型是否正确回答任何问题。
若基线也不对，则 engram 再完美也无法验证——**需要先让基线答对**。

