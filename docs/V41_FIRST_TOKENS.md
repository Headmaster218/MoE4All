

---

## Round 63：显存还有余量，但 arena 不由它决定

### 更正设备规格

```
device_name: Intel(R) Graphics (BMG G31)
vram_total_bytes = 34,242,297,856 = 31.9 GiB
vram_available   = 30,772,559,872 = 28.7 GiB
```

**不是 24 GiB，是 31.9 GiB。** 我先前写"24 GiB 显存"是错的。

### 于是余量存在

| 项 | 值 |
|---|---|
| 设备 arena | 21.27 GiB |
| 显存总量 | 31.9 GiB |
| **占比** | **67%** |

**约 10 GiB 空闲。** 这意味着提高命中率在**容量上**是可能的。

### 但 `INFR_NO_VRAM_GUARD=1` 无效

| | 默认 | `NO_VRAM_GUARD=1` |
|---|---|---|
| decode | 6.2 tok/s | **6.2 tok/s** |
| hit_rate | 61.0% | 61.0% |
| arena | 21.27 GiB | 21.27 GiB |

**arena 大小没变。**

### 为什么

preflight 那行写的是 **"provisional device arena at ubatch 256"** ——
**arena 是按 `ubatch` 算出来的，不是按显存余量算的。**

`INFR_NO_VRAM_GUARD` 解除的是显存保护，不改变 arena 的尺寸公式。

### 所以要提高命中率，得改 ubatch

但 `ubatch` 是 prefill 的批大小：调大能扩大 arena（更多专家常驻），
但会改变 prefill 的并行度——**可能拿 decode 换 prefill**。

这是个真实的权衡，值得测，但要注意两边都要量。

### 下一步

A/B `--ubatch 512`，同时记录 prefill 和 decode 两个数字。

