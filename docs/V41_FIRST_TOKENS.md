

---

## Round 64：撤回 Round 62 的带宽天花板结论

### 我用错了带宽上限

Round 62 说「decode 的 6.6 GB/s 相对 9.5 GB/s 上限已用 70%，本机天花板约 9 tok/s」。

**那个 9.5 GB/s 是错的。** 它来自本机**另一个引擎（Strata）的 PCIe probe**，
且是离散 GPU 的 host→device 单向读数。本机是 **Intel Arc（BMG G31），走统一内存**。

### 本仓库自己的探针

```
cargo test -p infr-vulkan probe_zero_fill_bandwidth_by_size -- --ignored --nocapture
```

```
zero-fill [CpuToGpu] mib= 32  ms=0.369  GBps= 89.46
zero-fill [CpuToGpu] mib= 47  ms=0.293  GBps=112.74
zero-fill [GpuOnly]  mib= 32  ms=0.722  GBps= 45.78
zero-fill [GpuOnly]  mib= 47  ms=0.728  GBps= 45.36
```

### 正确的对比

| | 带宽 | decode 6.6 GB/s 占比 |
|---|---|---|
| CpuToGpu（上限） | **112.74 GB/s** | **5.9%** |
| GpuOnly | 45.36 GB/s | 14.6% |

**decode 只用了带宽的 6%–15%。根本不是带宽受限。**

### 结论反转

| Round 62 的说法 | 实际情况 |
|---|---|
| 上限约 9 tok/s | **错误** |
| 27.7 需要 34.4 GB/s，本机给不了 | **本机给得起（上限 112）** |
| decode 已用 70% 带宽 | **只用 6%** |

**所以「decode 已到硬件极限」这个结论作废。** 还有大量空间，只是我还没找到瓶颈在哪。

### 教训

**用别的引擎/别的硬件测出的数字来判定当前系统的上限，是错的。**
必须用本仓库自己的探针，而且要注意该探针测的是什么（zero-fill 的 GpuOnly 是写带宽，
CpuToGpu 是传输带宽，两者不同）。

