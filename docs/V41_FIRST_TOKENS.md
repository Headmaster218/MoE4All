

---

## Round 66：长上下文 prefill 实测，以及 16K 以上的硬失败

### 方法修正

先用 `infr run <prompt>` 试 100K，直接失败：**`参数列表过长`**。
400KB 的 prompt 不能走命令行参数。

改用 **`infr bench`**（llama-bench 同接口，按 token 数生成 prompt）：

```
infr bench <model> -p <N> -n 0 --dev Vulkan1
```

### 实测结果

| 规模 | 结果 |
|---|---|
| **pp4096** | **27.5 t/s**（3 reps, 27.4–27.5, spread 0.4%） |
| **pp8192** | **26.3 t/s**（3 reps, 26.2–26.3, spread 0.3%） |
| **pp16384** | **失败**（跑了 20+ 分钟无输出） |
| **pp100000** | **失败**（26 分钟后 `device lost`） |

两次成功都自动选了 **`ubatch 3072`**（不是 `infr run` 默认的 256/512）。

### 重要发现：bench 的 ubatch 更大，prefill 更快

`infr run --ubatch 256` 是 25 tok/s，`infr run --ubatch 512` 是 27 tok/s，
而 **`infr bench` 自动用 3072，拿到 27.5**。

**所以在 prefill 场景，ubatch 越大越快**（与 Round 63 的观察一致）。

### 100K 失败的诊断

日志：

```
[moe-prefill] target_lanes=40 actual_lanes=5 resident_layers=0/40
[moe-prefill] streamed_layer_max=3928227840 ring_bytes=19074908160
```

- **`target_lanes=40` 但 `actual_lanes=5`** —— 只能同时跑 5 层
- `resident_layers=0/40` —— **一层都没常驻**
- `ring_bytes=19.07 GiB`，`streamed_layer_max=3.93 GiB`

进程等待位置是 `drm_syncobj_array_wait_timeout`（等 GPU fence），
26 分钟后 `devsandqueues-lost-device`。

**这是真实的规模上限，不是调用方式问题。**

### 设备未受损

失败后 `infr resources` 仍报 `vram_live: true`，`/dev/dri` 齐全。
**这次 device lost 是温和的，没触发驱动 wedge。**

### 当前基线

| 场景 | 实测 |
|---|---|
| prefill 4K | **27.5 t/s** |
| prefill 8K | **26.3 t/s** |
| prefill 16K+ | 不可用（device lost） |
| decode | 6.2 t/s @ 4K ctx |

**poorseek 的 100K@1063 t/s 在本机完全不可达**，连 16K 都跑不完。

