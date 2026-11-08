

---

## Round 67：16K 的坎与 ubatch 无关

### 排除 ubatch

上一轮 16K 失败时用的是自动选的 `ubatch 3072`。
怀疑是 chunk 太大，于是用 `-u 512` 重跑 16K：

**同样失败**，卡在同一位置 `drm_syncobj_array_wait_timeout`，20 分钟后设备丢失。

**所以不是 ubatch 的问题。**

### 矩阵

| 规模 | ubatch | 结果 |
|---|---|---|
| pp4096 | 3072 | ✅ 27.5 t/s |
| pp8192 | 3072 | ✅ 26.3 t/s |
| **pp16384** | **3072** | ❌ device lost |
| **pp16384** | **512** | ❌ device lost |
| pp100000 | 3072 | ❌ device lost |

**坎在 8K–16K 之间，与 batch 大小无关。**

### 卡住时的状态

```
41 个线程，全部 S (sleeping)
RSS 123.2 GB
wchan = drm_syncobj_array_wait_timeout
```

**所有线程都在等 GPU fence，没有一个在计算。** 这是 GPU 侧的挂起，
不是 host 死锁（host 死锁会有线程在 R 状态抢锁）。

### `actual_lanes=5` 不是 bug

查 `pager.rs:3260` 的 lane 规划：`actual_lanes` 是**"能装进固定 corridor 的
完整 Prefill streaming lane 数"**，显存约束下的必然结果，不是配置错误。

真正的问题是 16K 时 GPU 侧挂起，与 lane 数无关。

### 设备每次都安然无恙

三次 device lost 之后 `infr resources` 仍报 `vram_live: true`。
**这是可恢复的失效，不是驱动 wedge。**

### 结论

**本机可用的 prefill 上限是 8K（26.3 t/s）。**
16K 及以上会导致 GPU 挂起，与 ubatch 无关，属于引擎在这个模型+规模下的问题，
需要看 GPU 侧具体在等什么 fence 才能定位——这超出了一次会话能修的范围。

