

---

## Round 53：decode 瓶颈诊断 —— 不是带宽，是同步

### 先纠正我自己的一个错误推论

看到 `dma=54 MiB/层`，我算出「每 token 2.1 GiB，需要 59 GiB/s」。
**这个推论是错的**——那 54 MiB 是 prefill 大 batch 的量，不是 decode 的。

### decode 实测（12 token，ctx 4096）

```
decode 12 tok @ 6.2 tok/s
gpu cache: hits=19420 misses=12980 hit_rate=59.9%
device queue timeline: span=4.40s  main_busy=2.86s  outside=1.54s
```

| 项 | 值 |
|---|---|
| GPU 利用率 | **65%** |
| GPU 空闲 | **1.54s（35%）** |

### 空闲在哪：sync 拆解

```
sync: waits=1149  total=3.88s
  ├─ fence           628 次  2.25s   avg 3.58ms/次
  ├─ queue_wait_idle 521 次  1.63s
  └─ paging_explicit 520 次  1.64s
```

**12 个 token 等 628 次 fence —— 每 token 52 次同步，每次 3.58ms。**

### 已验证：专家缓存不是瓶颈

`INFR_MOE_SIZE_CACHE_BIAS=2.0`（日志确认 `+2.00 (explicit)` 生效）：

| | decode |
|---|---|
| bias off | 6.2 tok/s |
| bias 2.0 | **6.2 tok/s** |

**完全没变。** 所以「提高命中率」这条路是死的。

### 结论

decode 慢在**同步次数**，不在传输量、不在缓存命中率。

下一刀应该数 fence 是谁发的、能不能合并。

### 一个诚实的说明

这轮我先推错了带宽结论，是被 `main_busy` vs `span` 的差值纠正过来的。
**又一次：「先算了个像样的数」不如「先看一个实测的分解」。**

