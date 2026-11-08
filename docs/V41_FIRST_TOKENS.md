

---

## Round 54：fence 等待的源头 —— 每层一次全队列 wait-idle

### 代码事实

`crates/infr-vulkan/src/adapter.rs:16250` 构造等待线程时：

```rust
waiter: if cfg.decode_gpu_resident_waiter {
    Some(Arc::new(crate::recorder::FenceWaiter::new(
        Arc::clone(&be_.shared),
        !cfg.decode_gpu_resident_fence_wait,   // <-- queue_idle
    )?))
}
```

`FenceWaiter` 的等待线程（`recorder.rs:17158`）：

```rust
let result = if queue_idle {
    shared.queue_wait_idle_serialized()   // 整个队列同步
} else {
    Ok(())
}
.and_then(|()| unsafe { shared.device.wait_for_fences(&[job.fence], ...) });
```

而 `decode_gpu_resident_fence_wait` 的默认值是 **`false`**（`config/mod.rs:447`）。

### 于是默认路径是

**每一次 fence 等待之前，先对整个 Vulkan 队列做一次 `queue_wait_idle`。**

对应 Round 53 的实测：

```
sync: waits=1149 total=3.88s
  ├─ queue_wait_idle 521 次  1.63s   <- 就是这个
  └─ fence           628 次  2.25s
```

### 源码注释自己说明了代价

```
/// Optional host wait producer. Fence mode leaves the Vulkan queue mutex available to
/// submits; the queue-idle comparator uses the ordinary serialized gate and can delay
/// further submits.
```

**全队列 wait-idle 会占住队列锁，延迟后续 submit** —— 所以它不只是等，还在**阻断流水线**。

### 这就是一个 A/B 就能验的开关

`decode_gpu_resident_fence_wait = true` 会切到「只等自己那个 fence、不等整个队列」。

**但 `--set decode.gpu_resident_fence_wait=true` 报 `unknown config path`** ——
这批字段是宏生成的，配置路径前缀还没查清。**下一轮第一件事就是把路径试对，然后跑 A/B。**

### 自我批评

这轮我在「试配置路径」上又磨了一段时间，**和上一轮磨日志开关是同一个毛病（第三次）**。
好在核心事实是从源码直接读出来的，不依赖把开关点着。

