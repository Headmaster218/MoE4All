

---

## Round 55：A/B 否定 —— queue_wait_idle 不是瓶颈

### 做法

不找 CLI 开关（上轮在这上面耗了太久），直接**改默认值重编译**：

```rust
// crates/infr-core/src/config/mod.rs:447
decode_gpu_resident_fence_wait: bool = false  ->  true
```

`false` ⇒ `queue_idle = true` ⇒ 每次 fence 等待前做一次全队列 `queue_wait_idle`。
`true`  ⇒ `queue_idle = false` ⇒ 只等自己那个 fence。

### 结果

```
相同命令、相同模型、--max-new 12

fence_wait=false (默认):  decode 12 tok @ 6.2 tok/s
fence_wait=true  (A/B) :  decode 12 tok @ 6.3 tok/s
```

**6.2 → 6.3，噪声级别。**

### 结论

**Round 54 的「全队列 wait-idle 是 decode 瓶颈」被否定。**

那 1.63s 的 `queue_wait_idle` 确实是**在**等待，但它等的时间本来就要等
（在等 GPU 算完），换成等单个 fence 只是换了个等法，不省时间。

### 教训

这是本轮调查里第五个「看起来是瓶颈、A/B 一测就不是」的假设。
前四个：统一缩放、慢 copy、main_busy 当上限、带宽。

**凡是「时间花在哪」的推理，都必须用 A/B 证明「拿掉它真的更快」，
而不是只看它占了多少。**

### 已回退

默认值改回 `false`。**注意二进制需要重新编译回默认值再测别的**，
否则后续测量会建立在实验版本上。

