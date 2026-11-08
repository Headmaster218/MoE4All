

---

## Round 74：读源码定位到一处具体的"拿角色当比例"的用法

### 先纠正我自己

`runner.rs:6942-6962` 的注释和代码**已经实现了目标里的第 (2) 步**：

```rust
let owns_compressor = lw.dsv4_compressed.is_some();   // 按张量存在与否判所有权
```

`ape: cw.attention.ape.or_else(|| dsv41_zero_ape(c, zero_ape))` 也在（`runner.rs:7009`）。

**所以"给 Dsv4Compress 加 ape: Option"和"复用层不发射 compressor"都已完成。**
我前面几轮把它们当成待办是错的。

### 但发现一处真实的类型误用

`config.rs:524-530` 自己写着：

> *"on V4.1 it is a **role map, NOT a ratio**, so a consumer that wants a ratio out of it
> is reading the wrong number."*

而 `runner.rs:7158` 正是这种消费者：

```rust
visible: ((start_pos + 1) / (ratio4 as usize).max(1)) as u32
```

`ratio4` 来自 `c.layer_compress_ratio(l)`（`runner.rs:6751`）——**就是那个角色表**。

同样的用法还有三处：`7021`（`is_multiple_of(ratio4)`）、`7026`（`start_pos / ratio4`）、
以及 `ratio: ratio4` 传给 `Dsv4Compress`。

### 模型里的实际数据

```
compress_ratios = [0,0, 2 x18, 1 x20, 0,0,0]    共 43 项
block_count     = 40
```

角色分布（前 40 层）：**role=1 有 20 层，role=2 有 18 层，role=0 有 2 层**。

四个 kv source `[2,8,14,20]` 的角色分别是 **2, 2, 2, 1**。

### 这意味着什么

如果 role 2 真的被当成"2 倍压缩比"：
- `visible = (pos+1)/2` —— 每两个 token 合成一个
- `row = pos/2` —— 写第 pos/2 行

**但如果 V4.1 的实际压缩比不是 2**（V4 是 4 和 128），那么复用层读到的位置全是错的。
18 个 role=2 层 + 20 个 role=1 层，**几乎整个模型都受影响**——足以解释全局性的输出错误。

### 但我不能确定

`role=1` 时除以 1 等价于不压缩，看起来是"对的"。
`role=2` 时除以 2 **可能恰好就是对的**（角色 2 = 池化 2 倍），也可能不是。

**这需要外部资料确认 V4.1 的 CSA2 语义**，而不是靠猜。

### 顺带：`compress_ratios` 43 项 vs `block_count` 40

多出 3 项（`0,0,0`）。可能是 MTP 层的槽位，但也不排除是索引偏移。
**这个必须先确认**——如果 `layer_compress_ratio(l)` 取错了偏移，后面全部作废。

