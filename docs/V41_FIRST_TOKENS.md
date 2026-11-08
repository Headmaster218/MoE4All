

---

## Round 94：上传点找错了路径

### 我插探针的位置属于 qwen4exp

第 0 层的 bind 后面紧跟 `be.execute`，看起来是上传的正确位置。但细看那段代码：

```rust
let worker = ple_worker
    .as_ref()
    .ok_or_else(|| anyhow!("qwen4exp session has no PLE worker"))?;
for (&lane, range) in batch_lanes.iter().zip(&ranges) {
    ids.extend(parallel.prompts[lane][range.clone()].iter().map(|&token| token as i32));
    tickets.push(worker.submit_range(...));
}
```

**`ple_worker` 是硬性要求，没有它直接报错。** 这段只会跑在 qwen4exp 上——**V4.1 根本不进这里**。

所以 `ids`、`batches`、`ple_rows` 这些变量对 V4.1 无意义，我差点把 engram 的 gather 接在一条 V4.1 永不执行的路径上。

### 已确认可用的部分

| 事实 | 状态 |
|---|---|
| `cached`（已处理 token，`SeamKv::cached_tokens`）在调用点可见 | ✅ 探针验证通过 |
| `engram_emb_buf` 在 7 个调用点都可见 | ✅ |
| `gather_for_layer` 编译通过 | ✅ |

**缺的是"V4.1 的 prefill 走哪条路径"这个认知。**

### 这一轮的实际产出

- 排除了一个错误的接线位置（比接错后再 debug 便宜得多）
- 确认 `cached` 在调用点可用，下一轮直接用
- 工作树干净、编译通过

### 下一步

找 V4.1 的实际 prefill 路径，把 gather 接在**真正执行**的那条上。

