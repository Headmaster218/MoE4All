

---

## Round 97：`unbound tensor 3` 就是 engram，根因是 record-once 路径

### 第一步：确认 tensor 3 是什么

在 `prepare_decode_plan` 的 `be.compile(&g)` 之前打印前 6 个张量：

```
ENGRAM_T0 kind=Input    desc=[1],     dtype=I32
ENGRAM_T1 kind=Internal desc=[5120],  dtype=F32
ENGRAM_T2 kind=Input    desc=[1],     dtype=I32
ENGRAM_T3 kind=Input    desc=[6144],  dtype=F32   ← 就是它
ENGRAM_T4 kind=Input    desc=[18688], dtype=U32
ENGRAM_T5 kind=Input    desc=[1],     dtype=U32
```

**`[6144] F32` 正是 `n_cols(24) × key_length(256)`。**

**所以 tensor 3 就是 `engram_emb`。** Round 96 里我说"tensor 3 不是 engram 的 wkv"——
前半段（不是 wkv 的权重）对，后半段（不是 engram）错。**它就是 engram 的输入。**

### 第二步：为什么 bind 了还是 unbound

链路是：

```
bind_future(...)  ->  bindings  ->  ExecutionStage::new(*plan, bindings)
                                ->  execute_decode_stages(...)
```

**但 execute 有两条路径：**

```rust
if let Some((plan, b)) = &ro {
    // Record-once path: reuse the single compiled plan + bindings (no per-token rebuild).
    be.execute(plan.as_ref(), b)?;          // ← 用 ro 里的 b，不是 bind_future 的
} else if c.qwen4exp {
    ...
}
```

**`ro` 是 record-once 路径**，用的是**调用方预先建好的 `(plan, bindings)`**，
**完全绕过 `bind_future`**。我在 `bind_future` 里加的 engram bind 在这条路径上不生效。

### 这解释了所有矛盾

| 观察 | 解释 |
|---|---|
| `ENGRAM_DECODE_BIND holder=true buf=true` 打印了 | bind_future 确实执行了 |
| 但 tensor 3 仍然 unbound | **执行的不是 bind_future 的 bindings** |
| 报错在 op 67 | engram 的 `Linear` 在 ro 的 plan 里，但它没有 engram 的 binding |

### 我的诊断方法这次是对的

**打印张量 desc 一步定死了 `[6144]` 这个特征**，比查 label、查 kind 都快。
`label` 全是 `None`——这个项目没有给张量打标签，所以只能靠形状识别。

### 下一步

`ro` 的 `(plan, bindings)` 是调用方构建的。要么：
1. 在那里补 engram bind
2. 或在 record-once 场景下禁用 engram（但那等于 engram 永远不生效）

**先找到 `ro` 的构造点**（它是 `generate_dense_backend_inner` 的入参，在 `mod.rs` 或上层构造）。

