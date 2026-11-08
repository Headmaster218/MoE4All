

---

## Round 95：用运行时证据定位 —— engram 真的被执行了，但走的是 decode 路径

### 静态阅读查不出来，改用探针

前几轮我一直在静态读代码找“V4.1 走哪条 prefill 路径”，读了好几轮没定论。
这一轮改成**在 7 个 bind 点和发射闭包里各加一条 `eprintln!`，跑一次真实 V4.1**。

### 跑出来的结果（决定性）

```
ENGRAM_EMIT batch=1        （×4）
Error: backend: vulkan adapter: op 67 Linear failed: unbound tensor 3
```

**含义**：
1. **engram 的 Op 确实被发射并执行了**（`op 67 Linear` 就是 `wkv`）——
   证明 Round 87-88 写的发射代码是活的，不是死代码
2. **但 7 个 bind 点一个都没触发**（没有任何 `ENGRAM_BIND_SITE_*` 输出）
3. 所以 `wkv` 拿到的是未绑定的 tensor 3

**这才是当前模型答不对题的真正状态**：不是“engram 缺失”，而是
**“engram 已接入计算，但输入没接上”**。

### backtrace 指向真正的路径

```
0: {closure#48}                      runner.rs:5931   ← engram_apply 定义处
1: {closure#73}                      runner.rs:6352   ← 层循环里调用
2: {closure#283}                     runner.rs:13405  ← prepare_decode_plan
3: {closure#284}  ...checkpoints_constrained         runner.rs:13435
4: ...generate_vulkan_session_turn_with_checkpoints_constrained
5: infr_core/step_pipeline.rs:115
```

**V4.1 走的是 `prepare_decode_plan` 这条 decode 路径**，不是我以为的 prefill 路径。

### 而这条路径的注释，早就写明了原因

`runner.rs:1401`：

> *"Only immutable graph metadata reaches this closure: **no bindings, buffers or token history**."*

**这个闭包在设计上就拿不到 buffer 和 token 历史。** 而 engram 两者都需要。

**所以这 7 个 bind 点全部无效不是 bug——是它们压根不在这条执行路径上。**

### 这意味着什么

| 之前以为 | 实际 |
|---|---|
| bind 写好了，只是没在正确位置 | **bind 写在了永远不会执行的路径上** |
| 需要找 prefill 路径 | **V4.1 的 token 生成走 decode 路径** |
| 剩下的是“调用 gather” | **剩下的是“在这条 decode 路径上拿到 buffer 和 token 历史”** |

最后一条是**架构性问题**，不是接一根线：这个闭包明确禁止访问可变状态。

### 诊断代码已清除

`git checkout` 还原，工作树干净，`infr-llama` 编译通过。

### 下一步

两条路：
1. **改 `prepare_decode_plan` 闭包的签名**，让它能接收 engram buffer 和 token 历史
   —— 但这违背了它"只处理不可变元数据"的设计，可能影响 decode 的异步预编译
2. **找 decode 路径上真正做 bind 的地方**，在那里加 engram

**第 2 条更保守**，下轮先查 decode 循环里的 bind 是怎么做的。

