
---

## Round 176-177：输出侧无缺陷；逐层观测改用 `h_tap` 路径，不改签名

### 输出侧检查（Round 176）

| 项 | 结论 |
|---|---|
| `wload(&["output_norm.weight"])` | ✅ 最终 norm 张量名正确 |
| LM head 选择（`output.weight` 或 tied `token_embd`） | ✅ 逻辑正确 |
| logits 峰值尺度 | ✅ 15.6 vs 参考 14.98，同量级 |

**输出侧结构上没有缺陷。**

### 逐层观测：放弃改签名，改走 `h_tap`

**为什么放弃改签名：** `generate_dense_backend_inner` 有 7 个调用点、每个约 30 个**位置参数**。
插入新参数时一旦位置错，会编译通过但运行行为错乱 —— 而验证一次要加载 130 GB / 约 30 分钟。
**风险与收益不匹配。**

**新方案（下轮直接做）：**

`h_tap` 机制已经完整存在 —— 缓冲分配（`h_tap_buf`）、tensor 绑定（`h1.h_out`）、
host 下载（`be.download`）三段都现成，而且它只被 MTP 路径使用。

`runner.rs:9239` 处：

```rust
let h_out = if h_tap {
    let h_src = if c.qwen4exp { qwen_wide } else { lm_in };   // ← 改这里
    let ho = g.output(f32d(h_width * h_rows));
    g.push(Op::Copy { src: h_src, ... });
```

**把 `h_src` 从 `lm_in` 换成某层记下的 `hcr[0]`，层号由环境变量选**，
就能在**不改函数签名**的前提下读到任意一层的残差流。

**唯一需要的额外接线**：层循环构建时把 `hcr[0]` 的 `TensorId` 存到一个在
`generate_dense_backend_inner` **同一栈帧内**声明的 `Vec` 里（构建与执行都在这个函数体内，
不是之前误判的跨作用域问题），执行期据此选定 `h_src`。

### 已确认工作正常的项（累计）

输入敏感性 ✅（Round 170 修复后）· 上下文整合/注意力 ✅（Round 175，前缀增长→置信度 9.73%→16.65%）·
注意力结构 ✅ · 输出投影 ✅ · 缓冲维度 ✅（Round 172-173 全扫）

**剩余缺陷位于：语义 → 输出 token 的映射**，需要用逐层残差流数值才能继续定位。

