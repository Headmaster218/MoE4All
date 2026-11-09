

---

## Round 113：第一次端到端验证 —— 发现 V4.1 不走我以为的那条路径

### 三件事逐一定实

| 检查 | 结果 |
|---|---|
| 表是否加载 | **✅ `ENGRAM_LOADED` 打印了**（`deepseek41=true`、`layer_ids=[1,14]`）|
| upload 是否执行 | **❌ `ENGRAM_UPLOAD` 没打印** |
| `bind_layer_io` 是否被调用 | **❌ backtrace 探针也没打印** |

### 关键矛盾

**模型能跑完 prefill 35 tok、decode 2 tok，却既没走我插入 upload 的分支，
也没走 `bind_layer_io`。**

**那 engram 的绑定是怎么完成的？** 答案只能是：**根本没绑定，也没上传**——
而模型依然能跑，因为 `engram_emb` 是可选 input，未绑定不报错（Round 100 之前是报 unbound 的，
后来加了绑定才不报）。

**换句话说：现在跑的这条路径上，engram 完全不参与。**

### 两次错误判断

| 轮次 | 我以为 | 实际 |
|---|---|---|
| Round 112 | `10399` 的 `bind_parallel_layer_io` 是 V4.1 路径 | **没被调用** |
| 本轮 | 表加载了就会生效 | **加载了但不被使用** |

**共同点：我又一次靠"这个分支看起来对"来推断，而不是靠运行时证据。**

### 检查方法也错了一次

我用 `grep -c engram-mxfp8 /proc/PID/maps` 判断表是否映射，得到 0。
**这个方法是错的**——202 GB 的 mmap 在 `/proc/maps` 里的条目形式与我的 grep 不匹配，
且进程可能已退出。改用 `eprintln!` 后才拿到真答案。

### 正确的下一步

**不要再猜分支。** 直接在 `engram_apply` 闭包（发射 engram 计算的地方）里打 backtrace，
它会打印出**真正执行的那条路径**——Round 95 用过这个方法，一次就定位到
`prepare_decode_plan`。

### 状态

| 项 | 值 |
|---|---|
| 工作树 | 干净（诊断已回退）|
| 编译 | ✅ |
| 本地提交 | **84 个** |
| 运行 | 能跑、能出 token、**engram 未参与** |

