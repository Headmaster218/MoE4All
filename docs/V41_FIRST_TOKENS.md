

---

## Round 98：没能解决，但排除了三个候选路径

### 做了什么

按 Round 97 的结论找 `ro` 的构造点，找到 `13287: let ro = if dyn_replay {`，
在其中 `b.bind(h.logits, ...)` 前加了 engram bind。**仍是 `unbound tensor 3`。**

### 用探针排除的三条路径

| 探针 | 结果 | 结论 |
|---|---|---|
| `ENGRAM_RO_BIND`（在 `ro` 构造里） | **未打印** | `dyn_replay == false`，`ro` 是 `None` |
| `ENGRAM_BOUND id=3`（在 13833 的 bind_future） | **打印了，id 正是 3** | 绑定执行了，但**不生效** |
| `ENGRAM_REAL_BIND`（在 11689 的 bind_future） | **未打印** | 这条也不是执行路径 |

### 最有价值的一条观察

**`ENGRAM_BOUND id=3` 打印了、绑的就是 3，却仍报 unbound。**

这不是"绑错地方"，而是 **`bind_future` 返回的 `bindings` 没有到达 execute**。
它被 `next_stages` / `next_bundle` / `execute_decode_stages` 的某层转手丢掉了。

### 我这一轮的方法问题

**我在同一个 14000 行文件里追了三个不同的 `bind_future`，每个都试了一遍。**
正确的做法应该是**先确认"哪条路径在执行"，再去改那条路径**——而我是一边猜一边改，
每次编译 1 分钟、跑 30 秒，代价很高。

更有效的做法：在 **`resolve()` 报错的那个点**直接把 `bindings` 里已绑的 id 全部打印出来，
一次就能看到 engram 在不在里面。这一轮没想到。

### 当前状态

- 8 个 bind 点都在代码里，编译通过
- `unbound tensor 3` 仍在
- 工作树干净

### 下一步

在 `infr-vulkan/src/adapter.rs:462` 的 `resolve()` 失败处，打印该 `Bindings` 的完整 id 集合。
这一步能直接回答"engram 到底有没有进到执行用的 bindings 里"，
不用再靠追调用链猜。

