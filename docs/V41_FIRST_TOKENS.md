

---

## Round 99：拿到了决定性数据，但代码改乱了

### 决定性证据：bound_ids 里独独缺 3

在 `resolve()` 报错处打印该 `Bindings` 的完整 id 集合：

```
ENGRAM_RESOLVE_FAIL id=3
bound_ids=[0, 2, 4, 5, 6, 7, 8, ... 1163]
```

**有 0、有 2，独独没有 3。**

这彻底排除了"绑错对象"和"绑定被覆盖"，**是 engram 的 input 根本没进这份 bindings**。

### 顺着调用链找到 b1，但不是它

```
execute_decode_stages(main.bindings)
  ← execute_decode(main, ...)
  ← ExecutionStage::new(plan1, &b1)     runner.rs:14058
  ← let mut b1 = Bindings::new()         runner.rs:13976
```

在 `b1` 上补了 engram bind，**`bound_ids` 仍然缺 3**。

### 但探针给出了新线索

在 `bind_layer_io` 函数体内打印：

```
ENGRAM_BIND_LAYER_IO engram_holder=Some(TensorId(3))   ×2
```

**`bind_layer_io` 只被调用两次，handle 是对的** —— 但它**本身不负责 engram**。
这就是漏洞：`bind_layer_io` 是两条实际路径共用的绑定函数，engram 却不在里面。

### 然后我把代码改乱了

想让 `bind_layer_io` 内部绑 engram，需要给它加参数。它和 `bind_parallel_layer_io`
加起来有 **18 个调用点**。我用脚本批量插入，但：

- 第一次插到参数列表末尾 → 顺序错
- 第二次按 `primary_ple_state,` 定位 → 只匹配到 1 处（调用点里参数名各不相同）

**已经 `git checkout` 全部回退**，工作树干净、编译通过。

### 下一轮的正确做法

**不要用脚本批量改 18 个调用点。** 改为：
1. 在 `bind_layer_io` 里加参数（1 处签名）
2. 用编译器逐个报出调用点，**手工**补（每处 1 行）
3. 只有 18 处，手工比脚本出错少

**而且这次有明确的验证标准**：`bound_ids` 里是否出现 3。

