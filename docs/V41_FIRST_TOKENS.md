
---

## Round 159：hash 路由这个嫌疑不存在

### 探针不触发是正确行为

我加了个 `INFR_HASH_ID_PROBE` 想验证「hash 路由拿到的 token id 是否被传错」
（如果是，每一层 MoE 都会选错专家 —— 完美对应"语法完好、语义全错"）。
运行后毫无输出。查下去发现：

```rust
pub fn is_hash_moe_layer(&self, il: usize) -> bool {
    il < self.hash_layer_count      // 本 checkpoint 的 hash_layer_count = 0
}
```

**这个 checkpoint 一个 hash 路由层都没有。** 于是 `hash_gather = false`、`tok_ids` 不被声明、
`Op::GatherI32` 从不发射 —— **探针本来就不该触发。**

### 一条被排除的线索

**V4.1 的每一层 MoE 都走普通的学习型 router，没有按 token id 查表的 hash 路由。**
这条嫌疑线关闭。

探针保留在环境变量后面，并在代码里写下原因（免得下一个人再花一轮去查同一个不存在的路径）。

### 状态

问题仍未解决。本轮没有找到新原因，只是关闭了一条**经核实不存在**的路径。

**方法上的教训**：加探针前应先确认该代码路径有被执行的**结构前提**
（这里是 `hash_layer_count > 0`），而不是先编译再跑。

