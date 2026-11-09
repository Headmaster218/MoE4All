
---

## Round 203：rebase 后重新验证 —— 修复的实测数据仍然成立

git 历史被 rebase 到 `upstream/main` 后（PR #199），重跑同一条验证：

```
./infr run <model> "Hi" --max-new 1 --temp 0 --ctx 512 --dev Vulkan1
  with INFR_LOGIT_PROBE=1 INFR_NO_GPU_ARGMAX=1 INFR_NO_GPU_SAMPLE=1

top5 = [343:0.1718, 16:0.0876, 33:0.0575, 201:0.0390, 29:0.0268]
max = 15.6292   entropy = 5.391
```

**与 rebase 前逐位相同。** PR #199 里引用的 6.31% → 17.18% 是可信的。

### 为什么值得记这一笔

rebase 会重放 229 个补丁，理论上任何一处冲突解决都可能改变行为。**在把数据写进 PR 之前重跑一遍，
是让 PR 里每个数字都对得上它声称的树。** 这次对上了。

### 一个仍需注意的坑：`INFR_PROMPT_OVERRIDE` 的正确用法

对比两个引擎时必须保证 token 逐个相同。正确流程：

```bash
# 1. 让 infr 把你的问题渲染成完整 prompt 并落盘
INFR_PROMPT_DUMP=1 infr run <model> "<question>" --max-new 1 ...
cat /tmp/turn-prompt.txt

# 2. 用同一个 tokenizer 编码，确认 ids
llama-tokenize -m <model.gguf> -f /tmp/turn-prompt.txt --ids

# 3. 把同一份文件喂给 infr（绕过模板），同一组 ids 喂给参考引擎
INFR_PROMPT_OVERRIDE=/tmp/turn-prompt.txt infr run <model> x ...
strata-dsv41 ... --ids "<上一步的 ids>"
```

**为什么不能省这一步**：infr 的模板默认在 `<|Assistant|>` 后追加 `<|think>`，`--no-think` 追加的是
`</think>` 而不是空——两者都不是参考引擎拿到的干净结尾。**本会话有 10 次以上因为没做这一步
而拿"不同输入"做了对比、得出了错误结论。** 这个 flag 就是把那个坑填掉的。

