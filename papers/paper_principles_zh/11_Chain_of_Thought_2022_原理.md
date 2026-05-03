# Chain-of-Thought 原理导读

**论文标题**：`Chain-of-Thought Prompting Elicits Reasoning in Large Language Models`
**年份**：`2022`
**主题**：`推理提示`
**定位**：把“让模型写出中间推理过程”正式推上主舞台的里程碑论文。
**论文链接**：[arXiv](https://arxiv.org/abs/2201.11903)

## 1. 代表公式 / 关键表达
- 可把输出分解为：`p(y, r | x)`，其中 `r` 是中间推理链，`y` 是最终答案。
- CoT 的核心不在改模型，而在显式引入中间变量 `r` 来帮助求解。

## 2. 这篇论文要解决什么问题
- 很多复杂任务不是知识不足，而是模型不会稳定地展开多步推理。
- 直接要求最终答案时，模型容易跳步、犯算术错误或遗漏逻辑约束。
- 作者探索：如果给出带中间步骤的示例，能否显著提升推理表现。

## 3. 核心思想
- 在 few-shot prompt 中加入完整推理链，而不只给问题和答案。
- 让模型模仿“先推理，再给结论”的输出模式。
- 观察模型规模与 CoT 效果之间的关系，发现大模型受益更明显。

## 4. 方法原理拆解
- CoT 本质上不是改变参数，而是改变解题轨迹的展开方式。
- 中间步骤帮助模型把复杂问题分解成更小的局部决策，降低一次性直接预测最终答案的难度。
- 提示中的推理链还给模型示范了任务所需的思考形式。

## 5. 代码实现结构
- CoT 不是新模型结构，而是一层 `prompt orchestration` 和 `输出解析` 逻辑。
- 因此代码实现重点不在改 `Transformer`，而在 `prompt builder`、`reasoning parser`、`answer extractor` 三部分。
- 从工程视角看，它更像一个推理策略层，插在调用大模型 API 或本地模型推理接口之上。

### 5.1 Prompt Builder
- 输入问题后，先构造带有推理示例的 prompt。
- 样例通常形如：`问题 -> 推理过程 -> 最终答案`。
- 代码上最常见的是把 few-shot exemplars 存成模板列表，再和当前问题拼接。

### 5.2 推理调用层
- 模型本身仍是普通语言模型，调用接口没有变化。
- 关键差别是 prompt 中显式要求“逐步思考”或提供了多步推理示例。
- 因此实现核心仍是：
  `response = model.generate(prompt)`

### 5.3 输出解析层
- 由于 CoT 输出通常包含中间推理和最终答案，代码里往往要把两者分开。
- 常见做法是约定答案格式，例如最后一行以 `Answer:` 开头。
- 如果不做格式约束，后续自动评测和投票会变得很不稳定。

### 5.4 一个最小实现骨架

```python
def build_cot_prompt(question, exemplars):
    prompt = []
    for ex in exemplars:
        prompt.append(f"Q: {ex['q']}\nReasoning: {ex['r']}\nAnswer: {ex['a']}\n")
    prompt.append(f"Q: {question}\nReasoning:")
    return "\n".join(prompt)


def run_cot(model, question, exemplars):
    prompt = build_cot_prompt(question, exemplars)
    text = model.generate(prompt)
    reasoning, answer = parse_reasoning_and_answer(text)
    return {"reasoning": reasoning, "answer": answer}
```

### 5.5 如果做工程化 CoT
- 你通常还会增加 `answer extractor`、`validator`、`fallback prompt`。
- 因为 CoT 很容易输出过长、跑偏或不按格式收尾。
- 所以工程上的 CoT 往往是“提示模板 + 解析规则 + 错误重试”的组合。

### 5.6 实现时最值得注意的点
- CoT 的难点不是 forward，而是 prompt 设计与输出约束。
- 对自动化流水线来说，最终答案提取规则几乎和 prompt 本身一样重要。
- 如果后面要接 self-consistency，多条 CoT 的答案必须能被稳定归一化比较。

## 6. 训练或推理范式
- 这篇是推理时技巧，不需要重新训练模型。
- 但它启发了后续大量数据合成、过程监督、思维链蒸馏和 reasoning model 训练路线。
- 也就是说，最初是 prompt 技巧，后来演变成训练范式。

## 7. 为什么它有效
- 复杂推理本身有中间结构，中间文本让模型更容易显式表示这些结构。
- 大模型具备潜在推理能力，但需要合适的输出轨迹去激活。
- 显式步骤可以帮助局部纠错，并提升任务可解释性。

## 8. 局限与争议
- 并不是所有任务都适合长 CoT，简单任务可能反而被噪声拖累。
- 外显思维链不等于真实内部机制，解释性不能过度夸大。
- 单条 CoT 仍可能走偏，因此后来出现 self-consistency、verifier 等方法。

## 9. 对后续研究的影响
- 开启了 reasoning model 的主线叙事。
- 影响了 Self-Consistency、Tree of Thoughts、Step Verification 和 R1 等大量后续工作。
- 也改变了人们对“提示词”的理解：它不只是命令，更是思维轨迹模板。

## 10. 你应该怎样读这篇
- 读这篇时要抓住一个核心：推理不只是答案，还是过程设计。
- 建议和 Self-Consistency 连读，理解单链条与多链条推理策略的区别。
- 如果你做数学、代码、规划任务，这篇是必读起点。

## 11. 前置阅读
- `GPT-3`

## 12. 读完接着看
- `Self-Consistency`
- `Let's Verify Step by Step`
- `DeepSeek-R1`
