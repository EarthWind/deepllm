# Constitutional AI 原理导读

**论文标题**：`Constitutional AI: Harmlessness from AI Feedback`
**年份**：`2022/2023`
**主题**：`对齐 / AI 反馈`
**定位**：尝试用原则集与 AI 反馈替代大量人工偏好标注的重要论文。
**论文链接**：[arXiv](https://arxiv.org/abs/2212.08073)

## 1. 代表公式 / 关键表达
- 可抽象为：`y' = revise(y, constitution)`，先生成，再依据原则修订。
- 偏好学习阶段继续围绕“更符合原则的回答”进行优化。

## 2. 这篇论文要解决什么问题
- RLHF 有效，但对人工标注依赖高、成本大、扩展性有限。
- 如果想更系统地约束模型行为，需要一种更可扩展的监督来源。
- 作者尝试回答：能否用一套书面原则指导模型自我批判和自我改写。

## 3. 核心思想
- 预先给定一套“宪法”式原则，如安全、无害、诚实等。
- 让模型先生成回答，再根据原则自我批判并修订。
- 再利用这些 AI 反馈形成偏好信号，继续训练模型。

## 4. 方法原理拆解
- 关键思想是把高层规范显式化，而不是只让标注员隐式地表达喜好。
- 模型通过原则驱动的自评与改写，学习更可控的行为模式。
- 这种方式把对齐过程部分程序化和可审计化。

## 5. 代码实现结构
- Constitutional AI 的实现重点不在 backbone，而在 `生成 -> 自评 -> 修订 -> 再训练` 的对齐流水线。
- 从工程结构看，通常可以拆成 `Constitution Rule Set`、`Critique Generator`、`Revision Generator`、`Preference / SFT Trainer` 四部分。
- 它更像“自动化对齐数据工厂 + 后训练系统”，而不是新网络层。

### 5.1 Constitution Rule Set
- 首先需要把高层原则明确写成一组可读取的规则文本。
- 代码上这通常只是一个 rules 列表或配置文件。
- 但它在系统里的作用非常大，因为后续批判和修订都围绕这些规则展开。

### 5.2 Critique Generator
- 模型先生成初始回答。
- 然后第二次调用模型，要求它基于宪法规则批判前一个回答。
- 这一步在代码上经常表现为“同一个模型，不同 prompt 角色”的两次调用。

### 5.3 Revision Generator
- 有了批判文本后，再要求模型根据批判意见重写回答。
- 这样就形成了 `answer -> critique -> revised_answer` 的链条。
- 这些修订前后的样本，后续可以被用来做监督或偏好训练。

### 5.4 一个最小数据生成骨架

```python
def constitutional_rewrite(model, question, constitution):
    answer = model.generate(build_answer_prompt(question))
    critique = model.generate(build_critique_prompt(question, answer, constitution))
    revised = model.generate(build_revision_prompt(question, answer, critique, constitution))
    return {"answer": answer, "critique": critique, "revised": revised}
```

### 5.5 后训练接法
- 生成好的 `revised answer` 可以直接当 SFT 数据。
- 也可以把原始回答和修订回答组成 preference pair，再做偏好优化。
- 因此 Constitutional AI 本质上是在构造一种“带原则约束的训练信号”。

### 5.6 实现时最值得注意的点
- 难点不在生成三段文本，而在如何让批判真正对应规则，而不是泛泛而谈。
- 如果宪法原则写得太空泛，模型容易输出模板化自我批评，训练价值有限。
- 这类系统的工程核心是 prompt 设计、样本过滤和批量数据生成质量控制。

## 6. 训练或推理范式
- 训练流程结合了监督数据、AI 自我反馈和偏好优化。
- 与纯人工 RLHF 相比，它试图减少人工比较数据需求。
- 同时它也让对齐目标更容易被说明和修改。

## 7. 为什么它有效
- 高层规范如果写得清楚，可以在一定程度上替代大量零碎偏好标注。
- 模型自我批判能生成更多训练样本，提高数据扩展性。
- 原则驱动比纯偏好打分更便于审查和迭代。

## 8. 局限与争议
- 原则本身仍然由人制定，无法完全避免价值选择问题。
- 模型的自我批判质量受其自身能力限制。
- “更无害”不等于“更真实”，不同目标之间仍可能冲突。

## 9. 对后续研究的影响
- 推动 RLAIF、AI feedback 和自动化对齐成为重要路线。
- 补充了 InstructGPT 之后对齐方法的技术图谱。
- 让“对齐规则显式化”成为可实践的工程方向。

## 10. 你应该怎样读这篇
- 读这篇时要关注它如何把规范转成训练信号。
- 如果你关心安全对齐，这篇比单纯看聊天效果更有价值。
- 把它和 DPO 对照看，能看出“监督来源”和“优化方法”是两个不同维度。

## 11. 前置阅读
- `InstructGPT`

## 12. 读完接着看
- `DPO`
- `DeepSeek-R1`
