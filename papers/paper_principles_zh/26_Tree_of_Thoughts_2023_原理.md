# Tree of Thoughts 原理导读

**论文标题**：`Tree of Thoughts: Deliberate Problem Solving with Large Language Models`
**年份**：`2023`
**主题**：`搜索式推理`
**定位**：把单链条思维扩展为树状搜索的概念性代表作。
**论文链接**：[arXiv](https://arxiv.org/abs/2305.10601)

## 1. 代表公式 / 关键表达
- 可抽象为搜索过程：`state -> expand(thoughts) -> evaluate -> prune -> continue`。
- 这里搜索单元不是 token，而是更高层语义片段 `thought`。

## 2. 这篇论文要解决什么问题
- 单条 CoT 和单次采样仍可能陷入局部错误路径。
- 有些任务需要回溯、分支探索和中间状态评估，单链生成能力不足。
- 作者试图把经典搜索思想引入 LLM 推理过程。

## 3. 核心思想
- 把若干步推理组织成一个“thought”节点，而不是逐 token 贪心生成。
- 允许模型扩展多个候选 thought，形成搜索树。
- 结合启发式评估、保留和回溯机制寻找更优解。

## 4. 方法原理拆解
- LLM 同时扮演候选生成器和状态评估器。
- 系统在搜索树中展开多个分支，保留看起来更 promising 的路径继续探索。
- 这与经典 beam search 有相似处，但单位是语义 thought 而非 token。

## 5. 代码实现结构
- Tree of Thoughts 的实现重点是 `搜索控制器`，而不是模型结构本身。
- 代码结构通常拆成 `State Representation`、`Thought Generator`、`State Evaluator`、`Search Policy` 四部分。
- 从工程上看，它是把 LLM 包装进一个显式搜索框架，而不再只是单次 prompt 生成。

### 5.1 State Representation
- 每个搜索节点需要表示“当前已经走到哪一步”。
- 这通常由原问题加上当前 thought 序列组成。
- 因此代码里常见 `state = {question, thoughts_so_far}` 这样的结构。

### 5.2 Thought Generator
- 给定当前状态，模型生成若干候选 thought。
- 这些 thought 不是最终答案，而是中间解题动作或局部推理片段。
- 实现上常是一次 prompt 请求返回多个候选，或重复多次采样。

### 5.3 State Evaluator
- 对新扩展出的 thought 进行打分，判断哪些分支值得保留。
- 评估器可以仍然是 LLM 本身，也可以是规则或外部打分器。
- 这一步决定搜索是否高效，很多性能差异都出在这里。

### 5.4 Search Policy
- 搜索策略负责控制展开、剪枝、回溯和停止条件。
- 最简单的是 beam-style 保留 top-k 分支。
- 更复杂的实现可以模拟 DFS、BFS 或启发式搜索。

### 5.5 一个最小实现骨架

```python
def tree_of_thoughts(model, question, max_depth=3, beam_width=3):
    states = [{"question": question, "thoughts": []}]
    for _ in range(max_depth):
        candidates = []
        for state in states:
            thoughts = generate_thoughts(model, state)
            for thought in thoughts:
                new_state = {"question": question, "thoughts": state["thoughts"] + [thought]}
                score = evaluate_state(model, new_state)
                candidates.append((score, new_state))
        states = [s for _, s in sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]]
    return choose_best_answer(model, states)
```

### 5.6 实现时最值得注意的点
- 真正难点是状态评估是否稳定，否则搜索树会被错误打分带偏。
- 搜索深度和分支数稍微增大，计算成本就会迅速爆炸。
- 工程上通常需要很严格的 stopping rule 和分支剪枝策略。

## 6. 训练或推理范式
- 核心发生在推理阶段，不要求专门训练。
- 代价是计算量显著提升，尤其当分支和深度都增加时。
- 它更像是为复杂推理任务提供一个 deliberative inference 框架。

## 7. 为什么它有效
- 复杂问题往往需要试错、回退和多方案比较，树状搜索更符合这一点。
- 把推理粒度提升到 thought 有助于更高层规划。
- 显式探索多条路径能减少早期单点错误对全局的毁灭性影响。

## 8. 局限与争议
- 推理成本高，且搜索策略本身需要精心调节。
- LLM 作为评估器未必稳定，可能误判分支优劣。
- 工程上很多场景更偏好更轻量的 self-consistency 或 verifier 方法。

## 9. 对后续研究的影响
- 定义了“搜索式推理”这类问题的公共语言。
- 对后续 deliberation、test-time compute 和规划型推理研究影响很大。
- 虽然不一定总是最实用，但概念贡献非常强。

## 10. 你应该怎样读这篇
- 读这篇不要问“线上能不能直接用”，而要问“它重新定义了什么”。
- 它是 reasoning 研究中的概念路标论文。
- 建议和 Self-Consistency 对照阅读，一个是多样本投票，一个是显式搜索。

## 11. 前置阅读
- `Chain-of-Thought`
- `Self-Consistency`

## 12. 读完接着看
- `DeepSeek-R1`
