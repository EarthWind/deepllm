# 2017 年后 Transformer / LLM 必读论文导读

这份导读默认从 `Attention Is All You Need` 发布之后开始统计，不重复介绍该论文本身。目标不是机械罗列“所有论文”，而是筛出到今天仍然最值得通读原文、并且能串起 Transformer 到 LLM 主线演进的关键论文。

筛选标准：
- 对当前大模型主线仍有解释力
- 对后续研究和工程实践影响足够大
- 适合作为系统学习的骨架论文

阅读建议：
- 如果你是第一次系统梳理，先读“第一优先级”
- 如果你已经做过训练、微调或应用开发，再读“第二优先级”
- 如果你有明确方向，例如 RAG、多模态、推理模型、效率优化，再读“专题补充”

---

## 一、第一优先级：先吃透这 15 篇

这 15 篇基本定义了今天 LLM 的主线：预训练、扩展律、对齐、推理、效率与开源复现。

### 1. BERT (2018)
**论文**：`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`

**为什么重要**
- 它把 Transformer 真正送进 NLP 主流。
- 它确立了“先大规模预训练，再针对下游任务微调”的标准范式。
- 它让双向上下文建模和掩码语言模型成为核心方法论。

**看什么**
- MLM 和 NSP 的设计动机
- 为什么双向编码器适合理解任务
- BERT 为什么在当时几乎横扫 NLP benchmark

**今天怎么看**
- BERT 不是今天生成式 LLM 的直接祖先，但它是现代 Transformer 预训练时代真正的开端。
- 如果不理解 BERT，你会很难理解 GPT 路线为什么后来赢，以及 encoder-only / decoder-only / encoder-decoder 为什么会分化。

### 2. GPT (2018)
**论文**：`Improving Language Understanding by Generative Pre-Training`

**为什么重要**
- 这是 GPT 路线的起点。
- 它提出：只做从左到右的生成式语言建模，也能迁移到多类任务。
- 它为后面的 GPT-2、GPT-3、ChatGPT 奠定范式基础。

**看什么**
- 单向语言建模和任务适配方式
- 生成式预训练相对 BERT 式预训练的差异
- 为什么这个方向在早期还没有完全压倒 BERT

**今天怎么看**
- 这篇本身规模不大，但范式意义极强，是 GPT 系谱必须补的一篇。

### 3. GPT-2 (2019)
**论文**：`Language Models are Unsupervised Multitask Learners`

**为什么重要**
- GPT-2 让“语言模型本身就是通用任务接口”这件事第一次被广泛认真对待。
- 零样本、多任务、自然语言驱动行为开始变得可信。
- “模型越大，能力越通用”的趋势开始显现。

**看什么**
- 零样本任务表现
- 模型规模和泛化能力的关系
- 论文里对生成质量和泛化趋势的判断

**今天怎么看**
- GPT-2 不是最强里程碑，但它是从“研究原型”走向“通用模型”的关键一跳。

### 4. T5 (2019/2020)
**论文**：`Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`

**为什么重要**
- 它把几乎所有 NLP 任务统一成 text-to-text。
- 它给出一个极强的统一任务表达框架。
- 后来的 instruction tuning、prompt 化任务接口，都能从这里找到思想源头。

**看什么**
- 为什么要把所有任务都写成“输入文本 -> 输出文本”
- encoder-decoder 架构在统一任务接口上的优势
- 预训练目标和任务格式统一的收益

**今天怎么看**
- T5 不一定是今天主流聊天模型的核心架构，但它是“统一接口思想”的代表作。

### 5. GPT-3 (2020)
**论文**：`Language Models are Few-Shot Learners`

**为什么重要**
- 这是现代大模型时代真正的爆点之一。
- in-context learning 从现象变成研究对象。
- “不用梯度更新，只靠 prompt 和示例就能做任务”开始震动整个领域。

**看什么**
- few-shot / one-shot / zero-shot 的实验设计
- 上下文示例为什么能替代部分微调
- 规模增长如何改变能力边界

**今天怎么看**
- GPT-3 是理解 prompt engineering、上下文学习和大模型能力跃迁的必读论文。

### 6. Scaling Laws (2020)
**论文**：`Scaling Laws for Neural Language Models`

**为什么重要**
- 它首次系统量化了参数量、数据量、计算量与性能之间的关系。
- 它把“大模型训练”从经验驱动拉向规律驱动。
- 后面几乎所有训练预算决策都在回应这篇论文。

**看什么**
- loss 与模型规模、数据规模、计算量的幂律关系
- 为什么只一味增大参数不一定最优
- 这篇论文对训练预算规划的启发

**今天怎么看**
- 这是理解大模型训练经济学和扩展逻辑的核心论文。

### 7. RAG (2020)
**论文**：`Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks`

**为什么重要**
- 它正式确立了“参数知识 + 外部检索”的主线。
- 它把生成模型和检索系统接到了一起。
- 今天企业里绝大部分知识库问答、文档助手、私有化问答，思想源头都能追到这里。

**看什么**
- parametric memory 与 non-parametric memory 的分工
- 检索结果如何参与生成
- RAG 为什么比纯参数记忆更适合知识密集型任务

**今天怎么看**
- 如果你做应用，RAG 不是补充知识，而是主线必修课。

### 8. FLAN (2021)
**论文**：`Finetuned Language Models Are Zero-Shot Learners`

**为什么重要**
- 这篇把 instruction tuning 从零散技巧变成明确范式。
- 它证明：在大量指令任务上做微调，能显著改善零样本泛化。
- 后来的 instruction-following 模型基本都受它影响。

**看什么**
- 指令数据为何能提升泛化
- 多任务指令微调与传统监督微调的差别
- 为什么模型会“更像在听人话”

**今天怎么看**
- FLAN 是从“会续写”走向“会按要求办事”的关键桥梁。

### 9. RoFormer / RoPE (2021)
**论文**：`RoFormer: Enhanced Transformer with Rotary Position Embedding`

**为什么重要**
- 它提出的 RoPE 成为今天大多数开源 LLM 的默认位置编码方案。
- 长上下文外推、注意力中的相对位置信息表达，很多实现都建立在此基础上。

**看什么**
- 旋转位置编码的数学直觉
- 为什么它比传统绝对位置编码更适合 LLM
- 它对后续长上下文扩展的价值

**今天怎么看**
- 工程影响力极大，属于“论文不一定最常被讨论，但实现里天天都在用”的类型。

### 10. InstructGPT (2022)
**论文**：`Training language models to follow instructions with human feedback`

**为什么重要**
- 它把 RLHF 推成了主流对齐路线。
- 它解释了为什么“更小但更对齐”的模型，可以比“更大但未对齐”的模型更好用。
- ChatGPT 背后的关键技术脉络在这篇里最清楚。

**看什么**
- SFT、reward model、PPO 三阶段流程
- helpful / honest / harmless 这类对齐目标
- 对齐与纯语言建模目标之间的关系

**今天怎么看**
- 如果你只读一篇“为什么聊天模型会出现”的论文，就是它。

### 11. Chain-of-Thought (2022)
**论文**：`Chain-of-Thought Prompting Elicits Reasoning in Large Language Models`

**为什么重要**
- 它明确展示了：让模型写出中间推理步骤，会显著提升复杂任务表现。
- 现代 reasoning model 的很多外显形式，都能回溯到这篇。
- 它把“推理不仅是答案，更是过程”这个观念推到前台。

**看什么**
- few-shot CoT 的提示形式
- 为什么在足够大模型上效果突然变好
- 哪类任务最受益于中间推理链

**今天怎么看**
- 这是推理模型时代最基础的一篇起点论文。

### 12. Chinchilla (2022)
**论文**：`Training Compute-Optimal Large Language Models`

**为什么重要**
- 它修正了早期“优先堆参数”的粗放思路。
- 它指出在固定算力下，参数量和数据量应更均衡匹配。
- 这篇对实际训练预算和数据配比有极强指导意义。

**看什么**
- 为什么很多模型其实“数据不够吃”
- compute-optimal 的含义是什么
- 它如何修正前一阶段的 scaling 直觉

**今天怎么看**
- 理解这篇后，你会更清楚为什么高质量数据在大模型时代不是配角。

### 13. PaLM (2022)
**论文**：`PaLM: Scaling Language Modeling with Pathways`

**为什么重要**
- 它代表了超大规模训练时代的一个高点。
- 它系统展示了规模提升对推理、代码、知识和多任务能力的影响。
- 很多“能力涌现”讨论，都是围绕这代论文展开的。

**看什么**
- 大规模模型在多类 benchmark 上的变化
- chain-of-thought 与大模型规模结合后的效果
- 论文如何讨论能力跃迁与局限

**今天怎么看**
- PaLM 是理解“超大模型为何看起来突然变聪明”的代表论文之一。

### 14. FlashAttention (2022)
**论文**：`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`

**为什么重要**
- 它不是“换模型”，而是从系统层重做 attention 计算。
- 它极大提升了训练和推理效率。
- 后续大模型基础设施几乎绕不开 FlashAttention 系列。

**看什么**
- 为什么 attention 的瓶颈不只是 FLOPs，而是 IO
- 它如何在保持精确 attention 的同时减少显存读写
- 为什么工程实现能反过来影响研究方向

**今天怎么看**
- 如果你做训练系统、推理部署、长上下文优化，这篇是工程必读。

### 15. LLaMA (2023)
**论文**：`LLaMA: Open and Efficient Foundation Language Models`

**为什么重要**
- 它开启了开源 LLM 的大爆发。
- 它证明相对更小但数据更优、训练更扎实的模型，可以非常强。
- 后续 Mistral、Qwen、Yi、DeepSeek 等开源路线都受其深刻影响。

**看什么**
- 数据配比与训练策略
- 为什么较小参数量也能获得强能力
- 这篇对开源生态的历史作用

**今天怎么看**
- LLaMA 之后，研究和产业不再只能围着闭源模型转。

---

## 二、第二优先级：已经入门后再读的关键论文

### 16. Switch Transformer (2021)
**论文**：`Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity`

**导读**
- 这是现代 MoE 路线的核心起点之一。
- 它的核心思想是：不是每个 token 都激活全部参数，而是只走部分专家。
- 如果你想理解为什么今天很多前沿开源模型是 MoE，这是必读。

### 17. LoRA (2021/2022)
**论文**：`LoRA: Low-Rank Adaptation of Large Language Models`

**导读**
- 它让“低成本微调大模型”成为现实。
- 核心思想是只训练低秩增量，而不全量改动原模型。
- 今天 PEFT 生态几乎都以它为基准参照。

### 18. Self-Consistency (2022)
**论文**：`Self-Consistency Improves Chain of Thought Reasoning in Language Models`

**导读**
- 不只生成一条推理链，而是采样多条，再投票。
- 它体现了“推理能力不只靠训练，也靠推理时策略”的思想。
- 是 test-time compute 思路的重要前奏。

### 19. Constitutional AI (2022/2023)
**论文**：`Constitutional AI: Harmlessness from AI Feedback`

**导读**
- 这篇提出了用“原则集”替代大量人工偏好标注的一条对齐路线。
- 对理解 RLAIF、自动对齐、自我批判式生成很有帮助。
- 它补足了 InstructGPT 之后的对齐路线图。

### 20. Toolformer (2023)
**论文**：`Toolformer: Language Models Can Teach Themselves to Use Tools`

**导读**
- 这篇展示了模型可以学习何时调用外部工具。
- 它是“工具使用是模型能力的一部分”这条路线的重要早期作品。
- 对 agent 和函数调用理解很关键。

### 21. ReAct (2023)
**论文**：`ReAct: Synergizing Reasoning and Acting in Language Models`

**导读**
- 思考与行动交替进行，是 agent 领域最经典的范式之一。
- 它让“推理过程驱动工具调用”变得结构化。
- 今天很多 agent 框架本质上都还能看到 ReAct 的影子。

### 22. Self-Instruct (2023)
**论文**：`Self-Instruct: Aligning Language Models with Self-Generated Instructions`

**导读**
- 这篇让“模型合成指令数据，再反哺自己”成为一条成熟路线。
- 后来大量指令数据工程、蒸馏数据构造都能追溯到这里。
- 对数据生成和低成本对齐很重要。

### 23. DPO (2023)
**论文**：`Direct Preference Optimization: Your Language Model is Secretly a Reward Model`

**导读**
- 它提出无需显式 reward model + PPO，也能做偏好优化。
- 训练更简单，效果又强，因此迅速成为主流后训练方法之一。
- 如果你关心 alignment / preference tuning，这是必读。

### 24. QLoRA (2023)
**论文**：`QLoRA: Efficient Finetuning of Quantized LLMs`

**导读**
- 在量化模型上做高质量微调，大幅降低显存门槛。
- 它让个人和中小团队更容易参与大模型训练与适配。
- 是工程实践里最有落地价值的论文之一。

### 25. Let’s Verify Step by Step (2023)
**论文**：`Let's Verify Step by Step`

**导读**
- 它强调：不要只奖励最终答案，也要监督中间推理步骤。
- 这篇对 process supervision、verifier、推理模型训练影响很大。
- 如果你想理解为什么 reasoning model 不只是“长 CoT”，这篇非常关键。

### 26. Tree of Thoughts (2023)
**论文**：`Tree of Thoughts: Deliberate Problem Solving with Large Language Models`

**导读**
- 它把推理从单链条扩展到树状搜索。
- 工程上未必处处最优，但概念上极其重要。
- 很多“搜索式推理”“审议式推理”讨论，都受它影响。

### 27. Mixtral (2024)
**论文**：`Mixtral of Experts`

**导读**
- 这是开源 MoE 真正进入高实用性的代表作之一。
- 它让大家看到：开源模型也能在稀疏架构上做到非常强的效果。
- 对理解 2024 之后开源模型生态很重要。

### 28. Gemini 技术报告 (2023)
**论文**：`Gemini: A Family of Highly Capable Multimodal Models`

**导读**
- 这是多模态统一模型的重要代表。
- 它不只是“图文拼接”，而是体现统一多模态训练和能力组织的方向。
- 对理解未来基础模型为什么会天然多模态，很有帮助。

### 29. Mamba (2023)
**论文**：`Mamba: Linear-Time Sequence Modeling with Selective State Spaces`

**导读**
- 它不是 Transformer 主线内部优化，而是强有力的替代路线。
- 看这篇的价值在于理解：为什么大家持续寻找 attention 之外的序列建模方式。
- 它帮助你建立“后 Transformer 时代可能怎么走”的视角。

### 30. DeepSeek-R1 (2025)
**论文**：`DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning`

**导读**
- 这是开源 reasoning model 爆发的代表论文。
- 它把强化学习、长推理链、推理时计算这几件事真正推向大众关注中心。
- 如果你想理解 2025 年后 reasoning model 为什么成为主线，这篇必须读。

---

## 三、按主题补充：你有明确方向时再扩展

### A. 如果你重点看 RAG / 检索增强
- `REALM (2020)`：更早期地把检索纳入预训练/知识增强视角。
- `RAG (2020)`：主线起点，必须先读。
- `RETRO (2021)`：把海量外部语料检索与生成进一步深度绑定。

**导读建议**
- 先搞清楚：模型参数不是唯一知识载体。
- 再理解：检索是在训练时接入、推理时接入，还是两者都接入。
- 最后再看今天工程里的 chunk、embedding、rerank、agentic RAG，这样不容易只会调框架。

### B. 如果你重点看长上下文
- `Transformer-XL (2019)`：跨段记忆与长依赖问题的重要起点。
- `Longformer (2020)`：稀疏 attention 的经典代表。
- `ALiBi (2021)`：极简位置偏置方案，对长度外推讨论影响很大。
- `RoFormer / RoPE (2021)`：今天 LLM 最常用的位置编码。
- `FlashAttention (2022)`：长上下文工程效率核心。

**导读建议**
- 长上下文的难点不只有“能塞进去”，还包括训练稳定性、位置编码、注意力复杂度、推理显存。

### C. 如果你重点看 Agent / 工具调用
- `Toolformer (2023)`
- `ReAct (2023)`
- `Tree of Thoughts (2023)`

**导读建议**
- 先看“何时调工具”，再看“如何边想边做”，最后看“如何搜索多条推理路径”。

### D. 如果你重点看多模态
- `Flamingo (2022)`：少样本视觉语言模型的重要里程碑。
- `BLIP-2 (2023)`：高效连接视觉编码器和 LLM 的经典设计。
- `LLaVA (2023)`：开源多模态对话模型爆发起点之一。
- `Gemini (2023)`：统一多模态基础模型代表。

**导读建议**
- 多模态不只是“加图片输入”，关键是模态对齐、连接器设计、指令微调和推理能力迁移。

### E. 如果你重点看后训练与对齐
- `InstructGPT (2022)`
- `Constitutional AI (2022/2023)`
- `DPO (2023)`
- `Let’s Verify Step by Step (2023)`

**导读建议**
- 这条线最关键的问题不是“让模型更会说”，而是“让模型更像可用产品”。

---

## 四、最推荐的阅读顺序

### 路线 1：按历史主线读
1. `BERT`
2. `GPT`
3. `GPT-2`
4. `T5`
5. `GPT-3`
6. `Scaling Laws`
7. `RAG`
8. `FLAN`
9. `RoFormer / RoPE`
10. `InstructGPT`
11. `Chain-of-Thought`
12. `Chinchilla`
13. `PaLM`
14. `FlashAttention`
15. `LLaMA`
16. `DPO`
17. `DeepSeek-R1`

### 路线 2：如果你做应用开发
1. `GPT-3`
2. `InstructGPT`
3. `FLAN`
4. `RAG`
5. `ReAct`
6. `Toolformer`
7. `DPO`
8. `QLoRA`
9. `LLaMA`
10. `DeepSeek-R1`

### 路线 3：如果你做训练或微调
1. `Scaling Laws`
2. `Chinchilla`
3. `T5`
4. `InstructGPT`
5. `LoRA`
6. `QLoRA`
7. `DPO`
8. `Let’s Verify Step by Step`
9. `Switch Transformer`
10. `Mixtral`

### 路线 4：如果你做推理模型
1. `Chain-of-Thought`
2. `Self-Consistency`
3. `Tree of Thoughts`
4. `Let’s Verify Step by Step`
5. `DeepSeek-R1`

---

## 五、如果时间有限，怎么取舍

### 只读 5 篇
- `GPT-3`
- `InstructGPT`
- `Chinchilla`
- `Chain-of-Thought`
- `LLaMA`

**适合谁**
- 只想快速建立现代 LLM 主线认知的人。

### 只读 10 篇
- `BERT`
- `GPT-2`
- `T5`
- `GPT-3`
- `Scaling Laws`
- `InstructGPT`
- `Chinchilla`
- `Chain-of-Thought`
- `FlashAttention`
- `LLaMA`

**适合谁**
- 想兼顾理论、工程和应用视角的人。

### 只读“应用最相关”8 篇
- `GPT-3`
- `FLAN`
- `InstructGPT`
- `RAG`
- `ReAct`
- `DPO`
- `QLoRA`
- `LLaMA`

**适合谁**
- 正在做产品、知识库、agent、私有化部署的人。

---

## 六、阅读时最该抓住的 6 条主线

### 1. 预训练范式怎么变
- BERT 路线偏理解
- GPT 路线偏生成
- T5 提供统一任务接口

### 2. 为什么规模会带来能力变化
- `Scaling Laws` 给出规律
- `Chinchilla` 给出更合理的算力分配
- `PaLM` 展示超大规模能力现象

### 3. 为什么 LLM 不等于“会续写的模型”
- `FLAN` 让模型更会听指令
- `InstructGPT` 让模型更可交互
- `DPO` 让偏好优化更简单高效

### 4. 为什么推理会成为独立主线
- `Chain-of-Thought` 说明中间步骤的重要性
- `Self-Consistency` 强调推理时策略
- `Let’s Verify Step by Step` 强调过程监督
- `DeepSeek-R1` 把 reasoning model 推向新阶段

### 5. 为什么应用层离不开外部系统
- `RAG` 说明参数不是全部知识
- `Toolformer` 和 `ReAct` 说明工具调用不是外挂，而是能力扩展

### 6. 为什么工程实现会反过来决定研究边界
- `RoPE` 影响长上下文设计
- `FlashAttention` 决定训练和推理效率上限
- `LoRA / QLoRA` 决定谁能参与微调生态

---

## 七、一句话总结每条主线

- `BERT`：让 Transformer 成为 NLP 预训练基础设施。
- `GPT -> GPT-2 -> GPT-3`：让生成式语言模型成为通用接口。
- `T5`：让所有任务都能被统一表达。
- `Scaling Laws -> Chinchilla`：让扩展模型变成可计算、可规划的工程问题。
- `FLAN -> InstructGPT -> DPO`：让模型从“会生成”变成“会按人类偏好做事”。
- `CoT -> Self-Consistency -> Step Verification -> R1`：让推理从 prompt 技巧走向独立训练范式。
- `RAG -> Toolformer -> ReAct`：让模型具备连接外部知识和工具的能力。
- `RoPE -> FlashAttention -> LoRA / QLoRA`：让大模型真正可训练、可部署、可微调。
- `LLaMA -> Mixtral -> DeepSeek 系列`：让开源生态具备持续逼近前沿的能力。

---

## 八、最后的建议

如果你的目标是“真正看懂今天的大模型世界”，不要把论文当成孤立点，而要按下面的方式串起来：

1. 先理解预训练范式：`BERT / GPT / T5`
2. 再理解规模规律：`GPT-3 / Scaling Laws / Chinchilla / PaLM`
3. 再理解对齐与产品化：`FLAN / InstructGPT / DPO`
4. 再理解推理与 agent：`CoT / ReAct / Step Verification / DeepSeek-R1`
5. 最后理解工程现实：`RoPE / FlashAttention / LoRA / QLoRA / LLaMA`

真正值得反复读的，不是“论文数量”，而是这些论文背后的五条总问题：
- 模型如何获得知识
- 模型如何随着规模变强
- 模型如何学会遵循人类意图
- 模型如何进行更可靠的推理
- 模型如何在现实算力下被训练和部署

只要这五条主线吃透，后面再看新论文，基本都能迅速定位它是在补哪一块。
