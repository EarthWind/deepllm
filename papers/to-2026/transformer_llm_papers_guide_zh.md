# 2017—2025 Transformer / LLM 与基础模型必读论文导读

这份导读默认从 `Attention Is All You Need` 发布之后开始统计，不重复介绍该论文本身。收录边界固定在 **2025 年 12 月 31 日**：年份原则上采用论文首个公开版本的年份，跨年发表的论文会写成 `2021/2022`。目标不是机械罗列“所有论文”，而是筛出到 2025 年底仍然最值得通读原文、并且能串起 Transformer、LLM 与基础模型演进的关键论文。

全文分三层：前 15 篇是主线骨架，16—30 是进阶主线，31—90 是本次补充的扩展阅读。后者并不表示重要性更低，而是更依赖具体方向；例如做推理服务的人应把 `vLLM`、`GQA`、`GPTQ` 提到第一优先级，做视觉生成的人则应优先读 `CLIP`、`DDPM` 和 `Latent Diffusion`。

筛选标准：

- 对当前大模型主线仍有解释力
- 对后续研究和工程实践影响足够大
- 适合作为系统学习的骨架论文

阅读建议：

- 如果你是第一次系统梳理，先读“第一优先级”
- 如果你已经做过训练、微调或应用开发，再读“第二优先级”
- 如果你有明确方向，例如 RAG、多模态、推理模型、效率优化，再从“扩展阅读”和“专题补充”中选读
- 技术报告不等于完整方法论文；阅读 `GPT-4`、`Llama 3`、`Qwen3` 等报告时，要区分可复现方法、经验总结和未披露细节

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

## 三、扩展阅读：补齐 2017—2025 的关键拼图

下面 60 篇用于补齐原来 30 篇主论文之外的关键节点。它们按问题分组，而不是按热度排序；每一组先读排在前面的奠基工作，再根据自己的方向向后延伸。链接均指向论文原文或正式出版页。

### 3.1 表征学习、预训练、检索与数据

| 编号 | 论文 | 为什么值得读 | 阅读重点 |
|---:|---|---|---|
| 31 | [ULMFiT (2018)](https://arxiv.org/abs/1801.06146) — `Universal Language Model Fine-tuning for Text Classification` · [中文详解](31_ULMFiT_2018_原理.md) | 在 BERT 之前系统证明通用语言模型预训练可以迁移到下游任务。 | 判别式微调、逐层解冻、倾斜三角学习率。 |
| 32 | [ELMo (2018)](https://arxiv.org/abs/1802.05365) — `Deep Contextualized Word Representations` · [中文详解](32_ELMo_2018_原理.md) | 把“词的表示取决于上下文”推到 NLP 主舞台，是静态词向量到预训练模型的关键桥梁。 | 双向语言模型、分层表示、contextual embedding。 |
| 33 | [RoBERTa (2019)](https://arxiv.org/abs/1907.11692) — `A Robustly Optimized BERT Pretraining Approach` · [中文详解](33_RoBERTa_2019_原理.md) | 说明 BERT 的大量收益来自更充分的数据、训练和目标设计，而不只是新架构。 | 去掉 NSP、动态 masking、批量与训练时长。 |
| 34 | [BART (2019)](https://arxiv.org/abs/1910.13461) — `Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension` · [中文详解](34_BART_2019_原理.md) | 把双向编码器与自回归解码器结合起来，成为生成式 encoder-decoder 的经典基线。 | 文本破坏任务、去噪预训练、生成与理解的统一。 |
| 35 | [Sentence-BERT (2019)](https://arxiv.org/abs/1908.10084) — `Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks` · [中文详解](35_Sentence_BERT_2019_原理.md) | 让 BERT 产生可高效检索的句向量，直接影响语义搜索与后来的 RAG embedding。 | 双塔结构、对比/孪生训练、相似度检索。 |
| 36 | [DPR (2020)](https://arxiv.org/abs/2004.04906) — `Dense Passage Retrieval for Open-Domain Question Answering` · [中文详解](36_DPR_2020_原理.md) | 奠定现代稠密检索的双编码器范式，是理解 RAG 检索器的前置论文。 | in-batch negatives、问题/段落双塔、召回率评估。 |
| 37 | [The Pile (2020/2021)](https://arxiv.org/abs/2101.00027) — `The Pile: An 800GB Dataset of Diverse Text for Language Modeling` · [中文详解](37_The_Pile_2020_原理.md) | 代表开源大模型从“只开权重”走向公开数据配方的早期努力。 | 数据来源配比、去重与污染、数据治理。 |
| 38 | [Foundation Models Report (2021)](https://arxiv.org/abs/2108.07258) — `On the Opportunities and Risks of Foundation Models` · [中文详解](38_Foundation_Models_Report_2021_原理.md) | 系统定义并讨论“基础模型”范式，把技术、应用和社会风险放进同一框架。 | 同质化带来的杠杆与系统性风险、评测与治理。 |
| 39 | [OLMo (2024)](https://arxiv.org/abs/2402.00838) — `OLMo: Accelerating the Science of Language Models` · [中文详解](39_OLMo_2024_原理.md) | 不只开放权重，还开放数据、代码、日志和中间检查点，是可复现 LLM 科学的重要样板。 | 真正的开放性、训练过程分析、开放数据链路。 |

### 3.2 大规模训练、架构与推理系统

| 编号 | 论文 | 为什么值得读 | 阅读重点 |
|---:|---|---|---|
| 40 | [Megatron-LM (2019)](https://arxiv.org/abs/1909.08053) — `Training Multi-Billion Parameter Language Models Using Model Parallelism` · [中文详解](40_Megatron_LM_2019_原理.md) | 奠定大模型张量并行的工程主线，解释单卡放不下时如何拆分矩阵计算。 | tensor parallel、通信开销、与数据/流水线并行的组合。 |
| 41 | [ZeRO (2019/2020)](https://arxiv.org/abs/1910.02054) — `ZeRO: Memory Optimizations Toward Training Trillion Parameter Models` · [中文详解](41_ZeRO_2019_原理.md) | 把优化器状态、梯度和参数分片，显著改变大模型训练的显存边界。 | ZeRO 三阶段、冗余状态、通信—显存权衡。 |
| 42 | [GPTQ (2022)](https://arxiv.org/abs/2210.17323) — `Accurate Post-Training Quantization for Generative Pre-trained Transformers` · [中文详解](42_GPTQ_2022_原理.md) | 代表基于二阶信息的权重量化路线，让大模型低比特部署成为标准课题。 | 逐层误差补偿、3/4-bit 权重量化、精度—速度关系。 |
| 43 | [Speculative Decoding (2022/2023)](https://arxiv.org/abs/2211.17192) — `Fast Inference from Transformers via Speculative Decoding` · [中文详解](43_Speculative_Decoding_2022_原理.md) | 用小模型起草、大模型并行验证多个 token，在不改变目标分布的前提下加速解码。 | 接受/拒绝机制、无损采样、draft/target 配比。 |
| 44 | [GQA (2023)](https://arxiv.org/abs/2305.13245) — `GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints` · [中文详解](44_GQA_2023_原理.md) | 在 MHA 质量与 MQA 速度之间取得平衡，成为现代 LLM 降低 KV cache 的常用结构。 | query head 与 KV head 分组、uptraining、推理带宽。 |
| 45 | [AWQ (2023)](https://arxiv.org/abs/2306.00978) — `Activation-aware Weight Quantization for LLM Compression and Acceleration` · [中文详解](45_AWQ_2023_原理.md) | 展示少量显著权重通道决定量化质量，推动实用的 4-bit weight-only 部署。 | activation-aware scaling、显著通道、硬件友好量化。 |
| 46 | [FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691) — `Faster Attention with Better Parallelism and Work Partitioning` · [中文详解](46_FlashAttention2_2023_原理.md) | 在第一代 IO-aware attention 上进一步优化并行和工作划分，是理解现代高性能 attention kernel 的续篇。 | warp/线程块划分、非矩阵乘 FLOPs、occupancy。 |
| 47 | [vLLM / PagedAttention (2023)](https://arxiv.org/abs/2309.06180) — `Efficient Memory Management for Large Language Model Serving with PagedAttention` | 把操作系统分页思想用于 KV cache，成为高吞吐 LLM 服务的重要基础设施。 | 连续批处理、KV 分页、碎片与共享。 |
| 48 | [Mistral 7B (2023)](https://arxiv.org/abs/2310.06825) — `Mistral 7B` · [中文详解](48_Mistral_7B_2023_原理.md) | 用 GQA 与滑动窗口注意力证明小型开放模型也能以工程设计取得强性能。 | sliding-window attention、rolling buffer、模型尺寸与吞吐。 |
| 49 | [DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434) — `A Strong, Economical, and Efficient Mixture-of-Experts Language Model` · [中文详解](49_DeepSeek_V2_2024_原理.md) | 提出 MLA 与 DeepSeekMoE，直接连接高效 KV cache、稀疏专家和后来的 V3/R1。 | latent KV 压缩、细粒度专家、共享专家。 |
| 50 | [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437) — `DeepSeek-V3 Technical Report` · [中文详解](50_DeepSeek_V3_2024_原理.md) | 展示超大 MoE 的稳定、低成本训练，并引入无辅助损失负载均衡与多 token 预测。 | auxiliary-loss-free balancing、MTP、FP8 训练与系统协同。 |

### 3.3 对齐、代码与推理模型

| 编号 | 论文 | 为什么值得读 | 阅读重点 |
|---:|---|---|---|
| 51 | [Learning to Summarize from Human Feedback (2020)](https://arxiv.org/abs/2009.01325) · [中文详解](51_Learning_to_Summarize_from_Human_Feedback_2020_原理.md) | 是 InstructGPT 之前最清楚的语言模型偏好学习范例之一。 | 人类比较数据、reward model、RL 优化与代理指标偏差。 |
| 52 | [Codex / HumanEval (2021)](https://arxiv.org/abs/2107.03374) — `Evaluating Large Language Models Trained on Code` · [中文详解](52_Codex_HumanEval_2021_原理.md) | 证明代码可成为 LLM 的核心能力，并引入影响深远的 HumanEval 与 pass@k。 | 代码数据、函数级生成、执行式评测、数据污染。 |
| 53 | [Training Verifiers (2021)](https://arxiv.org/abs/2110.14168) — `Training Verifiers to Solve Math Word Problems` · [中文详解](53_Training_Verifiers_2021_原理.md) | 把生成候选与验证答案分开，预示 verifier 和 test-time sampling 的推理路线。 | GSM8K、solution verifier、采样后排序。 |
| 54 | [WebGPT (2021)](https://arxiv.org/abs/2112.09332) — `Browser-assisted Question-answering with Human Feedback` · [中文详解](54_WebGPT_2021_原理.md) | 把浏览、引用和人类偏好结合，是联网问答 Agent 与可追溯回答的重要前身。 | 行为克隆、浏览环境、引用支持、reward model。 |
| 55 | [STaR (2022)](https://arxiv.org/abs/2203.14465) — `Self-Taught Reasoner: Bootstrapping Reasoning With Reasoning` · [中文详解](55_STaR_2022_原理.md) | 用模型生成并筛选 rationale，再反复微调，开创推理轨迹自举路线。 | rationale filtering、rationalization、迭代自训练。 |
| 56 | [GPT-4 Technical Report (2023)](https://arxiv.org/abs/2303.08774) · [中文详解](56_GPT4_2023_原理.md) | 历史与评测价值很高，但架构、数据和训练细节披露有限，不适合作为可复现方法论文读。 | 多模态能力、可预测扩展、风险评测，以及“没有披露什么”。 |
| 57 | [DeepSeekMath (2024)](https://arxiv.org/abs/2402.03300) · [中文详解](57_DeepSeekMath_2024_原理.md) | 系统展示数学数据工程，并首次提出后来用于 R1 的 GRPO。 | 数学语料筛选、continued pretraining、GRPO。 |
| 58 | [Scaling LLM Test-Time Compute Optimally (2024)](https://arxiv.org/abs/2408.03314) · [中文详解](58_Scaling_Test_Time_Compute_2024_原理.md) | 把推理时计算当作可分配预算，说明何时搜索/验证比单纯增大模型更有效。 | 难度自适应预算、PRM 搜索、best-of-N 基线。 |
| 59 | [Llama 3 (2024)](https://arxiv.org/abs/2407.21783) — `The Llama 3 Herd of Models` · [中文详解](59_Llama3_2024_原理.md) | 是开放前沿模型在预训练、后训练、多语言、工具和安全上的完整经验总结。 | 15T token 级训练、数据过滤、SFT/RLHF/DPO、安全评测。 |
| 60 | [Kimi k1.5 (2025)](https://arxiv.org/abs/2501.12599) — `Scaling Reinforcement Learning with LLMs` · [中文详解](60_Kimi_k1.5_2025_原理.md) | 展示长上下文 RL 与 long-to-short 蒸馏如何同时推动文本和多模态推理。 | long-CoT RL、长度课程、策略优化、long2short。 |
| 61 | [s1 (2025)](https://arxiv.org/abs/2501.19393) — `Simple Test-Time Scaling` · [中文详解](61_s1_2025_原理.md) | 说明精挑的 1,000 条推理轨迹和简单 budget forcing 也能产生强推理行为。 | 数据质量/难度/多样性、budget forcing、复现实验边界。 |
| 62 | [Qwen3 (2025)](https://arxiv.org/abs/2505.09388) — `Qwen3 Technical Report` · [中文详解](62_Qwen3_2025_原理.md) | 把 thinking 与 non-thinking 统一在一套模型中，并提供显式思考预算。 | 混合推理模式、thinking budget、强到弱蒸馏、多语言。 |

### 3.4 RAG、Agent、长上下文与评测

| 编号 | 论文 | 为什么值得读 | 阅读重点 |
|---:|---|---|---|
| 63 | [TruthfulQA (2021)](https://arxiv.org/abs/2109.07958) · [中文详解](63_TruthfulQA_2021_原理.md) | 专门测试模型是否模仿人类常见谬误，提醒“更大”不自动等于“更真实”。 | imitation falsehood、真实性与信息量、评测者偏差。 |
| 64 | [HELM (2022)](https://arxiv.org/abs/2211.09110) — `Holistic Evaluation of Language Models` · [中文详解](64_HELM_2022_原理.md) | 把准确率之外的校准、鲁棒性、公平、偏见、毒性和效率纳入统一评测。 | 场景覆盖、多指标权衡、透明可复现评测。 |
| 65 | [Reflexion (2023)](https://arxiv.org/abs/2303.11366) — `Language Agents with Verbal Reinforcement Learning` · [中文详解](65_Reflexion_2023_原理.md) | 用语言化反馈和记忆改进下一轮尝试，影响了大量自反思 Agent 设计。 | 轨迹反馈、episodic memory、无需更新权重的改进。 |
| 66 | [Generative Agents (2023)](https://arxiv.org/abs/2304.03442) — `Generative Agents: Interactive Simulacra of Human Behavior` · [中文详解](66_Generative_Agents_2023_原理.md) | 展示记忆、反思与计划如何组成持续运行的社会型 Agent。 | memory stream、重要性/时近性/相关性检索、反思与规划。 |
| 67 | [Voyager (2023)](https://arxiv.org/abs/2305.16291) — `An Open-Ended Embodied Agent with Large Language Models` · [中文详解](67_Voyager_2023_原理.md) | 把自动课程、技能库和迭代提示结合到开放世界具身学习。 | skill library、automatic curriculum、环境反馈与代码执行。 |
| 68 | [MT-Bench & Chatbot Arena (2023)](https://arxiv.org/abs/2306.05685) — `Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena` · [中文详解](68_MT_Bench_Chatbot_Arena_2023_原理.md) | 奠定多轮对话、LLM-as-a-judge 与众包两两比较的评测范式。 | position/verbosity/self-enhancement bias、人类一致性、Elo 局限。 |
| 69 | [Lost in the Middle (2023)](https://arxiv.org/abs/2307.03172) · [中文详解](69_Lost_in_the_Middle_2023_原理.md) | 证明“上下文窗口很长”不等于“模型能可靠使用全部上下文”。 | U 形位置效应、多文档 QA、key-value retrieval。 |
| 70 | [SWE-bench (2023)](https://arxiv.org/abs/2310.06770) · [中文详解](70_SWE_bench_2023_原理.md) | 把代码评测从单函数生成推进到真实仓库 issue 修复，成为 coding agent 的关键基准。 | repository-level context、执行环境、F2P/P2P 测试驱动判定、污染风险。 |
| 71 | [Self-RAG (2023/2024)](https://arxiv.org/abs/2310.11511) — `Learning to Retrieve, Generate, and Critique through Self-Reflection` | 让模型学习何时检索、如何判断证据和如何自评，而不是固定塞入若干文档。 | reflection tokens、按需检索、事实性与引用质量。 |

### 3.5 多模态、生成模型、强化学习、科学 AI 与反思

| 编号 | 论文 | 为什么值得读 | 阅读重点 |
|---:|---|---|---|
| 72 | [AlphaZero (2017)](https://arxiv.org/abs/1712.01815) — `Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm` | 证明只给规则、通过自博弈即可在多个棋类达到超人水平，是现代自博弈 RL 的代表作。 | MCTS、policy/value network、自生成课程与搜索。 |
| 73 | [MuZero (2019/2020)](https://arxiv.org/abs/1911.08265) — `Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model` | 不显式重建环境，而是学习对规划有用的潜在动力学。 | representation/dynamics/prediction 三网络、潜空间 MCTS。 |
| 74 | [Vision Transformer (2020/2021)](https://arxiv.org/abs/2010.11929) — `An Image is Worth 16x16 Words` | 证明纯 Transformer 在足够数据下可成为视觉主干，打开统一架构之路。 | patch tokenization、位置编码、数据规模与归纳偏置。 |
| 75 | [DDPM (2020)](https://arxiv.org/abs/2006.11239) — `Denoising Diffusion Probabilistic Models` | 奠定现代扩散生成模型的标准训练与采样框架。 | 前向加噪、反向去噪、噪声预测目标、采样成本。 |
| 76 | [CLIP (2021)](https://arxiv.org/abs/2103.00020) — `Learning Transferable Visual Models From Natural Language Supervision` | 用海量图文对比学习获得开放词汇、零样本视觉能力，是现代多模态模型的重要地基。 | 图文双塔、对比损失、prompt ensemble、数据偏差。 |
| 77 | [AlphaFold2 (2021)](https://www.nature.com/articles/s41586-021-03819-2) — `Highly Accurate Protein Structure Prediction with AlphaFold` | 展示深度学习如何突破长期科学难题，是 AI for Science 的标志性工作。 | Evoformer、MSA/pair representation、结构模块、recycling 与置信度。 |
| 78 | [Decision Transformer (2021)](https://arxiv.org/abs/2106.01345) — `Reinforcement Learning via Sequence Modeling` | 把离线强化学习重写为按目标回报条件化的序列建模问题。 | return-to-go、轨迹 token 化、行为克隆与 RL 的边界。 |
| 79 | [Stochastic Parrots (2021)](https://dl.acm.org/doi/10.1145/3442188.3445922) — `On the Dangers of Stochastic Parrots` | 在大模型全面产品化前系统提出数据、能耗、偏见与伤害问题，补足纯能力叙事。 | 数据来源与同意、规模成本、下游伤害、文档化责任。 |
| 80 | [Latent Diffusion (2021/2022)](https://arxiv.org/abs/2112.10752) — `High-Resolution Image Synthesis with Latent Diffusion Models` | 在压缩潜空间做扩散，并用 cross-attention 接收文本条件，直接奠定 Stable Diffusion 路线。 | VAE 潜空间、cross-attention、classifier-free guidance、效率—细节权衡。 |
| 81 | [Toy Models of Superposition (2022)](https://arxiv.org/abs/2209.10652) | 用可控模型解释多义神经元和特征叠加，是现代机制可解释性的核心入口。 | 稀疏特征、superposition、相变与表示几何。 |
| 82 | [Whisper (2022)](https://arxiv.org/abs/2212.04356) — `Robust Speech Recognition via Large-Scale Weak Supervision` | 展示大规模弱监督可以把语音识别统一成鲁棒的序列到序列任务。 | 多任务 token、弱标注数据、跨语言迁移、长音频切分。 |
| 83 | [Segment Anything (2023)](https://arxiv.org/abs/2304.02643) | 把视觉分割变成 promptable foundation model，并建立数据引擎。 | image/prompt encoder、mask decoder、歧义感知输出、SA-1B 数据闭环。 |
| 84 | [AlphaGeometry (2024)](https://www.nature.com/articles/s41586-023-06747-5) — `Solving Olympiad Geometry without Human Demonstrations` | 把语言模型引导与符号演绎结合，展示合成数据和神经符号系统的推理价值。 | synthetic theorem generation、辅助点构造、symbolic deduction。 |

### 3.6 2025 年值得继续追踪的开放研究

这一组离当前时间最近，长期影响仍待检验。阅读时应更看重公开方法、消融和可复现证据，不要把作者报告的单次 SOTA 直接视为定论。

| 编号 | 论文 | 为什么值得读 | 阅读重点 |
|---:|---|---|---|
| 85 | [Search-R1 (2025)](https://arxiv.org/abs/2503.09516) — `Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning` | 把多轮搜索直接放进 RL rollout，使“边推理边检索”成为训练出来的策略。 | 多轮 search action、retrieved-token masking、结果奖励。 |
| 86 | [DAPO (2025)](https://arxiv.org/abs/2503.14476) — `An Open-Source LLM Reinforcement Learning System at Scale` | 公开大规模推理 RL 的代码、数据和稳定训练技巧，补足 R1 报告的复现缺口。 | Clip-Higher、dynamic sampling、token-level loss、overlong reward shaping。 |
| 87 | [Gemma 3 (2025)](https://arxiv.org/abs/2503.19786) — `Gemma 3 Technical Report` | 展示轻量开放模型如何结合视觉、多语言、长上下文与蒸馏。 | 局部/全局 attention 配比、KV cache、蒸馏与多模态后训练。 |
| 88 | [Limits of RLVR (2025)](https://arxiv.org/abs/2504.13837) — `Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?` | 用大 k 的 pass@k 质疑 RLVR 是否创造新能力，提供理解 R1 类训练的必要反方证据。 | pass@k 能力边界、分布重加权、RL 与蒸馏的区别。 |
| 89 | [DeepSeek-Prover-V2 (2025)](https://arxiv.org/abs/2504.21801) · [中文详解](89_DeepSeek_Prover_V2_2025_原理.md) | 把子目标分解、形式证明和 RL 结合，展示推理模型在 Lean 中生成可机器验证证明。 | recursive theorem proving、cold start、subgoal decomposition、形式验证。 |
| 90 | [Absolute Zero (2025)](https://arxiv.org/abs/2505.03335) — `Reinforced Self-play Reasoning with Zero Data` · [中文详解](90_Absolute_Zero_2025_原理.md) | 让模型同时提出可验证任务并求解，探索不依赖外部题库的自生成课程。 | proposer/solver、自博弈、代码执行器、可验证奖励与安全边界。 |

### 这 60 篇怎么取舍

- **只补 LLM 核心缺口**：`RoBERTa -> DPR -> ZeRO -> WebGPT -> GPTQ -> GQA -> vLLM -> DeepSeek-V2 -> DeepSeek-V3 -> Llama 3`
- **只补推理模型缺口**：`Training Verifiers -> STaR -> DeepSeekMath -> Test-Time Compute -> Kimi k1.5 -> s1 -> Qwen3`
- **只补多模态与生成**：`ViT -> CLIP -> DDPM -> Latent Diffusion -> Whisper -> Segment Anything`
- **只补 Agent 与评测**：`WebGPT -> HELM -> Reflexion -> MT-Bench / Arena -> Lost in the Middle -> SWE-bench -> Self-RAG`
- **只补系统工程**：`Megatron-LM -> ZeRO -> FlashAttention -> GPTQ / AWQ -> GQA -> vLLM -> DeepSeek-V2`
- **只补广义 AI 视野**：`AlphaZero -> MuZero -> AlphaFold2 -> Decision Transformer -> AlphaGeometry`
- **只追踪 2025 开放研究**：`Search-R1 -> DAPO -> Limits of RLVR -> DeepSeek-Prover-V2 -> Absolute Zero`

---

## 四、按主题补充：你有明确方向时再扩展

### A. 如果你重点看 RAG / 检索增强

- [DPR (2020)](https://arxiv.org/abs/2004.04906)：现代稠密检索双塔范式。
- [REALM (2020)](https://arxiv.org/abs/2002.08909)：更早期地把检索纳入预训练/知识增强视角。
- [RAG (2020)](https://arxiv.org/abs/2005.11401)：参数记忆与非参数记忆结合的主线起点。
- [FiD (2020)](https://arxiv.org/abs/2007.01282)：分别编码多段证据、在解码器中融合，建立强 open-domain QA 基线。
- [RETRO (2021)](https://arxiv.org/abs/2112.04426)：把海量外部语料检索与生成进一步深度绑定。
- [Atlas (2022)](https://arxiv.org/abs/2208.03299)：系统研究 retrieval-augmented LM 的 few-shot 能力。
- [Self-RAG (2023)](https://arxiv.org/abs/2310.11511)：学习按需检索、生成与自我批判。
- [Search-R1 (2025)](https://arxiv.org/abs/2503.09516)：用 RL 学会在推理过程中执行多轮搜索。

**导读建议**

- 先搞清楚：模型参数不是唯一知识载体。
- 再理解：检索是在训练时接入、推理时接入，还是两者都接入。
- 最后再看今天工程里的 chunk、embedding、rerank、引用评测和 agentic RAG，这样不容易只会调框架。

### B. 如果你重点看长上下文

- [Transformer-XL (2019)](https://arxiv.org/abs/1901.02860)：跨段记忆与长依赖问题的重要起点。
- [Longformer (2020)](https://arxiv.org/abs/2004.05150)：局部窗口加少量全局 token 的稀疏 attention 代表。
- [ALiBi (2021)](https://arxiv.org/abs/2108.12409)：极简位置偏置方案，对长度外推讨论影响很大。
- [RoFormer / RoPE (2021)](https://arxiv.org/abs/2104.09864)：今天 LLM 最常用的位置编码。
- [FlashAttention (2022)](https://arxiv.org/abs/2205.14135)：长上下文工程效率核心。
- [YaRN (2023)](https://arxiv.org/abs/2309.00071)：低成本扩展 RoPE 上下文窗口的实用方法。
- [Ring Attention (2023)](https://arxiv.org/abs/2310.01889)：把 blockwise attention 与设备环形通信结合，突破单设备上下文限制。
- [Lost in the Middle (2023)](https://arxiv.org/abs/2307.03172) · [中文详解](69_Lost_in_the_Middle_2023_原理.md)：检验模型是否真正使用了长上下文，而不只看窗口标称长度。

**导读建议**

- 长上下文的难点不只有“能塞进去”，还包括训练稳定性、位置编码、注意力复杂度、推理显存。
- 区分三件事：窗口长度、有效检索长度，以及长上下文下的推理质量。

### C. 如果你重点看 Agent / 工具调用

- [WebGPT (2021)](https://arxiv.org/abs/2112.09332) · [中文详解](54_WebGPT_2021_原理.md)：浏览、引用与人类反馈。
- [ReAct (2022/2023)](https://arxiv.org/abs/2210.03629)：推理与行动交替。
- [Toolformer (2023)](https://arxiv.org/abs/2302.04761)：自监督学习何时调用工具。
- [Reflexion (2023)](https://arxiv.org/abs/2303.11366) · [中文详解](65_Reflexion_2023_原理.md)：用语言反馈和记忆改进下一轮轨迹。
- [Generative Agents (2023)](https://arxiv.org/abs/2304.03442) · [中文详解](66_Generative_Agents_2023_原理.md)：用记忆流、反思与层级计划维持长期角色和社会互动。
- [Tree of Thoughts (2023)](https://arxiv.org/abs/2305.10601)：显式搜索多条推理路径。
- [Voyager (2023)](https://arxiv.org/abs/2305.16291) · [中文详解](67_Voyager_2023_原理.md)：开放世界中的自动课程、可执行技能库与环境反馈闭环。
- [WebArena (2023)](https://arxiv.org/abs/2307.13854)：在可复现网站环境中评测真实网页任务。
- [SWE-bench (2023)](https://arxiv.org/abs/2310.06770) · [中文详解](70_SWE_bench_2023_原理.md)：用真实 GitHub issue、历史仓库快照和隐藏测试评测代码 Agent。

**导读建议**

- 先看“何时调工具”，再看“如何边想边做”，最后看“如何搜索多条推理路径”。
- 不要只看任务成功率；还要看环境可复现性、轨迹成本、错误恢复和评测污染。

### D. 如果你重点看多模态

- [Vision Transformer (2020)](https://arxiv.org/abs/2010.11929)：视觉 token 化与 Transformer 视觉主干。
- [CLIP (2021)](https://arxiv.org/abs/2103.00020)：图文对比学习与开放词汇视觉能力。
- [Flamingo (2022)](https://arxiv.org/abs/2204.14198)：少样本视觉语言模型的重要里程碑。
- [BLIP-2 (2023)](https://arxiv.org/abs/2301.12597)：高效连接冻结视觉编码器和 LLM 的经典设计。
- [LLaVA (2023)](https://arxiv.org/abs/2304.08485)：开源多模态指令微调的爆发起点之一。
- [RT-2 (2023)](https://arxiv.org/abs/2307.15818)：把视觉语言模型的知识迁移到机器人动作 token。
- [Gemini (2023)](https://arxiv.org/abs/2312.11805)：统一多模态基础模型代表。

**导读建议**

- 多模态不只是“加图片输入”，关键是模态对齐、连接器设计、指令微调和推理能力迁移。
- 对比“冻结单模态专家后连接”和“原生联合训练”两条路线，不要只比较榜单分数。

### E. 如果你重点看后训练与对齐

- [Learning to Summarize from Human Feedback (2020)](https://arxiv.org/abs/2009.01325) · [中文详解](51_Learning_to_Summarize_from_Human_Feedback_2020_原理.md)
- [InstructGPT (2022)](https://arxiv.org/abs/2203.02155)
- [Constitutional AI (2022/2023)](https://arxiv.org/abs/2212.08073)
- [DPO (2023)](https://arxiv.org/abs/2305.18290)
- [Let’s Verify Step by Step (2023)](https://arxiv.org/abs/2305.20050)
- [The Instruction Hierarchy (2024)](https://arxiv.org/abs/2404.13208)：训练模型区分系统、开发者和用户等不同权限的指令。
- [DAPO (2025)](https://arxiv.org/abs/2503.14476)：大规模推理 RL 的开放训练配方与系统。

**导读建议**

- 这条线最关键的问题不是“让模型更会说”，而是“让模型更像可用产品”。
- 始终追问反馈来自谁、优化了什么代理目标，以及分布外是否仍然成立。

### F. 如果你重点看训练与推理系统

- `Megatron-LM -> ZeRO`：模型如何跨设备训练。
- `FlashAttention -> FlashAttention-2`：attention 如何从算法映射到存储层次和 GPU 并行。
- `GPTQ -> AWQ`：权重量化如何降低部署门槛。
- `GQA -> DeepSeek-V2 MLA`：如何减少不断增长的 KV cache。
- `Speculative Decoding -> vLLM`：如何同时优化单请求延迟和服务吞吐。

**导读建议**

- 分清算法复杂度、显存占用、内存带宽、通信成本和端到端吞吐；它们不是同一个指标。

### G. 如果你重点看评测

- [MMLU (2020)](https://arxiv.org/abs/2009.03300)：大规模多任务知识与理解评测。
- [TruthfulQA (2021)](https://arxiv.org/abs/2109.07958) · [中文详解](63_TruthfulQA_2021_原理.md)：模型是否复述常见谬误。
- [BIG-bench (2022)](https://arxiv.org/abs/2206.04615)：大规模协作式任务集合与能力边界。
- [HELM (2022)](https://arxiv.org/abs/2211.09110) · [中文详解](64_HELM_2022_原理.md)：多场景、多指标、透明评测。
- [MT-Bench / Chatbot Arena (2023)](https://arxiv.org/abs/2306.05685) · [中文详解](68_MT_Bench_Chatbot_Arena_2023_原理.md)：多轮开放题、LLM judge、人类两两偏好与系统性偏差。
- [SWE-bench (2023)](https://arxiv.org/abs/2310.06770) · [中文详解](70_SWE_bench_2023_原理.md)：真实软件工程任务、仓库级补丁与执行式判定。

**导读建议**

- 先检查数据污染、提示敏感性和 judge bias，再讨论小数点后的排名差异。
- benchmark 分数是对某套任务与判分器的测量，不是“智能”的完整定义。

### H. 如果你重点看图像生成与语音

- `DDPM (2020)`：扩散模型的标准起点。
- `DDIM (2020)`：非马尔可夫采样与大幅减少采样步数。
- `Latent Diffusion (2021/2022)`：潜空间扩散和文本 cross-attention。
- `Whisper (2022)`：大规模弱监督语音识别。
- `Segment Anything (2023)`：可提示分割模型与数据引擎。

**导读建议**

- 把生成质量、条件控制、采样速度和训练成本分开评价。

### I. 如果你重点看可解释性与安全

- [A Mathematical Framework for Transformer Circuits (2021)](https://transformer-circuits.pub/2021/framework/index.html)：用电路视角分解 attention 与 MLP。
- [Toy Models of Superposition (2022)](https://arxiv.org/abs/2209.10652)：理解 polysemanticity 与特征叠加。
- [Stochastic Parrots (2021)](https://dl.acm.org/doi/10.1145/3442188.3445922)：理解规模化语言模型的数据、环境与社会风险。
- [Red Teaming Language Models with Language Models (2022)](https://arxiv.org/abs/2202.03286)：用模型自动发现有害行为。
- [Sleeper Agents (2024)](https://arxiv.org/abs/2401.05566)：研究后门行为是否能在安全训练后持续存在。

**导读建议**

- 可解释性证据要区分相关性、因果干预和真正的机制解释；安全评测则要区分“没测到”与“没有风险”。

### J. 如果你想把视野扩到广义 AI

- `AlphaZero (2017)`：自博弈、搜索与策略/价值网络。
- `MuZero (2019/2020)`：学习只服务于规划的潜在世界模型。
- `AlphaFold2 (2021)`：AI for Science 的标志性突破。
- `Decision Transformer (2021)`：把强化学习重写为序列建模。
- `AlphaGeometry (2024)`：神经语言模型、合成数据与符号推理结合。
- `DeepSeek-Prover-V2 (2025)`：让自然语言推理落到可机器验证的形式证明。

**导读建议**

- 这些论文提醒我们：LLM 是当前主线，但搜索、环境交互、符号系统、科学先验和可靠验证仍是 AI 的核心组成。

---

## 五、最推荐的阅读顺序

### 路线 1：按历史主线读

1. `BERT`
2. `GPT`
3. `GPT-2`
4. `T5`
5. `GPT-3`
6. `Scaling Laws`
7. `DPR`
8. `RAG`
9. `FLAN`
10. `RoFormer / RoPE`
11. `InstructGPT`
12. `Chain-of-Thought`
13. `Chinchilla`
14. `PaLM`
15. `FlashAttention`
16. `LLaMA`
17. `DPO`
18. `vLLM / PagedAttention`
19. `Mistral / Mixtral`
20. `DeepSeek-V2`
21. `Llama 3`
22. `DeepSeek-V3`
23. `DeepSeek-R1`
24. `Qwen3`

### 路线 2：如果你做应用开发

1. `GPT-3`
2. `InstructGPT`
3. `FLAN`
4. `DPR`
5. `RAG`
6. `Self-RAG`
7. `ReAct`
8. `Toolformer`
9. `DPO`
10. `QLoRA`
11. `vLLM`
12. `SWE-bench`
13. `DeepSeek-R1`

### 路线 3：如果你做训练或微调

1. `Scaling Laws`
2. `Chinchilla`
3. `Megatron-LM`
4. `ZeRO`
5. `T5`
6. `InstructGPT`
7. `LoRA`
8. `QLoRA`
9. `DPO`
10. `Let’s Verify Step by Step`
11. `Switch Transformer`
12. `Mixtral`
13. `DeepSeek-V2`
14. `DeepSeek-V3`
15. `OLMo`

### 路线 4：如果你做推理模型

1. `Chain-of-Thought`
2. `Self-Consistency`
3. `Training Verifiers`
4. `STaR`
5. `Tree of Thoughts`
6. `Let’s Verify Step by Step`
7. `DeepSeekMath / GRPO`
8. `Scaling LLM Test-Time Compute Optimally`
9. `DeepSeek-R1`
10. `DAPO`
11. `Kimi k1.5`
12. `s1`
13. `Limits of RLVR`
14. `DeepSeek-Prover-V2`
15. `Absolute Zero`

### 路线 5：如果你做推理部署与性能优化

1. `Megatron-LM`
2. `ZeRO`
3. `FlashAttention`
4. `FlashAttention-2`
5. `GPTQ`
6. `AWQ`
7. `GQA`
8. `Speculative Decoding`
9. `vLLM / PagedAttention`
10. `DeepSeek-V2 / MLA`

### 路线 6：如果你做多模态与生成模型

1. `Vision Transformer`
2. `CLIP`
3. `DDPM`
4. `Latent Diffusion`
5. `Flamingo`
6. `BLIP-2`
7. `LLaVA`
8. `Gemini`
9. `Whisper`
10. `Segment Anything`
11. `Gemma 3`

### 路线 7：如果你做 Agent 与评测

1. `WebGPT`
2. `HELM`
3. `ReAct`
4. `Toolformer`
5. `Reflexion`
6. `Generative Agents`
7. `Voyager`
8. `MT-Bench / Chatbot Arena`
9. `Lost in the Middle`
10. `SWE-bench`
11. `Self-RAG`

### 路线 8：如果你想建立广义 AI 视野

1. `AlphaZero`
2. `MuZero`
3. `Vision Transformer`
4. `CLIP`
5. `AlphaFold2`
6. `Decision Transformer`
7. `DDPM`
8. `Latent Diffusion`
9. `AlphaGeometry`
10. `Toy Models of Superposition`

---

## 六、如果时间有限，怎么取舍

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

### 此次新增只读 10 篇

- `DPR`
- `Learning to Summarize from Human Feedback`
- `ZeRO`
- `GPTQ`
- `GQA`
- `vLLM / PagedAttention`
- `DeepSeek-V2`
- `DeepSeek-V3`
- `Scaling LLM Test-Time Compute Optimally`
- `SWE-bench`

**适合谁**

- 已经读过原来 30 篇，希望用最短路径补齐数据、系统、推理和评测缺口的人。

---

## 七、阅读时最该抓住的 9 条主线

### 1. 预训练范式怎么变

- BERT 路线偏理解
- GPT 路线偏生成
- T5 提供统一任务接口
- `ULMFiT / ELMo / RoBERTa / BART / Sentence-BERT / DPR` 解释这条范式并非一步形成

### 2. 为什么规模会带来能力变化

- `Scaling Laws` 给出规律
- `Chinchilla` 给出更合理的算力分配
- `PaLM` 展示超大规模能力现象
- `The Pile / OLMo` 提醒数据质量、透明性和复现同样重要

### 3. 为什么 LLM 不等于“会续写的模型”

- `FLAN` 让模型更会听指令
- `InstructGPT` 让模型更可交互
- `DPO` 让偏好优化更简单高效

### 4. 为什么推理会成为独立主线

- `Chain-of-Thought` 说明中间步骤的重要性
- `Self-Consistency` 强调推理时策略
- `Let’s Verify Step by Step` 强调过程监督
- `DeepSeekMath / DeepSeek-R1` 把 GRPO、可验证奖励和 reasoning model 推向新阶段
- `Test-Time Compute / Kimi k1.5 / s1` 说明推理预算本身也是扩展轴
- `DAPO / Limits of RLVR / DeepSeek-Prover-V2 / Absolute Zero` 展示开放复现、能力边界争论、形式证明与自生成课程四条 2025 前沿分支

### 5. 为什么应用层离不开外部系统

- `RAG` 说明参数不是全部知识
- `Toolformer` 和 `ReAct` 说明工具调用不是外挂，而是能力扩展
- `Self-RAG / WebGPT / SWE-bench` 把检索、引用、行动和真实环境评测接起来

### 6. 为什么工程实现会反过来决定研究边界

- `RoPE` 影响长上下文设计
- `FlashAttention` 决定训练和推理效率上限
- `LoRA / QLoRA` 决定谁能参与微调生态
- `ZeRO / GQA / GPTQ / vLLM` 决定模型能否被训练、放进显存并以足够吞吐服务

### 7. 为什么“窗口更长、分数更高”还不够

- `Lost in the Middle` 说明标称上下文长度不等于有效使用长度
- `HELM / TruthfulQA` 说明能力之外还要测校准、真实性、鲁棒性与风险
- `MT-Bench / Chatbot Arena` 说明自动 judge 和人类偏好也各有系统偏差

### 8. 为什么基础模型天然走向多模态

- `ViT / CLIP` 建立统一 token 与图文对齐基础
- `Flamingo / BLIP-2 / LLaVA / Gemini` 探索连接冻结专家、指令微调和原生多模态训练
- `DDPM / Latent Diffusion / Whisper / SAM` 说明基础模型范式已超出文本生成

### 9. 为什么 LLM 不是 AI 的全部

- `AlphaZero / MuZero` 强调搜索、环境模型和自博弈
- `AlphaFold2 / AlphaGeometry` 强调科学先验、符号系统与可验证输出
- `Toy Models of Superposition / Stochastic Parrots` 分别提醒我们理解内部机制与外部影响

---

## 八、一句话总结每条主线

- `BERT`：让 Transformer 成为 NLP 预训练基础设施。
- `GPT -> GPT-2 -> GPT-3`：让生成式语言模型成为通用接口。
- `T5`：让所有任务都能被统一表达。
- `Scaling Laws -> Chinchilla -> OLMo`：让扩展模型变成可计算、可审计、可复现的工程问题。
- `Human Feedback -> FLAN -> InstructGPT -> DPO`：让模型从“会生成”变成“会按人类偏好做事”。
- `CoT -> Verifier -> Process Supervision -> GRPO / R1 -> DAPO / Test-Time Scaling`：让推理从 prompt 技巧走向训练与推理计算协同的独立范式。
- `DPR -> RAG -> WebGPT -> Toolformer / ReAct -> Self-RAG`：让模型连接外部知识、引用、工具与环境。
- `Megatron / ZeRO -> FlashAttention -> GQA -> GPTQ / AWQ -> vLLM`：让大模型真正可训练、可压缩、可高吞吐部署。
- `LLaMA -> Mistral / Mixtral -> Llama 3 -> DeepSeek-V2 / V3 / R1 -> Qwen3`：让开放模型生态持续逼近前沿。
- `ViT -> CLIP -> Flamingo / BLIP-2 / LLaVA -> Gemini`：让基础模型从文本走向原生多模态。
- `DDPM -> Latent Diffusion`：让可扩展的去噪生成成为图像生成主线。
- `HELM -> MT-Bench / Arena -> SWE-bench`：让评测从静态单分数走向多指标、人类偏好和真实环境。
- `AlphaZero / MuZero -> AlphaFold2 / AlphaGeometry`：说明搜索、自博弈、科学结构与符号验证仍是通往更通用 AI 的关键。

---

## 九、最后的建议

如果你的目标是“真正看懂今天的大模型世界”，不要把论文当成孤立点，而要按下面的方式串起来：

1. 先理解预训练范式：`BERT / GPT / T5`
2. 再理解数据与规模规律：`GPT-3 / Scaling Laws / Chinchilla / PaLM / OLMo`
3. 再理解对齐与产品化：`Human Feedback / FLAN / InstructGPT / DPO`
4. 再理解推理训练：`CoT / Verifier / Process Supervision / GRPO / DeepSeek-R1`
5. 再理解知识与行动：`DPR / RAG / WebGPT / Toolformer / ReAct / Self-RAG`
6. 再理解工程现实：`Megatron / ZeRO / RoPE / FlashAttention / GQA / vLLM`
7. 最后按方向扩展：`多模态 / 生成模型 / Agent / 评测 / 可解释性 / AI for Science`

真正值得反复读的，不是“论文数量”，而是这些论文背后的八条总问题：

- 模型如何获得知识
- 模型如何随着规模变强
- 数据质量、来源和开放性如何改变结论
- 模型如何学会遵循人类意图
- 模型如何进行更可靠的推理
- 模型如何在现实算力下被训练和部署
- 模型如何连接工具、环境和其他模态
- 我们如何知道评测可信、行为安全、机制可理解

只要这八个问题吃透，后面再看新论文，基本都能迅速定位它是在补哪一块，也更不容易把一次榜单提升误判成范式突破。
