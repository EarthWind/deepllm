# 论文原理文档索引

`transformer_llm_papers_guide_zh.md` 现收录 30 篇主论文与 60 篇扩展阅读（共 90 篇，时间边界为 2025-12-31）。本目录当前提供其中 30 篇主论文的独立详细原理文档，并额外补充 `Transformer`、`ULMFiT`、`ELMo`、`RoBERTa`、`BART`、`Sentence-BERT`、`DPR`、`The Pile`、`Foundation Models Report`、`OLMo`、`Megatron-LM`、`ZeRO`、`GPTQ`、`Speculative Decoding`、`GQA` 与 `AWQ` 十六篇基础 / 迁移 / 检索 / 数据 / 治理 / 开放科学 / 大模型系统 / 推理优化论文，共 46 篇；其余扩展阅读暂以总导读和原文链接为主。
增强版已补充：`论文链接`、`代表公式 / 关键表达`、`前置阅读`、`读完接着看`。

## 文件列表

- `00_Transformer_2017_原理.md`: `Transformer` - 注意力 / 统一骨架
- `01_BERT_2018_原理.md`: `BERT` - 预训练 / 编码器
- `02_GPT_2018_原理.md`: `GPT` - 预训练 / 解码器
- `03_GPT2_2019_原理.md`: `GPT-2` - 大规模生成式预训练
- `04_T5_2020_原理.md`: `T5` - 统一任务接口 / 编码器-解码器
- `05_GPT3_2020_原理.md`: `GPT-3` - 规模扩展 / 上下文学习
- `06_Scaling_Laws_2020_原理.md`: `Scaling Laws` - 扩展律
- `07_RAG_2020_原理.md`: `RAG` - 检索增强生成
- `08_FLAN_2021_原理.md`: `FLAN` - 指令微调
- `09_RoFormer_RoPE_2021_原理.md`: `RoFormer / RoPE` - 位置编码 / 长上下文
- `10_InstructGPT_2022_原理.md`: `InstructGPT` - 对齐 / RLHF
- `11_Chain_of_Thought_2022_原理.md`: `Chain-of-Thought` - 推理提示
- `12_Chinchilla_2022_原理.md`: `Chinchilla` - 扩展律 / 训练配方
- `13_PaLM_2022_原理.md`: `PaLM` - 超大规模训练
- `14_FlashAttention_2022_原理.md`: `FlashAttention` - 系统优化 / 注意力实现
- `15_LLaMA_2023_原理.md`: `LLaMA` - 开源基础模型
- `16_Switch_Transformer_2021_原理.md`: `Switch Transformer` - MoE / 稀疏激活
- `17_LoRA_2022_原理.md`: `LoRA` - 参数高效微调
- `18_Self_Consistency_2022_原理.md`: `Self-Consistency` - 推理时策略
- `19_Constitutional_AI_2023_原理.md`: `Constitutional AI` - 对齐 / AI 反馈
- `20_Toolformer_2023_原理.md`: `Toolformer` - 工具使用
- `21_ReAct_2023_原理.md`: `ReAct` - Agent / 推理与行动
- `22_Self_Instruct_2023_原理.md`: `Self-Instruct` - 数据合成 / 指令对齐
- `23_DPO_2023_原理.md`: `DPO` - 偏好优化
- `24_QLoRA_2023_原理.md`: `QLoRA` - 低成本微调 / 量化
- `25_Lets_Verify_Step_by_Step_2023_原理.md`: `Let's Verify Step by Step` - 过程监督 / 推理训练
- `26_Tree_of_Thoughts_2023_原理.md`: `Tree of Thoughts` - 搜索式推理
- `27_Mixtral_2024_原理.md`: `Mixtral` - 开源 MoE
- `28_Gemini_2023_原理.md`: `Gemini` - 多模态基础模型
- `29_Mamba_2023_原理.md`: `Mamba` - 后 Transformer 路线
- `30_DeepSeek_R1_2025_原理.md`: `DeepSeek-R1` - Reasoning Model / 强化学习
- `31_ULMFiT_2018_原理.md`: `ULMFiT` - 通用语言模型微调 / 迁移学习
- `32_ELMo_2018_原理.md`: `ELMo` - 深层上下文词表示 / 特征式迁移
- `33_RoBERTa_2019_原理.md`: `RoBERTa` - BERT 预训练配方 / 动态掩码 / 充分训练
- `34_BART_2019_原理.md`: `BART` - 去噪序列到序列预训练 / 理解与生成统一
- `35_Sentence_BERT_2019_原理.md`: `Sentence-BERT` - 共享权重双塔 / 句向量 / 语义检索
- `36_DPR_2020_原理.md`: `DPR` - 问题 / 段落双编码器 / in-batch negatives / 稠密检索
- `37_The_Pile_2020_原理.md`: `The Pile` - 多源预训练语料 / 数据混合 / 去重与治理
- `38_Foundation_Models_Report_2021_原理.md`: `Foundation Models Report` - 涌现 / 同质化 / 生态系统 / 评测与治理
- `39_OLMo_2024_原理.md`: `OLMo` - 开放数据 / 训练日志 / 中间检查点 / 可复现语言模型科学
- `40_Megatron_LM_2019_原理.md`: `Megatron-LM` - 层内张量并行 / 词表并行 / 混合数据并行 / 大模型训练系统
- `41_ZeRO_2019_原理.md`: `ZeRO` - 数据并行状态分片 / Optimizer-Gradient-Parameter Partitioning / ZeRO-R / 显存优化
- `42_GPTQ_2022_原理.md`: `GPTQ` - Weight-only PTQ / 输入 Hessian / 逐列误差补偿 / 3-bit 与 4-bit 推理量化
- `43_Speculative_Decoding_2022_原理.md`: `Speculative Decoding` - 小模型起草 / 大模型并行验证 / 精确随机采样 / 自回归低延迟推理
- `44_GQA_2023_原理.md`: `GQA` - Query 分组共享 K/V / KV Cache 与带宽压缩 / MHA Checkpoint Uptraining
- `45_AWQ_2023_原理.md`: `AWQ` - Activation-aware Weight-only PTQ / 显著通道缩放 / INT3/INT4 Group Quantization / TinyChat

## 建议阅读顺序

0. 基础骨架：`Transformer`
1. 预训练主线：`ULMFiT -> ELMo -> GPT -> BERT -> RoBERTa -> GPT-2 -> BART -> T5 -> GPT-3`
2. 数据 / 扩展律主线：`Scaling Laws -> The Pile -> Foundation Models Report -> Chinchilla -> PaLM -> OLMo`
3. 对齐主线：`FLAN -> InstructGPT -> Constitutional AI -> DPO`
4. 推理主线：`Chain-of-Thought -> Self-Consistency -> Let's Verify Step by Step -> Tree of Thoughts -> DeepSeek-R1`
5. 工程主线：`Megatron-LM -> ZeRO -> RoPE -> FlashAttention -> LoRA -> GPTQ -> AWQ -> Speculative Decoding -> GQA -> QLoRA -> LLaMA -> Mixtral`
6. Agent / 检索 / 多模态补充：`Sentence-BERT -> DPR -> RAG -> Toolformer -> ReAct -> Gemini -> Mamba`
7. 基础模型治理与开放科学主线：`The Pile -> Foundation Models Report -> OLMo -> InstructGPT -> Constitutional AI`

## 说明

- 每份文档都采用统一结构，便于横向比较。
- 每篇均补了链接与代表公式，适合做逐篇精读起点。
- 内容聚焦原理、方法逻辑、影响和阅读建议，不追求逐节翻译原论文。
- 如果后续需要，我可以继续补每篇的逐节拆解版、图解版或公式推导版。
