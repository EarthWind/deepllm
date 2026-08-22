# 论文原理文档索引

`transformer_llm_papers_guide_zh.md` 现收录 30 篇主论文与 60 篇扩展阅读（共 90 篇，时间边界为 2025-12-31）。本目录当前提供其中 30 篇主论文的独立详细原理文档，并额外补充 `Transformer`、`ULMFiT`、`ELMo`、`RoBERTa`、`BART`、`Sentence-BERT`、`DPR`、`The Pile`、`Foundation Models Report`、`OLMo`、`Megatron-LM`、`ZeRO`、`GPTQ`、`Speculative Decoding`、`GQA`、`AWQ`、`FlashAttention-2`、`Mistral 7B`、`DeepSeek-V2`、`DeepSeek-V3`、`Learning to Summarize from Human Feedback`、`Codex / HumanEval`、`Training Verifiers`、`WebGPT`、`STaR`、`GPT-4 Technical Report`、`DeepSeekMath`、`Scaling LLM Test-Time Compute Optimally`、`Llama 3`、`Kimi k1.5`、`s1`、`Qwen3`、`TruthfulQA`、`HELM`、`Reflexion`、`Generative Agents`、`Voyager`、`MT-Bench / Chatbot Arena`、`Lost in the Middle`、`SWE-bench`、`Self-RAG`、`AlphaZero`、`MuZero`、`Vision Transformer`、`DDPM`、`CLIP`、`AlphaFold2`、`Decision Transformer`、`Stochastic Parrots`、`Latent Diffusion`、`Toy Models of Superposition`、`Whisper`、`Segment Anything` 与 `AlphaGeometry` 五十四篇基础 / 迁移 / 检索 / 数据 / 治理 / 开放科学 / 对齐 / 真实性与整体评测 / 代码生成 / 仓库级软件工程评测 / 数学推理 / 自举推理 / 搜索决策 / 世界模型 / 浏览 Agent / 社会模拟 / 具身 Agent / 验证器 / 人类偏好评测 / 长上下文评测 / 多模态 / 计算机视觉 / 生成模型 / 科学机器学习 / 大模型系统 / 推理优化 / 离线强化学习 / AI 治理 / 生成模型系统 / 机械可解释性 / 语音 / 视觉基础模型 / 神经符号推理论文，共 84 篇；其余扩展阅读暂以总导读和原文链接为主。
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
- `46_FlashAttention2_2023_原理.md`: `FlashAttention-2` - 非矩阵乘 FLOPs / 序列维 CTA 并行 / Split-Q Warp 工作划分 / A100 利用率
- `48_Mistral_7B_2023_原理.md`: `Mistral 7B` - GQA / Sliding Window Attention / Rolling KV Cache / Chunked Prefill
- `49_DeepSeek_V2_2024_原理.md`: `DeepSeek-V2` - MLA / DeepSeekMoE / Device-Limited Routing / YaRN / GRPO
- `50_DeepSeek_V3_2024_原理.md`: `DeepSeek-V3` - Auxiliary-Loss-Free Balancing / MTP / FP8 / DualPipe
- `51_Learning_to_Summarize_from_Human_Feedback_2020_原理.md`: `Learning to Summarize from Human Feedback` - 人类比较 / 奖励模型 / KL-PPO / Reward Overoptimization
- `52_Codex_HumanEval_2021_原理.md`: `Codex / HumanEval` - 代码预训练 / 函数级生成 / pass@k / 执行式评测
- `53_Training_Verifiers_2021_原理.md`: `Training Verifiers` - GSM8K / outcome verifier / best-of-N / test-time compute
- `54_WebGPT_2021_原理.md`: `WebGPT` - 文本浏览器 / 引用证据 / 人类反馈 / reward-model best-of-N
- `55_STaR_2022_原理.md`: `STaR` - 自生成 rationale / outcome filtering / rationalization / 迭代自训练
- `56_GPT4_2023_原理.md`: `GPT-4 Technical Report` - 多模态输入 / 可预测扩展 / 评测协议 / RLHF 与 RBRM / 系统安全 / 披露边界
- `57_DeepSeekMath_2024_原理.md`: `DeepSeekMath` - 数学网页迭代筛选 / 代码底座 / Continued Pretraining / CoT-PoT-Tool SFT / GRPO
- `58_Scaling_Test_Time_Compute_2024_原理.md`: `Scaling LLM Test-Time Compute Optimally` - 难度条件路由 / PRM 搜索 / Best-of-N / 串行修订 / 预训练与推理 FLOPs 交换
- `59_Llama3_2024_原理.md`: `The Llama 3 Herd of Models` - 15.6T Token 数据治理 / 405B 稠密扩展 / 四维并行 / 六轮后训练 / 长上下文与系统安全
- `60_Kimi_k1.5_2025_原理.md`: `Kimi k1.5` - 128K Long-CoT RL / Online Policy Mirror Descent / Partial Rollout / Length Penalty / Long2short / 多模态推理
- `61_s1_2025_原理.md`: `s1` - 1K 高信息密度推理轨迹 / Completion-only SFT / Budget Forcing / 顺序测试时计算 / Control-Scaling-Performance
- `62_Qwen3_2025_原理.md`: `Qwen3` - 36T Token 三阶段预训练 / Dense + MoE 模型族 / Hybrid Thinking / Thinking Budget / 四阶段后训练 / 强到弱蒸馏
- `63_TruthfulQA_2021_原理.md`: `TruthfulQA` - Imitative Falsehood / Truthfulness + Informativeness / 对抗过滤 / GPT-judge / MC1、MC2 与新版 Binary
- `64_HELM_2022_原理.md`: `HELM` - 场景 taxonomy / 适配协议 / 多指标矩阵 / 校准、鲁棒性、公平性与效率 / 透明可复现评测
- `65_Reflexion_2023_原理.md`: `Reflexion` - Verbal Reinforcement / Self-Reflection / Episodic Memory / 无权重更新的跨试次适应
- `66_Generative_Agents_2023_原理.md`: `Generative Agents` - Memory Stream / 三因素检索 / Reflection / 层级 Planning / 多 Agent 社会模拟
- `67_Voyager_2023_原理.md`: `Voyager` - Automatic Curriculum / Code as Action / Skill Library / Iterative Prompting / 具身终身学习
- `68_MT_Bench_Chatbot_Arena_2023_原理.md`: `MT-Bench / Chatbot Arena` - 多轮开放题 / LLM-as-a-Judge / 匿名人类偏好 / 位置与冗长偏差 / 一致率
- `69_Lost_in_the_Middle_2023_原理.md`: `Lost in the Middle` - 有效上下文 / U 形位置效应 / 多文档 QA / Key-value Retrieval / RAG 截断与重排
- `70_SWE_bench_2023_原理.md`: `SWE-bench` - 真实 GitHub Issue / 仓库级代码定位 / Unified Diff / F2P + P2P / 隔离执行评测
- `71_Self_RAG_2023_原理.md`: `Self-RAG` - Reflection Tokens / 按需检索 / 证据支持度 / Segment-level Beam Search
- `72_AlphaZero_2017_原理.md`: `AlphaZero` - Self-Play / Policy–Value Network / PUCT-MCTS / 广义策略迭代
- `73_MuZero_2019_原理.md`: `MuZero` - Learned Latent Dynamics / Reward–Policy–Value / MCTS / Reanalyze
- `74_Vision_Transformer_2020_原理.md`: `Vision Transformer` - Patch Embedding / Pre-LN Encoder / 大规模视觉预训练 / 迁移学习
- `75_DDPM_2020_原理.md`: `DDPM` - 高斯前向扩散 / 噪声预测 / 变分下界 / Score Matching / 逐步反向生成
- `76_CLIP_2021_原理.md`: `CLIP` - 图文双塔 / 批内对比损失 / 自然语言监督 / Prompt Ensemble / Zero-shot 开放词汇分类
- `77_AlphaFold2_2021_原理.md`: `AlphaFold2` - MSA / Evoformer / Triangle Update / IPA / FAPE / Recycling / 蛋白质结构预测
- `78_Decision_Transformer_2021_原理.md`: `Decision Transformer` - Offline RL / Return-to-go / Causal Transformer / 条件动作生成 / Credit Assignment
- `79_Stochastic_Parrots_2021_原理.md`: `On the Dangers of Stochastic Parrots` - 数据文档 / 代表性 / 环境成本 / 偏见 / 问责 / 社会技术系统
- `80_Latent_Diffusion_2021_原理.md`: `Latent Diffusion` - Autoencoder / Latent Space / U-Net / Cross-Attention / Text-to-Image / Sampling
- `81_Toy_Models_of_Superposition_2022_原理.md`: `Toy Models of Superposition` - Polysemanticity / Sparse Features / Feature Geometry / Phase Change / Mechanistic Interpretability
- `82_Whisper_2022_原理.md`: `Whisper` - Weak Supervision / Multilingual ASR / Encoder-Decoder / Speech Translation / Timestamp / Robustness
- `83_Segment_Anything_2023_原理.md`: `Segment Anything` - Promptable Segmentation / Image Encoder / Prompt Encoder / Mask Decoder / SA-1B / Zero-shot Transfer
- `84_AlphaGeometry_2024_原理.md`: `AlphaGeometry` - Neuro-symbolic Reasoning / Deductive Database / Algebraic Reasoning / Auxiliary Construction / Synthetic Proofs / Beam Search

## 建议阅读顺序

0. 基础骨架：`Transformer`
1. 预训练主线：`ULMFiT -> ELMo -> GPT -> BERT -> RoBERTa -> GPT-2 -> BART -> T5 -> GPT-3`
2. 数据 / 扩展律主线：`Scaling Laws -> The Pile -> Foundation Models Report -> Chinchilla -> PaLM -> OLMo -> Llama 3 -> Qwen3`
3. 对齐主线：`Learning to Summarize from Human Feedback -> TruthfulQA -> FLAN -> InstructGPT -> Constitutional AI -> DPO -> Llama 3 -> Kimi k1.5 -> Qwen3`
4. 推理主线：`Training Verifiers -> Chain-of-Thought -> STaR -> Self-Consistency -> Let's Verify Step by Step -> Tree of Thoughts -> DeepSeekMath -> Scaling LLM Test-Time Compute Optimally -> Kimi k1.5 -> s1 -> Qwen3 -> DeepSeek-R1`
5. 工程主线：`Megatron-LM -> ZeRO -> RoPE -> FlashAttention -> FlashAttention-2 -> LoRA -> GPTQ -> AWQ -> Speculative Decoding -> GQA -> QLoRA -> LLaMA -> Mistral 7B -> Mixtral -> Llama 3 -> DeepSeek-V2 -> DeepSeek-V3 -> Kimi k1.5 -> Qwen3`
6. Agent / 检索 / 多模态补充：`Sentence-BERT -> DPR -> RAG -> WebGPT -> Toolformer -> ReAct -> Self-RAG -> Generative Agents -> Reflexion -> Voyager -> GPT-4 -> Gemini -> Kimi k1.5 -> Qwen3 -> Mamba`
7. 基础模型治理与开放科学主线：`The Pile -> Foundation Models Report -> TruthfulQA -> HELM -> OLMo -> Learning to Summarize from Human Feedback -> InstructGPT -> Constitutional AI -> Llama 3`
8. 代码生成与执行评测主线：`GPT-3 -> Codex / HumanEval -> SWE-bench -> Reflexion -> Self-Consistency -> Let's Verify Step by Step -> DeepSeekMath -> Llama 3 -> Scaling LLM Test-Time Compute Optimally -> Kimi k1.5 -> s1 -> Qwen3`
9. 真实性、整体评测与证据化问答主线：`Foundation Models Report -> TruthfulQA -> HELM -> WebGPT -> Self-RAG -> InstructGPT -> GPT-4 -> MT-Bench / Chatbot Arena -> OLMo`
10. 长上下文容量与有效利用主线：`Transformer -> RoPE -> FlashAttention -> FlashAttention-2 -> GQA -> Lost in the Middle -> RAG -> Self-RAG`
11. 搜索、世界模型、验证与自我改进主线：`AlphaZero -> MuZero -> Training Verifiers -> Self-Consistency -> Let's Verify Step by Step -> Tree of Thoughts -> Scaling LLM Test-Time Compute Optimally -> DeepSeek-R1`
12. 视觉与多模态主线：`Transformer -> Vision Transformer -> CLIP -> GPT-4 -> Gemini -> Kimi k1.5 -> Qwen3`
13. 生成模型主线：`VAE / ELBO -> Diffusion Probabilistic Model -> NCSN -> DDPM -> DDIM -> Improved DDPM -> Score SDE -> Classifier Guidance -> Latent Diffusion`
14. 科学与几何建模主线：`Transformer -> AlphaFold1 -> AlphaFold2 -> AlphaFold-Multimer -> OpenFold`

## 说明

- 每份文档都采用统一结构，便于横向比较。
- 每篇均补了链接与代表公式，适合做逐篇精读起点。
- 内容聚焦原理、方法逻辑、影响和阅读建议，不追求逐节翻译原论文。
- 如果后续需要，我可以继续补每篇的逐节拆解版、图解版或公式推导版。
