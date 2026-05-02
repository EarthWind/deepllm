# 论文原理文档索引

本目录为 `transformer_llm_papers_guide_zh.md` 中 30 篇主论文的独立详细原理文档。
增强版已补充：`论文链接`、`代表公式 / 关键表达`、`前置阅读`、`读完接着看`。

## 文件列表

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

## 建议阅读顺序

1. 预训练主线：`BERT -> GPT -> GPT-2 -> T5 -> GPT-3`
2. 扩展律主线：`Scaling Laws -> Chinchilla -> PaLM`
3. 对齐主线：`FLAN -> InstructGPT -> Constitutional AI -> DPO`
4. 推理主线：`Chain-of-Thought -> Self-Consistency -> Let's Verify Step by Step -> Tree of Thoughts -> DeepSeek-R1`
5. 工程主线：`RoPE -> FlashAttention -> LoRA -> QLoRA -> LLaMA -> Mixtral`
6. Agent / 检索 / 多模态补充：`RAG -> Toolformer -> ReAct -> Gemini -> Mamba`

## 说明

- 每份文档都采用统一结构，便于横向比较。
- 每篇均补了链接与代表公式，适合做逐篇精读起点。
- 内容聚焦原理、方法逻辑、影响和阅读建议，不追求逐节翻译原论文。
- 如果后续需要，我可以继续补每篇的逐节拆解版、图解版或公式推导版。
