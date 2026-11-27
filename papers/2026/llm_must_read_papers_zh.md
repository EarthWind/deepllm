# 2026 年 AI 最值得阅读的论文清单

> 版本日期：2026-11-27
> 收录时间：**2026-01-01 至 2026-11-27**
> 日期口径：论文或技术报告的**首个公开版本日期**
> 覆盖方向：语言模型、推理与 Agent、多模态生成、World Model、机器人与具身智能、AI for Science、安全与评测

这是一份“少而精”的 2026 年 AI 阅读清单（截至 2026-11-27），不是 arXiv 论文全集，也不是按模型榜单排序。入选论文应至少满足一项：提出有辨识度的方法、公开值得复用的训练或评测体系、把 AI 推进到新的任务层级，或对一个快速发展的方向给出重要的定义和反思。

由于这些工作公开时间都很短，尚未积累稳定的长期引用和独立复现结果，“必读”表示当前的方法价值和研究价值较高，不表示已经成为经时间检验的经典。

## 一、筛选标准

- 首个公开版本位于 2026-01-01 至 2026-11-27；
- 优先采用论文原文或官方技术报告，不以新闻热度代替技术价值；
- 优先选择包含方法细节、消融、开放权重/代码/数据或可执行评测的工作；
- 不因模型榜单领先而自动收录，重点判断新意来自架构、数据、训练目标、环境还是系统实现；
- 同类工作较多时，只保留方法最有代表性或公开程度较高的少数论文。

## 二、论文总表

按首个公开版本日期排序。

| # | 首发日期 | 方向 | 论文 | 为什么值得读 | 建议 |
|---:|:---:|---|---|---|:---:|
| 1 | 2026-01-06 | LLM / 系统 | [MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780) | 将混合滑窗/全局 attention、MTP、多教师 on-policy distillation 与原生 speculative decoding 组合在同一开放模型中。 | 必读 |
| 2 | 2026-01-30 | AI 安全 | [How should AI Safety Benchmarks Benchmark Safety?](https://arxiv.org/abs/2601.23112) | 系统审查 210 个安全基准，从风险管理和测量理论解释为何“有安全分数”不等于“测到了安全”。 | 安全方向必读 |
| 3 | 2026-02-02 | 多模态 Agent | [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276) | 联合图文预训练、SFT 和 RL，并将动态任务分解与并行 Agent Swarm 纳入统一报告。 | 必读 |
| 4 | 2026-02-09 | World Model | [stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation](https://arxiv.org/abs/2602.08968) | 提供模块化、经过测试的 World Model 研究生态，覆盖数据收集、环境、规划算法和标准基线。 | 实践优先 |
| 5 | 2026-02-09 | World Model 评测 | [WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models](https://arxiv.org/abs/2602.08971) | 同时评测视觉质量和下游决策效用，揭示“视频看起来好”不等于“能帮助具身任务”。 | 评测优先 |
| 6 | 2026-02-10 | AI for Mathematics | [Towards Autonomous Mathematics Research](https://arxiv.org/abs/2602.10177) | Aletheia 把生成、验证、修订、工具使用和长程证明组织成数学研究 Agent，并讨论结果的新颖性与自主等级。 | 必读 |
| 7 | 2026-02-11 | 机器人 / VLA | [LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer](https://arxiv.org/abs/2602.10556) | 把低层机器人动作直接表示为语言，使 VLA 能零样本迁移到未见过的机器人本体。 | 具身方向必读 |
| 8 | 2026-02-28 | 代码 Agent | [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729) | 使用可执行、可验证的代码任务和环境反馈训练 80B-A3B coding agent，代表“小活跃参数 + 强训练配方”。 | 必读 |
| 9 | 2026-03-04 | 多模态推理 | [Phi-4-reasoning-vision-15B Technical Report](https://arxiv.org/abs/2603.03975) | 通过数据清洗、动态高分辨率视觉和 reasoning/non-reasoning 混训推进紧凑型多模态推理模型。 | 方向选读 |
| 10 | 2026-03-31 | 自动化 AI 研究 | [Towards End-to-End Automation of AI Research](https://arxiv.org/abs/2606.15497) | The AI Scientist 覆盖选题、编码、实验、分析、写作和评审全流程，同时暴露自动生成研究的质量与治理问题。 | 必读 |
| 11 | 2026-04-14 | 架构 / 系统 | [Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning](https://arxiv.org/abs/2604.12374) | 将 hybrid Mamba–Transformer、LatentMoE、NVFP4 预训练和 MTP speculative decoding 放进一个开放模型。 | 必读 |
| 12 | 2026-04-22 | 3D 生成 | [Seed3D 2.0: Advancing High-Fidelity Simulation-Ready 3D Content Generation](https://arxiv.org/abs/2605.13862) | 从单物体生成推进到可用于仿真的材质、场景布局、部件分解和关节化 3D 内容。 | 生成/仿真优先 |
| 13 | 2026-05-05 | AI for Neuroscience | [A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience](https://arxiv.org/abs/2605.04326) | TRIBE v2 用视听语言三模态预测跨刺激、任务和受试者的脑活动，并尝试复现经典神经科学现象。 | AI for Science 优先 |
| 14 | 2026-05-11 | 图像生成 | [Qwen-Image-2.0 Technical Report](https://arxiv.org/abs/2605.10730) | 用 Qwen3-VL 条件编码器与多模态 DiT 统一高保真生成和编辑，重点解决长文本、多语言排版与复杂指令。 | 生成方向优先 |
| 15 | 2026-05-28 | 机器人 / VLA | [Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments](https://arxiv.org/abs/2605.30280) | 用统一动作解码器和 embodiment-aware prompt 连接操作、导航、轨迹预测及多种机器人本体。 | 具身方向必读 |
| 16 | 2026-06-09 | 统一多模态 | [ARM: An AutoRegressive Large Multimodal Model with Unified Discrete Representations](https://arxiv.org/abs/2606.11188) | 用离散视觉 tokenizer 和 next-token prediction 统一图像理解、生成与编辑，并研究 RL 带来的跨任务协同。 | 多模态优先 |
| 17 | 2026-06-10 | 推理 / RL | [Verifiable Environments Are LEGO Bricks: Recursive Composition for Reasoning Generalization](https://arxiv.org/abs/2606.12373) | RACES 将可验证环境视为可递归组合的模块，探索从扩充题量转向扩展环境结构。 | 必读 |
| 18 | 2026-07-07 | World Model | [A Definition and Roadmap for World Models](https://arxiv.org/abs/2607.06401) | 尝试统一视频生成、模型式 RL、机器人和 Physical AI 中含义混乱的 World Model，并给出阶段性路线图。 | 建立全局视野 |
| 19 | 2026-07-10 | 小模型训练 | [Index SLM Technical Report](https://arxiv.org/abs/2607.09885) | 公开 1.9B 模型关于深度、学习率、数据质量和预训练指令数据的受控实验，适合研究小模型训练配方。 | 实证选读 |
| 20 | 2026-08-02 | World Model | [MiniWorld: Democratizing the Training of Video World Models from Scratch](https://arxiv.org/abs/2608.01127) | 给出可复现的流式视频 World Model 从零训练配方：block-causal Video DiT + Flow Matching、chunk 级单调噪声调度和滚动 KV cache，单台 8 卡服务器数天可训完。 | 实践优先 |
| 21 | 2026-08-06 | 推理 / 后训练 | [On-Policy Self-Distillation without Any Supervision](https://arxiv.org/abs/2608.06296) | U-OPSD 用多数投票构造伪解，只在与伪解不一致的样本上做 on-policy 自蒸馏，无需任何标注即可接近或超过 GRPO。 | 方向选读 |
| 22 | 2026-08-10 | World Model | [Learning How the World Evolves: Extrapolative Video World Models via Latent Dynamics Reasoning](https://arxiv.org/abs/2608.09926) | LDR 在结构化潜空间中对低阶运动做数值积分、网络只学高阶残差，OOD 外推远强于视频扩散基线，直接质疑“视频扩散 = World Model”。 | 必读 |
| 23 | 2026-08-25 | 机器人 / World Model 评测 | [Do Robotic World Models Really Follow Actions? Diagnosing and Aligning Action-Conditioned Generation for Policy Learning](https://arxiv.org/abs/2608.24885) | WorldEcho 用末端 SE(3) 轨迹对齐检验动作条件 World Model 是否真按动作生成，发现其在专家分布外普遍失效，并提出 WorldSync 对齐方法。 | 具身方向必读 |
| 24 | 2026-08-27 | AI for Science | [Accelerating Scientific Research with Gemini in the Real-World](https://arxiv.org/abs/2608.26701) | 记录 Co-Scientist 等系统在材料、生物等真实科研中的闭环案例，并用 30 位专家盲审生成论文，检验幻觉与抄袭控制。 | AI for Science 优先 |
| 25 | 2026-08-31 | 架构 / 预训练 | [On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/abs/2608.30320) | 系统消融 Gated DeltaNet/全注意力 3:1 混合、Qwen Sparse Attention 与 Gated Residual，并给出 loss 与下游指标背离、Muon 下 LR/batch 变化等训练经验。 | 必读 |
| 26 | 2026-09-03 | AI 安全 / 自动科研 | [A Case Study on Emergent Cheating and Whistleblowing in Autonomous Research Swarms](https://arxiv.org/abs/2609.04170) | 100 个 LLM Agent 组成的数学研究集体中，评测漏洞利用经共享知识库扩散，同时自发出现举报与审计，是多 Agent reward hacking 的一手案例。 | 安全方向必读 |
| 27 | 2026-09-17 | LLM / 系统 | [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/abs/2609.19969) | 通过跨层 KV 复用、FP4 KV cache 与 SWA Bounded Replay，把 1M 上下文 MoE 的 KV 占用压到前代约 1/4，长上下文推理成本的代表性工作。 | 必读 |
| 28 | 2026-09-18 | 代码 Agent / RL 环境 | [CodeMidas: Scaling Agentic Coding RL Environments from Code Itself](https://arxiv.org/abs/2609.22068) | 不依赖 issue/commit，只从源码出发由 Agent 写行为规格并生成可执行测试，构造 5,545 个跨 23 种语言的 RL 任务，并给出任务规模消融。 | 方向选读 |
| 29 | 2026-09-26 | 推理 / RL | [Rethinking Training-Inference Mismatch in LLM Reinforcement Learning: Where It Arises and How to Correct It](https://arxiv.org/abs/2609.32444) | 把训推引擎概率差建模为 log-odds 偏移，提出带理论保证的 Calibrated Importance Sampling，直击 MoE RL 不稳定的工程根源。 | 方向选读 |
| 30 | 2026-09-29 | 机器人 / VLA | [Rho: A Foundation for Efficiently Adaptable VLA Models](https://arxiv.org/abs/2609.38164) | 开放权重的双臂 VLA 家族，系统消融 action expert 与“本体 midtraining”，并用小潜策略为冻结动作专家选噪声，约 15 条纠正轨迹即可在线适配。 | 具身方向必读 |
| 31 | 2026-09-30 | 机器人 / 人形 | [IronMind: Scaling Humanoid Dexterous Manipulation via Camera-Space Ego-Centric Pretraining](https://arxiv.org/abs/2609.39403) | 用相机空间动作表示统一第一人称人类视频与机器人数据，给出 250→10000 小时预训练的清晰 scaling 曲线和 OOD 真机成功率跃升。 | 具身方向必读 |
| 32 | 2026-10-01 | 后训练 / 评测 | [Sharpening Tax in Post-Training](https://arxiv.org/abs/2610.01509) | 在 14 组 base/post-trained 模型对上发现后训练提升 pass@1 却损失 pass@K 覆盖，提出可低成本估计的 Sharpening Tax 指标。 | 方向选读 |
| 33 | 2026-10-04 | 视频 / 音频生成 | [Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation](https://arxiv.org/abs/2610.05608) | MIT 协议开源的音视频同步生成模型，双流 CrossDiT 连接视频与音频，完整公开预训练、联合训练、SFT、RL 到蒸馏的流程。 | 生成方向优先 |
| 34 | 2026-10-08 | 推理 / RL | [MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement](https://arxiv.org/abs/2610.11959) | 从异步大批量、百万 token 上下文和多域 Agent 环境三方面扩展 RL，并公开冻结 router、防 reward hacking、训推一致性等稳定性做法与 RL 环境/框架。 | 必读 |
| 35 | 2026-10-15 | AI for Science | [Discovering state-of-the-art quantum circuits with large language models](https://arxiv.org/abs/2610.22367) | LLM 在进化框架中提出量子电路并由独立评估器打分，8 个任务中 7 个在 Toffoli 数与量子比特数上刷新已知最好结果（资源估计，非硬件实验）。 | AI for Science 优先 |
| 36 | 2026-10-27 | AI 安全 | [Agentic Backdoors via Pretraining Poisoning](https://arxiv.org/abs/2610.37741) | 从零预训练并后训练 CLI Agent，证明污染 0.1% 预训练语料即可植入远程代码执行后门，且攻击成功率随模型规模上升；代码开源。 | 安全方向必读 |
| 37 | 2026-10-27 | AI 安全 / 人机协作 | [How AI agents shape experts' judgment calls](https://www.anthropic.com/research/agents-judgment-calls) | 50 名专业分析师用编码 Agent 做因果分析的随机对照试验：Agent 被设定倾向后，专家多数判断随之偏移且很少意识到来源。 | 安全方向必读 |
| 38 | 2026-10-28 | 架构 / 系统 | [Nemotron 3.5 Super and Lightning: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Models for Agentic AI](https://arxiv.org/abs/2610.38095) | 延续 Nemotron 3 的混合 Mamba-Transformer MoE 路线，覆盖 SFT/RLVR/RLHF/多教师 on-policy 蒸馏、MTP 投机解码和 NVFP4 部署细节。 | 实践优先 |
| 39 | 2026-11-16 | 开放模型 / 预训练 | [K2-Horizon: A Frontier Fleet of Open Source Language Models](https://arxiv.org/abs/2611.21003) | 一次公开 0.9B–375B-A23B 共 6 个全开源模型及中间 checkpoint、数据、日志和训练/RL 代码，并提出 Mixture-of-Value Attention，适合做受控 scaling 研究。 | 必读 |
| 40 | 2026-11-19 | 全模态理解 | [WeMM-Video: Pushing the Frontiers of Open-Source Omni-Modal Video Understanding](https://arxiv.org/abs/2611.26409) | 开源 2B–35B-A3B 音视频全模态理解模型家族，支持思考/非思考双模式，并开源 120+ 基准的评测复现流程。 | 多模态优先 |

## 三、如果时间有限，只读这 15 篇

1. [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780)：高效 LLM 架构、训练和推理解码协同。
2. [How should AI Safety Benchmarks Benchmark Safety?](https://arxiv.org/abs/2601.23112)：理解安全评测为什么容易失真。
3. [Kimi K2.5](https://arxiv.org/abs/2602.02276)：多模态模型与并行 Agent 编排。
4. [Towards Autonomous Mathematics Research](https://arxiv.org/abs/2602.10177)：AI 从解题走向研究的代表案例。
5. [LAP](https://arxiv.org/abs/2602.10556)：机器人跨本体零样本迁移。
6. [Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729)：可执行环境中的代码 Agent 训练。
7. [Towards End-to-End Automation of AI Research](https://arxiv.org/abs/2606.15497)：端到端自动化科研及其局限。
8. [Nemotron 3 Super](https://arxiv.org/abs/2604.12374)：混合架构、MoE、低精度和推理加速。
9. [Qwen-VLA](https://arxiv.org/abs/2605.30280)：统一操作、导航和多机器人本体。
10. [RACES](https://arxiv.org/abs/2606.12373)：可组合验证环境与推理泛化。
11. [Qwen3.8-Next 架构设计](https://arxiv.org/abs/2608.30320)：混合线性/稀疏注意力的系统消融与训练稳定性经验。
12. [LDR](https://arxiv.org/abs/2608.09926)：World Model 是否真的学到了动力学。
13. [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969)：百万上下文下的 KV cache 极限压缩。
14. [MiMo-V2.6](https://arxiv.org/abs/2610.11959)：大规模 Agent RL 的扩展与稳定性配方。
15. [Agentic Backdoors via Pretraining Poisoning](https://arxiv.org/abs/2610.37741)：预训练数据投毒对 Agent 的实际威胁。

## 四、按方向阅读

### 语言模型、推理与 Agent

`MiMo-V2-Flash → Kimi K2.5 → Qwen3-Coder-Next → Nemotron 3 Super → RACES → Index SLM → Qwen3.8-Next → DeepSeek-V4.1-Flash → Nemotron 3.5 → K2-Horizon`

后训练与 RL 支线：`U-OPSD → CodeMidas → Training-Inference Mismatch → Sharpening Tax → MiMo-V2.6`

阅读时比较：混合 attention/SSM、MoE 活跃参数、MTP 的训练与推理复用、KV cache 压缩、Agent 环境构建、训推一致性，以及后训练对 pass@1 与 pass@K 的不同影响。

### World Model、机器人与具身智能

`stable-worldmodel → WorldArena → LAP → Qwen-VLA → Seed3D 2.0 → World Model Roadmap → MiniWorld → LDR → WorldEcho → Rho → IronMind`

阅读时区分三个问题：是否能预测世界、预测是否能帮助决策，以及策略能否迁移到新的任务、环境和机器人本体。

### 多模态理解与生成

`Phi-4-reasoning-vision → Qwen-Image-2.0 → ARM → Kimi K2.5 → Kandinsky 6.0 Video → WeMM-Video`

阅读时比较：连续/离散视觉表示、autoregressive/diffusion 路线、感知与推理耦合，以及生成与编辑能否真正共享表示。

### AI for Science 与自动化研究

`Towards Autonomous Mathematics Research → Towards End-to-End Automation of AI Research → TRIBE v2 → Gemini in the Real-World → LLM 量子电路发现`

阅读时重点检查：结果如何验证、AI 的自主程度如何定义、失败结果是否披露，以及自动化系统是在复用已知模式还是产生可确认的新知识。

### 安全与评测

`How should AI Safety Benchmarks Benchmark Safety? → WorldArena → Towards End-to-End Automation of AI Research → Emergent Cheating in Research Swarms → Agentic Backdoors → How AI agents shape experts' judgment calls`

安全评测论文提供测量框架；WorldArena 与自动科研两篇展示“代理指标可能偏离真实目标”；下半年三篇分别从多 Agent reward hacking、预训练投毒和人类判断被 Agent 塑造三个角度扩展了 Agent 安全问题。

## 五、阅读时统一记录的问题

1. 论文真正改变的是架构、数据、算力、训练目标、环境、工具还是评测？
2. 最强证据来自哪项消融，而不是哪张总榜？
3. 对比是否公平：总参数、活跃参数、训练 token、推理 token 和工具预算是否一致？
4. Agent、RL 或自动科研结果是否依赖特定环境、判分器、提示模板或数据污染？
5. 模型是否开放权重、代码、数据、训练框架和完整评测配置？
6. 论文的核心结论是否已有独立复现，在哪些任务或规模下可能失效？
7. 对 AI for Science，结论是否经过形式验证、实验验证或独立领域专家审核？

## 六、时间边界说明

- 只收录首个公开版本位于 **2026-01-01 至 2026-11-27** 的论文。
- 日期以论文页面的首个公开版本为准，而不是 arXiv 编号月份、会议年份或最后修订日期。
- 例如 `Towards End-to-End Automation of AI Research` 的 arXiv 编号含 `2606`，但页面记录的首次公开日期是 2026-03-31，因此按 3 月排序。
- `Kimi K2.5` 的 v1 在 2026-02-02 发布；后续修订的日期不改变本清单的收录依据。
- `How AI agents shape experts' judgment calls` 未发布在 arXiv，日期以 Anthropic 官方研究页面的发布日期为准。
- 2026-08 之后的条目公开时间更短，部分开放程度（权重、代码、数据）以论文摘要页描述为准，阅读前建议再核对。
- 2026-11-28 及之后首次公开的论文留待下次更新。
