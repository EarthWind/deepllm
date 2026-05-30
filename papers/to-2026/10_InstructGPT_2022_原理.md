# InstructGPT 原理导读

**论文标题**：`Training language models to follow instructions with human feedback`
**年份**：`2022`
**主题**：`对齐 / RLHF`
**定位**：解释聊天模型为何突然变得可用的关键论文，RLHF 主线的经典起点。
**论文链接**：[arXiv](https://arxiv.org/abs/2203.02155)

## 1. 代表公式 / 关键表达
- 奖励模型学习偏好排序，可写成：`P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))`。
- RLHF 阶段可理解为在 KL 约束下最大化期望奖励：`max E[r(x, y)] - β KL(π || π_ref)`。

## 2. 这篇论文要解决什么问题
- 纯预训练模型即便能力强，也常常不按要求回答，容易输出冗长、危险或偏离意图的内容。
- 用户真正需要的是可交互、可控制、能遵循偏好的模型，而不是只会续写的模型。
- 作者要解决的就是“能力”和“可用性”之间的落差。

## 3. 核心思想
- 先用人工演示数据做监督微调，得到基本会听指令的初版模型。
- 再收集人类偏好比较数据，训练奖励模型。
- 最后用 PPO 等强化学习方法，让模型优化人类偏好下的行为。

## 4. 方法原理拆解
- SFT 阶段负责把模型从通用续写器变成基础助手。
- 奖励模型把“哪种回答更好”压缩成可学习的评分函数。
- RL 阶段则让模型在生成空间中偏向高奖励回答，形成更稳定的行为风格。

## 5. 代码实现结构
- InstructGPT 的代码结构不是一个单独模型类，而是一条三阶段训练流水线：`SFT -> Reward Model -> PPO/RLHF`。
- 因此工程实现一般会拆成 `基础策略模型`、`SFT 训练器`、`Reward Model`、`RL Trainer` 四类组件。
- 从代码设计上看，它更像一个训练系统，而不是一个新架构。

### 5.1 基础策略模型
- 底座通常是标准 decoder-only 语言模型，例如 GPT 系列主干。
- 在 SFT 阶段，它仍然只是做条件生成，只不过输入变成了“用户指令/对话上下文”。
- 代码层面一般可直接复用现有 causal LM 模型类。

### 5.2 SFT 训练器
- SFT 阶段把人工演示数据转成 `prompt -> response` 监督样本。
- 训练代码和普通指令微调非常接近，本质上还是 next-token prediction。
- 这一阶段产出的模型通常是后续 Reward Model 和 RL 阶段的起点。

### 5.3 Reward Model
- Reward Model 常共享与策略模型相似的 backbone，但头部不同。
- 它通常接收 `prompt + response`，输出一个标量分数而非词表 logits。
- 偏好对训练时，代码上常写成：
  `score_chosen` 和 `score_rejected` 的对比损失。

### 5.4 PPO / RLHF 训练器
- RL 阶段会让当前策略模型生成回答，再由 Reward Model 打分。
- 同时引入一个参考模型，防止策略偏离原始语言模型分布太远。
- 这一阶段的代码复杂度明显高于 SFT，因为它涉及采样、打分、优势估计、PPO 更新和 KL 约束。

### 5.5 一个最小流程骨架

```python
policy = CausalLM.from_pretrained(...)
sft_policy = train_sft(policy, sft_dataset)

reward_model = RewardModel.from_pretrained(...)
train_reward_model(reward_model, preference_pairs)

ref_policy = copy.deepcopy(sft_policy).eval()
rlhf_policy = train_with_ppo(
    policy=sft_policy,
    ref_policy=ref_policy,
    reward_model=reward_model,
    prompt_dataset=rlhf_prompts,
)
```

### 5.6 Reward Model 的最小形式

```python
class RewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(backbone.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        return self.value_head(hidden[:, -1]).squeeze(-1)
```

### 5.7 实现时最值得注意的点
- 这条流水线最大的复杂度不在模型 forward，而在数据组织与多阶段训练衔接。
- `prompt` 截断策略、response 长度、奖励归一化、KL 系数都会显著影响 RLHF 结果。
- 如果自己实现，最容易先做对的是 `SFT + Reward Model`，PPO 部分通常最难稳定。

## 6. 训练或推理范式
- 三阶段流程分别解决会做、知道什么更好、以及如何稳定朝更好方向优化的问题。
- 这套流程后来成为 RLHF 标准模板，即便很多细节在后续方法中被替换。
- 论文还强调：更小但更对齐的模型，用户偏好上可能胜过更大但未对齐的模型。

## 7. 为什么它有效
- 用户满意度并不等于语言模型对训练语料分布的拟合度，必须显式引入偏好信号。
- 人类比较反馈比绝对打分更稳定、更容易收集。
- RL 允许模型在连续生成空间中学习长程行为偏好，而不只是单点监督。

## 8. 局限与争议
- 成本高，需要大量人工标注和复杂训练管线。
- 奖励模型会引入偏差，甚至可能导致 reward hacking。
- RLHF 强化的是标注偏好，并不自动保证真实性或通用价值一致性。

## 9. 对后续研究的影响
- 直接推动了 ChatGPT 式产品体验的形成。
- 让“对齐”从伦理讨论变成具体工程路线。
- 为后来的 DPO、RLAIF、Constitutional AI 等方法提供了共同参照系。

## 10. 你应该怎样读这篇
- 读这篇时一定区分预训练能力与对齐能力，它们不是一回事。
- 如果你做产品，这篇比单纯看模型分数更重要。
- 它和 FLAN、DPO 连读，能建立后训练全景图。

## 11. 前置阅读
- `GPT-3`
- `FLAN`

## 12. 读完接着看
- `Constitutional AI`
- `DPO`
