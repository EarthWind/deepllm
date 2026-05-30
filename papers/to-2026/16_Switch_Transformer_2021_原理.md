# Switch Transformer 原理导读

**论文标题**：`Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity`
**年份**：`2021`
**主题**：`MoE / 稀疏激活`
**定位**：现代大规模 MoE 路线的关键起点之一。
**论文链接**：[arXiv](https://arxiv.org/abs/2101.03961)

## 1. 代表公式 / 关键表达
- MoE 的简化表达：`y = E_{g(x)}(x)`，其中门控 `g(x)` 为 token 选择一个专家。
- Switch 的关键是 top-1 路由，减少训练复杂度和通信负担。

## 2. 这篇论文要解决什么问题
- 稠密模型若继续放大到更高参数量，训练和推理成本会快速失控。
- 大家希望增加模型容量，但不想每个 token 都激活全部参数。
- 作者要解决的是如何通过稀疏激活实现“更大总参数、可控计算量”。

## 3. 核心思想
- 使用 mixture-of-experts，把部分前馈层替换为多个专家网络。
- 每个 token 只路由到极少数专家，常见是 top-1 或少量 top-k。
- 通过简单路由策略降低稀疏模型训练复杂度。

## 4. 方法原理拆解
- 门控网络根据 token 表示决定它去哪个专家处理。
- 这样总参数量可以非常大，但单次前向只激活少数专家，因此计算成本接近稠密小模型。
- 关键难点在于负载均衡、路由稳定性和分布式通信。

## 5. 代码实现结构
- Switch Transformer 的实现重点是把普通 FFN 层替换成 `MoE Layer`。
- 因此代码结构通常拆成 `Router`、`Experts`、`Dispatch/Combine`、`Load Balancing Loss` 四部分。
- 从源码角度看，它不是整模型都变了，而是某些 Transformer block 中的前馈层被稀疏专家层替换。

### 5.1 Router
- Router 接收每个 token 的隐藏状态，输出该 token 应该去哪个专家。
- Switch 的关键是 `top-1 routing`，也就是每个 token 只选一个专家。
- 代码上常见写法是：
  `expert_idx = argmax(router_logits, dim=-1)`

### 5.2 Experts
- 每个 expert 本质上通常就是一个前馈网络 MLP。
- 所有 expert 结构相同，但参数不同，因此可以形成专业化分工。
- 也就是说，MoE 并不是把 attention 稀疏化，而是把 FFN 稀疏化。

### 5.3 Dispatch 与 Combine
- 选好专家后，需要把属于同一专家的 token 聚到一起计算。
- 然后再把每个 expert 的输出散回原 token 顺序。
- 这一层是 MoE 代码最容易复杂化的地方，因为它牵涉到张量重排与跨设备通信。

### 5.4 一个最小 MoE 骨架

```python
class SwitchFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            FeedForward(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def forward(self, x):
        router_logits = self.router(x)
        expert_idx = router_logits.argmax(dim=-1)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = expert_idx == i
            if mask.any():
                out[mask] = expert(x[mask])
        return out, router_logits
```

### 5.5 负载均衡损失
- 如果没有额外约束，router 可能把绝大多数 token 都发给少数专家。
- 所以实现里通常还会加一个 auxiliary loss，鼓励专家负载更均匀。
- 这部分虽然不是主任务 loss，但对训练稳定性非常重要。

### 5.6 实现时最值得注意的点
- 真正难点不是写出 router，而是高效实现 token dispatch 和多设备专家并行。
- top-1 routing 虽然简化了复杂度，但仍会面临专家过载和 token 丢弃问题。
- 如果自己做 toy 实验，先在单机上实现逻辑正确性，再考虑分布式优化会更稳。

## 6. 训练或推理范式
- MoE 训练除了常规优化，还要处理专家不均衡和通信开销。
- Switch Transformer 试图用更简单的 top-1 路由减轻训练复杂度。
- 它让 trillion-parameter 级别容量变得更具可行性。

## 7. 为什么它有效
- 不同 token 不必共享完全相同的计算路径，专业化专家能提高容量利用率。
- 稀疏激活把“总参数量”和“单步计算量”部分解耦。
- 这为更大规模模型提供了新的扩展维度。

## 8. 局限与争议
- MoE 引入了额外系统复杂度，尤其是多设备通信和负载均衡。
- 专家路由不稳定会导致训练困难。
- 推理部署也比稠密模型更复杂，工程成本不可忽视。

## 9. 对后续研究的影响
- 为后续 Mixtral、DeepSeek-V2/V3 等 MoE 模型提供方法学基础。
- 让“稀疏扩展”成为与“稠密扩展”并列的主流路线。
- 推动前沿模型在容量与效率之间寻找新平衡。

## 10. 你应该怎样读这篇
- 读这篇时要理解它解决的是扩展成本问题，而不是简单追求新奇结构。
- 如果你关注前沿开源模型的 MoE 现象，这篇是源头级文献。
- 建议与 Mixtral 对照阅读，理解研究原型如何走向可用开源模型。

## 11. 前置阅读
- `Scaling Laws`
- `GPT-3`

## 12. 读完接着看
- `Mixtral`
