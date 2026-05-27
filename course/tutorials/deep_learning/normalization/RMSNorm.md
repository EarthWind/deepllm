# 为什么需要归一化？
在深度神经网络训练中，随着层数加深，每一层的输入分布会随着参数更新而持续变化，这种现象称为内部协变量偏移（Internal Covariate Shift）。它导致：梯度消失或爆炸、训练不稳定、收敛速度慢。

归一化层通过将激活值约束在稳定的分布范围内，有效缓解上述问题，是现代 Transformer 架构不可或缺的组件。

核心思路：将每一层的激活值标准化，使其均值为 0、方差为 1，再通过可学习参数恢复表达能力。


# LayerNorm 回顾
LayerNorm（Ba et al., 2016）在特征维度上对每个样本独立归一化，不依赖 batch 大小，非常适合序列模型：

```
LayerNorm 公式
// 计算均值和方差
μ = 1/n · Σ xᵢ
σ² = 1/n · Σ (xᵢ - μ)²

// 归一化
x̂ᵢ = (xᵢ - μ) / √(σ² + ε)

// 仿射变换（γ, β 可学习）
y = γ · x̂ + β
```

LayerNorm 包含两个核心操作：中心化（减去均值 μ）和缩放（除以标准差 σ），以及两个可学习参数 γ（增益）和 β（偏置）。

# RMSNorm 原理推导
[RMSNorm（Root Mean Square Layer Normalization，Zhang & Sennrich, 2019）](https://arxiv.org/abs/1910.07467) 提出了一个关键问题：LayerNorm 的成功，究竟来自中心化还是缩放？

核心假设：归一化的效果主要来自重缩放（Re-scaling），而非均值中心化。因此，可以去掉均值计算，只保留 RMS 归一化。

**RMS（均方根）的定义**
```
Root Mean Square
RMS(x) = √( 1/n · Σᵢ xᵢ² )

// 等价于：向量 x 的 L2 范数 / √n
RMS(x) = ‖x‖₂ / √n
```

**完整的 RMSNorm 公式**
```
RMSNorm 公式（完整）
// Step 1: 计算均方根（无需计算均值！）
RMS(x) = √( 1/n · Σᵢ xᵢ² + ε )

// Step 2: 归一化
x̄ᵢ = xᵢ / RMS(x)

// Step 3: 仿射变换（只有 γ，无 β！）
RMSNorm(x)ᵢ = γᵢ · x̄ᵢ
```

**与 LayerNorm 的数学关联**
当输入 x 的均值为 0 时，LayerNorm 退化为 RMSNorm。可以用假设检验理解：RMSNorm 是一种"零均值假设"下的 LayerNorm 近似，实践中这个假设在深度网络中近似成立。
```
当 μ → 0 时的化简
// LayerNorm 中的方差
σ² = 1/n · Σ (xᵢ - μ)²

// 当 μ ≈ 0：
σ² ≈ 1/n · Σ xᵢ²  =  RMS(x)²

// 因此 LayerNorm ≈ RMSNorm（μ → 0）
```

# 与 LayerNorm 的对比
| LayerNorm | RMSNorm |
| --- | --- |
| 计算均值 μ 和方差 σ² | 只计算 RMS（均方根） |
| 两遍扫描特征维度 | 一遍扫描即可 |
| 参数：γ 和 β（2n 个） | 参数：仅 γ（n 个）| 
| 中心化 + 缩放 | 只做缩放，无中心化 |
| 表达能力更强（含偏移） | 更快、更简洁 |
| 计算量更大 | 实践中效果相当 | 

| 维度	| LayerNorm | 	RMSNorm |
| --- | --- | --- |
| 计算复杂度 | O(n)，常数更大|	O(n)，约快 10–40% |
| 可学习参数 | γ, β（2n）|	γ（n） |
| 均值中心化 | 有 | 无 |
| 数值稳定性 |稳定 | 同等稳定 |  
| 主流模型采用 | BERT, GPT-2 | LLaMA, Qwen, Gemma, Mistral| 

# PyTorch 实现
```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        # dim: 特征维度; eps: 防止除零
        super().__init__()
        self.eps = eps
        # 唯一的可学习参数：增益 γ，初始化为全 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算 RMS，在最后一个维度（特征维）上
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms  # x̄ = x / RMS(x)

    def forward(self, x: torch.Tensor):
        # 在 float32 下计算以提高精度，再转回原类型
        output = self._norm(x.float()).type_as(x)
        return output * self.weight  # γ · x̄

# 使用示例
norm = RMSNorm(dim=4096)
x = torch.randn(2, 512, 4096)  # [batch, seq_len, dim]
out = norm(x)  # out.shape = (2, 512, 4096)
```

**关键实现细节**
- 注意代码中 `.add(eps).rsqrt()` 的写法：先加 `ε` 再取倒数平方根，等价于 `1 / sqrt(mean + ε)`，这比先 `sqrt` 再除法在数值上更稳定，且硬件友好。
- 将输入 `cast` 到 `float32` 再计算，是 LLaMA 实现中的工程实践，避免 `bfloat16` 精度不足导致的训练不稳定。


# 性能与应用
RMSNorm 约比 LayerNorm 快 10%～40%（取决于硬件和特征维度），原因在于：省去了均值的两次遍历、更少的内存读写、更好的硬件指令对齐。
如今最主流的开源大语言模型均已采用 RMSNorm：
- LLaMA 2/3
- Meta · Pre & Post-norm
- Qwen 2/3
- Alibaba · 全系列
- Mistral
- Mistral AI
- Gemma 2
- Google DeepMind
- DeepSeek
- DeepSeek AI
- Phi-3/4
- Microsoft
放置位置：现代 LLM 通常采用 Pre-Norm 结构，即在注意力和 FFN 子层之前做 RMSNorm，而非原始 Transformer 的 Post-Norm。Pre-Norm 使梯度路径更短，训练更稳定，配合 RMSNorm 是当前最优实践。

# 总结
RMSNorm 以极简的设计——去掉均值、只保留 RMS 缩放——在几乎不损失效果的前提下显著提升了训练和推理速度。它是 "less is more" 工程哲学的绝佳示例：抓住核心机制，抛弃冗余操作。如果你在搭建自己的 Transformer，RMSNorm 应该是你的默认选择。