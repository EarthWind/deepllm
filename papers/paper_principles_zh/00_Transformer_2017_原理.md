# Transformer 原理导读

**论文标题**：`Attention Is All You Need`
**作者**：Ashish Vaswani 等，Google Brain / Google Research，2017年
**年份**：`2017`
**主题**：`注意力 / 编码器-解码器`
**定位**：用纯注意力替代 RNN/CNN 的序列建模范式，成为后续 LLM 的统一骨架。
**论文链接**：[arXiv](https://arxiv.org/abs/1706.03762)

Transformer 最重要的价值不是“一个更强的翻译模型”，而是它给出了一个可扩展、可并行、可堆叠的序列建模积木：`(Self-Attention + FFN) × N`。从 BERT 的编码器、GPT 的解码器，到今天的各种 LLM 变体，都可以看作对这套积木的删减、改造与工程化。

你可以用一句话抓住它的核心：

**把“序列依赖”从“沿时间步递推”改成“在同一层里全局建图并一次性聚合”。**

## 1. 代表公式 / 关键表达

- Scaled Dot-Product Attention：`Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`
- Multi-Head Attention：`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`，`MHA = Concat(head_1..head_h) W^O`
- 位置编码（正弦/余弦）：
  - `PE(pos,2i) = sin(pos / 10000^{2i/d_model})`
  - `PE(pos,2i+1) = cos(pos / 10000^{2i/d_model})`
- 每个子层外的残差与归一化：`x -> Sublayer(x) -> x + Dropout(...) -> LayerNorm(...)`（原论文为 post-norm 结构）

## 2. 这篇论文要解决什么问题

在 2017 年主流序列到序列（Seq2Seq）系统里，RNN/LSTM 仍是机器翻译的核心：

- **并行性差**：RNN 需要按时间步递推，难以充分利用 GPU。
- **长程依赖困难**：虽然有注意力缓解，但主干仍是递推结构，路径长度随序列长度增长。
- **工程复杂**：为了速度常引入 CNN 或复杂的注意力变体，系统难以简化。

作者的目标很明确：

- 设计一个不依赖递推和卷积的 Seq2Seq 架构；
- 在翻译任务上做到更强、更快、更易并行。

## 3. 核心思想

- 用 **自注意力（self-attention）** 作为“信息路由器”：每个 token 直接从全局收集相关上下文。
- 用 **多头（multi-head）** 把一个注意力拆成多个子空间的注意力，提升表达能力。
- 用 **位置编码（positional encoding）** 注入顺序信息，弥补注意力本身的置换不变性。
- 用 **编码器-解码器堆叠** 保留传统 Seq2Seq 的可解释结构：编码器负责理解源句，解码器负责自回归生成目标句。

## 4. 方法原理拆解

### 4.1 编码器（Encoder）

编码器是 `N` 层相同结构的堆叠，每层包含两个子层：

- `Multi-Head Self-Attention`
- `Position-wise Feed-Forward Network (FFN)`

直觉上：attention 负责“从全局找信息并混合”，FFN 负责“对每个位置做非线性变换（升维-激活-降维）”。

### 4.2 解码器（Decoder）

解码器同样是 `N` 层堆叠，但每层包含三个子层：

- `Masked Multi-Head Self-Attention`（因果 mask，防止看到未来 token）
- `Multi-Head Cross-Attention`（Q 来自解码器，K/V 来自编码器输出）
- `FFN`

这让解码器同时具备两种能力：

- 对已生成前缀做自回归建模（masked self-attention）
- 按需从源句表征中检索信息（cross-attention）

### 4.3 为什么要 “Scaled”

`QK^T` 的尺度会随着 `d_k` 增大而增大，softmax 容易进入饱和区导致梯度变小。
用 `1/sqrt(d_k)` 缩放可让数值更稳定，这是 Transformer 训练稳定性的关键小细节之一。

### 4.4 Mask 的两种常见形态

- `padding mask`：屏蔽掉 padding 位置，使其不会被关注。
- `causal (look-ahead) mask`：解码器自注意力中使用上三角 mask，保证自回归。

工程实现里通常把 mask 转成对 attention logits 的加法：把被屏蔽位置加上一个极小值（如 `-1e9`），再 softmax。

## 5. 代码实现结构

如果按工程实现拆分，Transformer 常被写成：

- `Embedding + PositionalEncoding`
- `Encoder`（`N × EncoderLayer`）
- `Decoder`（`N × DecoderLayer`）
- `Generator / LM Head`（线性层映射到词表）

### 5.1 一个最小实现骨架

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attn_mask=None):
        b, tq, _ = q.shape
        _, tk, _ = k.shape

        q = self.wq(q).view(b, tq, self.num_heads, self.d_head).transpose(1, 2)
        k = self.wk(k).view(b, tk, self.num_heads, self.d_head).transpose(1, 2)
        v = self.wv(v).view(b, tk, self.num_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = scores.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).contiguous().view(b, tq, self.d_model)
        return self.wo(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, src_mask))
        x = self.norm2(x + self.ffn(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, memory, memory, src_mask))
        x = self.norm3(x + self.ffn(x))
        return x
```

### 5.2 从源码角度最值得注意的点

- `mask` 的形状广播：常见形状是 `(batch, 1, 1, seq_len)` 或 `(batch, 1, tgt_len, src_len)`，和 `scores` 对齐。
- `reshape / transpose`：多头拆分与拼接最容易写错维度顺序。
- `softmax` 的数值稳定性：注意 logits 的 dtype、以及 mask 的极小值。
- 权重共享：翻译/LM 场景常把输入 embedding 与输出投影权重绑定。

## 6. 训练或推理范式

- 训练任务在原论文中是机器翻译（WMT14 En-De / En-Fr），解码器为自回归，训练使用 teacher forcing。
- 常见训练细节：`Adam`、`warmup` 学习率调度、`label smoothing`、`dropout`。
- 推理阶段通常用 `beam search`（机器翻译场景）。

## 7. 为什么它有效

- **更短的依赖路径**：任意两 token 之间的交互在同一层内就能完成，路径长度从 `O(n)` 变为 `O(1)`（忽略层数）。
- **更强的并行性**：训练不需要按时间步递推，吞吐显著提升。
- **可扩展的积木结构**：堆更深、更宽、更大 batch 更自然，为后续“规模化就是能力”的路线打基础。

## 8. 局限与争议

- **二次复杂度**：全注意力在长度上是 `O(n^2)`，长序列成本高，直接催生了稀疏注意力、线性注意力、FlashAttention 等方向。
- **位置建模并不完美**：绝对位置编码对长度外推有限，后续出现 RoPE、ALiBi 等改进。
- **训练稳定性依赖细节**：归一化位置（post-norm vs pre-norm）、初始化与学习率调度都会显著影响收敛。

## 9. 对后续研究的影响

- 定义了现代大模型的默认骨架：encoder-only（BERT）、decoder-only（GPT）、encoder-decoder（T5）。
- 推动“模型并行 + 数据并行 + 更大规模”的工程路径成为主流。
- 让“注意力机制”从辅助模块上升为序列建模主干。

## 10. 你应该怎样读这篇

- 优先把结构读成积木：`Attention / FFN / Residual+Norm / Mask / Position`，不要陷在翻译实验细节。
- 对照实现时重点盯 `mask`、`QKV` 形状、`softmax` 之前的 logits 处理。
- 读完后再看 RoPE、FlashAttention，会更容易理解它们到底在改哪里。

## 11. 模型规模

原论文中常被引用的两个配置：

- `Transformer base`：`d_model=512`，`layers=6`，`heads=8`，`d_ff=2048`
- `Transformer big`：`d_model=1024`，`layers=6`，`heads=16`，`d_ff=4096`

## 12. 历史地位与局限

Transformer 的历史地位在于它把“序列建模”变成了一套可大规模堆叠与优化的通用算子组合，为后续预训练语言模型、对齐、推理、Agent 等几乎所有 LLM 方向提供了统一底座。

它的局限同样清晰：长度成本高、位置表示需要改进、工程实现对细节敏感。后续十年的研究几乎都可以理解为：在不破坏这套积木可扩展性的前提下，解决这些局限。

## 13. 前置阅读

## 14. 读完接着看
- `RoFormer / RoPE`
- `FlashAttention`
- `T5`


## 15. 参考文档
- [PaperNotes/Transformer 论文精读.md](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/PaperNotes/Transformer%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md)