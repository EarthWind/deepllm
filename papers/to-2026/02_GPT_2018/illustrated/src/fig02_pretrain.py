"""02 · 无监督预训练：因果语言模型的前向计算、因果掩码与模型配置"""
from lib import *

P = []
P.append(header("无监督预训练：12 层 Transformer 解码器做“下一个 token 预测”",
                "Eq. 2：h_{0} = UW_{e} + W_{p}；h_{l} = transformer_block(h_{l-1})；P(u) = softmax(h_{n}W_{e}^{T})（Section 3.1、4.1）",
                2, "核心机制 ①"))

# ======================= ① forward pass =======================
x0, y0, w0, h0 = 60, 178, 520, 660
P.append(rect(x0, y0, w0, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x0 + 24, y0 + 28, 1, fill=BLUE))
P.append(text(x0 + 46, y0 + 34, "前向计算（自下而上）", 17, weight=700))
cx = x0 + w0 / 2
bw = 360
bx = cx - bw / 2
# output
P.append(box(bx, y0 + 64, bw, 58, "P(u) = softmax(h_{n} W_{e}^{T})", 17, fill=ORANGE_L, stroke=ORANGE, color=ORANGE,
             sub="输出层复用输入嵌入矩阵 W_{e}（权重共享）", sub_size=13, sub_color=INK))
P.append(arrow(cx, y0 + 160, cx, y0 + 124, INK, 1.6))
P.append(text(cx + 10, y0 + 148, "h_{n}", 14, fill=MUTED))
# stack
P.append(rect(bx - 14, y0 + 162, bw + 28, 296, PANEL, LINE, 1.2, 12, dash="6 5"))
P.append(text(bx + bw + 22, y0 + 182, "×12", 16, weight=700, fill=MUTED, anchor="end"))
P.append(text(bx, y0 + 186, "transformer_block", 14, weight=700, fill=MUTED, family=MONO))
layers = [("Add & LayerNorm", PANEL2, LINE, INK), ("Feed Forward  768→3072→768", ORANGE_L, ORANGE, INK),
          ("Add & LayerNorm", PANEL2, LINE, INK), ("Masked Multi-Head Self-Attention", BLUE_L, BLUE, BLUE)]
for i, (lab, f, s, c) in enumerate(layers):
    y = y0 + 200 + i * 56
    P.append(box(bx + 10, y, bw - 20, 42, lab, 15, fill=f, stroke=s, color=c, weight=600))
    if i < 3:
        P.append(arrow(cx, y + 56, cx, y + 44, MUTED, 1.3))
P.append(text(bx + 10, y0 + 440, "12 个头 · d_{model} = 768 · 每个子层外有残差连接", 13, fill=MUTED))
P.append(arrow(cx, y0 + 492, cx, y0 + 460, INK, 1.6))
P.append(text(cx + 10, y0 + 482, "h_{0}", 14, fill=MUTED))
# embeddings
P.append(box(bx, y0 + 494, bw, 58, "h_{0} = U W_{e} + W_{p}", 17, fill=BLUE_L, stroke=BLUE, color=BLUE,
             sub="W_{e}：token 嵌入　W_{p}：可学习位置嵌入（非正弦）", sub_size=13, sub_color=INK))
P.append(arrow(cx, y0 + 584, cx, y0 + 554, INK, 1.6))
toks = ["u_{-k}", "…", "u_{-2}", "u_{-1}"]
for i, t in enumerate(toks):
    P.append(chip(bx + 20 + i * 84, y0 + 586, 72, 32, t, size=14))
P.append(text(cx, y0 + 640, "U：上下文 token 的 one-hot 矩阵（BPE 词表）", 13, fill=MUTED, anchor="middle"))

# ======================= ② causal mask =======================
x1, w1 = 600, 470
P.append(rect(x1, y0, w1, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x1 + 24, y0 + 28, 2, fill=BLUE))
P.append(text(x1 + 46, y0 + 34, "因果掩码：每个位置只看自己和左边", 17, weight=700))
T = ["The", "cat", "sat", "on", "the", "mat"]
n = len(T)
cell = 46
gx, gy = x1 + 110, y0 + 110
for j, t in enumerate(T):
    P.append(ctext(gx + j * cell + cell / 2, gy - 18, t, 13, fill=MUTED, weight=600))
P.append(text(gx + n * cell / 2, gy - 44, "Key（被看的位置）", 13, fill=MUTED, anchor="middle"))
for i, t in enumerate(T):
    P.append(text(gx - 12, gy + i * cell + cell / 2 + 5, t, 13, fill=MUTED, weight=600, anchor="end"))
P.append(text(gx - 76, gy + n * cell / 2, "Query", 13, fill=MUTED, anchor="middle", rotate=-90))


def fill(r, c):
    return BLUE if c == r else (BLUE_L if c < r else PANEL2)


P.append(grid(gx, gy, n, n, cell, fill_fn=fill, stroke="#FFFFFF", sw=2,
              values=[["" if c <= r else "−∞" for c in range(n)] for r in range(n)], vsize=13,
              vcolor_fn=lambda r, c: FAINT))
# targets
tx = gx + n * cell + 30
P.append(text(tx + 30, gy - 18, "目标", 13, fill=ORANGE, weight=700, anchor="middle"))
tgt = ["cat", "sat", "on", "the", "mat", "…"]
for i, t in enumerate(tgt):
    P.append(arrow(gx + n * cell + 4, gy + i * cell + cell / 2, tx - 2, gy + i * cell + cell / 2, ORANGE, 1.4))
    P.append(chip(tx, gy + i * cell + 10, 60, cell - 20, t, fill=ORANGE_L, stroke=ORANGE, color=ORANGE, size=13))
ly = gy + n * cell + 34
P.append(rect(x1 + 24, ly, 18, 18, BLUE_L, BLUE_L, 1, 3))
P.append(text(x1 + 50, ly + 14, "可见（深色为自身）", 13))
P.append(rect(x1 + 200, ly, 18, 18, PANEL2, LINE, 1, 3))
P.append(text(x1 + 226, ly + 14, "被遮住（softmax 前加 −∞）", 13))
P.append(rect(x1 + 24, ly + 40, w1 - 48, 152, BLUE_LL, BLUE, 1.2, 10))
P.append(text(x1 + 44, ly + 72, ["一条长 512 的序列，**一次前向得到 512 个预测**：",
                                 "第 i 行只依赖 u_{1..i}，预测 u_{i+1}。",
                                 "监督信号来自文本自身，不需要人工标注；",
                                 "推理时把预测出的 token 接回输入即可续写。"], 14, lh=1.65))

# ======================= ③ recipe =======================
x2, w2 = 1090, 450
P.append(rect(x2, y0, w2, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x2 + 24, y0 + 28, 3, fill=ORANGE))
P.append(text(x2 + 46, y0 + 34, "预训练配置（Section 4.1）", 17, weight=700))
rows = [("结构", "12 层 decoder-only，768 维，12 头"),
        ("FFN", "内层 3072 维，激活 GELU"),
        ("位置", "可学习位置嵌入"),
        ("词表", "BPE，40,000 次合并"),
        ("批次", "64 条 × 512 个连续 token"),
        ("轮数", "100 epochs"),
        ("优化", "Adam，峰值 lr 2.5e-4"),
        ("调度", "前 2,000 步线性升温，再余弦降到 0"),
        ("初始化", "N(0, 0.02)（得益于 LayerNorm）"),
        ("正则", "残差 / 嵌入 / 注意力 dropout 0.1"),
        ("", "改进版 L2，w = 0.01"),
        ("预处理", "ftfy 清洗 + spaCy 分词")]
ry = y0 + 66
for i, (k, v) in enumerate(rows):
    y = ry + i * 34
    if i % 2 == 0:
        P.append(rect(x2 + 16, y - 4, w2 - 32, 34, PANEL, PANEL, 1, 6))
    P.append(text(x2 + 32, y + 18, k, 14, weight=700, fill=BLUE))
    P.append(text(x2 + 110, y + 18, v, 14))
by = ry + len(rows) * 34 + 14
P.append(rect(x2 + 16, by, w2 - 32, 162, ORANGE_LL, ORANGE, 1.2, 10))
P.append(text(x2 + 34, by + 30, "为什么选 BooksCorpus", 15, weight=700, fill=ORANGE))
P.append(text(x2 + 34, by + 60, ["7,000+ 本书，有大段连续文本，", "模型能学会利用长距离上下文。",
                                  "同规模的 1B Word Benchmark 按句打乱，",
                                  "长距离结构被破坏。token 级困惑度 18.4。"], 14, lh=1.6))

P.append(footer(2))
render("02-unsupervised-pretraining", "".join(P))
