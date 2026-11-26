"""03 · 核心机制：Multi-Head Attention 与三种注意力用法"""
import math
from lib import *

P = []
P.append(header("Multi-Head Attention：8 个低维头并行看不同关系，再拼回 512 维",
                "左：一个多头注意力层内部；右：同一个模块在 Transformer 里的三种接法（Q 与 K/V 来源、掩码不同）",
                3, "核心机制 ②"))

# ======================= LEFT: multi-head =======================
P.append(step_badge(76, 196, 1))
P.append(text(98, 202, "拆成 h 个头 → 各自注意力 → 拼接 → 线性融合", 19, weight=700))

# input
ix, iy = 70, 300
P.append(rect(ix, iy, 70, 250, PANEL2, MUTED, 1.4, 8))
P.append(ctext(ix + 35, iy + 110, "X", 24, fill=INK, family=MATH, italic=True, weight=700))
P.append(ctext(ix + 35, iy + 150, ["n×512"], 13, fill=MUTED))

# heads
patterns = {
    0: lambda r, c: 0.85 if c == max(r - 1, 0) else 0.08,                  # previous token
    1: lambda r, c: 0.8 if (r + c) % 5 == 0 or c == 4 else 0.1,           # long range / anchor
    2: lambda r, c: 0.85 if r == c else 0.1,                               # self
}
head_desc = ["例：关注前一个词", "例：关注远处关键词", "例：关注自身"]
hx = 220
hy0 = 238
hstep = 118
for k in range(3):
    y = hy0 + k * hstep
    P.append(poly_arrow([(ix + 70, iy + 125), (ix + 100, iy + 125), (ix + 100, y + 45), (hx - 4, y + 45)], stroke=MUTED, sw=1.3))
    P.append(rect(hx, y, 300, 92, "#FFFFFF", BLUE, 1.6, 10))
    P.append(text(hx + 14, y + 24, f"head_{{{k + 1 if k < 2 else 8}}}", 16, weight=700, fill=BLUE, family=FONT))
    P.append(text(hx + 14, y + 48, f"XW_{{i}}^{{Q}}, XW_{{i}}^{{K}}, XW_{{i}}^{{V}}", 13, fill=MUTED))
    P.append(text(hx + 14, y + 68, "512 → 64", 13, fill=MUTED))
    P.append(text(hx + 14, y + 86, head_desc[k], 12, fill=INK))
    P.append(grid(hx + 168, y + 10, 5, 5, 14, lambda r, c, k=k: mix("#FFFFFF", BLUE, patterns[k](r, c))))
    P.append(text(hx + 246, y + 40, ["Attn", "n×64"], 12, fill=MUTED))
    if k == 1:
        P.append(ctext(hx + 150, y + hstep - 13, "⋮  共 h = 8 个头，并行计算", 14, fill=MUTED))
# concat
cx_ = 570
for k in range(3):
    y = hy0 + k * hstep + 45
    P.append(arrow(hx + 302, y, cx_ - 4, y if k == 1 else (y + (hy0 + hstep + 45 - y) * 0.0), MUTED, 1.3))
P.append(rect(cx_, hy0 + 10, 64, 300, PANEL2, INK, 1.5, 8))
for i in range(8):
    P.append(rect(cx_ + 12, hy0 + 26 + i * 34, 40, 28, mix(BLUE_L, BLUE, 0.1 + 0.08 * i), "none", 0, 3))
P.append(ctext(cx_ + 32, hy0 + 335, "Concat", 14, weight=700))
P.append(ctext(cx_ + 32, hy0 + 354, "n×(8·64)", 12, fill=MUTED))
P.append(arrow(cx_ + 66, hy0 + 160, 668, hy0 + 160, INK, 1.8))
P.append(box(672, hy0 + 112, 92, 96, "W^{O}", 20, fill=ORANGE_L, stroke=ORANGE, color=ORANGE, sub="512×512", sub_size=12))
P.append(arrow(766, hy0 + 160, 792, hy0 + 160, INK, 1.8))
P.append(text(732, hy0 + 92, "输出 n×512", 13, fill=MUTED, anchor="middle"))

# formula + cost
P.append(rect(60, 620, 740, 82, PANEL, LINE, 1.2, 10))
P.append(segs(80, 652, [("MultiHead(Q,K,V) = Concat(head_{1}, …, head_{h}) ",), ("W^{O}", ORANGE, 700)], 17, family=MATH))
P.append(segs(80, 684, [("head_{i} = Attention(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V})     ",),
                        ("h·d_{k} = 8×64 = 512 → 总计算量≈单头全维", BLUE, 600)], 15))

# ablation mini chart (Table 3 A) — keep compute constant
ax0, ay0 = 80, 745
P.append(text(ax0 - 10, ay0 - 14, "头数消融（Table 3-A，计算量不变，EN-DE dev BLEU）", 14, weight=700))
data = [(1, 24.9), (4, 25.5), (8, 25.8), (16, 25.8), (32, 25.4)]
for i, (h, b) in enumerate(data):
    x = ax0 + i * 100
    bw = 70
    hh = (b - 24.5) * 42
    col = ORANGE if h == 8 else (BLUE_L if h != 1 else "#E5E7EB")
    P.append(rect(x, ay0 + 86 - hh, bw, hh, col, "none", 0, 3))
    P.append(ctext(x + bw / 2, ay0 + 78 - hh, f"{b}", 13, fill=INK, weight=600))
    P.append(ctext(x + bw / 2, ay0 + 102, f"h={h}", 12, fill=MUTED))
P.append(text(ax0 + 512, ay0 + 40, ["单头比最佳低 0.9", "头太多也会下降"], 13, fill=MUTED))

# divider
P.append(line(830, 180, 830, 850, LINE, 1.2, dash="4 5"))

# ======================= RIGHT: three uses =======================
rx = 860
P.append(step_badge(rx + 16, 196, 2, fill=ORANGE))
P.append(text(rx + 38, 202, "同一模块，三种接法", 19, weight=700, fill=ORANGE))

uses = [
    ("Encoder Self-Attention", "Q, K, V 都来自上一层 Encoder 输出", "每个源 token 看全部源 token（无掩码）",
     "full", BLUE, ["I", "love", "NLP", "."], ["I", "love", "NLP", "."]),
    ("Decoder Masked Self-Attention", "Q, K, V 都来自上一层 Decoder 输出", "位置 i 只能看 ≤ i：未来位置置 −∞",
     "causal", ORANGE, ["<s>", "我", "爱", "NLP"], ["<s>", "我", "爱", "NLP"]),
    ("Encoder–Decoder Cross-Attention", "Q 来自 Decoder；K, V 来自 Encoder Memory", "每个目标位置查询全部源 token",
     "cross", BLUE, ["<s>", "我", "爱", "NLP"], ["I", "love", "NLP", "."]),
]
cy = 232
for k, (name, src, rule, kind, col, rows, cols) in enumerate(uses):
    y = cy + k * 205
    P.append(rect(rx, y, 680, 190, "#FFFFFF", col, 1.6, 12))
    P.append(text(rx + 20, y + 34, name, 18, weight=700, fill=col))
    P.append(text(rx + 20, y + 66, src, 15, fill=INK))
    P.append(text(rx + 20, y + 92, rule, 15, fill=MUTED))
    # where used
    where = {"full": "位置：每个 Encoder 层的第 1 个子层",
             "causal": "位置：每个 Decoder 层的第 1 个子层",
             "cross": "位置：每个 Decoder 层的第 2 个子层"}[kind]
    P.append(text(rx + 20, y + 120, where, 13, fill=FAINT))
    if kind == "causal":
        P.append(text(rx + 20, y + 152, ["保证自回归：训练时可并行，", "又不会偷看答案"], 13, fill=ORANGE, lh=1.3))
    if kind == "cross":
        P.append(text(rx + 20, y + 152, ["行 = 目标 m 个位置，列 = 源 n 个位置", "注意力矩阵形状 m×n"], 13, fill=BLUE, lh=1.3))
    if kind == "full":
        P.append(text(rx + 20, y + 152, ["双向：同时利用左右上下文"], 13, fill=BLUE))

    # mask grid
    gx, gy, cs = rx + 500, y + 40, 28
    for j, t in enumerate(cols):
        P.append(ctext(gx + j * cs + cs / 2, gy - 10, t, 11, fill=MUTED))
    for i, t in enumerate(rows):
        P.append(text(gx - 6, gy + i * cs + cs / 2 + 4, t, 11, fill=MUTED, anchor="end"))

    def fill(r, c, kind=kind, col=col):
        if kind == "causal" and c > r:
            return "#E5E7EB"
        return mix("#FFFFFF", col, 0.25 + 0.5 * ((r * 7 + c * 3) % 5) / 5)
    vals = None
    if kind == "causal":
        vals = [["−∞" if c > r else "" for c in range(4)] for r in range(4)]
    P.append(grid(gx, gy, 4, 4, cs, fill, values=vals, vsize=10, vcolor_fn=lambda r, c: MUTED))
    P.append(text(gx + 2 * cs, gy + 4 * cs + 18, "列：K/V 来源", 11, fill=MUTED, anchor="middle"))

P.append(footer(3))
render("03-multi-head-attention", "".join(P))
