"""05 · 一层内部：残差 + LayerNorm + 逐位置 FFN"""
from lib import *

P = []
P.append(header("层内部构造：残差 + LayerNorm 包住两个子层，FFN 负责逐 token 加工",
                "以一个 Encoder 层为例（Decoder 多一个 Cross-Attention 子层，包法相同）；所有子层输入输出都是 n×512，才能直接相加",
                5, "核心机制 ④"))

P.append(step_badge(76, 196, 1))
P.append(text(98, 202, "数据路径：Sublayer → Dropout → 残差相加 → LayerNorm（Post-Norm）", 19, weight=700))

cy = 318
bh = 64
P.append(chip(60, cy - 24, 70, 48, "x", fill=PANEL2, size=20))
P.append(text(95, cy + 44, "n×512", 12, fill=MUTED, anchor="middle"))


def b(x, w, label, col=INK, fill="#FFFFFF", sub=None, h=bh):
    return box(x, cy - h / 2, w, h, label, 16, fill=fill, stroke=col, color=col, sub=sub, sub_size=12)


P.append(arrow(132, cy, 176, cy))
P.append(b(180, 170, ["Multi-Head", "Self-Attention"], BLUE))
P.append(arrow(352, cy, 386, cy))
P.append(b(390, 90, "Dropout", MUTED, PANEL2, h=44))
P.append(arrow(482, cy, 515, cy))
P.append(plus_node(530, cy, 15))
P.append(arrow(546, cy, 576, cy))
P.append(b(580, 120, "LayerNorm", INK, PANEL2))
P.append(arrow(702, cy, 784, cy))
P.append(text(722, cy - 10, "x′", 16, fill=INK, anchor="middle", family=MATH, italic=True))
P.append(b(788, 160, ["Feed-Forward", "(FFN)"], ORANGE, ORANGE_LL))
P.append(arrow(950, cy, 984, cy))
P.append(b(988, 90, "Dropout", MUTED, PANEL2, h=44))
P.append(arrow(1080, cy, 1113, cy))
P.append(plus_node(1128, cy, 15))
P.append(arrow(1144, cy, 1174, cy))
P.append(b(1178, 120, "LayerNorm", INK, PANEL2))
P.append(arrow(1300, cy, 1340, cy))
P.append(chip(1344, cy - 24, 110, 48, "输出 n×512", fill=PANEL2, size=15))
P.append(text(1470, cy + 6, "→ 下一层", 15, fill=MUTED))

# residual skips
P.append(poly_arrow([(150, cy), (150, 248), (530, 248), (530, cy - 17)], stroke=BLUE, sw=2.2, r=12))
P.append(poly_arrow([(750, cy), (750, 248), (1128, 248), (1128, cy - 17)], stroke=BLUE, sw=2.2, r=12))
P.append(text(340, 240, "残差捷径：x 原样跳过子层", 14, fill=BLUE, weight=600, anchor="middle"))
P.append(text(940, 240, "残差捷径：x′ 原样跳过子层", 14, fill=BLUE, weight=600, anchor="middle"))
# sublayer brackets
for x1, x2, lab in [(170, 712, "子层 1：注意力（token 之间交换信息）"), (776, 1310, "子层 2：FFN（每个 token 内部变换）")]:
    P.append(path(f"M{x1},{cy + 46} L{x1},{cy + 56} L{x2},{cy + 56} L{x2},{cy + 46}", MUTED, 1.2))
    P.append(text((x1 + x2) / 2, cy + 78, lab, 14, fill=MUTED, anchor="middle"))

# formula band
fy = 414
P.append(rect(60, fy, W - 120, 56, BLUE_LL, BLUE, 1.2, 10))
P.append(segs(84, fy + 36, [("每个子层：  output = ",), ("LayerNorm( x + Dropout( Sublayer(x) ) )", BLUE, 700)], 20))
P.append(text(W - 84, fy + 26, "原论文为 Post-Norm：先相加，后归一化", 14, fill=INK, anchor="end", weight=600))
P.append(text(W - 84, fy + 46, "P_{drop} = 0.1；embedding + PE 之和也做 dropout", 13, fill=MUTED, anchor="end"))

# ======================= lower panels =======================
ly = 500
# --- Panel A: division of labour
P.append(step_badge(76, ly + 6, 2))
P.append(text(98, ly + 12, "分工：横向混合 vs 纵向加工", 18, weight=700))
ax_ = 90
toks = ["我", "爱", "吃", "苹果", "。"]
ty = ly + 92
for i, t in enumerate(toks):
    x = ax_ + i * 82
    P.append(circle(x + 20, ty, 20, "#FFFFFF", BLUE, 1.6))
    P.append(ctext(x + 20, ty, t, 13, fill=INK))
for (i, j, h) in [(0, 3, 46), (1, 3, 30), (2, 3, 18), (3, 4, 18), (0, 2, 32)]:
    x1, x2 = ax_ + i * 82 + 20, ax_ + j * 82 + 20
    P.append(path(f"M{x1},{ty - 22} Q{(x1 + x2) / 2},{ty - 22 - h} {x2},{ty - 22}", BLUE, 1.4, opacity=0.8))
P.append(text(ax_ - 10, ty + 44, "Attention：token ↔ token，按相关性路由信息", 14, fill=BLUE, weight=600))
fy2 = ty + 76
for i, t in enumerate(toks):
    x = ax_ + i * 82
    P.append(circle(x + 20, fy2, 16, "#FFFFFF", MUTED, 1.2))
    P.append(ctext(x + 20, fy2, t, 11, fill=INK))
    P.append(arrow(x + 20, fy2 + 17, x + 20, fy2 + 37, ORANGE, 1.6))
    P.append(rect(x, fy2 + 40, 40, 30, ORANGE_L, ORANGE, 1.3, 6))
    P.append(ctext(x + 20, fy2 + 55, "FFN", 11, fill=ORANGE, weight=700))
P.append(text(ax_ - 10, fy2 + 96, ["FFN：同一个 MLP 独立作用于每个位置", "同层各位置共享参数；不同层参数不同"], 14, fill=ORANGE, weight=600))

P.append(line(540, ly - 10, 540, 850, LINE, 1.2, dash="4 5"))

# --- Panel B: FFN zoom
bx = 570
P.append(step_badge(bx + 6, ly + 6, 3, fill=ORANGE))
P.append(text(bx + 28, ly + 12, "FFN 放大：先升维 ×4，再降回", 18, weight=700, fill=ORANGE))
vy = ly + 44
c = 15
# input 512
for i in range(8):
    P.append(rect(bx + 20, vy + i * c, 20, c, mix("#FFFFFF", "#64748B", 0.2 + 0.08 * ((i * 3) % 7)), "#FFFFFF", 1, 0))
P.append(ctext(bx + 30, vy + 8 * c + 16, "512", 13, fill=MUTED))
P.append(arrow(bx + 46, vy + 60, bx + 120, vy + 60, INK, 1.6))
P.append(ctext(bx + 83, vy + 46, "W_{1}, b_{1}", 13, fill=INK, family=MATH))
P.append(ctext(bx + 83, vy + 80, "512×2048", 11, fill=MUTED))
# hidden 2048 with ReLU zeros
zero = {1, 4, 5, 9, 12, 13, 17, 20, 22}
hcells = 24
hc = 8
for i in range(hcells):
    f = "#E5E7EB" if i in zero else mix(ORANGE_L, ORANGE, 0.2 + 0.6 * ((i * 5) % 7) / 7)
    P.append(rect(bx + 128, vy - 0 + i * hc - 0, 30, hc, f, "#FFFFFF", 1, 0))
P.append(ctext(bx + 143, vy + hcells * hc + 16, "2048", 13, fill=MUTED))
P.append(text(bx + 166, vy + 70, ["ReLU", "灰 = 被截为 0"], 12, fill=MUTED))
P.append(arrow(bx + 250, vy + 60, bx + 324, vy + 60, INK, 1.6))
P.append(ctext(bx + 287, vy + 46, "W_{2}, b_{2}", 13, fill=INK, family=MATH))
P.append(ctext(bx + 287, vy + 80, "2048×512", 11, fill=MUTED))
for i in range(8):
    P.append(rect(bx + 332, vy + i * c, 20, c, mix(ORANGE_L, ORANGE, 0.25 + 0.08 * ((i * 5) % 7)), "#FFFFFF", 1, 0))
P.append(ctext(bx + 342, vy + 8 * c + 16, "512", 13, fill=MUTED))
P.append(text(bx + 370, vy + 50, ["对每个", "token", "各算一次"], 12, fill=MUTED))
P.append(rect(bx, vy + 226, 470, 64, ORANGE_LL, ORANGE, 1.2, 8))
P.append(text(bx + 16, vy + 252, "FFN(x) = max(0, xW_{1} + b_{1}) W_{2} + b_{2}", 17, family=MATH))
P.append(text(bx + 16, vy + 278, "等价视角：两个 kernel size = 1 的卷积；d_{ff} = 2048 = 4·d_{model}", 13, fill=MUTED))

P.append(line(1080, ly - 10, 1080, 850, LINE, 1.2, dash="4 5"))

# --- Panel C: params + LayerNorm
cx_ = 1110
P.append(step_badge(cx_ + 6, ly + 6, 4))
P.append(text(cx_ + 28, ly + 12, "参数在哪 & LayerNorm 在做什么", 18, weight=700))
py = ly + 46
P.append(text(cx_, py + 4, "每层参数（base，忽略偏置）", 14, weight=600))
for k, (lab, v, col, frm) in enumerate([("Attention", 1.05, BLUE, "4·d^{2}"), ("FFN", 2.10, ORANGE, "2·d·d_{ff}")]):
    yy = py + 20 + k * 34
    P.append(text(cx_ + 80, yy + 18, lab, 13, anchor="end", fill=INK))
    P.append(rect(cx_ + 88, yy + 4, v * 90, 20, col, "none", 0, 3))
    P.append(text(cx_ + 96 + v * 90, yy + 19, f"{frm} ≈ {v:.2f}M", 13, fill=INK))
P.append(text(cx_, py + 104, "→ FFN 约占每层 2/3 参数", 14, fill=ORANGE, weight=700))

ny = py + 132
P.append(text(cx_, ny + 6, "LayerNorm：对单个 token 的 512 维归一化", 14, weight=600))
vals = [0.9, 0.2, 0.75, 0.1, 0.55, 0.95, 0.3, 0.65]
for i, v in enumerate(vals):
    P.append(rect(cx_ + i * 20, ny + 22, 18, 18, mix("#FFFFFF", "#475569", v), "none", 0, 2))
P.append(arrow(cx_ + 168, ny + 31, cx_ + 214, ny + 31, INK, 1.4))
P.append(ctext(cx_ + 191, ny + 18, "−μ, ÷σ", 11, fill=MUTED))
for i, v in enumerate(vals):
    P.append(rect(cx_ + 222 + i * 20, ny + 22, 18, 18, mix(BLUE_LL, BLUE, 0.25 + 0.5 * v), "none", 0, 2))
P.append(text(cx_, ny + 66, "均值 0、方差 1，再 ×γ + β（可学习）；与 batch 无关", 12, fill=MUTED))

P.append(rect(cx_, ny + 84, 430, 76, PANEL, LINE, 1.2, 8))
P.append(text(cx_ + 14, ny + 106, "后续演进（非本文）", 13, weight=700, fill=MUTED))
P.append(text(cx_ + 14, ny + 128, "GPT-2 等改为 Pre-Norm：x + Sublayer(LN(x))", 13, fill=INK))
P.append(text(cx_ + 14, ny + 148, "残差主干不被归一化打断，深层训练更稳", 13, fill=MUTED))

P.append(footer(5))
render("05-residual-layernorm-ffn", "".join(P))
