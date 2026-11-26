"""02 · 核心机制：Scaled Dot-Product Attention 逐步计算"""
import math
from lib import *

P = []
P.append(header("Scaled Dot-Product Attention：一次注意力到底算了什么",
                "以 4 个 token 为例：打分 → 缩放 → 归一化 → 加权求和；每一步都是矩阵乘或逐元素运算，可整段并行",
                2, "核心机制 ①"))

TOK = ["我", "爱", "吃", "苹果"]
# raw scores S = QK^T (d_k = 64, sqrt = 8)
S = [[24, 8, 4, 12],
     [10, 20, 14, 18],
     [6, 12, 22, 30],
     [4, 10, 26, 34]]
SC = [[v / 8 for v in r] for r in S]
A = []
for r in SC:
    m = max(r)
    e = [math.exp(v - m) for v in r]
    s = sum(e)
    A.append([v / s for v in e])

# ---------- Stage 1: X -> Q, K, V ----------
y0 = 190
P.append(step_badge(76, y0 + 6, 1))
P.append(text(98, y0 + 12, "线性投影：同一个 X 投出三种角色", 19, weight=700))

gx, gy = 70, 250
cell = 15
P.append(text(gx, gy - 12, "X", 20, weight=700, family=MATH, italic=True))
P.append(text(gx + 22, gy - 12, "n×d_{model}", 14, fill=MUTED))
for i, t in enumerate(TOK):
    P.append(text(gx - 8, gy + i * 34 + 22, t, 15, anchor="end", fill=INK))
P.append(grid(gx, gy, 4, 7, cell, lambda r, c: mix("#FFFFFF", "#94A3B8", 0.25 + 0.5 * ((r * 3 + c * 5) % 7) / 7),
              cell_h=34))

roles = [("Q", "W^{Q}", "查询：我要找什么", BLUE, BLUE_L, 232),
         ("K", "W^{K}", "键：我有什么特征", BLUE, BLUE_L, 372),
         ("V", "W^{V}", "值：我能提供的内容", ORANGE, ORANGE_L, 512)]
for name, wname, desc, col, colL, yy in roles:
    P.append(poly_arrow([(gx + 7 * cell + 6, gy + 68), (205, gy + 68), (205, yy + 50), (280, yy + 50)], stroke=MUTED, sw=1.5))
    P.append(text(214, yy + 42, f"× {wname}", 14, fill=MUTED, family=MATH))
    P.append(grid(290, yy + 10, 4, 4, 20, lambda r, c, colL=colL, col=col: mix(colL, col, 0.12 + 0.55 * ((r * 2 + c * 3) % 5) / 5)))
    P.append(text(290, yy + 2, name, 20, weight=700, fill=col, family=MATH, italic=True))
    P.append(text(312, yy + 2, "n×d_{k}" if name != "V" else "n×d_{v}", 13, fill=MUTED))
    P.append(text(290, yy + 112, desc, 14, fill=MUTED))
P.append(text(70, 440, ["d_{model}=512", "d_{k}=d_{v}=64"], 14, fill=MUTED))
P.append(text(70, 500, ["三个矩阵", "各自可学习"], 14, fill=MUTED))

# divider
P.append(line(400, 180, 400, 760, LINE, 1.2, dash="4 5"))

# ---------- Stage 2: QK^T ----------
sx = 440
P.append(step_badge(sx + 6, y0 + 6, 2))
P.append(text(sx + 28, y0 + 12, "打分：S = QK^{T}", 19, weight=700))
cs = 46
gx2, gy2 = sx + 54, 270
P.append(text(gx2 + 2 * cs, gy2 - 30, "key →", 14, fill=MUTED, anchor="middle"))
for i, t in enumerate(TOK):
    P.append(ctext(gx2 + i * cs + cs / 2, gy2 - 12, t, 14, fill=INK))
    P.append(text(gx2 - 8, gy2 + i * cs + cs / 2 + 5, t, 14, anchor="end"))
P.append(text(gx2 - 40, gy2 + 2 * cs, "query ↓", 14, fill=MUTED, anchor="middle", rotate=-90))
P.append(grid(gx2, gy2, 4, 4, cs, lambda r, c: mix("#FFFFFF", BLUE, 0.08 + 0.5 * S[r][c] / 34),
              values=[[str(v) for v in r] for r in S], vsize=15))
P.append(text(sx, 495, ["S_{ij} = q_{i} · k_{j}：第 i 个 query", "与第 j 个 key 的匹配度", "形状 n×n，一次矩阵乘算完"], 15, fill=INK))

# ---------- Stage 3: scale ----------
tx = 720
P.append(arrow(gx2 + 4 * cs + 10, gy2 + 2 * cs, tx + 40, gy2 + 2 * cs, INK, 2))
P.append(ctext((gx2 + 4 * cs + 10 + tx + 40) / 2, gy2 + 2 * cs - 16, "÷ √d_{k}", 17, fill=ORANGE, weight=700, family=MATH))
P.append(ctext((gx2 + 4 * cs + 10 + tx + 40) / 2, gy2 + 2 * cs + 20, "= ÷ 8", 14, fill=MUTED))
gx3 = tx + 50
P.append(step_badge(tx + 30, y0 + 6, 3, fill=ORANGE))
P.append(text(tx + 52, y0 + 12, "缩放", 19, weight=700, fill=ORANGE))
P.append(grid(gx3, gy2, 4, 4, cs, lambda r, c: mix("#FFFFFF", BLUE, 0.08 + 0.5 * S[r][c] / 34),
              values=[[f"{v:.2g}" if v != int(v) else str(int(v)) for v in r] for r in SC], vsize=15))

# why-scale callout
cx, cy, cw, ch = 690, 495, 330, 230
P.append(rect(cx, cy, cw, ch, ORANGE_LL, ORANGE, 1.4, 10))
P.append(text(cx + 16, cy + 30, "为什么要除以 √d_{k}？", 17, weight=700, fill=ORANGE))
P.append(text(cx + 16, cy + 62, ["设 q、k 各分量独立，均值 0、方差 1", "⇒ q·k 的方差 = d_{k}（=64，标准差 8）",
                                  "⇒ 分数过大，softmax 接近 one-hot", "⇒ 梯度极小，训练困难",
                                  "**缩放后方差回到 1，softmax 不饱和**"], 15, lh=1.5))
# tiny softmax saturation sketch
bx, by = cx + 220, cy + 175
P.append(text(cx + 16, cy + 214, "未缩放 vs 缩放后的一行权重：", 13, fill=MUTED))
un = [0.0, 0.0, 0.0, 1.0]
sc_row = A[3]
for i in range(4):
    P.append(rect(bx + i * 12, by + 40 - 36 * un[i], 9, 36 * un[i] + 0.5, FAINT, FAINT, 0.5, 1))
    P.append(rect(bx + 60 + i * 12, by + 40 - 36 * sc_row[i], 9, 36 * sc_row[i] + 0.5, ORANGE, ORANGE, 0.5, 1))
P.append(line(bx - 4, by + 40, bx + 108, by + 40, MUTED, 1))

# divider
P.append(line(1050, 180, 1050, 760, LINE, 1.2, dash="4 5"))

# ---------- Stage 4: (mask) + softmax ----------
ax = 1080
P.append(step_badge(ax + 6, y0 + 6, 4))
P.append(text(ax + 28, y0 + 12, "(+M) → softmax：逐行归一化", 19, weight=700))
gx4, gy4 = ax + 40, 250
for i, t in enumerate(TOK):
    P.append(ctext(gx4 + i * 40 + 20, gy4 - 12, t, 13))
    P.append(text(gx4 - 8, gy4 + i * 40 + 25, t, 13, anchor="end"))
P.append(grid(gx4, gy4, 4, 4, 40, lambda r, c: mix("#FFFFFF", BLUE, 0.05 + 0.9 * A[r][c]),
              values=[[f"{v:.2f}" for v in r] for r in A], vsize=12,
              vcolor_fn=lambda r, c: "#FFFFFF" if A[r][c] > 0.5 else INK))
P.append(text(gx4 + 170, gy4 + 70, ["A：注意力权重", "每行之和 = 1", "颜色越深 = 越关注"], 14, fill=MUTED))
P.append(rect(ax, 430, 150, 54, "#FFFFFF", MUTED, 1.2, 8, dash="5 4"))
P.append(ctext(ax + 75, 457, ["+M（可选掩码）", "非法位置设为 −∞"], 12, fill=MUTED))
P.append(text(ax + 165, 452, ["Decoder 中屏蔽", "“未来” token"], 13, fill=MUTED))

# ---------- Stage 5: A V ----------
P.append(step_badge(ax + 6, 520, 5, fill=ORANGE))
P.append(text(ax + 28, 526, "加权求和：Z = AV", 19, weight=700, fill=ORANGE))
# row of A for 苹果 × V rows
vy = 560
P.append(text(ax, vy + 18, "“苹果”的输出", 14, fill=INK, weight=600))
P.append(text(ax, vy + 40, "z_{苹果} = Σ_{j} A_{苹果,j} · v_{j}", 15, family=FONT))
for j, t in enumerate(TOK):
    yy = vy + 62 + j * 30
    P.append(text(ax + 10, yy + 18, f"{A[3][j]:.2f} ×", 13, fill=INK))
    P.append(grid(ax + 60, yy + 4, 1, 4, 20, lambda r, c, j=j: mix(ORANGE_L, ORANGE, 0.15 + 0.15 * ((j + c) % 4)), cell_h=20))
    P.append(text(ax + 148, yy + 19, f"v_{{{t}}}", 13, fill=MUTED))
P.append(path(f"M{ax + 215},{vy + 66} Q{ax + 240},{vy + 66} {ax + 240},{vy + 120} L{ax + 240},{vy + 120}", MUTED, 1.4))
P.append(path(f"M{ax + 215},{vy + 180} Q{ax + 240},{vy + 180} {ax + 240},{vy + 126}", MUTED, 1.4))
P.append(arrow(ax + 242, vy + 123, ax + 290, vy + 123, ORANGE, 2))
P.append(grid(ax + 300, vy + 113, 1, 4, 22, lambda r, c: mix(ORANGE_L, ORANGE, 0.55 + 0.1 * c), cell_h=22))
P.append(text(ax + 300, vy + 160, ["主要混入“吃”", "和“苹果”自己"], 13, fill=MUTED))

# ---------- bottom formula band ----------
fy = 780
P.append(rect(60, fy - 2, W - 120, 62, PANEL, LINE, 1.2, 10))
P.append(segs(90, fy + 38, [("Attention(Q, K, V) = softmax( ",), ("QK^{T} / √d_{k}", BLUE, 700), (" + M ) V",)], 24, family=MATH))
P.append(text(770, fy + 26, "形状：(n×d_{k})(d_{k}×n) → n×n 权重；  (n×n)(n×d_{v}) → n×d_{v} 输出", 15, fill=MUTED))
P.append(text(770, fy + 48, "代价：n×n 矩阵带来 O(n^{2}·d) 计算，换来任意两位置一步直连", 15, fill=MUTED))

P.append(footer(2))
render("02-scaled-dot-product-attention", "".join(P))
