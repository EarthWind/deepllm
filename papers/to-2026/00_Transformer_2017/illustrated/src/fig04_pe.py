"""04 · 位置编码：sin/cos 多频率信号"""
import math
from lib import *

P = []
P.append(header("Positional Encoding：给“无序”的注意力注入位置信息",
                "注意力本身对 token 顺序不敏感；在输入端把一组不同频率的 sin/cos 信号加到 embedding 上，模型就能区分“谁在前、谁在后”",
                4, "核心机制 ③"))

# ======================= ① why =======================
P.append(step_badge(76, 196, 1))
P.append(text(98, 202, "为什么需要", 19, weight=700))
bx, by = 60, 228
P.append(rect(bx, by, 360, 300, PANEL, LINE, 1.2, 10))
P.append(text(bx + 18, by + 32, "没有位置信息时：", 15, weight=700))
for i, t in enumerate(["猫", "追", "狗"]):
    P.append(chip(bx + 18 + i * 52, by + 48, 44, 32, t, size=15))
for i, t in enumerate(["狗", "追", "猫"]):
    P.append(chip(bx + 198 + i * 52, by + 48, 44, 32, t, size=15))
P.append(ctext(bx + 180, by + 64, "vs", 14, fill=MUTED))
P.append(text(bx + 18, by + 112, ["自注意力只看“集合”里谁和谁相似，", "打乱输入顺序 → 输出只是跟着重排", "（置换等变），两句话无从区分"],
              14, fill=INK, lh=1.5))
P.append(line(bx + 18, by + 180, bx + 342, by + 180, LINE, 1))
P.append(text(bx + 18, by + 208, "加上位置编码后：", 15, weight=700, fill=BLUE))
P.append(segs(bx + 18, by + 240, [("x_{pos} = ",), ("Embedding(token)·√d_{model}", INK, 600)], 15))
P.append(segs(bx + 40, by + 266, [("+ ",), ("PE(pos)", BLUE, 700), ("   (相加，不是拼接)", MUTED, 400)], 15))
P.append(text(bx + 18, by + 290, "同一个词在不同位置 → 不同的输入向量", 13, fill=MUTED))

# formula box
fx, fy = 60, 548
P.append(rect(fx, fy, 360, 132, BLUE_LL, BLUE, 1.4, 10))
P.append(text(fx + 18, fy + 30, "论文公式（固定，不需训练）", 15, weight=700, fill=BLUE))
P.append(text(fx + 18, fy + 64, "PE_{(pos, 2i)} = sin(pos / 10000^{2i/d})", 16, family=MATH))
P.append(text(fx + 18, fy + 96, "PE_{(pos, 2i+1)} = cos(pos / 10000^{2i/d})", 16, family=MATH))
P.append(text(fx + 18, fy + 122, "d = d_{model} = 512；i 为维度对序号", 13, fill=MUTED))

# ======================= ② heatmap =======================
hx0 = 470
P.append(step_badge(hx0 + 6, 196, 2, fill=ORANGE))
P.append(text(hx0 + 28, 202, "PE 矩阵长什么样（pos 0–49 × sin 分量 i=0–63）", 19, weight=700))
d = 512
NP, ND = 50, 64
cw, ch = 8.4, 8.4
gx, gy = hx0 + 40, 240


def pe(pos, j):
    i = j // 2
    ang = pos / (10000 ** (2 * i / d))
    return math.sin(ang) if j % 2 == 0 else math.cos(ang)


cells = []
for p in range(NP):
    for j in range(ND):
        v = pe(p, 2 * j)
        col = mix("#FFFFFF", ORANGE, v) if v >= 0 else mix("#FFFFFF", BLUE, -v)
        cells.append(f'<rect x="{gx + j * cw:.1f}" y="{gy + p * ch:.1f}" width="{cw + 0.3:.1f}" height="{ch + 0.3:.1f}" fill="{col}"/>')
P.append("".join(cells))
P.append(rect(gx, gy, ND * cw, NP * ch, "none", MUTED, 1, 0))
# axes
P.append(text(gx - 10, gy + 8, "0", 12, fill=MUTED, anchor="end"))
P.append(text(gx - 10, gy + NP * ch, "49", 12, fill=MUTED, anchor="end"))
P.append(text(gx - 26, gy + NP * ch / 2, "位置 pos ↓", 13, fill=MUTED, anchor="middle", rotate=-90))
P.append(text(gx, gy + NP * ch + 20, "i = 0（维度 0）", 12, fill=MUTED))
P.append(text(gx + ND * cw, gy + NP * ch + 20, "i = 63（维度 126）→", 12, fill=MUTED, anchor="end"))
# annotations
P.append(path(f"M{gx + 20},{gy + NP * ch + 32} L{gx + 20},{gy + NP * ch + 46}", INK, 1.2))
P.append(text(gx, gy + NP * ch + 66, ["低维：高频", "相邻位置差别大"], 13, fill=INK, weight=600))
P.append(path(f"M{gx + ND * cw - 30},{gy + NP * ch + 32} L{gx + ND * cw - 30},{gy + NP * ch + 46}", INK, 1.2))
P.append(text(gx + ND * cw, gy + NP * ch + 66, ["高维：低频", "变化很慢，编码大尺度位置"], 13, fill=INK, weight=600, anchor="end"))
# legend
lx, ly = gx + ND * cw + 18, gy + 10
for k in range(11):
    v = 1 - k / 5
    col = mix("#FFFFFF", ORANGE, v) if v >= 0 else mix("#FFFFFF", BLUE, -v)
    P.append(rect(lx, ly + k * 16, 14, 16, col, "none", 0, 0))
P.append(text(lx + 20, ly + 12, "+1", 12, fill=MUTED))
P.append(text(lx + 20, ly + 92, "0", 12, fill=MUTED))
P.append(text(lx + 20, ly + 174, "−1", 12, fill=MUTED))
P.append(text(gx + 4, 830, "波长从 2π 到 10000·2π 几何递增：像转速不同的指针，组合起来唯一标识位置", 14, fill=BLUE, weight=600))

# ======================= ③ curves + relative =======================
rx = 1100
P.append(step_badge(rx + 6, 196, 3))
P.append(text(rx + 28, 202, "三个维度上的信号 & 相对位置", 19, weight=700))
curves = []
for i, col in [(0, ORANGE), (32, BLUE), (96, MUTED)]:
    period = 2 * math.pi * 10000 ** (2 * i / d)
    curves.append((i, col, f"i = {i}（周期 ≈ {period:.1f}）" if period < 10 else f"i = {i}（周期 ≈ {period:.0f}）"))
cx0, cy0, cwid, chgt = rx + 10, 240, 430, 60
for k, (i, col, lab) in enumerate(curves):
    yb = cy0 + k * 92
    P.append(rect(cx0, yb, cwid, chgt, "#FFFFFF", LINE, 1, 6))
    P.append(line(cx0, yb + chgt / 2, cx0 + cwid, yb + chgt / 2, LINE, 1, dash="3 3"))
    pts = []
    for s_ in range(0, 201):
        pos = s_ * 50 / 200
        v = math.sin(pos / (10000 ** (2 * i / d)))
        pts.append(f"{cx0 + s_ * cwid / 200:.1f},{yb + chgt / 2 - v * (chgt / 2 - 6):.1f}")
    P.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2"/>')
    P.append(text(cx0, yb + chgt + 18, f"sin 分量，{lab}", 12, fill=col if col != MUTED else INK))
P.append(text(cx0 + cwid, cy0 + 3 * 92 + 0, "pos 0 → 50", 12, fill=MUTED, anchor="end"))

ry = 540
P.append(rect(rx, ry, 450, 186, ORANGE_LL, ORANGE, 1.4, 10))
P.append(text(rx + 18, ry + 30, "关键性质：PE_{pos+k} 是 PE_{pos} 的线性函数", 15, weight=700, fill=ORANGE))
P.append(text(rx + 18, ry + 64, "对每一对 (sin, cos)，平移 k 等价于旋转角度 ωk：", 14))
# matrix
mx, my = rx + 24, ry + 78
P.append(text(mx, my + 14, "[sin ω(p+k), cos ω(p+k)]^{T} = R_{k} · [sin ωp, cos ωp]^{T}", 14, family=MATH))
P.append(text(mx, my + 54, "R_{k} =", 15, family=MATH))
P.append(path(f"M{mx + 62},{my + 30} L{mx + 56},{my + 30} L{mx + 56},{my + 72} L{mx + 62},{my + 72}", INK, 1.3))
for r_, row in enumerate([("cos ωk", "sin ωk"), ("−sin ωk", "cos ωk")]):
    P.append(text(mx + 118, my + 47 + r_ * 20, row[0], 13, family=MATH, anchor="end"))
    P.append(text(mx + 186, my + 47 + r_ * 20, row[1], 13, family=MATH, anchor="end"))
P.append(path(f"M{mx + 192},{my + 30} L{mx + 198},{my + 30} L{mx + 198},{my + 72} L{mx + 192},{my + 72}", INK, 1.3))
P.append(text(mx + 216, my + 56, "← 二维旋转，只依赖 k", 13, fill=ORANGE, weight=600))
P.append(text(rx + 18, ry + 174, "矩阵只依赖偏移 k → 便于注意力学习“相对距离”", 14, fill=INK, weight=600))

# learned vs sinusoid
ly2 = 742
P.append(rect(rx, ly2, 450, 80, PANEL, LINE, 1.2, 10))
P.append(text(rx + 18, ly2 + 28, "可学习位置嵌入 vs 正弦编码（Table 3-E，dev BLEU）", 14, weight=700))
for k, (lab, v, col) in enumerate([("learned", 25.7, BLUE_L), ("sinusoid", 25.8, ORANGE)]):
    yy = ly2 + 42 + k * 20
    P.append(text(rx + 80, yy + 13, lab, 12, anchor="end", fill=MUTED))
    P.append(rect(rx + 88, yy, (v - 24) * 80, 15, col, "none", 0, 3))
    P.append(text(rx + 94 + (v - 24) * 80, yy + 13, f"{v}", 12, weight=600))
P.append(text(rx + 440, ly2 + 66, ["几乎相同；选 sin：", "可能外推到更长序列"], 12, fill=MUTED, anchor="end"))

P.append(footer(4))
render("04-positional-encoding", "".join(P))
