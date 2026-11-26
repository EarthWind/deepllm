"""07 · 关键结果：更高 BLEU，更少训练算力；以及为什么快"""
import math
from lib import *

P = []
P.append(header("关键结果：BLEU 更高，训练成本低一个数量级",
                "WMT 2014 newstest2014（Table 2）；右侧解释为什么能更快：每层顺序操作 O(1)、任意两位置路径 O(1)（Table 1）",
                7, "实验结果"))

# ======================= ① EN-DE scatter =======================
x0, y0, w0, h0 = 60, 178, 700, 660
P.append(rect(x0, y0, w0, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x0 + 24, y0 + 28, 1, fill=ORANGE))
P.append(text(x0 + 46, y0 + 34, "EN→DE：BLEU vs 训练算力（越靠左上越好）", 17, weight=700))
gx, gy, gw, gh = x0 + 80, y0 + 70, 580, 490
lx0, lx1 = math.log10(2e18), math.log10(4e20)
b0, b1 = 24, 29


def X(f):
    return gx + (math.log10(f) - lx0) / (lx1 - lx0) * gw


def Y(b):
    return gy + gh - (b - b0) / (b1 - b0) * gh


for b in range(24, 30):
    P.append(line(gx, Y(b), gx + gw, Y(b), "#EEF2F7", 1))
    P.append(text(gx - 10, Y(b) + 4, str(b), 12, fill=MUTED, anchor="end"))
for e, lab in [(1e19, "10^{19}"), (1e20, "10^{20}")]:
    P.append(line(X(e), gy, X(e), gy + gh, "#EEF2F7", 1))
    P.append(text(X(e), gy + gh + 20, lab, 12, fill=MUTED, anchor="middle"))
P.append(line(gx, gy + gh, gx + gw, gy + gh, MUTED, 1.2))
P.append(line(gx, gy, gx, gy + gh, MUTED, 1.2))
P.append(text(gx + gw, gy + gh + 42, "训练 FLOPs（对数轴）", 13, fill=MUTED, anchor="end"))
P.append(text(gx - 48, gy + gh / 2, "BLEU", 13, fill=MUTED, anchor="middle", rotate=-90))

pts = [("GNMT + RL", 24.6, 2.3e19, "single", (8, 16)),
       ("ConvS2S", 25.16, 9.6e18, "single", (-8, 18)),
       ("MoE", 26.03, 2.0e19, "single", (10, 5)),
       ("GNMT + RL 集成", 26.30, 1.8e20, "ens", (-10, 28)),
       ("ConvS2S 集成", 26.36, 7.7e19, "ens", (-10, -12)),
       ("Transformer base", 27.3, 3.3e18, "ours", (12, 5)),
       ("Transformer big", 28.4, 2.3e19, "ours", (12, 5))]
for name, b, f, kind, (dx, dy) in pts:
    cx, cy = X(f), Y(b)
    if kind == "ours":
        P.append(circle(cx, cy, 8, ORANGE, ORANGE))
        P.append(text(cx + dx, cy + dy, f"{name}  {b}", 15, fill=ORANGE, weight=700))
    elif kind == "ens":
        P.append(f'<rect x="{cx - 6}" y="{cy - 6}" width="12" height="12" fill="#FFFFFF" stroke="{BLUE}" stroke-width="2" transform="rotate(45 {cx} {cy})"/>')
        P.append(text(cx + dx, cy + dy, f"{name} {b}", 13, fill=BLUE, anchor="end" if dx < 0 else "start"))
    else:
        P.append(circle(cx, cy, 6, BLUE_L, BLUE, 1.6))
        P.append(text(cx + dx, cy + dy, f"{name} {b}", 13, fill=INK, anchor="end" if dx < 0 else "start"))
# +2.0 bracket
best_prev = Y(26.36)
P.append(line(X(2.3e19) - 24, best_prev, X(2.3e19) - 24, Y(28.4), ORANGE, 1.6, dash="4 3"))
P.append(line(X(2.3e19) - 30, best_prev, X(2.3e19) + 60, best_prev, ORANGE, 1, dash="2 3"))
P.append(text(X(2.3e19) - 16, Y(27.75), ["+2.0 BLEU", "超过此前最佳", "（含集成）"], 13, fill=ORANGE, weight=700))
# legend
lgy = y0 + h0 - 52
P.append(circle(x0 + 34, lgy, 6, BLUE_L, BLUE, 1.6))
P.append(text(x0 + 46, lgy + 5, "此前单模型", 13, fill=MUTED))
P.append(f'<rect x="{x0 + 152}" y="{lgy - 6}" width="12" height="12" fill="#FFFFFF" stroke="{BLUE}" stroke-width="2" transform="rotate(45 {x0 + 158} {lgy})"/>')
P.append(text(x0 + 172, lgy + 5, "此前集成模型", 13, fill=MUTED))
P.append(circle(x0 + 290, lgy, 7, ORANGE, ORANGE))
P.append(text(x0 + 302, lgy + 5, "Transformer", 13, fill=MUTED))
P.append(text(x0 + 24, y0 + h0 - 20, "base 用最少算力（3.3×10^{18}）已超过所有已发表模型，包括集成", 14, fill=INK, weight=600))

# ======================= ② EN-FR bars =======================
x1, w1 = 790, 400
P.append(rect(x1, y0, w1, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x1 + 24, y0 + 28, 2, fill=ORANGE))
P.append(text(x1 + 46, y0 + 34, "EN→FR：BLEU 与训练算力", 17, weight=700))
fr = [("GNMT + RL", 39.92, "1.4×10^{20}", BLUE_L),
      ("ConvS2S", 40.46, "1.5×10^{20}", BLUE_L),
      ("MoE", 40.56, "1.2×10^{20}", BLUE_L),
      ("Deep-Att 集成", 40.4, "8.0×10^{20}", "#C7D2FE"),
      ("GNMT 集成", 41.16, "1.1×10^{21}", "#C7D2FE"),
      ("ConvS2S 集成", 41.29, "1.2×10^{21}", "#C7D2FE"),
      ("Transformer big", 41.8, "2.3×10^{19}", ORANGE)]
bx, by = x1 + 120, y0 + 80
for i, (n, b, c, col) in enumerate(fr):
    yy = by + i * 68
    wbar = (b - 38) * 55
    P.append(text(bx - 10, yy + 20, n, 13, anchor="end", fill=ORANGE if col == ORANGE else INK,
                  weight=700 if col == ORANGE else 400))
    P.append(rect(bx, yy + 4, wbar, 24, col, "none", 0, 3))
    P.append(text(bx + wbar + 6, yy + 21, f"{b}", 13, weight=700 if col == ORANGE else 500,
                  fill=ORANGE if col == ORANGE else INK))
    P.append(text(bx, yy + 46, f"算力 {c}", 11, fill=ORANGE if col == ORANGE else MUTED))
P.append(text(bx, by - 14, "BLEU 轴从 38 起", 11, fill=FAINT))
P.append(text(x1 + 24, y0 + h0 - 44, ["单模型新 SOTA，训练算力不到此前", "最好单模型的 1/4（v7 表值 41.8；会议版 41.0）"], 13, fill=INK, lh=1.45))

# ======================= ③ why fast: Table 1 =======================
x2, w2 = 1220, 320
P.append(rect(x2, y0, w2, 370, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x2 + 24, y0 + 28, 3))
P.append(text(x2 + 46, y0 + 34, "为什么快：层类型对比", 17, weight=700))
cols = ["每层计算", "顺序操作", "最长路径"]
rows = [("Self-Attn", ["O(n^{2}·d)", "O(1)", "O(1)"], ORANGE),
        ("RNN", ["O(n·d^{2})", "O(n)", "O(n)"], INK),
        ("CNN", ["O(k·n·d^{2})", "O(1)", "O(log_{k} n)"], INK)]
tx = x2 + 84
for j, cname in enumerate(cols):
    P.append(ctext(tx + j * 74 + 37, y0 + 72, cname, 12, fill=MUTED))
for i, (name, vals, col) in enumerate(rows):
    yy = y0 + 92 + i * 48
    if col == ORANGE:
        P.append(rect(x2 + 12, yy, w2 - 24, 40, ORANGE_LL, "none", 0, 6))
    P.append(text(x2 + 22, yy + 25, name, 13, weight=700, fill=col))
    for j, v in enumerate(vals):
        P.append(ctext(tx + j * 74 + 37, yy + 20, v, 12, fill=col, family=FONT))
P.append(text(x2 + 22, y0 + 266, ["顺序操作 O(1) → 整段序列并行训练", "路径 O(1) → 长距离依赖一步可达", "n < d 时（句子几十 token，d=512）", "自注意力每层甚至比 RNN 更省"], 12, fill=INK, lh=1.45))

# ======================= ④ generalization =======================
gy2 = y0 + 390
P.append(rect(x2, gy2, w2, h0 - 390, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x2 + 24, gy2 + 28, 4))
P.append(text(x2 + 46, gy2 + 34, "迁移：英语成分句法分析", 16, weight=700))
P.append(text(x2 + 22, gy2 + 66, "4 层 Transformer，几乎不调参（WSJ 23 F1）", 12, fill=MUTED))
for i, (lab, v) in enumerate([("仅 WSJ 40K 句", 91.3), ("半监督 ≈17M 句", 92.7)]):
    yy = gy2 + 84 + i * 50
    P.append(text(x2 + 22, yy + 16, lab, 13))
    P.append(rect(x2 + 22, yy + 24, (v - 88) * 52, 16, ORANGE, "none", 0, 3))
    P.append(text(x2 + 28 + (v - 88) * 52, yy + 37, f"{v}", 13, weight=700, fill=ORANGE))
P.append(text(x2 + 22, gy2 + 194, "仅次于 RNN Grammar（93.3）", 12, fill=MUTED))

P.append(footer(7))
render("07-key-results", "".join(P))
