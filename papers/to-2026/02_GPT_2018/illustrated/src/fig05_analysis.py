"""05 · 为什么有效：迁移层数实验 + zero-shot 启发式 + Transformer vs LSTM"""
from lib import *

P = []
P.append(header("为什么有效：每一层都学到了有用的东西，语言模型本身已会“做任务”",
                "Section 5 · Figure 2（左：迁移层数；右：zero-shot 表现随预训练步数变化）· Table 5（LSTM 对照）",
                5, "分析"))
y0, h0 = 178, 660

# ======================= ① layers transferred =======================
x0, w0 = 60, 640
P.append(rect(x0, y0, w0, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x0 + 24, y0 + 28, 1, fill=BLUE))
P.append(text(x0 + 46, y0 + 34, "迁移前 k 层，其余层随机初始化", 17, weight=700))
# layer stack illustration (k = 5 as example)
sx, sy = x0 + 40, y0 + 62
k = 5
for i in range(12):
    yy = sy + (11 - i) * 29
    on = i < k
    P.append(rect(sx, yy, 150, 24, BLUE_L if on else "#FFFFFF", BLUE if on else LINE, 1.3, 5,
                  dash=None if on else "4 3"))
    P.append(ctext(sx + 75, yy + 12, f"第 {i + 1} 层" + ("（迁移）" if on else "（随机）"), 12,
                   fill=BLUE if on else FAINT, weight=600 if on else 400))
P.append(rect(sx, sy + 12 * 29, 150, 24, BLUE, BLUE, 1, 5))
P.append(ctext(sx + 75, sy + 12 * 29 + 12, "嵌入层（迁移）", 12, fill="#FFFFFF", weight=700))
P.append(text(sx + 75, sy + 13 * 29 + 18, "例：k = 5", 13, fill=MUTED, anchor="middle"))
# schematic curve
gx, gy, gw, gh = x0 + 260, y0 + 90, 340, 330
P.append(line(gx, gy + gh, gx + gw, gy + gh, MUTED, 1.2))
P.append(line(gx, gy, gx, gy + gh, MUTED, 1.2))
for i in (0, 6, 12):
    xx = gx + i / 12 * gw
    P.append(text(xx, gy + gh + 20, str(i), 12, fill=MUTED, anchor="middle"))
P.append(text(gx + gw / 2, gy + gh + 44, "迁移的 Transformer 层数", 13, fill=MUTED, anchor="middle"))
P.append(text(gx - 16, gy + gh / 2, "Dev 准确率", 13, fill=MUTED, anchor="middle", rotate=-90))
pts_m = [(0, .18), (1, .30), (2, .33), (3, .40), (4, .45), (5, .48), (6, .55), (7, .60), (8, .64), (9, .68),
         (10, .74), (11, .78), (12, .82)]
pts_r = [(0, .10), (1, .16), (2, .15), (3, .26), (4, .30), (5, .34), (6, .42), (7, .44), (8, .45), (9, .48),
         (10, .55), (11, .58), (12, .62)]


def poly(pts, col):
    d = " ".join(f"{'M' if i == 0 else 'L'}{gx + a / 12 * gw:.1f},{gy + gh - b * gh:.1f}" for i, (a, b) in enumerate(pts))
    return path(d, col, 2.6)


P.append(poly(pts_m, ORANGE))
P.append(poly(pts_r, BLUE))
P.append(text(gx + gw - 4, gy + gh - .82 * gh - 12, "MultiNLI", 13, fill=ORANGE, weight=700, anchor="end"))
P.append(text(gx + gw - 4, gy + gh - .62 * gh + 34, "RACE", 13, fill=BLUE, weight=700, anchor="end"))
P.append(text(gx + 8, gy + 12, "趋势示意（据 Fig. 2 左重绘，非精确数值）", 12, fill=FAINT))
ny = y0 + 500
P.append(rect(x0 + 20, ny, w0 - 40, 140, BLUE_LL, BLUE, 1.2, 10))
P.append(text(x0 + 40, ny + 32, ["• 只迁移嵌入层就有提升（与预训练词向量的经验一致）；",
                                  "• **每多迁移一层都继续变好**，MultiNLI 上全量迁移最多 +9%；",
                                  "• 说明预训练模型的**每一层**都包含对下游任务有用的功能。"], 14, lh=1.7))

# ======================= ② zero-shot heuristics =======================
x1, w1 = 720, 820
P.append(rect(x1, y0, w1, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x1 + 24, y0 + 28, 2, fill=ORANGE))
P.append(text(x1 + 46, y0 + 34, "不微调，直接用语言模型的概率“做任务”（zero-shot 启发式）", 17, weight=700))
cards = [
    ("CoLA · 语法可接受性", ["句子的平均 token 对数概率", "超过阈值 → 判为“合语法”"], "log P(句子) > τ ?"),
    ("SST-2 · 情感", ["在句末接上 “very”，只比较", "下一个词是 positive 还是 negative"], "“… very” → positive / negative"),
    ("RACE · 阅读理解", ["在文章 + 问题条件下，", "选平均对数概率最高的答案"], "argmax_{k} log P(a_{k} | z, q)"),
    ("DPRD · 指代消解（Winograd）", ["把代词分别替换成两个候选，", "看哪种替换让后文概率更高"], "P(后文 | 候选 A) vs P(后文 | 候选 B)"),
]
cw, ch = (w1 - 60) / 2, 150
for i, (t, lines, f) in enumerate(cards):
    cx_ = x1 + 20 + (i % 2) * (cw + 20)
    cy_ = y0 + 60 + (i // 2) * (ch + 16)
    P.append(rect(cx_, cy_, cw, ch, ORANGE_LL, ORANGE, 1.2, 10))
    P.append(text(cx_ + 18, cy_ + 30, t, 15, weight=700, fill=ORANGE))
    P.append(text(cx_ + 18, cy_ + 60, lines, 14, lh=1.55))
    P.append(rect(cx_ + 18, cy_ + 104, cw - 36, 32, "#FFFFFF", LINE, 1, 6))
    P.append(ctext(cx_ + cw / 2, cy_ + 120, f, 13, fill=INK, weight=600))
by = y0 + 60 + 2 * (ch + 16) + 4
P.append(rect(x1 + 20, by, (w1 - 60) / 2, h0 - (by - y0) - 20, BLUE_LL, BLUE, 1.2, 10))
P.append(text(x1 + 38, by + 32, "观察（Fig. 2 右）", 15, weight=700, fill=BLUE))
P.append(text(x1 + 38, by + 62, ["随着预训练步数增加，这些启发式", "的表现**稳定上升**：生成式预训练",
                                  "在学语言的同时学到了任务相关能力；", "同样做法下 LSTM 的波动更大。", "", "（图中分数在随机猜测与当时单模型", "SOTA 之间归一化）"], 14, lh=1.6))
rx = x1 + 40 + (w1 - 60) / 2
P.append(rect(rx, by, (w1 - 60) / 2, h0 - (by - y0) - 20, "#FFFFFF", LINE, 1.2, 10))
P.append(text(rx + 18, by + 32, "Transformer vs 单层 LSTM（Table 5）", 15, weight=700))
bars = [("Transformer（完整）", 74.7, ORANGE), ("LSTM（2048 单元）", 69.1, MUTED)]
for i, (lab, v, col) in enumerate(bars):
    yy = by + 58 + i * 50
    P.append(text(rx + 18, yy + 18, lab, 13))
    bw_ = (v - 50) / 30 * 200
    P.append(rect(rx + 160, yy, bw_, 26, col if col != MUTED else PANEL2, col, 1.2, 4))
    P.append(text(rx + 168 + bw_, yy + 19, f"{v}", 14, fill=col, weight=700))
P.append(text(rx + 18, by + 176, ["同一框架换成 LSTM，平均分降 5.6；", "只在 MRPC 一项上 LSTM 更好。"], 13, fill=MUTED, lh=1.5))

P.append(footer(5))
render("05-why-it-works", "".join(P))
