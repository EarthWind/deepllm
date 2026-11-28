"""05 · 规模与泛化：越大越好（Table 3 逐规模数值）；不是背题（Table 6 重叠分析）"""
import math
from lib import *

P = []
P.append(header("规模与泛化：模型越大越好，而且不是靠背测试集",
                "左：Table 3 四种规模的 zero-shot 结果（原文数值）；右：Section 4 / Table 6，用 8-gram 重叠检查训练集是否“泄题”",
                5, "规模与泛化"))
P.append(legend(W - 60, 62, ("lm", "base")))
y0, h0 = 178, 660

# ======================= ① scaling small multiples =======================
x0, w0 = 60, 880
P.append(rect(x0, y0, w0, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x0 + 24, y0 + 28, 1, fill=INK))
P.append(text(x0 + 46, y0 + 34, "参数量增大，zero-shot 表现稳定变好", 17, weight=700))
P.append(line(x0 + 500, y0 + 28, x0 + 530, y0 + 28, LM, 3))
P.append(circle(x0 + 515, y0 + 28, 5, LM, LM))
P.append(text(x0 + 538, y0 + 33, "WebText 语言模型", 13, fill=LM, weight=600))
P.append(line(x0 + 680, y0 + 28, x0 + 712, y0 + 28, MUTED, 2, dash="6 4"))
P.append(text(x0 + 720, y0 + 33, "此前 SOTA", 13, fill=MUTED))

sizes = [117, 345, 762, 1542]
charts = [("LAMBADA 准确率 %", "↑ 越高越好", [45.99, 55.48, 60.12, 63.24], 59.23, (40, 70)),
          ("CBT 普通名词 准确率 %", "↑ 越高越好", [87.65, 92.35, 93.45, 93.30], 85.7, (84, 96)),
          ("WikiText-2 困惑度", "↓ 越低越好", [29.41, 22.76, 19.93, 18.34], 39.14, (15, 42)),
          ("PTB 困惑度", "↓ 越低越好", [65.85, 47.33, 40.31, 35.76], 46.54, (30, 70))]
gw, gh = 390, 230
for k, (title, hint, vals, sota, (lo, hi)) in enumerate(charts):
    gx = x0 + 34 + (k % 2) * (gw + 40)
    gy = y0 + 70 + (k // 2) * (gh + 40)
    P.append(text(gx, gy + 4, title, 14, weight=700))
    P.append(text(gx + gw, gy + 4, hint, 12, fill=MUTED, anchor="end"))
    px0, px1 = gx + 46, gx + gw - 20
    py0, py1 = gy + 30, gy + gh - 34

    def X(s):
        return px0 + (math.log(s) - math.log(117)) / (math.log(1542) - math.log(117)) * (px1 - px0)

    def Y(v):
        return py1 - (v - lo) / (hi - lo) * (py1 - py0)

    P.append(line(px0, py1, px1, py1, LINE, 1.2))
    for t in (lo, (lo + hi) / 2, hi):
        P.append(text(px0 - 8, Y(t) + 4, f"{t:g}", 11, fill=FAINT, anchor="end"))
        P.append(line(px0, Y(t), px1, Y(t), "#F1F5F9", 1))
    for s in sizes:
        P.append(text(X(s), py1 + 18, f"{s}M" if s < 1000 else "1.5B", 11, fill=MUTED, anchor="middle"))
    P.append(line(px0, Y(sota), px1, Y(sota), MUTED, 1.6, dash="6 4"))
    if k == 0:
        P.append(text(px0 + 4, Y(sota) - 6, f"SOTA {sota}", 11, fill=MUTED))
    else:
        P.append(text(px1, Y(sota) - 6, f"SOTA {sota}", 11, fill=MUTED, anchor="end"))
    pts = " L".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(sizes, vals))
    P.append(path("M" + pts, LM, 2.6))
    for s, v in zip(sizes, vals):
        P.append(circle(X(s), Y(v), 5, LM, LM))
    P.append(text(X(1542) - 8, Y(vals[-1]) + (18 if hint.startswith("↑") else -10), f"{vals[-1]}", 12,
                  fill=LM, weight=700, anchor="end"))
    P.append(text(X(117) + 8, Y(vals[0]) + (18 if hint.startswith("↑") else -10), f"{vals[0]}", 12, fill=LM, anchor="start"))

P.append(rect(x0 + 24, y0 + h0 - 82, w0 - 48, 62, PANEL, LINE, 1, 8))
P.append(text(x0 + 42, y0 + h0 - 56, ["117M 时 LAMBADA 和 PTB 还不如 SOTA，到 1.5B 四项全部超过。横轴为参数量（对数刻度）。",
                                       "1.5B 仍在**欠拟合** WebText：训练集和留出集的困惑度随规模一起下降，二者很接近（Figure 4）。"],
              13, lh=1.6))

# ======================= ② memorization =======================
x1, w1 = 960, 580
P.append(rect(x1, y0, w1, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x1 + 24, y0 + 28, 2, fill=INK))
P.append(text(x1 + 46, y0 + 34, "是背下来的吗？测试集 8-gram 重叠率", 17, weight=700))
P.append(text(x1 + 24, y0 + 62, "把 WebText 训练集的全部 8-gram 放进 Bloom filter，再查测试集", 12, fill=MUTED))
P.append(rect(x1 + 24, y0 + 78, 14, 12, INK, INK, 1, 2))
P.append(text(x1 + 44, y0 + 89, "与该数据集自己的训练集", 12))
P.append(rect(x1 + 230, y0 + 78, 14, 12, BASE_L, FAINT, 1, 2))
P.append(text(x1 + 250, y0 + 89, "与 WebText 训练集", 12))
t6 = [("PTB", 2.67, 0.88), ("WikiText-2", 0.66, 1.63), ("enwik8", 7.50, 6.31), ("text8", 2.34, 3.94),
      ("WikiText-103", 9.09, 2.42), ("1BW", 13.19, 3.75)]
bx = x1 + 130
for i, (n, own, wt) in enumerate(t6):
    y = y0 + 104 + i * 42
    P.append(text(bx - 12, y + 22, n, 13, weight=600, anchor="end"))
    P.append(rect(bx, y + 4, own / 14 * 330, 14, "#4B5563", "#4B5563", 1, 2))
    P.append(text(bx + own / 14 * 330 + 6, y + 16, f"{own:.2f}%", 11))
    P.append(rect(bx, y + 21, wt / 14 * 330, 14, BASE_L, FAINT, 1, 2))
    P.append(text(bx + wt / 14 * 330 + 6, y + 33, f"{wt:.2f}%", 11, fill=MUTED))
ty = y0 + 396
P.append(text(x1 + 24, ty, ["与 WebText 的重叠平均 **3.2%**，反而低于各数据集", "自己训练集与测试集之间的平均 **5.9%**。"], 13, lh=1.55))
facts = [("LAMBADA", "去掉所有有重叠的样本：困惑度 8.6 → 8.7，准确率 63.2% → 62.9%"),
         ("CoQA", "新闻域约 15% 文档在 WebText 里，该域高约 3 F1；5 个域平均只多 0.5–1.0 F1"),
         ("Winograd", "273 题中只有 10 题有重叠，其中只有 1 题的上下文泄露了答案")]
for i, (n, s) in enumerate(facts):
    y = ty + 52 + i * 52
    P.append(rect(x1 + 24, y, w1 - 48, 44, PANEL, LINE, 1, 8))
    P.append(text(x1 + 40, y + 27, n, 13, weight=700))
    P.append(text(x1 + 130, y + 27, s, 12))
P.append(text(x1 + 24, y0 + h0 - 22, "结论：重叠带来小而稳定的好处，但不是 GPT-2 成绩的主要来源。", 13, weight=700))

P.append(footer(5))
render("05-scale-and-generalization", "".join(P))
