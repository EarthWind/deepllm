"""06 · 训练配方：数据、warmup 学习率、正则化、解码"""
from lib import *

P = []
P.append(header("训练配方：让 Transformer 在 8 张 P100 上 12 小时练成",
                "架构之外，这些训练细节同样写进了论文：长度分桶、warmup + 反平方根衰减、Dropout 与 Label Smoothing、checkpoint 平均与 beam search",
                6, "训练细节"))

TOP, BOT = 180, 520
PH = 320


def panel(x, y, w, h, n, title, col=INK):
    s = rect(x, y, w, h, "#FFFFFF", LINE, 1.2, 12)
    s += step_badge(x + 24, y + 28, n, fill=col)
    s += text(x + 46, y + 34, title, 17, weight=700, fill=col)
    return s


# ---------- ① data & batching ----------
x0, w0 = 60, 450
P.append(panel(x0, TOP, w0, PH, 1, "数据与批处理"))
P.append(text(x0 + 24, TOP + 70, "WMT 2014 EN→DE：约 4.5M 句对", 14, weight=600))
P.append(text(x0 + 24, TOP + 92, "BPE，源/目标共享词表 ≈ 37K", 13, fill=MUTED))
P.append(text(x0 + 24, TOP + 122, "WMT 2014 EN→FR：36M 句", 14, weight=600))
P.append(text(x0 + 24, TOP + 144, "word-piece 词表 32K", 13, fill=MUTED))
# length buckets
by = TOP + 172
P.append(text(x0 + 24, by, "按长度近似分桶 → 每批 ≈ 25K 源 + 25K 目标 token", 13, fill=BLUE, weight=600))
buckets = [[8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 9], [18, 19, 18, 20, 19], [34, 33, 35]]
bx = x0 + 24
for k, bucket in enumerate(buckets):
    yy = by + 10 + k * 34
    P.append(text(bx, yy + 24, f"batch {k + 1}", 12, fill=MUTED))
    xx = bx + 62
    for L in bucket:
        P.append(rect(xx, yy + 8, L * 0.95, 22, mix(BLUE_L, BLUE, 0.2 + 0.25 * k), "#FFFFFF", 1, 3))
        xx += L * 0.95 + 2
P.append(text(x0 + 24, TOP + PH - 12, "长度相近 → padding 少，GPU 利用率高", 12, fill=MUTED))

# ---------- ② LR schedule ----------
x1, w1 = 540, 560
P.append(panel(x1, TOP, w1, PH, 2, "学习率：线性 warmup，再按 1/√step 衰减", ORANGE))
P.append(text(x1 + 24, TOP + 66, "lrate = d_{model}^{-0.5} · min(step^{-0.5}, step · warmup^{-1.5})", 15, family=MATH))
gx, gy, gw, gh = x1 + 70, TOP + 92, 450, 170
P.append(line(gx, gy + gh, gx + gw, gy + gh, MUTED, 1.2))
P.append(line(gx, gy, gx, gy + gh, MUTED, 1.2))
ymax = 8e-4
smax = 100000
warm = 4000


def lr(s):
    s = max(s, 1)
    return 512 ** -0.5 * min(s ** -0.5, s * warm ** -1.5)


def X(s):
    return gx + s / smax * gw


def Y(v):
    return gy + gh - v / ymax * gh


P.append(rect(X(0), gy, X(warm) - X(0), gh, ORANGE_LL, "none", 0, 0))
pts = " ".join(f"{X(s):.1f},{Y(lr(s)):.1f}" for s in list(range(0, warm, 100)) + list(range(warm, smax + 1, 500)))
P.append(f'<polyline points="{pts}" fill="none" stroke="{ORANGE}" stroke-width="2.6"/>')
peak = lr(warm)
P.append(circle(X(warm), Y(peak), 4.5, ORANGE, ORANGE))
P.append(text(X(warm) + 10, Y(peak) + 4, f"峰值 ≈ {peak * 1e4:.1f}×10^{{-4}}（step 4000）", 13, fill=ORANGE, weight=600))
P.append(text(X(warm) + 4, gy + gh - 8, "warmup 4000 步", 12, fill=ORANGE))
end = lr(smax)
P.append(text(X(smax), Y(end) - 10, f"{end * 1e4:.1f}×10^{{-4}}", 12, fill=MUTED, anchor="end"))
for v in [0, 2e-4, 4e-4, 6e-4, 8e-4]:
    P.append(text(gx - 8, Y(v) + 4, f"{v * 1e4:.0f}e-4" if v else "0", 11, fill=MUTED, anchor="end"))
for s in [0, 25000, 50000, 75000, 100000]:
    P.append(text(X(s), gy + gh + 16, f"{s // 1000}K", 11, fill=MUTED, anchor="middle"))
P.append(text(gx + gw, gy + gh + 34, "训练步数（base 共 100K 步）", 12, fill=MUTED, anchor="end"))
P.append(text(x1 + 24, TOP + PH - 12, "早期参数随机，大步长易发散 → 先小后大；之后慢慢收敛", 12, fill=MUTED))

# ---------- ③ optimizer & hardware ----------
x2, w2 = 1130, 410
P.append(panel(x2, TOP, w2, PH, 3, "优化器与算力"))
rows = [("优化器", "Adam"), ("β_{1} / β_{2}", "0.9 / 0.98"), ("ε", "10^{-9}"), ("硬件", "1 台机器 · 8×P100")]
for i, (k, v) in enumerate(rows):
    yy = TOP + 72 + i * 28
    P.append(text(x2 + 24, yy, k, 14, fill=MUTED))
    P.append(text(x2 + w2 - 24, yy, v, 14, weight=600, anchor="end"))
P.append(line(x2 + 24, TOP + 180, x2 + w2 - 24, TOP + 180, LINE, 1))
for i, (lab, steps, t, tot, col) in enumerate([("base", "100K 步", "0.4 s/步", "≈ 12 小时", BLUE),
                                               ("big", "300K 步", "1.0 s/步", "≈ 3.5 天", ORANGE)]):
    yy = TOP + 200 + i * 52
    P.append(tag(x2 + 24, yy, lab, fill=col))
    P.append(text(x2 + 90, yy + 17, f"{steps} × {t}", 14))
    P.append(text(x2 + w2 - 24, yy + 17, tot, 16, weight=700, fill=col, anchor="end"))
P.append(text(x2 + 24, TOP + PH - 12, "注意 β_{2}=0.98（常见默认 0.999）", 12, fill=MUTED))

# ---------- ④ teacher forcing vs inference ----------
P.append(panel(x0, BOT, w0, PH, 4, "训练并行 vs 推理逐步", BLUE))
ty = BOT + 66
P.append(text(x0 + 24, ty, "训练（Teacher Forcing）：一次前向算完所有位置", 14, weight=600, fill=BLUE))
inp = ["<s>", "我", "爱", "NLP"]
out = ["我", "爱", "NLP", "</s>"]
for i in range(4):
    xx = x0 + 40 + i * 98
    P.append(chip(xx, ty + 74, 70, 28, inp[i], size=13))
    P.append(arrow(xx + 35, ty + 72, xx + 35, ty + 48, BLUE, 1.5))
    P.append(chip(xx, ty + 16, 70, 28, out[i], fill=BLUE_L, stroke=BLUE, color=BLUE, size=13))
P.append(text(x0 + 24, ty + 126, "输入 = 目标右移一位；因果掩码保证位置 i 看不到 i 之后", 12, fill=MUTED))
P.append(line(x0 + 24, ty + 142, x0 + w0 - 24, ty + 142, LINE, 1))
P.append(text(x0 + 24, ty + 168, "推理：每步生成 1 个 token，再接回输入", 14, weight=600))
seq = ["<s>", "<s> 我", "<s> 我 爱", "… </s>"]
for i, sq in enumerate(seq):
    xx = x0 + 24 + i * 104
    P.append(chip(xx, ty + 184, 96, 28, sq, size=12))
    if i < 3:
        P.append(arrow(xx + 97, ty + 198, xx + 103, ty + 198, MUTED, 1.2))
P.append(text(x0 + 24, ty + 238, "推理仍是串行的；Transformer 的提速主要在训练", 12, fill=MUTED))

# ---------- ⑤ regularization ----------
P.append(panel(x1, BOT, w1, PH, 5, "正则化：Dropout + Label Smoothing", ORANGE))
ry = BOT + 70
P.append(text(x1 + 24, ry, "Residual Dropout：P_{drop} = 0.1", 14, weight=600))
P.append(text(x1 + 24, ry + 20, "子层输出、embedding+PE 之和（big EN-DE 为 0.3）", 12, fill=MUTED))
P.append(text(x1 + 24, ry + 54, "Label Smoothing：ε_{ls} = 0.1", 14, weight=600))
# target distribution bars
labels = ["我", "你", "他", "爱", "吃"]
for k, (name, dist, col) in enumerate([("one-hot 目标", [1, 0, 0, 0, 0], FAINT),
                                       ("平滑后（示意 5 类）", [0.92, 0.02, 0.02, 0.02, 0.02], ORANGE)]):
    gx0 = x1 + 34 + k * 250
    base = ry + 160
    P.append(text(gx0, ry + 80, name, 12, fill=MUTED))
    for i, v in enumerate(dist):
        h = v * 62 + 1
        P.append(rect(gx0 + i * 34, base - h, 24, h, col, "none", 0, 2))
        P.append(ctext(gx0 + i * 34 + 12, base + 12, labels[i], 11, fill=MUTED))
        if v:
            P.append(ctext(gx0 + i * 34 + 12, base - h - 8, f"{v:g}", 10, fill=INK))
P.append(text(x1 + 24, BOT + PH - 34, "让模型“不那么确定”：困惑度变差，但准确率与 BLEU 提升", 13, fill=INK))
P.append(text(x1 + 24, BOT + PH - 12, "消融（dev BLEU）：base 25.8 · 去掉 ε_{ls} 25.3 · 去掉 dropout 24.6", 12, fill=MUTED))

# ---------- ⑥ decoding tricks ----------
P.append(panel(x2, BOT, w2, PH, 6, "收尾：平均与解码"))
items = [("Checkpoint 平均", ["base：最后 5 个（每 10 分钟存 1 个）", "big：最后 20 个"]),
         ("Beam Search", ["beam size = 4，长度惩罚 α = 0.6"]),
         ("最大输出长度", ["输入长度 + 50，能提前结束就结束"])]
yy = BOT + 74
for k, (h, lines) in enumerate(items):
    P.append(circle(x2 + 30, yy - 5, 4, BLUE, BLUE))
    P.append(text(x2 + 42, yy, h, 14, weight=600))
    P.append(text(x2 + 42, yy + 22, lines, 13, fill=MUTED, lh=1.45))
    yy += 30 + 20 * len(lines) + 16
P.append(text(x2 + 24, BOT + PH - 12, "超参均在开发集 newstest2013 上选定", 12, fill=MUTED))

P.append(footer(6))
render("06-training-recipe", "".join(P))
