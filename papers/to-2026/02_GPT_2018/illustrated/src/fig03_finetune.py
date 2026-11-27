"""03 · 有监督微调：Extract 位置 → 线性头；辅助 LM 损失；新增参数与超参"""
from lib import *

P = []
P.append(header("有监督微调：取最后一个 token 的表示接线性分类头，同时保留语言模型损失",
                "Eq. 3：P(y | x^{1..m}) = softmax(h_{l}^{m} W_{y})；Eq. 5：L_{3} = L_{2} + λ·L_{1}（Section 3.2、4.1，Table 5）",
                3, "核心机制 ②"))

y0, h0 = 178, 660
# ======================= ① two heads on one trunk =======================
x0, w0 = 60, 600
P.append(rect(x0, y0, w0, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x0 + 24, y0 + 28, 1, fill=ORANGE))
P.append(text(x0 + 46, y0 + 34, "同一次前向，两个输出头", 17, weight=700))
seq = ["Start", "x^{1}", "x^{2}", "…", "x^{m}", "Extract"]
cw_, cg = 76, 12
sx0 = x0 + (w0 - (len(seq) * cw_ + (len(seq) - 1) * cg)) / 2
cxs = [sx0 + i * (cw_ + cg) + cw_ / 2 for i in range(len(seq))]
# inputs
iy = y0 + 590
for i, t in enumerate(seq):
    special = t in ("Start", "Extract")
    P.append(chip(cxs[i] - cw_ / 2, iy, cw_, 34, t, fill=ORANGE_L if special else PANEL2,
                  stroke=ORANGE if special else LINE, color=ORANGE if special else INK, size=14))
    P.append(arrow(cxs[i], iy - 2, cxs[i], iy - 26, MUTED, 1.3))
P.append(text(x0 + w0 / 2, iy + 58, "起止 token 与分隔符的嵌入随机初始化，随微调一起学习", 13, fill=ORANGE, anchor="middle",
              weight=600))
# trunk
ty, th = y0 + 380, 180
P.append(rect(sx0 - 14, ty, cxs[-1] + cw_ / 2 - sx0 + 28, th, BLUE_L, BLUE, 1.6, 12))
P.append(ctext(x0 + w0 / 2, ty + 70, "预训练好的 12 层 Transformer  Θ", 19, fill=BLUE, weight=700))
P.append(ctext(x0 + w0 / 2, ty + 104, "全部参数参与微调（不冻结）；因果掩码不变", 14, fill=INK))
P.append(ctext(x0 + w0 / 2, ty + 136, "Extract 在最后，能看到整条输入", 14, fill=INK))
# per-position hidden states
hy = ty - 62
for i, t in enumerate(seq):
    last = i == len(seq) - 1
    P.append(arrow(cxs[i], ty - 2, cxs[i], hy + 34, BLUE if not last else ORANGE, 1.4))
    P.append(chip(cxs[i] - cw_ / 2, hy, cw_, 32, "h_{l}^{m}" if last else "h", fill=ORANGE if last else "#FFFFFF",
                  stroke=ORANGE if last else BLUE, color="#FFFFFF" if last else BLUE, size=14, weight=700))
# LM head (all positions except Extract)
ly = y0 + 150
P.append(rect(cxs[0] - cw_ / 2, ly, cxs[-2] + cw_ / 2 - (cxs[0] - cw_ / 2), 58, BLUE_LL, BLUE, 1.4, 10, dash="6 4"))
P.append(ctext((cxs[0] + cxs[-2]) / 2, ly + 20, "LM 头 softmax(hW_{e}^{T})", 15, fill=BLUE, weight=700))
P.append(ctext((cxs[0] + cxs[-2]) / 2, ly + 42, "逐位置预测下一个 token → 辅助损失 L_{1}", 13, fill=INK))
for i in range(len(seq) - 1):
    P.append(arrow(cxs[i], hy - 2, cxs[i], ly + 60, BLUE, 1.2, dash="4 3"))
# classifier head
kx = cxs[-1]
P.append(box(kx - 70, y0 + 132, 140, 44, "Linear W_{y}", 15, fill=ORANGE_L, stroke=ORANGE, color=ORANGE))
P.append(arrow(kx, hy - 2, kx, y0 + 178, ORANGE, 2))
P.append(box(kx - 70, y0 + 66, 140, 44, "softmax → ŷ", 15, fill=ORANGE, stroke=ORANGE, color="#FFFFFF"))
P.append(arrow(kx, y0 + 130, kx, y0 + 112, ORANGE, 2))
P.append(text(kx - 86, y0 + 92, "任务损失 L_{2}", 14, fill=ORANGE, weight=700, anchor="end"))

# ======================= ② auxiliary LM: does it help? =======================
x1, w1 = 680, 470
P.append(rect(x1, y0, w1, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x1 + 24, y0 + 28, 2, fill=BLUE))
P.append(text(x1 + 46, y0 + 34, "辅助 LM 损失：L_{3} = L_{2} + 0.5·L_{1}", 17, weight=700))
P.append(text(x1 + 24, y0 + 70, ["作者给出的两个理由：(a) 提升监督模型的泛化，", "(b) 加快收敛。下图为 Table 5 消融：",
                                 "有辅助 LM 减去无辅助 LM 的得分差"], 14, fill=INK, lh=1.6))
data = [("MNLI", 0.7), ("QNLI", 1.2), ("QQP", 0.5), ("RTE", 1.6),
        ("SST-2", -0.7), ("STS-B", -1.2), ("CoLA", -2.5), ("MRPC", -2.6)]
bx0, by0 = x1 + 100, y0 + 160
zero = x1 + 270
scale = 52
rowh = 40
P.append(line(zero, by0 - 8, zero, by0 + len(data) * rowh, MUTED, 1.2))
for i, (name, d) in enumerate(data):
    y = by0 + i * rowh
    col = BLUE if d > 0 else ORANGE
    w = abs(d) * scale
    xx = zero if d > 0 else zero - w
    P.append(text(x1 + 84, y + 21, name, 14, weight=600, anchor="end"))
    P.append(rect(xx, y + 6, w, 22, BLUE_L if d > 0 else ORANGE_L, col, 1.3, 4))
    lab = f"+{d:.1f}" if d > 0 else f"{d:.1f}"
    P.append(text(zero + w + 8 if d > 0 else zero - w - 8, y + 22, lab, 14, fill=col, weight=700,
                  anchor="start" if d > 0 else "end"))
P.append(text(zero + 6, by0 - 14, "有辅助 LM 更好 →", 12, fill=BLUE, weight=600))
P.append(text(zero - 6, by0 - 14, "← 去掉反而更好", 12, fill=ORANGE, weight=600, anchor="end"))
ny = by0 + len(data) * rowh + 18
P.append(rect(x1 + 20, ny, w1 - 40, 116, BLUE_LL, BLUE, 1.2, 10))
P.append(text(x1 + 38, ny + 30, ["论文结论：**大数据集（NLI、QQP）受益，小数据集不受益**；",
                                  "RTE（2,490 例）是例外。8 项平均分：",
                                  "有辅助 LM 74.7，无辅助 LM 75.0，差别很小。"], 14, lh=1.65))

# ======================= ③ what is new + hyperparameters =======================
x2, w2 = 1170, 370
P.append(rect(x2, y0, w2, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x2 + 24, y0 + 28, 3, fill=ORANGE))
P.append(text(x2 + 46, y0 + 34, "新增什么、怎么训", 17, weight=700))
P.append(rect(x2 + 16, y0 + 58, w2 - 32, 150, ORANGE_LL, ORANGE, 1.2, 10))
P.append(text(x2 + 34, y0 + 88, "微调新增的参数只有", 15, weight=700, fill=ORANGE))
P.append(text(x2 + 34, y0 + 120, ["• 输出层 W_{y}（768 × 类别数）", "• 起止 token、分隔符 $ 的嵌入",
                                   "其余全部来自预训练"], 14, lh=1.65))
rows = [("学习率", "6.25e-5"), ("批大小", "32"), ("轮数", "3 epochs（多数任务）"),
        ("调度", "线性衰减，前 0.2% 步升温"), ("dropout", "分类器 0.1"), ("λ", "0.5"),
        ("其余", "沿用预训练超参")]
ry = y0 + 228
for i, (k, v) in enumerate(rows):
    y = ry + i * 34
    if i % 2 == 0:
        P.append(rect(x2 + 16, y - 4, w2 - 32, 34, PANEL, PANEL, 1, 6))
    P.append(text(x2 + 30, y + 18, k, 14, weight=700, fill=BLUE))
    P.append(text(x2 + 112, y + 18, v, 14))
cy = ry + len(rows) * 34 + 14
P.append(rect(x2 + 16, cy, w2 - 32, h0 - (cy - y0) - 16, PANEL2, LINE, 1.2, 10))
P.append(text(x2 + 34, cy + 30, "对比此前做法", 15, weight=700))
P.append(text(x2 + 34, cy + 60, ["预训练表示只当特征，每个任务", "另起一套模型和大量新参数；",
                                  "GPT 微调很快，3 轮就够。"], 14, lh=1.6))

P.append(footer(3))
render("03-supervised-finetuning", "".join(P))
