"""06 · 关键结果：12 个数据集 9 个 SOTA；消融：预训练最关键"""
from lib import *

P = []
P.append(header("关键结果：12 个数据集中 9 个刷新 SOTA；去掉预训练平均分掉 14.8",
                "左：Table 2–4，此前最佳为表中最好成绩（含集成模型）；右：Table 5 消融，8 个 GLUE 任务的平均分",
                6, "实验结果"))
y0, h0 = 178, 660

# ======================= ① dumbbell: prev best vs GPT =======================
x0, w0 = 60, 900
P.append(rect(x0, y0, w0, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x0 + 24, y0 + 28, 1, fill=ORANGE))
P.append(text(x0 + 46, y0 + 34, "此前最佳 vs GPT（Finetuned Transformer LM）", 17, weight=700))
P.append(circle(x0 + 560, y0 + 28, 7, "#FFFFFF", MUTED, 2))
P.append(text(x0 + 574, y0 + 33, "此前最佳", 13, fill=MUTED))
P.append(circle(x0 + 660, y0 + 28, 7, ORANGE, ORANGE))
P.append(text(x0 + 674, y0 + 33, "GPT（刷新）", 13, fill=ORANGE, weight=600))
P.append(circle(x0 + 780, y0 + 28, 7, BLUE, BLUE))
P.append(text(x0 + 794, y0 + 33, "GPT（未刷新）", 13, fill=BLUE, weight=600))
groups = [("推理", [("MNLI-m", 80.6, 82.1), ("SNLI", 89.3, 89.9), ("SciTail", 83.3, 88.3), ("QNLI", 82.3, 88.1),
                  ("RTE", 61.7, 56.0)]),
          ("问答", [("Story Cloze", 77.6, 86.5), ("RACE", 53.3, 59.0)]),
          ("相似度", [("MRPC", 86.0, 82.3), ("STS-B", 81.0, 82.0), ("QQP", 66.1, 70.3)]),
          ("分类", [("CoLA", 35.0, 45.4), ("SST-2", 93.2, 91.3)])]
ax0, ax1 = x0 + 230, x0 + 760
v0, v1 = 30, 95


def X(v):
    return ax0 + (v - v0) / (v1 - v0) * (ax1 - ax0)


rowh = 36
yy = y0 + 72
top = yy
for v in range(30, 100, 10):
    P.append(line(X(v), top - 8, X(v), top + 12 * rowh + 3 * 10, "#EEF2F7", 1))
    P.append(text(X(v), top + 12 * rowh + 3 * 10 + 20, str(v), 12, fill=MUTED, anchor="middle"))
for gi, (g, rows) in enumerate(groups):
    P.append(rect(x0 + 20, yy - 4, 60, len(rows) * rowh - 4, PANEL2, PANEL2, 1, 6))
    P.append(ctext(x0 + 50, yy - 4 + (len(rows) * rowh - 4) / 2, list(g), 13, weight=700, fill=MUTED, lh=1.2))
    for name, prev, ours in rows:
        cy = yy + rowh / 2 - 4
        win = ours > prev
        col = ORANGE if win else BLUE
        P.append(text(x0 + 210, cy + 5, name, 14, weight=600, anchor="end"))
        P.append(line(X(prev), cy, X(ours), cy, col, 3, opacity=0.45))
        P.append(circle(X(prev), cy, 7, "#FFFFFF", MUTED, 2))
        P.append(circle(X(ours), cy, 8, col, col))
        d = ours - prev
        lab = f"{ours}（{'+' if d > 0 else ''}{d:.1f}）"
        right = max(prev, ours)
        P.append(text(X(right) + 14, cy + 5, lab, 13, fill=col, weight=700))
        yy += rowh
    yy += 10
P.append(text(ax1, top + 12 * rowh + 3 * 10 + 20, "准确率 / F1 / 相关系数（×100）", 12, fill=MUTED, anchor="end").replace(f'x="{ax1}"', f'x="{X(30) - 14}"'))
P.append(rect(x0 + 20, y0 + h0 - 62, w0 - 40, 46, ORANGE_LL, ORANGE, 1.2, 10))
P.append(text(x0 + 40, y0 + h0 - 33, "GLUE 总分 **72.8**（此前最佳 68.9）；数据集规模从 STS-B 约 5.7k 到 SNLI 约 550k 都有效。", 14))

# ======================= ② ablation =======================
x1, w1 = 980, 560
P.append(rect(x1, y0, w1, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x1 + 24, y0 + 28, 2, fill=BLUE))
P.append(text(x1 + 46, y0 + 34, "消融：哪个部分最重要（Table 5）", 17, weight=700))
abl = [("完整模型", "Transformer + 预训练 + 辅助 LM", 74.7, ORANGE),
       ("去掉辅助 LM", "微调时只用 L_{2}", 75.0, BLUE),
       ("换成 LSTM", "单层 2048 单元，其余相同", 69.1, MUTED),
       ("去掉预训练", "Transformer 直接在任务上训练", 59.9, INK)]
bx, bw_max = x1 + 210, 300
for i, (t, sub, v, col) in enumerate(abl):
    y = y0 + 72 + i * 74
    P.append(text(x1 + 24, y + 22, t, 15, weight=700))
    P.append(text(x1 + 24, y + 44, sub, 12, fill=MUTED))
    w = (v - 50) / 30 * bw_max
    fill = {ORANGE: ORANGE, BLUE: BLUE_L, MUTED: PANEL2, INK: "#FFFFFF"}[col]
    P.append(rect(bx, y + 8, w, 32, fill, col if col != INK else INK, 1.3, 5, dash="5 3" if col == INK else None))
    P.append(text(bx + w + 10, y + 30, f"{v}", 16, weight=800, fill=col if col != MUTED else INK))
P.append(text(bx, y0 + 72 + 4 * 74 + 4, "平均分（横轴从 50 起）", 12, fill=MUTED))
ky = y0 + 404
notes = [("−14.8", "去掉预训练：所有任务都下降，CoLA 从 45.4 跌到 18.9", INK),
         ("−5.6", "换成 LSTM：只有 MRPC 一项更好，说明 Transformer 更利于迁移", MUTED),
         ("+0.3", "去掉辅助 LM 平均略高：小数据集不受益，大数据集受益", BLUE)]
for i, (big, s, col) in enumerate(notes):
    y = ky + i * 76
    P.append(rect(x1 + 20, y, w1 - 40, 64, PANEL if i != 2 else BLUE_LL, LINE if i != 2 else BLUE, 1.2, 10))
    P.append(text(x1 + 40, y + 42, big, 24, weight=800, fill=col if col != MUTED else ORANGE))
    P.append(text(x1 + 130, y + 38, s, 13))

P.append(footer(6))
render("06-key-results", "".join(P))
