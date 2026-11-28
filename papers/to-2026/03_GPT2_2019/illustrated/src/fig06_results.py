"""06 · 关键结果：语言建模 8 个数据集赢 7 个；下游任务强在阅读理解，其余离实用还远"""
from lib import *

P = []
P.append(header("关键结果：语言建模几乎全胜；下游任务“会做了”，但还做不好",
                "左：Table 3，1.5B 模型 zero-shot 对比此前 SOTA；右：Section 3.5–3.8 与 Discussion。GPT-2 一律为蓝色，赢 / 输用 ✓ / ✗ 表示",
                6, "实验结果"))
P.append(legend(W - 60, 62, ("lm", "base")))
y0, h0 = 178, 660

# ======================= ① Table 3 =======================
x0, w0 = 60, 900
P.append(rect(x0, y0, w0, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x0 + 24, y0 + 28, 1, fill=INK))
P.append(text(x0 + 46, y0 + 34, "语言建模：8 个数据集中 7 个刷新 SOTA（不做任何训练或微调）", 17, weight=700))
rows = [("LAMBADA", "困惑度 ↓", 99.8, 8.63, False), ("LAMBADA", "准确率 ↑", 59.23, 63.24, True),
        ("CBT 普通名词", "准确率 ↑", 85.7, 93.30, True), ("CBT 命名实体", "准确率 ↑", 82.3, 89.05, True),
        ("WikiText-2", "困惑度 ↓", 39.14, 18.34, False), ("PTB", "困惑度 ↓", 46.54, 35.76, False),
        ("enwik8", "BPB ↓", 0.99, 0.93, False), ("text8", "BPC ↓", 1.08, 0.98, False),
        ("WikiText-103", "困惑度 ↓", 18.3, 17.48, False), ("1BW", "困惑度 ↓", 21.8, 42.16, False)]
cols = [x0 + 34, x0 + 190, x0 + 300, x0 + 380, x0 + 470]
hy = y0 + 70
for j, hname in enumerate(["数据集", "指标", "此前 SOTA", "GPT-2", ""]):
    P.append(text(cols[j], hy, hname, 13, weight=700, fill=MUTED if j != 3 else LM))
P.append(text(x0 + 700, hy, "相对此前 SOTA 的改进", 13, weight=700, fill=MUTED, anchor="middle"))
zero = x0 + 700
P.append(line(zero, hy + 14, zero, hy + 14 + len(rows) * 44, INK, 1.2))
for i, (n, m, sota, ours, higher) in enumerate(rows):
    y = hy + 22 + i * 44
    win = ours > sota if higher else ours < sota
    rel = (ours - sota) / sota if higher else (sota - ours) / sota
    if i % 2 == 0:
        P.append(rect(x0 + 20, y - 4, w0 - 40, 40, PANEL, PANEL, 1, 6))
    P.append(text(cols[0], y + 22, n, 14, weight=600))
    P.append(text(cols[1], y + 22, m, 13, fill=MUTED))
    P.append(text(cols[2], y + 22, f"{sota:g}", 14, fill=MUTED))
    P.append(text(cols[3], y + 22, f"{ours:g}", 14, weight=800, fill=LM))
    P.append(text(cols[4], y + 22, "✓" if win else "✗", 16, weight=800, fill=INK if win else FAINT))
    w = abs(rel) * 150
    if win:
        P.append(rect(zero, y + 8, w, 20, LM_L, LM, 1.2, 3))
        P.append(text(zero + w + 8, y + 23, f"+{rel * 100:.1f}%", 12, fill=LM, weight=700))
    else:
        P.append(rect(zero - w, y + 8, w, 20, LM_LL, LM, 1.2, 3))
        P.append(text(zero - w - 8, y + 23, f"−{abs(rel) * 100:.1f}%", 12, fill=LM, weight=700, anchor="end"))
fy = hy + 22 + len(rows) * 44 + 8
P.append(rect(x0 + 20, fy, w0 - 40, 70, PANEL, LINE, 1, 8))
P.append(text(x0 + 38, fy + 28, ["**唯一输的是 1BW**：它是最大的数据集，而且按句子打乱，长距离结构全被破坏。",
                                  "小数据集（PTB、WikiText-2）和考长距离依赖的（LAMBADA、CBT）提升最大。"], 13, lh=1.6))

# ======================= ② downstream =======================
x1, w1 = 980, 560
P.append(rect(x1, y0, w1, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x1 + 24, y0 + 28, 2, fill=INK))
P.append(text(x1 + 46, y0 + 34, "下游任务：离参考系统还有多远", 17, weight=700))
P.append(text(x1 + 24, y0 + 62, "灰色整条 = 参考系统的分数，蓝色 = GPT-2 达到的部分", 12, fill=MUTED))
tasks = [("阅读理解 · CoQA", 55, 89, "55 F1", "人类 89 F1", "4 个 baseline 中追平 / 超过 3 个"),
         ("摘要 · CNN/DM", 21.40, 32.75, "21.40", "SOTA 32.75", "只比随机抽 3 句（20.98）略好"),
         ("翻译 · WMT-14 法→英", 11.5, 33.5, "11.5 BLEU", "无监督 SOTA 33.5", "英→法 只有 5 BLEU"),
         ("问答 · Natural Questions", 4.1, 30, "4.1%", "检索式系统 30–50%", "参考取下限 30%")]
for i, (n, v, ref, vt, rt, note) in enumerate(tasks):
    y = y0 + 92 + i * 110
    P.append(text(x1 + 24, y + 16, n, 14, weight=700))
    P.append(text(x1 + w1 - 24, y + 16, f"{v / ref * 100:.0f}%", 14, weight=800, fill=LM, anchor="end"))
    bw = w1 - 48
    P.append(rect(x1 + 24, y + 28, bw, 22, BASE_L, FAINT, 1, 4))
    P.append(rect(x1 + 24, y + 28, bw * v / ref, 22, LM, LM, 1, 4))
    P.append(text(x1 + 24, y + 70, vt, 12, fill=LM, weight=700))
    P.append(text(x1 + w1 - 24, y + 70, rt, 12, fill=MUTED, anchor="end"))
    P.append(text(x1 + 24, y + 90, note, 12, fill=MUTED))
qy = y0 + 530
P.append(rect(x1 + 24, qy, w1 - 48, 112, PANEL, LINE, 1, 8))
P.append(text(x1 + 42, qy + 26, ["论文自己的判断（Discussion）：", "阅读理解已能和监督 baseline 竞争；摘要等任务",
                                  "只是“在做这件事”，量化指标仍很初级——",
                                  "**zero-shot 离实际可用还很远**，但规模一大就会变好。"], 13, lh=1.5))

P.append(footer(6))
render("06-key-results", "".join(P))
