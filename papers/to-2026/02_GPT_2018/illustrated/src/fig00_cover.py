"""00 · 封面：一句话贡献 + “一个模型、两种头”的视觉锚点"""
from lib import *

P = []
# ---------------- left: title block ----------------
P.append(text(60, 92, "论文方法图解 · 共 7 张", 17, fill=BLUE, weight=700))
P.append(text(60, 156, ["Improving Language Understanding", "by Generative Pre-Training"], 35, weight=800, lh=1.25))
P.append(text(60, 252, "GPT-1：生成式预训练 + 判别式微调", 26, weight=600))
P.append(text(60, 284, "Radford, Narasimhan, Salimans, Sutskever · OpenAI 2018", 14, fill=MUTED))

P.append(rect(60, 310, 640, 150, BLUE_LL, BLUE, 1.4, 12))
P.append(text(84, 344, "一句话", 15, weight=700, fill=BLUE))
P.append(text(84, 380, ["先在大量无标注长文本上训练一个 Transformer 解码器语言模型，",
                        "再把**整个模型**微调到下游任务，只加一层线性输出：",
                        "**一个任务无关的模型，在 12 个数据集中的 9 个上刷新 SOTA。**"], 17, lh=1.55))

P.append(text(60, 498, "两个阶段", 15, weight=700, fill=MUTED))
stages = [("① 无监督预训练", "BooksCorpus 上做下一个 token 预测", BLUE, BLUE_L),
          ("② 有监督微调", "加 Linear 头 + 辅助 LM 损失", ORANGE, ORANGE_L)]
for i, (a, b_, col, fill) in enumerate(stages):
    x = 60 + i * 324
    P.append(rect(x, 512, 312, 72, fill, col, 1.5, 10))
    P.append(ctext(x + 156, 536, a, 17, fill=col, weight=700))
    P.append(ctext(x + 156, 564, b_, 13, fill=INK))
P.append(arrow(373, 548, 383, 548, INK, 2))

P.append(text(60, 622, "目录", 15, weight=700, fill=MUTED))
toc = ["01 方法总览：两阶段训练", "02 无监督预训练：因果语言模型",
       "03 有监督微调：线性头 + 辅助 LM", "04 任务输入改写：结构化输入变序列",
       "05 为什么有效：迁移层数与 zero-shot", "06 关键结果与消融"]
for i, t in enumerate(toc):
    col, row = i % 2, i // 2
    P.append(text(60 + col * 330, 648 + row * 25, t, 14))

# ---------------- right: one model, two heads ----------------
rx, ry, rw, rh = 760, 60, 780, 620
P.append(rect(rx, ry, rw, rh, "#FFFFFF", LINE, 1.2, 12))
P.append(text(rx + 24, ry + 38, "同一个 Transformer，先学“续写”，再学“判断”", 18, weight=700))
P.append(text(rx + 24, ry + 64, "预训练得到的全部参数都被下游任务继承；微调只新增输出层 W_{y} 与分隔符嵌入", 14, fill=MUTED))

# shared trunk
tx, tw = rx + 250, 280
P.append(rect(tx - 16, ry + 234, tw + 32, 256, PANEL, LINE, 1.2, 14, dash="6 5"))
P.append(text(tx + tw + 26, ry + 252, "×12", 16, weight=700, fill=MUTED))
blocks = [("Masked Multi-Head Self-Attention", BLUE_L, BLUE), ("Add & LayerNorm", PANEL2, LINE),
          ("Feed Forward（GELU）", ORANGE_L, ORANGE), ("Add & LayerNorm", PANEL2, LINE)]
by = ry + 430
for i, (lab, f, s) in enumerate(blocks):
    y = by - i * 60
    P.append(box(tx, y, tw, 44, lab, 15, fill=f, stroke=s, weight=600))
    if i:
        P.append(arrow(tx + tw / 2, y + 60, tx + tw / 2, y + 46, MUTED, 1.4))
P.append(box(tx, ry + 520, tw, 44, "Token 嵌入 + 可学习位置嵌入", 15, fill="#FFFFFF", stroke=INK))
P.append(arrow(tx + tw / 2, ry + 520, tx + tw / 2, ry + 476, INK, 1.6))
toks = ["The", "cat", "sat", "on", "the"]
for i, t in enumerate(toks):
    P.append(chip(tx - 10 + i * 62, ry + 580, 54, 28, t, size=13))
P.append(arrow(tx + tw / 2, ry + 578, tx + tw / 2, ry + 566, INK, 1.4))

# two heads
P.append(box(rx + 60, ry + 100, 240, 64, "Text Prediction", 16, fill=BLUE_L, stroke=BLUE, color=BLUE,
             sub="预训练：预测下一个 token", sub_size=13))
P.append(box(rx + 480, ry + 100, 240, 64, "Task Classifier", 16, fill=ORANGE_L, stroke=ORANGE, color=ORANGE,
             sub="微调：W_{y} → 标签", sub_size=13))
P.append(poly_arrow([(tx + 70, ry + 234), (tx + 70, ry + 200), (rx + 180, ry + 200), (rx + 180, ry + 166)], BLUE, 2))
P.append(poly_arrow([(tx + 210, ry + 234), (tx + 210, ry + 200), (rx + 600, ry + 200), (rx + 600, ry + 166)], ORANGE, 2))
P.append(text(rx + 60, ry + 252, ["阶段 ①", "无标注文本", "L_{1}：语言模型"], 14, fill=BLUE, weight=600, lh=1.5))
P.append(text(rx + 720, ry + 252, ["阶段 ②", "有标注数据", "L_{3} = L_{2} + λ·L_{1}"], 14, fill=ORANGE, weight=600,
              anchor="end", lh=1.5))

# ---------------- bottom stat cards ----------------
cy = 712
stats = [("9 / 12", "数据集刷新 SOTA", "覆盖推理、问答、相似度、分类", BLUE),
         ("86.5", "Story Cloze 准确率", "此前最佳 77.6，+8.9", ORANGE),
         ("59.0", "RACE 准确率", "此前最佳 53.3（9 模型集成），+5.7", ORANGE),
         ("72.8", "GLUE 总分", "此前最佳 68.9", BLUE)]
cw = (W - 120 - 3 * 20) / 4
for i, (big, lab, sub, col) in enumerate(stats):
    x = 60 + i * (cw + 20)
    P.append(rect(x, cy, cw, 128, "#FFFFFF", LINE, 1.2, 12))
    P.append(text(x + 22, cy + 52, big, 34, weight=800, fill=col))
    P.append(text(x + 22, cy + 84, lab, 15, weight=600))
    P.append(text(x + 22, cy + 110, sub, 13, fill=MUTED))

P.append(text(60, H - 22, SOURCE, 13, fill=FAINT))
P.append(text(W - 60, H - 22, f"00 / {TOTAL_PAGES:02d}", 15, fill=MUTED, weight=600, anchor="end"))
render("00-cover", "".join(P))
