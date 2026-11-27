"""04 · 任务输入改写：把结构化输入拼成一条序列，模型结构不变"""
from lib import *

P = []
P.append(header("任务输入改写：结构化输入拼成 token 序列，同一个模型处理所有任务",
                "所有序列都加随机初始化的 起止 token（Start / Extract），句子之间用分隔符 $；只有输入格式随任务变化（Section 3.3，Figure 1）",
                4, "核心机制 ③"))

SEQ_X = 330
TF_X, TF_W = 1090, 150
LIN_X, LIN_W = 1280, 100
OUT_X = 1410


def seq(x, y, items, h=36):
    """items: [(label, kind, width)] kind: s(special) d(delim) t(text)"""
    style = {"s": (ORANGE_L, ORANGE, ORANGE), "d": (PANEL2, MUTED, INK), "t": ("#FFFFFF", LINE, INK),
             "a": (BLUE_LL, BLUE, BLUE)}
    out = []
    for lab, k, w in items:
        f, s, c = style[k]
        out.append(chip(x, y, w, h, lab, fill=f, stroke=s, color=c, size=14, weight=600 if k != "t" else 400))
        x += w + 6
    return "".join(out), x


def lane(y, h, title, data, col):
    s = [rect(60, y, W - 120, h, "#FFFFFF", LINE, 1.2, 12)]
    s.append(rect(60, y, 8, h, col, col, 1, 4))
    s.append(text(88, y + 38, title, 19, weight=700, fill=col))
    s.append(text(88, y + 66, data, 13, fill=MUTED))
    return "".join(s)


def tail(y, out_lab, out_sub=None, from_x=None, join=False):
    s = []
    if from_x is not None:
        s.append(arrow(from_x + 4, y, TF_X - 4, y, INK, 1.5))
    s.append(box(TF_X, y - 20, TF_W, 40, "Transformer", 15, fill=BLUE_L, stroke=BLUE, color=BLUE))
    if not join:
        s.append(arrow(TF_X + TF_W + 2, y, LIN_X - 4, y, INK, 1.5))
        s.append(box(LIN_X, y - 20, LIN_W, 40, "Linear", 15, fill=ORANGE_L, stroke=ORANGE, color=ORANGE))
        if out_lab:
            s.append(arrow(LIN_X + LIN_W + 2, y, OUT_X - 4, y, INK, 1.5))
            s.append(text(OUT_X, y + 6, out_lab, 15, weight=700))
            if out_sub:
                s.append(text(OUT_X, y + 26, out_sub, 12, fill=MUTED))
    return "".join(s)


y = 172
# ---------- classification ----------
h = 118
P.append(lane(y, h, "分类", "SST-2 · CoLA", BLUE))
s_, ex = seq(SEQ_X, y + 30, [("Start", "s", 70), ("Text", "t", 380), ("Extract", "s", 84)])
P.append(s_)
P.append(text(SEQ_X, y + 94, "例（示意）：“a gorgeous, witty movie.” → 正面", 13, fill=MUTED))
P.append(tail(y + 48, "类别", "softmax", from_x=ex))
y += h + 14
# ---------- entailment ----------
h = 118
P.append(lane(y, h, "蕴含 / 推理", "MNLI · SNLI · SciTail · QNLI · RTE", BLUE))
s_, ex = seq(SEQ_X, y + 30, [("Start", "s", 70), ("Premise p", "t", 220), ("$", "d", 44), ("Hypothesis h", "t", 220),
                             ("Extract", "s", 84)])
P.append(s_)
P.append(text(SEQ_X, y + 94, "前提与假设按顺序拼接，中间放分隔符 $ → 蕴含 / 矛盾 / 中立", 13, fill=MUTED))
P.append(tail(y + 48, "3 类", "softmax", from_x=ex))
y += h + 14
# ---------- similarity ----------
h = 176
P.append(lane(y, h, "语义相似度", "MRPC · QQP · STS-B", ORANGE))
r1, r2 = y + 26, y + 84
s1, ex1 = seq(SEQ_X, r1, [("Start", "s", 70), ("Text 1", "t", 200), ("$", "d", 44), ("Text 2", "t", 200),
                          ("Extract", "s", 84)])
s2, ex2 = seq(SEQ_X, r2, [("Start", "s", 70), ("Text 2", "t", 200), ("$", "d", 44), ("Text 1", "t", 200),
                          ("Extract", "s", 84)])
P.append(s1 + s2)
P.append(tail(r1 + 18, None, from_x=ex1, join=True))
P.append(tail(r2 + 18, None, from_x=ex2, join=True))
px, py = TF_X + TF_W + 46, (r1 + r2) / 2 + 18
P.append(poly_arrow([(TF_X + TF_W + 2, r1 + 18), (px, r1 + 18), (px, py - 16)], INK, 1.5))
P.append(poly_arrow([(TF_X + TF_W + 2, r2 + 18), (px, r2 + 18), (px, py + 16)], INK, 1.5))
P.append(plus_node(px, py, 14))
P.append(box(LIN_X + 40, py - 20, LIN_W - 10, 40, "Linear", 15, fill=ORANGE_L, stroke=ORANGE, color=ORANGE))
P.append(arrow(px + 16, py, LIN_X + 36, py, INK, 1.5))
P.append(arrow(LIN_X + LIN_W + 32, py, OUT_X + 24, py, INK, 1.5))
P.append(text(OUT_X + 30, py + 6, "相似度", 15, weight=700))
P.append(text(SEQ_X, y + 148, "两句没有先后之分：两种顺序各过一遍模型，把两个 h_{l}^{m} **逐元素相加**后再接 Linear", 13, fill=ORANGE,
              weight=600))
y += h + 14
# ---------- multiple choice ----------
h = 206
P.append(lane(y, h, "问答 / 常识推理", "RACE · Story Cloze", BLUE))
rows = [("Answer 1", y + 22), ("Answer 2", y + 70), ("Answer N", y + 130)]
ys = []
for lab, ry in rows:
    s_, ex = seq(SEQ_X, ry, [("Start", "s", 70), ("Context z ; Question q", "t", 280), ("$", "d", 44),
                             (lab, "a", 120), ("Extract", "s", 84)])
    P.append(s_)
    P.append(arrow(ex + 4, ry + 18, TF_X - 4, ry + 18, INK, 1.5))
    P.append(box(TF_X, ry - 2, TF_W, 40, "Transformer", 15, fill=BLUE_L, stroke=BLUE, color=BLUE))
    P.append(arrow(TF_X + TF_W + 2, ry + 18, LIN_X - 4, ry + 18, INK, 1.5))
    P.append(box(LIN_X, ry - 2, LIN_W, 40, "Linear", 15, fill=ORANGE_L, stroke=ORANGE, color=ORANGE))
    ys.append(ry + 18)
P.append(ctext(SEQ_X + 250, y + 112, "⋮", 18, fill=MUTED))
P.append(ctext(TF_X + TF_W / 2, y + 112, "⋮", 18, fill=MUTED))
# softmax over N
sx = OUT_X + 6
P.append(rect(sx, ys[0] - 10, 34, ys[-1] - ys[0] + 20, ORANGE_LL, ORANGE, 1.4, 6))
P.append(ctext(sx + 17, (ys[0] + ys[-1]) / 2, ["s", "o", "f", "t", "m", "a", "x"], 11, fill=ORANGE, weight=700, lh=1.05))
for yy in ys:
    P.append(arrow(LIN_X + LIN_W + 2, yy, sx - 2, yy, INK, 1.4))
P.append(text(sx + 46, (ys[0] + ys[-1]) / 2 - 4, ["选项", "分布"], 15, weight=700, lh=1.3))
P.append(text(SEQ_X, y + 192, "每个候选答案单独拼一条 [z; q; $; a_{k}]，各自得到一个分数，再在 N 个选项上做 softmax", 13, fill=MUTED))

P.append(footer(4))
render("04-task-input-transformations", "".join(P))
