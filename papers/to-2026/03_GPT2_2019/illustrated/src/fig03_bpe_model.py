"""03 · 字节级 BPE 与模型改动"""
from lib import *

P = []
P.append(header("输入与模型：字节级 BPE 让任何字符串都能建模；结构沿用 GPT-1，改了几处",
                "Section 2.2–2.3：输入不做小写化、不出现 <UNK>；模型改用 Pre-LayerNorm，最大 48 层、1.5B 参数",
                3, "输入表示与模型"))
P.append(legend(W - 60, 62, ("lm", "base")))
y0, h0 = 178, 660

# ======================= ① byte-level BPE =======================
x0, w0 = 60, 760
P.append(rect(x0, y0, w0, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x0 + 24, y0 + 28, 1, fill=INK))
P.append(text(x0 + 46, y0 + 34, "四种输入表示：各有什么问题", 17, weight=700))
rows = [("词级", "大数据上效果最好", "要分词、小写化，生词只能变成 <UNK>"),
        ("字节级", "基础词表只有 256，任何字符串都能编码", "在 1B Word 等大数据上打不过词级"),
        ("Unicode 码点 BPE", "高频词整词表示", "基础词表就要 13 万+；常用 BPE 词表才 3.2–6.4 万"),
        ("字节级 BPE（GPT-2）", "高频序列整词，罕见序列退回字节", "兼顾词级的效果和字节级的通用性")]
for i, (n, pro, con) in enumerate(rows):
    y = y0 + 56 + i * 62
    last = i == 3
    P.append(rect(x0 + 24, y, w0 - 48, 54, "#FFFFFF" if last else PANEL, INK if last else LINE, 2 if last else 1, 8))
    P.append(text(x0 + 42, y + 33, n, 15, weight=700))
    P.append(text(x0 + 250, y + 23, "+ " + pro, 13))
    P.append(text(x0 + 250, y + 43, ("= " if last else "− ") + con, 13, fill=INK if last else MUTED,
                  weight=600 if last else 400))

yb = y0 + 318
P.append(line(x0 + 24, yb, x0 + w0 - 24, yb, LINE, 1))
P.append(text(x0 + 24, yb + 34, "直接在字节上跑 BPE 的坑，以及修正", 16, weight=700))
P.append(text(x0 + 24, yb + 76, "不加限制", 14, weight=700, fill=MUTED))
cx = x0 + 130
for t in ["dog", "dog.", "dog!", "dog?"]:
    P.append(chip(cx, yb + 54, 74, 34, t, PANEL2, MUTED, INK, 15, 600))
    cx += 84
P.append(text(cx + 8, yb + 68, ["同一个词占了 4 个词表位，", "浪费有限的词表和模型容量"], 13, fill=MUTED, lh=1.4))
P.append(text(x0 + 24, yb + 142, "GPT-2", 14, weight=700))
cx = x0 + 130
P.append(chip(cx, yb + 120, 74, 34, "dog", "#FFFFFF", INK, INK, 15, 700))
cx += 84
for t in [".", "!", "?"]:
    P.append(chip(cx, yb + 120, 40, 34, t, "#FFFFFF", INK, INK, 15, 700))
    cx += 50
P.append(text(cx + 8, yb + 134, ["“dog”只占 1 个位置，", "标点各自单独成 token"], 13, lh=1.4))
P.append(rect(x0 + 24, yb + 176, w0 - 48, 76, PANEL, LINE, 1, 8))
P.append(text(x0 + 42, yb + 204, ["规则：**不允许跨字符类别合并**（字母、数字、标点互不合并）；",
                                  "空格例外：空格可以和后面的单词合并成一个 token（如 ␣dog），压缩率明显更好。"], 13, lh=1.6))
P.append(text(x0 + 24, yb + 290, ["结果：词表 **50,257**，能给**任意 Unicode 字符串**算概率，",
                                  "所以可以直接在各种语言建模数据集上评测，不受它们各自分词、预处理的限制。"], 14, lh=1.6))

# ======================= ② model =======================
x1, w1 = 840, 700
P.append(rect(x1, y0, w1, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x1 + 24, y0 + 28, 2, fill=INK))
P.append(text(x1 + 46, y0 + 34, "模型：GPT-1 的 Transformer 解码器 + 3 处改动", 17, weight=700))
P.append(rect(x1 + w1 - 200, y0 + 18, 16, 16, LM, LM, 1, 3))
P.append(text(x1 + w1 - 176, y0 + 31, "深色 = 改动处", 13, fill=LM, weight=600))

bx, bw = x1 + 50, 220
cxm = bx + bw / 2


def blk(y, label, changed=False, h=32):
    if changed:
        return box(bx, y, bw, h, label, 14, fill=LM, stroke=LM, color="#FFFFFF", weight=700)
    return box(bx, y, bw, h, label, 14, fill=LM_LL, stroke=LM, color=LM, weight=600)


P.append(chip(bx, y0 + 60, bw, 32, "softmax → 下一个 token", "#FFFFFF", LINE, INK, 13, 600))
P.append(arrow(cxm, y0 + 110, cxm, y0 + 94, INK, 1.4))
P.append(blk(y0 + 110, "LayerNorm（新增）", True))
P.append(arrow(cxm, y0 + 160, cxm, y0 + 144, INK, 1.4))
P.append(rect(bx - 30, y0 + 160, bw + 60, 236, PANEL, LINE, 1.2, 12, dash="6 5"))
P.append(text(bx + bw + 38, y0 + 180, "×N", 15, weight=700, fill=MUTED))
P.append(plus_node(cxm, y0 + 180, 11))
P.append(blk(y0 + 200, "Feed Forward"))
P.append(blk(y0 + 240, "LayerNorm（前移）", True))
P.append(plus_node(cxm, y0 + 292, 11))
P.append(blk(y0 + 312, "Masked Self-Attention"))
P.append(blk(y0 + 352, "LayerNorm（前移）", True))
P.append(blk(y0 + 410, "Token 嵌入 + 位置嵌入"))
for ya, yb_ in [(y0 + 410, y0 + 386), (y0 + 352, y0 + 346), (y0 + 312, y0 + 305), (y0 + 280, y0 + 274),
                (y0 + 240, y0 + 234), (y0 + 200, y0 + 193)]:
    P.append(arrow(cxm, ya, cxm, yb_, INK, 1.4))
P.append(line(cxm, y0 + 169, cxm, y0 + 160, INK, 1.4))
P.append(poly_arrow([(cxm, y0 + 400), (bx - 16, y0 + 400), (bx - 16, y0 + 292), (cxm - 13, y0 + 292)], MUTED, 1.4, r=6))
P.append(poly_arrow([(cxm, y0 + 278), (bx - 16, y0 + 278), (bx - 16, y0 + 180), (cxm - 13, y0 + 180)], MUTED, 1.4, r=6))
P.append(text(bx - 22, y0 + 240, "残差", 12, fill=MUTED, anchor="end"))

tx = x1 + 360
changes = [("1", "LayerNorm 移到每个子块的输入处", ["GPT-1 放在子块之后；类似", "pre-activation ResNet"]),
           ("2", "最后一个 block 后再加一层 LayerNorm", ["Pre-LN 的输出没有归一化，", "所以在顶端补一层"]),
           ("3", "残差层权重初始化乘 1/√N", ["N = 残差层数；层数越多，", "残差路径上累加得越多"])]
for i, (n, t, sub) in enumerate(changes):
    y = y0 + 64 + i * 106
    P.append(step_badge(tx + 12, y + 12, n, fill=INK, r=12))
    P.append(text(tx + 34, y + 18, t, 14, weight=700))
    P.append(text(tx + 34, y + 44, sub, 13, fill=MUTED, lh=1.45))
P.append(text(tx, y0 + 394, ["另外：词表 50,257，上下文 512 → 1024，", "batch 512"], 13, fill=INK, lh=1.45))

ty = y0 + 470
P.append(line(x1 + 24, ty - 12, x1 + w1 - 24, ty - 12, LINE, 1))
P.append(text(x1 + 24, ty + 12, "四种规模（Table 2），学习率各自按 WebText 留出集调", 14, weight=700))
hdr = ["参数", "层数", "d_model", "相当于"]
sizes = [("117M", 12, 768, "GPT-1"), ("345M", 24, 1024, "BERT-Large 的规模"), ("762M", 36, 1280, ""),
         ("1542M", 48, 1600, "GPT-2")]
cols = [x1 + 40, x1 + 150, x1 + 230, x1 + 330]
for j, hname in enumerate(hdr):
    P.append(text(cols[j], ty + 44, hname, 13, weight=700, fill=MUTED))
for i, (p_, l, d, note) in enumerate(sizes):
    y = ty + 70 + i * 30
    big = i == 3
    P.append(text(cols[0], y, p_, 14, weight=800 if big else 600, fill=LM if big else INK))
    P.append(text(cols[1], y, str(l), 14))
    P.append(text(cols[2], y, str(d), 14))
    P.append(text(cols[3], y, note, 13, fill=MUTED if not big else LM, weight=700 if big else 400))
    w = int(p_[:-1]) / 1542 * 180
    P.append(rect(x1 + 480, y - 14, w, 16, LM_L, LM, 2.2 if big else 1, 3))

P.append(footer(3))
render("03-bpe-and-model", "".join(P))
