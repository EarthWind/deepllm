"""01 · 方法总览：两阶段训练，同一组参数 Θ 从预训练流向微调"""
from lib import *

P = []
P.append(header("方法总览：先在无标注文本上学语言模型，再把整个网络微调到下游任务",
                "两个阶段共用同一个 12 层 Transformer 解码器；第 ② 阶段只新增 W_{y} 和分隔符嵌入（Section 3）",
                1, "方法总览"))


def panel(x, y, w, h, title, sub, col, fill):
    s = [rect(x, y, w, h, fill, col, 1.6, 14)]
    s.append(text(x + 24, y + 38, title, 20, weight=700, fill=col))
    s.append(text(x + 24, y + 64, sub, 14, fill=MUTED))
    return "".join(s)


# ======================= Stage 1 =======================
ax, ay, aw, ah = 60, 178, 640, 500
P.append(panel(ax, ay, aw, ah, "① 无监督预训练", "目标：最大化语料的对数似然 L_{1}（Eq. 1–2）", BLUE, BLUE_LL))
cx = ax + aw / 2
# data
P.append(box(ax + 40, ay + 88, aw - 80, 62, "BooksCorpus：7,000+ 本未出版图书", 17, fill="#FFFFFF", stroke=BLUE,
             sub="含大段连续文本，能学到长距离依赖；BPE 词表（40,000 次合并）", sub_size=13))
P.append(arrow(cx, ay + 152, cx, ay + 172, BLUE, 1.8))
# token stream with window
toks = ["u_{i-k}", "…", "u_{i-2}", "u_{i-1}", "u_{i}"]
tw, tg = 92, 12
tx0 = cx - (len(toks) * tw + (len(toks) - 1) * tg) / 2
for i, t in enumerate(toks):
    hl = i == len(toks) - 1
    P.append(chip(tx0 + i * (tw + tg), ay + 176, tw, 34, t, fill=ORANGE_L if hl else "#FFFFFF",
                  stroke=ORANGE if hl else LINE, color=ORANGE if hl else INK, size=15))
P.append(rect(tx0 - 6, ay + 170, 4 * (tw + tg) - tg + 12, 46, "none", BLUE, 1.4, 8, dash="5 4"))
P.append(text(tx0 - 6, ay + 234, "上下文窗口 k 个 token（训练时序列长 512）", 13, fill=BLUE, weight=600))
P.append(text(tx0 + 4 * (tw + tg) + tw / 2, ay + 234, "要预测的下一个", 13, fill=ORANGE, weight=600, anchor="middle"))
P.append(arrow(cx, ay + 244, cx, ay + 266, BLUE, 1.8))
P.append(box(ax + 120, ay + 268, aw - 240, 62, "12 层 Transformer 解码器  Θ", 18, fill=BLUE_L, stroke=BLUE,
             color=BLUE, sub="Masked Self-Attention：只能看左侧 token", sub_size=13, sub_color=INK))
P.append(arrow(cx, ay + 332, cx, ay + 354, BLUE, 1.8))
P.append(box(ax + 120, ay + 356, aw - 240, 44, "P(u_{i} | u_{i-k}, …, u_{i-1}; Θ)", 17, fill="#FFFFFF", stroke=BLUE,
             weight=600))
P.append(rect(ax + 40, ay + 418, aw - 80, 60, "#FFFFFF", LINE, 1.2, 10))
P.append(ctext(cx, ay + 448, "L_{1}(U) = Σ_{i} log P(u_{i} | u_{i-k}, …, u_{i-1}; Θ)", 19, family=MATH, weight=600))

# ======================= hand-off =======================
mx = ax + aw + 10
P.append(arrow(mx + 10, ay + 299, mx + 170, ay + 299, INK, 3))
P.append(ctext(mx + 90, ay + 270, "全部参数 Θ", 16, weight=700))
P.append(ctext(mx + 90, ay + 330, ["作为微调起点", "（不冻结）"], 14, fill=MUTED))

# ======================= Stage 2 =======================
bx, by, bw, bh = 900, 178, 640, 500
P.append(panel(bx, by, bw, bh, "② 有监督微调", "目标：L_{3} = L_{2} + λ·L_{1}，λ = 0.5（Eq. 3–5）", ORANGE, ORANGE_LL))
cx2 = bx + bw / 2
P.append(box(bx + 40, by + 88, bw - 80, 62, "有标注数据 C：(x^{1}, …, x^{m}, y)", 17, fill="#FFFFFF", stroke=ORANGE,
             sub="推理 / 问答 / 相似度 / 分类，共 12 个数据集", sub_size=13))
P.append(arrow(cx2, by + 152, cx2, by + 172, ORANGE, 1.8))
seq = [("Start", PANEL2, LINE, INK), ("Premise", "#FFFFFF", LINE, INK), ("Delim", PANEL2, LINE, INK),
       ("Hypothesis", "#FFFFFF", LINE, INK), ("Extract", ORANGE_L, ORANGE, ORANGE)]
ws = [70, 120, 70, 130, 84]
sx = cx2 - (sum(ws) + 4 * 8) / 2
for (t, f, s_, c), w in zip(seq, ws):
    P.append(chip(sx, by + 176, w, 34, t, fill=f, stroke=s_, color=c, size=14))
    sx += w + 8
P.append(text(cx2, by + 234, "任务输入改写成一条 token 序列（见第 04 页）", 13, fill=ORANGE, weight=600, anchor="middle"))
P.append(arrow(cx2, by + 244, cx2, by + 266, ORANGE, 1.8))
P.append(box(bx + 120, by + 268, bw - 240, 62, "同一个 Transformer  Θ（继承）", 18, fill=BLUE_L, stroke=BLUE,
             color=BLUE, sub="取最后一个 token（Extract）的顶层激活 h_{l}^{m}", sub_size=13, sub_color=INK))
P.append(arrow(cx2, by + 332, cx2, by + 354, ORANGE, 1.8))
P.append(box(bx + 120, by + 356, bw - 240, 44, "softmax(h_{l}^{m} W_{y}) → ŷ", 17, fill=ORANGE_L, stroke=ORANGE,
             color=ORANGE, weight=700))
P.append(rect(bx + 40, by + 418, bw - 80, 60, "#FFFFFF", LINE, 1.2, 10))
P.append(ctext(cx2, by + 448, "L_{2}(C) = Σ_{(x,y)} log P(y | x^{1}, …, x^{m})", 19, family=MATH, weight=600))

# ======================= bottom: before vs GPT =======================
cy = 700
cards = [
    ("此前的迁移方式", ["预训练词向量只迁移词级信息；", "ELMo 等把预训练表示当作特征，", "仍要为每个任务设计专用模型"], MUTED, PANEL2, LINE),
    ("GPT 的做法", ["整个网络一起迁移，所有层都参与微调；", "结构化输入改写成序列，", "模型结构几乎不改"], BLUE, BLUE_LL, BLUE),
    ("微调新增的参数", ["只有输出层 W_{y}", "和分隔符 / 起止 token 的嵌入；", "通常 3 个 epoch 即可收敛"], ORANGE, ORANGE_LL, ORANGE),
]
cw = (W - 120 - 2 * 20) / 3
for i, (t, lines, col, fill, stroke) in enumerate(cards):
    x = 60 + i * (cw + 20)
    P.append(rect(x, cy, cw, 150, fill, stroke, 1.3, 12))
    P.append(text(x + 22, cy + 34, t, 17, weight=700, fill=col if col != MUTED else INK))
    P.append(text(x + 22, cy + 68, lines, 15, lh=1.6))

P.append(footer(1))
render("01-method-overview", "".join(P))
