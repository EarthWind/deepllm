"""00 · 封面：一句话贡献 + RNN vs Self-Attention 视觉锚点"""
from lib import *

P = []
# ---------------- left: title block ----------------
P.append(text(60, 92, "论文方法图解 · 共 8 张", 17, fill=BLUE, weight=700))
P.append(text(60, 160, "Attention Is All You Need", 50, weight=800))
P.append(text(60, 208, "Transformer：完全基于注意力的序列转换模型", 26, weight=600, fill=INK))
P.append(text(60, 242, "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin · NeurIPS 2017", 14, fill=MUTED))

# one-liner
P.append(rect(60, 272, 640, 150, BLUE_LL, BLUE, 1.4, 12))
P.append(text(84, 306, "一句话", 15, weight=700, fill=BLUE))
P.append(text(84, 342, ["去掉 RNN 与 CNN，只用注意力完成 token 间的信息交互：", "**训练可整段并行，任意两位置一步直连**，并成为", "BERT、GPT、T5 与几乎所有 LLM 的骨架。"],
              17, lh=1.55))

# three building blocks
P.append(text(60, 460, "三块积木", 15, weight=700, fill=MUTED))
blocks = [("Attention", "跨 token 路由信息", BLUE, BLUE_L),
          ("FFN", "逐 token 加工特征", ORANGE, ORANGE_L),
          ("残差 + LayerNorm", "让深层堆叠可训练", INK, PANEL2)]
for i, (a, b_, col, fill) in enumerate(blocks):
    x = 60 + i * 216
    P.append(rect(x, 474, 204, 72, fill, col, 1.5, 10))
    P.append(ctext(x + 102, 498, a, 16, fill=col, weight=700))
    P.append(ctext(x + 102, 526, b_, 13, fill=INK))

# contents
P.append(text(60, 584, "目录", 15, weight=700, fill=MUTED))
toc = ["01 方法总览：Encoder-Decoder 数据流", "02 Scaled Dot-Product Attention",
       "03 Multi-Head 与三种注意力", "04 Positional Encoding",
       "05 残差 + LayerNorm + FFN", "06 训练配方", "07 关键结果"]
for i, t in enumerate(toc):
    col, row = i % 2, i // 2
    P.append(text(60 + col * 330, 610 + row * 25, t, 14, fill=INK))

# ---------------- right: RNN vs Self-Attention ----------------
TOK = ["小猫", "没", "过", "马路", "因为", "它", "累了"]
rx, rw = 760, 780
tx0, tstep = rx + 50, 104


def tok_row(y, hl=None):
    s = []
    for i, t in enumerate(TOK):
        x = tx0 + i * tstep
        if i in (hl or {}):
            fill, stroke, color = hl[i]
        else:
            fill, stroke, color = PANEL2, LINE, INK
        s.append(chip(x, y, 80, 34, t, fill=fill, stroke=stroke, color=color, size=15))
    return "".join(s)


# RNN panel
py = 60
P.append(rect(rx, py, rw, 256, "#FFFFFF", LINE, 1.2, 12))
P.append(text(rx + 24, py + 36, "RNN / LSTM：信息沿时间步逐个传递", 18, weight=700))
P.append(text(rx + rw - 24, py + 36, "顺序操作 O(n) · 路径长度 O(n)", 14, fill=MUTED, anchor="end", weight=600))
hy = py + 130
for i in range(len(TOK)):
    cx = tx0 + i * tstep + 40
    on = 0 <= i <= 5
    P.append(circle(cx, hy, 22, BLUE_L if on else "#FFFFFF", BLUE if on else MUTED, 1.6))
    P.append(ctext(cx, hy, f"h_{{{i + 1}}}", 13, fill=INK, family=FONT))
    P.append(arrow(cx, hy + 54, cx, hy + 25, MUTED, 1.3))
    if i < len(TOK) - 1:
        P.append(arrow(cx + 23, hy, cx + tstep - 25, hy, BLUE if i < 5 else MUTED, 2.2 if i < 5 else 1.4))
P.append(tok_row(hy + 56, {0: (BLUE_L, BLUE, BLUE), 5: (BLUE_L, BLUE, BLUE)}))
P.append(path(f"M{tx0 + 40},{hy - 34} Q{tx0 + 40 + 2.5 * tstep},{hy - 70} {tx0 + 5 * tstep + 40},{hy - 34}", BLUE, 1.6,
              arrow=True, dash="5 4"))
P.append(text(tx0 + 40 + 2.5 * tstep, hy - 62, "“它”要找回“小猫”：信息需穿过 5 步，且必须串行计算", 14, fill=BLUE, anchor="middle", weight=600))

# Self-attention panel
sy = 336
P.append(rect(rx, sy, rw, 344, ORANGE_LL, ORANGE, 1.6, 12))
P.append(text(rx + 24, sy + 36, "Self-Attention：任意两个位置一步直连", 18, weight=700, fill=ORANGE))
P.append(text(rx + rw - 24, sy + 36, "顺序操作 O(1) · 路径长度 O(1)", 14, fill=ORANGE, anchor="end", weight=700))
ty = sy + 248
src = 5
weights = {0: 0.62, 1: 0.03, 2: 0.04, 3: 0.08, 4: 0.06, 6: 0.17}
# faint all-pairs background
for i in range(len(TOK)):
    for j in range(i + 1, len(TOK)):
        x1, x2 = tx0 + i * tstep + 40, tx0 + j * tstep + 40
        h = 22 + (j - i) * 18
        P.append(path(f"M{x1},{ty - 2} Q{(x1 + x2) / 2},{ty - 2 - h} {x2},{ty - 2}", "#FDBA74", 0.8, opacity=0.6))
for j, w in weights.items():
    x1, x2 = tx0 + src * tstep + 40, tx0 + j * tstep + 40
    h = 70 + abs(j - src) * 46
    P.append(path(f"M{x1},{ty - 2} Q{(x1 + x2) / 2},{ty - 2 - h} {x2},{ty - 2}", ORANGE, 1 + 9 * w, opacity=0.9))
    P.append(ctext(x2, ty + 52, f"{w:.2f}", 13 if w < 0.5 else 16, fill=ORANGE, weight=700))
P.append(tok_row(ty, {5: (ORANGE, ORANGE, "#FFFFFF"), 0: (ORANGE_L, ORANGE, ORANGE)}))
P.append(text(rx + 24, sy + 330, "“它”的注意力权重（示意）：一步把最多的注意力放到“小猫”上；所有位置同时计算", 14, fill=INK))

# ---------------- bottom stat cards ----------------
cy = 712
stats = [("28.4", "BLEU · WMT14 英→德", "比此前最佳（含集成）高 2.0 以上"),
         ("41.8", "BLEU · WMT14 英→法", "单模型 SOTA（arXiv v7 表值）"),
         ("3.3×10^{18}", "FLOPs · base 模型训练算力", "8×P100 约 12 小时，已超全部旧模型"),
         ("65M / 213M", "参数 · base / big", "N=6，d_{model}=512 / 1024")]
cw = (W - 120 - 3 * 20) / 4
for i, (big, lab, sub) in enumerate(stats):
    x = 60 + i * (cw + 20)
    P.append(rect(x, cy, cw, 128, "#FFFFFF", LINE, 1.2, 12))
    P.append(text(x + 22, cy + 52, big, 34, weight=800, fill=ORANGE if i < 2 else BLUE))
    P.append(text(x + 22, cy + 84, lab, 15, weight=600))
    P.append(text(x + 22, cy + 110, sub, 13, fill=MUTED))

P.append(text(60, H - 22, "Vaswani et al., Attention Is All You Need, NeurIPS 2017 · arXiv:1706.03762v7", 13, fill=FAINT))
P.append(text(W - 60, H - 22, f"00 / {TOTAL_PAGES:02d}", 15, fill=MUTED, weight=600, anchor="end"))
render("00-cover", "".join(P))
