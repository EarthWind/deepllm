"""01 · 方法总览：Encoder-Decoder 数据流"""
from lib import *

P = []
P.append(header("方法总览：从源句到下一个 token 的完整数据流",
                "Encoder 把源句编码成上下文表示 Memory；Decoder 一边看已生成的前缀，一边通过 Cross-Attention 查询 Memory",
                1, "Transformer 架构"))


def stack_shadow(x, y, w, h, n=2, gap=7):
    s = []
    for i in range(n, 0, -1):
        s.append(rect(x + i * gap, y - i * gap, w, h, "#FFFFFF", LINE, 1.2, 14))
    return "".join(s)


def sublayer_row(x0, y, items, hgt):
    """items: list of (kind, label, sub, width, color, fill). Returns svg and list of (x,w)."""
    s, pos, x = [], [], x0
    for kind, label, sub, w, col, fill in items:
        if kind == "norm":
            s.append(rect(x, y, w, hgt, PANEL2, MUTED, 1.2, 8))
            s.append(text(x + w / 2 + 5, y + hgt / 2, "Add & Norm", 13, fill=INK, anchor="middle", rotate=-90, weight=600))
        else:
            s.append(box(x, y, w, hgt, label, 16, fill=fill, stroke=col, sw=1.8, color=col if col != INK else INK,
                         sub=sub, sub_size=13))
        pos.append((x, w))
        x += w + 26
    for i in range(len(pos) - 1):
        a = pos[i][0] + pos[i][1]
        s.append(arrow(a + 2, y + hgt / 2, pos[i + 1][0] - 3, y + hgt / 2, INK, 1.6))
    return "".join(s), pos


def residual(x_from, x_to, y_top, y_row, color=MUTED):
    return poly_arrow([(x_from, y_row), (x_from, y_top), (x_to, y_top), (x_to, y_row - 2)], stroke=color, sw=1.3, r=8)


def tokens(x, y, toks, color=INK, fill=PANEL2, stroke=LINE):
    s = []
    for i, t in enumerate(toks):
        s.append(chip(x, y + i * 40, 110, 32, t, fill=fill, stroke=stroke, color=color, size=15))
    return "".join(s)


# ======================= ENCODER ROW =======================
ey = 200
P.append(tag(60, ey - 16, "Encoder", fill=BLUE))
P.append(text(60, ey + 32, "源句 (n 个 token)", 14, fill=MUTED))
P.append(tokens(60, ey + 46, ["I", "love", "NLP"]))
P.append(arrow(178, ey + 106, 214, ey + 106))
P.append(box(218, ey + 46, 170, 122, ["Input", "Embedding"], 16, fill="#FFFFFF", stroke=INK,
             sub=["× √d_{model}", "⊕ 位置编码 PE"], sub_size=13))
P.append(arrow(390, ey + 106, 440, ey + 106))
P.append(text(400, ey + 96, "X", 14, fill=MUTED, family=MATH, italic=True))

gx, gy, gw, gh = 448, ey + 4, 610, 200
P.append(stack_shadow(gx, gy, gw, gh))
P.append(rect(gx, gy, gw, gh, BLUE_LL, BLUE, 1.8, 14))
P.append(text(gx + 16, gy + 26, "Encoder Layer", 16, weight=700, fill=BLUE))
P.append(text(gx + gw - 16, gy + 26, "× N = 6（参数不共享）", 15, weight=700, fill=BLUE, anchor="end"))
row_y, row_h = gy + 70, 108
items = [("attn", ["Multi-Head", "Self-Attention"], "Q=K=V=X", 180, BLUE, "#FFFFFF"),
         ("norm", None, None, 40, INK, None),
         ("ffn", ["Feed", "Forward"], "512→2048→512", 150, INK, "#FFFFFF"),
         ("norm", None, None, 40, INK, None)]
s, pos = sublayer_row(gx + 40, row_y, items, row_h)
P.append(s)
P.append(residual(gx + 24, pos[1][0] + 20, row_y - 18, row_y))
P.append(residual(pos[1][0] + pos[1][1] + 13, pos[3][0] + 20, row_y - 18, row_y))
P.append(arrow(gx + 2, row_y + row_h / 2, gx + 38, row_y + row_h / 2))
P.append(text(gx + 20, row_y - 24, "残差", 12, fill=MUTED, anchor="middle"))
P.append(text(gx + 16, gy + gh + 26, "自注意力：每个源 token 一步看到整句", 14, fill=BLUE, weight=600))

# memory
mx, my, mw, mh = 1110, ey + 34, 200, 150
P.append(arrow(gx + gw + 2, ey + 106, mx - 4, ey + 106, BLUE, 2.4))
P.append(rect(mx, my, mw, mh, BLUE, BLUE, 1.5, 12))
P.append(ctext(mx + mw / 2, my + 36, "Memory", 21, fill="#FFFFFF", weight=700))
P.append(ctext(mx + mw / 2, my + 70, "Z ∈ ℝ^{n×512}", 16, fill="#FFFFFF"))
P.append(ctext(mx + mw / 2, my + 108, ["每个源 token 的", "上下文表示"], 13, fill=BLUE_L))

# ======================= DECODER ROW =======================
dy = 500
P.append(tag(60, dy - 16, "Decoder", fill=ORANGE))
P.append(text(60, dy + 32, "已生成前缀（右移一位）", 14, fill=MUTED))
P.append(tokens(60, dy + 46, ["<s>", "我", "爱"]))
P.append(arrow(178, dy + 106, 214, dy + 106))
P.append(box(218, dy + 46, 170, 122, ["Output", "Embedding"], 16, fill="#FFFFFF", stroke=INK,
             sub=["× √d_{model}", "⊕ 位置编码 PE"], sub_size=13))
P.append(arrow(390, dy + 106, 440, dy + 106))

gx2, gy2, gw2, gh2 = 448, dy + 4, 862, 210
P.append(stack_shadow(gx2, gy2, gw2, gh2))
P.append(rect(gx2, gy2, gw2, gh2, ORANGE_LL, ORANGE, 1.8, 14))
P.append(text(gx2 + 16, gy2 + 26, "Decoder Layer", 16, weight=700, fill=ORANGE))
P.append(text(gx2 + gw2 - 16, gy2 + 26, "× N = 6", 15, weight=700, fill=ORANGE, anchor="end"))
row_y2, row_h2 = gy2 + 76, 108
items2 = [("attn", ["Masked", "Self-Attention"], "只看左侧 token", 176, ORANGE, "#FFFFFF"),
          ("norm", None, None, 40, INK, None),
          ("cross", ["Cross-", "Attention"], ["Q ← Decoder", "K, V ← Memory"], 176, BLUE, "#FFFFFF"),
          ("norm", None, None, 40, INK, None),
          ("ffn", ["Feed", "Forward"], "逐位置", 130, INK, "#FFFFFF"),
          ("norm", None, None, 40, INK, None)]
s, pos2 = sublayer_row(gx2 + 40, row_y2, items2, row_h2)
P.append(s)
P.append(arrow(gx2 + 2, row_y2 + row_h2 / 2, gx2 + 38, row_y2 + row_h2 / 2))
P.append(residual(gx2 + 24, pos2[1][0] + 20, row_y2 - 18, row_y2))
P.append(residual(pos2[1][0] + pos2[1][1] + 13, pos2[3][0] + 20, row_y2 - 18, row_y2))
P.append(residual(pos2[3][0] + pos2[3][1] + 13, pos2[5][0] + 20, row_y2 - 18, row_y2))

# memory -> cross attention (into every decoder layer)
cx = pos2[2][0] + pos2[2][1] / 2
P.append(poly_arrow([(mx + mw / 2, my + mh), (mx + mw / 2, 440), (cx, 440), (cx, row_y2 - 2)], stroke=BLUE, sw=2.6, r=14))
P.append(text(cx + 12, 432, "K, V：送入每一个 Decoder 层", 14, fill=BLUE, weight=700))

# output head
ox = 1350
P.append(arrow(gx2 + gw2 + 2, dy + 110, ox - 4, dy + 110))
P.append(box(ox, dy + 30, 200, 70, "Linear", 17, fill="#FFFFFF", stroke=INK, sub="与 Embedding 共享权重", sub_size=12))
P.append(arrow(ox + 100, dy + 102, ox + 100, dy + 124))
P.append(box(ox, dy + 126, 200, 44, "Softmax", 17, fill="#FFFFFF", stroke=INK))
P.append(arrow(ox + 100, dy + 172, ox + 100, dy + 194))
P.append(text(ox, dy + 214, "P(下一个 token)", 14, fill=MUTED))
probs = [("NLP", 0.72), ("你", 0.11), ("学习", 0.06)]
for i, (t, p) in enumerate(probs):
    yy = dy + 226 + i * 24
    P.append(text(ox + 40, yy + 15, t, 13, anchor="end"))
    P.append(rect(ox + 48, yy + 3, 140 * p, 16, ORANGE if i == 0 else ORANGE_L, "none", 0, 3))
    P.append(text(ox + 54 + 140 * p, yy + 15, f"{p:.2f}", 12, fill=MUTED))

# loop back for inference
P.append(path(f"M{ox - 12},{dy + 250} C{ox - 70},{dy + 300} 140,{dy + 300} 115,{dy + 214}", MUTED, 1.4, arrow=True, dash="5 4"))
P.append(text(800, dy + 316, "推理：把生成的 token 接到前缀末尾，自回归逐个生成（beam size 4，长度惩罚 α=0.6）", 15, fill=MUTED, anchor="middle"))

# ======================= hyper-parameter panel =======================
hx, hy, hw, hh = 1340, 172, 200, 250
P.append(rect(hx, hy, hw, hh, PANEL, LINE, 1.2, 10))
P.append(text(hx + 16, hy + 30, "base 配置", 16, weight=700))
rows = [("N", "6 + 6 层"), ("d_{model}", "512"), ("d_{ff}", "2048"), ("h", "8 头"),
        ("d_{k}=d_{v}", "64"), ("P_{drop}", "0.1"), ("参数量", "65M")]
for i, (k, v) in enumerate(rows):
    yy = hy + 62 + i * 26
    P.append(text(hx + 16, yy, k, 15, fill=MUTED, family=FONT))
    P.append(text(hx + hw - 16, yy, v, 15, fill=INK, weight=600, anchor="end"))
P.append(text(hx + 16, hy + hh - 6, "big：d=1024, d_{ff}=4096, h=16", 12, fill=MUTED))

# training note
P.append(footer(1))
P.append(text(800, dy + 342, "训练：Teacher Forcing + 因果掩码，目标序列所有位置一次前向并行算完", 15, fill=ORANGE, anchor="middle", weight=600))
render("01-method-overview", "".join(P))
