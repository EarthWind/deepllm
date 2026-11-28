"""02 · WebText：用 Reddit 的人工筛选换数据质量；网页里本来就有任务示范"""
from lib import *

P = []
P.append(header("WebText：让人替你筛网页，再从里面找“任务示范”",
                "Section 2.1：只抓 Reddit 上至少 3 karma 的外链；数据越杂，模型能见到的“自然出现的任务”越多",
                2, "训练数据"))
P.append(legend(W - 60, 62, ("prompt", "base")))
y0, h0 = 178, 660

# ======================= ① funnel =======================
x0, w0 = 60, 700
P.append(rect(x0, y0, w0, h0, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x0 + 24, y0 + 28, 1, fill=INK))
P.append(text(x0 + 46, y0 + 34, "数据怎么来的", 17, weight=700))
steps = [("Reddit 帖子里的外链", "至少 3 karma；把“有人觉得有用、有趣”当作人工筛选"),
         ("4500 万个链接", "抓下网页"),
         ("抽取正文", "Dragnet + Newspaper 两种抽取器组合"),
         ("清洗", "只保留 2017 年 12 月前的链接；去重 + 启发式清洗"),
         ("删掉所有 Wikipedia 文档", "很多评测集取材于 Wikipedia；删掉以免训练和测试重叠")]
cx = x0 + 220
for i, (a, b_) in enumerate(steps):
    y = y0 + 70 + i * 92
    w = 340 - i * 34
    P.append(rect(cx - w / 2, y, w, 56, PANEL2, MUTED if i else INK, 1.4, 10))
    P.append(ctext(cx, y + 28, a, 16, weight=700))
    P.append(text(x0 + 410, y + 24, b_.split("；")[0] if "；" in b_ else b_, 13, fill=MUTED))
    if "；" in b_:
        P.append(text(x0 + 410, y + 44, b_.split("；")[1], 13, fill=MUTED))
    P.append(arrow(cx, y + 58, cx, y + 88, INK, 1.6))
P.append(rect(cx - 150, y0 + 530, 300, 76, "#FFFFFF", INK, 2.2, 12))
P.append(ctext(cx, y0 + 556, "WebText", 22, weight=800))
P.append(ctext(cx, y0 + 588, "800 万+ 文档 · 40GB 文本", 15, weight=600))
P.append(text(x0 + 410, y0 + 560, ["比 BooksCorpus 杂得多：", "新闻、教程、论坛、评论……"], 13, fill=MUTED, lh=1.5))
P.append(text(x0 + 24, y0 + h0 - 22, "注：论文结果都用 WebText 的初版（链接截至 2017 年 12 月）。", 12, fill=FAINT))

# ======================= ② why not existing corpora =======================
x1, w1 = 780, 760
h1 = 250
P.append(rect(x1, y0, w1, h1, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x1 + 24, y0 + 28, 2, fill=INK))
P.append(text(x1 + 46, y0 + 34, "为什么不用现成的语料", 17, weight=700))
opts = [("单一领域", ["新闻 / Wikipedia / 小说", "任务示范的种类太少"], "✗"),
        ("Common Crawl", ["量大几个数量级，但大量", "文档“基本读不懂”"], "✗"),
        ("按目标任务挑子集", ["Trinh & Le 的做法", "等于提前假设了任务"], "✗"),
        ("WebText", ["来源杂 + 人筛过", "不针对任何任务"], "✓")]
cw = (w1 - 48 - 3 * 12) / 4
for i, (t, lines, mark) in enumerate(opts):
    x = x1 + 24 + i * (cw + 12)
    last = i == 3
    P.append(rect(x, y0 + 60, cw, 166, PANEL if not last else "#FFFFFF", INK if last else LINE, 2 if last else 1.2, 10))
    P.append(ctext(x + cw / 2, y0 + 94, t, 15, weight=700))
    P.append(ctext(x + cw / 2, y0 + 140, lines, 13, fill=MUTED, lh=1.5))
    P.append(ctext(x + cw / 2, y0 + 196, mark, 22, weight=800, fill=INK if last else FAINT))

# ======================= ③ naturally occurring demonstrations =======================
y2 = y0 + h1 + 20
h2 = h0 - h1 - 20
P.append(rect(x1, y2, w1, h2, "#FFFFFF", LINE, 1.2, 12))
P.append(step_badge(x1 + 24, y2 + 28, 3, fill=INK))
P.append(text(x1 + 46, y2 + 34, "网页里本来就有“任务示范”（Table 1 摘录）", 17, weight=700))
P.append(text(x1 + 46, y2 + 58, "橙色这几个字，就是网页作者自己写下的“翻译提示”；模型为了预测后文，会顺带学会翻译", 13, fill=MUTED))
quotes = [
    [[("\"I'm not the cleverest man in the world, but ", INK, 400), ("like they say in French:", PROMPT, 700)],
     [("Je ne suis pas un imbecile [I'm not a fool].\"", INK, 400)]],
    [[("\"Mentez mentez, il en restera toujours quelque chose,\" ", INK, 400), ("which translates as,", PROMPT, 700)],
     [("\"Lie lie and something will always remain.\"", INK, 400)]],
    [[("\"Brevet Sans Garantie Du Gouvernement\", ", INK, 400), ("translated to English:", PROMPT, 700)],
     [("\"Patented without government warranty\".", INK, 400)]],
]
for i, q in enumerate(quotes):
    y = y2 + 80 + i * 78
    P.append(rect(x1 + 24, y, w1 - 48, 66, PANEL, LINE, 1, 8))
    P.append(line(x1 + 24, y + 8, x1 + 24, y + 58, PROMPT, 3))
    P.append(segs(x1 + 42, y + 27, q[0], 14))
    P.append(segs(x1 + 42, y + 51, q[1], 14))
P.append(rect(x1 + 24, y2 + h2 - 66, w1 - 48, 48, "#FFFFFF", LINE, 1.2, 8))
P.append(text(x1 + 42, y2 + h2 - 36,
              "WebText 刻意去掉了非英文网页，法语只剩约 **10MB**（约为常用无监督翻译法语语料的 1/500）。", 13))

P.append(footer(2))
render("02-webtext", "".join(P))
