"""00 · 封面：一句话贡献 + “同一个模型，换一段提示就换一个任务”的视觉锚点"""
from lib import *

P = []
# ---------------- left: title block ----------------
P.append(text(60, 92, "论文方法图解 · 共 7 张", 17, fill=INK, weight=700))
P.append(text(60, 156, ["Language Models are", "Unsupervised Multitask Learners"], 35, weight=800, lh=1.25))
P.append(text(60, 252, "GPT-2：把任务写成文字，语言模型直接续写出答案", 24, weight=600))
P.append(text(60, 284, "Radford, Wu, Child, Luan, Amodei, Sutskever · OpenAI 2019", 14, fill=MUTED))

P.append(rect(60, 310, 640, 150, PANEL, LINE, 1.4, 12))
P.append(text(84, 344, "一句话", 15, weight=700, fill=MUTED))
P.append(text(84, 380, ["在 800 万篇网页（WebText，40GB）上训练一个 15 亿参数的语言模型，",
                        "**不做任何微调**，只在输入里写一段提示，模型就能接着写出答案：",
                        "**8 个语言建模数据集中 7 个 zero-shot 刷新 SOTA，CoQA 55 F1。**"], 17, lh=1.55))

P.append(text(60, 498, "两样东西", 15, weight=700, fill=MUTED))
P.append(legend(700, 494, ("lm", "prompt")))
cards = [("一个大语言模型", "1.5B 参数，只学“预测下一个 token”", LM, LM_L),
         ("把任务写成文字", "TL;DR:  ·  A:  ·  english = french", PROMPT, PROMPT_L)]
for i, (a, b_, col, fill) in enumerate(cards):
    x = 60 + i * 340
    P.append(rect(x, 512, 300, 72, fill, col, 1.5, 10))
    P.append(ctext(x + 150, 536, a, 17, fill=col, weight=700))
    P.append(ctext(x + 150, 564, b_, 13, fill=INK))
P.append(ctext(380, 548, "+", 24, fill=INK, weight=700))

P.append(text(60, 622, "目录", 15, weight=700, fill=MUTED))
toc = ["01 方法总览：任务也是一段文字", "02 WebText：用人的筛选换数据质量",
       "03 字节级 BPE 与模型改动", "04 Zero-shot：用提示唤起任务",
       "05 规模与泛化：越大越好，不是背题", "06 关键结果：强在哪、弱在哪"]
for i, t in enumerate(toc):
    col, row = i % 2, i // 2
    P.append(text(60 + col * 330, 648 + row * 25, t, 14))

# ---------------- right: one model, many prompts ----------------
rx, ry, rw, rh = 760, 60, 780, 620
P.append(rect(rx, ry, rw, rh, "#FFFFFF", LINE, 1.2, 12))
P.append(text(rx + 24, ry + 38, "同一个模型、同一组参数：换一段提示，就换一个任务", 18, weight=700))
P.append(legend(rx + rw - 24, ry + 560))
P.append(segs(rx + 24, ry + 72, [("p(output | input, ", INK, 400), ("task", PROMPT, 700), (")", INK, 400),
                                 ("　—— task 不是新模块，而是输入里的几个字", MUTED, 400)], 16, family=MATH))
P.append(segs(rx + 24, ry + 104, [("训练：", INK, 700), ("在 WebText 上预测下一个 token　", MUTED, 400),
                                  ("使用：", INK, 700), ("参数不动，只换提示", MUTED, 400)], 14))

rows = [([("新闻全文 …", "ctx"), ("TL;DR:", "prompt")], "摘要"),
         ([("dog = chien", "prompt"), ("cat = chat", "prompt"), ("bird", "ctx"), ("=", "prompt")], "oiseau"),
         ([("文档 …", "ctx"), ("对话历史 …", "ctx"), ("A:", "prompt")], "答案"),
         ([("示例问答对 …", "prompt"), ("新问题", "ctx")], "短答案")]
mx, mw = rx + 470, 150  # model box
my, mh = ry + 130, 394
P.append(rect(mx, my, mw, mh, LM_LL, LM, 2, 14))
P.append(ctext(mx + mw / 2, my + mh / 2 - 30, "GPT-2", 26, fill=LM, weight=800))
P.append(ctext(mx + mw / 2, my + mh / 2 + 10, ["48 层 Transformer", "1.5B 参数"], 14, fill=INK, lh=1.5))
P.append(ctext(mx + mw / 2, my + mh / 2 + 70, ["预测下一个", "token"], 13, fill=LM, weight=600, lh=1.4))
for i, (seq, out) in enumerate(rows):
    y = ry + 150 + i * 100
    svg, end = prompt_seq(rx + 24, y, seq, h=34, size=13)
    # right-align the strip against the model box
    shift = (mx - 30) - end
    P.append(f'<g transform="translate({shift:.0f},0)">{svg}</g>')
    P.append(arrow(mx - 26, y + 17, mx - 4, y + 17, INK, 1.6))
    P.append(arrow(mx + mw + 4, y + 17, mx + mw + 30, y + 17, LM, 1.8))
    w = estimate_width(out, 14) + 26
    P.append(rect(mx + mw + 36, y, w, 34, LM_LL, LM, 1.2, 6, dash="4 3"))
    P.append(ctext(mx + mw + 36 + w / 2, y + 17, out, 14, fill=LM, weight=700))
P.append(text(rx + 24, ry + 590, "输入格式按原文 Section 3.5–3.8（摘要、翻译、阅读理解、开放域问答）；具体词句为示意。虚线框 = GPT-2 生成的内容。",
              12, fill=MUTED))

# ---------------- bottom stat cards (all are GPT-2 results -> LM colour) ----------------
cy = 712
stats = [("7 / 8", "语言建模数据集 zero-shot SOTA", "唯一没赢的 1BW：42.16 vs 此前 21.8"),
         ("55 F1", "CoQA 阅读理解", "人类 89 F1；没用 12.7 万+ 条训练问答"),
         ("8.6", "LAMBADA 困惑度", "此前最佳 99.8；准确率 19% → 52.66%"),
         ("1.5B", "参数，仍然欠拟合 WebText", "GPT-1 为 117M，大了一个数量级以上")]
cw = (W - 120 - 3 * 20) / 4
for i, (big, lab, sub) in enumerate(stats):
    x = 60 + i * (cw + 20)
    P.append(rect(x, cy, cw, 128, "#FFFFFF", LINE, 1.2, 12))
    P.append(text(x + 22, cy + 52, big, 34, weight=800, fill=LM))
    P.append(text(x + 22, cy + 84, lab, 15, weight=600))
    P.append(text(x + 22, cy + 110, sub, 12, fill=MUTED))

P.append(text(60, H - 22, SOURCE, 13, fill=FAINT))
P.append(text(W - 60, H - 22, f"00 / {TOTAL_PAGES:02d}", 15, fill=MUTED, weight=600, anchor="end"))
render("00-cover", "".join(P))
