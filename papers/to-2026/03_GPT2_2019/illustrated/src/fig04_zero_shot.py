"""04 · Zero-shot：每个任务怎么用提示唤起，效果如何"""
from lib import *

P = []
P.append(header("Zero-shot：同一个模型，靠提示格式切换任务",
                "Section 3.5–3.8：全部不做微调；输入怎么写、怎么解码都按原文。灰条为对照系统，蓝条为 GPT-2（1.5B）",
                4, "Zero-shot 任务"))
P.append(legend(W - 60, 62))


def hbar(x, y, label, val, vmax, width, kind="base", vtxt=None, lw=170, h=18, light=False):
    fill, stroke, col = {"lm": (LM_L if light else LM, LM, LM), "base": (BASE_L, FAINT, INK)}[kind]
    out = [text(x + lw - 10, y + h - 4, label, 12, fill=col if kind == "lm" else MUTED, weight=600 if kind == "lm" else 400,
                anchor="end")]
    w = max(2, val / vmax * width)
    out.append(rect(x + lw, y, w, h, fill, stroke, 1, 3))
    out.append(text(x + lw + w + 8, y + h - 4, vtxt or f"{val}", 12, fill=col, weight=700 if kind == "lm" else 400))
    return "".join(out)


def card(x, y, w, h, n, title, seq, decode):
    out = [rect(x, y, w, h, "#FFFFFF", LINE, 1.2, 12), step_badge(x + 24, y + 28, n, fill=INK),
           text(x + 46, y + 34, title, 17, weight=700)]
    s, end = prompt_seq(x + 24, y + 54, seq, h=32, size=13)
    out.append(s)
    out.append(text(x + 24, y + 108, decode, 12, fill=MUTED))
    return "".join(out)


cw, ch = 730, 262
xa, xb = 60, 60 + cw + 20
ya, yb = 178, 178 + ch + 16

# ---- A: CoQA ----
P.append(card(xa, ya, cw, ch, 1, "阅读理解 · CoQA",
              [("文档 …", "ctx"), ("对话历史 Q/A …", "ctx"), ("A:", "prompt"), ("答案", "out")],
              "贪心解码"))
P.append(text(xa + 24, ya + 160, "55", 40, weight=800, fill=LM))
P.append(text(xa + 82, ya + 160, "F1（开发集）", 15, weight=600))
P.append(text(xa + 24, ya + 192, ["追平或超过 4 个 baseline 中的 3 个——", "它们都用了 12.7 万+ 条人工标注问答"], 13, lh=1.5))
P.append(hbar(xa + 330, ya + 140, "GPT-2", 55, 89, 200, "lm", "55"))
P.append(hbar(xa + 330, ya + 168, "人类", 89, 89, 200, "base", "89"))
P.append(text(xa + 330 + 170, ya + 208, ["监督 SOTA（BERT）已接近人类；", "GPT-2 常靠简单启发式：", "问 who 就挑文中人名"], 12,
              fill=MUTED, lh=1.45))

# ---- B: summarization ----
P.append(card(xb, ya, cw, ch, 2, "摘要 · CNN / Daily Mail",
              [("新闻全文 …", "ctx"), ("TL;DR:", "prompt"), ("摘要", "out")],
              "Top-k 随机采样（k = 2）生成 100 个 token，取前 3 句；指标为 ROUGE 1/2/L 平均（Table 4）"))
sm = [("Bottom-Up Sum（SOTA）", 32.75, "base", False), ("Lede-3（前 3 句）", 31.55, "base", False),
      ("Seq2Seq + Attn", 23.99, "base", False), ("GPT-2 + TL;DR:", 21.40, "lm", False),
      ("随机 3 句", 20.98, "base", False), ("GPT-2 去掉 TL;DR:", 15.03, "lm", True)]
for i, (lab, v, k, light) in enumerate(sm):
    P.append(hbar(xb + 24, ya + 124 + i * 22, lab, v, 33, 220, k, f"{v:.2f}", lw=180, h=16, light=light))
P.append(rect(xb + 506, ya + 196, 200, 50, PROMPT_LL, PROMPT, 1.2, 8))
P.append(text(xb + 518, ya + 217, ["去掉 TL;DR: 降 **6.4**：", "提示真的在“调用”摘要任务"], 12, fill=INK, lh=1.45))
P.append(text(xb + 506, ya + 140, ["只比随机 3 句略好；", "常把车祸里几辆车、", "logo 在帽子还是衣服上", "这类细节弄错"], 12,
              fill=MUTED, lh=1.45))

# ---- C: translation ----
P.append(card(xa, yb, cw, ch, 3, "翻译 · WMT-14 英法",
              [("english = french", "prompt"), ("english = french", "prompt"), ("…", "ctx"), ("english sentence", "ctx"),
               ("=", "prompt"), ("french", "out")],
              "上下文里先放若干“英文句子 = 法文句子”示例对；贪心解码，取生成的第一句"))
P.append(hbar(xa + 24, yb + 132, "最好的无监督翻译（法→英）", 33.5, 33.5, 300, "base", "33.5 BLEU", lw=210))
P.append(hbar(xa + 24, yb + 158, "GPT-2 法→英", 11.5, 33.5, 300, "lm", "11.5 BLEU", lw=210))
P.append(hbar(xa + 24, yb + 184, "GPT-2 英→法", 5, 33.5, 300, "lm", "5 BLEU", lw=210))
P.append(text(xa + 24, yb + 228, ["法→英 超过了几个无监督翻译 baseline；英→法 略差于用双语词典逐词替换。",
                                  "意外之处：WebText 刻意去掉了非英文网页，法语只有约 10MB。"], 12, fill=MUTED, lh=1.45))

# ---- D: QA ----
P.append(card(xb, yb, cw, ch, 4, "开放域问答 · Natural Questions",
              [("示例问答对 …", "prompt"), ("新问题", "ctx"), ("短答案", "out")],
              "上下文里先放示例问答对，让模型学会“短答案”的格式；按 exact match 计分"))
P.append(hbar(xb + 24, yb + 132, "检索 + 抽取式系统", 30, 50, 300, "base", " ", lw=180))
P.append(rect(xb + 24 + 180 + 180, yb + 132, 120, 18, "#FFFFFF", FAINT, 1, 3, dash="4 3"))
P.append(text(xb + 24 + 180 + 308, yb + 146, "30–50%", 12))
P.append(hbar(xb + 24, yb + 158, "GPT-2（1.5B）", 4.1, 50, 300, "lm", "4.1%", lw=180))
P.append(hbar(xb + 24, yb + 184, "按问题类型答最常见答案", 1.0, 50, 300, "base", "1.0%（最小模型也没超过）", lw=180))
P.append(text(xb + 24, yb + 228, ["GPT-2 答对的题数是最小模型的 5.3 倍：容量是关键。",
                                  "它对自己的把握估得很准：最有把握的 1% 问题上正确率 63.1%。"], 12, fill=MUTED, lh=1.45))

# ---- bottom strip: no prompt needed ----
ys = yb + ch + 16
hs = 838 - ys
P.append(rect(60, ys, W - 120, hs, PANEL, LINE, 1.2, 12))
P.append(text(84, ys + 30, "不用提示：直接比较语言模型给出的概率", 15, weight=700))
items = [("LAMBADA", "预测需要长上下文的句末词", "63.24%", "准确率（加停用词过滤；不加为 52.66%）"),
         ("Children's Book Test", "10 个候选词各填进去，比整句概率", "93.3% / 89.1%", "普通名词 / 命名实体"),
         ("Winograd Schema", "代词换成不同候选，比较整句概率", "70.70%", "比此前最佳高 7 个点（仅 273 题）")]
iw = (W - 120 - 48) / 3
for i, (n, how, big, sub) in enumerate(items):
    x = 84 + i * iw
    P.append(text(x, ys + 60, n, 14, weight=700))
    P.append(text(x, ys + 80, how, 12, fill=MUTED))
    P.append(text(x + 250, ys + 66, big, 22, weight=800, fill=LM))
    P.append(text(x + 250, ys + 86, sub, 11, fill=MUTED))

P.append(footer(4))
render("04-zero-shot-prompts", "".join(P))
