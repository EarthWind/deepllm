"""重建博客插图：手工 SVG 架构图 + matplotlib 数据图。

python -m pip install -r project/laya/code/requirements-figures.txt
MPLCONFIGDIR=/tmp/laya-mpl python project/laya/code/build_figures.py
"""
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images"
OUT.mkdir(exist_ok=True)
DATA = json.loads((ROOT / "data/benchmark_extract.json").read_text())
BLUE, GOLD, INK, MUTED = "#2864b4", "#c17a24", "#192d43", "#52667a"


class Diagram:
    def __init__(self, title, subtitle, height=750):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height}" viewBox="0 0 1400 {height}" role="img">',
                      f'<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>',
                      '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0 0 L10 4 L0 8" fill="#72879b"/></marker></defs>',
                      f'<rect width="1400" height="{height}" fill="#f7f9fc"/>',
                      '<g font-family="Noto Sans CJK SC,Microsoft YaHei,sans-serif">']
        self.text(50, 58, title, 32, INK, weight=700)
        self.text(50, 98, subtitle, 19, MUTED)

    def text(self, x, y, value, size=21, color=INK, weight=400):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(value)}</text>')

    def box(self, x, y, width, height, title, lines=(), color=BLUE):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="white" stroke="#d4dee8" stroke-width="2"/>')
        self.parts.append(f'<rect x="{x}" y="{y+14}" width="5" height="{height-28}" fill="{color}"/>')
        self.text(x+22, y+39, title, 23, color, 700)
        for i, line in enumerate(lines):
            self.text(x+22, y+77+i*32, line, 19)

    def arrow(self, x1, y1, x2, y2):
        self.parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="#72879b" stroke-width="2.5" marker-end="url(#arrow)"/>')

    def save(self, name, footer):
        self.text(50, self.height-26, footer, 17, MUTED)
        self.parts.append('</g></svg>')
        (OUT / name).write_text("\n".join(self.parts), encoding="utf-8")


def schematics():
    d = Diagram("Laya：从文本到有类型的决策", "一次批量前向计算；每个问题拥有自己的输入序列与输出分布", 780)
    d.box(50, 155, 300, 180, "输入", ["state：文本 / JSON / 对话", "questions：类型 + 指令", "criteria：候选标签或等级"])
    d.box(425, 155, 430, 180, "双向编码器", ["ModernBERT-large / mmBERT-base", "每题构造一条 [问题 + 选项 + 文本]", "H ∈ R^(B × L × d)"])
    d.box(930, 155, 420, 180, "类型感知决策头", ["H + type embedding", "2 层 Transformer（全序列）", "从每个选项的 [MASK] 提取表示"])
    d.arrow(350, 245, 425, 245); d.arrow(855, 245, 930, 245)
    d.box(50, 430, 300, 215, "类型化答案", ["choice：argmax + 概率字典", "score：等级索引的期望", "noul：P(true)", "output_tokens = 0"], GOLD)
    d.box(425, 430, 430, 215, "共享选项评分器", ["LayerNorm → Linear → GELU", "→ Linear(1)，每个 marker 一个 logit", "按题选择温度 T，softmax(z / T)", "标签空间可在请求时改变"])
    d.box(930, 430, 420, 215, "另一路：action head", ["[CLS] + top1 / margin / entropy / K", "小型 MLP → act / escalate", "公开 notebook 中 act 损失权重为 0", "不能把它视作已可靠训练的拒答器"], GOLD)
    d.parts.append('<path d="M1140 335 V385 H640 V430" fill="none" stroke="#72879b" stroke-width="2.5" marker-end="url(#arrow)"/>')
    d.arrow(1140, 385, 1140, 430); d.arrow(425, 538, 350, 538)
    d.arrow(855, 540, 930, 540)
    d.save("architecture.svg", "依据 common.py / agent.py；箭头表示张量流。action 分支使用未按题校准的 logits 派生特征。")

    d = Diagram("一次前向调用 ≠ 文本只编码一次", "两个问题对应 batch 中两行；问题之间没有联合注意力或一致性约束", 730)
    for y, question, opts in [(155, "Q1：choice · 应转哪个部门？", "[MASK] billing   [MASK] technical   [MASK] sales"),
                               (325, "Q2：noul · 是否明确要求退款？", "[MASK] false: …   [MASK] true: …")]:
        d.box(50, y, 800, 135, question, ["[CLS] 指令 [SEP] " + opts, "[SEP] 同一份 state 的副本 [SEP]"])
    d.box(950, 215, 400, 195, "collate_items", ["padding + attention_mask", "marker_pos + marker_mask", "堆叠为 [2, L]，调用模型一次", "两行各自计算 attention"])
    d.arrow(850, 220, 950, 280); d.arrow(850, 390, 950, 350)
    d.box(50, 530, 1300, 125, "长度预算会影响模型实际看见什么", ["max_len 是序列硬截断；head_max_len 是问题头的启发式预算，并非严格上界。", "77 个选项 × 最少 4 token = 308 token，仅选项就可能超过 192 / 256 的问题头预算。"], GOLD)
    d.save("sequence-batch.svg", "依据 build_sequence / _encode_state / predict_batch；对话 list 保留尾部，字符串与字典默认保留前部。")

    d = Diagram("公开微调 notebook：RLCD + soft-label CE", "在 logit 空间探索概率分布；奖励不是对自然语言回答进行打分", 750)
    d.box(50, 150, 360, 165, "模型输出 μθ", ["训练目标：教师软分布 t", "G = 4 组高斯噪声", "对有效选项投影到零和空间"])
    d.box(505, 150, 365, 165, "候选分布 qg", ["zg = stopgrad(μθ) + εg", "qg = softmax(zg)", "相同输入，探索不同概率报告"])
    d.box(965, 150, 385, 165, "分布奖励 Rg", ["log score + 0.75 spherical", "score 类型再减 RPS", "组均值基线 + 标准差归一化"])
    d.arrow(410, 230, 505, 230); d.arrow(870, 230, 965, 230)
    d.box(50, 430, 360, 190, "监督项 LCE", ["−Σ tk log softmax(μθ)k", "系数 1.0", "与策略梯度共同更新模型"], GOLD)
    d.box(505, 430, 365, 190, "策略项 LRL", ["−mean(Ag · log πθ(zg))", "detach 固定采样点与 advantage", "梯度穿过分布均值 μθ"])
    d.box(965, 430, 385, 190, "训练后校准", ["保留样本拟合温度", "清除旧的 bucket 覆盖项", "实际推理仍会将温度夹紧", "独立测试集评估最终概率"], GOLD)
    d.arrow(230, 315, 230, 430); d.arrow(690, 315, 690, 430)
    d.arrow(1150, 315, 1150, 375); d.arrow(1150, 375, 690, 375); d.arrow(690, 375, 690, 430)
    d.save("training.svg", "教学重绘；当前 notebook 对 action head 使用 0 权重，不含 PPO clipping 或显式参考策略 KL。")

    d = Diagram("Router：规则选择 checkpoint，随后才运行神经网络", "按优先级命中即返回；路由规则本身不需要加载模型权重", 860)
    steps = [("1  显式 model", "直接指定 english / multilingual / typed-decisions"),
             ("2  显式 task", "typed_decisions 对应领域微调 checkpoint"),
             ("3  工作流 ID 集合", "仅 auto_task_detection=True 生效；精确匹配四类 schema"),
             ("4  显式 lang", "en → english；其他语言 → multilingual"),
             ("5  lang_guess", "单次请求提示优先，再看 Router 级别提示"),
             ("6  字符与词表规则", "非拉丁文字 / 非英语线索 → multilingual"),
             ("7  default", "短文本或缺乏证据时回退；默认 english")]
    for i, (title, detail) in enumerate(steps):
        y = 142+i*89
        d.box(50, y, 1280, 72, title, [])
        d.text(390, y+43, detail, 21)
        if i < len(steps)-1:
            d.arrow(110, y+72, 110, y+89)
    d.save("routing.svg", "注意：自动工作流匹配排在 lang 之前；启用后可能优先选中英语领域模型。来源：Router.route。")


def charts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    fonts = list(Path("/usr/share/fonts/opentype/noto").glob("NotoSansCJK-Regular.ttc"))
    if fonts:
        font_manager.fontManager.addfont(str(fonts[0]))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(fonts[0])).get_name()
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "text.color": INK, "axes.labelcolor": INK,
                         "axes.edgecolor": "#adb9c7", "savefig.facecolor": "white",
                         "svg.fonttype": "path", "axes.unicode_minus": False})
    def save(fig, name):
        fig.savefig(OUT / f"{name}.png", dpi=170)
        svg_path = OUT / f"{name}.svg"
        fig.savefig(svg_path)
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8))
    fig.subplots_adjust(left=.075, right=.975, top=.79, bottom=.23, wspace=.28)
    fig.suptitle("T4：批处理降低每题均摊耗时，整次请求仍会变慢", fontsize=19, x=.075, ha="left")
    fig.text(.075, .875, "上游 2026-09-19 记录 · 预热后 p50 · 两图使用不同纵轴单位", color=MUTED)
    counts = [1, 5, 10, 50]
    for model, label, color, marker in [("laya", "English", BLUE, "o"),
                                         ("laya-multilingual", "Multilingual", GOLD, "s")]:
        rows = DATA["t4"]["latency"][model]
        ys = [rows[f"{n}_questions"]["p50_ms"] for n in counts]
        ps = [rows[f"{n}_questions"]["ms_per_question"] for n in counts]
        for ax, values in zip(axes, [ys, ps]):
            ax.plot(counts, values, color=color, marker=marker, label=label, linewidth=2)
            for x, y in zip(counts, values):
                if ax is axes[0] and x in (1, 5):
                    continue  # 短批次点接近零线；精确值见正文，避免标注压到刻度。
                ax.annotate(f"{y:.1f}", (x, y), xytext=(3, 9 if model=="laya" else -18),
                            textcoords="offset points", fontsize=10, color=color)
    for ax, title, unit in zip(axes, ["整次调用延迟", "每题均摊耗时"], ["毫秒 / 调用", "毫秒 / 问题"]):
        ax.set_title(title); ax.set_ylabel(unit); ax.set_xlabel("每次调用的问题数")
        ax.set_xticks(counts); ax.set_xlim(-2, 55); ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e4e9ee"); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left")
    fig.text(.075, .045, "来源：research/results/t4_colab_benchmark.json → latency；本次未重跑 GPU。\n不含置信区间。72.3 ms / 10 题 ≈ 7.23 ms/题，不表示单题响应只需 7.23 ms。", fontsize=10, color=MUTED)
    save(fig, "latency")

    fig, ax = plt.subplots(figsize=(13.8, 6.8))
    fig.subplots_adjust(left=.20, right=.96, top=.79, bottom=.22)
    suites = [("jev.ag_news", "AG News（训练来源内）"), ("jev.emotion", "Emotion（来源外）"),
              ("jev.banking77_full", "Banking77（77 选项）"), ("app.guardrails_jailbreak", "Jailbreak（来源外）"),
              ("app.moderation_toxicity", "Toxicity（来源外）")]
    pos = np.arange(len(suites))
    for j, (model, label, color) in enumerate([("english", "English", BLUE),
                                              ("multilingual", "Multilingual", GOLD),
                                              ("typed-decisions", "Typed-decisions", "#748297")]):
        values = [DATA["apps"]["suites"][k][model]["accuracy"]*100 for k, _ in suites]
        bars = ax.barh(pos + (j-1)*.24, values, height=.22, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=3, fontsize=10)
    ax.set_yticks(pos, [label for _, label in suites]); ax.invert_yaxis()
    ax.set_xlim(0, 107); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_xlabel("准确率（%）")
    ax.grid(axis="x", color="#e4e9ee"); ax.set_axisbelow(True)
    fig.suptitle("跨任务表现：同一种接口，对应不同的能力边界", fontsize=19, x=.075, ha="left")
    fig.text(.075, .875, "上游 CPU 测量 · Laya 0.2.1 · 每任务每模型 n=400 · 同一任务内比较", color=MUTED)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3)
    fig.text(.075, .04, "来源：research/results/app_benchmark_results.json；本次未重跑模型，未给出置信区间。\n“训练来源内”不等于已证实样本泄漏；不同任务标签数与分布不同，横向高低不能直接衡量难度。", fontsize=10, color=MUTED)
    save(fig, "task-results")

    from decision_math import entropy_confidence
    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.subplots_adjust(left=.10, right=.95, bottom=.22, top=.80)
    p = np.linspace(.5, 1, 151)
    ax.plot(p, p, color=GOLD, linestyle="--", label="noul confidence = max(p, 1−p)", linewidth=2.5)
    ax.plot(p, [entropy_confidence([1-x, x]) for x in p], color=BLUE,
            label="2-option choice confidence = 1−H(p)/log 2", linewidth=2.5)
    ax.scatter([.9, .9], [.9, entropy_confidence([.1,.9])], c=[GOLD, BLUE], zorder=3)
    ax.annotate("0.900", (.9,.9), xytext=(8,-4), textcoords="offset points", color=GOLD)
    ax.annotate("0.531", (.9,entropy_confidence([.1,.9])), xytext=(8,-4), textcoords="offset points", color=BLUE)
    ax.set(xlim=(.5,1), ylim=(0,1.03), xlabel="最大类别概率 p", ylabel="API confidence 字段")
    ax.grid(color="#e4e9ee"); ax.legend(frameon=False, loc="upper left", fontsize=11)
    fig.suptitle("相同概率分布，两种 confidence 定义", fontsize=19, x=.10, ha="left")
    fig.text(.10, .86, "理论曲线：由源码公式计算，不是模型校准实验", color=MUTED)
    fig.text(.10, .055, "依据 agent.py::_decode_answers 与 common.py::confidence_from_probs。\nchoice / score 的归一化熵不是“预测正确的概率”，跨类型统一阈值会改变接受范围。", fontsize=10, color=MUTED)
    save(fig, "confidence")


if __name__ == "__main__":
    schematics()
    charts()
    print("Generated 4 schematic SVGs and 3 chart pairs (PNG + SVG).")
