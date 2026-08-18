#!/usr/bin/env python3
"""MT-Bench / Chatbot Arena 的零依赖评测教学实现。

覆盖：
1. pairwise 与 single-answer prompt；
2. [[A]] / [[B]] / [[C]] 与 [[rating]] 的严格解析；
3. 交换位置后保守合并，诊断 position bias；
4. 含/不含平局的一致率、平均对手胜率；
5. 可选 Elo 与 bootstrap 区间（补充工具，不是论文正文核心指标）。

脚本不调用在线 LLM；把 call_judge() 接到任意模型 API 即可用于真实评测。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
import re
from statistics import mean
from typing import Iterable, Literal


RawVerdict = Literal["A", "B", "tie", "error"]
Winner = str  # 规范化后是模型名、"tie" 或 "error"


PAIRWISE_SYSTEM = """\
Act as an impartial judge. Compare two assistants on helpfulness, relevance,
accuracy, depth, creativity, and instruction following. Do not let response
order, assistant name, or answer length influence the decision. Explain briefly,
then end with exactly [[A]], [[B]], or [[C]] for a tie."""


SINGLE_SYSTEM = """\
Act as an impartial judge. Evaluate the answer for helpfulness, relevance,
accuracy, depth, creativity, and instruction following. Explain briefly, then
end with a rating from 1 to 10 in the exact form [[rating]]."""


@dataclass(frozen=True)
class Conversation:
    questions: tuple[str, ...]
    answers: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.questions) != len(self.answers):
            raise ValueError("每一轮问题都必须有对应回答")
        if not self.questions:
            raise ValueError("对话不能为空")


@dataclass(frozen=True)
class SwappedJudgment:
    """同一对回答的两次判决：原顺序与交换顺序。"""

    item_id: str
    model_1: str
    model_2: str
    original_output: str  # model_1 在 A，model_2 在 B
    swapped_output: str   # model_2 在 A，model_1 在 B


@dataclass(frozen=True)
class Vote:
    item_id: str
    model_a: str
    model_b: str
    winner: Winner  # model_a / model_b / tie
    judge: str = "human"

    def __post_init__(self) -> None:
        allowed = {self.model_a, self.model_b, "tie"}
        if self.winner not in allowed:
            raise ValueError(f"winner 必须属于 {allowed}，实际为 {self.winner!r}")


def format_conversation(name: str, conversation: Conversation) -> str:
    lines = [f"<|START OF {name} CONVERSATION|>"]
    for question, answer in zip(conversation.questions, conversation.answers):
        lines.extend((f"User: {question}", f"{name}: {answer}"))
    lines.append(f"<|END OF {name} CONVERSATION|>")
    return "\n\n".join(lines)


def build_pairwise_prompt(
    conversation_a: Conversation,
    conversation_b: Conversation,
    reference: Conversation | None = None,
) -> str:
    """完整展示两段多轮对话，避免裁判把第二轮指代串线。"""

    if conversation_a.questions != conversation_b.questions:
        raise ValueError("A/B 必须回答相同问题")
    blocks: list[str] = []
    if reference is not None:
        if reference.questions != conversation_a.questions:
            raise ValueError("参考答案必须对应相同问题")
        blocks.append(format_conversation("REFERENCE", reference))
    blocks.append(format_conversation("ASSISTANT A", conversation_a))
    blocks.append(format_conversation("ASSISTANT B", conversation_b))
    if len(conversation_a.questions) > 1:
        blocks.append("Focus the verdict on the answer to the final user question.")
    return PAIRWISE_SYSTEM + "\n\n" + "\n\n".join(blocks)


def build_single_prompt(
    conversation: Conversation,
    reference: Conversation | None = None,
) -> str:
    blocks: list[str] = []
    if reference is not None:
        blocks.append(format_conversation("REFERENCE", reference))
    blocks.append(format_conversation("ASSISTANT", conversation))
    if len(conversation.questions) > 1:
        blocks.append("Grade the answer to the final user question.")
    return SINGLE_SYSTEM + "\n\n" + "\n\n".join(blocks)


def parse_pairwise(output: str) -> RawVerdict:
    """只接受唯一一个合法 marker；多个互相冲突的 marker 视为错误。"""

    markers = re.findall(r"\[\[([ABC])\]\]", output.upper())
    unique = set(markers)
    if len(unique) != 1:
        return "error"
    return {"A": "A", "B": "B", "C": "tie"}[unique.pop()]  # type: ignore[return-value]


def parse_single_rating(output: str) -> float | None:
    matches = re.findall(r"\[\[(\d+(?:\.\d+)?)\]\]", output)
    if len(matches) != 1:
        return None
    rating = float(matches[0])
    return rating if 1.0 <= rating <= 10.0 else None


def canonical_winner(
    raw: RawVerdict,
    model_in_a: str,
    model_in_b: str,
) -> Winner:
    if raw == "A":
        return model_in_a
    if raw == "B":
        return model_in_b
    return raw


def reconcile_swapped(judgment: SwappedJudgment) -> Winner:
    """论文的保守策略：同一模型在两个位置都胜才判胜，否则记平局。"""

    first = canonical_winner(
        parse_pairwise(judgment.original_output), judgment.model_1, judgment.model_2
    )
    second = canonical_winner(
        parse_pairwise(judgment.swapped_output), judgment.model_2, judgment.model_1
    )
    if "error" in {first, second}:
        return "error"
    if first == second and first in {judgment.model_1, judgment.model_2}:
        return first
    # 两次都 tie，或一次胜/一次 tie，或交换后翻转，都采用保守平局。
    return "tie"


def position_bias_bucket(judgment: SwappedJudgment) -> str:
    """按两次原始位置判决区分一致、偏第一、偏第二与错误。"""

    original = parse_pairwise(judgment.original_output)
    swapped = parse_pairwise(judgment.swapped_output)
    if "error" in {original, swapped}:
        return "error"
    if original == "A" and swapped == "A":
        return "biased_first"
    if original == "B" and swapped == "B":
        return "biased_second"
    mapped_1 = canonical_winner(original, judgment.model_1, judgment.model_2)
    mapped_2 = canonical_winner(swapped, judgment.model_2, judgment.model_1)
    if mapped_1 == mapped_2:
        return "consistent"
    return "inconsistent"


def position_bias_report(judgments: Iterable[SwappedJudgment]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for judgment in judgments:
        counts[position_bias_bucket(judgment)] += 1
        total += 1
    if total == 0:
        return {}
    return {name: count / total for name, count in sorted(counts.items())}


def labels_by_item(votes: Iterable[Vote], judge: str) -> dict[str, Winner]:
    return {vote.item_id: vote.winner for vote in votes if vote.judge == judge}


def agreement(
    votes: Iterable[Vote],
    judge_x: str,
    judge_y: str,
    include_ties: bool,
) -> tuple[float, int]:
    """同一 item 上两个 judge 标签相同的比例。

    论文的人-人一致率还会从同一问题的多名标注者中抽不同个体；这里展示
    两条已对齐 judge 序列的最小版本。
    """

    xs = labels_by_item(votes, judge_x)
    ys = labels_by_item(votes, judge_y)
    pairs = []
    for item_id in xs.keys() & ys.keys():
        x, y = xs[item_id], ys[item_id]
        if not include_ties and (x == "tie" or y == "tie"):
            continue
        pairs.append(x == y)
    if not pairs:
        raise ValueError("没有满足条件的重叠投票")
    return sum(pairs) / len(pairs), len(pairs)


def mt_bench_score(outputs: Iterable[str]) -> tuple[float, int]:
    ratings = [parse_single_rating(output) for output in outputs]
    valid = [rating for rating in ratings if rating is not None]
    if not valid:
        raise ValueError("没有合法的 [[rating]]")
    return mean(valid), len(valid)


def pairwise_records(
    votes: Iterable[Vote], include_ties: bool = True
) -> dict[tuple[str, str], list[float]]:
    """从每个模型视角记录胜负；可把平局计 0.5 或直接排除。"""

    records: dict[tuple[str, str], list[float]] = defaultdict(list)
    for vote in votes:
        if vote.winner == "tie":
            if not include_ties:
                continue
            score_a = score_b = 0.5
        elif vote.winner == vote.model_a:
            score_a, score_b = 1.0, 0.0
        else:
            score_a, score_b = 0.0, 1.0
        records[(vote.model_a, vote.model_b)].append(score_a)
        records[(vote.model_b, vote.model_a)].append(score_b)
    return records


def average_opponent_win_rate(
    votes: Iterable[Vote], include_ties: bool = True
) -> dict[str, float]:
    """先算对每个对手的胜率，再对实际交手对手等权平均。

    论文说明 average win rate 可含或不含 tie；默认采用平局计 0.5 的版本。
    """

    records = pairwise_records(votes, include_ties=include_ties)
    by_model: dict[str, list[float]] = defaultdict(list)
    for (model, _opponent), results in records.items():
        by_model[model].append(mean(results))
    return {model: mean(rates) for model, rates in sorted(by_model.items())}


def elo_ratings(
    votes: Iterable[Vote],
    k: float = 16.0,
    initial: float = 1000.0,
) -> dict[str, float]:
    """Arena 风格补充：在线 Elo 依赖投票顺序，不是论文正文的核心统计。"""

    ratings: dict[str, float] = defaultdict(lambda: initial)
    for vote in votes:
        ra, rb = ratings[vote.model_a], ratings[vote.model_b]
        expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        if vote.winner == "tie":
            actual_a = 0.5
        else:
            actual_a = 1.0 if vote.winner == vote.model_a else 0.0
        delta = k * (actual_a - expected_a)
        ratings[vote.model_a] += delta
        ratings[vote.model_b] -= delta
    return dict(sorted(ratings.items(), key=lambda pair: pair[1], reverse=True))


def bootstrap_win_rate_interval(
    votes: list[Vote],
    model: str,
    repeats: int = 1000,
    seed: int = 7,
) -> tuple[float, float]:
    """对 battle 记录有放回抽样，给排名附上不确定性。"""

    if not votes:
        raise ValueError("votes 不能为空")
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(repeats):
        resampled = [rng.choice(votes) for _ in votes]
        rates = average_opponent_win_rate(resampled)
        if model in rates:
            samples.append(rates[model])
    if not samples:
        raise ValueError(f"没有模型 {model!r} 的投票")
    samples.sort()
    lo = samples[math.floor(0.025 * (len(samples) - 1))]
    hi = samples[math.floor(0.975 * (len(samples) - 1))]
    return lo, hi


def verbosity_attack_failure_rate(
    judgments: Iterable[tuple[str, str]],
) -> float:
    """每项为 (原答案在 A 的输出, 冗长复制版在 A 的输出)。

    第二个输出若裁判选择 A，表示无新增信息的加长答案成功骗过裁判。
    """

    rows = list(judgments)
    if not rows:
        raise ValueError("至少需要一项攻击测试")
    failures = sum(parse_pairwise(padded_output) == "A" for _, padded_output in rows)
    return failures / len(rows)


def synthetic_votes() -> list[Vote]:
    """小型合成数据，仅用于演示聚合；不是论文原始投票。"""

    battles = [
        ("q01", "Alpha", "Beta", "Alpha"),
        ("q02", "Alpha", "Beta", "Alpha"),
        ("q03", "Alpha", "Beta", "tie"),
        ("q04", "Alpha", "Gamma", "Gamma"),
        ("q05", "Alpha", "Gamma", "Alpha"),
        ("q06", "Beta", "Gamma", "Gamma"),
        ("q07", "Beta", "Gamma", "Gamma"),
        ("q08", "Beta", "Gamma", "tie"),
        ("q09", "Gamma", "Alpha", "Alpha"),
        ("q10", "Beta", "Alpha", "Alpha"),
    ]
    return [Vote(*battle, judge="human") for battle in battles]


def run_self_checks() -> None:
    assert parse_pairwise("Reasoning... [[A]]") == "A"
    assert parse_pairwise("[[A]] but maybe [[B]]") == "error"
    assert parse_single_rating("Rating: [[8.5]]") == 8.5
    assert parse_single_rating("Rating: [[11]]") is None
    consistent = SwappedJudgment("x", "M1", "M2", "[[A]]", "[[B]]")
    first_bias = SwappedJudgment("y", "M1", "M2", "[[A]]", "[[A]]")
    assert reconcile_swapped(consistent) == "M1"
    assert reconcile_swapped(first_bias) == "tie"
    assert position_bias_bucket(first_bias) == "biased_first"


def main() -> None:
    run_self_checks()

    q = ("What is 17 × 6?", "Explain the calculation in one sentence.")
    answer_a = Conversation(q, ("102", "17 × 6 = 102."))
    answer_b = Conversation(q, ("96", "Adding six groups of 17 gives 96."))
    reference = Conversation(q, ("102", "17 × 6 = 102."))
    prompt = build_pairwise_prompt(answer_a, answer_b, reference)
    print("PROMPT_PREVIEW")
    print("\n".join(prompt.splitlines()[:10]), "\n...")

    swapped = [
        SwappedJudgment("q1", "Alpha", "Beta", "Reason... [[A]]", "Reason... [[B]]"),
        SwappedJudgment("q2", "Alpha", "Beta", "[[A]]", "[[A]]"),
        SwappedJudgment("q3", "Alpha", "Beta", "[[B]]", "[[B]]"),
        SwappedJudgment("q4", "Alpha", "Beta", "[[C]]", "[[C]]"),
    ]
    print("\nPOSITION_BIAS", position_bias_report(swapped))
    print("CONSERVATIVE", [reconcile_swapped(row) for row in swapped])

    ratings = ["Good. [[8]]", "Mostly correct. [[7.5]]", "Excellent. [[9]]"]
    print("MT_BENCH", mt_bench_score(ratings))

    # 两组已对齐标签，演示 S1（含 tie）与 S2（排除 tie）。
    human_labels = ["Alpha", "Beta", "tie", "Alpha", "Beta"]
    gpt_labels = ["Alpha", "Beta", "Alpha", "Alpha", "Beta"]
    aligned: list[Vote] = []
    for index, (human, gpt) in enumerate(zip(human_labels, gpt_labels), start=1):
        item_id = f"a{index}"
        aligned.append(Vote(item_id, "Alpha", "Beta", human, "human"))
        aligned.append(Vote(item_id, "Alpha", "Beta", gpt, "gpt-judge"))
    print("AGREEMENT_S1", agreement(aligned, "human", "gpt-judge", include_ties=True))
    print("AGREEMENT_S2", agreement(aligned, "human", "gpt-judge", include_ties=False))

    votes = synthetic_votes()
    rates = average_opponent_win_rate(votes)
    print("AVERAGE_WIN_RATE", {name: round(rate, 3) for name, rate in rates.items()})
    print("ELO_SUPPLEMENT", {name: round(score, 1) for name, score in elo_ratings(votes).items()})
    print("ALPHA_95CI", tuple(round(x, 3) for x in bootstrap_win_rate_interval(votes, "Alpha")))

    attacks = [
        ("Original better. [[B]]", "Longer but repetitive wins. [[A]]"),
        ("Original better. [[B]]", "No new information; tie. [[C]]"),
    ]
    print("VERBOSITY_ATTACK_FAILURE", verbosity_attack_failure_rate(attacks))


if __name__ == "__main__":
    main()
