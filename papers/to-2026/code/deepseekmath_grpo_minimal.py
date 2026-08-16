#!/usr/bin/env python3
"""DeepSeekMath 数据去污染与 GRPO 的零依赖教学实现。

脚本聚焦论文中最容易被二手解读混淆的三件事：

1. 用 benchmark 子串的精确 n-gram 匹配删除污染片段；
2. 对同一问题的一组回答做组内 reward 标准化；
3. 分别构造 outcome supervision 与 process supervision 的 token advantage，
   再计算 clipped surrogate 和直接加入目标的 KL 项。

这不是 DeepSeekMath 官方训练代码。真实系统还需要语言模型前向/反向、奖励模型、
token mask、分布式 rollout、旧策略快照和优化器更新。这里使用手写 token log-prob
轨迹，让每一项数学量都能单独检查。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TokenPolicyTrace:
    """一个回答中，每个有效 token 在新、旧、参考策略下的 log probability。"""

    old_logps: tuple[float, ...]
    new_logps: tuple[float, ...]
    ref_logps: tuple[float, ...]

    def __post_init__(self) -> None:
        lengths = {len(self.old_logps), len(self.new_logps), len(self.ref_logps)}
        if len(lengths) != 1 or not self.old_logps:
            raise ValueError("three non-empty log-probability sequences must align")


def contiguous_ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    """返回连续 n-gram；短文本不产生 n-gram。"""

    if n <= 0:
        raise ValueError("n must be positive")
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def decontaminate_segments(
    segments: Iterable[str], benchmark_texts: Iterable[str], n: int = 10
) -> list[str]:
    """删除与任一 benchmark 共享精确 n-gram 的文本片段。

    论文对不足 10 个、但至少 3 个 token 的短 benchmark 使用其完整 token 序列。
    这里把同一规则显式实现出来。分词仅按空白切分，生产版本应使用与语料处理
    一致的规范化和 tokenizer。
    """

    fingerprints: set[tuple[str, ...]] = set()
    for text in benchmark_texts:
        tokens = text.casefold().split()
        width = n if len(tokens) >= n else len(tokens)
        if width >= 3:
            fingerprints.update(contiguous_ngrams(tokens, width))

    clean: list[str] = []
    for segment in segments:
        tokens = segment.casefold().split()
        contaminated = any(
            fingerprint in contiguous_ngrams(tokens, len(fingerprint))
            for fingerprint in fingerprints
        )
        if not contaminated:
            clean.append(segment)
    return clean


def group_normalize(values: Sequence[float], epsilon: float = 1e-8) -> list[float]:
    """A_i = (r_i - mean(r)) / std(r)，零方差组安全返回全 0。"""

    if not values:
        raise ValueError("values must not be empty")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std < epsilon:
        return [0.0] * len(values)
    return [(value - mean) / std for value in values]


def outcome_advantages(
    rewards: Sequence[float], token_lengths: Sequence[int]
) -> list[list[float]]:
    """Outcome supervision：一个回答的所有 token 共用组内标准化结果。"""

    if len(rewards) != len(token_lengths):
        raise ValueError("one reward and one token length are required per answer")
    normalized = group_normalize(rewards)
    return [[advantage] * length for advantage, length in zip(normalized, token_lengths)]


def process_advantages(
    step_rewards: Sequence[Sequence[float]],
    step_end_tokens: Sequence[Sequence[int]],
    token_lengths: Sequence[int],
) -> list[list[float]]:
    """Process supervision：token 优势是其后续步骤标准化 reward 之和。

    论文先在该组所有步骤的 reward 集合 R 上做一次标准化；对 token t，再累加
    所有结束位置不早于 t 的步骤 reward。token 下标从 0 开始。
    """

    if not (len(step_rewards) == len(step_end_tokens) == len(token_lengths)):
        raise ValueError("answers, step boundaries and token lengths must align")
    if any(len(rs) != len(ends) for rs, ends in zip(step_rewards, step_end_tokens)):
        raise ValueError("each process reward needs one step end token")

    flattened = [reward for rewards in step_rewards for reward in rewards]
    normalized_flat = iter(group_normalize(flattened))
    normalized_steps = [
        [next(normalized_flat) for _ in rewards] for rewards in step_rewards
    ]

    advantages: list[list[float]] = []
    for rewards, ends, length in zip(
        normalized_steps, step_end_tokens, token_lengths
    ):
        if any(end < 0 or end >= length for end in ends):
            raise ValueError("step end token must fall inside its answer")
        advantages.append(
            [
                sum(reward for reward, end in zip(rewards, ends) if end >= token)
                for token in range(length)
            ]
        )
    return advantages


def positive_kl_estimator(new_logp: float, ref_logp: float) -> float:
    """论文采用的单样本 KL 估计：x - log(x) - 1，x = pi_ref / pi_theta。"""

    log_x = ref_logp - new_logp
    x = math.exp(log_x)
    return x - log_x - 1.0


def grpo_objective(
    traces: Sequence[TokenPolicyTrace],
    advantages: Sequence[Sequence[float]],
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
) -> float:
    """计算论文式 GRPO 目标；优化器训练时最小化它的相反数。"""

    if len(traces) != len(advantages) or not traces:
        raise ValueError("one non-empty advantage sequence is required per answer")

    answer_objectives: list[float] = []
    for trace, token_advantages in zip(traces, advantages):
        if len(trace.old_logps) != len(token_advantages):
            raise ValueError("token advantages and policy traces must align")
        token_objectives: list[float] = []
        for old, new, ref, advantage in zip(
            trace.old_logps,
            trace.new_logps,
            trace.ref_logps,
            token_advantages,
        ):
            ratio = math.exp(new - old)
            clipped = min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
            surrogate = min(ratio * advantage, clipped * advantage)
            token_objectives.append(
                surrogate - beta * positive_kl_estimator(new, ref)
            )
        answer_objectives.append(sum(token_objectives) / len(token_objectives))
    return sum(answer_objectives) / len(answer_objectives)


def majority_at_k(answers: Sequence[str], gold: str) -> float:
    """多数投票是否命中；平票时按首次出现次序确定，便于教学复现。"""

    if not answers:
        raise ValueError("answers must not be empty")
    counts = {answer: answers.count(answer) for answer in dict.fromkeys(answers)}
    winner = max(counts, key=counts.get)
    return float(winner == gold)


def pass_at_k(answers: Sequence[str], gold: str) -> float:
    """这组采样中是否至少出现一次正确答案。"""

    return float(any(answer == gold for answer in answers))


def demo() -> None:
    raw_segments = [
        "A proof about prime numbers with a fresh construction.",
        "Tom has 3 red balls and buys 4 blue balls how many balls total now?",
        "An unrelated geometry note about cyclic quadrilaterals.",
    ]
    benchmarks = [
        "Tom has 3 red balls and buys 4 blue balls how many balls total now?"
    ]
    clean = decontaminate_segments(raw_segments, benchmarks)

    rewards = [0.20, 0.90, 0.40, 0.90]
    lengths = [3, 3, 3, 3]
    outcome = outcome_advantages(rewards, lengths)

    process = process_advantages(
        step_rewards=[[0.2, 0.8], [0.7, 0.9]],
        step_end_tokens=[[1, 2], [1, 2]],
        token_lengths=[3, 3],
    )

    traces = [
        TokenPolicyTrace(
            old_logps=(-1.0, -1.2, -0.8),
            new_logps=(-0.9, -1.1, -0.9),
            ref_logps=(-1.1, -1.3, -0.9),
        )
        for _ in rewards
    ]
    objective = grpo_objective(traces, outcome)

    print(f"decontamination: {len(raw_segments)} -> {len(clean)} segments")
    print("group rewards:    ", rewards)
    print("outcome advantage:", [round(row[0], 4) for row in outcome])
    print("process advantage:", [[round(x, 4) for x in row] for row in process])
    print(f"GRPO objective:    {objective:.6f}")
    print(
        "Maj@4 / Pass@4:    ",
        majority_at_k(["7", "7", "8", "7"], "7"),
        pass_at_k(["6", "8", "7", "9"], "7"),
    )

    assert len(clean) == 2
    assert abs(sum(row[0] for row in outcome)) < 1e-8
    assert group_normalize([1.0, 1.0]) == [0.0, 0.0]
    assert positive_kl_estimator(-1.0, -1.0) == 0.0


if __name__ == "__main__":
    demo()
