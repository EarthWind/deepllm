#!/usr/bin/env python3
"""Speculative decoding 的零依赖教学实现。

它刻意把真实 Transformer 的一次并行 verification forward 抽象成
``target_distributions = [...]``。列表里有 gamma+1 个位置的分布，但
``target_rounds`` 只增加一次；真实系统会用 causal mask 在一次 target
forward 中同时得到这些 logits。

运行：
    python3 papers/to-2026/code/speculative_decoding_minimal.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Callable, Iterable, Mapping, Sequence


Distribution = dict[str, float]
Model = Callable[[Sequence[str]], Distribution]


def ordered_support(
    first: Mapping[object, float], second: Mapping[object, float]
) -> list[object]:
    """稳定的并集顺序，避免 Python hash 随机化改变演示输出。"""
    return list(dict.fromkeys((*first.keys(), *second.keys())))


def normalize(weights: Mapping[str, float]) -> Distribution:
    """把非负权重归一化成概率分布。"""
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("概率权重不能为负")
    total = math.fsum(weights.values())
    if total <= 0.0:
        raise ValueError("至少需要一个正权重")
    return {token: value / total for token, value in weights.items()}


def sample_categorical(distribution: Mapping[str, float], rng: random.Random) -> str:
    """不依赖 NumPy 的 categorical sampling。"""
    threshold = rng.random()
    cumulative = 0.0
    last_token = ""
    for token, probability in distribution.items():
        cumulative += probability
        last_token = token
        if threshold < cumulative:
            return token
    # 浮点累加可能略小于 1；最后一个 token 是安全回退。
    return last_token


def acceptance_probability(
    target: Mapping[str, float], draft: Mapping[str, float]
) -> float:
    """beta = sum_x min(p(x), q(x)) = 1 - TV(p, q)。"""
    vocabulary = ordered_support(target, draft)
    return math.fsum(min(target.get(x, 0.0), draft.get(x, 0.0)) for x in vocabulary)


def residual_distribution(
    target: Mapping[str, float], draft: Mapping[str, float]
) -> Distribution:
    """拒绝 draft token 后使用 norm(max(0, p-q)) 修正。"""
    vocabulary = ordered_support(target, draft)
    positive_part = {
        token: max(0.0, target.get(token, 0.0) - draft.get(token, 0.0))
        for token in vocabulary
    }
    return normalize(positive_part)


def speculative_sample_once(
    target: Distribution, draft: Distribution, rng: random.Random
) -> tuple[str, bool]:
    """单位置 speculative sampling；返回 (token, draft 是否被接受)。"""
    proposal = sample_categorical(draft, rng)
    q = draft[proposal]  # proposal 来自 q，因此 q > 0。
    accept = rng.random() < min(1.0, target.get(proposal, 0.0) / q)
    if accept:
        return proposal, True
    correction = sample_categorical(residual_distribution(target, draft), rng)
    return correction, False


@dataclass
class DecodeStats:
    target_rounds: int = 0
    target_positions: int = 0
    draft_calls: int = 0
    checked_proposals: int = 0
    accepted_proposals: int = 0
    discarded_proposals: int = 0

    @property
    def measured_acceptance(self) -> float:
        if self.checked_proposals == 0:
            return 0.0
        return self.accepted_proposals / self.checked_proposals


def speculative_round(
    prefix: Sequence[str],
    target_model: Model,
    draft_model: Model,
    gamma: int,
    rng: random.Random,
    stats: DecodeStats,
) -> list[str]:
    """完成一轮 draft -> parallel verify -> accept/correct。

    每轮一定提交至少 1 个 token；全部 gamma 个提案接受时，会再从目标
    模型的第 gamma+1 个分布采一个 bonus token。
    """
    if gamma < 1:
        raise ValueError("gamma 必须 >= 1")

    draft_tokens: list[str] = []
    draft_distributions: list[Distribution] = []
    running_prefix = list(prefix)

    # 1) 小模型串行提出 gamma 个候选。
    for _ in range(gamma):
        q = draft_model(running_prefix)
        proposal = sample_categorical(q, rng)
        draft_distributions.append(q)
        draft_tokens.append(proposal)
        running_prefix.append(proposal)
        stats.draft_calls += 1

    # 2) 教学写法是循环；真实 Transformer 把 gamma+1 个位置放进同一次
    #    causal target forward。故 target_rounds += 1，而不是 gamma+1。
    target_distributions = [
        target_model(list(prefix) + draft_tokens[:position])
        for position in range(gamma + 1)
    ]
    stats.target_rounds += 1
    stats.target_positions += gamma + 1

    # 3) 只能接受第一个拒绝点之前的连续前缀。
    committed: list[str] = []
    for position, proposal in enumerate(draft_tokens):
        p = target_distributions[position]
        q = draft_distributions[position]
        stats.checked_proposals += 1
        ratio = p.get(proposal, 0.0) / q[proposal]
        if rng.random() < min(1.0, ratio):
            committed.append(proposal)
            stats.accepted_proposals += 1
            continue

        # 4a) 首次拒绝：后续草稿作废，并从残差分布采修正 token。
        stats.discarded_proposals += gamma - position
        correction = sample_categorical(residual_distribution(p, q), rng)
        committed.append(correction)
        return committed

    # 4b) 全接受：目标 forward 已经给出了下一个位置的分布，免费多采一个。
    bonus = sample_categorical(target_distributions[gamma], rng)
    committed.append(bonus)
    return committed


def speculative_decode(
    prefix: Sequence[str],
    target_model: Model,
    draft_model: Model,
    gamma: int,
    max_new_tokens: int,
    rng: random.Random,
    eos_token: str | None = None,
) -> tuple[list[str], DecodeStats]:
    output = list(prefix)
    stats = DecodeStats()
    while len(output) - len(prefix) < max_new_tokens:
        block = speculative_round(output, target_model, draft_model, gamma, rng, stats)
        for token in block:
            if len(output) - len(prefix) >= max_new_tokens:
                break
            output.append(token)
            if eos_token is not None and token == eos_token:
                return output, stats
    return output, stats


def target_only_decode(
    prefix: Sequence[str],
    target_model: Model,
    max_new_tokens: int,
    rng: random.Random,
) -> list[str]:
    output = list(prefix)
    for _ in range(max_new_tokens):
        output.append(sample_categorical(target_model(output), rng))
    return output


def expected_tokens_per_round(alpha: float, gamma: int) -> float:
    if alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def expected_speedup(alpha: float, gamma: int, c: float) -> float:
    return expected_tokens_per_round(alpha, gamma) / (1.0 + gamma * c)


def total_variation(first: Mapping[object, float], second: Mapping[object, float]) -> float:
    support = ordered_support(first, second)
    return 0.5 * math.fsum(abs(first.get(x, 0.0) - second.get(x, 0.0)) for x in support)


def frequencies(samples: Iterable[object]) -> dict[object, float]:
    counter = Counter(samples)
    total = counter.total()
    return {item: count / total for item, count in counter.items()}


VOCABULARY = ("A", "B", "C", "D")
BASE_TARGET = (0.52, 0.25, 0.15, 0.08)


def _state(prefix: Sequence[str]) -> int:
    last = VOCABULARY.index(prefix[-1]) if prefix else 0
    return (len(prefix) + last) % len(VOCABULARY)


def toy_target(prefix: Sequence[str]) -> Distribution:
    state = _state(prefix)
    return {
        token: BASE_TARGET[(index - state) % len(VOCABULARY)]
        for index, token in enumerate(VOCABULARY)
    }


def toy_draft(prefix: Sequence[str]) -> Distribution:
    """比 target 便宜且接近，但并不相同的玩具草稿模型。"""
    p = toy_target(prefix)
    shifted = list(p.values())[1:] + list(p.values())[:1]
    return normalize(
        {
            token: 0.72 * p[token] + 0.28 * shifted[index]
            for index, token in enumerate(VOCABULARY)
        }
    )


def show_exactness_ledger() -> None:
    p = {"A": 0.50, "B": 0.30, "C": 0.20}
    q = {"A": 0.60, "B": 0.10, "C": 0.30}
    beta = acceptance_probability(p, q)
    residual = residual_distribution(p, q)

    print("[1] 单 token 精确性账本")
    print("token  accepted mass  correction mass  total  target")
    reconstructed: Distribution = {}
    for token in p:
        accepted_mass = min(p[token], q[token])
        correction_mass = (1.0 - beta) * residual.get(token, 0.0)
        reconstructed[token] = accepted_mass + correction_mass
        print(
            f"{token:>5}  {accepted_mass:13.3f}  {correction_mass:15.3f}"
            f"  {reconstructed[token]:5.3f}  {p[token]:6.3f}"
        )
    assert total_variation(p, reconstructed) < 1e-12

    trials = 100_000
    rng = random.Random(7)
    samples = [speculative_sample_once(p, q, rng)[0] for _ in range(trials)]
    empirical = frequencies(samples)
    empirical_tv = total_variation(p, empirical)
    print(f"beta={beta:.3f}, empirical TV({trials:,} samples)={empirical_tv:.4f}\n")
    assert empirical_tv < 0.01


def show_sequence_distribution() -> None:
    """比较长度为 3 的完整序列分布，而非只检查第一个 token。"""
    trials = 50_000
    baseline_rng = random.Random(11)
    speculative_rng = random.Random(29)
    baseline: list[tuple[str, ...]] = []
    speculative: list[tuple[str, ...]] = []

    for _ in range(trials):
        baseline.append(tuple(target_only_decode([], toy_target, 3, baseline_rng)))
        output, _ = speculative_decode([], toy_target, toy_draft, 3, 3, speculative_rng)
        speculative.append(tuple(output))

    tv = total_variation(frequencies(baseline), frequencies(speculative))
    print("[2] 自回归序列分布 Monte Carlo 校验")
    print(f"length=3, trials={trials:,}, empirical TV={tv:.4f}\n")
    assert tv < 0.03


def show_cost_model() -> None:
    alpha = acceptance_probability(toy_target([]), toy_draft([]))
    c = 0.05
    candidates = [(gamma, expected_speedup(alpha, gamma, c)) for gamma in range(1, 11)]
    best_gamma, best_speedup = max(candidates, key=lambda item: item[1])

    print("[3] 论文中的简化性能模型")
    print(f"alpha={alpha:.3f}, draft/target cost c={c:.2f}")
    for gamma, speedup in candidates:
        expected = expected_tokens_per_round(alpha, gamma)
        marker = "  <-- best" if gamma == best_gamma else ""
        print(f"gamma={gamma:2d}: E[tokens]={expected:.3f}, speedup={speedup:.3f}x{marker}")
    print(f"best gamma={best_gamma}, predicted speedup={best_speedup:.3f}x\n")


def show_one_decode() -> None:
    max_new_tokens = 32
    output, stats = speculative_decode(
        [], toy_target, toy_draft, gamma=4, max_new_tokens=max_new_tokens, rng=random.Random(42)
    )
    print("[4] 一次玩具解码")
    print("output:", " ".join(output))
    print(f"naive target serial rounds : {max_new_tokens}")
    print(f"speculative target rounds  : {stats.target_rounds}")
    print(f"target positions evaluated : {stats.target_positions}")
    print(f"draft calls                : {stats.draft_calls}")
    print(f"checked/accepted proposals : {stats.checked_proposals}/{stats.accepted_proposals}")
    print(f"measured acceptance        : {stats.measured_acceptance:.3f}")
    print(f"discarded draft proposals  : {stats.discarded_proposals}")
    assert len(output) == max_new_tokens
    assert stats.target_rounds <= max_new_tokens


def main() -> None:
    show_exactness_ledger()
    show_sequence_distribution()
    show_cost_model()
    show_one_decode()


if __name__ == "__main__":
    main()
