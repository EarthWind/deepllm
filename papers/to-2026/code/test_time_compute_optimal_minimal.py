#!/usr/bin/env python3
"""Compute-optimal test-time scaling 的零依赖教学实现。

对应 Snell et al. (2024) 的几个核心机制：

1. 用模型 pass@1 或 verifier 平均分，把问题划入 5 个难度分位；
2. 对相同最终答案累加 verifier 分数，实现 best-of-N weighted；
3. 在验证数据上为每个 (difficulty, budget) 选择最优策略；
4. 用两折交叉验证避免在同一批问题上选策略又报分；
5. 分解串行 revision 与并行 sampling 的固定生成预算；
6. 计算论文中预训练 FLOPs 与推理 FLOPs 的交换倍率。

它不是论文官方实现，也不会调用真实语言模型或 PRM。demo 中的 correctness 是
确定性构造的玩具数据，只用来检查路由和预算数学，不能当作论文实验结果。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Candidate:
    """一条完整候选答案，以及 verifier 对它的最终分数。"""

    answer: str
    verifier_score: float
    steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.verifier_score <= 1.0:
            raise ValueError("verifier_score must be in [0, 1]")


@dataclass(frozen=True)
class StrategyRecord:
    """一个问题在固定预算、固定策略下是否答对。"""

    question_id: str
    difficulty: int
    budget: int
    strategy: str
    correct: bool

    def __post_init__(self) -> None:
        if not 1 <= self.difficulty <= 5:
            raise ValueError("difficulty must be an integer from 1 to 5")
        if self.budget <= 0:
            raise ValueError("budget must be positive")


@dataclass(frozen=True)
class ComputeOptimalPolicy:
    """按 (难度, 预算) 查表的 prompt-adaptive 推理策略。"""

    table: Mapping[tuple[int, int], str]

    def choose(self, difficulty: int, budget: int) -> str:
        if not 1 <= difficulty <= 5:
            raise ValueError("difficulty must be an integer from 1 to 5")
        exact = self.table.get((difficulty, budget))
        if exact is not None:
            return exact

        available = [
            known_budget
            for known_difficulty, known_budget in self.table
            if known_difficulty == difficulty
        ]
        if not available:
            raise KeyError(f"no strategy learned for difficulty {difficulty}")
        nearest = min(available, key=lambda known: (abs(known - budget), known))
        return self.table[(difficulty, nearest)]


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def monte_carlo_step_value(rollout_successes: Sequence[bool]) -> float:
    """PRM 的 soft target：从某一步继续 rollout 后的成功比例。"""

    return mean([float(success) for success in rollout_successes])


def binary_cross_entropy(target: float, prediction: float) -> float:
    """论文附录中的 soft-label PRM 损失。"""

    if not 0.0 <= target <= 1.0:
        raise ValueError("target must be in [0, 1]")
    if not 0.0 < prediction < 1.0:
        raise ValueError("prediction must be strictly inside (0, 1)")
    return -(target * math.log(prediction) + (1.0 - target) * math.log(1.0 - prediction))


def weighted_best_of_n(candidates: Sequence[Candidate]) -> Candidate:
    """按最终答案聚合 verifier 质量，再返回获胜答案中的最高分候选。

    论文的 best-of-N weighted 对相同 final answer 的 verifier 分数求和，选择总分
    最高的答案，而不是直接选择单条分数最高的轨迹。
    """

    if not candidates:
        raise ValueError("candidates must not be empty")

    totals: dict[str, float] = defaultdict(float)
    for candidate in candidates:
        totals[candidate.answer] += candidate.verifier_score

    winner = max(totals, key=lambda answer: (totals[answer], answer))
    return max(
        (candidate for candidate in candidates if candidate.answer == winner),
        key=lambda candidate: candidate.verifier_score,
    )


def majority_vote(candidates: Sequence[Candidate]) -> str:
    """对所有轨迹扁平多数投票；平票时使用 verifier 总分。"""

    if not candidates:
        raise ValueError("candidates must not be empty")
    counts: dict[str, int] = defaultdict(int)
    scores: dict[str, float] = defaultdict(float)
    for candidate in candidates:
        counts[candidate.answer] += 1
        scores[candidate.answer] += candidate.verifier_score
    return max(counts, key=lambda answer: (counts[answer], scores[answer], answer))


def hierarchical_revision_selection(chains: Sequence[Sequence[Candidate]]) -> Candidate:
    """先在每条 revision chain 内选，再跨 chain 做第二次 weighted 选择。"""

    if not chains or any(not chain for chain in chains):
        raise ValueError("every revision chain must contain at least one candidate")
    chain_winners = [weighted_best_of_n(chain) for chain in chains]
    return weighted_best_of_n(chain_winners)


def difficulty_quintiles(question_values: Mapping[str, float]) -> dict[str, int]:
    """把模型成功率 / verifier 均分从高到低划成五个等频难度桶。

    level 1 最容易，level 5 最难。相同 value 在教学实现中可能跨桶；生产实现应
    明确 tie policy，并在版本化数据上固化边界。
    """

    if not question_values:
        raise ValueError("question_values must not be empty")
    if any(not 0.0 <= value <= 1.0 for value in question_values.values()):
        raise ValueError("all question values must be in [0, 1]")

    ordered = sorted(question_values, key=lambda qid: (-question_values[qid], qid))
    size = len(ordered)
    return {
        question_id: min(5, (rank * 5) // size + 1)
        for rank, question_id in enumerate(ordered)
    }


def fit_compute_optimal_policy(records: Iterable[StrategyRecord]) -> ComputeOptimalPolicy:
    """每个 difficulty × budget 单元格选择验证准确率最高的策略。"""

    grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record.difficulty, record.budget, record.strategy)].append(
            float(record.correct)
        )
    if not grouped:
        raise ValueError("records must not be empty")

    cells: dict[tuple[int, int], list[tuple[float, str]]] = defaultdict(list)
    for (difficulty, budget, strategy), outcomes in grouped.items():
        cells[(difficulty, budget)].append((mean(outcomes), strategy))

    table = {
        cell: max(options, key=lambda item: (item[0], item[1]))[1]
        for cell, options in cells.items()
    }
    return ComputeOptimalPolicy(table=table)


def stable_two_fold(question_id: str) -> int:
    """无需随机库的稳定两折划分。"""

    return sum((index + 1) * ord(char) for index, char in enumerate(question_id)) % 2


def two_fold_policy_accuracy(records: Sequence[StrategyRecord]) -> float:
    """在一折选路由策略，在另一折按该策略报准确率，再交换。

    每个 held-out 问题应为各候选 strategy 都有一条记录，否则无法公平路由。
    """

    if not records:
        raise ValueError("records must not be empty")
    heldout_scores: list[float] = []

    for heldout_fold in (0, 1):
        train = [r for r in records if stable_two_fold(r.question_id) != heldout_fold]
        test = [r for r in records if stable_two_fold(r.question_id) == heldout_fold]
        policy = fit_compute_optimal_policy(train)
        lookup = {
            (r.question_id, r.budget, r.strategy): r.correct
            for r in test
        }
        questions = sorted({(r.question_id, r.difficulty, r.budget) for r in test})
        for question_id, difficulty, budget in questions:
            strategy = policy.choose(difficulty, budget)
            key = (question_id, budget, strategy)
            if key not in lookup:
                raise KeyError(f"missing held-out result for {key}")
            heldout_scores.append(float(lookup[key]))

    return mean(heldout_scores)


def sequential_parallel_layout(total_budget: int, chain_length: int) -> tuple[int, int]:
    """把 N 次生成排成 parallel_chains × sequential_steps。

    chain_length=1 等价于纯并行；chain_length=total_budget 等价于一条纯串行链。
    """

    if total_budget <= 0 or chain_length <= 0:
        raise ValueError("budgets must be positive")
    if total_budget % chain_length != 0:
        raise ValueError("chain_length must divide total_budget exactly")
    parallel_chains = total_budget // chain_length
    return parallel_chains, chain_length


def lookahead_generation_cost(beams: int, lookahead_steps: int) -> int:
    """论文的生成预算近似：N × (k + 1)。"""

    if beams <= 0 or lookahead_steps < 0:
        raise ValueError("beams must be positive and lookahead_steps non-negative")
    return beams * (lookahead_steps + 1)


def pretraining_flops(parameters: float, pretraining_tokens: float) -> float:
    return 6.0 * parameters * pretraining_tokens


def inference_flops(parameters: float, inference_tokens: float) -> float:
    return 2.0 * parameters * inference_tokens


def matched_small_model_inference_multiplier(
    parameter_multiplier: float, inference_to_pretraining_ratio: float
) -> float:
    """小模型可增加多少倍推理计算，才与 M 倍大模型的总 FLOPs 相同。

    由 X=6ND_pre、Y=2ND_inf 和 X+T·Y=M(X+Y) 得：
    T = M + 3(D_pre / D_inf)(M-1) = M + 3(M-1)/R。
    """

    if parameter_multiplier <= 1.0:
        raise ValueError("parameter_multiplier must be greater than 1")
    if inference_to_pretraining_ratio <= 0.0:
        raise ValueError("inference_to_pretraining_ratio must be positive")
    return parameter_multiplier + 3.0 * (parameter_multiplier - 1.0) / (
        inference_to_pretraining_ratio
    )


def make_toy_records(budget: int = 64) -> list[StrategyRecord]:
    """生成确定性的演示数据；目标正确率不是论文数值。"""

    target_accuracy = {
        1: {"parallel": 0.85, "beam": 0.70, "revision": 0.95},
        2: {"parallel": 0.75, "beam": 0.65, "revision": 0.85},
        3: {"parallel": 0.55, "beam": 0.75, "revision": 0.65},
        4: {"parallel": 0.35, "beam": 0.55, "revision": 0.45},
        5: {"parallel": 0.15, "beam": 0.10, "revision": 0.10},
    }
    records: list[StrategyRecord] = []
    for difficulty in range(1, 6):
        for question_index in range(40):
            question_id = f"d{difficulty}-q{question_index:02d}"
            for strategy_index, strategy in enumerate(("parallel", "beam", "revision")):
                fraction = ((question_index * 17 + strategy_index * 7) % 40) / 40.0
                correct = fraction < target_accuracy[difficulty][strategy]
                records.append(
                    StrategyRecord(
                        question_id=question_id,
                        difficulty=difficulty,
                        budget=budget,
                        strategy=strategy,
                        correct=correct,
                    )
                )
    return records


def demo() -> None:
    candidates = [
        Candidate(answer="12", verifier_score=0.91),
        Candidate(answer="13", verifier_score=0.63),
        Candidate(answer="13", verifier_score=0.62),
    ]
    winner = weighted_best_of_n(candidates)

    values = {f"q{i}": value for i, value in enumerate((0.95, 0.8, 0.7, 0.5, 0.3, 0.1))}
    bins = difficulty_quintiles(values)

    records = make_toy_records()
    policy = fit_compute_optimal_policy(records)
    cross_validated_accuracy = two_fold_policy_accuracy(records)

    print("weighted winner:", winner.answer, "(single-best score is 12)")
    print("difficulty bins:", bins)
    print("toy compute-optimal routing at budget=64:")
    for difficulty in range(1, 6):
        print(f"  difficulty {difficulty}: {policy.choose(difficulty, 64)}")
    print(f"two-fold toy accuracy: {cross_validated_accuracy:.3f}")
    print("parallel × sequential layouts for N=64:")
    for chain_length in (1, 4, 8, 64):
        print(" ", sequential_parallel_layout(64, chain_length))
    print("14x model FLOPs-matched small-model inference multipliers:")
    for ratio in (0.16, 0.79, 22.0):
        multiplier = matched_small_model_inference_multiplier(14.0, ratio)
        print(f"  R={ratio:>5}: {multiplier:7.2f}x")

    assert winner.answer == "13"
    assert lookahead_generation_cost(16, 3) == 64
    assert sequential_parallel_layout(64, 8) == (8, 8)
    assert math.isclose(monte_carlo_step_value([True, False, True, True]), 0.75)
    assert math.isclose(
        matched_small_model_inference_multiplier(14.0, 0.16),
        257.75,
    )


if __name__ == "__main__":
    demo()
