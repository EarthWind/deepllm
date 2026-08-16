#!/usr/bin/env python3
"""Kimi k1.5 长上下文强化学习的零依赖教学实现。

对应《Kimi k1.5: Scaling Reinforcement Learning with LLMs》公开的几项机制：

1. 用 10 次高温采样 pass rate 估计相对模型难度；
2. 按 ``1 - success_rate`` 做 prioritized prompt sampling；
3. 论文公式 (3) 的 group-centered mirror-descent proxy；
4. 只惩罚冗长、不奖励错误短答的 group-relative length reward；
5. 固定分段预算的 partial rollout，以及旧前缀可从 loss 排除的语义；
6. shortest rejection sampling、长度偏好 DPO pair 和权重平均；
7. 代码测试用例的 7/10 共识与 9/10 题目接纳门槛。

它不是 Moonshot AI 官方实现，也不训练或调用真实模型。官方报告未公开模型
参数量、checkpoint、完整 RL 数据或训练代码；这里仅复现可由论文公式独立验证
的调度和目标函数算术。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Mapping, Sequence


def difficulty_from_outcomes(outcomes: Sequence[bool]) -> float:
    """返回 [0, 1] 难度：difficulty = 1 - pass_rate。

    论文用 SFT model 对每题做 10 次较高温采样；这里允许任意非空次数，
    便于测试，但 demo 使用 10 次。
    """

    if not outcomes:
        raise ValueError("outcomes must not be empty")
    pass_rate = sum(outcomes) / len(outcomes)
    return 1.0 - pass_rate


def is_easy_to_hack(
    no_cot_guesses: Sequence[Hashable],
    ground_truth: Hashable,
    *,
    attempts: int = 8,
) -> bool:
    """无 CoT 猜测在前 N 次内命中，就视作容易 hack。

    论文实验采用 N=8，并排除多选、判断、证明题等难以可靠做 outcome
    verification 的题型。
    """

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    return ground_truth in no_cot_guesses[:attempts]


def prioritized_sampling_probabilities(
    success_rates: Sequence[float],
) -> tuple[float, ...]:
    """论文 prioritized sampling：p_i ∝ 1 - s_i。

    如果所有题当前都 100% 成功，论文公式的分母为 0；教学实现退化为均匀
    分布，避免调度器崩溃。这是工程兜底，不是论文新增结论。
    """

    if not success_rates:
        raise ValueError("success_rates must not be empty")
    if any(not 0.0 <= rate <= 1.0 for rate in success_rates):
        raise ValueError("success rates must lie in [0, 1]")

    weights = tuple(1.0 - rate for rate in success_rates)
    total = sum(weights)
    if total == 0.0:
        uniform = 1.0 / len(weights)
        return tuple(uniform for _ in weights)
    return tuple(weight / total for weight in weights)


def curriculum_indices(
    difficulties: Sequence[float],
    *,
    warmup: bool,
    hard_threshold: float = 0.6,
) -> tuple[int, ...]:
    """warmup 使用全部题；之后只保留 hard prompts 的最小课程示例。"""

    if any(not 0.0 <= value <= 1.0 for value in difficulties):
        raise ValueError("difficulties must lie in [0, 1]")
    if not 0.0 <= hard_threshold <= 1.0:
        raise ValueError("hard_threshold must lie in [0, 1]")
    if warmup:
        return tuple(range(len(difficulties)))
    return tuple(
        index
        for index, difficulty in enumerate(difficulties)
        if difficulty >= hard_threshold
    )


def length_rewards(
    correctness: Sequence[bool],
    lengths: Sequence[int],
) -> tuple[float, ...]:
    """实现论文 Section 2.3.3 的 group-relative length reward。

    lambda_i = 0.5 - (len_i - min_len) / (max_len - min_len)

    - 正确回答取得完整 lambda，因此短正确为正、长正确可为负；
    - 错误回答取得 min(0, lambda)，所以错误短答不会因“短”获得正奖励；
    - 组内等长时全部为 0。
    """

    if not correctness or len(correctness) != len(lengths):
        raise ValueError("correctness and lengths must have equal non-zero size")
    if any(length <= 0 for length in lengths):
        raise ValueError("lengths must be positive")

    min_len = min(lengths)
    max_len = max(lengths)
    if min_len == max_len:
        return tuple(0.0 for _ in lengths)

    rewards: list[float] = []
    span = max_len - min_len
    for correct, length in zip(correctness, lengths):
        relative = 0.5 - (length - min_len) / span
        rewards.append(relative if correct else min(0.0, relative))
    return tuple(rewards)


def combined_outcome_rewards(
    correctness: Sequence[bool],
    lengths: Sequence[int],
    *,
    length_weight: float,
) -> tuple[float, ...]:
    """把 binary outcome reward 与加权长度奖励相加。"""

    if length_weight < 0.0:
        raise ValueError("length_weight must be non-negative")
    extra = length_rewards(correctness, lengths)
    return tuple(
        float(correct) + length_weight * length_reward
        for correct, length_reward in zip(correctness, extra)
    )


def mirror_descent_target_probabilities(
    reference_probabilities: Sequence[float],
    rewards: Sequence[float],
    *,
    tau: float,
) -> tuple[float, ...]:
    """论文公式 (2) 的离散 closed-form target。

    pi*(j) ∝ pi_ref(j) * exp(reward_j / tau)
    """

    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if not reference_probabilities or len(reference_probabilities) != len(rewards):
        raise ValueError("probabilities and rewards must have equal non-zero size")
    if any(probability < 0.0 for probability in reference_probabilities):
        raise ValueError("probabilities must be non-negative")
    reference_total = sum(reference_probabilities)
    if not math.isclose(reference_total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("reference probabilities must sum to one")

    logits = [
        (-math.inf if probability == 0.0 else math.log(probability))
        + reward / tau
        for probability, reward in zip(reference_probabilities, rewards)
    ]
    maximum = max(logits)
    unnormalized = [
        0.0 if logit == -math.inf else math.exp(logit - maximum)
        for logit in logits
    ]
    normalizer = sum(unnormalized)
    return tuple(value / normalizer for value in unnormalized)


def mirror_descent_proxy_loss(
    current_logprobs: Sequence[float],
    reference_logprobs: Sequence[float],
    rewards: Sequence[float],
    *,
    tau: float,
) -> float:
    """产生论文公式 (3) 梯度的可读 proxy loss。

    Samples 来自每轮冻结的 reference policy。对每个 prompt 的 k 个回答，用
    reward mean 作 baseline：

      objective_j = (r_j-r_bar)*log pi - tau/2*(log pi-log pi_ref)^2

    返回其负均值，供梯度下降。真实训练会按 token、batch、并行策略处理；
    此函数使用整条序列 log-prob，只演示公开公式。
    """

    size = len(rewards)
    if size == 0 or len(current_logprobs) != size or len(reference_logprobs) != size:
        raise ValueError("all inputs must have equal non-zero size")
    if tau < 0.0:
        raise ValueError("tau must be non-negative")

    reward_mean = sum(rewards) / size
    objective = 0.0
    for current, reference, reward in zip(
        current_logprobs, reference_logprobs, rewards
    ):
        advantage = reward - reward_mean
        log_ratio = current - reference
        objective += advantage * current - 0.5 * tau * log_ratio**2
    return -objective / size


@dataclass(frozen=True)
class RolloutSegment:
    start: int
    end: int
    on_policy: bool

    @property
    def token_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PartialRollout:
    trajectory_id: str
    target_tokens: int
    generated_tokens: int = 0

    @property
    def complete(self) -> bool:
        return self.generated_tokens >= self.target_tokens


def advance_partial_rollout(
    state: PartialRollout,
    *,
    segment_budget: int,
) -> tuple[PartialRollout, tuple[RolloutSegment, ...]]:
    """复用旧前缀，只生成不超过 segment_budget 的新后缀。

    返回的 segments 显式标出旧前缀 ``on_policy=False``、本轮新段为 True，
    对应论文“某些旧 segment 可排除在 loss 之外”的语义。
    """

    if segment_budget <= 0 or state.target_tokens <= 0:
        raise ValueError("budgets and target length must be positive")
    if not 0 <= state.generated_tokens <= state.target_tokens:
        raise ValueError("generated_tokens is outside the trajectory")
    if state.complete:
        return state, (RolloutSegment(0, state.target_tokens, False),)

    new_end = min(state.target_tokens, state.generated_tokens + segment_budget)
    segments: list[RolloutSegment] = []
    if state.generated_tokens:
        segments.append(RolloutSegment(0, state.generated_tokens, False))
    segments.append(RolloutSegment(state.generated_tokens, new_end, True))
    return (
        PartialRollout(state.trajectory_id, state.target_tokens, new_end),
        tuple(segments),
    )


def partial_rollout_cost(target_tokens: int) -> int:
    """只生成每个 token 一次时的 generation token 数。"""

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    return target_tokens


def regenerate_from_scratch_cost(target_tokens: int, segment_budget: int) -> int:
    """对照项：每轮从头生成到当前边界所需 token 数。"""

    if target_tokens <= 0 or segment_budget <= 0:
        raise ValueError("budgets must be positive")
    cost = 0
    boundary = segment_budget
    while boundary < target_tokens:
        cost += boundary
        boundary += segment_budget
    return cost + target_tokens


@dataclass(frozen=True)
class Response:
    name: str
    tokens: int
    correct: bool


def shortest_correct_response(responses: Sequence[Response]) -> Response:
    """论文 shortest rejection sampling 的核心选择规则。"""

    correct = [response for response in responses if response.correct]
    if not correct:
        raise ValueError("at least one correct response is required")
    return min(correct, key=lambda response: (response.tokens, response.name))


def length_preference_pairs(
    responses: Sequence[Response],
    *,
    correct_negative_ratio: float = 1.5,
) -> tuple[tuple[Response, Response], ...]:
    """用最短正确回答构造论文 long2short DPO pair。

    负样本包括比 chosen 更长的错误回答，以及长度至少为 chosen 1.5 倍的
    正确回答。
    """

    if correct_negative_ratio <= 1.0:
        raise ValueError("correct_negative_ratio must exceed one")
    chosen = shortest_correct_response(responses)
    rejected = [
        response
        for response in responses
        if response.name != chosen.name
        and response.tokens > chosen.tokens
        and (
            not response.correct
            or response.tokens >= correct_negative_ratio * chosen.tokens
        )
    ]
    return tuple((chosen, response) for response in rejected)


def average_model_weights(
    long_weights: Sequence[float],
    short_weights: Sequence[float],
) -> tuple[float, ...]:
    """论文 model merging 的逐元素简单平均示意。"""

    if not long_weights or len(long_weights) != len(short_weights):
        raise ValueError("weight vectors must have equal non-zero size")
    return tuple((left + right) / 2.0 for left, right in zip(long_weights, short_weights))


def accepted_generated_test_cases(
    outputs_by_case: Sequence[Sequence[Hashable]],
    *,
    case_consensus: int = 7,
) -> tuple[int, ...]:
    """10 份 ground-truth submissions 中至少 7 份输出一致才保留测试。"""

    accepted: list[int] = []
    for case_index, outputs in enumerate(outputs_by_case):
        if not outputs:
            continue
        counts: dict[Hashable, int] = {}
        for output in outputs:
            counts[output] = counts.get(output, 0) + 1
        if max(counts.values()) >= case_consensus:
            accepted.append(case_index)
    return tuple(accepted)


def accept_problem(full_suite_passes: Sequence[bool], *, threshold: int = 9) -> bool:
    """10 份 ground-truth submissions 中至少 9 份通过整套测试。"""

    if not full_suite_passes:
        raise ValueError("full_suite_passes must not be empty")
    return sum(full_suite_passes) >= threshold


def demo() -> None:
    outcomes = (True, False, False, True, False, False, False, False, False, False)
    difficulty = difficulty_from_outcomes(outcomes)
    priorities = prioritized_sampling_probabilities((0.9, 0.4, 0.1))

    correctness = (True, True, False, False)
    lengths = (1_000, 2_000, 1_200, 3_000)
    length_bonus = length_rewards(correctness, lengths)
    rewards = combined_outcome_rewards(correctness, lengths, length_weight=0.2)
    proxy = mirror_descent_proxy_loss(
        current_logprobs=(-2.0, -2.4, -3.1, -3.8),
        reference_logprobs=(-2.2, -2.3, -3.0, -3.6),
        rewards=rewards,
        tau=0.1,
    )

    target = 13_000
    budget = 4_096
    state = PartialRollout("math-trajectory", target)
    rounds = 0
    while not state.complete:
        state, segments = advance_partial_rollout(state, segment_budget=budget)
        rounds += 1
        assert sum(segment.token_count for segment in segments if segment.on_policy) <= budget

    partial_cost = partial_rollout_cost(target)
    naive_cost = regenerate_from_scratch_cost(target, budget)

    responses = (
        Response("long-correct", 6_000, True),
        Response("short-correct", 3_200, True),
        Response("short-wrong", 2_500, False),
        Response("long-wrong", 5_000, False),
    )
    chosen = shortest_correct_response(responses)
    pairs = length_preference_pairs(responses)

    print("Kimi k1.5 disclosed-mechanism arithmetic:")
    print(f"  prompt difficulty from 10 samples = {difficulty:.1%}")
    print("  prioritized probabilities          =", tuple(round(x, 3) for x in priorities))
    print("  group-relative length rewards      =", tuple(round(x, 3) for x in length_bonus))
    print(f"  mirror-descent proxy loss          = {proxy:.4f}")
    print(f"  partial rollout rounds             = {rounds}")
    print(f"  generated tokens: partial / naive  = {partial_cost} / {naive_cost}")
    print(f"  avoided repeated-prefix tokens     = {naive_cost - partial_cost}")
    print(f"  shortest correct response          = {chosen.name} ({chosen.tokens} tokens)")
    print("  length-DPO rejected responses      =", tuple(pair[1].name for pair in pairs))

    assert math.isclose(difficulty, 0.8)
    assert math.isclose(sum(priorities), 1.0)
    assert length_bonus == (0.5, 0.0, 0.0, -0.5)
    assert rounds == 4
    assert naive_cost == 37_576
    assert chosen.name == "short-correct"
    assert tuple(pair[1].name for pair in pairs) == ("long-correct", "long-wrong")
    assert is_easy_to_hack(("A", "B", "42"), "42")
    assert accepted_generated_test_cases(((1,) * 7 + (2,) * 3, (1,) * 6 + (2,) * 4)) == (0,)
    assert accept_problem((True,) * 9 + (False,))


if __name__ == "__main__":
    demo()
