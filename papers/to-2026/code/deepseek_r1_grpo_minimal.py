#!/usr/bin/env python3
"""DeepSeek-R1 / GRPO 的零依赖教学实现。

这个脚本只演示论文中的关键数学与数据流：

1. 同一问题采样一组回答；
2. 用可验证规则计算 accuracy / format reward；
3. 在组内标准化 reward，得到 group-relative advantage；
4. 计算 clipped surrogate 与论文给出的 KL 估计；
5. 用 rejection sampling 留下可供后续 SFT 的正确、可读样本。

它不是官方训练代码：一个回答在这里用一个聚合 log-probability 表示，真实系统会
处理每个 token、padding mask、分布式 rollout、旧策略快照与优化器更新。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Completion:
    """一个 rollout，以及三个策略在该 rollout 上的聚合 log probability。"""

    text: str
    old_logp: float
    new_logp: float
    ref_logp: float


@dataclass(frozen=True)
class ScoredCompletion:
    completion: Completion
    accuracy: float
    format: float
    reward: float
    advantage: float


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def extract_answer(text: str) -> str | None:
    """抽取教学模板中的最终答案；生产 verifier 需要更严格的规范化。"""

    match = ANSWER_RE.search(text)
    return match.group(1).strip() if match else None


def exact_match_reward(text: str, expected_answer: str) -> float:
    """最小 accuracy verifier：规范化空白后做精确匹配。"""

    answer = extract_answer(text)
    if answer is None:
        return 0.0
    normalize = lambda value: " ".join(value.split()).casefold()
    return float(normalize(answer) == normalize(expected_answer))


def format_reward(text: str) -> float:
    """检查 reasoning 在 answer 之前，且两段都非空。"""

    think = THINK_RE.search(text)
    answer = ANSWER_RE.search(text)
    valid = (
        think is not None
        and answer is not None
        and bool(think.group(1).strip())
        and bool(answer.group(1).strip())
        and think.end() <= answer.start()
    )
    return float(valid)


def group_relative_advantages(
    rewards: Sequence[float], epsilon: float = 1e-8
) -> list[float]:
    """A_i = (r_i - mean(r)) / std(r)，零方差组安全返回全 0。"""

    if not rewards:
        raise ValueError("rewards must not be empty")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std < epsilon:
        return [0.0 for _ in rewards]
    return [(reward - mean) / std for reward in rewards]


def paper_kl_estimator(new_logp: float, ref_logp: float) -> float:
    """论文式非负 KL 项：x - log(x) - 1，其中 x = pi_ref / pi_theta。"""

    log_ratio_ref_over_policy = ref_logp - new_logp
    ratio_ref_over_policy = math.exp(log_ratio_ref_over_policy)
    return ratio_ref_over_policy - log_ratio_ref_over_policy - 1.0


def clipped_surrogate(
    completion: Completion,
    advantage: float,
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
) -> float:
    """单个 rollout 的 GRPO surrogate；训练时应最大化返回值。"""

    ratio = math.exp(completion.new_logp - completion.old_logp)
    clipped_ratio = min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
    policy_term = min(ratio * advantage, clipped_ratio * advantage)
    return policy_term - beta * paper_kl_estimator(
        completion.new_logp, completion.ref_logp
    )


def score_group(
    completions: Sequence[Completion],
    expected_answer: str,
    accuracy_weight: float = 1.0,
    format_weight: float = 1.0,
) -> list[ScoredCompletion]:
    """用 R1-Zero 风格的两个规则信号给一组 rollout 打分。"""

    components: list[tuple[float, float, float]] = []
    for completion in completions:
        accuracy = exact_match_reward(completion.text, expected_answer)
        fmt = format_reward(completion.text)
        total = accuracy_weight * accuracy + format_weight * fmt
        components.append((accuracy, fmt, total))

    advantages = group_relative_advantages([item[2] for item in components])
    return [
        ScoredCompletion(completion, accuracy, fmt, total, advantage)
        for completion, (accuracy, fmt, total), advantage in zip(
            completions, components, advantages
        )
    ]


def grpo_group_objective(
    scored: Sequence[ScoredCompletion],
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
) -> float:
    """组内取平均；优化器使用 loss = -objective。"""

    if not scored:
        raise ValueError("scored group must not be empty")
    values = [
        clipped_surrogate(item.completion, item.advantage, clip_epsilon, beta)
        for item in scored
    ]
    return sum(values) / len(values)


def rejection_sample(
    completions: Iterable[Completion],
    expected_answer: str,
    extra_filter: Callable[[str], bool] | None = None,
) -> list[Completion]:
    """保留答案正确、结构可读的样本，模拟 R1 后续 SFT 数据筛选。"""

    accepted = []
    for completion in completions:
        valid = (
            exact_match_reward(completion.text, expected_answer) == 1.0
            and format_reward(completion.text) == 1.0
        )
        if valid and (extra_filter is None or extra_filter(completion.text)):
            accepted.append(completion)
    return accepted


def demo() -> None:
    prompt = "What is 12 * 7?"
    expected = "84"
    rollouts = [
        Completion(
            "<think>12 * 7 = 84.</think><answer>84</answer>",
            old_logp=-2.10,
            new_logp=-2.00,
            ref_logp=-2.25,
        ),
        Completion(
            "The answer is 84.",
            old_logp=-1.85,
            new_logp=-1.90,
            ref_logp=-1.80,
        ),
        Completion(
            "<think>12 * 7 = 82.</think><answer>82</answer>",
            old_logp=-2.30,
            new_logp=-2.20,
            ref_logp=-2.25,
        ),
        Completion(
            "<think>I add twelve seven times: 84.</think><answer>84</answer>",
            old_logp=-2.45,
            new_logp=-2.32,
            ref_logp=-2.50,
        ),
    ]

    scored = score_group(rollouts, expected)
    objective = grpo_group_objective(scored)
    accepted = rejection_sample(rollouts, expected)

    print(f"prompt: {prompt}")
    print("idx  acc  fmt  reward  advantage  answer")
    for index, item in enumerate(scored):
        answer = extract_answer(item.completion.text) or "<missing>"
        print(
            f"{index:>3}  {item.accuracy:>3.0f}  {item.format:>3.0f}"
            f"  {item.reward:>6.2f}  {item.advantage:>9.4f}  {answer}"
        )
    print(f"group objective: {objective:.6f}")
    print(f"accepted for SFT: {len(accepted)}/{len(rollouts)}")

    assert abs(sum(item.advantage for item in scored)) < 1e-9
    assert group_relative_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    assert len(accepted) == 2
    assert all(math.isfinite(paper_kl_estimator(-2.0, ref)) for ref in (-2.5, -2.0, -1.5))


if __name__ == "__main__":
    demo()
