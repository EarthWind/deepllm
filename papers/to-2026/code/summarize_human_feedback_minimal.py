#!/usr/bin/env python3
"""Dependency-free checks for Learning to Summarize from Human Feedback.

This is not a language-model trainer. It isolates the small pieces of math that
are easiest to lose inside a full RLHF system:

1. pairwise Bradley--Terry reward-model loss;
2. sampled token-level KL shaping plus the terminal reward-model score;
3. generalized advantage estimation (GAE);
4. PPO's clipped policy objective; and
5. best-of-N selection as a training-free baseline.

Run:
    python3 papers/to-2026/code/summarize_human_feedback_minimal.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def _require_same_length(*sequences: Sequence[object]) -> None:
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError(f"sequences must have equal length, got {sorted(lengths)}")


def log_sigmoid(value: float) -> float:
    """Numerically stable log(sigmoid(value))."""

    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def pairwise_reward_loss(
    preferred_rewards: Sequence[float],
    rejected_rewards: Sequence[float],
) -> float:
    """Mean -log(sigmoid(r_preferred - r_rejected))."""

    _require_same_length(preferred_rewards, rejected_rewards)
    if not preferred_rewards:
        raise ValueError("at least one preference pair is required")
    losses = [
        -log_sigmoid(preferred - rejected)
        for preferred, rejected in zip(preferred_rewards, rejected_rewards)
    ]
    return sum(losses) / len(losses)


def preference_probability(reward_a: float, reward_b: float) -> float:
    """Bradley--Terry probability that summary A is preferred to summary B."""

    margin = reward_a - reward_b
    if margin >= 0.0:
        return 1.0 / (1.0 + math.exp(-margin))
    exp_margin = math.exp(margin)
    return exp_margin / (1.0 + exp_margin)


def kl_shaped_token_rewards(
    policy_token_logprobs: Sequence[float],
    reference_token_logprobs: Sequence[float],
    *,
    reward_model_score: float,
    beta: float,
) -> list[float]:
    """Construct per-token rewards for one sampled summary.

    For sampled token y_t, the non-score reward is

        -beta * (log pi(y_t | s_t) - log pi_SFT(y_t | s_t)).

    The reward model scores the complete summary, so its score is added only to
    the final token. A sampled log-ratio may be negative; its expectation under
    the policy is the non-negative forward KL.
    """

    _require_same_length(policy_token_logprobs, reference_token_logprobs)
    if not policy_token_logprobs:
        raise ValueError("a rollout needs at least one generated token")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")

    rewards = [
        -beta * (policy_logprob - reference_logprob)
        for policy_logprob, reference_logprob in zip(
            policy_token_logprobs,
            reference_token_logprobs,
        )
    ]
    rewards[-1] += reward_model_score
    return rewards


def generalized_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    *,
    bootstrap_value: float = 0.0,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> list[float]:
    """Compute GAE from per-token rewards and pre-action state values."""

    _require_same_length(rewards, values)
    if not rewards:
        raise ValueError("a rollout needs at least one generated token")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")

    advantages = [0.0] * len(rewards)
    next_value = bootstrap_value
    next_advantage = 0.0
    for index in reversed(range(len(rewards))):
        td_error = rewards[index] + gamma * next_value - values[index]
        next_advantage = td_error + gamma * gae_lambda * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages


@dataclass(frozen=True)
class PPODiagnostics:
    loss: float
    objective: float
    ratios: tuple[float, ...]
    clipped_ratios: tuple[float, ...]
    clipped_fraction: float


def ppo_clipped_policy_loss(
    old_action_logprobs: Sequence[float],
    new_action_logprobs: Sequence[float],
    advantages: Sequence[float],
    *,
    clip_epsilon: float = 0.2,
) -> PPODiagnostics:
    """Compute the standard PPO clipped surrogate on sampled tokens."""

    _require_same_length(old_action_logprobs, new_action_logprobs, advantages)
    if not old_action_logprobs:
        raise ValueError("at least one sampled action is required")
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be in (0, 1)")

    low, high = 1.0 - clip_epsilon, 1.0 + clip_epsilon
    ratios = [
        math.exp(new_logprob - old_logprob)
        for old_logprob, new_logprob in zip(
            old_action_logprobs,
            new_action_logprobs,
        )
    ]
    clipped_ratios = [min(max(ratio, low), high) for ratio in ratios]
    surrogate_terms = [
        min(ratio * advantage, clipped_ratio * advantage)
        for ratio, clipped_ratio, advantage in zip(
            ratios,
            clipped_ratios,
            advantages,
        )
    ]
    objective = sum(surrogate_terms) / len(surrogate_terms)
    clipped_count = sum(
        not math.isclose(ratio, clipped_ratio)
        for ratio, clipped_ratio in zip(ratios, clipped_ratios)
    )
    return PPODiagnostics(
        loss=-objective,
        objective=objective,
        ratios=tuple(ratios),
        clipped_ratios=tuple(clipped_ratios),
        clipped_fraction=clipped_count / len(ratios),
    )


def best_of_n(summaries: Sequence[str], rewards: Sequence[float]) -> tuple[str, int]:
    """Return the highest-reward candidate and its original index."""

    _require_same_length(summaries, rewards)
    if not summaries:
        raise ValueError("at least one candidate is required")
    best_index = max(range(len(rewards)), key=rewards.__getitem__)
    return summaries[best_index], best_index


def best_of_n_kl(n: int) -> float:
    """Paper's idealized KL expression for best-of-N against its base policy.

    This assumes iid candidates and a continuous score distribution without
    ties, matching the analytic calculation reported in the paper.
    """

    if n < 1:
        raise ValueError("n must be positive")
    return math.log(n) - (n - 1.0) / n


def _demo() -> None:
    equal_reward_loss = pairwise_reward_loss([0.0], [0.0])
    assert math.isclose(equal_reward_loss, math.log(2.0))
    assert pairwise_reward_loss([2.0], [-1.0]) < equal_reward_loss
    assert math.isclose(preference_probability(0.0, 0.0), 0.5)

    policy_logprobs = [-0.40, -0.70, -0.20]
    reference_logprobs = [-0.45, -0.60, -0.30]
    terminal_score = 1.20
    beta = 0.05
    rewards = kl_shaped_token_rewards(
        policy_logprobs,
        reference_logprobs,
        reward_model_score=terminal_score,
        beta=beta,
    )
    expected_return = terminal_score - beta * sum(
        policy - reference
        for policy, reference in zip(policy_logprobs, reference_logprobs)
    )
    assert math.isclose(sum(rewards), expected_return)

    advantages = generalized_advantages(
        rewards,
        values=[0.50, 0.55, 0.65],
        gamma=1.0,
        gae_lambda=0.95,
    )
    ppo = ppo_clipped_policy_loss(
        old_action_logprobs=policy_logprobs,
        new_action_logprobs=[-0.05, -1.05, -0.18],
        advantages=advantages,
        clip_epsilon=0.2,
    )

    candidates = (
        "遗漏了关键事实的摘要",
        "覆盖核心事实且没有添加原文之外信息的摘要",
        "流畅但含有幻觉的摘要",
    )
    selected, selected_index = best_of_n(candidates, [0.1, 1.4, 0.6])
    assert selected_index == 1

    print(f"pairwise loss at equal rewards : {equal_reward_loss:.6f}")
    print(f"P(A preferred), margin=3      : {preference_probability(2, -1):.6f}")
    print(f"token rewards                 : {[round(x, 4) for x in rewards]}")
    print(f"GAE advantages                : {[round(x, 4) for x in advantages]}")
    print(f"PPO clipped fraction          : {ppo.clipped_fraction:.2%}")
    print(f"best-of-3 KL expression       : {best_of_n_kl(3):.6f}")
    print(f"selected summary              : {selected}")
    print("all checks passed")


if __name__ == "__main__":
    _demo()
