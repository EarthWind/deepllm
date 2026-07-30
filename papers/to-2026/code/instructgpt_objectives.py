#!/usr/bin/env python3
"""Dependency-free checks for the core InstructGPT / RLHF objectives.

This is not a language-model trainer.  It isolates the math that is easiest to
hide inside a large RLHF stack:

1. a common instruction-SFT implementation computes next-token loss on
   response tokens only (the paper does not disclose its token-mask detail);
2. the reward model learns pairwise preferences with a Bradley–Terry loss;
3. sampled token log-ratios form the KL-shaped reward used during PPO;
4. PPO clips overly large policy updates; and
5. PPO-ptx adds a pretraining log-likelihood term to reduce alignment tax.

Run:
    python3 papers/to-2026/code/instructgpt_objectives.py
"""

from __future__ import annotations

import itertools
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


def sft_response_nll(
    token_logprobs: Sequence[Sequence[float]],
    response_masks: Sequence[Sequence[bool]],
) -> float:
    """Average negative log-likelihood over response tokens only.

    ``token_logprobs[b][t]`` is log p(token_t | tokens_<t).  The mask is false
    on prompt/padding positions and true on target response positions.
    """

    _require_same_length(token_logprobs, response_masks)
    selected: list[float] = []
    for row_logprobs, row_mask in zip(token_logprobs, response_masks):
        _require_same_length(row_logprobs, row_mask)
        selected.extend(
            logprob
            for logprob, is_response in zip(row_logprobs, row_mask)
            if is_response
        )
    if not selected:
        raise ValueError("at least one response token is required")
    return -sum(selected) / len(selected)


def pairwise_reward_loss(
    chosen_rewards: Sequence[float],
    rejected_rewards: Sequence[float],
) -> float:
    """Mean -log sigmoid(r_chosen - r_rejected)."""

    _require_same_length(chosen_rewards, rejected_rewards)
    if not chosen_rewards:
        raise ValueError("at least one preference pair is required")
    losses = [
        -log_sigmoid(chosen - rejected)
        for chosen, rejected in zip(chosen_rewards, rejected_rewards)
    ]
    return sum(losses) / len(losses)


def ranked_reward_loss(rewards_best_to_worst: Sequence[float]) -> tuple[float, int]:
    """Use every ordered pair implied by one K-way ranking.

    InstructGPT labelers ranked K=4..9 completions for a prompt.  A ranking of
    K responses implies C(K, 2) pairwise comparisons, but all pairs from one
    prompt should remain grouped rather than being treated as independent
    prompt-level datapoints.
    """

    if len(rewards_best_to_worst) < 2:
        raise ValueError("a ranking needs at least two completions")

    pairs = list(itertools.combinations(rewards_best_to_worst, 2))
    loss = pairwise_reward_loss(
        chosen_rewards=[chosen for chosen, _ in pairs],
        rejected_rewards=[rejected for _, rejected in pairs],
    )
    return loss, len(pairs)


def kl_shaped_token_rewards(
    policy_token_logprobs: Sequence[float],
    reference_token_logprobs: Sequence[float],
    *,
    reward_model_score: float,
    beta: float,
) -> list[float]:
    """Build per-token rewards with the RM score added at the final token.

    For one sampled response token y_t:

        non_score_reward_t = -beta * (log pi(y_t) - log pi_ref(y_t))

    The expectation of the sampled log-ratio under pi is KL(pi || pi_ref).
    An individual sampled term may be negative; only its expectation is
    guaranteed to be non-negative.
    """

    _require_same_length(policy_token_logprobs, reference_token_logprobs)
    if not policy_token_logprobs:
        raise ValueError("a rollout needs at least one response token")
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


@dataclass(frozen=True)
class PPODiagnostics:
    """Transparent return value for a clipped PPO policy update."""

    loss: float
    objective: float
    probability_ratios: tuple[float, ...]
    clipped_ratios: tuple[float, ...]
    clipped_fraction: float


def ppo_clipped_policy_loss(
    old_action_logprobs: Sequence[float],
    new_action_logprobs: Sequence[float],
    advantages: Sequence[float],
    *,
    clip_epsilon: float = 0.2,
) -> PPODiagnostics:
    """Compute the standard clipped PPO policy loss for sampled actions."""

    _require_same_length(
        old_action_logprobs,
        new_action_logprobs,
        advantages,
    )
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
        probability_ratios=tuple(ratios),
        clipped_ratios=tuple(clipped_ratios),
        clipped_fraction=clipped_count / len(ratios),
    )


def ppo_ptx_sequence_objective(
    *,
    reward_model_score: float,
    policy_response_logprobs: Sequence[float],
    reference_response_logprobs: Sequence[float],
    pretraining_token_logprobs: Sequence[float] = (),
    beta: float,
    gamma: float,
) -> float:
    """Equation-2-style sequence objective, before PPO's clipped estimator.

    The first two terms are the reward-model score and sampled KL penalty.
    The final term is the log-likelihood of tokens from the pretraining
    distribution.  Setting gamma=0 recovers the paper's PPO variant.
    """

    _require_same_length(
        policy_response_logprobs,
        reference_response_logprobs,
    )
    if beta < 0.0 or gamma < 0.0:
        raise ValueError("beta and gamma must be non-negative")

    sampled_log_ratio = sum(
        policy_logprob - reference_logprob
        for policy_logprob, reference_logprob in zip(
            policy_response_logprobs,
            reference_response_logprobs,
        )
    )
    return (
        reward_model_score
        - beta * sampled_log_ratio
        + gamma * sum(pretraining_token_logprobs)
    )


def _check_objectives() -> None:
    """Run small deterministic invariants for every objective."""

    # Prompt positions (mask=False) do not affect the SFT response loss.
    sft_loss = sft_response_nll(
        token_logprobs=(
            (-9.0, -8.0, -0.2, -0.4),
            (-7.0, -0.1, -0.3, -6.0),
        ),
        response_masks=(
            (False, False, True, True),
            (False, True, True, False),
        ),
    )
    assert math.isclose(sft_loss, 0.25)

    # K=4 ranked completions imply C(4,2)=6 preference pairs.
    ranked_loss, pair_count = ranked_reward_loss((2.0, 1.0, 0.0, -1.0))
    tied_loss, _ = ranked_reward_loss((0.0, 0.0, 0.0, 0.0))
    assert pair_count == 6
    assert ranked_loss < tied_loss

    # Token-shaped KL rewards sum to the same reward-minus-log-ratio objective.
    policy_logprobs = (-0.2, -0.5, -0.3)
    reference_logprobs = (-0.3, -0.4, -0.7)
    shaped = kl_shaped_token_rewards(
        policy_logprobs,
        reference_logprobs,
        reward_model_score=1.7,
        beta=0.02,
    )
    sequence_objective = ppo_ptx_sequence_objective(
        reward_model_score=1.7,
        policy_response_logprobs=policy_logprobs,
        reference_response_logprobs=reference_logprobs,
        beta=0.02,
        gamma=0.0,
    )
    assert math.isclose(sum(shaped), sequence_objective)

    # Large policy ratios are clipped to [0.8, 1.2] for epsilon=0.2.
    ppo = ppo_clipped_policy_loss(
        old_action_logprobs=(0.0, 0.0),
        new_action_logprobs=(math.log(2.0), math.log(0.5)),
        advantages=(1.0, -1.0),
        clip_epsilon=0.2,
    )
    assert ppo.clipped_ratios == (1.2, 0.8)
    assert math.isclose(ppo.objective, 0.2)

    print("All InstructGPT objective checks passed.")
    print(f"SFT response-token NLL: {sft_loss:.4f}")
    print(f"RM ranking pairs/loss: {pair_count} / {ranked_loss:.4f}")
    print(f"KL-shaped rollout return: {sum(shaped):.4f}")
    print(
        "PPO raw/clipped ratios:",
        tuple(round(value, 2) for value in ppo.probability_ratios),
        "/",
        ppo.clipped_ratios,
    )


if __name__ == "__main__":
    _check_objectives()
