#!/usr/bin/env python3
"""A zero-dependency executable sketch of DAPO's core operators.

This file is deliberately small: it does not train a language model or launch
distributed rollouts.  It isolates the pieces that are easiest to confuse when
reading the paper:

1. group-relative advantages and zero-gradient group filtering;
2. asymmetric PPO clipping (Clip-Higher);
3. sample-level versus token-level loss aggregation;
4. the soft overlong reward used near the response-length limit.

Run:
    python3 dapo_minimal.py
    python3 dapo_minimal.py --test
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Rollout:
    """One sampled response after rule-based verification.

    ``log_ratios`` contains log pi_theta - log pi_old for policy-generated
    tokens.  A real trainer obtains one ratio per valid response token.
    """

    name: str
    reward: float
    log_ratios: tuple[float, ...]


def group_relative_advantages(rewards: Sequence[float]) -> list[float]:
    """Normalize rewards inside one prompt's rollout group.

    DAPO samples G responses for the same prompt.  If every reward is equal,
    the standard deviation is zero and the group provides no policy-gradient
    direction; returning zeros makes that fact explicit.
    """

    if not rewards:
        raise ValueError("rewards must not be empty")
    mean = fmean(rewards)
    variance = fmean((reward - mean) ** 2 for reward in rewards)
    std = math.sqrt(variance)
    if std < 1e-12:
        return [0.0 for _ in rewards]
    return [(reward - mean) / std for reward in rewards]


def is_informative_group(rewards: Sequence[float]) -> bool:
    """Whether a group has both successful and unsuccessful responses."""

    return bool(rewards) and min(rewards) < max(rewards)


def dynamic_sample(
    candidate_groups: Iterable[Sequence[Rollout]], target_groups: int
) -> list[list[Rollout]]:
    """Keep mixed-reward groups until the effective prompt batch is full.

    Production DAPO keeps generating new groups when the buffer is short.  This
    finite example consumes an existing stream and raises if it cannot fill the
    requested effective batch.
    """

    if target_groups <= 0:
        raise ValueError("target_groups must be positive")

    buffer: list[list[Rollout]] = []
    for group in candidate_groups:
        group = list(group)
        if is_informative_group([rollout.reward for rollout in group]):
            buffer.append(group)
        if len(buffer) == target_groups:
            return buffer

    raise RuntimeError(
        f"only collected {len(buffer)} informative groups; "
        f"need {target_groups}"
    )


def clipped_surrogate(
    log_ratio: float,
    advantage: float,
    eps_low: float = 0.20,
    eps_high: float = 0.28,
) -> float:
    """One token's DAPO clipped surrogate objective (to be maximized)."""

    if eps_low < 0.0 or eps_high < 0.0:
        raise ValueError("clip ranges must be non-negative")
    ratio = math.exp(log_ratio)
    clipped_ratio = min(max(ratio, 1.0 - eps_low), 1.0 + eps_high)
    return min(ratio * advantage, clipped_ratio * advantage)


def sample_level_objective(
    groups: Sequence[Sequence[Rollout]], eps_low: float = 0.20, eps_high: float = 0.28
) -> float:
    """Original GRPO reduction: token mean per sequence, then sequence mean."""

    sequence_values: list[float] = []
    for group in groups:
        advantages = group_relative_advantages([item.reward for item in group])
        for rollout, advantage in zip(group, advantages):
            if not rollout.log_ratios:
                continue
            token_values = [
                clipped_surrogate(value, advantage, eps_low, eps_high)
                for value in rollout.log_ratios
            ]
            sequence_values.append(fmean(token_values))
    if not sequence_values:
        raise ValueError("no policy tokens found")
    return fmean(sequence_values)


def token_level_objective(
    groups: Sequence[Sequence[Rollout]], eps_low: float = 0.20, eps_high: float = 0.28
) -> float:
    """DAPO reduction: average the clipped surrogate over all valid tokens."""

    token_values: list[float] = []
    for group in groups:
        advantages = group_relative_advantages([item.reward for item in group])
        for rollout, advantage in zip(group, advantages):
            token_values.extend(
                clipped_surrogate(value, advantage, eps_low, eps_high)
                for value in rollout.log_ratios
            )
    if not token_values:
        raise ValueError("no policy tokens found")
    return fmean(token_values)


def soft_overlong_penalty(
    response_length: int,
    max_length: int = 20_480,
    cache_length: int = 4_096,
) -> float:
    """DAPO's piecewise length reward in [-1, 0].

    The paper allows 16,384 unpenalized tokens and then linearly increases the
    penalty during a 4,096-token cache, giving an actual cap of 20,480.
    """

    if response_length < 0:
        raise ValueError("response_length must be non-negative")
    if cache_length <= 0 or cache_length > max_length:
        raise ValueError("cache_length must be in (0, max_length]")

    expected_length = max_length - cache_length
    if response_length <= expected_length:
        return 0.0
    if response_length <= max_length:
        return (expected_length - response_length) / cache_length
    return -1.0


def make_demo_groups() -> list[list[Rollout]]:
    """Create three candidate groups; only the middle group is informative."""

    all_correct = [
        Rollout("easy-a", 1.0, (math.log(1.02),)),
        Rollout("easy-b", 1.0, (math.log(0.99),)),
    ]
    mixed = [
        Rollout("short-correct", 1.0, (math.log(1.10), math.log(1.50))),
        Rollout(
            "long-wrong",
            -1.0,
            (math.log(0.60), math.log(0.90), math.log(1.05), math.log(1.15)),
        ),
    ]
    all_wrong = [
        Rollout("hard-a", -1.0, (math.log(1.01),)),
        Rollout("hard-b", -1.0, (math.log(0.98),)),
    ]
    return [all_correct, mixed, all_wrong]


def run_tests() -> None:
    advantages = group_relative_advantages([-1.0, 1.0])
    assert advantages == [-1.0, 1.0]
    assert group_relative_advantages([1.0, 1.0]) == [0.0, 0.0]

    candidates = make_demo_groups()
    assert not is_informative_group([1.0, 1.0])
    assert is_informative_group([1.0, -1.0])
    assert not is_informative_group([-1.0, -1.0])
    kept = dynamic_sample(candidates, target_groups=1)
    assert [item.name for item in kept[0]] == ["short-correct", "long-wrong"]

    # Positive advantage is upper-clipped at 1.28; negative advantage with a
    # too-small ratio is lower-clipped at 0.8 in the pessimistic surrogate.
    assert math.isclose(clipped_surrogate(math.log(1.50), 1.0), 1.28)
    assert math.isclose(clipped_surrogate(math.log(0.60), -1.0), -0.80)

    # Unequal response lengths make the two reductions intentionally differ.
    sample_value = sample_level_objective(kept)
    token_value = token_level_objective(kept)
    assert not math.isclose(sample_value, token_value)

    # Toy 20-token cap with a 4-token soft cache mirrors 20,480 / 4,096.
    assert soft_overlong_penalty(16, 20, 4) == 0.0
    assert soft_overlong_penalty(18, 20, 4) == -0.5
    assert soft_overlong_penalty(20, 20, 4) == -1.0
    assert soft_overlong_penalty(21, 20, 4) == -1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run assertions")
    args = parser.parse_args()

    if args.test:
        run_tests()
        print("all DAPO minimal tests passed")
        return

    candidates = make_demo_groups()
    kept = dynamic_sample(candidates, target_groups=1)
    rewards = [item.reward for item in kept[0]]
    print("candidate groups:", len(candidates))
    print("informative groups kept:", len(kept))
    print("kept rewards:", rewards)
    print("group-relative advantages:", group_relative_advantages(rewards))
    print("sample-level objective:", round(sample_level_objective(kept), 4))
    print("token-level objective:", round(token_level_objective(kept), 4))
    print("toy overlong curve:")
    for length in (15, 16, 17, 18, 19, 20, 21):
        print(f"  length={length:2d}, penalty={soft_overlong_penalty(length, 20, 4):5.2f}")


if __name__ == "__main__":
    main()
