#!/usr/bin/env python3
"""Dependency-free checks for HumanEval's pass@k evaluation math.

This script deliberately does NOT execute generated code. Model completions are
untrusted programs and must be evaluated inside a real security sandbox. The
functions below consume only already-computed pass/fail labels and token log
probabilities, isolating the safe, reproducible parts of the Codex paper:

1. the unbiased pass@k estimator;
2. aggregation across problems;
3. the bias of the tempting plug-in estimator; and
4. mean-log-probability reranking versus sum-log-probability reranking.

Run:
    python3 papers/to-2026/code/humaneval_pass_at_k_minimal.py
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Return 1 - C(n-c, k) / C(n, k) without huge binomial integers.

    The value is the probability that a uniformly drawn size-k subset from the
    n evaluated candidates contains at least one of the c correct candidates.
    It is also the paper's unbiased estimator of a problem's pass@k.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if not 0 <= num_correct <= num_samples:
        raise ValueError("num_correct must be in [0, num_samples]")
    if not 1 <= k <= num_samples:
        raise ValueError("k must be in [1, num_samples]")

    num_failed = num_samples - num_correct
    if num_failed < k:
        return 1.0

    # Equivalent to 1 - comb(num_failed, k) / comb(num_samples, k),
    # but avoids constructing enormous integers in array-oriented code.
    all_failed_probability = 1.0
    for denominator in range(num_failed + 1, num_samples + 1):
        all_failed_probability *= 1.0 - k / denominator
    return 1.0 - all_failed_probability


def exact_subset_pass_at_k(labels: Sequence[bool], k: int) -> float:
    """Enumerate subsets for tiny examples; useful only as a correctness oracle."""

    if not 1 <= k <= len(labels):
        raise ValueError("k must be in [1, len(labels)]")
    subsets = itertools.combinations(labels, k)
    solved = [any(subset) for subset in subsets]
    return sum(solved) / len(solved)


def naive_plugin_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """The tempting but biased finite-sample plug-in estimate from the paper."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if not 0 <= num_correct <= num_samples:
        raise ValueError("num_correct must be in [0, num_samples]")
    if k < 1:
        raise ValueError("k must be positive")
    empirical_pass_at_1 = num_correct / num_samples
    return 1.0 - (1.0 - empirical_pass_at_1) ** k


@dataclass(frozen=True)
class ProblemResults:
    task_id: str
    passed: tuple[bool, ...]


def dataset_pass_at_k(results: Iterable[ProblemResults], k: int) -> float:
    """Average the per-problem unbiased estimates, as HumanEval does."""

    materialized = tuple(results)
    if not materialized:
        raise ValueError("at least one problem is required")

    estimates = []
    for problem in materialized:
        if len(problem.passed) < k:
            raise ValueError(
                f"{problem.task_id} has {len(problem.passed)} samples, fewer than k={k}"
            )
        estimates.append(
            estimate_pass_at_k(
                num_samples=len(problem.passed),
                num_correct=sum(problem.passed),
                k=k,
            )
        )
    return sum(estimates) / len(estimates)


@dataclass(frozen=True)
class RankedCandidate:
    name: str
    token_logprobs: tuple[float, ...]

    @property
    def sum_logprob(self) -> float:
        return sum(self.token_logprobs)

    @property
    def mean_logprob(self) -> float:
        if not self.token_logprobs:
            raise ValueError("a candidate must contain at least one scored token")
        return self.sum_logprob / len(self.token_logprobs)


def choose_by_mean_logprob(candidates: Sequence[RankedCandidate]) -> RankedCandidate:
    """Paper's practical heuristic when unit-test results are unavailable."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    return max(candidates, key=lambda candidate: candidate.mean_logprob)


def choose_by_sum_logprob(candidates: Sequence[RankedCandidate]) -> RankedCandidate:
    """Length-biased comparison included to make the paper's warning concrete."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    return max(candidates, key=lambda candidate: candidate.sum_logprob)


def _demo() -> None:
    # With one success among 200 candidates, half of all size-100 subsets contain
    # that success. This is an easy-to-audit pass@100 example.
    assert math.isclose(estimate_pass_at_k(200, 1, 100), 0.5)
    assert estimate_pass_at_k(200, 0, 100) == 0.0
    assert estimate_pass_at_k(200, 101, 100) == 1.0

    labels = (True, False, False, False, False)
    combinatorial = estimate_pass_at_k(5, 1, 2)
    enumerated = exact_subset_pass_at_k(labels, 2)
    assert math.isclose(combinatorial, enumerated)
    assert math.isclose(combinatorial, 0.4)

    # The plug-in estimator treats repeated draws from a finite observed pool as
    # independent and underestimates this finite-sample example.
    unbiased = estimate_pass_at_k(10, 1, 5)
    naive = naive_plugin_pass_at_k(10, 1, 5)
    assert math.isclose(unbiased, 0.5)
    assert naive < unbiased

    dataset = (
        ProblemResults("HumanEval/toy-0", (True, True, False, False, False)),
        ProblemResults("HumanEval/toy-1", (False, False, False, False, False)),
        ProblemResults("HumanEval/toy-2", (True, True, True, True, True)),
    )
    assert math.isclose(dataset_pass_at_k(dataset, 1), 7 / 15)
    assert math.isclose(dataset_pass_at_k(dataset, 2), (0.7 + 0.0 + 1.0) / 3)

    # Sum log-probability systematically favors shorter completions. Mean
    # log-probability removes that first-order length effect.
    short_lower_quality = RankedCandidate("short", (-0.20, -0.20))
    long_higher_quality = RankedCandidate("long", (-0.10,) * 10)
    candidates = (short_lower_quality, long_higher_quality)
    assert choose_by_sum_logprob(candidates).name == "short"
    assert choose_by_mean_logprob(candidates).name == "long"

    print(f"pass@100 with n=200, c=1     : {estimate_pass_at_k(200, 1, 100):.6f}")
    print(f"unbiased pass@5, n=10, c=1  : {unbiased:.6f}")
    print(f"naive plug-in estimate       : {naive:.6f}")
    print(f"toy dataset pass@1           : {dataset_pass_at_k(dataset, 1):.6f}")
    print(f"toy dataset pass@2           : {dataset_pass_at_k(dataset, 2):.6f}")
    print(f"sum-logprob selection        : {choose_by_sum_logprob(candidates).name}")
    print(f"mean-logprob selection       : {choose_by_mean_logprob(candidates).name}")
    print("all checks passed; no generated code was executed")


if __name__ == "__main__":
    _demo()
