#!/usr/bin/env python3
"""Zero-dependency tools for reading the Limits of RLVR paper.

This is not a language-model trainer.  It isolates four measurements that are
easy to conflate when discussing reinforcement learning with verifiable
rewards (RLVR):

1. the unbiased pass@k estimator used by the paper;
2. pass@1 (sampling efficiency) versus large-k problem coverage;
3. the base-only / RL-only / both / neither coverage partition;
4. perplexity and the sampling-efficiency shortfall.

Run:
    python3 limits_rlvr_minimal.py
    python3 limits_rlvr_minimal.py --test
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from statistics import fmean
from typing import Sequence


@dataclass(frozen=True)
class SampleCounts:
    """Correct-sample counts for the same problems and sampling budget."""

    name: str
    total_samples: int
    correct_per_problem: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.total_samples <= 0:
            raise ValueError("total_samples must be positive")
        if not self.correct_per_problem:
            raise ValueError("at least one problem is required")
        if any(
            count < 0 or count > self.total_samples
            for count in self.correct_per_problem
        ):
            raise ValueError("correct counts must lie in [0, total_samples]")


def pass_at_k_from_count(total: int, correct: int, k: int) -> float:
    """Unbiased pass@k estimate for one problem.

    With ``total=n`` sampled completions and ``correct=c`` successes, the
    paper follows the HumanEval estimator

        1 - C(n-c, k) / C(n, k).

    The product/log form below avoids constructing huge binomial integers and
    remains numerically stable when the success probability is small.
    """

    if total <= 0:
        raise ValueError("total must be positive")
    if correct < 0 or correct > total:
        raise ValueError("correct must lie in [0, total]")
    if k <= 0 or k > total:
        raise ValueError("k must lie in [1, total]")
    if correct == 0:
        return 0.0
    if total - correct < k:
        return 1.0

    log_failure = 0.0
    for offset in range(k):
        log_failure += math.log(total - correct - offset)
        log_failure -= math.log(total - offset)
    return -math.expm1(log_failure)


def dataset_pass_at_k(samples: SampleCounts, k: int) -> float:
    """Average the per-problem unbiased pass@k estimates."""

    return fmean(
        pass_at_k_from_count(samples.total_samples, count, k)
        for count in samples.correct_per_problem
    )


def observed_pass_at_1(samples: SampleCounts) -> float:
    """The pass@1 estimate, equal to mean empirical accuracy."""

    return fmean(
        count / samples.total_samples for count in samples.correct_per_problem
    )


def coverage_partition(
    base: SampleCounts, rlvr: SampleCounts
) -> dict[str, float]:
    """Partition problems by whether either model solved them at least once.

    A nonzero count means the problem was observed as solvable within the
    common sampling budget.  This is a finite-budget empirical boundary, not a
    proof about mathematical support at infinitely many samples.
    """

    _require_comparable(base, rlvr)
    counts = {"both": 0, "base_only": 0, "rlvr_only": 0, "neither": 0}

    for base_correct, rl_correct in zip(
        base.correct_per_problem, rlvr.correct_per_problem
    ):
        base_solved = base_correct > 0
        rl_solved = rl_correct > 0
        if base_solved and rl_solved:
            counts["both"] += 1
        elif base_solved:
            counts["base_only"] += 1
        elif rl_solved:
            counts["rlvr_only"] += 1
        else:
            counts["neither"] += 1

    size = len(base.correct_per_problem)
    return {name: count / size for name, count in counts.items()}


def sampling_efficiency_shortfall(
    base: SampleCounts, rlvr: SampleCounts, large_k: int
) -> float:
    """How far RLVR pass@1 is below the base model's large-k potential.

    The paper calls this idea the sampling-efficiency gap.  We expose the sign
    explicitly as a positive shortfall:

        base pass@large_k - RLVR pass@1.

    A smaller nonnegative value means the learned policy converts more of the
    base model's sampled potential into one-shot accuracy.
    """

    _require_comparable(base, rlvr)
    return dataset_pass_at_k(base, large_k) - observed_pass_at_1(rlvr)


def perplexity(token_log_probabilities: Sequence[float]) -> float:
    """Sequence perplexity exp(-mean(log p_t))."""

    if not token_log_probabilities:
        raise ValueError("at least one token log-probability is required")
    if any(value > 0.0 or not math.isfinite(value) for value in token_log_probabilities):
        raise ValueError("log probabilities must be finite and no greater than zero")
    return math.exp(-fmean(token_log_probabilities))


def bernoulli_pass_at_k(success_probability: float, k: int) -> float:
    """Idealized independent-sampling curve, useful for intuition only."""

    if not 0.0 <= success_probability <= 1.0:
        raise ValueError("success_probability must lie in [0, 1]")
    if k <= 0:
        raise ValueError("k must be positive")
    return 1.0 - (1.0 - success_probability) ** k


def _require_comparable(first: SampleCounts, second: SampleCounts) -> None:
    if first.total_samples != second.total_samples:
        raise ValueError("models must use the same sample budget")
    if len(first.correct_per_problem) != len(second.correct_per_problem):
        raise ValueError("models must be evaluated on the same problem count")


def toy_models() -> tuple[SampleCounts, SampleCounts, SampleCounts]:
    """A four-problem thought experiment, not measurements from the paper.

    RLVR moves probability toward two already-solvable problems, so its
    pass@1 rises.  It assigns no observed success to the third problem, so the
    base model wins at large k.  Distillation adds success on a fourth problem,
    illustrating how a stronger teacher can expand observed coverage.
    """

    base = SampleCounts("Base", 256, (64, 8, 2, 0))
    rlvr = SampleCounts("RLVR", 256, (154, 26, 0, 0))
    distilled = SampleCounts("Distilled", 256, (150, 30, 5, 2))
    return base, rlvr, distilled


def paper_training_dynamics() -> tuple[tuple[str, float, float], ...]:
    """Omni-MATH training-set values from paper v5, Appendix Table 4."""

    return (
        ("Qwen2.5-7B Base", 9.9, 67.2),
        ("GRPO step 150", 26.1, 66.3),
        ("GRPO step 300", 33.6, 65.3),
        ("GRPO step 450", 42.5, 64.3),
    )


def run_demo() -> None:
    base, rlvr, distilled = toy_models()
    models = (base, rlvr, distilled)
    ks = (1, 2, 8, 32, 128, 256)

    print("Toy finite-sample boundary (illustrative, not paper data)")
    print("model      " + "".join(f"pass@{k:<5}" for k in ks))
    for model in models:
        values = "".join(f"{dataset_pass_at_k(model, k):<10.3f}" for k in ks)
        print(f"{model.name:<11}{values}")

    print("\nObserved coverage partition: Base vs RLVR")
    for name, fraction in coverage_partition(base, rlvr).items():
        print(f"  {name:<10} {fraction:>6.1%}")

    gap = sampling_efficiency_shortfall(base, rlvr, large_k=256)
    print(f"\nBase pass@256 - RLVR pass@1: {gap:.3f}")

    print("\nPaper v5 training dynamics (Omni-MATH train)")
    print("checkpoint          pass@1  pass@256")
    for checkpoint, pass1, pass256 in paper_training_dynamics():
        print(f"{checkpoint:<20}{pass1:>6.1f}{pass256:>10.1f}")


def run_tests() -> None:
    expected = 1.0 - math.comb(8, 3) / math.comb(10, 3)
    assert math.isclose(pass_at_k_from_count(10, 2, 3), expected)
    assert pass_at_k_from_count(10, 0, 10) == 0.0
    assert pass_at_k_from_count(10, 1, 10) == 1.0

    base, rlvr, distilled = toy_models()
    for model in (base, rlvr, distilled):
        curve = [dataset_pass_at_k(model, k) for k in (1, 2, 8, 32, 128, 256)]
        assert all(left <= right for left, right in zip(curve, curve[1:]))

    assert observed_pass_at_1(rlvr) > observed_pass_at_1(base)
    assert dataset_pass_at_k(base, 128) > dataset_pass_at_k(rlvr, 128)
    assert dataset_pass_at_k(distilled, 256) > dataset_pass_at_k(base, 256)

    partition = coverage_partition(base, rlvr)
    assert partition == {
        "both": 0.5,
        "base_only": 0.25,
        "rlvr_only": 0.0,
        "neither": 0.25,
    }
    assert math.isclose(sum(partition.values()), 1.0)

    assert math.isclose(perplexity((math.log(0.5), math.log(0.5))), 2.0)
    assert math.isclose(bernoulli_pass_at_k(0.1, 2), 0.19)

    dynamics = paper_training_dynamics()
    assert all(a[1] < b[1] for a, b in zip(dynamics, dynamics[1:]))
    assert all(a[2] > b[2] for a, b in zip(dynamics, dynamics[1:]))

    try:
        pass_at_k_from_count(4, 1, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid k should fail")

    print("all tests passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run self-tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test:
        run_tests()
    else:
        run_demo()


if __name__ == "__main__":
    main()
