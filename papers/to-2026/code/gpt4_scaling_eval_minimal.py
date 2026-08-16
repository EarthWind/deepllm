#!/usr/bin/env python3
"""Executable teaching examples for the GPT-4 Technical Report.

This is NOT GPT-4 source code. The report does not disclose model size,
architecture details, hardware, training compute, or dataset construction.

The script only reproduces four public methodological ideas:

1. fit a compute power law with an irreducible-loss term;
2. aggregate HumanEval-style pass rates in log space;
3. measure expected calibration error (ECE);
4. audit evaluation examples for substring overlap with training documents.

All scaling measurements below are synthetic. They are deliberately not fitted
to GPT-4 because the paper does not publish the underlying numerical run data.

Run:
    python3 papers/to-2026/code/gpt4_scaling_eval_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isclose, log
from random import Random
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PowerLawFit:
    """Parameters of y(C) = a * C**b + c and a log-space fit score."""

    a: float
    b: float
    c: float
    mean_squared_log_error: float

    def predict(self, compute: float) -> float:
        if compute <= 0:
            raise ValueError("compute must be positive")
        return self.a * compute**self.b + self.c


def _linear_regression(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need at least two paired observations")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("x observations must not all be equal")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    return intercept, slope


def fit_power_law_with_floor(
    computes: Sequence[float],
    losses: Sequence[float],
    *,
    floor_steps: int = 2_000,
) -> PowerLawFit:
    """Fit L(C)=a*C**b+c by deterministic grid search over c.

    Once c is fixed, log(L-c)=log(a)+b*log(C) is ordinary linear
    regression. Searching c keeps the example dependency-free. Production
    analysis should use uncertainty estimates and held-out validation too.
    """

    if len(computes) != len(losses) or len(computes) < 3:
        raise ValueError("need at least three paired observations")
    if any(value <= 0 for value in computes + losses):
        raise ValueError("compute and loss must be positive")
    if floor_steps < 2:
        raise ValueError("floor_steps must be at least 2")

    minimum_loss = min(losses)
    # A valid floor must be below every observed loss. The range intentionally
    # includes zero and stops just short of the smallest measurement.
    candidates = [minimum_loss * 0.999 * i / (floor_steps - 1) for i in range(floor_steps)]
    log_compute = [log(value) for value in computes]
    best: PowerLawFit | None = None

    for floor in candidates:
        transformed = [log(loss - floor) for loss in losses]
        log_a, exponent = _linear_regression(log_compute, transformed)
        coefficient = exp(log_a)
        residuals = [
            log(loss - floor) - (log_a + exponent * log(compute))
            for compute, loss in zip(computes, losses)
        ]
        score = sum(value * value for value in residuals) / len(residuals)
        fit = PowerLawFit(coefficient, exponent, floor, score)
        if best is None or fit.mean_squared_log_error < best.mean_squared_log_error:
            best = fit

    assert best is not None
    return best


def mean_log_pass_rate(pass_rates: Iterable[float], *, epsilon: float = 1e-12) -> float:
    """Return the mean log pass rate used by the capability-scaling idea.

    Zero empirical success cannot be logged. The report avoids that regime by
    restricting analysis to problems every compared model solves at least once
    under a large sample budget. ``epsilon`` here only prevents a crash; it is
    not a substitute for that identifiability requirement.
    """

    rates = list(pass_rates)
    if not rates:
        raise ValueError("pass_rates must not be empty")
    if any(rate < 0 or rate > 1 for rate in rates):
        raise ValueError("pass rates must lie in [0, 1]")
    return sum(log(max(rate, epsilon)) for rate in rates) / len(rates)


def geometric_mean_pass_rate(pass_rates: Iterable[float]) -> float:
    """Convert mean log pass rate back to an interpretable geometric mean."""

    return exp(mean_log_pass_rate(pass_rates))


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 10,
) -> float:
    """Compute standard equal-width expected calibration error."""

    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidence/correct arrays must be paired and non-empty")
    if any(value < 0 or value > 1 for value in confidences):
        raise ValueError("confidences must lie in [0, 1]")
    if bins <= 0:
        raise ValueError("bins must be positive")

    total = len(confidences)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            i
            for i, confidence in enumerate(confidences)
            if lower <= confidence < upper or (index == bins - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        accuracy = sum(bool(correct[i]) for i in members) / len(members)
        mean_confidence = sum(confidences[i] for i in members) / len(members)
        error += len(members) / total * abs(accuracy - mean_confidence)
    return error


def normalize_for_overlap(text: str) -> str:
    """Approximate the paper's normalization: keep letters and numbers only."""

    return "".join(character.casefold() for character in text if character.isalnum())


def sampled_substring_overlap(
    evaluation_example: str,
    training_documents: Sequence[str],
    *,
    substring_length: int = 50,
    samples: int = 3,
    seed: int = 0,
) -> bool:
    """Approximate Appendix C's randomized substring contamination check.

    This detector inherits the paper's stated limitations: paraphrases can be
    false negatives and generic repeated phrases can be false positives.
    """

    normalized_eval = normalize_for_overlap(evaluation_example)
    normalized_train = [normalize_for_overlap(document) for document in training_documents]
    if not normalized_eval:
        return False
    if len(normalized_eval) <= substring_length:
        probes = [normalized_eval]
    else:
        rng = Random(seed)
        maximum_start = len(normalized_eval) - substring_length
        starts = [rng.randint(0, maximum_start) for _ in range(samples)]
        probes = [normalized_eval[start : start + substring_length] for start in starts]
    return any(probe in document for probe in probes for document in normalized_train)


def _demo_scaling() -> None:
    # Hidden toy law: 1.7*C^-0.095 + 0.82. The tiny deterministic deviations
    # simulate noisy observations. C=1 (the target run) is held out.
    computes = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    deviations = [0.006, -0.004, 0.003, -0.002, 0.002, -0.001, 0.001]
    observed = [1.7 * value**-0.095 + 0.82 + noise for value, noise in zip(computes, deviations)]
    fit = fit_power_law_with_floor(computes, observed)
    predicted = fit.predict(1.0)
    hidden_target = 1.7 + 0.82
    relative_error = abs(predicted - hidden_target) / hidden_target
    assert fit.b < 0 and relative_error < 0.03
    print("synthetic scaling fit:")
    print(f"  L(C) = {fit.a:.4f} * C^{fit.b:.4f} + {fit.c:.4f}")
    print(f"  target prediction at C=1: {predicted:.4f}")
    print(f"  held-out relative error:  {relative_error:.2%}")


def _demo_evaluation() -> None:
    rates = [0.80, 0.50, 0.20]
    arithmetic_mean = sum(rates) / len(rates)
    geometric_mean = geometric_mean_pass_rate(rates)
    assert geometric_mean < arithmetic_mean
    print("capability aggregation:")
    print(f"  arithmetic mean pass rate: {arithmetic_mean:.3f}")
    print(f"  geometric mean pass rate:  {geometric_mean:.3f}")

    calibrated_confidence = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    calibrated_correct = [False, False, False, True, False, True, True, True]
    overconfident = [min(0.99, value + 0.12) for value in calibrated_confidence]
    ece_before = expected_calibration_error(calibrated_confidence, calibrated_correct, bins=4)
    ece_after = expected_calibration_error(overconfident, calibrated_correct, bins=4)
    assert ece_after > ece_before
    print("toy calibration check:")
    print(f"  ECE before confidence shift: {ece_before:.3f}")
    print(f"  ECE after confidence shift:  {ece_after:.3f}")

    exam = "A transformer model predicts the next token in a document using context."
    corpus = [
        "Unrelated material about evaluation.",
        "A transformer model predicts the next token in a document using context, then it is aligned.",
    ]
    assert sampled_substring_overlap(exam, corpus, substring_length=30, seed=4)
    assert not sampled_substring_overlap("A completely separate held-out question", corpus, seed=4)
    print("  contamination audit: overlap and clean cases detected")


def _self_check() -> None:
    assert isclose(geometric_mean_pass_rate([0.25, 1.0]), 0.5)
    assert expected_calibration_error([0.5, 0.5], [True, False], bins=2) == 0.0
    _demo_scaling()
    _demo_evaluation()
    print("all checks passed")


if __name__ == "__main__":
    _self_check()
