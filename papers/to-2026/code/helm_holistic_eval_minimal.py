#!/usr/bin/env python3
"""A dependency-free, HELM-inspired evaluation miniature.

This is teaching code, not a reimplementation of stanford-crfm/helm.  It
focuses on the measurement ideas in the HELM paper: calibration, selective
prediction, worst-case robustness, group disparities, efficiency, coverage,
stakeholder-dependent aggregation, Pareto frontiers, and paired bootstrap
uncertainty.

Run:
    python papers/to-2026/code/helm_holistic_eval_minimal.py
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Prediction:
    """One classification result plus optional audit metadata."""

    example_id: str
    correct: bool
    confidence: float
    group: str | None = None
    perturbation: str = "original"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")


@dataclass(frozen=True)
class MetricSpec:
    """How to orient and normalize one heterogeneous metric."""

    low: float
    high: float
    higher_is_better: bool = True

    def normalize(self, value: float) -> float:
        if self.high <= self.low:
            raise ValueError("high must be greater than low")
        unit = min(1.0, max(0.0, (value - self.low) / (self.high - self.low)))
        return unit if self.higher_is_better else 1.0 - unit


def exact_match(prediction: str, reference: str) -> float:
    """A deliberately transparent exact-match metric."""

    normalize = lambda text: " ".join(text.strip().lower().split())
    return float(normalize(prediction) == normalize(reference))


def _equal_mass_bins(items: Sequence[Prediction], n_bins: int) -> list[list[Prediction]]:
    """Split sorted predictions into nearly equal-sized bins."""

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    ordered = sorted(items, key=lambda item: item.confidence)
    return [
        ordered[start:end]
        for index in range(n_bins)
        if (start := index * len(ordered) // n_bins)
        < (end := (index + 1) * len(ordered) // n_bins)
    ]


def expected_calibration_error(predictions: Sequence[Prediction], n_bins: int = 10) -> float:
    """ECE with HELM's equal-mass-bin convention."""

    if not predictions:
        raise ValueError("ECE needs at least one prediction")
    total = len(predictions)
    ece = 0.0
    for bucket in _equal_mass_bins(predictions, min(n_bins, total)):
        accuracy = statistics.fmean(float(item.correct) for item in bucket)
        confidence = statistics.fmean(item.confidence for item in bucket)
        ece += len(bucket) / total * abs(accuracy - confidence)
    return ece


def selective_curve(predictions: Sequence[Prediction]) -> list[tuple[float, float]]:
    """Accuracy when retaining only the most confident fraction."""

    if not predictions:
        raise ValueError("selective prediction needs data")
    ordered = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    curve: list[tuple[float, float]] = []
    correct_so_far = 0
    for count, item in enumerate(ordered, start=1):
        correct_so_far += int(item.correct)
        curve.append((count / len(ordered), correct_so_far / count))
    return curve


def coverage_accuracy_auc(predictions: Sequence[Prediction]) -> float:
    """Trapezoidal area under the coverage-accuracy curve."""

    curve = [(0.0, selective_curve(predictions)[0][1]), *selective_curve(predictions)]
    return sum(
        (right_x - left_x) * (left_y + right_y) / 2
        for (left_x, left_y), (right_x, right_y) in zip(curve, curve[1:])
    )


def accuracy_at_coverage(predictions: Sequence[Prediction], coverage: float) -> float:
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must lie in (0, 1]")
    ordered = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    kept = max(1, math.ceil(coverage * len(ordered)))
    return statistics.fmean(float(item.correct) for item in ordered[:kept])


def worst_case_robustness(predictions: Sequence[Prediction]) -> float:
    """Mean over examples of their worst perturbation score."""

    by_example: dict[str, list[float]] = {}
    for item in predictions:
        by_example.setdefault(item.example_id, []).append(float(item.correct))
    if not by_example:
        raise ValueError("robustness needs data")
    return statistics.fmean(min(scores) for scores in by_example.values())


def group_statistics(predictions: Sequence[Prediction]) -> tuple[dict[str, float], float, float]:
    """Return group accuracies, worst-group accuracy, and max-min gap."""

    grouped: dict[str, list[float]] = {}
    for item in predictions:
        if item.group is not None:
            grouped.setdefault(item.group, []).append(float(item.correct))
    if not grouped:
        raise ValueError("group analysis needs group labels")
    scores = {group: statistics.fmean(values) for group, values in grouped.items()}
    return scores, min(scores.values()), max(scores.values()) - min(scores.values())


def training_energy_kwh(
    num_accelerators: int,
    accelerator_watts: float,
    training_hours: float,
    pue: float = 1.1,
) -> float:
    """HELM's disclosed training-energy approximation."""

    if min(num_accelerators, accelerator_watts, training_hours, pue) < 0:
        raise ValueError("energy inputs must be non-negative")
    return num_accelerators * accelerator_watts * training_hours * pue / 1000.0


def emissions_kg_co2e(energy_kwh: float, carbon_kg_per_kwh: float) -> float:
    return energy_kwh * carbon_kg_per_kwh


def metric_coverage(
    scores: Mapping[str, float | None], expected_metrics: Iterable[str]
) -> tuple[int, int, float]:
    """Coverage is explicit: missing is neither zero nor silently discarded."""

    expected = list(expected_metrics)
    observed = sum(scores.get(metric) is not None for metric in expected)
    return observed, len(expected), observed / len(expected) if expected else 1.0


def weighted_utility(
    scores: Mapping[str, float | None],
    specs: Mapping[str, MetricSpec],
    weights: Mapping[str, float],
) -> float:
    """Stakeholder-specific utility that fails closed on missing metrics."""

    if not weights or any(weight < 0 for weight in weights.values()):
        raise ValueError("weights must be non-empty and non-negative")
    missing = [name for name in weights if scores.get(name) is None]
    if missing:
        raise ValueError(f"cannot aggregate missing metrics: {missing}")
    denominator = sum(weights.values())
    if denominator == 0:
        raise ValueError("at least one weight must be positive")
    return sum(
        weights[name] * specs[name].normalize(float(scores[name]))
        for name in weights
    ) / denominator


def pareto_frontier(
    models: Mapping[str, Mapping[str, float]],
    higher_is_better: Mapping[str, bool],
) -> list[str]:
    """Models not strictly dominated on every requested metric."""

    metrics = list(higher_is_better)

    def at_least_as_good(a: Mapping[str, float], b: Mapping[str, float], metric: str) -> bool:
        return a[metric] >= b[metric] if higher_is_better[metric] else a[metric] <= b[metric]

    def strictly_better(a: Mapping[str, float], b: Mapping[str, float], metric: str) -> bool:
        return a[metric] > b[metric] if higher_is_better[metric] else a[metric] < b[metric]

    frontier = []
    for candidate, candidate_scores in models.items():
        dominated = any(
            other != candidate
            and all(at_least_as_good(other_scores, candidate_scores, metric) for metric in metrics)
            and any(strictly_better(other_scores, candidate_scores, metric) for metric in metrics)
            for other, other_scores in models.items()
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier)


def paired_bootstrap_difference(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    samples: int = 5000,
    seed: int = 7,
) -> tuple[float, tuple[float, float]]:
    """Paired mean difference and percentile 95% confidence interval."""

    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("paired bootstrap needs equal non-empty sequences")
    differences = [a - b for a, b in zip(scores_a, scores_b)]
    rng = random.Random(seed)
    bootstrap = sorted(
        statistics.fmean(rng.choice(differences) for _ in differences)
        for _ in range(samples)
    )
    lower = bootstrap[int(0.025 * samples)]
    upper = bootstrap[min(samples - 1, int(0.975 * samples))]
    return statistics.fmean(differences), (lower, upper)


def demo() -> None:
    original = [
        Prediction("q1", True, 0.92, "A"),
        Prediction("q2", True, 0.79, "A"),
        Prediction("q3", False, 0.76, "B"),
        Prediction("q4", True, 0.64, "B"),
        Prediction("q5", False, 0.61, "A"),
        Prediction("q6", False, 0.35, "B"),
    ]
    perturbed = original + [
        Prediction("q1", True, 0.85, "A", "lowercase"),
        Prediction("q2", False, 0.62, "A", "typo"),
        Prediction("q3", False, 0.70, "B", "extra_space"),
        Prediction("q4", True, 0.58, "B", "lowercase"),
        Prediction("q5", False, 0.55, "A", "typo"),
        Prediction("q6", False, 0.30, "B", "extra_space"),
    ]

    group_scores, worst_group, group_gap = group_statistics(original)
    print("accuracy:", statistics.fmean(float(item.correct) for item in original))
    print("ECE (equal-mass bins):", round(expected_calibration_error(original, 3), 4))
    print("accuracy@50% coverage:", round(accuracy_at_coverage(original, 0.5), 4))
    print("coverage-accuracy AUC:", round(coverage_accuracy_auc(original), 4))
    print("worst-case perturbation accuracy:", round(worst_case_robustness(perturbed), 4))
    print("group accuracy / worst / gap:", group_scores, worst_group, group_gap)

    energy = training_energy_kwh(64, 400, 24 * 14, pue=1.1)
    print("training energy / emissions:", round(energy, 1), "kWh /", round(emissions_kg_co2e(energy, 0.39), 1), "kgCO2e")

    seven_metrics = [
        "accuracy", "calibration", "robustness", "fairness",
        "bias", "toxicity", "efficiency",
    ]
    model_a: dict[str, float | None] = {
        "accuracy": 0.81,
        "calibration": 0.09,  # ECE: lower is better
        "robustness": 0.66,
        "fairness": 0.72,
        "bias": 0.18,         # disparity: lower is better
        "toxicity": 0.04,     # rate: lower is better
        "efficiency": None,    # intentionally unavailable
    }
    print("coverage:", metric_coverage(model_a, seven_metrics))

    # Aggregation only after directions, ranges, coverage, and stakeholder
    # weights are explicit.  Exclude efficiency here on purpose and disclose it.
    specs = {
        "accuracy": MetricSpec(0, 1),
        "calibration": MetricSpec(0, 0.5, higher_is_better=False),
        "robustness": MetricSpec(0, 1),
        "toxicity": MetricSpec(0, 0.5, higher_is_better=False),
    }
    weights = {"accuracy": 0.4, "calibration": 0.2, "robustness": 0.3, "toxicity": 0.1}
    print("disclosed weighted utility:", round(weighted_utility(model_a, specs, weights), 4))

    models = {
        "accurate-but-costly": {"accuracy": 0.85, "ece": 0.13, "latency": 1.8},
        "balanced": {"accuracy": 0.82, "ece": 0.08, "latency": 0.9},
        "fast": {"accuracy": 0.77, "ece": 0.10, "latency": 0.3},
        "dominated": {"accuracy": 0.75, "ece": 0.14, "latency": 1.1},
    }
    print("Pareto frontier:", pareto_frontier(models, {"accuracy": True, "ece": False, "latency": False}))

    delta, interval = paired_bootstrap_difference(
        [1, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 0],
    )
    print("paired delta / 95% CI:", round(delta, 3), tuple(round(x, 3) for x in interval))


if __name__ == "__main__":
    demo()
