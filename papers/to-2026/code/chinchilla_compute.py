#!/usr/bin/env python3
"""Dependency-free Chinchilla compute-allocation calculator.

The script implements two deliberately separate estimators:

1. the practical ``D ~= 20 N`` rule calibrated by the paper's Approach 1 and
   the actual 70B / 1.4T Chinchilla run;
2. the paper's Approach 3 parametric loss model and its closed-form optimum.

These are empirical planning tools, not universal laws. The default FLOPs
budget is the paper's quoted Gopher budget, 5.76e23 FLOPs.

Run:
    python3 papers/to-2026/code/chinchilla_compute.py
    python3 papers/to-2026/code/chinchilla_compute.py --compute 1e22
    python3 papers/to-2026/code/chinchilla_compute.py --csv /tmp/isoflop.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


FLOPS_PER_PARAMETER_TOKEN = 6.0
GOPHER_FLOPS = 5.76e23


@dataclass(frozen=True)
class ChinchillaLossLaw:
    """Approach 3: L(N, D) = E + A/N**alpha + B/D**beta.

    Coefficients are the rounded values reported in Appendix D.2 of Hoffmann
    et al. (2022). N and D are raw counts, not values measured in billions.
    """

    irreducible_loss: float = 1.69
    parameter_coefficient: float = 406.4
    data_coefficient: float = 410.7
    parameter_exponent: float = 0.34
    data_exponent: float = 0.28

    def __post_init__(self) -> None:
        values = (
            self.irreducible_loss,
            self.parameter_coefficient,
            self.data_coefficient,
            self.parameter_exponent,
            self.data_exponent,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all loss-law coefficients must be positive")

    @property
    def model_compute_exponent(self) -> float:
        """a in N_opt(C) proportional to C**a."""

        return self.data_exponent / (
            self.parameter_exponent + self.data_exponent
        )

    @property
    def data_compute_exponent(self) -> float:
        """b in D_opt(C) proportional to C**b."""

        return self.parameter_exponent / (
            self.parameter_exponent + self.data_exponent
        )

    @property
    def allocation_constant(self) -> float:
        """G in the paper's closed-form efficient frontier."""

        numerator = self.parameter_exponent * self.parameter_coefficient
        denominator = self.data_exponent * self.data_coefficient
        exponent = 1.0 / (
            self.parameter_exponent + self.data_exponent
        )
        return (numerator / denominator) ** exponent

    def predict(self, params: float, tokens: float) -> float:
        _validate_positive("params", params)
        _validate_positive("tokens", tokens)
        model_term = self.parameter_coefficient / (
            params**self.parameter_exponent
        )
        data_term = self.data_coefficient / (
            tokens**self.data_exponent
        )
        return self.irreducible_loss + model_term + data_term

    def closed_form_optimum(self, compute: float) -> "TrainingPlan":
        """Minimize the printed rounded loss law under C = 6ND."""

        _validate_positive("compute", compute)
        effective_budget = compute / FLOPS_PER_PARAMETER_TOKEN
        params = (
            self.allocation_constant
            * effective_budget**self.model_compute_exponent
        )
        tokens = (
            effective_budget**self.data_compute_exponent
            / self.allocation_constant
        )
        return make_plan(
            "Printed-law optimum",
            params=params,
            tokens=tokens,
            loss_law=self,
        )


@dataclass(frozen=True)
class TrainingPlan:
    label: str
    params: float
    tokens: float
    approximate_flops: float
    tokens_per_parameter: float
    predicted_loss: float


def _validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def estimate_training_flops(params: float, tokens: float) -> float:
    """Return the common dense-Transformer approximation C ~= 6ND."""

    _validate_positive("params", params)
    _validate_positive("tokens", tokens)
    return FLOPS_PER_PARAMETER_TOKEN * params * tokens


def make_plan(
    label: str,
    params: float,
    tokens: float,
    loss_law: ChinchillaLossLaw,
) -> TrainingPlan:
    _validate_positive("params", params)
    _validate_positive("tokens", tokens)
    return TrainingPlan(
        label=label,
        params=params,
        tokens=tokens,
        approximate_flops=estimate_training_flops(params, tokens),
        tokens_per_parameter=tokens / params,
        predicted_loss=loss_law.predict(params, tokens),
    )


def ratio_rule_plan(
    compute: float,
    tokens_per_parameter: float = 20.0,
    *,
    loss_law: ChinchillaLossLaw | None = None,
) -> TrainingPlan:
    """Solve C=6ND together with the heuristic D=rN."""

    _validate_positive("compute", compute)
    _validate_positive("tokens_per_parameter", tokens_per_parameter)
    law = loss_law or ChinchillaLossLaw()
    params = math.sqrt(
        compute
        / (FLOPS_PER_PARAMETER_TOKEN * tokens_per_parameter)
    )
    tokens = tokens_per_parameter * params
    return make_plan(
        f"Ratio rule ({tokens_per_parameter:g}:1)",
        params=params,
        tokens=tokens,
        loss_law=law,
    )


def fixed_model_plan(
    compute: float,
    params: float,
    *,
    label: str | None = None,
    loss_law: ChinchillaLossLaw | None = None,
) -> TrainingPlan:
    """Spend a fixed FLOPs budget on a specified dense model size."""

    _validate_positive("compute", compute)
    _validate_positive("params", params)
    law = loss_law or ChinchillaLossLaw()
    tokens = compute / (FLOPS_PER_PARAMETER_TOKEN * params)
    return make_plan(
        label or f"Fixed model {format_quantity(params)}",
        params=params,
        tokens=tokens,
        loss_law=law,
    )


def _logspace(low: float, high: float, count: int) -> list[float]:
    _validate_positive("low", low)
    _validate_positive("high", high)
    if high <= low:
        raise ValueError("high must be greater than low")
    if count < 3:
        raise ValueError("count must be at least 3")
    ratio = high / low
    return [
        low * ratio ** (index / (count - 1))
        for index in range(count)
    ]


def isoflop_sweep(
    compute: float,
    min_params: float,
    max_params: float,
    points: int = 121,
    *,
    loss_law: ChinchillaLossLaw | None = None,
) -> list[TrainingPlan]:
    """Evaluate the fitted loss along one fixed-compute hyperbola."""

    law = loss_law or ChinchillaLossLaw()
    return [
        fixed_model_plan(
            compute,
            params,
            label=f"Sweep {index:03d}",
            loss_law=law,
        )
        for index, params in enumerate(
            _logspace(min_params, max_params, points)
        )
    ]


def best_plan(plans: Iterable[TrainingPlan]) -> TrainingPlan:
    candidates = list(plans)
    if not candidates:
        raise ValueError("plans must not be empty")
    return min(candidates, key=lambda plan: plan.predicted_loss)


def write_sweep_csv(
    path: Path,
    plans: Sequence[TrainingPlan],
) -> None:
    """Write an auditable IsoFLOP sweep that can be plotted elsewhere."""

    if not plans:
        raise ValueError("plans must not be empty")
    optimum = best_plan(plans)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "params",
                "tokens",
                "approximate_flops",
                "tokens_per_parameter",
                "predicted_loss",
                "is_grid_minimum",
            ),
        )
        writer.writeheader()
        for plan in plans:
            writer.writerow(
                {
                    "params": f"{plan.params:.12g}",
                    "tokens": f"{plan.tokens:.12g}",
                    "approximate_flops": (
                        f"{plan.approximate_flops:.12g}"
                    ),
                    "tokens_per_parameter": (
                        f"{plan.tokens_per_parameter:.12g}"
                    ),
                    "predicted_loss": f"{plan.predicted_loss:.12g}",
                    "is_grid_minimum": plan is optimum,
                }
            )


def format_quantity(value: float) -> str:
    _validate_positive("value", value)
    units = (
        (1e15, "Q"),
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    )
    for scale, suffix in units:
        if value >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:.3g}"


def _format_table(plans: Sequence[TrainingPlan]) -> str:
    headers = ("plan", "params", "tokens", "D/N", "pred. loss")
    rows = [
        (
            plan.label,
            format_quantity(plan.params),
            format_quantity(plan.tokens),
            f"{plan.tokens_per_parameter:.2f}",
            f"{plan.predicted_loss:.4f}",
        )
        for plan in plans
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(
            header.ljust(widths[index])
            for index, header in enumerate(headers)
        ),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(
            cell.ljust(widths[index])
            for index, cell in enumerate(row)
        )
        for row in rows
    )
    return "\n".join(lines)


def _self_test() -> None:
    law = ChinchillaLossLaw()
    assert math.isclose(
        law.model_compute_exponent,
        0.28 / 0.62,
        rel_tol=1e-12,
    )
    assert math.isclose(
        law.data_compute_exponent,
        0.34 / 0.62,
        rel_tol=1e-12,
    )
    assert math.isclose(
        law.model_compute_exponent + law.data_compute_exponent,
        1.0,
        rel_tol=1e-12,
    )

    ratio_plan = ratio_rule_plan(GOPHER_FLOPS, 20, loss_law=law)
    assert math.isclose(
        ratio_plan.approximate_flops,
        GOPHER_FLOPS,
        rel_tol=1e-12,
    )
    assert math.isclose(
        ratio_plan.params,
        69.2820323027551e9,
        rel_tol=1e-12,
    )

    closed = law.closed_form_optimum(GOPHER_FLOPS)
    assert math.isclose(
        closed.approximate_flops,
        GOPHER_FLOPS,
        rel_tol=1e-12,
    )
    model_marginal = (
        law.parameter_exponent
        * law.parameter_coefficient
        / closed.params**law.parameter_exponent
    )
    data_marginal = (
        law.data_exponent
        * law.data_coefficient
        / closed.tokens**law.data_exponent
    )
    assert math.isclose(model_marginal, data_marginal, rel_tol=1e-12)

    sweep = isoflop_sweep(
        GOPHER_FLOPS,
        min_params=1e9,
        max_params=1e12,
        points=601,
        loss_law=law,
    )
    grid_best = best_plan(sweep)
    assert abs(math.log(grid_best.params / closed.params)) < 0.02
    assert law.predict(1e9, 2e10) > law.predict(1e9, 2e11)

    try:
        ratio_rule_plan(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative compute was not rejected")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Chinchilla compute-allocation estimates."
    )
    parser.add_argument(
        "--compute",
        type=float,
        default=GOPHER_FLOPS,
        help="training budget in FLOPs (default: 5.76e23)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=20.0,
        help="tokens-per-parameter heuristic (default: 20)",
    )
    parser.add_argument(
        "--candidate-params",
        type=float,
        nargs="*",
        default=(70e9, 280e9),
        help="fixed model sizes to compare, as raw parameter counts",
    )
    parser.add_argument(
        "--min-params",
        type=float,
        default=1e8,
        help="minimum parameter count for IsoFLOP CSV sweep",
    )
    parser.add_argument(
        "--max-params",
        type=float,
        default=1e12,
        help="maximum parameter count for IsoFLOP CSV sweep",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=161,
        help="number of logarithmic sweep points",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="optional path for the IsoFLOP sweep CSV",
    )
    return parser.parse_args()


def main() -> None:
    _self_test()
    args = _parse_args()
    law = ChinchillaLossLaw()
    plans = [
        ratio_rule_plan(args.compute, args.ratio, loss_law=law),
        law.closed_form_optimum(args.compute),
    ]
    plans.extend(
        fixed_model_plan(
            args.compute,
            params,
            label=f"Fixed model {format_quantity(params)}",
            loss_law=law,
        )
        for params in args.candidate_params
    )

    print(f"Budget: {args.compute:.4e} FLOPs")
    print("Approximation: C = 6ND")
    print(_format_table(plans))
    print()
    print(
        "Approach 3 exponents: "
        f"N_opt ~ C^{law.model_compute_exponent:.4f}, "
        f"D_opt ~ C^{law.data_compute_exponent:.4f}"
    )
    print(
        "Note: the printed coefficients are empirical and rounded; they do "
        "not exactly reconstruct every paper table. Predicted loss is "
        "comparable only within the paper's setup."
    )

    if args.csv is not None:
        sweep = isoflop_sweep(
            args.compute,
            args.min_params,
            args.max_params,
            args.points,
            loss_law=law,
        )
        write_sweep_csv(args.csv, sweep)
        print(f"Wrote {len(sweep)} IsoFLOP rows to {args.csv}")

    print("All self-tests passed.")


if __name__ == "__main__":
    main()
