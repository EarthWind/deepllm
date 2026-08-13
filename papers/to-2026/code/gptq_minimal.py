#!/usr/bin/env python3
"""A zero-dependency, inspectable GPTQ teaching implementation.

This file intentionally favors the equations over performance.  It implements
the fixed shared-column order and the exact inverse-Hessian downdate inherited
from OBQ.  Production GPTQ replaces the repeated downdate with a Cholesky
factor and batches the outer updates into matrix multiplications.

Run:
    python3 papers/to-2026/code/gptq_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence


Matrix = list[list[float]]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def transpose(a: Sequence[Sequence[float]]) -> Matrix:
    return [list(col) for col in zip(*a)]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def inverse(a: Sequence[Sequence[float]]) -> Matrix:
    """Invert a small dense matrix with pivoted Gauss-Jordan elimination."""
    n = len(a)
    aug = [list(row) + [float(i == j) for j in range(n)] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular Hessian: increase damping or calibration diversity")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [x - factor * y for x, y in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


@dataclass(frozen=True)
class AffineRowQuantizer:
    """Uniform asymmetric min-max grid, including zero in the fitted range."""

    bits: int
    scale: float
    zero: int

    @classmethod
    def fit(cls, row: Sequence[float], bits: int) -> "AffineRowQuantizer":
        if bits < 2:
            raise ValueError("this demo expects at least 2 bits")
        qmax = (1 << bits) - 1
        minimum = min(0.0, min(row))
        maximum = max(0.0, max(row))
        if minimum == maximum:
            return cls(bits=bits, scale=1.0, zero=0)
        scale = (maximum - minimum) / qmax
        zero = round(-minimum / scale)
        return cls(bits=bits, scale=scale, zero=max(0, min(qmax, zero)))

    def quantize(self, value: float) -> float:
        qmax = (1 << self.bits) - 1
        code = round(value / self.scale) + self.zero
        code = max(0, min(qmax, code))
        return self.scale * (code - self.zero)


def fit_row_quantizers(weights: Sequence[Sequence[float]], bits: int) -> list[AffineRowQuantizer]:
    return [AffineRowQuantizer.fit(row, bits) for row in weights]


def hessian_from_samples(samples: Sequence[Sequence[float]], damp_percent: float) -> Matrix:
    """Return 2 X X^T / n + lambda I for samples stored as [n, columns]."""
    if not samples:
        raise ValueError("at least one calibration sample is required")
    columns = len(samples[0])
    if any(len(sample) != columns for sample in samples):
        raise ValueError("all calibration samples must have the same width")
    x = transpose(samples)  # Paper notation: X is [columns, tokens].
    h = matmul(x, transpose(x))
    factor = 2.0 / len(samples)
    h = [[factor * value for value in row] for row in h]
    mean_diagonal = sum(h[i][i] for i in range(columns)) / columns
    damping = damp_percent * mean_diagonal
    for i in range(columns):
        h[i][i] += damping
    return h


def round_to_nearest(weights: Sequence[Sequence[float]], bits: int) -> Matrix:
    quantizers = fit_row_quantizers(weights, bits)
    return [[quantizers[r].quantize(value) for value in row] for r, row in enumerate(weights)]


def remove_first_inverse(hinv: Matrix) -> Matrix:
    """Schur-complement downdate after the first active column is quantized."""
    if len(hinv) == 1:
        return []
    diagonal = hinv[0][0]
    return [
        [hinv[i][j] - hinv[i][0] * hinv[0][j] / diagonal for j in range(1, len(hinv))]
        for i in range(1, len(hinv))
    ]


def gptq_reference(
    weights: Sequence[Sequence[float]],
    calibration_samples: Sequence[Sequence[float]],
    bits: int = 3,
    damp_percent: float = 0.01,
) -> tuple[Matrix, list[dict[str, float]]]:
    """Quantize W one shared input column at a time with GPTQ compensation.

    W has shape [output_features, input_features].  Every row uses the same
    column order.  The quantization grid is fitted once per original row.
    """
    if not weights or not weights[0]:
        raise ValueError("weights must be a non-empty matrix")
    rows, columns = len(weights), len(weights[0])
    if any(len(row) != columns for row in weights):
        raise ValueError("weight rows must have the same width")
    if any(len(sample) != columns for sample in calibration_samples):
        raise ValueError("sample width must equal input_features")

    working = [list(row) for row in weights]
    quantized = zeros(rows, columns)
    quantizers = fit_row_quantizers(weights, bits)
    hinv = inverse(hessian_from_samples(calibration_samples, damp_percent))
    trace: list[dict[str, float]] = []

    for column in range(columns):
        diagonal = hinv[0][0]
        mean_abs_error = 0.0
        mean_abs_compensation = 0.0
        for row in range(rows):
            current = working[row][column]
            qvalue = quantizers[row].quantize(current)
            # Paper/official-code convention: err = (w - q) / [H^-1]_{qq}.
            scaled_error = (current - qvalue) / diagonal
            mean_abs_error += abs(current - qvalue)
            for offset in range(len(hinv)):
                target = column + offset
                delta = scaled_error * hinv[0][offset]
                working[row][target] -= delta
                if offset:
                    mean_abs_compensation += abs(delta)
            quantized[row][column] = qvalue

        trace.append(
            {
                "column": float(column),
                "hinv_diagonal": diagonal,
                "mean_abs_rounding_error": mean_abs_error / rows,
                "mean_abs_future_update": mean_abs_compensation / max(1, rows * (columns - column - 1)),
            }
        )
        hinv = remove_first_inverse(hinv)

    return quantized, trace


def reconstruction_mse(
    original: Sequence[Sequence[float]],
    quantized: Sequence[Sequence[float]],
    samples: Sequence[Sequence[float]],
) -> float:
    """Compute mean ||W x - Q x||^2 over calibration samples and output rows."""
    total = 0.0
    count = 0
    for sample in samples:
        for wrow, qrow in zip(original, quantized):
            y = sum(w * x for w, x in zip(wrow, sample))
            yhat = sum(q * x for q, x in zip(qrow, sample))
            total += (y - yhat) ** 2
            count += 1
    return total / count


def ideal_weight_payload(parameter_count: int, bits: int) -> tuple[float, float]:
    """Return ideal payload in decimal GB and binary GiB, excluding metadata."""
    byte_count = parameter_count * bits / 8
    return byte_count / 1e9, byte_count / (1024**3)


def make_correlated_demo(seed: int = 7) -> tuple[Matrix, Matrix]:
    rng = random.Random(seed)
    samples: Matrix = []
    for _ in range(96):
        z0, z1, z2, z3 = (rng.gauss(0.0, 1.0) for _ in range(4))
        noise = lambda: rng.gauss(0.0, 0.12)
        samples.append(
            [
                z0,
                0.96 * z0 + noise(),
                z1,
                -0.90 * z1 + noise(),
                z2,
                0.82 * z2 + noise(),
                z3,
                0.62 * z0 + 0.68 * z3 + noise(),
            ]
        )
    weights = [[rng.uniform(-1.8, 1.8) for _ in range(8)] for _ in range(5)]
    return weights, samples


def format_row(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:+.3f}" for value in values) + "]"


def main() -> None:
    weights, samples = make_correlated_demo()
    bits = 3
    rtn = round_to_nearest(weights, bits)
    gptq, trace = gptq_reference(weights, samples, bits=bits)
    rtn_mse = reconstruction_mse(weights, rtn, samples)
    gptq_mse = reconstruction_mse(weights, gptq, samples)

    print("GPTQ fixed-order teaching demo")
    print(f"shape                 : {len(weights)} x {len(weights[0])}")
    print(f"calibration samples   : {len(samples)}")
    print(f"weight precision      : {bits}-bit")
    print(f"RTN reconstruction MSE: {rtn_mse:.8f}")
    print(f"GPTQ reconstruction MSE: {gptq_mse:.8f}")
    print(f"relative reduction    : {(1.0 - gptq_mse / rtn_mse) * 100:.2f}%")
    example_row = 1  # This deterministic row contains a GPTQ/RTN decision difference.
    print(f"\nexample output row {example_row}")
    print("FP weights:", format_row(weights[example_row]))
    print("RTN       :", format_row(rtn[example_row]))
    print("GPTQ      :", format_row(gptq[example_row]))
    print("\nfirst three shared-column decisions")
    for item in trace[:3]:
        print(
            f"column={int(item['column'])} "
            f"diag={item['hinv_diagonal']:.6f} "
            f"round_err={item['mean_abs_rounding_error']:.6f} "
            f"future_update={item['mean_abs_future_update']:.6f}"
        )

    print("\nideal 175B weight payload (metadata/workspace excluded)")
    for precision in (16, 4, 3):
        gb, gib = ideal_weight_payload(175_000_000_000, precision)
        print(f"{precision:>2}-bit: {gb:7.2f} GB = {gib:7.2f} GiB")

    if not gptq_mse < rtn_mse:
        raise AssertionError("the deterministic correlated demo should improve over RTN")


if __name__ == "__main__":
    main()
