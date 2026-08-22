"""A tiny, dependency-free toy model of superposition.

We learn a linear tied autoencoder W^T W for sparse feature vectors x.  When
there are more features than hidden dimensions, the optimizer can place several
feature directions in one neuron/space using nearly-orthogonal (often
approximately geometric) directions.  This is a pedagogical simulation, not a
reproduction of every experiment in the paper.

Run ``python toy_superposition.py --test`` or just ``python toy_superposition.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import random
from typing import Sequence


Matrix = list[list[float]]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mat_vec(matrix: Matrix, vector: Sequence[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def transpose_mat_vec(matrix: Matrix, vector: Sequence[float]) -> list[float]:
    if not matrix:
        return []
    return [sum(matrix[i][j] * vector[i] for i in range(len(matrix))) for j in range(len(matrix[0]))]


def sample_sparse_features(n_features: int, sparsity: float, samples: int, rng: random.Random) -> Matrix:
    """Bernoulli feature matrix X with shape [samples, n_features]."""
    return [[1.0 if rng.random() < sparsity else 0.0 for _ in range(n_features)] for _ in range(samples)]


def normalize_columns(w: Matrix) -> None:
    d = len(w)
    n = len(w[0])
    for j in range(n):
        norm = math.sqrt(sum(w[i][j] ** 2 for i in range(d))) or 1.0
        for i in range(d):
            w[i][j] /= norm


def train_tied_autoencoder(
    x_rows: Matrix, hidden_dim: int, *, steps: int = 2000, lr: float = 0.03, seed: int = 0
) -> Matrix:
    """Minimize ||W^T W x - x||² with SGD and unit-norm feature columns."""
    if not x_rows or hidden_dim < 1:
        raise ValueError("need non-empty data and positive hidden dimension")
    n_features = len(x_rows[0])
    rng = random.Random(seed)
    w = [[rng.gauss(0.0, 0.15) for _ in range(n_features)] for _ in range(hidden_dim)]
    normalize_columns(w)
    for step in range(steps):
        x = x_rows[step % len(x_rows)]
        h = mat_vec(w, x)
        reconstruction = transpose_mat_vec(w, h)
        error = [reconstruction[j] - x[j] for j in range(n_features)]
        # d ||W^T W x - x||² / dW = 2 h e^T + 2 (W e) x^T
        w_error = mat_vec(w, error)
        for i in range(hidden_dim):
            for j in range(n_features):
                grad = 2.0 * h[i] * error[j] + 2.0 * w_error[i] * x[j]
                w[i][j] -= lr * grad
        if step % 25 == 0:
            normalize_columns(w)
    normalize_columns(w)
    return w


def reconstruction_mse(w: Matrix, x_rows: Matrix) -> float:
    total = 0.0
    count = 0
    for x in x_rows:
        y = transpose_mat_vec(w, mat_vec(w, x))
        total += sum((a - b) ** 2 for a, b in zip(y, x)) / len(x)
        count += 1
    return total / count


def feature_coherence(w: Matrix) -> float:
    """Mean absolute cosine similarity between distinct feature directions."""
    n = len(w[0])
    values = []
    for a in range(n):
        for b in range(a + 1, n):
            va = [w[i][a] for i in range(len(w))]
            vb = [w[i][b] for i in range(len(w))]
            values.append(abs(dot(va, vb)))
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class SweepResult:
    hidden_dim: int
    n_features: int
    sparsity: float
    mse: float
    coherence: float


def run_sweep() -> list[SweepResult]:
    results = []
    for hidden_dim, n_features in [(8, 8), (4, 8), (2, 8)]:
        rng = random.Random(7)
        x = sample_sparse_features(n_features, 0.12, 320, rng)
        w = train_tied_autoencoder(x, hidden_dim, steps=2500, lr=0.025, seed=7)
        results.append(SweepResult(hidden_dim, n_features, 0.12, reconstruction_mse(w, x), feature_coherence(w)))
    return results


def demo() -> None:
    print("sparse features, tied linear autoencoder")
    print("hidden  features  sparsity   mse       mean |cos(feature_i, feature_j)|")
    for result in run_sweep():
        print(f"{result.hidden_dim:>6}  {result.n_features:>8}  {result.sparsity:>8.2f}  {result.mse:>7.4f}  {result.coherence:>10.4f}")
    print("\nInterpretation: when features exceed dimensions, several directions share the same hidden space;")
    print("the exact geometry depends on sparsity, loss, optimization and normalization.")


def run_tests() -> None:
    rng = random.Random(1)
    x = sample_sparse_features(4, 0.2, 20, rng)
    assert len(x) == 20 and len(x[0]) == 4
    w = train_tied_autoencoder(x, 2, steps=80, lr=0.02, seed=1)
    assert len(w) == 2 and len(w[0]) == 4
    assert reconstruction_mse(w, x) >= 0.0
    assert 0.0 <= feature_coherence(w) <= 1.0
    print("all tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_tests() if args.test else demo()
