"""Dependency-free LoRA reference: training, rank, and weight merging.

This script uses Python lists and scalar loops so the complete LoRA mechanism
is visible without PyTorch or NumPy.  It demonstrates:

* a frozen base weight W0;
* trainable A and B with delta_W = (alpha / rank) * B @ A;
* Gaussian A plus zero B initialization, preserving the initial base output;
* gradients for A and B on a small mean-squared-error task;
* the parameter-count reduction r * (d_in + d_out) versus d_in * d_out; and
* merge/unmerge equivalence for zero-overhead inference.

It is an educational implementation, not a fast training library.

Run:
    python3 papers/to-2026/code/lora_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


Vector = list[float]
Matrix = list[Vector]
Example = tuple[Vector, Vector]


def _zeros(rows: int, columns: int) -> Matrix:
    return [[0.0 for _ in range(columns)] for _ in range(rows)]


def _copy(matrix: Matrix) -> Matrix:
    return [row[:] for row in matrix]


def _random_gaussian_matrix(
    rows: int,
    columns: int,
    rng: random.Random,
    *,
    std: float,
) -> Matrix:
    return [
        [rng.gauss(0.0, std) for _ in range(columns)]
        for _ in range(rows)
    ]


def _matvec(matrix: Matrix, vector: Sequence[float]) -> Vector:
    if not matrix or len(matrix[0]) != len(vector):
        raise ValueError("matrix and vector shapes are incompatible")
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix must be rectangular")
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    if not left or not right or not right[0]:
        raise ValueError("matrices must be non-empty")
    shared = len(left[0])
    if any(len(row) != shared for row in left):
        raise ValueError("left matrix must be rectangular")
    if len(right) != shared:
        raise ValueError("matrix shapes are incompatible")
    columns = len(right[0])
    if any(len(row) != columns for row in right):
        raise ValueError("right matrix must be rectangular")
    return [
        [
            sum(left_row[k] * right[k][j] for k in range(shared))
            for j in range(columns)
        ]
        for left_row in left
    ]


def _outer(left: Sequence[float], right: Sequence[float]) -> Matrix:
    return [[a * b for b in right] for a in left]


def _transpose_matvec(matrix: Matrix, vector: Sequence[float]) -> Vector:
    if len(matrix) != len(vector):
        raise ValueError("matrix and vector shapes are incompatible")
    columns = len(matrix[0])
    return [
        sum(matrix[i][j] * vector[i] for i in range(len(matrix)))
        for j in range(columns)
    ]


def _matrix_add(left: Matrix, right: Matrix, *, scale: float = 1.0) -> Matrix:
    if len(left) != len(right) or any(
        len(a) != len(b) for a, b in zip(left, right)
    ):
        raise ValueError("matrix shapes must match")
    return [
        [a + scale * b for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def _max_abs_difference(left: Matrix, right: Matrix) -> float:
    if len(left) != len(right) or any(
        len(a) != len(b) for a, b in zip(left, right)
    ):
        raise ValueError("matrix shapes must match")
    return max(
        abs(a - b)
        for left_row, right_row in zip(left, right)
        for a, b in zip(left_row, right_row)
    )


def _vector_max_abs_difference(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("vector shapes must match")
    return max(abs(a - b) for a, b in zip(left, right))


def _matrix_rank(matrix: Matrix, *, tolerance: float = 1e-10) -> int:
    """Compute rank by Gaussian elimination; sufficient for this tiny demo."""

    work = _copy(matrix)
    rows = len(work)
    columns = len(work[0])
    pivot_row = 0

    for column in range(columns):
        pivot = max(
            range(pivot_row, rows),
            key=lambda row: abs(work[row][column]),
            default=pivot_row,
        )
        if pivot_row >= rows or abs(work[pivot][column]) <= tolerance:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        pivot_value = work[pivot_row][column]
        for j in range(column, columns):
            work[pivot_row][j] /= pivot_value
        for row in range(rows):
            if row == pivot_row:
                continue
            factor = work[row][column]
            for j in range(column, columns):
                work[row][j] -= factor * work[pivot_row][j]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


@dataclass(frozen=True)
class TrainingStats:
    loss: float
    grad_a_norm: float
    grad_b_norm: float


class LoRALinear:
    """A linear layer with frozen W0 and trainable low-rank update B @ A."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: float,
        seed: int = 0,
    ) -> None:
        if not 0 < rank <= min(in_features, out_features):
            raise ValueError("rank must be in (0, min(in_features, out_features)]")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")

        rng = random.Random(seed)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.base_weight = _random_gaussian_matrix(
            out_features,
            in_features,
            rng,
            std=1.0 / math.sqrt(in_features),
        )
        # Original LoRA initialization: A is random and B is zero.  Therefore
        # B @ A == 0 and the initial function exactly matches the base layer.
        self.lora_a = _random_gaussian_matrix(
            rank,
            in_features,
            rng,
            std=0.02,
        )
        self.lora_b = _zeros(out_features, rank)
        self.merged = False
        self._merged_delta: Matrix | None = None

    @property
    def trainable_parameter_count(self) -> int:
        return self.rank * (self.in_features + self.out_features)

    @property
    def full_finetune_parameter_count(self) -> int:
        return self.in_features * self.out_features

    def delta_weight(self) -> Matrix:
        return [
            [self.scaling * value for value in row]
            for row in _matmul(self.lora_b, self.lora_a)
        ]

    def forward(self, inputs: Sequence[float]) -> Vector:
        if len(inputs) != self.in_features:
            raise ValueError("input width does not match in_features")
        base_output = _matvec(self.base_weight, inputs)
        if self.merged:
            return base_output

        compressed = _matvec(self.lora_a, inputs)
        adaptation = _matvec(self.lora_b, compressed)
        return [
            base + self.scaling * update
            for base, update in zip(base_output, adaptation)
        ]

    def evaluate_loss(self, examples: Sequence[Example]) -> float:
        if not examples:
            raise ValueError("at least one training example is required")
        total = 0.0
        for inputs, target in examples:
            prediction = self.forward(inputs)
            total += 0.5 * sum(
                (actual - expected) ** 2
                for actual, expected in zip(prediction, target)
            )
        return total / len(examples)

    def train_step(
        self,
        examples: Sequence[Example],
        *,
        learning_rate: float,
    ) -> TrainingStats:
        if self.merged:
            raise RuntimeError("unmerge before changing LoRA parameters")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not examples:
            raise ValueError("at least one training example is required")

        grad_a = _zeros(self.rank, self.in_features)
        grad_b = _zeros(self.out_features, self.rank)
        total_loss = 0.0

        for inputs, target in examples:
            compressed = _matvec(self.lora_a, inputs)
            base_output = _matvec(self.base_weight, inputs)
            adaptation = _matvec(self.lora_b, compressed)
            prediction = [
                base + self.scaling * update
                for base, update in zip(base_output, adaptation)
            ]
            error = [
                actual - expected
                for actual, expected in zip(prediction, target)
            ]
            total_loss += 0.5 * sum(value * value for value in error)

            # dL/dB = scaling * error @ (A x)^T
            sample_grad_b = _outer(error, compressed)
            # dL/dA = scaling * (B^T error) @ x^T
            back_to_rank = _transpose_matvec(self.lora_b, error)
            sample_grad_a = _outer(back_to_rank, inputs)
            for i in range(self.out_features):
                for j in range(self.rank):
                    grad_b[i][j] += self.scaling * sample_grad_b[i][j]
            for i in range(self.rank):
                for j in range(self.in_features):
                    grad_a[i][j] += self.scaling * sample_grad_a[i][j]

        inverse_batch = 1.0 / len(examples)
        grad_a_norm = math.sqrt(
            sum(value * value for row in grad_a for value in row)
        ) * inverse_batch
        grad_b_norm = math.sqrt(
            sum(value * value for row in grad_b for value in row)
        ) * inverse_batch

        # Simultaneous SGD update: both gradients were computed using the old
        # A and B.  The frozen base_weight is intentionally untouched.
        for i in range(self.rank):
            for j in range(self.in_features):
                self.lora_a[i][j] -= (
                    learning_rate * inverse_batch * grad_a[i][j]
                )
        for i in range(self.out_features):
            for j in range(self.rank):
                self.lora_b[i][j] -= (
                    learning_rate * inverse_batch * grad_b[i][j]
                )

        return TrainingStats(
            loss=total_loss * inverse_batch,
            grad_a_norm=grad_a_norm,
            grad_b_norm=grad_b_norm,
        )

    def merge(self) -> None:
        if self.merged:
            raise RuntimeError("LoRA weights are already merged")
        delta = self.delta_weight()
        self.base_weight = _matrix_add(self.base_weight, delta)
        self._merged_delta = delta
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged or self._merged_delta is None:
            raise RuntimeError("LoRA weights are not merged")
        self.base_weight = _matrix_add(
            self.base_weight,
            self._merged_delta,
            scale=-1.0,
        )
        self._merged_delta = None
        self.merged = False


def _build_low_rank_task(layer: LoRALinear, *, seed: int) -> list[Example]:
    """Create targets whose required change has rank at most layer.rank."""

    rng = random.Random(seed)
    target_a = _random_gaussian_matrix(
        layer.rank,
        layer.in_features,
        rng,
        std=0.35,
    )
    target_b = _random_gaussian_matrix(
        layer.out_features,
        layer.rank,
        rng,
        std=0.35,
    )
    target_delta = _matmul(target_b, target_a)

    examples: list[Example] = []
    for _ in range(36):
        inputs = [rng.uniform(-1.0, 1.0) for _ in range(layer.in_features)]
        base = _matvec(layer.base_weight, inputs)
        update = _matvec(target_delta, inputs)
        examples.append((inputs, [a + b for a, b in zip(base, update)]))
    return examples


def _demo() -> None:
    layer = LoRALinear(
        in_features=12,
        out_features=10,
        rank=2,
        alpha=2.0,
        seed=7,
    )
    examples = _build_low_rank_task(layer, seed=19)
    base_before_training = _copy(layer.base_weight)
    a_before_first_step = _copy(layer.lora_a)
    probe = examples[0][0]

    # Because B starts at zero, the adapter initially changes nothing.
    base_probe = _matvec(layer.base_weight, probe)
    assert _vector_max_abs_difference(layer.forward(probe), base_probe) < 1e-15
    initial_loss = layer.evaluate_loss(examples)

    first_step = layer.train_step(examples, learning_rate=0.08)
    # At step 1, dL/dA contains B^T and is exactly zero; dL/dB is non-zero.
    assert first_step.grad_a_norm == 0.0
    assert first_step.grad_b_norm > 0.0
    assert _max_abs_difference(layer.lora_a, a_before_first_step) == 0.0

    for _ in range(2499):
        layer.train_step(examples, learning_rate=0.08)
    final_loss = layer.evaluate_loss(examples)

    assert final_loss < initial_loss * 1e-4
    assert _max_abs_difference(layer.base_weight, base_before_training) == 0.0
    assert _matrix_rank(layer.delta_weight()) <= layer.rank

    unmerged_output = layer.forward(probe)
    layer.merge()
    merged_output = layer.forward(probe)
    assert _vector_max_abs_difference(unmerged_output, merged_output) < 1e-12
    layer.unmerge()
    restored_output = layer.forward(probe)
    assert _vector_max_abs_difference(unmerged_output, restored_output) < 1e-12

    print(f"base parameters:      {layer.full_finetune_parameter_count}")
    print(f"LoRA parameters:      {layer.trainable_parameter_count}")
    print(f"initial delta rank:   0")
    print(f"learned delta rank:   {_matrix_rank(layer.delta_weight())}")
    print(f"initial loss:         {initial_loss:.8f}")
    print(f"final loss:           {final_loss:.8f}")
    print("base weight changed:  no")
    print("merge equivalence:    passed")


if __name__ == "__main__":
    _demo()
