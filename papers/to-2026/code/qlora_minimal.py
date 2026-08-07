#!/usr/bin/env python3
"""Dependency-free, inspectable QLoRA teaching implementation.

This file demonstrates the mechanics that are easy to hide behind CUDA kernels:

1. block-wise absmax quantization into the 16-value NF4 lookup table;
2. packing two 4-bit indices into one byte;
3. double-quantizing the per-block scales;
4. dequantizing the frozen base weight during the forward pass;
5. backpropagating only into the high-precision LoRA matrices;
6. reproducing the paper's 4.500 -> 4.127 bits/parameter memory ledger.

It is deliberately a CPU reference, not a replacement for bitsandbytes.  The paper
uses FP8 for the second quantization; to stay dependency-free, this file uses a
symmetric int8 proxy with the same 8-bit storage cost.  Production training should
use the tested CUDA/ROCm/XPU kernels supplied by bitsandbytes.

Run:
    python papers/to-2026/code/qlora_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence


# The exact lookup table used by bitsandbytes for the NF4 quantization type.
# A stored nibble is an INDEX into this table; NF4 is not sign/exponent/mantissa.
NF4 = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


def chunks(values: Sequence[float], size: int) -> Iterable[Sequence[float]]:
    """Yield consecutive slices without padding the last block."""
    if size <= 0:
        raise ValueError("block size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def nearest_index(value: float, codebook: Sequence[float] = NF4) -> int:
    """Return the nearest codebook entry; ties go to the smaller index."""
    return min(range(len(codebook)), key=lambda index: abs(codebook[index] - value))


def pack_nibbles(indices: Sequence[int]) -> bytes:
    """Pack two unsigned 4-bit values into each byte (low nibble first)."""
    packed = bytearray()
    for offset in range(0, len(indices), 2):
        low = indices[offset]
        high = indices[offset + 1] if offset + 1 < len(indices) else 0
        if not 0 <= low < 16 or not 0 <= high < 16:
            raise ValueError("a packed NF4 index must be in [0, 15]")
        packed.append(low | (high << 4))
    return bytes(packed)


def unpack_nibbles(packed: bytes, count: int) -> list[int]:
    """Reverse :func:`pack_nibbles`, dropping a possible padding nibble."""
    if count < 0 or count > 2 * len(packed):
        raise ValueError("invalid number of nibbles")
    result: list[int] = []
    for value in packed:
        result.extend((value & 0x0F, value >> 4))
    return result[:count]


@dataclass(frozen=True)
class DoubleQuantizedScales:
    """An inspectable 8-bit approximation of the paper's FP8 scale storage.

    The first-level absmax scales are positive and clustered.  Following QLoRA,
    we subtract their mean, then quantize residuals in groups of 256.  Each int8
    code costs eight bits and each group keeps one FP32 second-level absmax.
    Python object overhead is intentionally excluded from the theoretical ledger.
    """

    codes: bytes
    second_scales: tuple[float, ...]
    mean: float
    count: int
    block_size: int = 256

    @classmethod
    def from_scales(
        cls, scales: Sequence[float], block_size: int = 256
    ) -> "DoubleQuantizedScales":
        if not scales:
            return cls(b"", (), 0.0, 0, block_size)

        mean = sum(scales) / len(scales)
        residuals = [scale - mean for scale in scales]
        encoded = bytearray()
        second_scales: list[float] = []

        for block in chunks(residuals, block_size):
            absmax = max((abs(value) for value in block), default=0.0)
            second_scales.append(absmax)
            for value in block:
                signed_code = 0 if absmax == 0.0 else round(127.0 * value / absmax)
                signed_code = max(-127, min(127, signed_code))
                encoded.append(signed_code + 127)  # map [-127, 127] to [0, 254]

        return cls(bytes(encoded), tuple(second_scales), mean, len(scales), block_size)

    def dequantize(self) -> list[float]:
        if len(self.codes) != self.count:
            raise ValueError("corrupt double-quantized scale state")
        result: list[float] = []
        for index, encoded in enumerate(self.codes):
            absmax = self.second_scales[index // self.block_size]
            residual = ((encoded - 127) / 127.0) * absmax
            result.append(max(0.0, self.mean + residual))
        return result


@dataclass(frozen=True)
class QuantizedMatrix:
    """Row-major block-wise NF4 matrix with a frozen double-quantized scale state."""

    rows: int
    columns: int
    packed_indices: bytes
    scales: DoubleQuantizedScales
    weight_block_size: int = 64

    @classmethod
    def quantize(
        cls,
        matrix: Sequence[Sequence[float]],
        weight_block_size: int = 64,
        scale_block_size: int = 256,
    ) -> "QuantizedMatrix":
        rows = len(matrix)
        columns = len(matrix[0]) if rows else 0
        if rows == 0 or columns == 0:
            raise ValueError("matrix must be non-empty")
        if any(len(row) != columns for row in matrix):
            raise ValueError("matrix rows must have equal length")

        flat = [value for row in matrix for value in row]
        indices: list[int] = []
        scales: list[float] = []

        for block in chunks(flat, weight_block_size):
            absmax = max((abs(value) for value in block), default=0.0)
            scales.append(absmax)
            if absmax == 0.0:
                zero_index = NF4.index(0.0)
                indices.extend([zero_index] * len(block))
            else:
                indices.extend(nearest_index(value / absmax) for value in block)

        return cls(
            rows=rows,
            columns=columns,
            packed_indices=pack_nibbles(indices),
            scales=DoubleQuantizedScales.from_scales(scales, scale_block_size),
            weight_block_size=weight_block_size,
        )

    def dequantize(self) -> list[list[float]]:
        """Materialize the BF16-compute analogue as Python float values."""
        count = self.rows * self.columns
        indices = unpack_nibbles(self.packed_indices, count)
        scales = self.scales.dequantize()
        flat = [
            NF4[index] * scales[position // self.weight_block_size]
            for position, index in enumerate(indices)
        ]
        return [
            flat[start : start + self.columns]
            for start in range(0, len(flat), self.columns)
        ]

    def linear(self, vector: Sequence[float]) -> list[float]:
        """Compute y = W x after on-demand dequantization of frozen W."""
        if len(vector) != self.columns:
            raise ValueError("input dimension does not match the matrix")
        weight = self.dequantize()
        return [sum(w * x for w, x in zip(row, vector)) for row in weight]

    def theoretical_bits(self) -> int:
        """Payload bits under the paper's 4-bit + 8-bit/FP32 scale accounting."""
        weight_count = self.rows * self.columns
        first_scale_count = math.ceil(weight_count / self.weight_block_size)
        second_scale_count = math.ceil(first_scale_count / self.scales.block_size)
        return 4 * weight_count + 8 * first_scale_count + 32 * second_scale_count


def zeros(rows: int, columns: int) -> list[list[float]]:
    return [[0.0 for _ in range(columns)] for _ in range(rows)]


def random_matrix(
    rows: int, columns: int, rng: random.Random, std: float
) -> list[list[float]]:
    return [[rng.gauss(0.0, std) for _ in range(columns)] for _ in range(rows)]


@dataclass
class QLoRALinear:
    """A frozen NF4 base plus trainable full-precision LoRA factors.

    Shapes follow W[out, in], A[rank, in], B[out, rank]:

        y = dequant(W_NF4) x + (alpha / rank) B A x

    In a real implementation, fused kernels avoid materializing the full W matrix.
    """

    base: QuantizedMatrix
    rank: int
    alpha: float
    a: list[list[float]]
    b: list[list[float]]

    @classmethod
    def create(
        cls, base: QuantizedMatrix, rank: int, alpha: float, seed: int = 0
    ) -> "QLoRALinear":
        if rank <= 0:
            raise ValueError("rank must be positive")
        rng = random.Random(seed)
        # LoRA commonly starts one factor randomly and the other at zero, making
        # the initial delta exactly zero while preserving an initial gradient.
        a = random_matrix(rank, base.columns, rng, std=0.02)
        b = zeros(base.rows, rank)
        return cls(base, rank, alpha, a, b)

    @property
    def multiplier(self) -> float:
        return self.alpha / self.rank

    def forward_with_cache(
        self, vector: Sequence[float]
    ) -> tuple[list[float], list[float]]:
        base_output = self.base.linear(vector)
        low_rank = [
            sum(value * x for value, x in zip(row, vector)) for row in self.a
        ]
        output = [
            base_value
            + self.multiplier
            * sum(self.b[out_index][r] * low_rank[r] for r in range(self.rank))
            for out_index, base_value in enumerate(base_output)
        ]
        return output, low_rank

    def forward(self, vector: Sequence[float]) -> list[float]:
        return self.forward_with_cache(vector)[0]

    def train_batch(
        self,
        examples: Sequence[tuple[Sequence[float], Sequence[float]]],
        learning_rate: float,
    ) -> float:
        """One full-batch SGD step; gradients are created only for A and B."""
        if not examples:
            raise ValueError("training batch must be non-empty")

        grad_a = zeros(self.rank, self.base.columns)
        grad_b = zeros(self.base.rows, self.rank)
        squared_error = 0.0
        normalizer = len(examples) * self.base.rows

        for vector, target in examples:
            prediction, low_rank = self.forward_with_cache(vector)
            if len(target) != self.base.rows:
                raise ValueError("target dimension does not match the layer")

            grad_output: list[float] = []
            for predicted, expected in zip(prediction, target):
                error = predicted - expected
                squared_error += error * error
                grad_output.append(2.0 * error / normalizer)

            # Important: both gradients below use the same pre-update parameters.
            for out_index in range(self.base.rows):
                for r in range(self.rank):
                    grad_b[out_index][r] += (
                        self.multiplier * grad_output[out_index] * low_rank[r]
                    )

            for r in range(self.rank):
                grad_low_rank = self.multiplier * sum(
                    grad_output[out_index] * self.b[out_index][r]
                    for out_index in range(self.base.rows)
                )
                for in_index, value in enumerate(vector):
                    grad_a[r][in_index] += grad_low_rank * value

        for r in range(self.rank):
            for in_index in range(self.base.columns):
                self.a[r][in_index] -= learning_rate * grad_a[r][in_index]
        for out_index in range(self.base.rows):
            for r in range(self.rank):
                self.b[out_index][r] -= learning_rate * grad_b[out_index][r]

        return squared_error / normalizer


def mean_squared_error(
    model: QLoRALinear,
    examples: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> float:
    total = 0.0
    count = 0
    for vector, target in examples:
        for predicted, expected in zip(model.forward(vector), target):
            total += (predicted - expected) ** 2
            count += 1
    return total / count


def make_low_rank_task(
    base: QuantizedMatrix,
    rank: int,
    count: int,
    seed: int,
) -> list[tuple[list[float], list[float]]]:
    """Create targets that differ from the quantized base by a rank-r update."""
    rng = random.Random(seed)
    true_a = random_matrix(rank, base.columns, rng, std=0.45)
    true_b = random_matrix(base.rows, rank, rng, std=0.45)
    examples: list[tuple[list[float], list[float]]] = []

    for _ in range(count):
        vector = [rng.gauss(0.0, 1.0) for _ in range(base.columns)]
        base_output = base.linear(vector)
        latent = [sum(w * x for w, x in zip(row, vector)) for row in true_a]
        target = [
            base_output[out_index]
            + sum(true_b[out_index][r] * latent[r] for r in range(rank))
            for out_index in range(base.rows)
        ]
        examples.append((vector, target))
    return examples


def qlora_bits_per_parameter(
    weight_block_size: int = 64, scale_block_size: int = 256
) -> tuple[float, float, float]:
    """Return (without DQ, with DQ, saved), matching the paper's ledger."""
    without_double_quant = 4.0 + 32.0 / weight_block_size
    with_double_quant = (
        4.0
        + 8.0 / weight_block_size
        + 32.0 / (weight_block_size * scale_block_size)
    )
    return (
        without_double_quant,
        with_double_quant,
        without_double_quant - with_double_quant,
    )


def decimal_gigabytes(parameter_count: int, bits_per_parameter: float) -> float:
    """Theoretical payload in decimal GB; excludes activations and allocator overhead."""
    return parameter_count * bits_per_parameter / 8.0 / 1_000_000_000


def run_self_checks() -> None:
    sample = list(range(16)) + [15, 0, 7]
    assert unpack_nibbles(pack_nibbles(sample), len(sample)) == sample
    assert 0.0 in NF4 and len(NF4) == 16

    original = [[0.0, -0.2, 0.5], [1.0, -1.1, 0.04]]
    quantized = QuantizedMatrix.quantize(original, weight_block_size=4)
    restored = quantized.dequantize()
    assert len(restored) == 2 and len(restored[0]) == 3
    assert all(math.isfinite(value) for row in restored for value in row)

    without_dq, with_dq, saved = qlora_bits_per_parameter()
    assert math.isclose(without_dq, 4.5)
    assert math.isclose(with_dq, 4.126953125)
    assert math.isclose(saved, 0.373046875)


def main() -> None:
    run_self_checks()
    rng = random.Random(7)
    full_precision_weight = random_matrix(5, 8, rng, std=0.35)
    frozen_base = QuantizedMatrix.quantize(full_precision_weight)
    frozen_snapshot = (frozen_base.packed_indices, frozen_base.scales)

    examples = make_low_rank_task(frozen_base, rank=2, count=96, seed=11)
    model = QLoRALinear.create(frozen_base, rank=2, alpha=2.0, seed=19)
    initial_loss = mean_squared_error(model, examples)

    for step in range(1, 401):
        model.train_batch(examples, learning_rate=0.16)
        if step in {1, 25, 100, 200, 400}:
            print(f"step={step:>3}  mse={mean_squared_error(model, examples):.8f}")

    final_loss = mean_squared_error(model, examples)
    assert final_loss < initial_loss * 0.01, (initial_loss, final_loss)
    assert frozen_snapshot == (frozen_base.packed_indices, frozen_base.scales)

    without_dq, with_dq, saved = qlora_bits_per_parameter()
    parameters = 65_000_000_000
    print("\nPaper memory ledger (payload only):")
    print(f"  NF4 + FP32 scales : {without_dq:.6f} bits/parameter")
    print(f"  NF4 + double quant: {with_dq:.6f} bits/parameter")
    print(f"  saved              : {saved:.6f} bits/parameter")
    print(f"  65B saving         : {decimal_gigabytes(parameters, saved):.2f} GB")
    print(f"\nbase frozen: yes; loss {initial_loss:.6f} -> {final_loss:.8f}")


if __name__ == "__main__":
    main()
