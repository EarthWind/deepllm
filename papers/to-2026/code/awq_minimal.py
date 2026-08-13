#!/usr/bin/env python3
"""AWQ 的零依赖教学实现。

它刻意只实现论文中最重要的机制：

1. 用校准 activation 的逐输入通道平均绝对值衡量 saliency；
2. 在 alpha 网格上搜索 s = mean(|X|) ** alpha；
3. 验证 W @ X = (W * s) @ (X / s) 的等价变换；
4. 对缩放后的权重做 group-wise 低比特伪量化；
5. 可选地搜索逐输出行、逐 group 的 clipping 阈值。

代码只依赖 Python 标准库，适合核对公式，不是高性能量化器或推理 kernel。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


Matrix = list[list[float]]


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def linear(weight: Matrix, samples: Matrix) -> Matrix:
    """计算一批行向量输入：Y = X @ W^T。"""
    return [
        [sum(w * x for w, x in zip(row, sample)) for row in weight]
        for sample in samples
    ]


def mse(a: Matrix, b: Matrix) -> float:
    values = [
        (x - y) ** 2
        for row_a, row_b in zip(a, b)
        for x, y in zip(row_a, row_b)
    ]
    return sum(values) / len(values)


def max_abs_diff(a: Matrix, b: Matrix) -> float:
    return max(
        abs(x - y)
        for row_a, row_b in zip(a, b)
        for x, y in zip(row_a, row_b)
    )


def activation_scales(samples: Matrix) -> list[float]:
    """论文/官方代码使用的逐输入通道 mean(abs(x))。"""
    n_channels = len(samples[0])
    return [
        sum(abs(sample[channel]) for sample in samples) / len(samples)
        for channel in range(n_channels)
    ]


def awq_scales(act_scales: Sequence[float], alpha: float) -> list[float]:
    """s = s_X ** alpha，并复现官方代码的几何中心归一化。"""
    scales = [max(value, 1e-4) ** alpha for value in act_scales]
    normalizer = math.sqrt(max(scales) * min(scales))
    return [value / normalizer for value in scales]


def transform(weight: Matrix, samples: Matrix, scales: Sequence[float]) -> tuple[Matrix, Matrix]:
    """返回 W' = W diag(s) 与 X' = X diag(s)^-1。"""
    scaled_weight = [
        [value * scales[channel] for channel, value in enumerate(row)]
        for row in weight
    ]
    scaled_samples = [
        [value / scales[channel] for channel, value in enumerate(sample)]
        for sample in samples
    ]
    return scaled_weight, scaled_samples


def quantize_group(values: Sequence[float], bits: int) -> list[float]:
    """官方仓库风格的 asymmetric min-max fake quantization。"""
    qmin, qmax = 0, 2**bits - 1
    low, high = min(values), max(values)
    scale = max((high - low) / qmax, 1e-8)
    zero = int(clamp(round(-low / scale), qmin, qmax))
    return [
        (int(clamp(round(value / scale) + zero, qmin, qmax)) - zero) * scale
        for value in values
    ]


def pseudo_quantize(weight: Matrix, bits: int, group_size: int) -> Matrix:
    result: Matrix = []
    for row in weight:
        if len(row) % group_size:
            raise ValueError("输入维度必须能被 group_size 整除")
        quantized_row: list[float] = []
        for start in range(0, len(row), group_size):
            quantized_row.extend(quantize_group(row[start : start + group_size], bits))
        result.append(quantized_row)
    return result


def restore_input_coordinates(weight: Matrix, scales: Sequence[float]) -> Matrix:
    """把量化后的 W' 除以 s，得到在原始输入坐标下的等效权重。"""
    return [
        [value / scales[channel] for channel, value in enumerate(row)]
        for row in weight
    ]


def output_error(reference_weight: Matrix, candidate_weight: Matrix, samples: Matrix) -> float:
    return mse(linear(reference_weight, samples), linear(candidate_weight, samples))


@dataclass(frozen=True)
class SearchResult:
    alpha: float
    scales: list[float]
    effective_weight: Matrix
    error: float
    history: list[tuple[float, float]]


def search_awq(
    weight: Matrix,
    samples: Matrix,
    bits: int,
    group_size: int,
    n_grid: int = 20,
) -> SearchResult:
    """复现官方搜索空间 alpha = 0, 1/n_grid, ..., (n_grid-1)/n_grid。"""
    act = activation_scales(samples)
    history: list[tuple[float, float]] = []
    best: SearchResult | None = None

    for index in range(n_grid):
        alpha = index / n_grid
        scales = awq_scales(act, alpha)
        scaled_weight, _ = transform(weight, samples, scales)
        quantized_scaled = pseudo_quantize(scaled_weight, bits, group_size)
        effective = restore_input_coordinates(quantized_scaled, scales)
        error = output_error(weight, effective, samples)
        history.append((alpha, error))
        if best is None or error < best.error:
            best = SearchResult(alpha, scales, effective, error, [])

    assert best is not None
    return SearchResult(best.alpha, best.scales, best.effective_weight, best.error, history)


def clip_group_for_output(
    values: Sequence[float],
    group_inputs: Matrix,
    bits: int,
    ratios: Iterable[float],
) -> list[float]:
    """逐 group 搜索 clipping；用该 group 的局部输出 MSE 选阈值。"""
    original = [sum(w * x for w, x in zip(values, sample)) for sample in group_inputs]
    abs_max = max(abs(value) for value in values)
    best_values = list(values)
    best_error = math.inf

    for ratio in ratios:
        bound = abs_max * ratio
        clipped = [clamp(value, -bound, bound) for value in values]
        quantized = quantize_group(clipped, bits)
        candidate = [
            sum(w * x for w, x in zip(quantized, sample))
            for sample in group_inputs
        ]
        error = sum((a - b) ** 2 for a, b in zip(original, candidate)) / len(original)
        if error < best_error:
            best_error = error
            best_values = quantized
    return best_values


def pseudo_quantize_with_clip(
    weight: Matrix,
    samples: Matrix,
    bits: int,
    group_size: int,
    n_grid: int = 20,
    max_shrink: float = 0.5,
) -> Matrix:
    """论文 clipping 思路的简化版；正式实现会向量化并分批处理输出通道。"""
    ratios = [1.0 - index / n_grid for index in range(int(max_shrink * n_grid))]
    result: Matrix = []
    for row in weight:
        quantized_row: list[float] = []
        for start in range(0, len(row), group_size):
            group_inputs = [sample[start : start + group_size] for sample in samples]
            quantized_row.extend(
                clip_group_for_output(
                    row[start : start + group_size], group_inputs, bits, ratios
                )
            )
        result.append(quantized_row)
    return result


def protect_columns(weight: Matrix, quantized: Matrix, columns: set[int]) -> Matrix:
    """仅用于复现论文观察：把指定输入通道对应权重保留为 FP。"""
    return [
        [weight[r][c] if c in columns else quantized[r][c] for c in range(len(weight[r]))]
        for r in range(len(weight))
    ]


def top_indices(values: Sequence[float], count: int) -> set[int]:
    return set(sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:count])


def column_weight_norms(weight: Matrix) -> list[float]:
    return [
        math.sqrt(sum(row[channel] ** 2 for row in weight))
        for channel in range(len(weight[0]))
    ]


def make_demo() -> tuple[Matrix, Matrix]:
    """构造“高 activation 通道并非最大权重通道”的可复现实验。"""
    weight = [
        [3.20, 0.42, -2.70, 1.80, -0.31, 2.50, -1.60, 1.10],
        [-2.90, -0.37, 2.40, -1.30, 0.28, -2.80, 1.90, -0.80],
        [2.60, 0.33, -3.10, 1.50, -0.35, 2.20, -1.70, 0.90],
        [-3.30, -0.46, 2.80, -1.10, 0.25, -2.40, 1.50, -1.20],
    ]
    amplitudes = [0.12, 6.0, 0.18, 0.10, 8.0, 0.15, 0.20, 0.11]
    rng = random.Random(2023)
    samples = [
        [amplitude * rng.uniform(-1.0, 1.0) for amplitude in amplitudes]
        for _ in range(256)
    ]
    return weight, samples


def gibibytes(byte_count: float) -> float:
    return byte_count / 2**30


def main() -> None:
    weight, samples = make_demo()
    bits, group_size = 3, 4
    reference = linear(weight, samples)

    print("[1] 等价缩放在量化前不改变线性层")
    scales = awq_scales(activation_scales(samples), alpha=0.5)
    scaled_weight, scaled_samples = transform(weight, samples, scales)
    print(f"max |WX - (W diag(s))(diag(s)^-1 X)| = {max_abs_diff(reference, linear(scaled_weight, scaled_samples)):.3e}")

    print("\n[2] activation saliency 比 weight magnitude 更贴近输出误差")
    rtn = pseudo_quantize(weight, bits, group_size)
    act_top = top_indices(activation_scales(samples), 2)
    weight_top = top_indices(column_weight_norms(weight), 2)
    random_top = {0, 3}
    print(f"toy protected ratio = 2/8（用于放大现象；论文观察为 0.1%-1%）")
    print(f"RTN output MSE             = {mse(reference, linear(rtn, samples)):.6f}")
    print(f"protect activation top {sorted(act_top)} = {mse(reference, linear(protect_columns(weight, rtn, act_top), samples)):.6f}")
    print(f"protect weight top     {sorted(weight_top)} = {mse(reference, linear(protect_columns(weight, rtn, weight_top), samples)):.6f}")
    print(f"protect fixed random   {sorted(random_top)} = {mse(reference, linear(protect_columns(weight, rtn, random_top), samples)):.6f}")

    print("\n[3] 搜索 alpha，不保留任何 FP 权重")
    result = search_awq(weight, samples, bits, group_size)
    print(f"best alpha = {result.alpha:.2f}")
    print(f"RTN output MSE = {output_error(weight, rtn, samples):.6f}")
    print(f"AWQ output MSE = {result.error:.6f}")
    print("alpha search:")
    print("  " + "  ".join(f"{alpha:.2f}:{error:.4f}" for alpha, error in result.history))

    print("\n[4] 在最佳缩放后做简化的 clipping 搜索")
    best_scaled_weight, best_scaled_samples = transform(weight, samples, result.scales)
    clipped_scaled = pseudo_quantize_with_clip(
        best_scaled_weight, best_scaled_samples, bits, group_size
    )
    clipped_effective = restore_input_coordinates(clipped_scaled, result.scales)
    print(f"AWQ + clipping output MSE = {output_error(weight, clipped_effective, samples):.6f}")

    print("\n[5] 7B 参数 W4-g128 的理想权重载荷")
    params = 7_000_000_000
    raw = params * 4 / 8
    metadata = params / 128 * 4  # 每组一个 FP16 scale + 一个 FP16 zero
    fp16 = params * 2
    print(f"FP16 payload                 = {gibibytes(fp16):.2f} GiB")
    print(f"INT4 codes                   = {gibibytes(raw):.2f} GiB")
    print(f"scale/zero metadata (估算)  = {gibibytes(metadata):.2f} GiB")
    print(f"W4-g128 total (不含其他张量) = {gibibytes(raw + metadata):.2f} GiB")


if __name__ == "__main__":
    main()
