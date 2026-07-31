"""Dependency-free reference implementation of FlashAttention's core math.

This file is deliberately written with Python lists and scalar loops.  It is
meant to make tiling, online softmax, and backward recomputation auditable; it
is NOT a fast GPU kernel and should only be used with tiny matrices.

Implemented ideas from the 2022 FlashAttention paper:

* stable dense attention as a correctness oracle;
* tiled forward attention without materializing an N x N score/probability
  matrix;
* causal masking inside each score tile;
* saving one log-sum-exp number per query row; and
* tiled backward that recomputes probabilities from Q, K, and the saved
  normalization statistic instead of saving the full probability matrix.

Run:
    python3 papers/to-2026/code/flash_attention_minimal.py
"""

from __future__ import annotations

import math
from typing import Iterable


Matrix = list[list[float]]


def _check_rectangular(name: str, matrix: Matrix) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError(f"{name} must be a non-empty matrix")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    return len(matrix), width


def _validate_qkv(q: Matrix, k: Matrix, v: Matrix) -> tuple[int, int, int]:
    n_q, d_q = _check_rectangular("q", q)
    n_k, d_k = _check_rectangular("k", k)
    n_v, d_v = _check_rectangular("v", v)
    if n_q != n_k or n_k != n_v:
        raise ValueError("this compact example expects self-attention (same length)")
    if d_q != d_k:
        raise ValueError("q and k must have the same head dimension")
    return n_q, d_q, d_v


def _dot(x: Iterable[float], y: Iterable[float]) -> float:
    return sum(a * b for a, b in zip(x, y))


def _zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def naive_attention(
    q: Matrix,
    k: Matrix,
    v: Matrix,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> tuple[Matrix, Matrix, list[float]]:
    """Materialize the full probability matrix as a correctness oracle.

    Returns ``(output, probabilities, logsumexp)``.  A production framework's
    ordinary attention path may fuse some operations, but this explicit form
    exposes the quadratic intermediate that FlashAttention avoids.
    """

    n, d, d_v_width = _validate_qkv(q, k, v)
    tau = 1.0 / math.sqrt(d) if scale is None else scale
    output = _zeros(n, d_v_width)
    probabilities = _zeros(n, n)
    logsumexp = [0.0 for _ in range(n)]

    for i in range(n):
        last_key = i + 1 if causal else n
        scores = [tau * _dot(q[i], k[j]) for j in range(last_key)]
        row_max = max(scores)
        unnormalized = [math.exp(score - row_max) for score in scores]
        denominator = sum(unnormalized)
        logsumexp[i] = row_max + math.log(denominator)

        for j, weight in enumerate(unnormalized):
            probability = weight / denominator
            probabilities[i][j] = probability
            for value_dim in range(d_v_width):
                output[i][value_dim] += probability * v[j][value_dim]

    return output, probabilities, logsumexp


def flash_attention_forward(
    q: Matrix,
    k: Matrix,
    v: Matrix,
    *,
    block_q: int = 2,
    block_k: int = 3,
    causal: bool = False,
    scale: float | None = None,
) -> tuple[Matrix, list[float]]:
    """Tiled exact attention using online softmax.

    The outer loop follows Algorithm 1 of the paper: keep a K/V block "on
    chip", then visit every Q block.  ``running_max``, ``running_sum``, and
    ``numerator`` are the per-row state needed to merge score blocks.

    In real CUDA these objects move between HBM and SRAM and the inner work is
    parallel.  Here they remain Python lists so every update is easy to read.
    """

    n, d, d_v = _validate_qkv(q, k, v)
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block sizes must be positive")
    tau = 1.0 / math.sqrt(d) if scale is None else scale

    running_max = [-math.inf for _ in range(n)]
    running_sum = [0.0 for _ in range(n)]
    numerator = _zeros(n, d_v)

    # Paper order: K/V tiles outside, Q tiles inside.  It reuses each K/V tile
    # while repeatedly streaming Q and the row statistics through fast SRAM.
    for key_start in range(0, n, block_k):
        key_end = min(key_start + block_k, n)

        for query_start in range(0, n, block_q):
            query_end = min(query_start + block_q, n)
            if causal and key_start >= query_end:
                # This whole tile lies strictly above the causal diagonal.
                continue

            for i in range(query_start, query_end):
                valid_key_end = min(key_end, i + 1) if causal else key_end
                if key_start >= valid_key_end:
                    continue

                tile_scores = [
                    tau * _dot(q[i], k[j])
                    for j in range(key_start, valid_key_end)
                ]
                tile_max = max(tile_scores)
                new_max = max(running_max[i], tile_max)

                # If a later block raises the maximum, every earlier
                # contribution must be expressed in the new exponential scale.
                old_scale = (
                    0.0
                    if running_max[i] == -math.inf
                    else math.exp(running_max[i] - new_max)
                )
                tile_weights = [math.exp(score - new_max) for score in tile_scores]
                new_sum = old_scale * running_sum[i] + sum(tile_weights)

                for value_dim in range(d_v):
                    tile_value = 0.0
                    for offset, weight in enumerate(tile_weights):
                        j = key_start + offset
                        tile_value += weight * v[j][value_dim]
                    numerator[i][value_dim] = (
                        old_scale * numerator[i][value_dim] + tile_value
                    )

                running_max[i] = new_max
                running_sum[i] = new_sum

    output = [
        [value / running_sum[i] for value in numerator[i]] for i in range(n)
    ]
    logsumexp = [
        running_max[i] + math.log(running_sum[i]) for i in range(n)
    ]
    return output, logsumexp


def naive_attention_backward(
    q: Matrix,
    k: Matrix,
    v: Matrix,
    d_output: Matrix,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> tuple[Matrix, Matrix, Matrix]:
    """Dense analytic backward used as the gradient oracle."""

    n, d, d_v_width = _validate_qkv(q, k, v)
    d_rows, d_width = _check_rectangular("d_output", d_output)
    if (d_rows, d_width) != (n, d_v_width):
        raise ValueError("d_output must have the same shape as attention output")
    tau = 1.0 / math.sqrt(d) if scale is None else scale
    _, probabilities, _ = naive_attention(q, k, v, causal=causal, scale=tau)

    d_q = _zeros(n, d)
    d_k = _zeros(n, d)
    d_v = _zeros(n, d_v_width)

    for i in range(n):
        d_probability = [_dot(d_output[i], v[j]) for j in range(n)]
        softmax_dot = sum(
            probabilities[i][j] * d_probability[j] for j in range(n)
        )
        for j in range(n):
            probability = probabilities[i][j]
            d_score = probability * (d_probability[j] - softmax_dot)
            for value_dim in range(d_v_width):
                d_v[j][value_dim] += probability * d_output[i][value_dim]
            for head_dim in range(d):
                d_q[i][head_dim] += tau * d_score * k[j][head_dim]
                d_k[j][head_dim] += tau * d_score * q[i][head_dim]

    return d_q, d_k, d_v


def flash_attention_backward(
    q: Matrix,
    k: Matrix,
    v: Matrix,
    output: Matrix,
    d_output: Matrix,
    logsumexp: list[float],
    *,
    block_q: int = 2,
    block_k: int = 3,
    causal: bool = False,
    scale: float | None = None,
) -> tuple[Matrix, Matrix, Matrix]:
    """Tiled backward that recomputes one probability tile at a time.

    ``P_ij = exp(S_ij - logsumexp_i)`` is reconstructed only for the current
    tile.  The identity ``sum_j P_ij dP_ij = dO_i dot O_i`` removes a second
    reduction over the complete attention row.
    """

    n, d, d_v_width = _validate_qkv(q, k, v)
    if _check_rectangular("output", output) != (n, d_v_width):
        raise ValueError("output shape does not match v")
    if _check_rectangular("d_output", d_output) != (n, d_v_width):
        raise ValueError("d_output shape does not match output")
    if len(logsumexp) != n:
        raise ValueError("logsumexp must contain one number per query row")
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block sizes must be positive")
    tau = 1.0 / math.sqrt(d) if scale is None else scale

    d_q = _zeros(n, d)
    d_k = _zeros(n, d)
    d_v = _zeros(n, d_v_width)
    softmax_dot = [_dot(d_output[i], output[i]) for i in range(n)]

    for key_start in range(0, n, block_k):
        key_end = min(key_start + block_k, n)

        for query_start in range(0, n, block_q):
            query_end = min(query_start + block_q, n)
            if causal and key_start >= query_end:
                continue

            for i in range(query_start, query_end):
                valid_key_end = min(key_end, i + 1) if causal else key_end
                for j in range(key_start, valid_key_end):
                    score = tau * _dot(q[i], k[j])
                    probability = math.exp(score - logsumexp[i])
                    d_probability = _dot(d_output[i], v[j])
                    d_score = probability * (d_probability - softmax_dot[i])

                    for value_dim in range(d_v_width):
                        d_v[j][value_dim] += probability * d_output[i][value_dim]
                    for head_dim in range(d):
                        d_q[i][head_dim] += tau * d_score * k[j][head_dim]
                        d_k[j][head_dim] += tau * d_score * q[i][head_dim]

    return d_q, d_k, d_v


def _max_abs_difference(left: Matrix, right: Matrix) -> float:
    if _check_rectangular("left", left) != _check_rectangular("right", right):
        raise ValueError("matrix shapes differ")
    return max(
        abs(a - b)
        for left_row, right_row in zip(left, right)
        for a, b in zip(left_row, right_row)
    )


def _loss(
    q: Matrix,
    k: Matrix,
    v: Matrix,
    d_output: Matrix,
    *,
    causal: bool,
) -> float:
    output, _, _ = naive_attention(q, k, v, causal=causal)
    return sum(_dot(row, grad_row) for row, grad_row in zip(output, d_output))


def _finite_difference(
    q: Matrix,
    k: Matrix,
    v: Matrix,
    d_output: Matrix,
    target: str,
    row: int,
    col: int,
    *,
    causal: bool,
    epsilon: float = 1e-5,
) -> float:
    matrices = {"q": q, "k": k, "v": v}
    matrix = matrices[target]
    original = matrix[row][col]
    matrix[row][col] = original + epsilon
    loss_plus = _loss(q, k, v, d_output, causal=causal)
    matrix[row][col] = original - epsilon
    loss_minus = _loss(q, k, v, d_output, causal=causal)
    matrix[row][col] = original
    return (loss_plus - loss_minus) / (2.0 * epsilon)


def _example_matrix(rows: int, cols: int, phase: float) -> Matrix:
    return [
        [math.sin((i + 1) * (j + 2) + phase) / 2.0 for j in range(cols)]
        for i in range(rows)
    ]


def self_test() -> None:
    """Compare tiled forward/backward with dense math and finite differences."""

    n, d, d_v = 6, 4, 3
    q = _example_matrix(n, d, 0.1)
    k = _example_matrix(n, d, 0.7)
    v = _example_matrix(n, d_v, 1.3)
    d_output = _example_matrix(n, d_v, 2.1)

    worst_forward = 0.0
    worst_backward = 0.0
    latest_grads: tuple[Matrix, Matrix, Matrix] | None = None

    for causal in (False, True):
        dense_output, _, _ = naive_attention(q, k, v, causal=causal)
        dense_grads = naive_attention_backward(q, k, v, d_output, causal=causal)

        # Exactness must not depend on how we split the score matrix.
        for block_q, block_k in ((1, 1), (2, 3), (4, 2), (8, 8)):
            tiled_output, logsumexp = flash_attention_forward(
                q,
                k,
                v,
                block_q=block_q,
                block_k=block_k,
                causal=causal,
            )
            tiled_grads = flash_attention_backward(
                q,
                k,
                v,
                tiled_output,
                d_output,
                logsumexp,
                block_q=block_q,
                block_k=block_k,
                causal=causal,
            )
            worst_forward = max(
                worst_forward, _max_abs_difference(dense_output, tiled_output)
            )
            worst_backward = max(
                worst_backward,
                *(
                    _max_abs_difference(dense, tiled)
                    for dense, tiled in zip(dense_grads, tiled_grads)
                ),
            )
            latest_grads = tiled_grads

    assert worst_forward < 1e-12, worst_forward
    assert worst_backward < 1e-12, worst_backward
    assert latest_grads is not None

    # Check one element from each input against numerical differentiation.
    checks = (("q", 2, 1, 0), ("k", 4, 3, 1), ("v", 1, 2, 2))
    worst_finite_difference = 0.0
    for name, row, col, gradient_index in checks:
        numerical = _finite_difference(
            q, k, v, d_output, name, row, col, causal=True
        )
        analytic = latest_grads[gradient_index][row][col]
        worst_finite_difference = max(
            worst_finite_difference, abs(numerical - analytic)
        )
    assert worst_finite_difference < 1e-9, worst_finite_difference

    print("FlashAttention educational reference: all checks passed")
    print(f"  max forward error:          {worst_forward:.3e}")
    print(f"  max backward error:         {worst_backward:.3e}")
    print(f"  max finite-difference error:{worst_finite_difference: .3e}")


if __name__ == "__main__":
    self_test()
