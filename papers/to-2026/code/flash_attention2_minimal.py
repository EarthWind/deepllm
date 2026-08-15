"""Dependency-free model of the FlashAttention-2 scheduling ideas.

This is intentionally scalar Python, not a fast GPU kernel.  It separates
three ideas that are easy to conflate when reading the paper:

1. FlashAttention v1 style: keep normalized output and update it after every
   K/V tile;
2. FlashAttention-2 style: one worker owns a Q row tile, keeps an unnormalized
   output accumulator, and divides only after scanning all K/V tiles; and
3. the grid-level parallelism created by assigning independent Q row tiles
   (forward) or K/V column tiles (backward) to thread blocks.

Both forward functions compute exact dense softmax attention in real
arithmetic.  The operation counters are pedagogical: they count row-level
normalization events, not GPU instructions or wall-clock time.

Run:
    python3 papers/to-2026/code/flash_attention2_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


Matrix = list[list[float]]


@dataclass
class WorkStats:
    """Small ledger for comparing the two loop organizations."""

    score_tiles: int = 0
    row_normalizations: int = 0
    q_tile_loads: int = 0
    kv_tile_loads: int = 0


@dataclass(frozen=True)
class Worker:
    """A logical CUDA thread-block assignment, not an actual CUDA object."""

    batch: int
    head: int
    tile: int
    axis: str


def _shape(name: str, matrix: Matrix) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError(f"{name} must be a non-empty matrix")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    return len(matrix), width


def _validate(q: Matrix, k: Matrix, v: Matrix) -> tuple[int, int, int]:
    n_q, d_q = _shape("q", q)
    n_k, d_k = _shape("k", k)
    n_v, d_v = _shape("v", v)
    if not (n_q == n_k == n_v):
        raise ValueError("this compact demo expects self-attention")
    if d_q != d_k:
        raise ValueError("q and k must have the same head dimension")
    return n_q, d_q, d_v


def _dot(x: Iterable[float], y: Iterable[float]) -> float:
    return sum(a * b for a, b in zip(x, y))


def _zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _valid_key_end(*, query: int, key_end: int, causal: bool) -> int:
    return min(key_end, query + 1) if causal else key_end


def naive_attention(
    q: Matrix,
    k: Matrix,
    v: Matrix,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> Matrix:
    """Dense materializing reference used as the correctness oracle."""

    n, d, d_v = _validate(q, k, v)
    tau = 1.0 / math.sqrt(d) if scale is None else scale
    output = _zeros(n, d_v)

    for i in range(n):
        stop = i + 1 if causal else n
        scores = [tau * _dot(q[i], k[j]) for j in range(stop)]
        row_max = max(scores)
        weights = [math.exp(score - row_max) for score in scores]
        denominator = sum(weights)
        for j, weight in enumerate(weights):
            probability = weight / denominator
            for value_dim in range(d_v):
                output[i][value_dim] += probability * v[j][value_dim]
    return output


def flash_v1_style_forward(
    q: Matrix,
    k: Matrix,
    v: Matrix,
    *,
    block_q: int = 2,
    block_k: int = 3,
    causal: bool = False,
    scale: float | None = None,
) -> tuple[Matrix, WorkStats]:
    """Model the original paper's K/V-outer loop and normalized O updates.

    A production v1 kernel performs vectorized tile operations.  This scalar
    version exposes the important dependency: every K/V tile revisits Q, O,
    m, and l, and O is normalized again after each merge.
    """

    n, d, d_v = _validate(q, k, v)
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block sizes must be positive")
    tau = 1.0 / math.sqrt(d) if scale is None else scale
    output = _zeros(n, d_v)
    running_max = [-math.inf for _ in range(n)]
    running_sum = [0.0 for _ in range(n)]
    stats = WorkStats()

    # FlashAttention v1 Algorithm 1: K/V column tiles outside.
    for key_start in range(0, n, block_k):
        key_end = min(key_start + block_k, n)
        stats.kv_tile_loads += 1

        for query_start in range(0, n, block_q):
            query_end = min(query_start + block_q, n)
            if causal and key_start >= query_end:
                continue
            stats.q_tile_loads += 1
            stats.score_tiles += 1

            for i in range(query_start, query_end):
                valid_end = _valid_key_end(
                    query=i, key_end=key_end, causal=causal
                )
                if key_start >= valid_end:
                    continue

                scores = [
                    tau * _dot(q[i], k[j])
                    for j in range(key_start, valid_end)
                ]
                tile_max = max(scores)
                new_max = max(running_max[i], tile_max)
                old_scale = (
                    0.0
                    if running_max[i] == -math.inf
                    else math.exp(running_max[i] - new_max)
                )
                weights = [math.exp(score - new_max) for score in scores]
                new_sum = old_scale * running_sum[i] + sum(weights)

                # O is already normalized, so its old numerator is l_old * O.
                old_coefficient = old_scale * running_sum[i] / new_sum
                tile_coefficient = 1.0 / new_sum
                for value_dim in range(d_v):
                    tile_value = sum(
                        weight * v[j][value_dim]
                        for j, weight in zip(
                            range(key_start, valid_end), weights
                        )
                    )
                    output[i][value_dim] = (
                        old_coefficient * output[i][value_dim]
                        + tile_coefficient * tile_value
                    )

                running_max[i] = new_max
                running_sum[i] = new_sum
                stats.row_normalizations += 1

    return output, stats


def flash_attention2_forward(
    q: Matrix,
    k: Matrix,
    v: Matrix,
    *,
    block_q: int = 2,
    block_k: int = 3,
    causal: bool = False,
    scale: float | None = None,
) -> tuple[Matrix, list[float], WorkStats]:
    """Model FA-2's Q-outer loop and unnormalized output accumulator.

    Each logical worker owns one Q row tile.  It loads that Q tile once,
    scans all legal K/V tiles, keeps ``numerator``, ``running_max`` and
    ``running_sum`` locally, and performs a single final normalization.
    """

    n, d, d_v = _validate(q, k, v)
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block sizes must be positive")
    tau = 1.0 / math.sqrt(d) if scale is None else scale
    output = _zeros(n, d_v)
    logsumexp = [0.0 for _ in range(n)]
    stats = WorkStats()

    # FlashAttention-2 Algorithm 1: independent Q row tiles outside.
    for query_start in range(0, n, block_q):
        query_end = min(query_start + block_q, n)
        stats.q_tile_loads += 1

        running_max = [-math.inf for _ in range(query_start, query_end)]
        running_sum = [0.0 for _ in range(query_start, query_end)]
        numerator = _zeros(query_end - query_start, d_v)

        for key_start in range(0, n, block_k):
            if causal and key_start >= query_end:
                break
            key_end = min(key_start + block_k, n)
            stats.kv_tile_loads += 1
            stats.score_tiles += 1

            for local_i, i in enumerate(range(query_start, query_end)):
                valid_end = _valid_key_end(
                    query=i, key_end=key_end, causal=causal
                )
                if key_start >= valid_end:
                    continue

                scores = [
                    tau * _dot(q[i], k[j])
                    for j in range(key_start, valid_end)
                ]
                tile_max = max(scores)
                new_max = max(running_max[local_i], tile_max)
                old_scale = (
                    0.0
                    if running_max[local_i] == -math.inf
                    else math.exp(running_max[local_i] - new_max)
                )
                weights = [math.exp(score - new_max) for score in scores]

                running_sum[local_i] = (
                    old_scale * running_sum[local_i] + sum(weights)
                )
                for value_dim in range(d_v):
                    tile_value = sum(
                        weight * v[j][value_dim]
                        for j, weight in zip(
                            range(key_start, valid_end), weights
                        )
                    )
                    numerator[local_i][value_dim] = (
                        old_scale * numerator[local_i][value_dim] + tile_value
                    )
                running_max[local_i] = new_max

        for local_i, i in enumerate(range(query_start, query_end)):
            output[i] = [
                value / running_sum[local_i] for value in numerator[local_i]
            ]
            logsumexp[i] = running_max[local_i] + math.log(
                running_sum[local_i]
            )
            stats.row_normalizations += 1

    return output, logsumexp, stats


def forward_worker_grid(
    *, batch: int, heads: int, sequence: int, block_q: int
) -> list[Worker]:
    """Logical FA-2 forward grid: (batch, head, Q-row tile)."""

    if min(batch, heads, sequence, block_q) <= 0:
        raise ValueError("grid dimensions must be positive")
    tiles = math.ceil(sequence / block_q)
    return [
        Worker(b, h, tile, "Q rows")
        for b in range(batch)
        for h in range(heads)
        for tile in range(tiles)
    ]


def backward_worker_grid(
    *, batch: int, heads: int, sequence: int, block_k: int
) -> list[Worker]:
    """Logical FA-2 backward grid: (batch, head, K/V-column tile)."""

    if min(batch, heads, sequence, block_k) <= 0:
        raise ValueError("grid dimensions must be positive")
    tiles = math.ceil(sequence / block_k)
    return [
        Worker(b, h, tile, "K/V columns")
        for b in range(batch)
        for h in range(heads)
        for tile in range(tiles)
    ]


def _max_error(left: Matrix, right: Matrix) -> float:
    return max(
        abs(a - b)
        for left_row, right_row in zip(left, right)
        for a, b in zip(left_row, right_row)
    )


def _demo() -> None:
    q = [
        [1.0, 0.0, 0.5, -0.5],
        [0.0, 1.0, -0.5, 0.5],
        [1.0, 1.0, 0.0, 0.5],
        [-0.5, 0.5, 1.0, 0.0],
        [0.5, -1.0, 0.5, 1.0],
        [1.5, 0.5, -0.5, 0.0],
    ]
    k = [
        [0.5, 0.0, 1.0, -0.5],
        [0.0, 1.0, 0.5, 0.0],
        [1.0, 0.5, -0.5, 0.5],
        [-0.5, 1.0, 0.0, 1.0],
        [0.5, -0.5, 1.0, 0.5],
        [1.0, 1.0, 0.5, -1.0],
    ]
    v = [
        [1.0, 0.0, 0.5],
        [0.0, 1.0, -0.5],
        [1.0, 1.0, 0.0],
        [-1.0, 0.5, 1.0],
        [0.5, -1.0, 0.5],
        [1.5, 0.5, -1.0],
    ]

    for causal in (False, True):
        reference = naive_attention(q, k, v, causal=causal)
        v1, v1_stats = flash_v1_style_forward(
            q, k, v, block_q=2, block_k=2, causal=causal
        )
        v2, logsumexp, v2_stats = flash_attention2_forward(
            q, k, v, block_q=2, block_k=2, causal=causal
        )
        error_v1 = _max_error(reference, v1)
        error_v2 = _max_error(reference, v2)
        assert error_v1 < 1e-12
        assert error_v2 < 1e-12
        assert all(math.isfinite(value) for value in logsumexp)

        mode = "causal" if causal else "non-causal"
        print(f"{mode}: max error v1={error_v1:.3e}, v2={error_v2:.3e}")
        print(
            "  row normalizations: "
            f"v1-style={v1_stats.row_normalizations}, "
            f"FA-2-style={v2_stats.row_normalizations}"
        )
        print(
            "  logical tile loads: "
            f"v1 Q={v1_stats.q_tile_loads}, KV={v1_stats.kv_tile_loads}; "
            f"v2 Q={v2_stats.q_tile_loads}, KV={v2_stats.kv_tile_loads}"
        )

    # Long sequences often force a small batch.  FA-2 exposes row/column tiles
    # as extra independent work, instead of offering only batch * heads CTAs.
    batch, heads, sequence, tile = 1, 8, 4096, 128
    v1_workers = batch * heads
    v2_forward = forward_worker_grid(
        batch=batch, heads=heads, sequence=sequence, block_q=tile
    )
    v2_backward = backward_worker_grid(
        batch=batch, heads=heads, sequence=sequence, block_k=tile
    )
    print(f"v1-style logical workers: {v1_workers}")
    print(f"FA-2 forward logical workers: {len(v2_forward)}")
    print(f"FA-2 backward logical workers: {len(v2_backward)}")


if __name__ == "__main__":
    _demo()
