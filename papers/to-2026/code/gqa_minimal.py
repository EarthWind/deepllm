#!/usr/bin/env python3
"""Grouped-Query Attention (GQA) 的零依赖教学实现。

展示：
1. GQA-H == MHA，GQA-1 == MQA；
2. native GQA 与先 repeat K/V 再做 MHA 的数学结果相同；
3. 从 MHA checkpoint 按组 mean-pool K/V；
4. KV Cache 与 K/V 投影参数量如何随 num_kv_heads 缩放。

这份代码只用于解释数学和布局，不是高性能 kernel。真实部署应让 kernel
直接消费较少的 K/V heads，不能把完整 KV Cache 物化复制到 query heads。

运行：
    python3 papers/to-2026/code/gqa_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


Vector = list[float]
SequenceVectors = list[Vector]
Heads = list[SequenceVectors]
Projection = list[list[Vector]]  # [heads][d_model][head_dim]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("dot 的两个向量长度必须相同")
    return math.fsum(a * b for a, b in zip(left, right))


def softmax(values: Sequence[float]) -> Vector:
    if not values:
        raise ValueError("softmax 输入不能为空")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = math.fsum(exponentials)
    return [value / denominator for value in exponentials]


def weighted_sum(weights: Sequence[float], vectors: Sequence[Vector]) -> Vector:
    if len(weights) != len(vectors) or not vectors:
        raise ValueError("weights 与 vectors 必须非空且长度相同")
    width = len(vectors[0])
    return [
        math.fsum(weight * vector[index] for weight, vector in zip(weights, vectors))
        for index in range(width)
    ]


def max_abs_diff(first: Heads, second: Heads) -> float:
    return max(
        abs(a - b)
        for first_head, second_head in zip(first, second)
        for first_token, second_token in zip(first_head, second_head)
        for a, b in zip(first_token, second_token)
    )


def group_for_query(query_head: int, num_query_heads: int, num_kv_heads: int) -> int:
    """把连续 query heads 均匀映射到 KV heads。"""
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_kv_heads 必须整除 num_query_heads")
    queries_per_kv = num_query_heads // num_kv_heads
    return query_head // queries_per_kv


def grouped_query_attention(queries: Heads, keys: Heads, values: Heads) -> Heads:
    """直接按 group 索引 K/V，不物化重复副本。

    Shapes:
        queries: [H, T_q, d_h]
        keys:    [G, T_kv, d_h]
        values:  [G, T_kv, d_h]
        output:  [H, T_q, d_h]
    """
    num_query_heads = len(queries)
    num_kv_heads = len(keys)
    if not queries or not keys or len(keys) != len(values):
        raise ValueError("Q/K/V head 数不合法")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("G 必须整除 H")

    head_dim = len(queries[0][0])
    scale = 1.0 / math.sqrt(head_dim)
    output: Heads = []

    for query_head, query_sequence in enumerate(queries):
        kv_head = group_for_query(query_head, num_query_heads, num_kv_heads)
        key_sequence = keys[kv_head]
        value_sequence = values[kv_head]
        if len(key_sequence) != len(value_sequence):
            raise ValueError("K/V 序列长度必须相同")

        head_output: SequenceVectors = []
        for query in query_sequence:
            logits = [dot(query, key) * scale for key in key_sequence]
            probabilities = softmax(logits)
            head_output.append(weighted_sum(probabilities, value_sequence))
        output.append(head_output)
    return output


def repeat_kv_heads(kv_heads: Heads, num_query_heads: int) -> Heads:
    """教学用物化 repeat；真实 kernel 应避免复制 Cache。"""
    if num_query_heads % len(kv_heads) != 0:
        raise ValueError("KV heads 必须整除 query heads")
    repeats = num_query_heads // len(kv_heads)
    return [
        [[*vector] for vector in sequence]
        for sequence in kv_heads
        for _ in range(repeats)
    ]


def mean_pool_projection(projection: Projection, num_kv_heads: int) -> Projection:
    """按连续组 mean-pool MHA 的 K 或 V projection heads。"""
    num_query_heads = len(projection)
    if not projection or num_query_heads % num_kv_heads != 0:
        raise ValueError("目标 KV heads 必须整除原始 heads")

    heads_per_group = num_query_heads // num_kv_heads
    d_model = len(projection[0])
    head_dim = len(projection[0][0])
    pooled: Projection = []

    for group in range(num_kv_heads):
        start = group * heads_per_group
        source_heads = projection[start : start + heads_per_group]
        pooled.append(
            [
                [
                    math.fsum(head[row][column] for head in source_heads)
                    / heads_per_group
                    for column in range(head_dim)
                ]
                for row in range(d_model)
            ]
        )
    return pooled


def first_head_projection(projection: Projection, num_kv_heads: int) -> Projection:
    """论文消融中的另一基线：每组只保留第一个 head。"""
    heads_per_group = len(projection) // num_kv_heads
    return [
        [[*row] for row in projection[group * heads_per_group]]
        for group in range(num_kv_heads)
    ]


def expand_projection(grouped: Projection, num_query_heads: int) -> Projection:
    repeats = num_query_heads // len(grouped)
    return [
        [[*row] for row in head]
        for head in grouped
        for _ in range(repeats)
    ]


def squared_projection_error(original: Projection, approximation: Projection) -> float:
    return math.fsum(
        (source - estimate) ** 2
        for source_head, estimate_head in zip(original, approximation)
        for source_row, estimate_row in zip(source_head, estimate_head)
        for source, estimate in zip(source_row, estimate_row)
    )


def random_heads(
    rng: random.Random, num_heads: int, sequence_length: int, head_dim: int
) -> Heads:
    return [
        [
            [rng.uniform(-1.0, 1.0) for _ in range(head_dim)]
            for _ in range(sequence_length)
        ]
        for _ in range(num_heads)
    ]


def random_projection(
    rng: random.Random, num_heads: int, d_model: int, head_dim: int
) -> Projection:
    return [
        [
            [rng.uniform(-0.5, 0.5) for _ in range(head_dim)]
            for _ in range(d_model)
        ]
        for _ in range(num_heads)
    ]


def gibibytes(num_bytes: int) -> float:
    return num_bytes / 1024**3


def mebibytes(num_bytes: int) -> float:
    return num_bytes / 1024**2


def kv_cache_bytes(
    *,
    batch_size: int,
    sequence_length: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    """K 和 V 两份 Cache，不含 allocator/padding/metadata。"""
    return (
        batch_size
        * sequence_length
        * num_layers
        * 2
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )


def attention_projection_parameters(
    *, d_model: int, num_query_heads: int, num_kv_heads: int, head_dim: int
) -> int:
    """无 bias 的 Wq/Wk/Wv/Wo 参数量。"""
    q = d_model * num_query_heads * head_dim
    k_and_v = 2 * d_model * num_kv_heads * head_dim
    output = num_query_heads * head_dim * d_model
    return q + k_and_v + output


@dataclass(frozen=True)
class Layout:
    name: str
    num_query_heads: int
    num_kv_heads: int

    @property
    def queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads


def show_endpoint_equivalences() -> None:
    rng = random.Random(7)
    queries = random_heads(rng, num_heads=4, sequence_length=2, head_dim=3)

    # GQA-H：每个 query head 有独立 K/V，定义上就是 MHA。
    mha_keys = random_heads(rng, num_heads=4, sequence_length=5, head_dim=3)
    mha_values = random_heads(rng, num_heads=4, sequence_length=5, head_dim=3)
    mha = grouped_query_attention(queries, mha_keys, mha_values)
    gqa_h = grouped_query_attention(queries, mha_keys, mha_values)

    # GQA-1：一个共享 K/V head，定义上就是 MQA。
    shared_keys = random_heads(rng, num_heads=1, sequence_length=5, head_dim=3)
    shared_values = random_heads(rng, num_heads=1, sequence_length=5, head_dim=3)
    mqa = grouped_query_attention(queries, shared_keys, shared_values)
    gqa_1 = grouped_query_attention(queries, shared_keys, shared_values)

    print("[1] 两个端点")
    print(f"GQA-H vs MHA max |diff| = {max_abs_diff(gqa_h, mha):.3e}")
    print(f"GQA-1 vs MQA max |diff| = {max_abs_diff(gqa_1, mqa):.3e}\n")
    assert max_abs_diff(gqa_h, mha) == 0.0
    assert max_abs_diff(gqa_1, mqa) == 0.0


def show_native_vs_repeat() -> None:
    rng = random.Random(11)
    queries = random_heads(rng, num_heads=8, sequence_length=3, head_dim=4)
    keys = random_heads(rng, num_heads=2, sequence_length=6, head_dim=4)
    values = random_heads(rng, num_heads=2, sequence_length=6, head_dim=4)

    native = grouped_query_attention(queries, keys, values)
    repeated = grouped_query_attention(
        queries,
        repeat_kv_heads(keys, num_query_heads=8),
        repeat_kv_heads(values, num_query_heads=8),
    )
    difference = max_abs_diff(native, repeated)

    print("[2] Native GQA 与物化 repeat_kv 的数学等价性")
    print(f"shapes: Q=[8,3,4], K/V=[2,6,4], queries_per_kv=4")
    print(f"max |native - repeated| = {difference:.3e}")
    print("注意：结果等价，不代表内存流量等价；生产 kernel 不应复制完整 Cache。\n")
    assert difference < 1e-12


def show_checkpoint_conversion() -> None:
    rng = random.Random(19)
    original = random_projection(rng, num_heads=8, d_model=6, head_dim=4)
    mean = mean_pool_projection(original, num_kv_heads=2)
    first = first_head_projection(original, num_kv_heads=2)

    mean_error = squared_projection_error(original, expand_projection(mean, 8))
    first_error = squared_projection_error(original, expand_projection(first, 8))

    print("[3] MHA checkpoint -> GQA-2 checkpoint")
    print("W_K/W_V: [8, d_model, d_h] --每 4 heads 求均值--> [2, d_model, d_h]")
    print(f"mean-pool squared error = {mean_error:.6f}")
    print(f"first-head squared error = {first_error:.6f}")
    print("均值是组内 Frobenius 最小二乘中心；但 attention 有 softmax，仍需 uptraining。\n")
    assert mean_error <= first_error + 1e-12


def show_memory_and_parameters() -> None:
    common = dict(
        batch_size=1,
        sequence_length=4096,
        num_layers=32,
        head_dim=128,
        bytes_per_element=2,
    )
    d_model = 4096
    layouts = [
        Layout("MHA", 32, 32),
        Layout("GQA-8", 32, 8),
        Layout("GQA-4", 32, 4),
        Layout("MQA", 32, 1),
    ]

    print("[4] KV Cache 与 attention 投影参数（示例配置）")
    print("layout  H/G  KV cache   attention params")
    for layout in layouts:
        cache = kv_cache_bytes(num_kv_heads=layout.num_kv_heads, **common)
        parameters = attention_projection_parameters(
            d_model=d_model,
            num_query_heads=layout.num_query_heads,
            num_kv_heads=layout.num_kv_heads,
            head_dim=common["head_dim"],
        )
        print(
            f"{layout.name:<7} {layout.queries_per_kv:>3}  "
            f"{mebibytes(cache):>7.1f} MiB  {parameters / 1e6:>8.2f} M"
        )

    mha_cache = kv_cache_bytes(num_kv_heads=32, **common)
    gqa_cache = kv_cache_bytes(num_kv_heads=8, **common)
    assert math.isclose(gibibytes(mha_cache), 2.0)
    assert math.isclose(mebibytes(gqa_cache), 512.0)
    assert mha_cache // gqa_cache == 4


def main() -> None:
    show_endpoint_equivalences()
    show_native_vs_repeat()
    show_checkpoint_conversion()
    show_memory_and_parameters()


if __name__ == "__main__":
    main()
