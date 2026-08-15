"""Dependency-free reference for the core Mistral 7B inference ideas.

This file is intentionally scalar Python, not a fast attention kernel.  It
keeps the mechanisms that are easy to conflate separate:

1. grouped-query attention maps 32 query heads to 8 KV heads;
2. sliding-window attention limits the number of visible token positions;
3. a rolling KV cache reuses ``position % window`` slots; and
4. chunked prefill computes the same local causal attention as token-by-token
   decoding while materializing only a bounded amount of old context.

Run:
    python3 papers/to-2026/code/mistral7b_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


Vector = list[float]
Heads = list[Vector]
Sequence = list[Heads]


@dataclass(frozen=True)
class Mistral7BConfig:
    dim: int = 4096
    n_layers: int = 32
    head_dim: int = 128
    hidden_dim: int = 14336
    n_heads: int = 32
    n_kv_heads: int = 8
    window_size: int = 4096
    context_len: int = 8192
    vocab_size: int = 32000
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.dim != self.n_heads * self.head_dim:
            raise ValueError("dim must equal n_heads * head_dim")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    @property
    def queries_per_kv_head(self) -> int:
        return self.n_heads // self.n_kv_heads


def parameter_ledger(config: Mistral7BConfig) -> dict[str, int]:
    """Count the checkpoint parameters implied by the public architecture.

    The count includes untied input and output embeddings, two RMSNorm weight
    vectors per block, and the final RMSNorm.  Linear biases are absent.
    """

    d = config.dim
    kv_dim = config.n_kv_heads * config.head_dim
    embedding = config.vocab_size * d
    attention_per_layer = d * d + 2 * d * kv_dim + d * d
    swiglu_per_layer = 3 * d * config.hidden_dim
    norms_per_layer = 2 * d
    blocks = config.n_layers * (
        attention_per_layer + swiglu_per_layer + norms_per_layer
    )
    lm_head = 0 if config.tie_word_embeddings else embedding
    total = embedding + blocks + d + lm_head
    return {
        "token_embedding": embedding,
        "attention_per_layer": attention_per_layer,
        "swiglu_per_layer": swiglu_per_layer,
        "norms_per_layer": norms_per_layer,
        "all_transformer_blocks": blocks,
        "final_norm": d,
        "lm_head": lm_head,
        "total": total,
    }


def kv_cache_bytes(
    config: Mistral7BConfig,
    cached_tokens: int,
    *,
    bytes_per_element: int = 2,
    kv_heads: int | None = None,
) -> int:
    """Ideal K/V payload for one sequence; allocator overhead is excluded."""

    heads = config.n_kv_heads if kv_heads is None else kv_heads
    return (
        2
        * config.n_layers
        * heads
        * cached_tokens
        * config.head_dim
        * bytes_per_element
    )


def query_to_kv_head(
    query_head: int, *, n_query_heads: int, n_kv_heads: int
) -> int:
    if n_query_heads % n_kv_heads:
        raise ValueError("n_query_heads must be divisible by n_kv_heads")
    if not 0 <= query_head < n_query_heads:
        raise IndexError("query_head out of range")
    return query_head // (n_query_heads // n_kv_heads)


def _dot(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


def _weighted_sum(weights: list[float], values: list[Vector]) -> Vector:
    width = len(values[0])
    return [
        sum(weight * value[column] for weight, value in zip(weights, values))
        for column in range(width)
    ]


def _softmax(scores: list[float]) -> list[float]:
    row_max = max(scores)
    exponentials = [math.exp(score - row_max) for score in scores]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _validate_tensors(q: Sequence, k: Sequence, v: Sequence) -> tuple[int, int, int]:
    if not q or not k or not v or not (len(q) == len(k) == len(v)):
        raise ValueError("q, k, and v must have the same non-zero sequence length")
    n_query_heads = len(q[0])
    n_kv_heads = len(k[0])
    if n_query_heads % n_kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    head_dim = len(q[0][0])
    for token in q:
        if len(token) != n_query_heads or any(len(head) != head_dim for head in token):
            raise ValueError("q must be rectangular")
    for tensor in (k, v):
        for token in tensor:
            if len(token) != n_kv_heads or any(len(head) != head_dim for head in token):
                raise ValueError("k and v must match the KV-head and head dimensions")
    return n_query_heads, n_kv_heads, head_dim


def materialized_local_gqa(
    q: Sequence, k: Sequence, v: Sequence, *, window: int
) -> Sequence:
    """Correctness oracle for causal local GQA.

    Here ``window`` is a cardinality: query position i sees at most W keys,
    namely [max(0, i-W+1), i].  Some papers express the left boundary as i-W;
    stating the cardinality avoids that common off-by-one ambiguity.
    """

    n_query_heads, n_kv_heads, head_dim = _validate_tensors(q, k, v)
    if window <= 0:
        raise ValueError("window must be positive")
    scale = 1.0 / math.sqrt(head_dim)
    output: Sequence = []

    for position, query_heads in enumerate(q):
        start = max(0, position - window + 1)
        token_output: Heads = []
        for query_head, query in enumerate(query_heads):
            kv_head = query_to_kv_head(
                query_head,
                n_query_heads=n_query_heads,
                n_kv_heads=n_kv_heads,
            )
            positions = list(range(start, position + 1))
            scores = [scale * _dot(query, k[index][kv_head]) for index in positions]
            weights = _softmax(scores)
            values = [v[index][kv_head] for index in positions]
            token_output.append(_weighted_sum(weights, values))
        output.append(token_output)
    return output


@dataclass(frozen=True)
class KVEntry:
    position: int
    key: Heads
    value: Heads


class RollingKVCache:
    """A fixed-size cache whose physical slot is absolute_position % window."""

    def __init__(self, window: int) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self._slots: list[KVEntry | None] = [None] * window

    def append(self, position: int, key: Heads, value: Heads) -> None:
        self._slots[position % self.window] = KVEntry(position, key, value)

    def visible(self, query_position: int) -> list[KVEntry]:
        start = max(0, query_position - self.window + 1)
        return sorted(
            (
                entry
                for entry in self._slots
                if entry is not None and start <= entry.position <= query_position
            ),
            key=lambda entry: entry.position,
        )

    def physical_layout(self) -> list[int | None]:
        return [None if entry is None else entry.position for entry in self._slots]


def _attend_entries(
    query_heads: Heads,
    entries: list[KVEntry],
    *,
    n_kv_heads: int,
) -> Heads:
    n_query_heads = len(query_heads)
    head_dim = len(query_heads[0])
    scale = 1.0 / math.sqrt(head_dim)
    output: Heads = []
    for query_head, query in enumerate(query_heads):
        kv_head = query_to_kv_head(
            query_head,
            n_query_heads=n_query_heads,
            n_kv_heads=n_kv_heads,
        )
        scores = [scale * _dot(query, entry.key[kv_head]) for entry in entries]
        weights = _softmax(scores)
        values = [entry.value[kv_head] for entry in entries]
        output.append(_weighted_sum(weights, values))
    return output


def rolling_decode(q: Sequence, k: Sequence, v: Sequence, *, window: int) -> Sequence:
    """Decode one token at a time with a bounded rolling cache."""

    _, n_kv_heads, _ = _validate_tensors(q, k, v)
    cache = RollingKVCache(window)
    output: Sequence = []
    for position in range(len(q)):
        cache.append(position, k[position], v[position])
        output.append(
            _attend_entries(
                q[position], cache.visible(position), n_kv_heads=n_kv_heads
            )
        )
    return output


def chunked_prefill(
    q: Sequence,
    k: Sequence,
    v: Sequence,
    *,
    window: int,
    chunk_size: int,
) -> Sequence:
    """Prefill chunks while applying causal and local masks inside each chunk.

    A whole chunk may be computed in parallel: every query reads the old cache
    plus keys from the current chunk, then filters them by absolute position.
    Only after the chunk is complete are its K/V entries committed to the
    rolling cache.
    """

    _, n_kv_heads, _ = _validate_tensors(q, k, v)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    cache = RollingKVCache(window)
    output: Sequence = []

    for chunk_start in range(0, len(q), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(q))
        old_entries = [entry for entry in cache._slots if entry is not None]
        current_entries = [
            KVEntry(position, k[position], v[position])
            for position in range(chunk_start, chunk_end)
        ]
        candidates = old_entries + current_entries

        for position in range(chunk_start, chunk_end):
            left = max(0, position - window + 1)
            visible = sorted(
                (
                    entry
                    for entry in candidates
                    if left <= entry.position <= position
                ),
                key=lambda entry: entry.position,
            )
            output.append(
                _attend_entries(q[position], visible, n_kv_heads=n_kv_heads)
            )

        for entry in current_entries:
            cache.append(entry.position, entry.key, entry.value)
    return output


def _random_sequence(
    rng: random.Random, tokens: int, heads: int, head_dim: int
) -> Sequence:
    return [
        [
            [rng.uniform(-1.0, 1.0) for _ in range(head_dim)]
            for _ in range(heads)
        ]
        for _ in range(tokens)
    ]


def _assert_close(left: Sequence, right: Sequence, *, tolerance: float = 1e-12) -> None:
    if len(left) != len(right):
        raise AssertionError("sequence lengths differ")
    for left_token, right_token in zip(left, right):
        for left_head, right_head in zip(left_token, right_token):
            for left_value, right_value in zip(left_head, right_head):
                if abs(left_value - right_value) > tolerance:
                    raise AssertionError(f"{left_value} != {right_value}")


def _mib(byte_count: int) -> float:
    return byte_count / (1024**2)


def main() -> None:
    config = Mistral7BConfig()
    ledger = parameter_ledger(config)
    assert ledger["total"] == 7_241_732_096

    print("Mistral 7B architecture ledger")
    print(f"  query heads / KV heads: {config.n_heads} / {config.n_kv_heads}")
    print(f"  query heads per KV head: {config.queries_per_kv_head}")
    print(f"  parameters from public dimensions: {ledger['total']:,} (~7.24B)")

    rolling_gqa = kv_cache_bytes(config, config.window_size)
    full_gqa_8k = kv_cache_bytes(config, config.context_len)
    full_gqa_32k = kv_cache_bytes(config, 32768)
    full_mha_32k = kv_cache_bytes(config, 32768, kv_heads=config.n_heads)
    print("\nIdeal bf16 KV payload for one sequence")
    print(f"  rolling GQA, W=4096: {_mib(rolling_gqa):.0f} MiB")
    print(f"  full GQA, N=8192:     {_mib(full_gqa_8k):.0f} MiB")
    print(f"  full GQA, N=32768:    {_mib(full_gqa_32k):.0f} MiB")
    print(f"  full MHA, N=32768:    {_mib(full_mha_32k):.0f} MiB")
    print(f"  SWA factor at 32K:    {full_gqa_32k / rolling_gqa:.0f}x")
    print(f"  GQA+SWA vs full MHA:  {full_mha_32k / rolling_gqa:.0f}x")

    # A small shape keeps the scalar correctness test instant.  It uses the
    # same 4:1 query/KV grouping ratio as Mistral while keeping two KV groups
    # so that the mapping remains visible.
    rng = random.Random(7)
    tokens, q_heads, kv_heads, head_dim, window = 11, 8, 2, 3, 4
    q = _random_sequence(rng, tokens, q_heads, head_dim)
    k = _random_sequence(rng, tokens, kv_heads, head_dim)
    v = _random_sequence(rng, tokens, kv_heads, head_dim)

    reference = materialized_local_gqa(q, k, v, window=window)
    decoded = rolling_decode(q, k, v, window=window)
    prefilled = chunked_prefill(q, k, v, window=window, chunk_size=window)
    _assert_close(reference, decoded)
    _assert_close(reference, prefilled)

    cache = RollingKVCache(window)
    for position in range(7):
        cache.append(position, k[position], v[position])
    assert cache.physical_layout() == [4, 5, 6, 3]
    assert [entry.position for entry in cache.visible(6)] == [3, 4, 5, 6]

    print("\nCorrectness checks")
    print("  materialized local GQA == rolling decode: yes")
    print("  materialized local GQA == chunked prefill: yes")
    print(f"  slots after positions 0..6 with W=4: {cache.physical_layout()}")
    print("  visible absolute positions for query 6: [3, 4, 5, 6]")


if __name__ == "__main__":
    main()
