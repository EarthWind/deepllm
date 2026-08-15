"""Dependency-free reference for DeepSeek-V2's MLA and DeepSeekMoE.

This is scalar Python for understanding and testing, not a production kernel.
It demonstrates four invariants from the paper:

1. the generation cache stores a joint KV latent plus a decoupled RoPE key;
2. reconstructing every historical K/V and absorbing the up-projections are
   mathematically equivalent in real arithmetic;
3. shared experts always run while only Top-K routed experts run per token;
4. device-limited routing first bounds the candidate devices, then selects
   experts inside those devices.

Run:
    python3 papers/to-2026/code/deepseek_v2_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


Vector = list[float]
Matrix = list[Vector]


@dataclass(frozen=True)
class DeepSeekV2Config:
    hidden_size: int = 5120
    n_layers: int = 60
    n_heads: int = 128
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    dense_intermediate_size: int = 12288
    moe_intermediate_size: int = 1536
    n_shared_experts: int = 2
    n_routed_experts: int = 160
    n_experts_per_token: int = 6
    first_k_dense_replace: int = 1
    vocab_size: int = 102400
    tie_word_embeddings: bool = False

    @property
    def cached_elements_per_token_per_layer(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim


def parameter_ledger(config: DeepSeekV2Config) -> dict[str, int]:
    """Rebuild the rounded 236B / 21B claims from public dimensions.

    The active count includes both embeddings, all attention/router weights,
    two shared plus six selected expert FFNs, and the first dense FFN.  Papers
    may apply slightly different conventions, so the headline remains 21B.
    """

    d = config.hidden_size
    q_width = config.n_heads * (
        config.qk_nope_head_dim + config.qk_rope_head_dim
    )
    kv_up_width = config.n_heads * (
        config.qk_nope_head_dim + config.v_head_dim
    )
    value_width = config.n_heads * config.v_head_dim

    attention_per_layer = (
        d * config.q_lora_rank
        + config.q_lora_rank * q_width
        + d * (config.kv_lora_rank + config.qk_rope_head_dim)
        + config.kv_lora_rank * kv_up_width
        + value_width * d
        + config.q_lora_rank
        + config.kv_lora_rank
    )
    expert_mlp = 3 * d * config.moe_intermediate_size
    routed_experts_per_layer = config.n_routed_experts * expert_mlp
    shared_experts_per_layer = config.n_shared_experts * expert_mlp
    router_per_layer = d * config.n_routed_experts
    moe_layers = config.n_layers - config.first_k_dense_replace
    all_moe = moe_layers * (
        routed_experts_per_layer
        + shared_experts_per_layer
        + router_per_layer
    )
    dense_ffn = (
        config.first_k_dense_replace
        * 3
        * d
        * config.dense_intermediate_size
    )
    block_norms = config.n_layers * 2 * d
    embedding = config.vocab_size * d
    lm_head = 0 if config.tie_word_embeddings else embedding
    final_norm = d
    total = (
        embedding
        + lm_head
        + final_norm
        + block_norms
        + config.n_layers * attention_per_layer
        + dense_ffn
        + all_moe
    )

    active_experts = moe_layers * (
        config.n_shared_experts + config.n_experts_per_token
    ) * expert_mlp
    active = (
        embedding
        + lm_head
        + final_norm
        + block_norms
        + config.n_layers * attention_per_layer
        + dense_ffn
        + moe_layers * router_per_layer
        + active_experts
    )
    return {
        "embedding": embedding,
        "lm_head": lm_head,
        "attention_per_layer": attention_per_layer,
        "attention_all_layers": config.n_layers * attention_per_layer,
        "expert_mlp": expert_mlp,
        "routed_experts_per_moe_layer": routed_experts_per_layer,
        "shared_experts_per_moe_layer": shared_experts_per_layer,
        "router_per_moe_layer": router_per_layer,
        "all_moe_layers": all_moe,
        "dense_ffn": dense_ffn,
        "total": total,
        "active_per_token": active,
    }


def kv_cache_elements(
    config: DeepSeekV2Config, *, mechanism: str, gqa_groups: int = 8
) -> int:
    """Elements cached per token per layer under comparable head dimensions."""

    if mechanism == "mha":
        return 2 * config.n_heads * config.qk_nope_head_dim
    if mechanism == "gqa":
        return 2 * gqa_groups * config.qk_nope_head_dim
    if mechanism == "mqa":
        return 2 * config.qk_nope_head_dim
    if mechanism == "mla":
        return config.cached_elements_per_token_per_layer
    raise ValueError(f"unknown mechanism: {mechanism}")


def cache_bytes(
    config: DeepSeekV2Config,
    *,
    sequence_length: int,
    mechanism: str,
    bits_per_element: int = 16,
    gqa_groups: int = 8,
) -> float:
    elements = (
        sequence_length
        * config.n_layers
        * kv_cache_elements(
            config, mechanism=mechanism, gqa_groups=gqa_groups
        )
    )
    return elements * bits_per_element / 8


def _dot(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    return [_dot(row, vector) for row in matrix]


def _transpose_matvec(matrix: Matrix, vector: Vector) -> Vector:
    return [
        sum(matrix[row][column] * vector[row] for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    ]


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    return [
        [
            sum(left_row[k] * right[k][column] for k in range(len(right)))
            for column in range(len(right[0]))
        ]
        for left_row in left
    ]


def _add(left: Vector, right: Vector) -> Vector:
    return [x + y for x, y in zip(left, right)]


def _softmax(scores: Vector) -> Vector:
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _split(vector: Vector, parts: int) -> list[Vector]:
    if len(vector) % parts:
        raise ValueError("vector width must be divisible by parts")
    width = len(vector) // parts
    return [vector[index * width : (index + 1) * width] for index in range(parts)]


def _column_block(matrix: Matrix, start: int, width: int) -> Matrix:
    return [row[start : start + width] for row in matrix]


def _rope(vector: Vector, position: int, theta: float = 10_000.0) -> Vector:
    if len(vector) % 2:
        raise ValueError("RoPE width must be even")
    rotated: Vector = []
    for pair in range(0, len(vector), 2):
        frequency = theta ** (-pair / len(vector))
        angle = position * frequency
        cosine, sine = math.cos(angle), math.sin(angle)
        x0, x1 = vector[pair], vector[pair + 1]
        rotated.extend([x0 * cosine - x1 * sine, x0 * sine + x1 * cosine])
    return rotated


@dataclass(frozen=True)
class ToyMLAWeights:
    q_down: Matrix
    q_nope_up: list[Matrix]
    q_rope_up: list[Matrix]
    kv_down: Matrix
    k_nope_up: list[Matrix]
    v_up: list[Matrix]
    k_rope: Matrix
    output: Matrix


def _random_matrix(rng: random.Random, rows: int, columns: int) -> Matrix:
    scale = 1.0 / math.sqrt(columns)
    return [
        [rng.uniform(-scale, scale) for _ in range(columns)]
        for _ in range(rows)
    ]


def make_toy_mla_weights(
    *,
    rng: random.Random,
    d_model: int,
    n_heads: int,
    q_rank: int,
    kv_rank: int,
    nope_dim: int,
    rope_dim: int,
    value_dim: int,
) -> ToyMLAWeights:
    return ToyMLAWeights(
        q_down=_random_matrix(rng, q_rank, d_model),
        q_nope_up=[
            _random_matrix(rng, nope_dim, q_rank) for _ in range(n_heads)
        ],
        q_rope_up=[
            _random_matrix(rng, rope_dim, q_rank) for _ in range(n_heads)
        ],
        kv_down=_random_matrix(rng, kv_rank, d_model),
        k_nope_up=[
            _random_matrix(rng, nope_dim, kv_rank) for _ in range(n_heads)
        ],
        v_up=[
            _random_matrix(rng, value_dim, kv_rank) for _ in range(n_heads)
        ],
        k_rope=_random_matrix(rng, rope_dim, d_model),
        output=_random_matrix(rng, d_model, n_heads * value_dim),
    )


@dataclass(frozen=True)
class MLACacheEntry:
    kv_latent: Vector
    rope_key: Vector


def _prepare_mla(
    hidden_states: list[Vector], weights: ToyMLAWeights
) -> tuple[list[Vector], list[list[Vector]], list[MLACacheEntry]]:
    query_latents: list[Vector] = []
    query_rope: list[list[Vector]] = []
    cache: list[MLACacheEntry] = []
    for position, hidden in enumerate(hidden_states):
        q_latent = _matvec(weights.q_down, hidden)
        query_latents.append(q_latent)
        query_rope.append(
            [
                _rope(_matvec(head_matrix, q_latent), position)
                for head_matrix in weights.q_rope_up
            ]
        )
        cache.append(
            MLACacheEntry(
                kv_latent=_matvec(weights.kv_down, hidden),
                rope_key=_rope(_matvec(weights.k_rope, hidden), position),
            )
        )
    return query_latents, query_rope, cache


def mla_naive(
    hidden_states: list[Vector], weights: ToyMLAWeights
) -> list[Vector]:
    """Reference path: reconstruct every historical content K and V."""

    query_latents, query_rope, cache = _prepare_mla(hidden_states, weights)
    n_heads = len(weights.q_nope_up)
    value_dim = len(weights.v_up[0])
    scale = 1.0 / math.sqrt(
        len(weights.q_nope_up[0]) + len(weights.q_rope_up[0])
    )
    output: list[Vector] = []
    for position in range(len(hidden_states)):
        head_outputs: list[Vector] = []
        for head in range(n_heads):
            q_nope = _matvec(weights.q_nope_up[head], query_latents[position])
            scores: Vector = []
            values: list[Vector] = []
            for past in range(position + 1):
                k_nope = _matvec(
                    weights.k_nope_up[head], cache[past].kv_latent
                )
                value = _matvec(weights.v_up[head], cache[past].kv_latent)
                scores.append(
                    scale
                    * (
                        _dot(q_nope, k_nope)
                        + _dot(query_rope[position][head], cache[past].rope_key)
                    )
                )
                values.append(value)
            probabilities = _softmax(scores)
            head_outputs.append(
                [
                    sum(
                        probability * value[column]
                        for probability, value in zip(probabilities, values)
                    )
                    for column in range(value_dim)
                ]
            )
        output.append(_matvec(weights.output, sum(head_outputs, [])))
    return output


def mla_absorbed(
    hidden_states: list[Vector], weights: ToyMLAWeights
) -> list[Vector]:
    """Inference path: attend directly to cached latents.

    For each head, U_K is absorbed into the query-side projection and U_V is
    absorbed into that head's output-projection block.  Historical content K
    and V tensors are never reconstructed.
    """

    query_latents, query_rope, cache = _prepare_mla(hidden_states, weights)
    n_heads = len(weights.q_nope_up)
    value_dim = len(weights.v_up[0])
    scale = 1.0 / math.sqrt(
        len(weights.q_nope_up[0]) + len(weights.q_rope_up[0])
    )
    absorbed_value_outputs = []
    for head in range(n_heads):
        output_block = _column_block(
            weights.output, head * value_dim, value_dim
        )
        absorbed_value_outputs.append(_matmul(output_block, weights.v_up[head]))

    output: list[Vector] = []
    for position in range(len(hidden_states)):
        token_output = [0.0] * len(hidden_states[0])
        for head in range(n_heads):
            q_nope = _matvec(weights.q_nope_up[head], query_latents[position])
            latent_query = _transpose_matvec(weights.k_nope_up[head], q_nope)
            scores = [
                scale
                * (
                    _dot(latent_query, cache[past].kv_latent)
                    + _dot(query_rope[position][head], cache[past].rope_key)
                )
                for past in range(position + 1)
            ]
            probabilities = _softmax(scores)
            latent_value = [
                sum(
                    probabilities[past] * cache[past].kv_latent[column]
                    for past in range(position + 1)
                )
                for column in range(len(cache[0].kv_latent))
            ]
            token_output = _add(
                token_output,
                _matvec(absorbed_value_outputs[head], latent_value),
            )
        output.append(token_output)
    return output


def device_limited_topk(
    logits: Vector,
    *,
    experts_per_device: int,
    max_devices: int,
    top_k: int,
) -> tuple[list[int], list[int], Vector]:
    """Select candidate devices by their best affinity, then Top-K experts."""

    if len(logits) % experts_per_device:
        raise ValueError("experts must be evenly partitioned across devices")
    probabilities = _softmax(logits)
    n_devices = len(logits) // experts_per_device
    device_scores = [
        max(
            probabilities[
                device * experts_per_device : (device + 1) * experts_per_device
            ]
        )
        for device in range(n_devices)
    ]
    selected_devices = sorted(
        range(n_devices), key=lambda index: device_scores[index], reverse=True
    )[:max_devices]
    candidates = [
        expert
        for device in selected_devices
        for expert in range(
            device * experts_per_device, (device + 1) * experts_per_device
        )
    ]
    selected_experts = sorted(
        candidates, key=lambda index: probabilities[index], reverse=True
    )[:top_k]
    return selected_devices, selected_experts, probabilities


def _toy_expert(vector: Vector, expert_id: int) -> Vector:
    """A deterministic stand-in for one SwiGLU expert."""

    gain = 0.02 * (expert_id + 3)
    return [
        math.tanh(value * gain + (column + 1) * 0.01)
        for column, value in enumerate(vector)
    ]


def deepseek_moe_forward(
    vector: Vector,
    router_logits: Vector,
    *,
    n_shared_experts: int,
    experts_per_device: int,
    max_devices: int,
    top_k: int,
    routed_scaling_factor: float = 1.0,
) -> tuple[Vector, list[int], list[int]]:
    """Always run shared experts and sparsely mix device-limited routed ones."""

    devices, experts, probabilities = device_limited_topk(
        router_logits,
        experts_per_device=experts_per_device,
        max_devices=max_devices,
        top_k=top_k,
    )
    output = vector[:]
    for shared in range(n_shared_experts):
        output = _add(output, _toy_expert(vector, -(shared + 1)))
    for expert in experts:
        contribution = _toy_expert(vector, expert)
        gate = routed_scaling_factor * probabilities[expert]
        output = _add(output, [gate * value for value in contribution])
    return output, devices, experts


def _assert_close(
    left: list[Vector], right: list[Vector], *, tolerance: float = 1e-11
) -> float:
    maximum = 0.0
    for left_row, right_row in zip(left, right):
        for left_value, right_value in zip(left_row, right_row):
            difference = abs(left_value - right_value)
            maximum = max(maximum, difference)
            if difference > tolerance:
                raise AssertionError(f"{left_value} != {right_value}")
    return maximum


def _gib(byte_count: float) -> float:
    return byte_count / (1024**3)


def main() -> None:
    config = DeepSeekV2Config()
    ledger = parameter_ledger(config)
    assert ledger["total"] == 235_741_434_880
    assert ledger["active_per_token"] == 21_375_800_320

    print("DeepSeek-V2 architecture ledger")
    print(f"  total parameters from public dimensions: {ledger['total']:,} (~236B)")
    print(
        "  active parameters under this counting convention: "
        f"{ledger['active_per_token']:,} (~21B)"
    )
    print("  MoE layers: 59; per layer: 2 shared + Top-6 of 160 routed")

    print("\nKV elements per token per layer")
    for mechanism in ("mha", "gqa", "mqa", "mla"):
        elements = kv_cache_elements(config, mechanism=mechanism)
        print(f"  {mechanism.upper():3}: {elements:,}")
    mha = kv_cache_elements(config, mechanism="mha")
    mla = kv_cache_elements(config, mechanism="mla")
    print(f"  MLA / same-shape MHA: {mla / mha:.3%}")
    print(f"  reduction: {1.0 - mla / mha:.3%}")

    sequence_length = 131_072
    mha_bytes = cache_bytes(
        config, sequence_length=sequence_length, mechanism="mha"
    )
    mla_bf16_bytes = cache_bytes(
        config, sequence_length=sequence_length, mechanism="mla"
    )
    mla_6bit_bytes = cache_bytes(
        config,
        sequence_length=sequence_length,
        mechanism="mla",
        bits_per_element=6,
    )
    print("\nIdeal one-sequence cache across 60 layers at 128K")
    print(f"  same-shape MHA bf16: {_gib(mha_bytes):.2f} GiB")
    print(f"  MLA bf16:            {_gib(mla_bf16_bytes):.2f} GiB")
    print(f"  MLA at 6 bits/elem:  {_gib(mla_6bit_bytes):.2f} GiB")

    rng = random.Random(2024)
    d_model, n_heads = 6, 3
    q_rank, kv_rank = 4, 3
    nope_dim, rope_dim, value_dim = 2, 2, 2
    weights = make_toy_mla_weights(
        rng=rng,
        d_model=d_model,
        n_heads=n_heads,
        q_rank=q_rank,
        kv_rank=kv_rank,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        value_dim=value_dim,
    )
    hidden_states = [
        [rng.uniform(-1.0, 1.0) for _ in range(d_model)] for _ in range(7)
    ]
    naive = mla_naive(hidden_states, weights)
    absorbed = mla_absorbed(hidden_states, weights)
    maximum_difference = _assert_close(naive, absorbed)

    router_logits = [rng.uniform(-2.0, 2.0) for _ in range(20)]
    _, devices, experts = deepseek_moe_forward(
        hidden_states[-1],
        router_logits,
        n_shared_experts=2,
        experts_per_device=5,
        max_devices=2,
        top_k=3,
    )
    assert len(devices) == 2
    assert len(experts) == 3
    assert all(expert // 5 in devices for expert in experts)

    print("\nCorrectness checks")
    print("  reconstructed K/V MLA == absorbed latent MLA: yes")
    print(f"  maximum numerical difference: {maximum_difference:.3e}")
    print(f"  toy routed devices: {devices}")
    print(f"  toy Top-3 experts: {experts}")
    print("  both shared experts also executed: yes")


if __name__ == "__main__":
    main()
