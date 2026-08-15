"""Dependency-free reference for the new mechanisms in DeepSeek-V3.

This scalar Python program is an explanatory oracle, not a training kernel. It
demonstrates four ideas from the technical report:

1. the public inference dimensions reconstruct roughly 671B total parameters
   and 37B parameters activated for one token;
2. auxiliary-loss-free routing selects experts with corrected scores ``s+b``
   but computes gate weights from the original sigmoid affinities ``s``;
3. fine-grained E4M3 scaling protects small values from distant outliers;
4. one-depth MTP predicts token i+2 while preserving the causal chain through
   the known embedding of token i+1.

Run:
    python3 papers/to-2026/code/deepseek_v3_minimal.py
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
import random
from typing import Iterable


Vector = list[float]
Matrix = list[Vector]


@dataclass(frozen=True)
class DeepSeekV3Config:
    hidden_size: int = 7168
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    dense_intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    n_experts_per_token: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    routed_scaling_factor: float = 2.5
    vocab_size: int = 129280
    tie_word_embeddings: bool = False
    mtp_depth: int = 1

    @property
    def n_moe_layers(self) -> int:
        return self.n_layers - self.n_dense_layers

    @property
    def cached_elements_per_token_per_layer(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim


def parameter_ledger(config: DeepSeekV3Config) -> dict[str, int]:
    """Rebuild the rounded 671B / 37B claims from inference dimensions.

    The main-model ledger includes both untied vocabulary matrices, all
    attention/router parameters, the selected experts, and normalization
    weights. The detachable MTP training module is deliberately excluded from
    the headline inference count.
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

    expert = 3 * d * config.moe_intermediate_size
    # The released inference implementation stores one correction bias per
    # routed expert in addition to the learned router centroid matrix.
    router = d * config.n_routed_experts + config.n_routed_experts
    dense_ffns = (
        config.n_dense_layers
        * 3
        * d
        * config.dense_intermediate_size
    )
    all_moe = config.n_moe_layers * (
        (config.n_routed_experts + config.n_shared_experts) * expert
        + router
    )
    active_moe = config.n_moe_layers * (
        (config.n_experts_per_token + config.n_shared_experts) * expert
        + router
    )
    embedding = config.vocab_size * d
    lm_head = 0 if config.tie_word_embeddings else embedding
    norms = config.n_layers * 2 * d + d

    shared_non_moe = (
        embedding
        + lm_head
        + config.n_layers * attention_per_layer
        + dense_ffns
        + norms
    )
    return {
        "embedding": embedding,
        "lm_head": lm_head,
        "attention_per_layer": attention_per_layer,
        "attention_all_layers": config.n_layers * attention_per_layer,
        "expert": expert,
        "router_per_moe_layer": router,
        "dense_ffns": dense_ffns,
        "all_moe_layers": all_moe,
        "total": shared_non_moe + all_moe,
        "active_per_token": shared_non_moe + active_moe,
    }


def training_cost_ledger() -> dict[str, int | float]:
    pretraining = 2_664_000
    context_extension = 119_000
    post_training = 5_000
    total = pretraining + context_extension + post_training
    return {
        "pretraining_gpu_hours": pretraining,
        "context_gpu_hours": context_extension,
        "post_training_gpu_hours": post_training,
        "total_gpu_hours": total,
        "usd_at_two_dollars_per_hour": total * 2,
        "pretraining_hours_per_trillion_tokens": pretraining / 14.8,
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _topk(indices: Iterable[int], scores: Vector, k: int) -> list[int]:
    return sorted(indices, key=lambda index: (-scores[index], index))[:k]


@dataclass(frozen=True)
class RoutingDecision:
    affinities: Vector
    corrected_scores: Vector
    selected_groups: list[int]
    selected_experts: list[int]
    gate_weights: Vector


def auxiliary_loss_free_route(
    logits: Vector,
    correction_bias: Vector,
    *,
    n_groups: int,
    n_limited_groups: int,
    top_k: int,
    routed_scaling_factor: float,
) -> RoutingDecision:
    """V3-style sigmoid routing with correction bias and group limits.

    For the released 671B model, 256 experts form 8 groups. A group score is
    the sum of its best two corrected expert scores (K/M = 8/4 = 2), the best
    four groups remain candidates, and the final eight experts are selected.
    """

    if len(logits) != len(correction_bias):
        raise ValueError("one correction bias is required per expert")
    if len(logits) % n_groups:
        raise ValueError("experts must be evenly divisible into groups")
    if top_k % n_limited_groups:
        raise ValueError("this reference expects K/M to be integral")

    affinities = [_sigmoid(logit) for logit in logits]
    corrected = [
        affinity + bias
        for affinity, bias in zip(affinities, correction_bias)
    ]
    experts_per_group = len(logits) // n_groups
    best_per_group = top_k // n_limited_groups
    group_scores: Vector = []
    for group in range(n_groups):
        start = group * experts_per_group
        members = range(start, start + experts_per_group)
        strongest = _topk(members, corrected, best_per_group)
        group_scores.append(sum(corrected[index] for index in strongest))

    selected_groups = _topk(range(n_groups), group_scores, n_limited_groups)
    candidates = [
        expert
        for group in selected_groups
        for expert in range(
            group * experts_per_group,
            (group + 1) * experts_per_group,
        )
    ]
    selected_experts = _topk(candidates, corrected, top_k)

    # Correction bias decides *which* experts run. The original, uncorrected
    # sigmoid affinity decides how their outputs are mixed.
    denominator = sum(affinities[index] for index in selected_experts)
    gate_weights = [
        routed_scaling_factor * affinities[index] / denominator
        for index in selected_experts
    ]
    return RoutingDecision(
        affinities=affinities,
        corrected_scores=corrected,
        selected_groups=selected_groups,
        selected_experts=selected_experts,
        gate_weights=gate_weights,
    )


def update_correction_bias(
    correction_bias: Vector,
    load: list[int],
    *,
    gamma: float,
) -> Vector:
    """Raise underloaded biases and lower overloaded biases by gamma."""

    if len(correction_bias) != len(load):
        raise ValueError("load and correction bias must have equal width")
    target = sum(load) / len(load)
    updated: Vector = []
    for bias, expert_load in zip(correction_bias, load):
        if expert_load > target:
            updated.append(bias - gamma)
        elif expert_load < target:
            updated.append(bias + gamma)
        else:
            updated.append(bias)
    return updated


def _coefficient_of_variation(load: list[int]) -> float:
    mean = sum(load) / len(load)
    variance = sum((value - mean) ** 2 for value in load) / len(load)
    return math.sqrt(variance) / mean


def simulate_routing_balance(
    token_logits: list[Vector],
    config: DeepSeekV3Config,
    *,
    batch_size: int = 64,
    gamma: float = 0.01,
) -> tuple[float, float, Vector]:
    """Compare fixed-bias routing against batch-feedback routing."""

    def route_all(adaptive: bool) -> tuple[list[int], Vector]:
        bias = [0.0] * config.n_routed_experts
        total_load = [0] * config.n_routed_experts
        batch_load = [0] * config.n_routed_experts
        for token_index, logits in enumerate(token_logits, start=1):
            decision = auxiliary_loss_free_route(
                logits,
                bias,
                n_groups=config.n_expert_groups,
                n_limited_groups=config.n_limited_groups,
                top_k=config.n_experts_per_token,
                routed_scaling_factor=config.routed_scaling_factor,
            )
            for expert in decision.selected_experts:
                total_load[expert] += 1
                batch_load[expert] += 1
            if adaptive and token_index % batch_size == 0:
                bias = update_correction_bias(bias, batch_load, gamma=gamma)
                batch_load = [0] * config.n_routed_experts
        return total_load, bias

    fixed_load, _ = route_all(adaptive=False)
    adaptive_load, final_bias = route_all(adaptive=True)
    return (
        _coefficient_of_variation(fixed_load),
        _coefficient_of_variation(adaptive_load),
        final_bias,
    )


def _e4m3fn_positive_levels() -> Vector:
    """Finite non-negative values of NVIDIA-style E4M3FN."""

    levels = [0.0]
    # Exponent field 0: subnormals, exponent 1-bias = -6.
    levels.extend((mantissa / 8.0) * 2.0**-6 for mantissa in range(1, 8))
    # Exponent fields 1..14: ordinary normal values.
    for exponent_field in range(1, 15):
        exponent = exponent_field - 7
        levels.extend(
            (1.0 + mantissa / 8.0) * 2.0**exponent
            for mantissa in range(8)
        )
    # E4M3FN repurposes most of field 15 for finite values; 448 is max.
    levels.extend(
        (1.0 + mantissa / 8.0) * 2.0**8 for mantissa in range(7)
    )
    return sorted(set(levels))


E4M3FN_LEVELS = _e4m3fn_positive_levels()
E4M3FN_MAX = E4M3FN_LEVELS[-1]


def _nearest_positive_e4m3(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= E4M3FN_MAX:
        return E4M3FN_MAX
    right = bisect_left(E4M3FN_LEVELS, value)
    left = right - 1
    if value - E4M3FN_LEVELS[left] <= E4M3FN_LEVELS[right] - value:
        return E4M3FN_LEVELS[left]
    return E4M3FN_LEVELS[right]


def quantize_e4m3_group(values: Vector) -> tuple[Vector, float]:
    """Online max-abs scaling followed by E4M3FN quantize/dequantize."""

    maximum = max((abs(value) for value in values), default=0.0)
    if maximum == 0.0:
        return values[:], 1.0
    scale = maximum / E4M3FN_MAX
    dequantized = [
        math.copysign(_nearest_positive_e4m3(abs(value) / scale), value)
        * scale
        for value in values
    ]
    return dequantized, scale


def quantize_e4m3_tiles(
    values: Vector, *, tile_size: int
) -> tuple[Vector, Vector]:
    output: Vector = []
    scales: Vector = []
    for start in range(0, len(values), tile_size):
        dequantized, scale = quantize_e4m3_group(
            values[start : start + tile_size]
        )
        output.extend(dequantized)
        scales.append(scale)
    return output, scales


def mean_squared_error(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    return sum((x - y) ** 2 for x, y in zip(left, right)) / len(left)


def _rmsnorm(vector: Vector, epsilon: float = 1e-6) -> Vector:
    scale = math.sqrt(
        sum(value * value for value in vector) / len(vector) + epsilon
    )
    return [value / scale for value in vector]


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    return [
        sum(weight * value for weight, value in zip(row, vector))
        for row in matrix
    ]


def mtp_fuse(
    previous_depth_hidden: Vector,
    next_input_embedding: Vector,
    projection: Matrix,
) -> Vector:
    """Equation 21: fuse h_i^(k-1) with Emb(t_(i+k))."""

    joined = _rmsnorm(previous_depth_hidden) + _rmsnorm(next_input_embedding)
    return _matvec(projection, joined)


def mtp_target_pairs(tokens: list[int], *, depth: int) -> list[tuple[int, int]]:
    """Return (representation position, target token) for one MTP depth.

    depth=0 is ordinary next-token prediction. depth=1 is the one additional
    token used by the released DeepSeek-V3 configuration.
    """

    if depth < 0:
        raise ValueError("depth must be non-negative")
    offset = depth + 1
    return [
        (position, tokens[position + offset])
        for position in range(len(tokens) - offset)
    ]


def cross_entropy(probabilities: list[Vector], targets: list[int]) -> float:
    if len(probabilities) != len(targets):
        raise ValueError("one probability row is required per target")
    return -sum(
        math.log(max(row[target], 1e-12))
        for row, target in zip(probabilities, targets)
    ) / len(targets)


def _target_favoring_rows(targets: list[int], vocab_size: int) -> list[Vector]:
    rows: list[Vector] = []
    for target in targets:
        row = [0.1 / (vocab_size - 1)] * vocab_size
        row[target] = 0.9
        rows.append(row)
    return rows


def main() -> None:
    config = DeepSeekV3Config()
    ledger = parameter_ledger(config)
    assert ledger["total"] == 671_026_419_200
    assert ledger["active_per_token"] == 37_552_297_472
    assert config.cached_elements_per_token_per_layer == 576

    print("DeepSeek-V3 architecture ledger")
    print(f"  total parameters:  {ledger['total']:,} (~671B)")
    print(f"  active per token:  {ledger['active_per_token']:,} (~37B)")
    print(
        "  MoE layout:       "
        f"{config.n_moe_layers} layers, 1 shared + Top-8 of 256 routed"
    )
    print("  MLA cache width:   512 + 64 = 576 elements/token/layer")

    cost = training_cost_ledger()
    assert cost["total_gpu_hours"] == 2_788_000
    print("\nOfficial-run training cost ledger")
    print(
        "  pretrain / context / post: "
        f"{cost['pretraining_gpu_hours']:,} / "
        f"{cost['context_gpu_hours']:,} / "
        f"{cost['post_training_gpu_hours']:,} H800 GPU hours"
    )
    print(f"  total:                    {cost['total_gpu_hours']:,}")
    print(
        "  paper's $2/hour estimate: "
        f"${cost['usd_at_two_dollars_per_hour'] / 1e6:.3f}M"
    )

    rng = random.Random(2024)
    token_logits: list[Vector] = []
    for _ in range(1_024):
        # A persistent preference for the first 16 experts creates imbalance;
        # per-token noise still leaves room for correction-bias feedback.
        token_logits.append(
            [
                rng.gauss(1.2 if expert < 16 else 0.0, 0.7)
                for expert in range(config.n_routed_experts)
            ]
        )
    fixed_cv, adaptive_cv, final_bias = simulate_routing_balance(
        token_logits, config
    )
    assert adaptive_cv < fixed_cv

    example = auxiliary_loss_free_route(
        token_logits[-1],
        final_bias,
        n_groups=config.n_expert_groups,
        n_limited_groups=config.n_limited_groups,
        top_k=config.n_experts_per_token,
        routed_scaling_factor=config.routed_scaling_factor,
    )
    assert len(example.selected_groups) == 4
    assert len(example.selected_experts) == 8
    assert math.isclose(sum(example.gate_weights), 2.5)
    print("\nAuxiliary-loss-free routing")
    print(f"  fixed-bias load CV:       {fixed_cv:.3f}")
    print(f"  feedback-bias load CV:    {adaptive_cv:.3f}")
    print(f"  selected groups:          {example.selected_groups}")
    print(f"  selected experts:         {example.selected_experts}")
    print(f"  sum of original-s gates:  {sum(example.gate_weights):.3f}")
    print("  token assignments dropped: 0")

    # One distant outlier suppresses tiny values under a single global scale.
    # The 1,024-channel row gives us eight paper-sized 1x128 tiles: only the
    # final tile has to share its scale with the outlier.
    activations = [
        rng.uniform(-2e-4, 2e-4) for _ in range(1_023)
    ] + [100.0]
    tensor_dequantized, _ = quantize_e4m3_group(activations)
    tiled_dequantized, tiled_scales = quantize_e4m3_tiles(
        activations, tile_size=128
    )
    tensor_mse = mean_squared_error(activations, tensor_dequantized)
    tiled_mse = mean_squared_error(activations, tiled_dequantized)
    assert tiled_mse < tensor_mse
    assert len(tiled_scales) == 8
    print("\nFine-grained E4M3 quantization toy")
    print(f"  tensor-wise MSE:          {tensor_mse:.3e}")
    print(f"  tile-wise MSE:            {tiled_mse:.3e}")
    print("  paper granularity:        activations 1x128, weights 128x128")

    tokens = [2, 5, 1, 4, 3, 0]
    next_pairs = mtp_target_pairs(tokens, depth=0)
    next_next_pairs = mtp_target_pairs(tokens, depth=1)
    assert next_pairs[0] == (0, 5)
    assert next_next_pairs[0] == (0, 1)

    projection = [
        [rng.uniform(-0.2, 0.2) for _ in range(8)] for _ in range(4)
    ]
    fused = mtp_fuse(
        previous_depth_hidden=[0.2, -0.1, 0.4, 0.7],
        next_input_embedding=[-0.3, 0.5, 0.1, 0.2],
        projection=projection,
    )
    assert len(fused) == 4

    main_targets = [target for _, target in next_pairs]
    mtp_targets = [target for _, target in next_next_pairs]
    main_loss = cross_entropy(
        _target_favoring_rows(main_targets, vocab_size=6), main_targets
    )
    mtp_loss = cross_entropy(
        _target_favoring_rows(mtp_targets, vocab_size=6), mtp_targets
    )
    loss_weight = 0.3
    total_loss = main_loss + loss_weight * mtp_loss
    print("\nOne-depth MTP alignment")
    print(f"  ordinary targets:         {main_targets}")
    print(f"  additional targets:       {mtp_targets}")
    print(f"  L_main + 0.3 L_MTP:       {total_loss:.6f}")
    print("  MTP module at inference:  discard or reuse as a draft head")


if __name__ == "__main__":
    main()
