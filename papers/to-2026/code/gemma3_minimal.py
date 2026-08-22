#!/usr/bin/env python3
"""Zero-dependency sketches of several Gemma 3 mechanisms.

This is not a model implementation.  It makes four report-level ideas
executable and testable:

1. five local sliding-window layers followed by one global layer;
2. the resulting structural KV-cache saving at long context;
3. the fixed 256 soft-token budget of one vision-encoder pass;
4. sparse teacher-logit distillation over 256 sampled vocabulary entries.

Run:
    python3 gemma3_minimal.py --model 4b --context 131072
    python3 gemma3_minimal.py --test
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Sequence


LOCAL = "local"
GLOBAL = "global"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture fields relevant to the educational calculations.

    Layer/head values follow the released Google DeepMind JAX implementation.
    Parameter counts and runtime allocations are intentionally not inferred
    from these few fields.
    """

    name: str
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    max_context: int
    local_window: int
    has_vision: bool


MODEL_CONFIGS = {
    "1b": ModelConfig("Gemma 3 1B", 26, 4, 1, 256, 32_768, 512, False),
    "4b": ModelConfig("Gemma 3 4B", 34, 8, 4, 256, 131_072, 1_024, True),
    "12b": ModelConfig("Gemma 3 12B", 48, 16, 8, 256, 131_072, 1_024, True),
    "27b": ModelConfig("Gemma 3 27B", 62, 32, 16, 128, 131_072, 1_024, True),
}


def attention_pattern(
    num_layers: int, local_layers_per_global: int = 5
) -> tuple[str, ...]:
    """Repeat L...L,G and truncate to ``num_layers``.

    Gemma 3 starts at a local layer.  With the default ratio, layer indices
    0..4 are local and index 5 is global.
    """

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if local_layers_per_global <= 0:
        raise ValueError("local_layers_per_global must be positive")
    period = local_layers_per_global + 1
    return tuple(
        GLOBAL if (layer + 1) % period == 0 else LOCAL
        for layer in range(num_layers)
    )


def visible_key_range(
    layer_index: int,
    query_position: int,
    local_window: int = 1_024,
    local_layers_per_global: int = 5,
) -> tuple[int, int]:
    """Return the half-open causal key range visible at one layer.

    This abstracts text self-attention only.  Gemma's multimodal mask has
    additional image-block semantics that are outside this minimal function.
    """

    if layer_index < 0 or query_position < 0:
        raise ValueError("indices must be non-negative")
    if local_window <= 0:
        raise ValueError("local_window must be positive")
    layer_type = attention_pattern(
        layer_index + 1, local_layers_per_global
    )[-1]
    end = query_position + 1
    if layer_type == GLOBAL:
        return (0, end)
    return (max(0, end - local_window), end)


def kv_cache_bytes(
    config: ModelConfig,
    context: int,
    bytes_per_element: int = 2,
    batch_size: int = 1,
) -> int:
    """Estimate raw K/V tensor bytes for Gemma 3's attention pattern.

    Formula:
        2 * batch * H_kv * head_dim * element_bytes
          * sum(cached_positions_per_layer)

    It excludes allocator overhead, temporary tensors, vision states, padding,
    quantization metadata and framework-specific cache layouts.
    """

    if not 0 < context <= config.max_context:
        raise ValueError(f"context must be in [1, {config.max_context}]")
    if bytes_per_element <= 0 or batch_size <= 0:
        raise ValueError("storage size and batch size must be positive")

    cached_positions = 0
    for layer_type in attention_pattern(config.num_layers):
        cached_positions += (
            context if layer_type == GLOBAL else min(context, config.local_window)
        )
    elements = (
        2
        * batch_size
        * config.num_kv_heads
        * config.head_dim
        * cached_positions
    )
    return elements * bytes_per_element


def all_global_kv_cache_bytes(
    config: ModelConfig,
    context: int,
    bytes_per_element: int = 2,
    batch_size: int = 1,
) -> int:
    """Raw K/V tensor bytes if every layer cached the full context."""

    if not 0 < context <= config.max_context:
        raise ValueError(f"context must be in [1, {config.max_context}]")
    elements = (
        2
        * batch_size
        * config.num_layers
        * config.num_kv_heads
        * config.head_dim
        * context
    )
    return elements * bytes_per_element


def vision_soft_tokens(
    image_count: int = 1,
    encoder_passes_per_image: int = 1,
    tokens_per_pass: int = 256,
) -> int:
    """Context tokens occupied by image encoder passes.

    A normal image needs one pass.  Pan & Scan can add crops, each separately
    resized and encoded; the report does not publish an exact crop planner, so
    callers provide the number of passes rather than pretending to reproduce it.
    """

    if image_count < 0 or encoder_passes_per_image <= 0 or tokens_per_pass <= 0:
        raise ValueError("image_count must be non-negative; other values positive")
    return image_count * encoder_passes_per_image * tokens_per_pass


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("values must not be empty")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def sparse_teacher_target(
    teacher_logits: Sequence[float], sampled_token_ids: Sequence[int]
) -> list[float]:
    """Zero non-sampled teacher targets and renormalize sampled logits.

    The paper samples 256 vocabulary entries per token, weighted by teacher
    probabilities.  This function receives the already-selected IDs so the
    probability-weighted sampling procedure can be tested separately.
    """

    if not teacher_logits:
        raise ValueError("teacher_logits must not be empty")
    sampled = list(sampled_token_ids)
    if not sampled or len(sampled) != len(set(sampled)):
        raise ValueError("sampled_token_ids must be non-empty and unique")
    if min(sampled) < 0 or max(sampled) >= len(teacher_logits):
        raise IndexError("sampled token id is outside the vocabulary")

    selected_probabilities = _softmax([teacher_logits[index] for index in sampled])
    target = [0.0] * len(teacher_logits)
    for index, probability in zip(sampled, selected_probabilities):
        target[index] = probability
    return target


def sparse_distillation_cross_entropy(
    teacher_logits: Sequence[float],
    student_logits: Sequence[float],
    sampled_token_ids: Sequence[int],
) -> float:
    """Cross-entropy from the sparse teacher target to the full student."""

    if len(teacher_logits) != len(student_logits):
        raise ValueError("teacher and student vocabularies must match")
    target = sparse_teacher_target(teacher_logits, sampled_token_ids)
    student_probabilities = _softmax(student_logits)
    return -sum(
        target_probability * math.log(max(student_probability, 1e-300))
        for target_probability, student_probability in zip(
            target, student_probabilities
        )
        if target_probability > 0.0
    )


def ideal_weight_storage_gb(parameters_billions: float, bits_per_weight: int) -> float:
    """Decimal-GB lower bound before scales, padding and metadata."""

    if parameters_billions <= 0 or bits_per_weight <= 0:
        raise ValueError("parameters and bits must be positive")
    return parameters_billions * bits_per_weight / 8.0


def gibibytes(byte_count: int) -> float:
    return byte_count / (1024**3)


def run_tests() -> None:
    assert attention_pattern(6) == (LOCAL, LOCAL, LOCAL, LOCAL, LOCAL, GLOBAL)
    assert attention_pattern(8) == (
        LOCAL,
        LOCAL,
        LOCAL,
        LOCAL,
        LOCAL,
        GLOBAL,
        LOCAL,
        LOCAL,
    )
    assert attention_pattern(34).count(GLOBAL) == 5
    assert visible_key_range(0, 2_000, 1_024) == (977, 2_001)
    assert visible_key_range(5, 2_000, 1_024) == (0, 2_001)

    config = MODEL_CONFIGS["4b"]
    gemma_cache = kv_cache_bytes(config, 131_072)
    global_cache = all_global_kv_cache_bytes(config, 131_072)
    assert 0 < gemma_cache < global_cache
    assert gemma_cache / global_cache < 0.20

    assert vision_soft_tokens() == 256
    assert vision_soft_tokens(image_count=2, encoder_passes_per_image=3) == 1_536

    teacher = [3.0, 1.0, -1.0, 0.0]
    target = sparse_teacher_target(teacher, [0, 3])
    assert math.isclose(sum(target), 1.0)
    assert target[1] == 0.0 and target[2] == 0.0
    aligned = sparse_distillation_cross_entropy(
        teacher, [3.0, -20.0, -20.0, 0.0], [0, 3]
    )
    reversed_student = sparse_distillation_cross_entropy(
        teacher, [0.0, -20.0, -20.0, 3.0], [0, 3]
    )
    assert aligned < reversed_student

    assert ideal_weight_storage_gb(4.0, 16) == 8.0
    assert ideal_weight_storage_gb(4.0, 4) == 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_CONFIGS, default="4b")
    parser.add_argument("--context", type=int)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        run_tests()
        print("all Gemma 3 minimal tests passed")
        return

    config = MODEL_CONFIGS[args.model]
    context = args.context or config.max_context
    pattern = attention_pattern(config.num_layers)
    local_count = pattern.count(LOCAL)
    global_count = pattern.count(GLOBAL)
    gemma_cache = kv_cache_bytes(config, context)
    global_cache = all_global_kv_cache_bytes(config, context)

    print(config.name)
    print(f"context: {context:,}")
    print(f"attention layers: {local_count} local + {global_count} global")
    print(f"local window: {config.local_window:,}")
    print(f"raw bf16 KV estimate: {gibibytes(gemma_cache):.3f} GiB")
    print(f"all-global comparison: {gibibytes(global_cache):.3f} GiB")
    print(f"structural cache ratio: {gemma_cache / global_cache:.2%}")
    print(f"one image encoder pass: {vision_soft_tokens():,} soft tokens")

    teacher = [2.5, 1.0, -0.5, 0.2]
    sampled_ids = [0, 3]
    print("sparse teacher target:", sparse_teacher_target(teacher, sampled_ids))


if __name__ == "__main__":
    main()
