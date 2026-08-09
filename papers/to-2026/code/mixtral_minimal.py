"""A zero-dependency, inspectable Mixtral-style sparse MoE.

This is teaching code, not an efficient kernel.  It makes four ideas explicit:

1. a linear router produces one logit per expert;
2. only the Top-K logits are normalized;
3. tokens are grouped (dispatched) by expert;
4. weighted expert outputs are added back to the original token positions.

Run:
    python3 papers/to-2026/code/mixtral_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from random import Random
from typing import Iterable, Sequence


Vector = list[float]
Matrix = list[Vector]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def linear(x: Sequence[float], weight: Matrix) -> Vector:
    """Apply a bias-free linear layer whose rows are output channels."""
    return [dot(row, x) for row in weight]


def silu(value: float) -> float:
    return value / (1.0 + exp(-value))


def add_scaled_(target: Vector, source: Sequence[float], scale: float) -> None:
    for index, value in enumerate(source):
        target[index] += scale * value


def top_k_gate(logits: Sequence[float], k: int) -> tuple[list[int], Vector]:
    """Return Top-K indices and a softmax normalized only over those logits."""
    if not 1 <= k <= len(logits):
        raise ValueError("k must be between 1 and the number of experts")

    indices = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:k]
    maximum = max(logits[index] for index in indices)
    numerators = [exp(logits[index] - maximum) for index in indices]
    denominator = sum(numerators)
    return indices, [value / denominator for value in numerators]


def random_matrix(rows: int, columns: int, rng: Random, scale: float) -> Matrix:
    return [
        [rng.uniform(-scale, scale) for _ in range(columns)]
        for _ in range(rows)
    ]


class SwiGLUExpert:
    """E(x) = W2(SiLU(W1 x) * (W3 x))."""

    def __init__(self, hidden_size: int, intermediate_size: int, rng: Random):
        scale = hidden_size**-0.5
        self.w1 = random_matrix(intermediate_size, hidden_size, rng, scale)
        self.w3 = random_matrix(intermediate_size, hidden_size, rng, scale)
        self.w2 = random_matrix(hidden_size, intermediate_size, rng, scale)

    def __call__(self, x: Sequence[float]) -> Vector:
        gate = linear(x, self.w1)
        value = linear(x, self.w3)
        hidden = [silu(g) * v for g, v in zip(gate, value)]
        return linear(hidden, self.w2)


@dataclass(frozen=True)
class Route:
    token: int
    expert: int
    slot: int
    weight: float


class SparseTopKMoE:
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        seed: int = 7,
    ):
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must not exceed num_experts")

        rng = Random(seed)
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.router = random_matrix(
            num_experts, hidden_size, rng, hidden_size**-0.5
        )
        self.experts = [
            SwiGLUExpert(hidden_size, intermediate_size, rng)
            for _ in range(num_experts)
        ]

    def route(self, tokens: Sequence[Sequence[float]]) -> list[Route]:
        routes: list[Route] = []
        for token_index, token in enumerate(tokens):
            logits = linear(token, self.router)
            expert_ids, weights = top_k_gate(logits, self.top_k)
            routes.extend(
                Route(token_index, expert_id, slot, weight)
                for slot, (expert_id, weight) in enumerate(
                    zip(expert_ids, weights)
                )
            )
        return routes

    def __call__(
        self, tokens: Sequence[Sequence[float]]
    ) -> tuple[list[Vector], list[Route]]:
        routes = self.route(tokens)
        outputs = [[0.0] * self.hidden_size for _ in tokens]

        # A real grouped-GEMM kernel performs the same logical dispatch in bulk.
        for expert_id, expert in enumerate(self.experts):
            expert_routes = [route for route in routes if route.expert == expert_id]
            for route in expert_routes:
                expert_output = expert(tokens[route.token])
                add_scaled_(outputs[route.token], expert_output, route.weight)

        return outputs, routes

    def dense_reference(self, tokens: Sequence[Sequence[float]]) -> list[Vector]:
        """Slow correctness oracle: no grouping, but identical Top-K math."""
        outputs: list[Vector] = []
        for token in tokens:
            logits = linear(token, self.router)
            expert_ids, weights = top_k_gate(logits, self.top_k)
            output = [0.0] * self.hidden_size
            for expert_id, weight in zip(expert_ids, weights):
                add_scaled_(output, self.experts[expert_id](token), weight)
            outputs.append(output)
        return outputs


@dataclass(frozen=True)
class MixtralShape:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 32000
    num_experts: int = 8
    top_k: int = 2


def parameter_ledger(shape: MixtralShape) -> dict[str, int]:
    """Reconstruct Mixtral 8x7B's full and per-token parameter counts."""
    d = shape.hidden_size
    h = shape.intermediate_size
    q_width = shape.num_attention_heads * shape.head_dim
    kv_width = shape.num_key_value_heads * shape.head_dim

    attention_per_layer = d * q_width + 2 * d * kv_width + q_width * d
    expert_per_layer = 3 * d * h  # W1, W2 and W3 in SwiGLU
    router_per_layer = d * shape.num_experts
    norms_per_layer = 2 * d

    shared_outside_blocks = 2 * shape.vocab_size * d + d
    shared_per_layer = attention_per_layer + router_per_layer + norms_per_layer
    full = shared_outside_blocks + shape.num_layers * (
        shared_per_layer + shape.num_experts * expert_per_layer
    )
    active = shared_outside_blocks + shape.num_layers * (
        shared_per_layer + shape.top_k * expert_per_layer
    )
    return {
        "expert_per_layer": expert_per_layer,
        "attention_per_layer": attention_per_layer,
        "full": full,
        "active": active,
    }


def maximum_absolute_error(left: Iterable[Vector], right: Iterable[Vector]) -> float:
    return max(
        abs(a - b)
        for left_row, right_row in zip(left, right)
        for a, b in zip(left_row, right_row)
    )


def main() -> None:
    tokens = [
        [0.2, -0.1, 0.7, 0.3],
        [-0.4, 0.8, 0.1, -0.2],
        [0.9, 0.1, -0.5, 0.4],
    ]
    moe = SparseTopKMoE(
        hidden_size=4,
        intermediate_size=6,
        num_experts=4,
        top_k=2,
    )

    sparse_output, routes = moe(tokens)
    reference_output = moe.dense_reference(tokens)
    error = maximum_absolute_error(sparse_output, reference_output)
    assert error < 1e-12

    print("Top-2 routes (token -> expert: weight):")
    for token_index in range(len(tokens)):
        selected = [route for route in routes if route.token == token_index]
        description = ", ".join(
            f"E{route.expert}: {route.weight:.4f}" for route in selected
        )
        print(f"  token {token_index} -> {description}")
        assert abs(sum(route.weight for route in selected) - 1.0) < 1e-12

    ledger = parameter_ledger(MixtralShape())
    print(f"max sparse-vs-reference error: {error:.3e}")
    print(f"Mixtral full parameters:   {ledger['full'] / 1e9:.3f}B")
    print(f"Mixtral active parameters: {ledger['active'] / 1e9:.3f}B")


if __name__ == "__main__":
    main()
