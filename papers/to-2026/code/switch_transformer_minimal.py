"""Dependency-free reference implementation of Switch Transformer routing.

The goal is to expose the algorithm, not to provide a fast MoE kernel.  This
small script implements the parts that are easy to hide behind framework APIs:

* float softmax routing and top-1 expert selection;
* fixed expert capacity and overflow-token dropping;
* the differentiable load-balancing objective's forward value;
* per-expert dispatch, gated expert output, and residual bypass; and
* routing statistics that are useful when debugging a real implementation.

It deliberately omits autograd and distributed all-to-all communication.  In a
real PyTorch/JAX implementation those mechanisms replace the Python loops, but
the routing decisions and invariants stay the same.

Run:
    python3 papers/to-2026/code/switch_transformer_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


Vector = list[float]
Matrix = list[Vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("dot-product vectors must have the same length")
    return sum(a * b for a, b in zip(left, right))


def _softmax(logits: Sequence[float]) -> Vector:
    if not logits:
        raise ValueError("softmax needs at least one logit")
    maximum = max(logits)
    unnormalized = [math.exp(value - maximum) for value in logits]
    denominator = sum(unnormalized)
    return [value / denominator for value in unnormalized]


def _add(left: Sequence[float], right: Sequence[float]) -> Vector:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    return [a + b for a, b in zip(left, right)]


def _scaled(vector: Sequence[float], scale: float) -> Vector:
    return [scale * value for value in vector]


def _random_matrix(
    rows: int,
    columns: int,
    rng: random.Random,
    *,
    scale: float,
) -> Matrix:
    return [
        [rng.uniform(-scale, scale) for _ in range(columns)]
        for _ in range(rows)
    ]


class FeedForwardExpert:
    """A tiny two-layer GELU FFN with expert-specific parameters."""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        rng: random.Random,
    ) -> None:
        # The paper uses a truncated normal with std=sqrt(s / fan_in) and
        # recommends s=0.1 instead of 1.0.  This demo uses a bounded uniform
        # distribution at the same rough magnitude only to keep runs stable.
        scale = math.sqrt(0.1 / model_dim)
        self.up = _random_matrix(hidden_dim, model_dim, rng, scale=scale)
        self.down = _random_matrix(model_dim, hidden_dim, rng, scale=scale)

    def __call__(self, token: Sequence[float]) -> Vector:
        preactivation = [_dot(row, token) for row in self.up]
        hidden = [
            0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))
            for value in preactivation
        ]
        return [_dot(row, hidden) for row in self.down]


@dataclass(frozen=True)
class RoutingResult:
    probabilities: Matrix
    expert_indices: list[int]
    gates: Vector
    positions: list[int]
    kept: list[bool]
    capacity: int
    selected_load: list[int]
    accepted_load: list[int]
    dropped_tokens: int
    auxiliary_loss: float


def load_balancing_loss(
    probabilities: Matrix,
    expert_indices: Sequence[int],
    *,
    alpha: float = 1e-2,
) -> float:
    """Compute alpha * N * sum_i(f_i * P_i) from the paper.

    ``f_i`` is the hard fraction of tokens selecting expert i and ``P_i`` is
    the mean router probability assigned to expert i.  During training only
    the probability term carries gradients; this dependency-free script only
    evaluates the scalar forward value.
    """

    if not probabilities:
        raise ValueError("probabilities must contain at least one token")
    token_count = len(probabilities)
    expert_count = len(probabilities[0])
    if expert_count == 0:
        raise ValueError("at least one expert is required")
    if len(expert_indices) != token_count:
        raise ValueError("one selected expert is required per token")
    if any(len(row) != expert_count for row in probabilities):
        raise ValueError("router probability rows must be rectangular")

    hard_fraction = [0.0 for _ in range(expert_count)]
    probability_fraction = [0.0 for _ in range(expert_count)]
    for token_probabilities, expert_index in zip(
        probabilities, expert_indices
    ):
        hard_fraction[expert_index] += 1.0 / token_count
        for expert_id, probability in enumerate(token_probabilities):
            probability_fraction[expert_id] += probability / token_count

    return alpha * expert_count * sum(
        hard * soft
        for hard, soft in zip(hard_fraction, probability_fraction)
    )


class Top1Router:
    def __init__(
        self,
        model_dim: int,
        expert_count: int,
        rng: random.Random,
    ) -> None:
        if model_dim <= 0 or expert_count <= 0:
            raise ValueError("model_dim and expert_count must be positive")
        self.expert_count = expert_count
        self.weights = _random_matrix(
            expert_count,
            model_dim,
            rng,
            scale=1.0 / math.sqrt(model_dim),
        )

    def __call__(
        self,
        tokens: Matrix,
        *,
        capacity_factor: float,
        alpha: float = 1e-2,
    ) -> RoutingResult:
        if not tokens:
            raise ValueError("the router needs at least one token")
        if capacity_factor <= 0.0:
            raise ValueError("capacity_factor must be positive")

        # A production mixed-precision model performs these logits and the
        # softmax in float32, then casts dispatch/combine tensors back down.
        probabilities = [
            _softmax([_dot(row, token) for row in self.weights])
            for token in tokens
        ]
        expert_indices = [
            max(range(self.expert_count), key=row.__getitem__)
            for row in probabilities
        ]
        gates = [
            row[expert_index]
            for row, expert_index in zip(probabilities, expert_indices)
        ]

        capacity = max(
            1,
            math.ceil(
                capacity_factor * len(tokens) / self.expert_count
            ),
        )
        selected_load = [0 for _ in range(self.expert_count)]
        accepted_load = [0 for _ in range(self.expert_count)]
        positions: list[int] = []
        kept: list[bool] = []

        # The cumsum-based position calculation in tensor implementations is
        # equivalent to these per-expert counters.  Earlier tokens get the
        # available slots; later overflow tokens are dropped for this branch.
        for expert_index in expert_indices:
            position = selected_load[expert_index]
            selected_load[expert_index] += 1
            is_kept = position < capacity
            positions.append(position)
            kept.append(is_kept)
            if is_kept:
                accepted_load[expert_index] += 1

        auxiliary_loss = load_balancing_loss(
            probabilities,
            expert_indices,
            alpha=alpha,
        )
        return RoutingResult(
            probabilities=probabilities,
            expert_indices=expert_indices,
            gates=gates,
            positions=positions,
            kept=kept,
            capacity=capacity,
            selected_load=selected_load,
            accepted_load=accepted_load,
            dropped_tokens=kept.count(False),
            auxiliary_loss=auxiliary_loss,
        )


class SwitchFeedForward:
    """Top-1 routed FFN branch; residual addition lives in ``SwitchBlock``."""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        expert_count: int,
        *,
        seed: int = 0,
    ) -> None:
        rng = random.Random(seed)
        self.model_dim = model_dim
        self.router = Top1Router(model_dim, expert_count, rng)
        self.experts = [
            FeedForwardExpert(model_dim, hidden_dim, rng)
            for _ in range(expert_count)
        ]

    def __call__(
        self,
        tokens: Matrix,
        *,
        capacity_factor: float = 1.0,
        alpha: float = 1e-2,
    ) -> tuple[Matrix, RoutingResult]:
        if any(len(token) != self.model_dim for token in tokens):
            raise ValueError("every token must have model_dim features")
        routing = self.router(
            tokens,
            capacity_factor=capacity_factor,
            alpha=alpha,
        )
        branch_output = [
            [0.0 for _ in range(self.model_dim)] for _ in tokens
        ]

        for token_id, token in enumerate(tokens):
            if not routing.kept[token_id]:
                # Zero means "skip this FFN branch".  The enclosing block's
                # residual connection still forwards the token representation.
                continue
            expert_id = routing.expert_indices[token_id]
            expert_output = self.experts[expert_id](token)
            branch_output[token_id] = _scaled(
                expert_output,
                routing.gates[token_id],
            )
        return branch_output, routing


class SwitchBlock:
    """Minimal residual wrapper around the sparse FFN branch."""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        expert_count: int,
        *,
        seed: int = 0,
    ) -> None:
        self.switch_ffn = SwitchFeedForward(
            model_dim,
            hidden_dim,
            expert_count,
            seed=seed,
        )

    def __call__(
        self,
        tokens: Matrix,
        *,
        capacity_factor: float = 1.0,
    ) -> tuple[Matrix, RoutingResult]:
        branch, routing = self.switch_ffn(
            tokens,
            capacity_factor=capacity_factor,
        )
        return [
            _add(token, update) for token, update in zip(tokens, branch)
        ], routing


def _demo() -> None:
    rng = random.Random(7)
    tokens = [
        [rng.uniform(-1.0, 1.0) for _ in range(8)]
        for _ in range(17)
    ]
    block = SwitchBlock(
        model_dim=8,
        hidden_dim=16,
        expert_count=4,
        seed=11,
    )
    output, routing = block(tokens, capacity_factor=1.0)

    assert len(output) == len(tokens)
    assert all(len(row) == 8 for row in output)
    assert sum(routing.selected_load) == len(tokens)
    assert all(
        load <= routing.capacity for load in routing.accepted_load
    )
    assert (
        sum(routing.accepted_load) + routing.dropped_tokens
        == len(tokens)
    )
    for token, result, is_kept in zip(tokens, output, routing.kept):
        if not is_kept:
            # Overflow token skipped the expert but survived via residual.
            assert result == token

    print(f"tokens:           {len(tokens)}")
    print(f"experts:          {len(routing.selected_load)}")
    print(f"expert capacity:  {routing.capacity}")
    print(f"selected load:    {routing.selected_load}")
    print(f"accepted load:    {routing.accepted_load}")
    print(f"dropped tokens:   {routing.dropped_tokens}")
    print(f"auxiliary loss:   {routing.auxiliary_loss:.6f}")


if __name__ == "__main__":
    _demo()
