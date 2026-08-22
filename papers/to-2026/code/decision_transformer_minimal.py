#!/usr/bin/env python3
"""Zero-dependency teaching implementation of Decision Transformer.

The file keeps the pieces that make the paper's formulation distinctive:

    rewards -> returns-to-go -> [R_t, s_t, a_t] token triplets
             -> causal self-attention -> action prediction at state tokens
             -> autoregressive target-return-conditioned rollout

It uses a tiny hand-written one-head Transformer with deterministic matrices,
not a useful RL policy.  The goal is to make token positions, the causal mask,
supervised action loss and return-to-go update inspectable with Python lists.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


Vector = List[float]
Matrix = List[Vector]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector dimensions must match")
    return sum(x * y for x, y in zip(a, b))


def add(a: Sequence[float], b: Sequence[float]) -> Vector:
    if len(a) != len(b):
        raise ValueError("vector dimensions must match")
    return [x + y for x, y in zip(a, b)]


def scale(a: Sequence[float], factor: float) -> Vector:
    return [factor * x for x in a]


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    return [dot(row, vector) for row in matrix]


def softmax(logits: Sequence[float]) -> Vector:
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def returns_to_go(rewards: Sequence[float]) -> Vector:
    """R-hat_t = sum_{t'=t}^{T-1} r_t', without discounting."""
    running = 0.0
    result = [0.0] * len(rewards)
    for index in range(len(rewards) - 1, -1, -1):
        running += rewards[index]
        result[index] = running
    return result


@dataclass(frozen=True)
class Token:
    timestep: int
    kind: str  # "return", "state", or "action"
    value: Vector


def interleave_trajectory(
    returns: Sequence[float],
    states: Sequence[Sequence[float]],
    actions: Sequence[Sequence[float]],
    timesteps: Sequence[int] | None = None,
) -> List[Token]:
    """Build (R_1,s_1,a_1,R_2,s_2,a_2,...) tokens."""
    if not (len(returns) == len(states) == len(actions)):
        raise ValueError("returns, states and actions must have equal length")
    if not returns:
        raise ValueError("trajectory must not be empty")
    if timesteps is None:
        timesteps = list(range(len(returns)))
    if len(timesteps) != len(returns):
        raise ValueError("timesteps must match trajectory length")
    tokens: List[Token] = []
    for timestep, target_return, state, action in zip(
        timesteps, returns, states, actions
    ):
        tokens.extend([
            Token(timestep, "return", [float(target_return)]),
            Token(timestep, "state", list(map(float, state))),
            Token(timestep, "action", list(map(float, action))),
        ])
    return tokens


def causal_attention(
    tokens: Matrix,
    q_projection: Matrix,
    k_projection: Matrix,
    v_projection: Matrix,
) -> Tuple[Matrix, Matrix]:
    """One-head scaled dot-product attention with a strict causal mask."""
    queries = [matvec(q_projection, token) for token in tokens]
    keys = [matvec(k_projection, token) for token in tokens]
    values = [matvec(v_projection, token) for token in tokens]
    key_width = len(keys[0])
    outputs: Matrix = []
    weights: Matrix = []
    for i, query in enumerate(queries):
        logits = [
            dot(query, keys[j]) / math.sqrt(key_width) if j <= i else -math.inf
            for j in range(len(tokens))
        ]
        attention = softmax(logits)
        weights.append(attention)
        outputs.append([
            sum(attention[j] * values[j][channel] for j in range(len(tokens)))
            for channel in range(len(values[0]))
        ])
    return outputs, weights


def layer_norm(vector: Sequence[float], eps: float = 1e-5) -> Vector:
    mean = sum(vector) / len(vector)
    variance = sum((value - mean) ** 2 for value in vector) / len(vector)
    scale_factor = 1.0 / math.sqrt(variance + eps)
    return [(value - mean) * scale_factor for value in vector]


class TinyDecisionTransformer:
    """A deterministic toy model; dimensions are intentionally tiny."""

    def __init__(self, state_dim: int = 2, action_dim: int = 1, hidden: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.return_projection = [
            [0.8],
            [0.0],
            [0.2],
            [0.0],
        ]
        self.state_projection = [
            [0.7, 0.1, 0.0, 0.0],
            [0.0, 0.7, 0.1, 0.0],
        ]
        self.action_projection = [[0.0, 0.0, 0.8, 0.2]]
        self.q_projection = [
            [0.6, 0.0, 0.2, 0.0],
            [0.0, 0.6, 0.0, 0.2],
            [0.2, 0.0, 0.6, 0.0],
            [0.0, 0.2, 0.0, 0.6],
        ]
        self.k_projection = [
            [0.6, 0.0, 0.2, 0.0],
            [0.0, 0.6, 0.0, 0.2],
            [0.2, 0.0, 0.6, 0.0],
            [0.0, 0.2, 0.0, 0.6],
        ]
        self.v_projection = [
            [0.5, 0.0, 0.0, 0.2],
            [0.0, 0.5, 0.2, 0.0],
            [0.0, 0.2, 0.5, 0.0],
            [0.2, 0.0, 0.0, 0.5],
        ]
        self.action_head = [[0.2, -0.1, 0.5, 0.3]]

    def embed(self, token: Token) -> Vector:
        if token.kind == "return":
            raw = token.value
            projected = matvec(self.return_projection, [raw[0]])
            return layer_norm(projected)
        if token.kind == "state":
            raw = token.value
            projected = [raw[0], raw[1], 0.0, 0.0]
            # The projection is a compact illustration, not a parameterized
            # matrix for every possible state dimension.
            return layer_norm([
                0.7 * projected[0] + 0.1 * projected[1],
                0.1 * projected[0] + 0.7 * projected[1],
                0.1 * projected[0],
                0.1 * projected[1],
            ])
        raw = token.value
        return layer_norm([0.0, 0.0, 0.8 * raw[0], 0.2 * raw[0]])

    def forward(self, tokens: Sequence[Token]) -> Tuple[Matrix, Matrix]:
        embedded = [self.embed(token) for token in tokens]
        attended, weights = causal_attention(
            embedded, self.q_projection, self.k_projection, self.v_projection
        )
        hidden = [add(x, y) for x, y in zip(embedded, attended)]
        return hidden, weights

    def action_predictions(self, tokens: Sequence[Token]) -> Matrix:
        hidden, _ = self.forward(tokens)
        return [
            matvec(self.action_head, hidden[index])[0: self.action_dim]
            for index, token in enumerate(tokens)
            if token.kind == "state"
        ]


def action_mse(predictions: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("prediction and target lengths must match")
    errors = [
        (prediction[dimension] - target[dimension]) ** 2
        for prediction, target in zip(predictions, targets)
        for dimension in range(len(target))
    ]
    return sum(errors) / len(errors)


def truncate_context(tokens: Sequence[Token], context_steps: int) -> List[Token]:
    """Keep the last K timesteps = last 3K modality tokens."""
    if context_steps <= 0:
        raise ValueError("context_steps must be positive")
    return list(tokens[-3 * context_steps :])


def rollout_step(
    model: TinyDecisionTransformer,
    target_return: float,
    state: Sequence[float],
    history: Sequence[Token],
    timestep: int,
    context_steps: int,
) -> Tuple[float, List[Token]]:
    """Choose one action, then caller can append the resulting reward/state."""
    prompt = list(history) + [
        Token(timestep, "return", [target_return]),
        Token(timestep, "state", list(map(float, state))),
    ]
    prompt = truncate_context(prompt, context_steps)
    hidden, _ = model.forward(prompt)
    state_indices = [i for i, token in enumerate(prompt) if token.kind == "state"]
    if not state_indices:
        raise ValueError("rollout prompt must contain a state token")
    action = matvec(model.action_head, hidden[state_indices[-1]])[0]
    return action, prompt


def demo() -> None:
    rewards = [1.0, 0.0, 2.0, -1.0]
    states = [[0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.8, 0.2]]
    actions = [[0.1], [0.2], [0.4], [0.0]]
    rtg = returns_to_go(rewards)
    tokens = interleave_trajectory(rtg, states, actions)
    model = TinyDecisionTransformer()
    predictions = model.action_predictions(tokens)
    _, attention = model.forward(tokens)
    print("trajectory length:       4 timesteps")
    print("token length:            3K =", len(tokens))
    print("returns-to-go:           ", [f"{value:.1f}" for value in rtg])
    print("state-token predictions: ", [f"{row[0]:.3f}" for row in predictions])
    print("action MSE:              ", f"{action_mse(predictions, actions):.5f}")
    state_index = 4  # state token at timestep 1
    print("causal weights at token 4:",
          " ".join(f"{value:.2f}" for value in attention[state_index]))
    action, prompt = rollout_step(
        model, target_return=5.0, state=[0.0, 1.0], history=[], timestep=0,
        context_steps=2,
    )
    print("target return at rollout: 5.0")
    print("first generated action: ", f"{action:.3f}")
    print("prompt tokens used:      ", [token.kind for token in prompt])


def assert_close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} != {expected} within {tolerance}")


def run_tests() -> None:
    assert returns_to_go([1.0, 2.0, -1.0]) == [2.0, 1.0, -1.0]
    tokens = interleave_trajectory(
        [3.0, 2.0], [[0.0, 1.0], [1.0, 0.0]], [[0.1], [0.2]]
    )
    assert [token.kind for token in tokens] == [
        "return", "state", "action", "return", "state", "action"
    ]
    model = TinyDecisionTransformer()
    hidden, weights = model.forward(tokens)
    assert len(hidden) == 6
    assert_close(sum(weights[4]), 1.0)
    assert all(value == 0.0 for value in weights[4][5:])
    assert weights[4][4] > 0.0

    predictions = model.action_predictions(tokens)
    assert len(predictions) == 2
    assert_close(action_mse(predictions, [[0.1], [0.2]]),
                 action_mse([[value[0]] for value in predictions], [[0.1], [0.2]]))
    assert len(truncate_context(tokens, 1)) == 3
    assert len(truncate_context(tokens, 2)) == 6

    action, prompt = rollout_step(
        model, target_return=10.0, state=[0.1, 0.9], history=[], timestep=0,
        context_steps=2,
    )
    assert isinstance(action, float)
    assert [token.kind for token in prompt] == ["return", "state"]
    print("all tests passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run invariant checks")
    args = parser.parse_args()
    run_tests() if args.test else demo()


if __name__ == "__main__":
    main()
