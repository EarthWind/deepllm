#!/usr/bin/env python3
"""Zero-dependency reference code for the core ideas in Mamba (2023).

This file is deliberately small and slow. It is NOT the official CUDA kernel and
not a full language model. It makes four paper-level mechanisms executable:

1. zero-order-hold (ZOH) discretization of a diagonal continuous-time SSM;
2. input-dependent delta, B and C while A remains a learned fixed parameter;
3. recurrent selective scan with a constant-size state;
4. an associative Blelloch prefix scan that matches the sequential recurrence.

Run:
    python3 papers/to-2026/code/mamba_selective_scan_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, expm1, log1p, tanh
from typing import Sequence


Vector = tuple[float, ...]


def softplus(value: float) -> float:
    """Numerically stable log(1 + exp(value))."""

    if value > 20.0:
        return value
    if value < -20.0:
        return exp(value)
    return log1p(exp(value))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = exp(-value)
        return 1.0 / (1.0 + inverse)
    positive = exp(value)
    return positive / (1.0 + positive)


def zoh_discretize(a: float, b: float, delta: float) -> tuple[float, float]:
    """Discretize dh/dt = a*h + b*x with a zero-order-held input.

    h_t = a_bar * h_{t-1} + b_bar * x_t
    a_bar = exp(delta * a)
    b_bar = integral_0^delta exp((delta-s)*a) ds * b

    expm1 is used because exp(delta*a)-1 loses precision near zero.
    """

    a_bar = exp(delta * a)
    if abs(a) < 1e-12:
        b_bar = delta * b
    else:
        b_bar = expm1(delta * a) / a * b
    return a_bar, b_bar


@dataclass(frozen=True)
class Affine:
    """Elementwise affine state map h -> a*h + b."""

    a: Vector
    b: Vector


def identity(width: int) -> Affine:
    return Affine((1.0,) * width, (0.0,) * width)


def compose(left: Affine, right: Affine) -> Affine:
    """Return right(left(h)); sequence order is left, then right.

    If left(h) = a_l*h+b_l and right(h) = a_r*h+b_r, then
    right(left(h)) = (a_r*a_l)h + (a_r*b_l+b_r).
    This operator is associative, which enables a parallel prefix scan.
    """

    if len(left.a) != len(right.a):
        raise ValueError("affine maps must have the same state width")
    return Affine(
        a=tuple(a_r * a_l for a_l, a_r in zip(left.a, right.a)),
        b=tuple(a_r * b_l + b_r for b_l, a_r, b_r in zip(left.b, right.a, right.b)),
    )


def blelloch_inclusive_scan(items: Sequence[Affine]) -> list[Affine]:
    """Work-efficient associative prefix scan; pads to a power of two.

    A GPU kernel uses a more sophisticated hierarchy and tensor layout. This
    function only demonstrates why a recurrence can be parallelized in training.
    """

    if not items:
        return []
    width = len(items[0].a)
    size = 1
    while size < len(items):
        size *= 2
    original = list(items) + [identity(width) for _ in range(size - len(items))]
    work = original.copy()

    # Up-sweep: reduce each tree node to the affine map for its whole segment.
    stride = 2
    while stride <= size:
        for right in range(stride - 1, size, stride):
            left = right - stride // 2
            work[right] = compose(work[left], work[right])
        stride *= 2

    # Down-sweep: turn segment totals into exclusive prefixes.
    work[-1] = identity(width)
    stride = size
    while stride >= 2:
        for right in range(stride - 1, size, stride):
            left = right - stride // 2
            left_total = work[left]
            parent_prefix = work[right]
            work[left] = parent_prefix
            work[right] = compose(parent_prefix, left_total)
        stride //= 2

    # Convert exclusive prefixes to inclusive prefixes.
    return [compose(work[index], original[index]) for index in range(len(items))]


@dataclass(frozen=True)
class SelectiveParameters:
    """Parameters generated for one token.

    delta has D values; B and C each have N values, matching Algorithm 2's
    broadcast structure. A is intentionally absent: it is fixed across tokens.
    """

    delta: Vector
    b: Vector
    c: Vector


def token_parameters(x: Vector, state_size: int) -> SelectiveParameters:
    """A deterministic toy replacement for Mamba's learned linear projections."""

    mean_x = sum(x) / len(x)
    delta = tuple(softplus(-1.8 + 1.25 * value) for value in x)
    b = tuple(0.55 + 0.25 * tanh(mean_x + 0.37 * index) for index in range(state_size))
    c = tuple(0.45 + 0.30 * tanh(mean_x - 0.29 * index) for index in range(state_size))
    return SelectiveParameters(delta=delta, b=b, c=c)


def build_selective_transforms(
    sequence: Sequence[Vector],
    *,
    a: Vector,
    state_size: int,
) -> tuple[list[Affine], list[SelectiveParameters]]:
    """Create token-dependent discretized recurrence maps.

    State layout is [channel 0's N states, channel 1's N states, ...].
    A has shape (D, N) and is fixed. Delta_t, B_t and C_t depend on x_t.
    """

    if not sequence:
        return [], []
    channels = len(sequence[0])
    if len(a) != channels * state_size:
        raise ValueError("A must contain channels * state_size values")

    transforms: list[Affine] = []
    generated: list[SelectiveParameters] = []
    for x in sequence:
        if len(x) != channels:
            raise ValueError("all tokens must have the same channel width")
        params = token_parameters(x, state_size)
        generated.append(params)
        a_bar: list[float] = []
        input_term: list[float] = []
        for channel, value in enumerate(x):
            for state_index in range(state_size):
                flat = channel * state_size + state_index
                transition, input_gain = zoh_discretize(
                    a[flat], params.b[state_index], params.delta[channel]
                )
                a_bar.append(transition)
                input_term.append(input_gain * value)
        transforms.append(Affine(tuple(a_bar), tuple(input_term)))
    return transforms, generated


def sequential_states(transforms: Sequence[Affine]) -> list[Vector]:
    """Readable O(L) recurrence used as the correctness oracle."""

    if not transforms:
        return []
    state = (0.0,) * len(transforms[0].a)
    states: list[Vector] = []
    for transform in transforms:
        state = tuple(a * value + b for a, value, b in zip(transform.a, state, transform.b))
        states.append(state)
    return states


def read_outputs(
    sequence: Sequence[Vector],
    states: Sequence[Vector],
    params: Sequence[SelectiveParameters],
    *,
    state_size: int,
    skip: float = 0.1,
) -> list[Vector]:
    """Apply selective C_t readout plus a small direct D*x_t skip."""

    outputs: list[Vector] = []
    for x, state, token_params in zip(sequence, states, params):
        channels = len(x)
        output = []
        for channel in range(channels):
            offset = channel * state_size
            readout = sum(
                token_params.c[index] * state[offset + index]
                for index in range(state_size)
            )
            output.append(readout + skip * x[channel])
        outputs.append(tuple(output))
    return outputs


def gate_demo(logit: float, previous: float, current: float) -> tuple[float, float]:
    """The N=1, A=-1, B=1 special case from the paper's Theorem 1."""

    delta = softplus(logit)
    a_bar, b_bar = zoh_discretize(-1.0, 1.0, delta)
    updated = a_bar * previous + b_bar * current
    gate = sigmoid(logit)
    expected = (1.0 - gate) * previous + gate * current
    if abs(updated - expected) > 1e-12:
        raise AssertionError("ZOH gate identity failed")
    return gate, updated


def _demo() -> None:
    sequence = [
        (0.2, -0.4),
        (1.1, 0.3),
        (-0.7, 0.8),
        (0.0, -1.2),
        (0.9, 0.5),
        (-0.2, 0.1),
        (0.6, -0.9),
    ]
    channels = len(sequence[0])
    state_size = 3
    # Stable, diagonal continuous-time A. It does not vary with the token.
    a = tuple(-0.4 - 0.15 * index for index in range(channels * state_size))

    transforms, params = build_selective_transforms(sequence, a=a, state_size=state_size)
    recurrent = sequential_states(transforms)
    scanned = [prefix.b for prefix in blelloch_inclusive_scan(transforms)]
    max_scan_error = max(
        abs(left - right)
        for state_a, state_b in zip(recurrent, scanned)
        for left, right in zip(state_a, state_b)
    )
    assert max_scan_error < 1e-12

    outputs = read_outputs(sequence, scanned, params, state_size=state_size)
    assert len(outputs) == len(sequence) and len(outputs[0]) == channels
    print(f"sequential vs parallel scan max error: {max_scan_error:.3e}")
    print(f"state scalars: D*N = {channels}*{state_size} = {channels * state_size}")
    print(f"sequence length: {len(sequence)}; final state size is still {len(scanned[-1])}")

    ignore_gate, ignore_value = gate_demo(-8.0, previous=7.0, current=100.0)
    select_gate, select_value = gate_demo(8.0, previous=7.0, current=100.0)
    print(f"small delta / ignore: g={ignore_gate:.6f}, state={ignore_value:.4f}")
    print(f"large delta / select: g={select_gate:.6f}, state={select_value:.4f}")
    assert abs(ignore_value - 7.0) < 0.04
    assert abs(select_value - 100.0) < 0.04
    print("all checks passed")


if __name__ == "__main__":
    _demo()
