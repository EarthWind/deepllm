#!/usr/bin/env python3
"""Zero-dependency teaching implementation of several AlphaFold2 ideas.

This is deliberately *not* a protein predictor.  It keeps four operations
visible with ordinary Python lists:

    MSA -> outer-product mean -> pair representation
    pair edges -> outgoing/incoming triangle multiplication
    residue frames + pair bias -> invariant point-attention weights
    local-frame comparison -> frame-aligned point error (FAPE)

The learned projections, gates and multi-head dimensions are collapsed in the
tiny examples.  The index contractions and rigid-motion invariants are the
parts worth tracing here.  No network access or third-party package is needed.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


Vector = List[float]
Matrix = List[Vector]
Tensor3 = List[Matrix]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector dimensions must match")
    return sum(x * y for x, y in zip(a, b))


def add(a: Sequence[float], b: Sequence[float]) -> Vector:
    if len(a) != len(b):
        raise ValueError("vector dimensions must match")
    return [x + y for x, y in zip(a, b)]


def sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    if len(a) != len(b):
        raise ValueError("vector dimensions must match")
    return [x - y for x, y in zip(a, b)]


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    return [dot(row, vector) for row in matrix]


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    return [list(column) for column in zip(*matrix)]


def softmax(values: Sequence[float]) -> Vector:
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def outer_product_mean(msa: Tensor3) -> Tensor3:
    """Unprojected OPM: [S,R,C] -> [R,R,C*C].

    AlphaFold2 first layer-normalizes and projects each MSA cell into two
    channel vectors.  It averages their outer products over sequences, then
    linearly projects C*C channels into the pair width.  Here the two learned
    projections are identities so the evolutionary contraction is explicit.
    """
    sequences = len(msa)
    residues = len(msa[0])
    channels = len(msa[0][0])
    if sequences == 0 or residues == 0 or channels == 0:
        raise ValueError("MSA dimensions must be non-empty")
    if any(len(row) != residues for row in msa):
        raise ValueError("all MSA rows must have the same residue count")

    pair = [[[0.0] * (channels * channels) for _ in range(residues)]
            for _ in range(residues)]
    for i in range(residues):
        for j in range(residues):
            for sequence in range(sequences):
                left = msa[sequence][i]
                right = msa[sequence][j]
                for a in range(channels):
                    for b in range(channels):
                        pair[i][j][a * channels + b] += left[a] * right[b]
            pair[i][j] = [value / sequences for value in pair[i][j]]
    return pair


def triangle_multiplication_outgoing(pair: Matrix) -> Matrix:
    """Scalar-channel contraction sum_k z[i,k] * z[j,k]."""
    residues = len(pair)
    return [
        [sum(pair[i][k] * pair[j][k] for k in range(residues))
         for j in range(residues)]
        for i in range(residues)
    ]


def triangle_multiplication_incoming(pair: Matrix) -> Matrix:
    """Scalar-channel contraction sum_k z[k,i] * z[k,j]."""
    residues = len(pair)
    return [
        [sum(pair[k][i] * pair[k][j] for k in range(residues))
         for j in range(residues)]
        for i in range(residues)
    ]


def msa_row_attention_with_pair_bias(
    row: Matrix,
    pair_bias: Matrix,
) -> Tuple[Matrix, Matrix]:
    """Single-head row attention over residues with z[i,j] as logit bias."""
    residues = len(row)
    width = len(row[0])
    weights: Matrix = []
    outputs: Matrix = []
    for i in range(residues):
        logits = [dot(row[i], row[j]) / math.sqrt(width) + pair_bias[i][j]
                  for j in range(residues)]
        attention = softmax(logits)
        weights.append(attention)
        outputs.append([
            sum(attention[j] * row[j][channel] for j in range(residues))
            for channel in range(width)
        ])
    return outputs, weights


@dataclass(frozen=True)
class Frame:
    """Rigid transform x_global = rotation @ x_local + translation."""

    rotation: Matrix
    translation: Vector

    def apply(self, local_point: Sequence[float]) -> Vector:
        return add(matvec(self.rotation, local_point), self.translation)

    def invert(self, global_point: Sequence[float]) -> Vector:
        return matvec(transpose(self.rotation), sub(global_point, self.translation))

    def left_compose(self, global_frame: "Frame") -> "Frame":
        rotation = [
            [sum(global_frame.rotation[i][k] * self.rotation[k][j]
                 for k in range(3)) for j in range(3)]
            for i in range(3)
        ]
        translation = global_frame.apply(self.translation)
        return Frame(rotation, translation)


def ipa_attention_weights(
    scalar_queries: Matrix,
    scalar_keys: Matrix,
    local_query_points: Matrix,
    local_key_points: Matrix,
    frames: Sequence[Frame],
    pair_bias: Matrix,
    point_weight: float = 1.0,
) -> Matrix:
    """One-head, one-point version of invariant point attention.

    The paper balances scalar, pair and point terms with fixed constants and
    learned positive per-head weights.  This compact form retains the defining
    fact: distances are measured after local points enter the global frame.
    """
    residues = len(frames)
    width = len(scalar_queries[0])
    global_queries = [frames[i].apply(local_query_points[i])
                      for i in range(residues)]
    global_keys = [frames[i].apply(local_key_points[i])
                   for i in range(residues)]
    result = []
    for i in range(residues):
        logits = []
        for j in range(residues):
            scalar = dot(scalar_queries[i], scalar_keys[j]) / math.sqrt(width)
            squared_distance = dot(
                sub(global_queries[i], global_keys[j]),
                sub(global_queries[i], global_keys[j]),
            )
            logits.append(scalar + pair_bias[i][j]
                          - 0.5 * point_weight * squared_distance)
        result.append(softmax(logits))
    return result


def fape(
    predicted_frames: Sequence[Frame],
    true_frames: Sequence[Frame],
    predicted_points: Matrix,
    true_points: Matrix,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
) -> float:
    """Average clamped error after observing every point from every frame."""
    if len(predicted_frames) != len(true_frames):
        raise ValueError("predicted and true frame counts must match")
    if len(predicted_points) != len(true_points):
        raise ValueError("predicted and true point counts must match")
    errors = []
    for predicted_frame, true_frame in zip(predicted_frames, true_frames):
        for predicted_point, true_point in zip(predicted_points, true_points):
            predicted_local = predicted_frame.invert(predicted_point)
            true_local = true_frame.invert(true_point)
            errors.append(min(clamp_distance, norm(sub(predicted_local, true_local)))
                          / length_scale)
    return sum(errors) / len(errors)


def rotation_z(angle_degrees: float) -> Matrix:
    angle = math.radians(angle_degrees)
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def identity_frame(x: float, y: float, z: float) -> Frame:
    return Frame(rotation_z(0.0), [x, y, z])


def max_abs_difference(left: Matrix, right: Matrix) -> float:
    return max(abs(a - b) for row_a, row_b in zip(left, right)
               for a, b in zip(row_a, row_b))


def demo() -> None:
    msa = [
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        [[0.9, 0.1], [0.7, 0.3], [0.1, 0.9]],
        [[0.2, 0.8], [0.3, 0.7], [0.9, 0.1]],
    ]
    pair_features = outer_product_mean(msa)
    pair_scalar = [[cell[0] + cell[3] for cell in row]
                   for row in pair_features]
    outgoing = triangle_multiplication_outgoing(pair_scalar)
    incoming = triangle_multiplication_incoming(pair_scalar)

    print("MSA shape:                 [3 sequences, 3 residues, 2 channels]")
    print("outer-product mean shape:  [3, 3, 4]")
    print("pair scalar (trace of OPM):")
    for row in pair_scalar:
        print(" ", " ".join(f"{value:5.2f}" for value in row))
    print(f"triangle outgoing z[0,2]: {outgoing[0][2]:.4f}")
    print(f"triangle incoming z[0,2]: {incoming[0][2]:.4f}")

    frames = [identity_frame(0.0, 0.0, 0.0),
              identity_frame(2.0, 0.0, 0.0),
              identity_frame(5.0, 0.0, 0.0)]
    q = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]
    k = [[0.9, 0.1], [0.7, 0.3], [0.1, 0.9]]
    local_points = [[0.0, 0.0, 0.0]] * 3
    zero_bias = [[0.0] * 3 for _ in range(3)]
    weights = ipa_attention_weights(
        q, k, local_points, local_points, frames, zero_bias, point_weight=0.3
    )
    print("IPA weights from residue 0:",
          " ".join(f"{value:.3f}" for value in weights[0]))

    global_motion = Frame(rotation_z(71.0), [10.0, -4.0, 2.5])
    moved_frames = [frame.left_compose(global_motion) for frame in frames]
    moved_weights = ipa_attention_weights(
        q, k, local_points, local_points, moved_frames, zero_bias,
        point_weight=0.3,
    )
    print("max IPA change after global rigid motion:",
          f"{max_abs_difference(weights, moved_weights):.3e}")

    true_points = [frame.translation for frame in frames]
    shifted_points = [[x + 0.4, y - 0.2, z] for x, y, z in true_points]
    print("FAPE for a small coordinate error:",
          f"{fape(frames, frames, shifted_points, true_points):.4f}")


def assert_close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} != {expected} within {tolerance}")


def run_tests() -> None:
    msa = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 1.0], [4.0, 3.0]],
    ]
    pair = outer_product_mean(msa)
    assert pair[0][1] == [5.5, 5.0, 5.0, 5.5]

    edges = [[1.0, 2.0], [3.0, 4.0]]
    assert triangle_multiplication_outgoing(edges) == [[5.0, 11.0], [11.0, 25.0]]
    assert triangle_multiplication_incoming(edges) == [[10.0, 14.0], [14.0, 20.0]]

    _, attention = msa_row_attention_with_pair_bias(
        [[1.0, 0.0], [0.0, 1.0]], [[0.0, 4.0], [0.0, 0.0]]
    )
    assert attention[0][1] > attention[0][0]
    assert_close(sum(attention[0]), 1.0)

    frames = [identity_frame(0.0, 0.0, 0.0), identity_frame(3.0, 1.0, 0.0)]
    scalar = [[1.0, 0.0], [0.0, 1.0]]
    points = [[0.2, 0.0, 0.0], [-0.1, 0.3, 0.0]]
    bias = [[0.0, 0.2], [-0.1, 0.0]]
    before = ipa_attention_weights(scalar, scalar, points, points, frames, bias)
    motion = Frame(rotation_z(123.0), [-7.0, 2.0, 9.0])
    after = ipa_attention_weights(
        scalar, scalar, points, points,
        [frame.left_compose(motion) for frame in frames], bias,
    )
    assert max_abs_difference(before, after) < 1e-12

    true_points = [[0.0, 0.0, 0.0], [3.0, 1.0, 0.0]]
    assert_close(fape(frames, frames, true_points, true_points), 0.0)
    moved_frames = [frame.left_compose(motion) for frame in frames]
    moved_points = [motion.apply(point) for point in true_points]
    assert_close(fape(moved_frames, moved_frames, moved_points, moved_points), 0.0)

    print("all tests passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run invariant checks")
    args = parser.parse_args()
    run_tests() if args.test else demo()


if __name__ == "__main__":
    main()
