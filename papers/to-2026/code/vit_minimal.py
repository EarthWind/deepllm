#!/usr/bin/env python3
"""A zero-dependency, forward-only Vision Transformer teaching implementation.

This file keeps the essential ViT data path visible:

    image -> non-overlapping patches -> linear embeddings
          -> [CLS] + learned positions -> Pre-LN Transformer
          -> classification logits

It intentionally omits batching, gradients, dropout, data loading and training.
The tiny dimensions make every operation inspectable with Python lists.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple


Vector = List[float]
Matrix = List[Vector]
Image = List[List[List[float]]]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """Multiply [m, k] by [k, n]."""
    assert a and b and len(a[0]) == len(b)
    return [
        [sum(row[k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for row in a
    ]


def add_rows(a: Matrix, b: Matrix) -> Matrix:
    assert len(a) == len(b) and len(a[0]) == len(b[0])
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def linear(x: Matrix, weight: Matrix, bias: Sequence[float] | None = None) -> Matrix:
    y = matmul(x, weight)
    if bias is not None:
        y = [[value + bias[j] for j, value in enumerate(row)] for row in y]
    return y


def softmax(row: Sequence[float]) -> Vector:
    peak = max(row)
    exps = [math.exp(value - peak) for value in row]
    total = sum(exps)
    return [value / total for value in exps]


def layer_norm(tokens: Matrix, eps: float = 1e-6) -> Matrix:
    output = []
    for token in tokens:
        mean = sum(token) / len(token)
        variance = sum((value - mean) ** 2 for value in token) / len(token)
        scale = 1.0 / math.sqrt(variance + eps)
        output.append([(value - mean) * scale for value in token])
    return output


def gelu(value: float) -> float:
    """The tanh approximation used in many Transformer implementations."""
    factor = math.sqrt(2.0 / math.pi)
    return 0.5 * value * (1.0 + math.tanh(factor * (value + 0.044715 * value**3)))


def patchify(image: Image, patch_size: int) -> Matrix:
    """Raster-scan an H x W x C image into N rows of P*P*C values."""
    height, width, channels = len(image), len(image[0]), len(image[0][0])
    if height % patch_size or width % patch_size:
        raise ValueError("height and width must be divisible by patch_size")

    patches: Matrix = []
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            patch = []
            for row in range(top, top + patch_size):
                for col in range(left, left + patch_size):
                    patch.extend(image[row][col][:channels])
            patches.append(patch)
    return patches


def conv_patch_embed(image: Image, patch_size: int, weight: Matrix, bias: Vector) -> Matrix:
    """Conv(kernel=P, stride=P) written as loops, to show its exact equivalence."""
    height, width, channels = len(image), len(image[0]), len(image[0][0])
    outputs: Matrix = []
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            output = bias[:]
            flattened_index = 0
            for row in range(top, top + patch_size):
                for col in range(left, left + patch_size):
                    for channel in range(channels):
                        pixel = image[row][col][channel]
                        for out_channel in range(len(output)):
                            output[out_channel] += pixel * weight[flattened_index][out_channel]
                        flattened_index += 1
            outputs.append(output)
    return outputs


def scaled_dot_product_attention(q: Matrix, k: Matrix, v: Matrix) -> Tuple[Matrix, Matrix]:
    scale = math.sqrt(len(q[0]))
    scores = [
        [sum(qi[d] * kj[d] for d in range(len(qi))) / scale for kj in k]
        for qi in q
    ]
    weights = [softmax(row) for row in scores]
    return matmul(weights, v), weights


def multi_head_attention(
    tokens: Matrix,
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    num_heads: int,
) -> Tuple[Matrix, List[Matrix]]:
    """Standard global MSA. Each returned attention map has shape T x T."""
    q, k, v = linear(tokens, wq), linear(tokens, wk), linear(tokens, wv)
    hidden = len(tokens[0])
    if hidden % num_heads:
        raise ValueError("hidden size must be divisible by num_heads")
    head_dim = hidden // num_heads

    per_head_outputs: List[Matrix] = []
    attention_maps: List[Matrix] = []
    for head in range(num_heads):
        start, end = head * head_dim, (head + 1) * head_dim
        head_q = [row[start:end] for row in q]
        head_k = [row[start:end] for row in k]
        head_v = [row[start:end] for row in v]
        head_output, head_weights = scaled_dot_product_attention(head_q, head_k, head_v)
        per_head_outputs.append(head_output)
        attention_maps.append(head_weights)

    concatenated = [
        [value for head in per_head_outputs for value in head[token_index]]
        for token_index in range(len(tokens))
    ]
    return linear(concatenated, wo), attention_maps


def bilinear_resize_position_grid(grid: Matrix, old_side: int, new_side: int) -> Matrix:
    """Resize patch positions; keep the separate [CLS] position outside this function."""
    if len(grid) != old_side * old_side:
        raise ValueError("grid length does not match old_side")
    if old_side == new_side:
        return [row[:] for row in grid]

    hidden = len(grid[0])
    resized: Matrix = []
    for new_y in range(new_side):
        source_y = new_y * (old_side - 1) / (new_side - 1)
        y0, y1 = math.floor(source_y), math.ceil(source_y)
        wy = source_y - y0
        for new_x in range(new_side):
            source_x = new_x * (old_side - 1) / (new_side - 1)
            x0, x1 = math.floor(source_x), math.ceil(source_x)
            wx = source_x - x0
            vector = []
            for d in range(hidden):
                top = grid[y0 * old_side + x0][d] * (1 - wx) + grid[y0 * old_side + x1][d] * wx
                bottom = grid[y1 * old_side + x0][d] * (1 - wx) + grid[y1 * old_side + x1][d] * wx
                vector.append(top * (1 - wy) + bottom * wy)
            resized.append(vector)
    return resized


@dataclass
class EncoderWeights:
    wq: Matrix
    wk: Matrix
    wv: Matrix
    wo: Matrix
    w1: Matrix
    b1: Vector
    w2: Matrix
    b2: Vector


class TinyVisionTransformer:
    """A deterministic ViT whose purpose is tracing shapes, not accuracy."""

    def __init__(
        self,
        image_size: int = 4,
        patch_size: int = 2,
        channels: int = 1,
        hidden: int = 8,
        mlp_hidden: int = 16,
        heads: int = 2,
        layers: int = 2,
        classes: int = 3,
        seed: int = 7,
    ) -> None:
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.hidden = hidden
        self.heads = heads
        self.num_patches = (image_size // patch_size) ** 2
        rng = random.Random(seed)

        def weights(rows: int, cols: int, scale: float = 0.18) -> Matrix:
            return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

        self.patch_weight = weights(patch_size * patch_size * channels, hidden)
        self.patch_bias = [0.0] * hidden
        self.class_token = weights(1, hidden)[0]
        self.positions = weights(self.num_patches + 1, hidden, scale=0.08)
        self.blocks = []
        for _ in range(layers):
            self.blocks.append(
                EncoderWeights(
                    wq=weights(hidden, hidden),
                    wk=weights(hidden, hidden),
                    wv=weights(hidden, hidden),
                    wo=weights(hidden, hidden),
                    w1=weights(hidden, mlp_hidden),
                    b1=[0.0] * mlp_hidden,
                    w2=weights(mlp_hidden, hidden),
                    b2=[0.0] * hidden,
                )
            )
        self.head_weight = weights(hidden, classes)
        self.head_bias = [0.0] * classes

    def encoder_block(self, tokens: Matrix, block: EncoderWeights) -> Tuple[Matrix, List[Matrix]]:
        # Pre-LN attention, then residual connection: z' = MSA(LN(z)) + z.
        attention_output, maps = multi_head_attention(
            layer_norm(tokens), block.wq, block.wk, block.wv, block.wo, self.heads
        )
        tokens = add_rows(tokens, attention_output)

        # Pre-LN MLP, then residual connection: z = MLP(LN(z')) + z'.
        mlp = linear(layer_norm(tokens), block.w1, block.b1)
        mlp = [[gelu(value) for value in row] for row in mlp]
        tokens = add_rows(tokens, linear(mlp, block.w2, block.b2))
        return tokens, maps

    def forward(self, image: Image) -> Tuple[Vector, List[List[Matrix]]]:
        patches = patchify(image, self.patch_size)
        patch_tokens = linear(patches, self.patch_weight, self.patch_bias)
        tokens = [self.class_token[:]] + patch_tokens
        tokens = add_rows(tokens, self.positions)

        all_attention = []
        for block in self.blocks:
            tokens, maps = self.encoder_block(tokens, block)
            all_attention.append(maps)

        class_representation = layer_norm([tokens[0]])
        logits = linear(class_representation, self.head_weight, self.head_bias)[0]
        return logits, all_attention


def demo_image() -> Image:
    """A 4x4 single-channel image with four visually distinct 2x2 patches."""
    values = [
        [0.0, 0.1, 0.8, 0.9],
        [0.2, 0.3, 1.0, 0.7],
        [0.9, 0.7, 0.2, 0.0],
        [1.0, 0.8, 0.1, 0.3],
    ]
    return [[[value] for value in row] for row in values]


def run_tests() -> None:
    image = demo_image()
    patches = patchify(image, 2)
    assert patches == [
        [0.0, 0.1, 0.2, 0.3],
        [0.8, 0.9, 1.0, 0.7],
        [0.9, 0.7, 1.0, 0.8],
        [0.2, 0.0, 0.1, 0.3],
    ]

    model = TinyVisionTransformer()
    dense_embedding = linear(patches, model.patch_weight, model.patch_bias)
    conv_embedding = conv_patch_embed(image, 2, model.patch_weight, model.patch_bias)
    for dense_row, conv_row in zip(dense_embedding, conv_embedding):
        assert all(abs(a - b) < 1e-12 for a, b in zip(dense_row, conv_row))

    logits, attention = model.forward(image)
    assert len(logits) == 3
    assert len(attention) == 2 and len(attention[0]) == 2
    assert all(len(row) == 5 for row in attention[0][0])  # [CLS] + 4 patches
    assert all(abs(sum(row) - 1.0) < 1e-9 for row in attention[0][0])

    old_grid = [[0.0], [1.0], [2.0], [3.0]]
    new_grid = bilinear_resize_position_grid(old_grid, old_side=2, new_side=3)
    assert len(new_grid) == 9
    assert new_grid[0] == [0.0] and new_grid[-1] == [3.0]
    assert new_grid[4] == [1.5]
    print("all tests passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run deterministic invariants")
    args = parser.parse_args()
    if args.test:
        run_tests()
        return

    image = demo_image()
    model = TinyVisionTransformer()
    logits, attention = model.forward(image)
    probabilities = softmax(logits)

    print("image shape:       4 x 4 x 1")
    print("patch shape:       2 x 2 x 1")
    print("patches:           4")
    print("encoder tokens:    5 ([CLS] + 4 patches)")
    print("logits:           ", [round(value, 5) for value in logits])
    print("probabilities:     ", [round(value, 5) for value in probabilities])
    print("layer-1 head-1 CLS attention:", [round(value, 5) for value in attention[0][0][0]])


if __name__ == "__main__":
    main()
