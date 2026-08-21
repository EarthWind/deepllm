#!/usr/bin/env python3
"""Zero-dependency teaching implementation of CLIP's mathematical core.

This file intentionally starts *after* the image and text encoders. It keeps
the part that makes CLIP CLIP completely visible with ordinary Python lists:

    encoder outputs -> L2 normalization -> N x N cosine logits
                    -> image-to-text CE + text-to-image CE
                    -> prompt-ensemble zero-shot classifier

It is not a replacement for OpenAI's released model and it does not train a
CNN or Transformer. The tiny deterministic example is designed for tracing
shapes, probabilities and temperature without third-party packages.
"""

from __future__ import annotations

import argparse
import math
from typing import List, Sequence, Tuple


Vector = List[float]
Matrix = List[Vector]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    return sum(x * y for x, y in zip(a, b))


def l2_normalize(vector: Sequence[float], eps: float = 1e-12) -> Vector:
    """Project one vector onto the unit hypersphere."""
    norm = math.sqrt(dot(vector, vector))
    if norm < eps:
        raise ValueError("cannot normalize a zero vector")
    return [value / norm for value in vector]


def normalize_rows(matrix: Sequence[Sequence[float]]) -> Matrix:
    if not matrix:
        raise ValueError("matrix must not be empty")
    return [l2_normalize(row) for row in matrix]


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    if not matrix or not matrix[0]:
        raise ValueError("matrix must not be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have equal length")
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(width)]


def softmax(logits: Sequence[float]) -> Vector:
    """Stable softmax: subtracting max leaves probabilities unchanged."""
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def similarity_logits(
    image_features: Sequence[Sequence[float]],
    text_features: Sequence[Sequence[float]],
    logit_scale: float,
) -> Matrix:
    """Return scaled cosine similarities with shape [images, texts]."""
    if logit_scale <= 0:
        raise ValueError("logit_scale must be positive")
    images = normalize_rows(image_features)
    texts = normalize_rows(text_features)
    if len(images[0]) != len(texts[0]):
        raise ValueError("image and text embeddings must share a dimension")
    return [[logit_scale * dot(image, text) for text in texts] for image in images]


def mean_diagonal_cross_entropy(logits: Sequence[Sequence[float]]) -> float:
    """Cross entropy when row i's correct target is column i."""
    if len(logits) != len(logits[0]):
        raise ValueError("paired CLIP training expects a square logits matrix")
    losses = []
    for index, row in enumerate(logits):
        probabilities = softmax(row)
        losses.append(-math.log(max(probabilities[index], 1e-300)))
    return sum(losses) / len(losses)


def clip_loss(
    image_features: Sequence[Sequence[float]],
    text_features: Sequence[Sequence[float]],
    logit_scale: float = 1.0 / 0.07,
) -> Tuple[float, float, float, Matrix]:
    """Symmetric CLIP loss and both directional components.

    Returns (symmetric_loss, image_to_text_loss, text_to_image_loss, logits).
    Each diagonal position i is assumed to be the real (image_i, text_i) pair.
    """
    if len(image_features) != len(text_features):
        raise ValueError("a training batch needs one text for every image")
    logits = similarity_logits(image_features, text_features, logit_scale)
    image_to_text = mean_diagonal_cross_entropy(logits)
    text_to_image = mean_diagonal_cross_entropy(transpose(logits))
    return (image_to_text + text_to_image) / 2.0, image_to_text, text_to_image, logits


def prompt_ensemble(prompt_features: Sequence[Sequence[float]]) -> Vector:
    """Average normalized prompt embeddings, then normalize once more."""
    normalized = normalize_rows(prompt_features)
    dimension = len(normalized[0])
    average = [
        sum(row[d] for row in normalized) / len(normalized)
        for d in range(dimension)
    ]
    return l2_normalize(average)


def build_zero_shot_classifier(
    class_prompt_features: Sequence[Sequence[Sequence[float]]],
) -> Matrix:
    """Create one normalized classifier weight per class."""
    return [prompt_ensemble(prompts) for prompts in class_prompt_features]


def zero_shot_predict(
    image_feature: Sequence[float],
    classifier: Sequence[Sequence[float]],
    logit_scale: float = 100.0,
) -> Tuple[int, Vector, Vector]:
    """Predict among text-defined classes; no learned task-specific head."""
    logits = similarity_logits([image_feature], classifier, logit_scale)[0]
    probabilities = softmax(logits)
    prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
    return prediction, probabilities, logits


def format_matrix(matrix: Sequence[Sequence[float]], digits: int = 2) -> str:
    return "\n".join(
        "  " + " ".join(f"{value:>{digits + 5}.{digits}f}" for value in row)
        for row in matrix
    )


def demo() -> None:
    # Pretend these rows came from an image encoder and a text Transformer.
    # Index-aligned rows describe the same concept: cat, aircraft, and food.
    image_features = [
        [1.00, 0.18, 0.05, 0.10],
        [0.03, 1.00, 0.12, 0.18],
        [0.12, 0.04, 1.00, 0.20],
    ]
    text_features = [
        [0.91, 0.16, 0.08, 0.18],
        [0.07, 0.95, 0.10, 0.22],
        [0.18, 0.05, 0.92, 0.15],
    ]

    loss, image_loss, text_loss, logits = clip_loss(
        image_features, text_features, logit_scale=1.0 / 0.07
    )
    print("paired batch:       3 images x 3 texts")
    print("logit scale:        1 / 0.07 = 14.2857")
    print("similarity logits:")
    print(format_matrix(logits))
    print(f"image -> text CE:   {image_loss:.5f}")
    print(f"text -> image CE:   {text_loss:.5f}")
    print(f"symmetric CLIP CE:  {loss:.5f}")

    # Two prompt variants for each of three classes. A real model would obtain
    # these rows by encoding strings such as "a photo of a {label}".
    class_names = ["cat", "aircraft", "food"]
    class_prompt_features = [
        [[0.95, 0.16, 0.06, 0.18], [0.87, 0.22, 0.10, 0.13]],
        [[0.06, 0.96, 0.11, 0.20], [0.10, 0.89, 0.07, 0.25]],
        [[0.17, 0.05, 0.94, 0.13], [0.10, 0.08, 0.88, 0.25]],
    ]
    classifier = build_zero_shot_classifier(class_prompt_features)
    query_image = [0.08, 0.11, 0.97, 0.20]
    predicted, probabilities, _ = zero_shot_predict(query_image, classifier)

    print("\nzero-shot classifier: 3 classes x 2 prompts")
    for name, probability in zip(class_names, probabilities):
        print(f"  {name:>8}: {probability * 100:6.2f}%")
    print(f"prediction:          {class_names[predicted]}")

    # Temperature changes confidence, not the ordering of fixed similarities.
    _, soft_probs, _ = zero_shot_predict(query_image, classifier, logit_scale=1.0)
    print(f"top probability at scale 1:   {max(soft_probs) * 100:.2f}%")
    print(f"top probability at scale 100: {max(probabilities) * 100:.2f}%")


def assert_close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} != {expected} within {tolerance}")


def run_tests() -> None:
    unit = l2_normalize([3.0, 4.0])
    assert_close(dot(unit, unit), 1.0)

    probabilities = softmax([1001.0, 1002.0, 1003.0])
    assert_close(sum(probabilities), 1.0)
    assert probabilities[2] > probabilities[1] > probabilities[0]

    images = [[1.0, 0.0], [0.0, 1.0]]
    aligned_texts = [[0.9, 0.1], [0.1, 0.9]]
    swapped_texts = [aligned_texts[1], aligned_texts[0]]
    aligned_loss, row_loss, column_loss, logits = clip_loss(images, aligned_texts)
    swapped_loss, _, _, _ = clip_loss(images, swapped_texts)
    assert aligned_loss < swapped_loss
    assert_close(aligned_loss, (row_loss + column_loss) / 2.0)
    expected_transpose = similarity_logits(aligned_texts, images, 1.0 / 0.07)
    assert transpose(logits) == expected_transpose

    ensemble = prompt_ensemble([[2.0, 0.0], [1.0, 1.0]])
    assert_close(dot(ensemble, ensemble), 1.0)

    classifier = build_zero_shot_classifier(
        [
            [[1.0, 0.0], [0.9, 0.1]],
            [[0.0, 1.0], [0.1, 0.9]],
        ]
    )
    low_prediction, low_probabilities, _ = zero_shot_predict(
        [0.8, 0.2], classifier, logit_scale=1.0
    )
    high_prediction, high_probabilities, _ = zero_shot_predict(
        [0.8, 0.2], classifier, logit_scale=100.0
    )
    assert low_prediction == high_prediction == 0
    assert max(high_probabilities) > max(low_probabilities)
    assert_close(sum(high_probabilities), 1.0)

    try:
        clip_loss([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
    except ValueError:
        pass
    else:
        raise AssertionError("unpaired batches must be rejected")

    print("all tests passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run invariant checks")
    args = parser.parse_args()
    if args.test:
        run_tests()
    else:
        demo()


if __name__ == "__main__":
    main()
