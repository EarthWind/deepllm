#!/usr/bin/env python3
"""A zero-dependency teaching model for Gemini 1.0 concepts.

This is NOT Gemini source code and does not reconstruct its private architecture.
It only makes three report-level ideas executable:

1. text, image, audio and video segments can be kept in one interleaved order;
2. modality-specific observations can be mapped to one model-width sequence;
3. an autoregressive decoder uses a causal mask, while MMLU-style evaluation can
   route between sampled chain-of-thought consensus and a greedy fallback.

Run:
    python3 papers/to-2026/code/gemini_multimodal_minimal.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import exp, sqrt
from typing import Iterable, Literal, Sequence, TypeVar


Modality = Literal["text", "image", "audio", "video"]
T = TypeVar("T")


@dataclass(frozen=True)
class Segment:
    """One user-visible segment in an interleaved prompt."""

    modality: Modality
    payload: object


@dataclass(frozen=True)
class ModelToken:
    """A toy token after frontend processing and width projection."""

    modality: Modality
    segment_index: int
    local_index: int
    vector: tuple[float, ...]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _project(raw: Sequence[float], modality: Modality, width: int) -> tuple[float, ...]:
    """Deterministic toy projector; real Gemini projectors are not disclosed."""

    phase = {"text": 0.17, "image": 0.43, "audio": 0.71, "video": 0.89}[modality]
    return tuple(
        sum(value * (((i + 1) * (j + 3)) % 7 - 3) for j, value in enumerate(raw))
        / max(1, len(raw))
        + phase
        for i in range(width)
    )


def _encode_text(text: str) -> list[tuple[float, ...]]:
    pieces = text.split()
    return [
        (
            min(len(piece), 16) / 16,
            sum(ch.lower() in "aeiou" for ch in piece) / max(1, len(piece)),
            sum(ord(ch) for ch in piece) % 257 / 257,
        )
        for piece in pieces
    ]


def _encode_image(image: Sequence[Sequence[float]], patch: int = 2) -> list[tuple[float, ...]]:
    """Patch a tiny grayscale image and return mean/contrast/position features."""

    height = len(image)
    width = len(image[0]) if height else 0
    features: list[tuple[float, ...]] = []
    for top in range(0, height, patch):
        for left in range(0, width, patch):
            pixels = [
                float(image[row][col])
                for row in range(top, min(top + patch, height))
                for col in range(left, min(left + patch, width))
            ]
            if not pixels:
                continue
            avg = _mean(pixels)
            variance = _mean([(value - avg) ** 2 for value in pixels])
            features.append((avg, sqrt(variance), top / max(1, height), left / max(1, width)))
    return features


def _encode_audio(samples: Sequence[float], frame: int = 4) -> list[tuple[float, ...]]:
    """Frame a waveform; Gemini reports using features from USM, not this frontend."""

    features: list[tuple[float, ...]] = []
    for start in range(0, len(samples), frame):
        chunk = [float(value) for value in samples[start : start + frame]]
        if not chunk:
            continue
        rms = sqrt(_mean([value * value for value in chunk]))
        crossings = sum((a >= 0) != (b >= 0) for a, b in zip(chunk, chunk[1:]))
        features.append((_mean(chunk), rms, crossings / max(1, len(chunk) - 1)))
    return features


def _encode_video(video: Sequence[Sequence[Sequence[float]]]) -> list[tuple[float, ...]]:
    """Treat video as ordered frames, matching the report's high-level description."""

    features: list[tuple[float, ...]] = []
    for time_index, frame in enumerate(video):
        time = time_index / max(1, len(video) - 1)
        for patch_feature in _encode_image(frame):
            features.append((*patch_feature, time))
    return features


def _uniform_cap(items: Sequence[T], budget: int) -> list[T]:
    if budget <= 0:
        return []
    if len(items) <= budget:
        return list(items)
    return [items[min(len(items) - 1, i * len(items) // budget)] for i in range(budget)]


def pack_interleaved(
    segments: Sequence[Segment],
    *,
    model_width: int = 8,
    per_segment_budget: int = 6,
) -> list[ModelToken]:
    """Encode every segment while preserving the user's cross-modal order."""

    encoders = {
        "text": _encode_text,
        "image": _encode_image,
        "audio": _encode_audio,
        "video": _encode_video,
    }
    packed: list[ModelToken] = []
    for segment_index, segment in enumerate(segments):
        raw_features = encoders[segment.modality](segment.payload)  # type: ignore[arg-type]
        for local_index, raw in enumerate(_uniform_cap(raw_features, per_segment_budget)):
            packed.append(
                ModelToken(
                    modality=segment.modality,
                    segment_index=segment_index,
                    local_index=local_index,
                    vector=_project(raw, segment.modality, model_width),
                )
            )
    return packed


def causal_mask(length: int) -> list[list[bool]]:
    """mask[q][k] is true exactly when query q may read key k."""

    return [[key <= query for key in range(length)] for query in range(length)]


def causal_self_attention(vectors: Sequence[Sequence[float]]) -> list[tuple[float, ...]]:
    """One-head, identity-projection attention used only to demonstrate causality."""

    if not vectors:
        return []
    width = len(vectors[0])
    scale = sqrt(width)
    outputs: list[tuple[float, ...]] = []
    for query_index, query in enumerate(vectors):
        visible = vectors[: query_index + 1]
        logits = [sum(a * b for a, b in zip(query, key)) / scale for key in visible]
        peak = max(logits)
        weights = [exp(logit - peak) for logit in logits]
        normalizer = sum(weights)
        outputs.append(
            tuple(
                sum(weight * value[col] for weight, value in zip(weights, visible)) / normalizer
                for col in range(width)
            )
        )
    return outputs


@dataclass(frozen=True)
class RoutedAnswer:
    answer: str
    route: Literal["sample-majority", "greedy-fallback"]
    consensus: float


def uncertainty_routed_answer(
    greedy_answer: str,
    sampled_answers: Iterable[str],
    *,
    threshold: float,
) -> RoutedAnswer:
    """Reproduce the decision rule—not the scores—of uncertainty-routed CoT."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    answers = list(sampled_answers)
    if not answers:
        return RoutedAnswer(greedy_answer, "greedy-fallback", 0.0)
    winner, votes = Counter(answers).most_common(1)[0]
    consensus = votes / len(answers)
    if consensus >= threshold:
        return RoutedAnswer(winner, "sample-majority", consensus)
    return RoutedAnswer(greedy_answer, "greedy-fallback", consensus)


def _demo() -> None:
    image = [
        [0.0, 0.1, 0.8, 0.9],
        [0.1, 0.2, 0.7, 1.0],
        [0.8, 0.7, 0.2, 0.1],
        [0.9, 0.8, 0.1, 0.0],
    ]
    video = [image, [list(reversed(row)) for row in image]]
    prompt = [
        Segment("text", "Compare this diagram"),
        Segment("image", image),
        Segment("text", "with the sound and clip"),
        Segment("audio", [0.0, 0.6, -0.4, 0.8, -0.7, 0.2, 0.1, -0.2]),
        Segment("video", video),
    ]
    tokens = pack_interleaved(prompt, per_segment_budget=4)
    ledger = [(token.modality, token.segment_index, token.local_index) for token in tokens]
    print("packed token ledger:")
    print(ledger)

    mask = causal_mask(len(tokens))
    assert all(mask[q][k] == (k <= q) for q in range(len(tokens)) for k in range(len(tokens)))

    original = [token.vector for token in tokens]
    changed_future = [*original[:-1], tuple(value + 100.0 for value in original[-1])]
    before = causal_self_attention(original)
    after = causal_self_attention(changed_future)
    max_earlier_change = max(
        abs(a - b)
        for left, right in zip(before[:-1], after[:-1])
        for a, b in zip(left, right)
    )
    assert max_earlier_change == 0.0
    print(f"causal future-leak check: max earlier change = {max_earlier_change:.1f}")

    confident = uncertainty_routed_answer("B", ["A"] * 26 + ["B"] * 6, threshold=0.75)
    uncertain = uncertainty_routed_answer("B", ["A"] * 18 + ["B"] * 14, threshold=0.75)
    assert confident.route == "sample-majority" and confident.answer == "A"
    assert uncertain.route == "greedy-fallback" and uncertain.answer == "B"
    print("high-consensus route:", confident)
    print("low-consensus route: ", uncertain)
    print("all checks passed")


if __name__ == "__main__":
    _demo()
