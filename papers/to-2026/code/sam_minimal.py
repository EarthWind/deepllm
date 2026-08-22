"""A dependency-free teaching sketch of SAM's promptable mask interface.

This is not Meta's SAM checkpoint.  It turns point/box prompts into a few
low-resolution candidate masks, upsamples them, and ranks them by a simple
stability heuristic.  The goal is to make prompt -> mask -> quality selection
executable without pretending to reproduce the ViT image encoder or mask
decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
from typing import Sequence


Mask = list[list[float]]


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    label: int  # 1 foreground, 0 background


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float


def gaussian_mask(height: int, width: int, points: Sequence[Point], box: Box | None = None, spread: float = 0.16) -> Mask:
    """Produce a soft prompt-conditioned mask in normalized image coordinates."""
    if height <= 0 or width <= 0 or spread <= 0:
        raise ValueError("invalid mask shape or spread")
    result = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            xn, yn = (x + 0.5) / width, (y + 0.5) / height
            score = 0.0
            for point in points:
                distance = ((xn - point.x) ** 2 + (yn - point.y) ** 2) / (2 * spread * spread)
                score += (1.0 if point.label else -1.0) * math.exp(-distance)
            if box is not None:
                inside = box.x0 <= xn <= box.x1 and box.y0 <= yn <= box.y1
                score += 0.35 if inside else -0.15
            result[y][x] = 1.0 / (1.0 + math.exp(-score * 8.0))
    return result


def resize_nearest(mask: Mask, height: int, width: int) -> Mask:
    if not mask or not mask[0] or height <= 0 or width <= 0:
        raise ValueError("invalid resize")
    src_h, src_w = len(mask), len(mask[0])
    return [[mask[min(src_h - 1, int(y * src_h / height))][min(src_w - 1, int(x * src_w / width))] for x in range(width)] for y in range(height)]


def stability_score(mask: Mask, low_threshold: float = 0.45, high_threshold: float = 0.55) -> float:
    """Fraction of pixels whose binary decision is stable across thresholds."""
    total = len(mask) * len(mask[0])
    stable = sum(1 for row in mask for value in row if value < low_threshold or value > high_threshold)
    return stable / total


def iou(pred: Mask, target: Mask, threshold: float = 0.5) -> float:
    if len(pred) != len(target) or len(pred[0]) != len(target[0]):
        raise ValueError("mask shapes differ")
    intersection = union = 0
    for a_row, b_row in zip(pred, target):
        for a, b in zip(a_row, b_row):
            aa, bb = a >= threshold, b >= threshold
            intersection += aa and bb
            union += aa or bb
    return intersection / union if union else 1.0


def candidate_masks(points: Sequence[Point], box: Box | None = None, low_res: tuple[int, int] = (8, 8), output: tuple[int, int] = (32, 32)) -> list[tuple[Mask, float]]:
    """Return multiple ambiguity-aware candidates, ranked by stability."""
    candidates = []
    for spread in (0.11, 0.16, 0.23):
        low = gaussian_mask(low_res[0], low_res[1], points, box, spread)
        full = resize_nearest(low, output[0], output[1])
        candidates.append((full, stability_score(full)))
    return sorted(candidates, key=lambda item: item[1], reverse=True)


def demo() -> None:
    prompt = [Point(0.47, 0.48, 1), Point(0.2, 0.2, 0)]
    box = Box(0.2, 0.2, 0.8, 0.8)
    candidates = candidate_masks(prompt, box)
    best, score = candidates[0]
    print("candidate masks:", len(candidates))
    print("best mask shape:", len(best), "x", len(best[0]))
    print("best stability score:", round(score, 3))
    print("center probability:", round(best[len(best) // 2][len(best[0]) // 2], 3))


def run_tests() -> None:
    mask = gaussian_mask(8, 8, [Point(0.5, 0.5, 1)])
    assert len(mask) == 8 and len(mask[0]) == 8
    enlarged = resize_nearest(mask, 16, 12)
    assert len(enlarged) == 16 and len(enlarged[0]) == 12
    assert 0.0 <= stability_score(mask) <= 1.0
    assert iou([[1.0, 0.0]], [[1.0, 0.0]]) == 1.0
    assert len(candidate_masks([Point(0.5, 0.5, 1)])) == 3
    print("all tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_tests() if args.test else demo()
