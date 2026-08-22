"""A dependency-free teaching implementation of the LDM equations.

This file does not train a VAE or a U-Net.  It makes the important interfaces
executable: latent scaling, forward noising, epsilon loss, cross-attention
conditioning, and one deterministic DDIM-style reverse step.
Run: ``python latent_diffusion_minimal.py --test``.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import random
from typing import Sequence


Vector = list[float]
Matrix = list[Vector]


def linear_beta_schedule(steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> Vector:
    if steps < 2:
        raise ValueError("steps must be >= 2")
    return [beta_start + (beta_end - beta_start) * i / (steps - 1) for i in range(steps)]


def cumulative_alphas(betas: Sequence[float]) -> Vector:
    product = 1.0
    result = []
    for beta in betas:
        product *= 1.0 - beta
        result.append(product)
    return result


def q_sample(x0: Sequence[float], timestep: int, alpha_bars: Sequence[float], noise: Sequence[float]) -> Vector:
    """Forward diffusion: x_t = sqrt(alpha_bar)x_0 + sqrt(1-alpha_bar)epsilon."""
    if not (0 <= timestep < len(alpha_bars)) or len(x0) != len(noise):
        raise ValueError("invalid timestep or vector lengths")
    a = alpha_bars[timestep]
    return [math.sqrt(a) * x + math.sqrt(1.0 - a) * e for x, e in zip(x0, noise)]


def mse(pred: Sequence[float], target: Sequence[float]) -> float:
    if len(pred) != len(target) or not pred:
        raise ValueError("vectors must be non-empty and have equal length")
    return sum((a - b) ** 2 for a, b in zip(pred, target)) / len(pred)


def cross_attention(query: Matrix, context: Matrix, *, temperature: float | None = None) -> Matrix:
    """Tiny single-head attention, useful for seeing text-to-image conditioning."""
    if not query or not context or any(len(row) != len(query[0]) for row in query + context):
        raise ValueError("query and context must be non-empty, same width")
    d = len(query[0])
    scale = temperature or math.sqrt(d)
    outputs: Matrix = []
    for q in query:
        logits = [sum(qi * ci for qi, ci in zip(q, c)) / scale for c in context]
        peak = max(logits)
        weights = [math.exp(v - peak) for v in logits]
        z = sum(weights)
        weights = [w / z for w in weights]
        outputs.append([sum(w * c[j] for w, c in zip(weights, context)) for j in range(d)])
    return outputs


def ddim_step(x_t: Sequence[float], eps_pred: Sequence[float], t: int, t_prev: int, alpha_bars: Sequence[float]) -> Vector:
    """Deterministic DDIM update with eta=0."""
    if t <= t_prev or t >= len(alpha_bars) or t_prev < 0:
        raise ValueError("require t > t_prev >= 0")
    a_t, a_prev = alpha_bars[t], alpha_bars[t_prev]
    x0_hat = [(x - math.sqrt(1 - a_t) * e) / math.sqrt(a_t) for x, e in zip(x_t, eps_pred)]
    return [math.sqrt(a_prev) * x0 + math.sqrt(1 - a_prev) * e for x0, e in zip(x0_hat, eps_pred)]


@dataclass(frozen=True)
class LatentShape:
    image_height: int
    image_width: int
    downsample_factor: int
    channels: int

    @property
    def latent_height(self) -> int:
        return self.image_height // self.downsample_factor

    @property
    def latent_width(self) -> int:
        return self.image_width // self.downsample_factor

    @property
    def pixel_elements(self) -> int:
        return 3 * self.image_height * self.image_width

    @property
    def latent_elements(self) -> int:
        return self.channels * self.latent_height * self.latent_width

    @property
    def element_ratio(self) -> float:
        return self.pixel_elements / self.latent_elements


def scaled_latent(z: Sequence[float], scale: float = 0.18215) -> Vector:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return [scale * value for value in z]


def demo() -> None:
    betas = linear_beta_schedule(8)
    alpha_bars = cumulative_alphas(betas)
    x0 = [0.4, -0.2, 0.8, 0.1]
    noise = [0.1, -0.3, 0.2, 0.4]
    xt = q_sample(x0, 4, alpha_bars, noise)
    recovered = ddim_step(xt, noise, 4, 3, alpha_bars)
    shape = LatentShape(512, 512, 8, 4)
    attention = cross_attention([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.2, 0.8]])
    print("latent shape:", shape.latent_height, "x", shape.latent_width, "x", shape.channels)
    print("pixel/latent element ratio:", round(shape.element_ratio, 1))
    print("x_t:", [round(v, 4) for v in xt])
    print("DDIM x_{t-1}:", [round(v, 4) for v in recovered])
    print("cross-attention:", [[round(v, 4) for v in row] for row in attention])


def run_tests() -> None:
    betas = linear_beta_schedule(10)
    alpha_bars = cumulative_alphas(betas)
    assert all(0 < a < 1 for a in alpha_bars)
    x0 = [1.0, -1.0]
    noise = [0.0, 0.0]
    assert q_sample(x0, 0, alpha_bars, noise)[0] < 1.0
    assert mse([1, 2], [1, 4]) == 2.0
    assert len(cross_attention([[1, 0]], [[1, 0], [0, 1]])[0]) == 2
    shape = LatentShape(512, 512, 8, 4)
    assert (shape.latent_height, shape.latent_width) == (64, 64)
    assert scaled_latent([2.0], 0.5) == [1.0]
    print("all tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_tests() if args.test else demo()
