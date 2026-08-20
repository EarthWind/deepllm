#!/usr/bin/env python3
"""A zero-dependency DDPM mathematics and sampling demonstration.

This file keeps the essential DDPM data path visible without a tensor library:

    x_0 --q(x_t | x_0)--> x_t --epsilon predictor--> p(x_{t-1} | x_t) --> x_0

To make reverse sampling runnable without training a U-Net, the data distribution
is the one-dimensional mixture 0.5*delta(-1) + 0.5*delta(+1).  Its Bayes-optimal
noise predictor E[epsilon | x_t] is available in closed form and stands in for a
trained neural network.  The diffusion equations are the same scalar equations
used independently at every coordinate of an image DDPM.

The script demonstrates:
  * the linear beta schedule from Ho et al. (2020);
  * one-shot forward noising at any timestep;
  * q(x_{t-1} | x_t, x_0) and the epsilon-parameterized reverse mean;
  * the simplified noise-prediction training objective;
  * ancestral sampling from x_T ~ N(0, 1).

It is educational code, not a high-quality image training implementation.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple


NoisePredictor = Callable[[float, int], float]


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed scalar coefficients, indexed from 1 through timesteps."""

    timesteps: int
    beta: Tuple[float, ...]
    alpha: Tuple[float, ...]
    alpha_bar: Tuple[float, ...]
    posterior_variance: Tuple[float, ...]

    @classmethod
    def linear(
        cls,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> "DiffusionSchedule":
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError("require 0 < beta_start <= beta_end < 1")

        # Index 0 is a sentinel: alpha_bar[0] = 1 means x_0 has no noise.
        beta = [0.0]
        alpha = [1.0]
        alpha_bar = [1.0]
        posterior_variance = [0.0]

        for index in range(timesteps):
            fraction = index / (timesteps - 1)
            beta_t = beta_start + fraction * (beta_end - beta_start)
            alpha_t = 1.0 - beta_t
            beta.append(beta_t)
            alpha.append(alpha_t)
            alpha_bar.append(alpha_bar[-1] * alpha_t)

        for t in range(1, timesteps + 1):
            beta_tilde = beta[t] * (1.0 - alpha_bar[t - 1]) / (1.0 - alpha_bar[t])
            posterior_variance.append(beta_tilde)

        return cls(
            timesteps=timesteps,
            beta=tuple(beta),
            alpha=tuple(alpha),
            alpha_bar=tuple(alpha_bar),
            posterior_variance=tuple(posterior_variance),
        )


def q_sample(x0: float, t: int, epsilon: float, schedule: DiffusionSchedule) -> float:
    """Draw x_t directly from q(x_t | x_0), without simulating steps 1..t."""
    alpha_bar_t = schedule.alpha_bar[t]
    return math.sqrt(alpha_bar_t) * x0 + math.sqrt(1.0 - alpha_bar_t) * epsilon


def predict_x0_from_epsilon(
    x_t: float,
    t: int,
    epsilon_prediction: float,
    schedule: DiffusionSchedule,
) -> float:
    """Invert the forward reparameterization using a predicted epsilon."""
    alpha_bar_t = schedule.alpha_bar[t]
    return (x_t - math.sqrt(1.0 - alpha_bar_t) * epsilon_prediction) / math.sqrt(alpha_bar_t)


def q_posterior_mean_variance(
    x0: float,
    x_t: float,
    t: int,
    schedule: DiffusionSchedule,
) -> Tuple[float, float]:
    """Parameters of q(x_{t-1} | x_t, x_0)."""
    alpha_bar_t = schedule.alpha_bar[t]
    alpha_bar_prev = schedule.alpha_bar[t - 1]
    coefficient_x0 = math.sqrt(alpha_bar_prev) * schedule.beta[t] / (1.0 - alpha_bar_t)
    coefficient_xt = (
        math.sqrt(schedule.alpha[t])
        * (1.0 - alpha_bar_prev)
        / (1.0 - alpha_bar_t)
    )
    mean = coefficient_x0 * x0 + coefficient_xt * x_t
    return mean, schedule.posterior_variance[t]


def reverse_mean_from_epsilon(
    x_t: float,
    t: int,
    epsilon_prediction: float,
    schedule: DiffusionSchedule,
) -> float:
    """Equation (11): epsilon prediction converted to the reverse Gaussian mean."""
    return (
        x_t
        - schedule.beta[t]
        / math.sqrt(1.0 - schedule.alpha_bar[t])
        * epsilon_prediction
    ) / math.sqrt(schedule.alpha[t])


def make_oracle_epsilon_predictor(
    schedule: DiffusionSchedule,
    data_scale: float = 1.0,
) -> NoisePredictor:
    """Return E[epsilon | x_t] for x_0 uniformly distributed on {-a, +a}.

    For this two-point distribution,

        E[x_0 | x_t] = a * tanh(sqrt(alpha_bar_t) * a * x_t / (1-alpha_bar_t)).

    Substitution into x_t = sqrt(alpha_bar_t)x_0 + sqrt(1-alpha_bar_t)epsilon
    gives the Bayes-optimal MSE noise predictor.  A trained U-Net approximates the
    same conditional expectation for a vastly more complex image distribution.
    """

    def predict(x_t: float, t: int) -> float:
        alpha_bar_t = schedule.alpha_bar[t]
        noise_variance = 1.0 - alpha_bar_t
        posterior_x0 = data_scale * math.tanh(
            math.sqrt(alpha_bar_t) * data_scale * x_t / noise_variance
        )
        return (x_t - math.sqrt(alpha_bar_t) * posterior_x0) / math.sqrt(noise_variance)

    return predict


def simple_loss_example(
    x0: float,
    t: int,
    epsilon: float,
    predictor: NoisePredictor,
    schedule: DiffusionSchedule,
) -> float:
    """One scalar instance of L_simple = ||epsilon - epsilon_theta(x_t,t)||^2."""
    x_t = q_sample(x0, t, epsilon, schedule)
    error = epsilon - predictor(x_t, t)
    return error * error


def p_sample(
    x_t: float,
    t: int,
    predictor: NoisePredictor,
    schedule: DiffusionSchedule,
    rng: random.Random,
) -> float:
    """Draw one ancestral reverse step with sigma_t^2 = beta_tilde_t."""
    epsilon_prediction = predictor(x_t, t)
    mean = reverse_mean_from_epsilon(x_t, t, epsilon_prediction, schedule)
    if t == 1:
        return mean  # Algorithm 2 uses z = 0 for the final step.
    return mean + math.sqrt(schedule.posterior_variance[t]) * rng.gauss(0.0, 1.0)


def sample_reverse_chain(
    count: int,
    predictor: NoisePredictor,
    schedule: DiffusionSchedule,
    rng: random.Random,
) -> List[float]:
    """Start from x_T ~ N(0,1) and run all T reverse transitions."""
    samples = [rng.gauss(0.0, 1.0) for _ in range(count)]
    for t in range(schedule.timesteps, 0, -1):
        samples = [p_sample(value, t, predictor, schedule, rng) for value in samples]
    return samples


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def schedule_rows(schedule: DiffusionSchedule) -> Iterable[Tuple[int, float, float, float]]:
    candidates = [1, schedule.timesteps // 4, schedule.timesteps // 2]
    candidates += [3 * schedule.timesteps // 4, schedule.timesteps]
    for t in sorted(set(max(1, value) for value in candidates)):
        alpha_bar_t = schedule.alpha_bar[t]
        snr = alpha_bar_t / (1.0 - alpha_bar_t)
        yield t, schedule.beta[t], alpha_bar_t, snr


def algebra_checks(schedule: DiffusionSchedule, rng: random.Random) -> Tuple[float, float]:
    """Numerically verify two formula identities used by DDPM."""
    max_x0_error = 0.0
    max_mean_error = 0.0
    for _ in range(1000):
        t = rng.randint(1, schedule.timesteps)
        x0 = rng.uniform(-1.0, 1.0)
        epsilon = rng.gauss(0.0, 1.0)
        x_t = q_sample(x0, t, epsilon, schedule)

        recovered = predict_x0_from_epsilon(x_t, t, epsilon, schedule)
        max_x0_error = max(max_x0_error, abs(recovered - x0))

        posterior_mean, _ = q_posterior_mean_variance(x0, x_t, t, schedule)
        epsilon_mean = reverse_mean_from_epsilon(x_t, t, epsilon, schedule)
        max_mean_error = max(max_mean_error, abs(posterior_mean - epsilon_mean))
    return max_x0_error, max_mean_error


def run_demo(samples: int, seed: int) -> None:
    schedule = DiffusionSchedule.linear()
    rng = random.Random(seed)
    predictor = make_oracle_epsilon_predictor(schedule)

    print("Ho et al. linear schedule")
    print(" t       beta_t     alpha_bar_t        SNR_t")
    for t, beta_t, alpha_bar_t, snr in schedule_rows(schedule):
        print(f"{t:4d}  {beta_t:11.7f}  {alpha_bar_t:14.8f}  {snr:12.6g}")

    x0_error, mean_error = algebra_checks(schedule, rng)
    print("\nFormula checks")
    print(f"max x0 reconstruction error : {x0_error:.3e}")
    print(f"max posterior-mean mismatch : {mean_error:.3e}")

    losses_oracle = []
    losses_zero = []
    for _ in range(20_000):
        x0 = -1.0 if rng.random() < 0.5 else 1.0
        t = rng.randint(1, schedule.timesteps)
        epsilon = rng.gauss(0.0, 1.0)
        losses_oracle.append(simple_loss_example(x0, t, epsilon, predictor, schedule))
        losses_zero.append(epsilon * epsilon)
    print("\nSimplified objective on random training tuples")
    print(f"Bayes epsilon predictor MSE : {mean(losses_oracle):.4f}")
    print(f"zero predictor MSE          : {mean(losses_zero):.4f}")

    generated = sample_reverse_chain(samples, predictor, schedule, rng)
    positive_fraction = sum(value >= 0.0 for value in generated) / len(generated)
    nearest_mode_error = mean([min(abs(value - 1.0), abs(value + 1.0)) for value in generated])
    print("\nReverse samples for target 0.5*delta(-1) + 0.5*delta(+1)")
    print(f"count                      : {len(generated)}")
    print(f"mean / std                 : {mean(generated):+.4f} / {math.sqrt(variance(generated)):.4f}")
    print(f"fraction at positive mode  : {positive_fraction:.4f}")
    print(f"mean distance to +/-1      : {nearest_mode_error:.4f}")
    print("first 16 samples           : " + " ".join(f"{value:+.3f}" for value in generated[:16]))


def run_tests() -> None:
    schedule = DiffusionSchedule.linear()
    rng = random.Random(123)
    x0_error, mean_error = algebra_checks(schedule, rng)
    assert x0_error < 1e-12, x0_error
    assert mean_error < 1e-11, mean_error
    assert schedule.posterior_variance[1] == 0.0
    assert 3e-5 < schedule.alpha_bar[-1] < 5e-5

    predictor = make_oracle_epsilon_predictor(schedule)
    generated = sample_reverse_chain(1000, predictor, schedule, rng)
    positive_fraction = sum(value >= 0.0 for value in generated) / len(generated)
    nearest_mode_error = mean([min(abs(value - 1.0), abs(value + 1.0)) for value in generated])
    assert 0.40 < positive_fraction < 0.60, positive_fraction
    assert nearest_mode_error < 0.03, nearest_mode_error
    print("all DDPM checks passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000, help="number of reverse samples")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test", action="store_true", help="run deterministic assertions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.test:
        run_tests()
    else:
        run_demo(args.samples, args.seed)


if __name__ == "__main__":
    main()
