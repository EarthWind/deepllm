#!/usr/bin/env python3
"""Dependency-free miniature of Direct Preference Optimization (DPO).

The NeurIPS 2023 DPO objective for one preference pair is

    -log sigmoid(beta * ((log pi(y_w|x) - log pi_ref(y_w|x))
                         - (log pi(y_l|x) - log pi_ref(y_l|x))))

This file demonstrates the algorithm-specific pieces without downloading a
language model or requiring PyTorch:

1. sum token log-probabilities over the *completion only*;
2. compute DPO loss, preference probability, and implicit rewards;
3. show the analytical gradients with respect to sequence log-probabilities;
4. train a tiny categorical policy from offline preference pairs.

The categorical demo is intentionally small: each prompt has three complete
candidate responses, represented by trainable logits.  It makes the DPO update
observable while preserving the same pairwise log-ratio algebra as an LM.

Run:

    python3 papers/to-2026/code/dpo_minimal.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log(sum(exp(values)))."""

    if not values:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(values)
    if math.isinf(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def log_softmax(logits: Sequence[float]) -> tuple[float, ...]:
    normalizer = logsumexp(logits)
    return tuple(value - normalizer for value in logits)


def sigmoid(value: float) -> float:
    """Stable logistic sigmoid."""

    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def softplus(value: float) -> float:
    """Stable log(1 + exp(value))."""

    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def sequence_log_probability(
    token_logits: Sequence[Sequence[float]],
    target_token_ids: Sequence[int],
    completion_mask: Sequence[bool],
) -> float:
    """Sum log p(target token) only where ``completion_mask`` is true.

    ``token_logits[t]`` is assumed to predict ``target_token_ids[t]`` already.
    A real causal LM must first shift logits left and labels right. Prompt and
    padding positions must be false in the mask. The original DPO implementation
    sums (rather than length-averages) response-token log-probabilities.
    """

    if not (
        len(token_logits) == len(target_token_ids) == len(completion_mask)
    ):
        raise ValueError("logits, targets, and mask must have the same length")

    total = 0.0
    selected = 0
    for position, (logits, target, keep) in enumerate(
        zip(token_logits, target_token_ids, completion_mask)
    ):
        if not logits:
            raise ValueError(f"position {position} has an empty vocabulary")
        if not 0 <= target < len(logits):
            raise ValueError(f"target {target} is invalid at position {position}")
        if keep:
            total += log_softmax(logits)[target]
            selected += 1
    if selected == 0:
        raise ValueError("completion mask selects no tokens")
    return total


@dataclass(frozen=True)
class PairLogProbabilities:
    """Sequence log-probabilities for one offline preference pair."""

    policy_chosen: float
    policy_rejected: float
    reference_chosen: float
    reference_rejected: float


@dataclass(frozen=True)
class DPOMetrics:
    loss: float
    preference_logit: float
    preference_probability: float
    policy_log_ratio: float
    reference_log_ratio: float
    relative_margin: float
    chosen_implicit_reward: float
    rejected_implicit_reward: float
    reward_margin: float
    dloss_d_policy_chosen: float
    dloss_d_policy_rejected: float


def dpo_metrics(logps: PairLogProbabilities, *, beta: float = 0.1) -> DPOMetrics:
    """Compute the original DPO loss and useful diagnostics for one pair."""

    if beta <= 0.0:
        raise ValueError("beta must be positive")

    policy_log_ratio = logps.policy_chosen - logps.policy_rejected
    reference_log_ratio = logps.reference_chosen - logps.reference_rejected
    relative_margin = policy_log_ratio - reference_log_ratio
    preference_logit = beta * relative_margin
    loss = softplus(-preference_logit)  # exactly -log sigmoid(logit)
    probability = sigmoid(preference_logit)

    chosen_reward = beta * (
        logps.policy_chosen - logps.reference_chosen
    )
    rejected_reward = beta * (
        logps.policy_rejected - logps.reference_rejected
    )

    # d[-log sigma(z)]/dz = -sigma(-z), z = beta * margin.
    gradient_scale = -beta * sigmoid(-preference_logit)
    return DPOMetrics(
        loss=loss,
        preference_logit=preference_logit,
        preference_probability=probability,
        policy_log_ratio=policy_log_ratio,
        reference_log_ratio=reference_log_ratio,
        relative_margin=relative_margin,
        chosen_implicit_reward=chosen_reward,
        rejected_implicit_reward=rejected_reward,
        reward_margin=chosen_reward - rejected_reward,
        dloss_d_policy_chosen=gradient_scale,
        dloss_d_policy_rejected=-gradient_scale,
    )


def dpo_batch_loss(
    examples: Iterable[PairLogProbabilities],
    *,
    beta: float = 0.1,
) -> float:
    losses = [dpo_metrics(example, beta=beta).loss for example in examples]
    if not losses:
        raise ValueError("a batch must contain at least one preference pair")
    return sum(losses) / len(losses)


@dataclass(frozen=True)
class PreferencePair:
    prompt_id: str
    chosen_index: int
    rejected_index: int


class CategoricalPolicy:
    """A tiny policy over complete responses for each prompt.

    This is not a token-level language model. It is a transparent stand-in in
    which each candidate response is one categorical action. For a pair sharing
    the same prompt, ``log pi(chosen) - log pi(rejected)`` equals the difference
    between their logits because the softmax normalizer cancels.
    """

    def __init__(self, logits_by_prompt: dict[str, Sequence[float]]) -> None:
        if not logits_by_prompt:
            raise ValueError("policy needs at least one prompt")
        self.logits_by_prompt = {
            prompt: [float(value) for value in logits]
            for prompt, logits in logits_by_prompt.items()
        }
        if any(not logits for logits in self.logits_by_prompt.values()):
            raise ValueError("every prompt needs at least one candidate")

    def copy(self) -> CategoricalPolicy:
        return CategoricalPolicy(self.logits_by_prompt)

    def log_probabilities(self, prompt_id: str) -> tuple[float, ...]:
        return log_softmax(self.logits_by_prompt[prompt_id])

    def probabilities(self, prompt_id: str) -> tuple[float, ...]:
        return tuple(math.exp(value) for value in self.log_probabilities(prompt_id))

    def pair_logps(
        self,
        reference: CategoricalPolicy,
        pair: PreferencePair,
    ) -> PairLogProbabilities:
        policy = self.log_probabilities(pair.prompt_id)
        ref = reference.log_probabilities(pair.prompt_id)
        return PairLogProbabilities(
            policy_chosen=policy[pair.chosen_index],
            policy_rejected=policy[pair.rejected_index],
            reference_chosen=ref[pair.chosen_index],
            reference_rejected=ref[pair.rejected_index],
        )

    def sgd_step(
        self,
        reference: CategoricalPolicy,
        pairs: Sequence[PreferencePair],
        *,
        beta: float,
        learning_rate: float,
    ) -> float:
        """Take one full-batch DPO step and return pre-update mean loss."""

        if not pairs:
            raise ValueError("training requires preference pairs")
        gradients = {
            prompt: [0.0] * len(logits)
            for prompt, logits in self.logits_by_prompt.items()
        }
        losses: list[float] = []

        for pair in pairs:
            metrics = dpo_metrics(self.pair_logps(reference, pair), beta=beta)
            losses.append(metrics.loss)

            # For responses sharing a prompt, the derivative of
            # log pi(chosen) - log pi(rejected) w.r.t. categorical logits is
            # +1 for chosen, -1 for rejected; softmax normalization cancels.
            gradients[pair.prompt_id][pair.chosen_index] += (
                metrics.dloss_d_policy_chosen
            )
            gradients[pair.prompt_id][pair.rejected_index] += (
                metrics.dloss_d_policy_rejected
            )

        scale = learning_rate / len(pairs)
        for prompt, prompt_gradients in gradients.items():
            for index, gradient in enumerate(prompt_gradients):
                self.logits_by_prompt[prompt][index] -= scale * gradient
        return sum(losses) / len(losses)


def preference_accuracy(
    policy: CategoricalPolicy,
    reference: CategoricalPolicy,
    pairs: Sequence[PreferencePair],
) -> float:
    correct = 0
    for pair in pairs:
        logps = policy.pair_logps(reference, pair)
        margin = (
            logps.policy_chosen
            - logps.policy_rejected
            - logps.reference_chosen
            + logps.reference_rejected
        )
        correct += margin > 0.0
    return correct / len(pairs)


RESPONSES = {
    "refund": (
        "Acknowledge the issue, explain the refund steps, and give a timeline.",
        "Refunds are impossible. Read the policy.",
        "I cannot help with that.",
    ),
    "science": (
        "Explain evaporation with a concise everyday example.",
        "Give an unrelated definition of gravity.",
        "Use jargon without answering the question.",
    ),
    "safety": (
        "Decline the harmful request and offer a safe alternative.",
        "Provide the requested harmful instructions.",
        "Refuse without any useful redirection.",
    ),
}


def demo() -> tuple[CategoricalPolicy, CategoricalPolicy, tuple[PreferencePair, ...]]:
    initial_logits = {
        "refund": (0.20, 0.10, -0.05),
        "science": (-0.10, 0.15, 0.05),
        "safety": (0.00, 0.20, -0.10),
    }
    policy = CategoricalPolicy(initial_logits)
    reference = policy.copy()  # frozen snapshot of the starting/SFT policy
    pairs = (
        PreferencePair("refund", 0, 1),
        PreferencePair("refund", 0, 2),
        PreferencePair("science", 0, 1),
        PreferencePair("science", 0, 2),
        PreferencePair("safety", 0, 1),
        PreferencePair("safety", 0, 2),
    )

    beta = 0.2
    for _ in range(200):
        policy.sgd_step(
            reference,
            pairs,
            beta=beta,
            learning_rate=1.0,
        )
    return policy, reference, pairs


def _self_check() -> None:
    equal = PairLogProbabilities(-2.0, -3.0, -2.0, -3.0)
    metrics = dpo_metrics(equal, beta=0.1)
    assert math.isclose(metrics.loss, math.log(2.0), rel_tol=1e-12)
    assert math.isclose(metrics.preference_probability, 0.5, rel_tol=1e-12)
    assert math.isclose(metrics.reward_margin, 0.0, abs_tol=1e-12)
    assert metrics.dloss_d_policy_chosen < 0.0
    assert metrics.dloss_d_policy_rejected > 0.0

    improved = PairLogProbabilities(-1.0, -4.0, -2.0, -3.0)
    assert dpo_metrics(improved, beta=0.1).loss < metrics.loss

    token_logits = (
        (5.0, 0.0),  # prompt position: deliberately very confident, but masked
        (0.0, 3.0),  # completion token 1
        (2.0, 0.0),  # completion token 0
    )
    masked = sequence_log_probability(
        token_logits,
        target_token_ids=(1, 1, 0),
        completion_mask=(False, True, True),
    )
    expected = log_softmax(token_logits[1])[1] + log_softmax(token_logits[2])[0]
    assert math.isclose(masked, expected, rel_tol=1e-12)

    policy, reference, pairs = demo()
    assert preference_accuracy(policy, reference, pairs) == 1.0
    assert dpo_batch_loss(
        (policy.pair_logps(reference, pair) for pair in pairs),
        beta=0.2,
    ) < math.log(2.0)


if __name__ == "__main__":
    _self_check()
    trained, frozen_reference, preference_pairs = demo()
    print("Tiny offline DPO training (reference remains frozen):")
    print(f"Preference accuracy: {preference_accuracy(trained, frozen_reference, preference_pairs):.0%}")
    for prompt_id, responses in RESPONSES.items():
        before = frozen_reference.probabilities(prompt_id)
        after = trained.probabilities(prompt_id)
        print(f"\n{prompt_id}:")
        for response, old_probability, new_probability in zip(
            responses, before, after
        ):
            print(
                f"  {old_probability:5.1%} -> {new_probability:5.1%}  {response}"
            )

    first_pair = preference_pairs[0]
    diagnostics = dpo_metrics(
        trained.pair_logps(frozen_reference, first_pair), beta=0.2
    )
    print("\nFirst-pair diagnostics:")
    print(f"  relative margin:       {diagnostics.relative_margin:.4f}")
    print(f"  preference probability:{diagnostics.preference_probability:8.2%}")
    print(f"  chosen implicit reward:{diagnostics.chosen_implicit_reward:8.4f}")
    print(f"  rejected implicit reward:{diagnostics.rejected_implicit_reward:6.4f}")
    print(f"  DPO loss:              {diagnostics.loss:.4f}")
