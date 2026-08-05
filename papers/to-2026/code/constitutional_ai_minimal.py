#!/usr/bin/env python3
"""Dependency-free miniature of the Constitutional AI training data flow.

The paper does not release a complete trainer.  This module isolates the parts
that are specific to Constitutional AI and makes them executable without a
model SDK:

1. repeatedly sample a principle, critique a response, and revise it;
2. turn *every* revision into supervised-learning data;
3. ask a feedback model to compare two responses with a constitutional rule;
4. preserve its calibrated A/B probabilities as a soft preference target;
5. train a scalar preference model with soft Bradley--Terry cross-entropy; and
6. expose the KL-regularized sequence objective used by the later RL stage.

The option-order randomization below is a production guard against position
bias.  The paper specifies an A/B multiple-choice format, but does not claim
this exact randomization helper as part of its implementation.

Adapt a local or remote language model to ``Generator`` and ``OptionScorer``.
Run the deterministic self-check with:

    python3 papers/to-2026/code/constitutional_ai_minimal.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence


Generator = Callable[[str, int], str]
OptionScorer = Callable[[str], tuple[float, float]]


@dataclass(frozen=True)
class Principle:
    """One constitutional rule expressed for the SL and RLAIF tasks."""

    name: str
    critique_request: str
    revision_request: str
    comparison_request: str


@dataclass(frozen=True)
class RevisionStep:
    """An auditable critique/revision transition for one sampled principle."""

    index: int
    principle_name: str
    response_before: str
    critique: str
    response_after: str


@dataclass(frozen=True)
class RevisionTrajectory:
    """The original response plus all sequential constitutional revisions."""

    user_prompt: str
    initial_response: str
    steps: tuple[RevisionStep, ...]

    @property
    def final_response(self) -> str:
        if not self.steps:
            return self.initial_response
        return self.steps[-1].response_after


@dataclass(frozen=True)
class SFTExample:
    """A prompt/revision pair used to move the policy on-distribution."""

    prompt: str
    response: str
    source: str
    revision_index: int | None


@dataclass(frozen=True)
class PreferenceExample:
    """A canonical pair and the AI probability that response A is preferred."""

    prompt: str
    response_a: str
    response_b: str
    probability_a_better: float
    principle_name: str
    displayed_order: str


def _validate_probability(value: float, *, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def normalize_binary_logprobs(logprob_a: float, logprob_b: float) -> float:
    """Return exp(log p(A)) / (exp(log p(A)) + exp(log p(B))) stably."""

    if not math.isfinite(logprob_a) or not math.isfinite(logprob_b):
        raise ValueError("binary option log-probabilities must be finite")
    maximum = max(logprob_a, logprob_b)
    weight_a = math.exp(logprob_a - maximum)
    weight_b = math.exp(logprob_b - maximum)
    return weight_a / (weight_a + weight_b)


def clamp_probability(
    probability: float,
    *,
    lower: float = 0.4,
    upper: float = 0.6,
) -> float:
    """Clamp an overconfident label, as in the paper's 40--60 CoT setting."""

    _validate_probability(probability, name="probability")
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("expected 0 <= lower <= upper <= 1")
    return min(max(probability, lower), upper)


def render_conversation(user_prompt: str, assistant_response: str) -> str:
    """Use the paper's simple Human/Assistant transcript convention."""

    return f"Human: {user_prompt.strip()}\n\nAssistant: {assistant_response.strip()}"


def build_initial_response_prompt(user_prompt: str) -> str:
    """Prompt an instruction-following assistant for the initial response."""

    return f"Human: {user_prompt.strip()}\n\nAssistant:"


def build_critique_prompt(
    user_prompt: str,
    response: str,
    principle: Principle,
) -> str:
    """Append a constitutional critique request to the current conversation."""

    return (
        f"{render_conversation(user_prompt, response)}\n\n"
        f"Critique Request: {principle.critique_request.strip()}\n\n"
        "Critique:"
    )


def build_revision_prompt(
    user_prompt: str,
    response: str,
    critique: str,
    principle: Principle,
) -> str:
    """Ask for a replacement response after making the critique explicit."""

    return (
        f"{render_conversation(user_prompt, response)}\n\n"
        f"Critique: {critique.strip()}\n\n"
        f"Revision Request: {principle.revision_request.strip()}\n\n"
        "Revision:"
    )


def constitutional_revision(
    generator: Generator,
    user_prompt: str,
    principles: Sequence[Principle],
    *,
    num_revisions: int = 4,
    rng: random.Random,
) -> RevisionTrajectory:
    """Generate one response and repeatedly critique/revise it.

    The paper independently samples a principle at every revision step.  It
    also uses few-shot examples to stabilize the critique/revision roles;
    production adapters can prepend those examples inside ``generator``.
    """

    if not principles:
        raise ValueError("at least one principle is required")
    if num_revisions <= 0:
        raise ValueError("num_revisions must be positive")

    initial_response = generator(
        build_initial_response_prompt(user_prompt),
        rng.randrange(2**31),
    ).strip()
    if not initial_response:
        raise ValueError("generator returned an empty initial response")

    current_response = initial_response
    steps: list[RevisionStep] = []
    for index in range(1, num_revisions + 1):
        principle = rng.choice(principles)
        critique = generator(
            build_critique_prompt(user_prompt, current_response, principle),
            rng.randrange(2**31),
        ).strip()
        revised = generator(
            build_revision_prompt(
                user_prompt,
                current_response,
                critique,
                principle,
            ),
            rng.randrange(2**31),
        ).strip()
        if not critique or not revised:
            raise ValueError(f"empty critique or revision at step {index}")
        steps.append(
            RevisionStep(
                index=index,
                principle_name=principle.name,
                response_before=current_response,
                critique=critique,
                response_after=revised,
            )
        )
        current_response = revised

    return RevisionTrajectory(
        user_prompt=user_prompt,
        initial_response=initial_response,
        steps=tuple(steps),
    )


def revision_sft_examples(trajectory: RevisionTrajectory) -> tuple[SFTExample, ...]:
    """Keep every revision, matching the paper rather than only the final one."""

    return tuple(
        SFTExample(
            prompt=trajectory.user_prompt,
            response=step.response_after,
            source=f"constitutional:{step.principle_name}",
            revision_index=step.index,
        )
        for step in trajectory.steps
    )


def helpfulness_sft_example(user_prompt: str, response: str) -> SFTExample:
    """Mark a helpful-only sample mixed in to reduce helpfulness regression."""

    if not user_prompt.strip() or not response.strip():
        raise ValueError("helpfulness examples require a prompt and response")
    return SFTExample(
        prompt=user_prompt.strip(),
        response=response.strip(),
        source="helpful_rlhf",
        revision_index=None,
    )


def build_feedback_prompt(
    user_prompt: str,
    displayed_a: str,
    displayed_b: str,
    principle: Principle,
) -> str:
    """Format constitutional feedback as the paper's A/B multiple choice."""

    return (
        "Consider the following conversation between a human and an assistant:\n"
        f"Human: {user_prompt.strip()}\n\n"
        f"{principle.comparison_request.strip()}\n\n"
        "Options:\n"
        f"(A) {displayed_a.strip()}\n"
        f"(B) {displayed_b.strip()}\n\n"
        "The answer is:"
    )


def make_ai_preference(
    option_scorer: OptionScorer,
    user_prompt: str,
    response_a: str,
    response_b: str,
    principles: Sequence[Principle],
    *,
    rng: random.Random,
    label_clamp: tuple[float, float] | None = None,
) -> PreferenceExample:
    """Create a soft RLAIF label while undoing randomized display order."""

    if not principles:
        raise ValueError("at least one principle is required")
    if not response_a.strip() or not response_b.strip():
        raise ValueError("both candidate responses must be non-empty")
    if response_a.strip() == response_b.strip():
        raise ValueError("candidate responses must be different")

    principle = rng.choice(principles)
    swapped = bool(rng.getrandbits(1))
    if swapped:
        displayed_a, displayed_b = response_b, response_a
        displayed_order = "B,A"
    else:
        displayed_a, displayed_b = response_a, response_b
        displayed_order = "A,B"

    feedback_prompt = build_feedback_prompt(
        user_prompt,
        displayed_a,
        displayed_b,
        principle,
    )
    logprob_displayed_a, logprob_displayed_b = option_scorer(feedback_prompt)
    probability_displayed_a = normalize_binary_logprobs(
        logprob_displayed_a,
        logprob_displayed_b,
    )
    probability_original_a = (
        1.0 - probability_displayed_a if swapped else probability_displayed_a
    )
    if label_clamp is not None:
        probability_original_a = clamp_probability(
            probability_original_a,
            lower=label_clamp[0],
            upper=label_clamp[1],
        )

    return PreferenceExample(
        prompt=user_prompt.strip(),
        response_a=response_a.strip(),
        response_b=response_b.strip(),
        probability_a_better=probability_original_a,
        principle_name=principle.name,
        displayed_order=displayed_order,
    )


def soft_preference_loss(
    reward_a: float,
    reward_b: float,
    probability_a_better: float,
) -> float:
    """Soft Bradley--Terry cross-entropy for a scalar preference model.

    If ``q`` is the AI feedback probability and ``d = r_a - r_b``, this is

        -q log sigmoid(d) - (1-q) log sigmoid(-d).
    """

    _validate_probability(probability_a_better, name="probability_a_better")
    difference = reward_a - reward_b
    if not math.isfinite(difference):
        raise ValueError("reward difference must be finite")

    def softplus(value: float) -> float:
        if value > 0.0:
            return value + math.log1p(math.exp(-value))
        return math.log1p(math.exp(value))

    q = probability_a_better
    return q * softplus(-difference) + (1.0 - q) * softplus(difference)


def kl_regularized_sequence_objective(
    *,
    preference_model_score: float,
    policy_token_logprobs: Sequence[float],
    reference_token_logprobs: Sequence[float],
    beta: float,
) -> float:
    """Reward minus the sampled policy/reference log-ratio.

    The CAI paper says the post-labeling PM/RL pipeline follows its earlier
    RLHF work.  This compact objective exposes that interface; a full trainer
    still needs a value model, advantages, PPO clipping, batching, and updates.
    """

    if len(policy_token_logprobs) != len(reference_token_logprobs):
        raise ValueError("policy and reference log-probabilities must align")
    if not policy_token_logprobs:
        raise ValueError("at least one sampled response token is required")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")

    sampled_log_ratio = sum(
        policy_logprob - reference_logprob
        for policy_logprob, reference_logprob in zip(
            policy_token_logprobs,
            reference_token_logprobs,
        )
    )
    return preference_model_score - beta * sampled_log_ratio


TOY_CONSTITUTION = (
    Principle(
        name="avoid-enabling-harm",
        critique_request=(
            "Identify whether the response enables harmful, dangerous, or "
            "illegal action, and explain the specific issue."
        ),
        revision_request=(
            "Rewrite the response so it does not enable harm while remaining "
            "clear, relevant, and as helpful as safely possible."
        ),
        comparison_request=(
            "Which response is less harmful while still engaging with the "
            "user's underlying need?"
        ),
    ),
    Principle(
        name="avoid-evasive-refusal",
        critique_request=(
            "Check whether the response is either unsafe or needlessly "
            "evasive, and identify a constructive safe alternative."
        ),
        revision_request=(
            "Produce a thoughtful, non-evasive response that explains any "
            "boundary and offers a safe alternative."
        ),
        comparison_request=(
            "Choose the response that is safer, more thoughtful, and less "
            "needlessly evasive."
        ),
    ),
)


def _toy_generator(prompt: str, seed: int) -> str:
    """Deterministic stand-in that makes the pipeline runnable, not realistic."""

    del seed
    if prompt.endswith("Critique:"):
        return (
            "The response promises operational help for unauthorized access. "
            "It should set a boundary and redirect to legitimate recovery."
        )
    if prompt.endswith("Revision:"):
        return (
            "I can't help bypass someone else's access controls. If this is "
            "your own account, I can help with the official recovery process."
        )
    return "I can provide instructions for bypassing the access control."


def _toy_option_scorer(prompt: str) -> tuple[float, float]:
    """Assign larger option log-probability to the constructive safe answer."""

    options = prompt.split("Options:\n", maxsplit=1)[1]
    option_a, option_b = options.split("\n(B) ", maxsplit=1)
    option_a = option_a.removeprefix("(A) ")
    option_b = option_b.split("\n\nThe answer is:", maxsplit=1)[0]

    def score(text: str) -> float:
        markers = ("can't help bypass", "official recovery", "safe alternative")
        return -0.2 if any(marker in text.casefold() for marker in markers) else -2.0

    return score(option_a), score(option_b)


def _run_checks() -> None:
    rng = random.Random(7)
    trajectory = constitutional_revision(
        _toy_generator,
        "How can I get into an account I do not own?",
        TOY_CONSTITUTION,
        num_revisions=4,
        rng=rng,
    )
    sft_examples = revision_sft_examples(trajectory)
    assert len(sft_examples) == 4
    assert [example.revision_index for example in sft_examples] == [1, 2, 3, 4]

    preference = make_ai_preference(
        _toy_option_scorer,
        trajectory.user_prompt,
        trajectory.final_response,
        trajectory.initial_response,
        TOY_CONSTITUTION,
        rng=rng,
    )
    assert preference.probability_a_better > 0.5

    aligned_loss = soft_preference_loss(
        reward_a=1.0,
        reward_b=-1.0,
        probability_a_better=preference.probability_a_better,
    )
    reversed_loss = soft_preference_loss(
        reward_a=-1.0,
        reward_b=1.0,
        probability_a_better=preference.probability_a_better,
    )
    assert aligned_loss < reversed_loss

    clamped = clamp_probability(preference.probability_a_better)
    assert 0.4 <= clamped <= 0.6

    objective = kl_regularized_sequence_objective(
        preference_model_score=1.5,
        policy_token_logprobs=(-0.2, -0.5, -0.3),
        reference_token_logprobs=(-0.3, -0.4, -0.6),
        beta=0.02,
    )
    assert math.isclose(objective, 1.494)

    print("All Constitutional AI pipeline checks passed.")
    print(f"Sequential revisions kept for SFT: {len(sft_examples)}")
    print(
        "AI soft preference P(final revision > initial response): "
        f"{preference.probability_a_better:.3f}"
    )
    print(
        "Soft PM loss (aligned / reversed): "
        f"{aligned_loss:.3f} / {reversed_loss:.3f}"
    )
    print(f"40--60 clamped label: {clamped:.3f}")
    print(f"KL-regularized sequence objective: {objective:.3f}")


if __name__ == "__main__":
    _run_checks()
