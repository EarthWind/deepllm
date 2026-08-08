#!/usr/bin/env python3
"""Dependency-free miniature of process-supervised reward modeling (PRM).

This module isolates the algorithm-specific pieces from *Let's Verify Step by
Step* (Lightman et al., 2023) without downloading a language model:

1. flatten the public PRM800K JSONL schema into prefix/step/label examples;
2. compute the three-way step classification loss (-1 / 0 / +1);
3. turn step logits into correctness probabilities;
4. aggregate a whole solution in log-space and perform best-of-N selection;
5. select "convincing wrong-answer" samples for active learning.

It is not a reproduction of the paper's GPT-4-based reward models.  A real PRM
replaces the toy logits with one LM forward pass over the problem and solution,
and emits a label prediction at every step boundary.

Run:

    python3 papers/to-2026/code/prm_minimal.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


NEGATIVE = -1
NEUTRAL = 0
POSITIVE = 1
RATINGS = (NEGATIVE, NEUTRAL, POSITIVE)
RATING_TO_INDEX = {rating: index for index, rating in enumerate(RATINGS)}


def logsumexp(values: Sequence[float]) -> float:
    """Numerically stable ``log(sum(exp(values)))``."""

    if not values:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(values)
    if math.isinf(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def log_softmax(logits: Sequence[float]) -> tuple[float, ...]:
    if len(logits) != len(RATINGS):
        raise ValueError("a PRM step needs logits for [-1, 0, +1]")
    normalizer = logsumexp(logits)
    return tuple(value - normalizer for value in logits)


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    return tuple(math.exp(value) for value in log_softmax(logits))


def step_cross_entropy(logits: Sequence[float], rating: int) -> float:
    """Negative log-likelihood for one human step label."""

    if rating not in RATING_TO_INDEX:
        raise ValueError(f"unknown rating {rating}; expected -1, 0, or +1")
    return -log_softmax(logits)[RATING_TO_INDEX[rating]]


def mean_step_loss(
    logits_batch: Sequence[Sequence[float]],
    ratings: Sequence[int],
) -> float:
    """Every labelled boundary contributes one classification target."""

    if len(logits_batch) != len(ratings):
        raise ValueError("logits and ratings must have the same batch size")
    if not ratings:
        raise ValueError("a training batch must contain at least one step")
    return sum(
        step_cross_entropy(logits, rating)
        for logits, rating in zip(logits_batch, ratings)
    ) / len(ratings)


@dataclass(frozen=True)
class StepTrainingExample:
    """One candidate step conditioned on its problem and accepted prefix."""

    problem: str
    prefix_steps: tuple[str, ...]
    candidate_step: str
    rating: int

    @property
    def model_text(self) -> str:
        prefix = "\n".join(self.prefix_steps)
        if prefix:
            return f"Problem: {self.problem}\n\n{prefix}\n{self.candidate_step}"
        return f"Problem: {self.problem}\n\n{self.candidate_step}"


def _chosen_text(step: Mapping[str, Any]) -> tuple[str | None, int | None]:
    """Return the trajectory continuation encoded by one PRM800K step."""

    chosen = step.get("chosen_completion")
    completions = step.get("completions", [])
    if chosen is not None:
        completion = completions[chosen]
        return str(completion["text"]), int(completion["rating"])

    # Phase 1 allowed a labeler to write a positive continuation when every
    # sampled candidate was negative.  Public records store that text here.
    human_completion = step.get("human_completion")
    if human_completion:
        if isinstance(human_completion, Mapping):
            text = human_completion.get("text")
        else:
            text = human_completion
        return str(text), POSITIVE
    return None, None


def flatten_prm800k_record(
    record: Mapping[str, Any],
    *,
    stop_after_first_negative: bool = True,
) -> tuple[StepTrainingExample, ...]:
    """Convert one public PRM800K record into step-classification examples.

    Every candidate completion is conditioned on the same accepted prefix.  The
    prefix then advances only through ``chosen_completion`` (or a phase-1 human
    completion).  Phase 2 normally has no continuation after the first error.
    """

    problem = str(record["question"]["problem"])
    labelled_steps = record["label"]["steps"]
    prefix: list[str] = []
    examples: list[StepTrainingExample] = []

    for step in labelled_steps:
        for completion in step.get("completions", []):
            rating = int(completion["rating"])
            if rating not in RATING_TO_INDEX:
                raise ValueError(f"invalid PRM800K rating: {rating}")
            if completion.get("flagged"):
                continue
            examples.append(
                StepTrainingExample(
                    problem=problem,
                    prefix_steps=tuple(prefix),
                    candidate_step=str(completion["text"]),
                    rating=rating,
                )
            )

        chosen_text, chosen_rating = _chosen_text(step)
        if chosen_text is None:
            break
        prefix.append(chosen_text)
        if stop_after_first_negative and chosen_rating == NEGATIVE:
            break

    return tuple(examples)


def correctness_probability(
    logits: Sequence[float],
    *,
    neutral_is_correct: bool = True,
) -> float:
    """Map [-1, 0, +1] logits to the paper's step-level score.

    The best paper configuration treats neutral as positive, so its mass is
    added to the positive class.  Set ``neutral_is_correct=False`` to reproduce
    the alternate convention in Appendix F.
    """

    negative, neutral, positive = softmax(logits)
    del negative
    return positive + neutral if neutral_is_correct else positive


def solution_log_score(
    step_logits: Sequence[Sequence[float]],
    *,
    reduction: str = "product",
    neutral_is_correct: bool = True,
    epsilon: float = 1e-12,
) -> float:
    """Reduce step probabilities to one ranking score.

    ``product`` is the paper's default and is implemented as a sum of logs to
    avoid underflow. ``minimum`` is the paper's main alternative.  ``geomean``
    is included as an explicit length-normalized diagnostic, not as a reported
    method from the paper.
    """

    if not step_logits:
        raise ValueError("a solution must contain at least one scored step")
    probabilities = [
        max(
            correctness_probability(
                logits,
                neutral_is_correct=neutral_is_correct,
            ),
            epsilon,
        )
        for logits in step_logits
    ]

    if reduction == "minimum":
        return math.log(min(probabilities))
    log_product = sum(math.log(probability) for probability in probabilities)
    if reduction == "product":
        return log_product
    if reduction == "geomean":
        return log_product / len(probabilities)
    raise ValueError("reduction must be 'product', 'minimum', or 'geomean'")


@dataclass(frozen=True)
class CandidateSolution:
    name: str
    final_answer: str
    answer_is_correct: bool
    step_logits: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ScoredSolution:
    candidate: CandidateSolution
    log_score: float

    @property
    def score(self) -> float:
        return math.exp(self.log_score)


def rank_solutions(
    candidates: Iterable[CandidateSolution],
    *,
    reduction: str = "product",
    neutral_is_correct: bool = True,
) -> tuple[ScoredSolution, ...]:
    """PRM best-of-N: score every sampled trace and rank descending."""

    scored = [
        ScoredSolution(
            candidate=candidate,
            log_score=solution_log_score(
                candidate.step_logits,
                reduction=reduction,
                neutral_is_correct=neutral_is_correct,
            ),
        )
        for candidate in candidates
    ]
    if not scored:
        raise ValueError("best-of-N needs at least one candidate")
    return tuple(
        sorted(scored, key=lambda item: (-item.log_score, item.candidate.name))
    )


@dataclass(frozen=True)
class PoolSample:
    sample_id: str
    answer_is_correct: bool
    selector_log_score: float


def select_convincing_samples(
    pool: Sequence[PoolSample],
    *,
    count: int,
    wrong_fraction: float = 0.8,
) -> tuple[PoolSample, ...]:
    """Small-scale active-learning policy used in the paper's ablation.

    Select high-scoring wrong-answer traces first (the current PRM is known to
    miss at least one step), then high-scoring samples from the remainder.
    """

    if not 0.0 <= wrong_fraction <= 1.0:
        raise ValueError("wrong_fraction must lie in [0, 1]")
    if not 0 < count <= len(pool):
        raise ValueError("count must be in [1, len(pool)]")

    ranked = sorted(pool, key=lambda item: (-item.selector_log_score, item.sample_id))
    wrong_target = min(int(count * wrong_fraction), sum(not x.answer_is_correct for x in ranked))
    chosen = [x for x in ranked if not x.answer_is_correct][:wrong_target]
    chosen_ids = {x.sample_id for x in chosen}
    chosen.extend(x for x in ranked if x.sample_id not in chosen_ids)
    return tuple(chosen[:count])


def _logits_for_correctness(probability: float) -> tuple[float, float, float]:
    """Build demo logits with no neutral mass and desired p(positive)."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    return (math.log(1.0 - probability), float("-inf"), math.log(probability))


def demo() -> None:
    record = {
        "question": {"problem": "If 2x + 3 = 11, find x."},
        "label": {
            "steps": [
                {
                    "completions": [
                        {"text": "Subtract 3: 2x = 8.", "rating": 1, "flagged": None},
                        {"text": "Subtract 3: 2x = 14.", "rating": -1, "flagged": None},
                    ],
                    "chosen_completion": 0,
                    "human_completion": None,
                },
                {
                    "completions": [
                        {"text": "Divide by 2: x = 4.", "rating": 1, "flagged": None}
                    ],
                    "chosen_completion": 0,
                    "human_completion": None,
                },
            ]
        },
    }
    examples = flatten_prm800k_record(record)
    print("flattened step labels:", [example.rating for example in examples])

    candidates = (
        CandidateSolution(
            name="lucky-final-answer",
            final_answer="4",
            answer_is_correct=True,
            step_logits=tuple(map(_logits_for_correctness, (0.97, 0.08))),
        ),
        CandidateSolution(
            name="sound-derivation",
            final_answer="4",
            answer_is_correct=True,
            step_logits=tuple(map(_logits_for_correctness, (0.91, 0.88, 0.93))),
        ),
        CandidateSolution(
            name="confident-wrong-answer",
            final_answer="7",
            answer_is_correct=False,
            step_logits=tuple(map(_logits_for_correctness, (0.95, 0.90, 0.12))),
        ),
    )
    ranking = rank_solutions(candidates)
    print("\nPRM best-of-N ranking (product):")
    for rank, item in enumerate(ranking, start=1):
        print(
            f"  {rank}. {item.candidate.name:<25} "
            f"score={item.score:.4f} answer={item.candidate.final_answer}"
        )

    pool = (
        PoolSample("wrong-obvious", False, math.log(0.10)),
        PoolSample("wrong-convincing", False, math.log(0.91)),
        PoolSample("wrong-medium", False, math.log(0.60)),
        PoolSample("right-high", True, math.log(0.95)),
        PoolSample("right-medium", True, math.log(0.70)),
    )
    selected = select_convincing_samples(pool, count=3, wrong_fraction=2 / 3)
    print("\nactive-learning selection:", [item.sample_id for item in selected])

    # Lightweight regression checks for the easy-to-break details.
    assert [example.rating for example in examples] == [1, -1, 1]
    assert ranking[0].candidate.name == "sound-derivation"
    assert selected[0].sample_id == "wrong-convincing"
    assert selected[1].sample_id == "wrong-medium"
    assert math.isclose(
        solution_log_score(
            tuple(map(_logits_for_correctness, (0.8, 0.5))),
            reduction="product",
        ),
        math.log(0.4),
    )
    print("\nself-checks passed")


if __name__ == "__main__":
    demo()
