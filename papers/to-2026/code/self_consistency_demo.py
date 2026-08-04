"""A dependency-free, model-agnostic Self-Consistency decoding demo.

The paper's method is an inference-time wrapper:

1. sample several chain-of-thought completions independently;
2. parse and normalize the final answer from every completion;
3. return the answer with the largest vote count.

This module deliberately does not call a specific model API. Adapt any local or
remote model to the ``Generator`` callable and pass it to
``run_self_consistency``.

Run:
    python3 papers/to-2026/code/self_consistency_demo.py
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Callable, Sequence


Generator = Callable[[str, float, int], str]
Normalizer = Callable[[str], str]

FINAL_ANSWER_PATTERN = re.compile(
    r"(?:The answer is|Final answer\s*:|答案(?:是|为)\s*[:：]?)"
    r"\s*(?P<answer>[^\n]+)",
    flags=re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"[-+]?\s*\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)


class NoValidAnswerError(ValueError):
    """Raised when no completion contains a usable final answer."""


class VoteTieError(ValueError):
    """Raised when majority voting does not have a unique winner."""


@dataclass(frozen=True)
class ParsedCompletion:
    """One completion after separating its rationale and final answer."""

    rationale: str
    raw_answer: str
    raw_text: str


@dataclass(frozen=True)
class SampledPath:
    """One auditable sample, including parser failures."""

    index: int
    seed: int
    raw_text: str
    rationale: str | None
    raw_answer: str | None
    normalized_answer: str | None
    error: str | None

    @property
    def is_valid(self) -> bool:
        return self.normalized_answer is not None


@dataclass(frozen=True)
class VoteGroup:
    """All valid samples that normalize to the same answer."""

    answer: str
    count: int
    share: float
    sample_indices: tuple[int, ...]


@dataclass(frozen=True)
class SelfConsistencyResult:
    """The winning answer plus enough evidence to inspect the decision."""

    answer: str
    consensus: float
    valid_rate: float
    votes: tuple[VoteGroup, ...]
    samples: tuple[SampledPath, ...]

    @property
    def valid_samples(self) -> int:
        return sum(sample.is_valid for sample in self.samples)


def parse_final_answer(text: str) -> ParsedCompletion:
    """Split a completion at its last explicit final-answer marker.

    Requiring a marker is intentional: taking the last number in an entire
    rationale can silently turn an intermediate value into the prediction.
    """

    raw = text.strip()
    matches = list(FINAL_ANSWER_PATTERN.finditer(raw))
    if not matches:
        raise ValueError(
            "missing final-answer marker; expected 'The answer is', "
            "'Final answer:', or '答案是'"
        )

    match = matches[-1]
    rationale = raw[: match.start()].strip()
    raw_answer = match.group("answer").strip().rstrip("。.")
    if not raw_answer:
        raise ValueError("final answer is empty")
    return ParsedCompletion(
        rationale=rationale,
        raw_answer=raw_answer,
        raw_text=raw,
    )


def normalize_numeric_answer(value: str) -> str:
    """Canonicalize the last number in an already-extracted answer span."""

    matches = NUMBER_PATTERN.findall(value)
    if not matches:
        raise ValueError(f"no numeric answer found in {value!r}")

    token = re.sub(r"[\s$,]", "", matches[-1])
    is_percent = token.endswith("%")
    if is_percent:
        token = token[:-1]
    try:
        number = Decimal(token)
    except InvalidOperation as error:
        raise ValueError(f"invalid numeric answer: {token!r}") from error

    # Decimal('-0').normalize() retains the sign; answers should not.
    if number == 0:
        number = Decimal(0)
    canonical = format(number.normalize(), "f")
    return f"{canonical}%" if is_percent else canonical


def normalize_choice_answer(value: str) -> str:
    """Canonicalize answers such as ``B``, ``(b)``, and ``option B``."""

    match = re.fullmatch(
        r"\s*(?:option\s*)?[\(\[]?([A-Ea-e])[\)\].]?\s*",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"no A-E choice found in {value!r}")
    return match.group(1).upper()


def normalize_text_answer(value: str) -> str:
    """A conservative normalizer for short, fixed-set textual answers."""

    text = value.casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("text answer is empty after normalization")
    return normalized


def sample_paths(
    generator: Generator,
    prompt: str,
    normalizer: Normalizer,
    *,
    num_samples: int,
    temperature: float = 0.7,
    base_seed: int = 0,
) -> tuple[SampledPath, ...]:
    """Generate independent paths and retain both valid and invalid samples.

    The loop is serial only to keep this example dependency-free. Production
    callers can issue the same independent requests concurrently or use an API's
    multi-completion parameter, while preserving one record per returned path.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive to sample diverse paths")

    samples: list[SampledPath] = []
    for index in range(num_samples):
        seed = base_seed + index
        raw_text = generator(prompt, temperature, seed)
        try:
            parsed = parse_final_answer(raw_text)
            normalized = normalizer(parsed.raw_answer)
        except ValueError as error:
            samples.append(
                SampledPath(
                    index=index,
                    seed=seed,
                    raw_text=raw_text,
                    rationale=None,
                    raw_answer=None,
                    normalized_answer=None,
                    error=str(error),
                )
            )
            continue

        samples.append(
            SampledPath(
                index=index,
                seed=seed,
                raw_text=raw_text,
                rationale=parsed.rationale,
                raw_answer=parsed.raw_answer,
                normalized_answer=normalized,
                error=None,
            )
        )
    return tuple(samples)


def majority_vote(
    samples: Sequence[SampledPath],
    *,
    min_valid_samples: int = 1,
) -> SelfConsistencyResult:
    """Aggregate normalized answers with an unweighted majority vote.

    A tie raises instead of silently selecting the first answer. In an online
    system, the caller can sample more paths, abstain, or invoke a verifier.
    """

    if min_valid_samples <= 0:
        raise ValueError("min_valid_samples must be positive")
    if not samples:
        raise NoValidAnswerError("no samples were provided")

    valid = [sample for sample in samples if sample.normalized_answer is not None]
    if len(valid) < min_valid_samples:
        raise NoValidAnswerError(
            f"needed {min_valid_samples} valid samples, got {len(valid)}"
        )

    counts = Counter(sample.normalized_answer for sample in valid)
    top_count = max(counts.values())
    winners = sorted(answer for answer, count in counts.items() if count == top_count)
    if len(winners) != 1:
        raise VoteTieError(f"vote tie between {winners}; sample more or abstain")

    indices: dict[str, list[int]] = defaultdict(list)
    for sample in valid:
        assert sample.normalized_answer is not None
        indices[sample.normalized_answer].append(sample.index)

    groups = tuple(
        VoteGroup(
            answer=answer,
            count=count,
            share=count / len(valid),
            sample_indices=tuple(indices[answer]),
        )
        for answer, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    winner = winners[0]
    return SelfConsistencyResult(
        answer=winner,
        consensus=top_count / len(valid),
        valid_rate=len(valid) / len(samples),
        votes=groups,
        samples=tuple(samples),
    )


def run_self_consistency(
    generator: Generator,
    prompt: str,
    normalizer: Normalizer,
    *,
    num_samples: int = 10,
    temperature: float = 0.7,
    base_seed: int = 0,
    min_valid_samples: int = 1,
) -> SelfConsistencyResult:
    """Run the complete sample -> parse -> normalize -> vote pipeline."""

    samples = sample_paths(
        generator,
        prompt,
        normalizer,
        num_samples=num_samples,
        temperature=temperature,
        base_seed=base_seed,
    )
    return majority_vote(samples, min_valid_samples=min_valid_samples)


class DemoGenerator:
    """Deterministic stand-in that makes the example runnable without an API."""

    def __init__(self, completions: Sequence[str]) -> None:
        if not completions:
            raise ValueError("completions must not be empty")
        self._completions = tuple(completions)

    def __call__(self, prompt: str, temperature: float, seed: int) -> str:
        del prompt, temperature
        return self._completions[seed % len(self._completions)]


def _self_test() -> None:
    assert normalize_numeric_answer("$1,392.00") == "1392"
    assert normalize_numeric_answer("9.0") == "9"
    assert normalize_numeric_answer("-0") == "0"
    assert normalize_choice_answer("option (b)") == "B"
    assert normalize_text_answer("New York.") == "new york"

    completions = (
        "23 - 20 = 3, then 3 + 6 = 9. The answer is 9.",
        "Add before subtracting: 23 + 6 - 20 = 9. Final answer: 9.0",
        "I accidentally ignored the sale. The answer is 29.",
        "Track the inventory in two steps. 答案是：$9.00。",
        "This malformed output deliberately has no answer marker.",
    )
    result = run_self_consistency(
        DemoGenerator(completions),
        "A word problem with a few-shot CoT prompt",
        normalize_numeric_answer,
        num_samples=5,
        temperature=0.7,
        min_valid_samples=3,
    )
    assert result.answer == "9"
    assert result.consensus == 0.75
    assert result.valid_rate == 0.8
    assert result.votes[0].sample_indices == (0, 1, 3)

    tied_samples = sample_paths(
        DemoGenerator(("The answer is 1.", "The answer is 2.")),
        "prompt",
        normalize_numeric_answer,
        num_samples=2,
        temperature=0.7,
    )
    try:
        majority_vote(tied_samples)
    except VoteTieError:
        pass
    else:
        raise AssertionError("a tied vote must not select a winner silently")


def main() -> None:
    _self_test()
    completions = (
        "Sell 20 first: 23 - 20 = 3; receive 6: 3 + 6 = 9. Answer is 9.",
        "Net change is -20 + 6 = -14; 23 - 14 = 9. The answer is 9.",
        "I forgot the final delivery. Final answer: 3.",
        "Compute in event order. 答案是 9。",
        "There are 23 + 6 items after the delivery. The answer is 29.",
    )
    result = run_self_consistency(
        DemoGenerator(completions),
        "Q: Lina has 23 apples, sells 20, then receives 6. How many remain?\nA:",
        normalize_numeric_answer,
        num_samples=5,
        temperature=0.7,
        min_valid_samples=3,
    )

    print(f"winner:     {result.answer}")
    print(f"consensus:  {result.consensus:.1%} of valid samples")
    print(f"valid rate: {result.valid_rate:.1%}")
    print("votes:")
    for group in result.votes:
        print(
            f"  answer={group.answer!r}, count={group.count}, "
            f"samples={group.sample_indices}"
        )
    print("All self-tests passed.")


if __name__ == "__main__":
    main()
