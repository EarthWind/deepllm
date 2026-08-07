#!/usr/bin/env python3
"""Dependency-free miniature of the Self-Instruct data pipeline.

The ACL 2023 paper grows instruction-tuning data in four stages:

1. sample tasks from a pool and ask a language model for new instructions;
2. identify whether each new task is a finite-label classification task;
3. generate input/output instances (output-first for classification, input-first
   for other tasks); and
4. reject invalid, conflicting, unsupported, or overly similar generations.

This file reproduces those algorithm-specific mechanics without an API key or
model download. ``ScriptedBackend`` is a deterministic stand-in for the four
language-model calls. Replace it with an adapter around a real model while
keeping the filtering, audit log, and SFT serialization model-independent.

Run:

    python3 papers/to-2026/code/self_instruct_minimal.py
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
UNSUPPORTED_WORDS = frozenset(
    {
        "image",
        "images",
        "graph",
        "graphs",
        "picture",
        "pictures",
        "file",
        "files",
        "map",
        "maps",
        "draw",
        "plot",
        "go to",
    }
)


@dataclass(frozen=True)
class Instance:
    input: str
    output: str


@dataclass(frozen=True)
class Task:
    instruction: str
    instances: tuple[Instance, ...]
    is_classification: bool
    source: str


@dataclass(frozen=True)
class FilterDecision:
    accepted: bool
    reason: str
    normalized_instruction: str
    max_similarity: float = 0.0
    most_similar_instruction: str | None = None


@dataclass(frozen=True)
class BootstrapResult:
    pool: tuple[Task, ...]
    generated: tuple[Task, ...]
    rejection_counts: dict[str, int]
    audit_log: tuple[FilterDecision, ...]


class GenerationBackend(Protocol):
    """The four model-assisted decisions used by Self-Instruct."""

    def generate_instructions(
        self,
        prompt_examples: Sequence[Task],
        round_index: int,
    ) -> Sequence[str]: ...

    def is_classification(self, instruction: str) -> bool: ...

    def generate_instances(
        self,
        instruction: str,
        *,
        output_first: bool,
    ) -> Sequence[Instance]: ...


def normalize_space(text: str) -> str:
    return " ".join(text.split()).strip()


def tokenize(text: str) -> tuple[str, ...]:
    """Small case-insensitive tokenizer for the dependency-free ROUGE demo."""

    return tuple(token.casefold() for token in TOKEN_RE.findall(text))


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Return longest-common-subsequence length using O(min(m, n)) memory."""

    if len(left) < len(right):
        short, long = left, right
    else:
        short, long = right, left
    previous = [0] * (len(short) + 1)
    for long_token in long:
        current = [0]
        for index, short_token in enumerate(short, start=1):
            if long_token == short_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    """Compute token-level ROUGE-L F1 without stemming.

    The released Self-Instruct code uses Google's ``rouge_score`` package. This
    implementation keeps the same LCS/F1 idea while using a tiny local tokenizer,
    so scores can differ slightly on punctuation-heavy text.
    """

    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = lcs_length(candidate_tokens, reference_tokens)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class InstructionFilter:
    """Paper-inspired instruction validity and novelty checks."""

    def __init__(
        self,
        *,
        max_rouge_l: float = 0.7,
        min_words_exclusive: int = 3,
        max_words: int = 150,
    ) -> None:
        if not 0.0 <= max_rouge_l <= 1.0:
            raise ValueError("max_rouge_l must be in [0, 1]")
        self.max_rouge_l = max_rouge_l
        self.min_words_exclusive = min_words_exclusive
        self.max_words = max_words

    def evaluate(
        self,
        candidate: str,
        existing_instructions: Sequence[str],
    ) -> FilterDecision:
        instruction = normalize_space(candidate)
        words = instruction.split()
        if not instruction:
            return FilterDecision(False, "empty", instruction)
        if len(words) <= self.min_words_exclusive:
            return FilterDecision(False, "too_short", instruction)
        if len(words) > self.max_words:
            return FilterDecision(False, "too_long", instruction)
        if instruction[0] in string.punctuation:
            return FilterDecision(False, "starts_with_punctuation", instruction)
        if not instruction[0].isascii():
            return FilterDecision(False, "non_ascii_start", instruction)
        if instruction.casefold().startswith("write a program"):
            return FilterDecision(False, "ambiguous_program_request", instruction)

        lowered = instruction.casefold()
        for blocked in UNSUPPORTED_WORDS:
            pattern = rf"\b{re.escape(blocked)}\b"
            if re.search(pattern, lowered):
                return FilterDecision(False, "unsupported_modality", instruction)

        similarities = [
            (rouge_l_f1(instruction, existing), existing)
            for existing in existing_instructions
        ]
        max_similarity, closest = max(similarities, default=(0.0, None))
        # The released code rejects scores strictly greater than 0.7.
        if max_similarity > self.max_rouge_l:
            return FilterDecision(
                False,
                "too_similar",
                instruction,
                max_similarity,
                closest,
            )
        return FilterDecision(
            True,
            "accepted",
            instruction,
            max_similarity,
            closest,
        )


def filter_instances(instances: Sequence[Instance]) -> tuple[Instance, ...]:
    """Apply the released pipeline's central instance-consistency checks."""

    cleaned: list[Instance] = []
    for instance in instances:
        item = Instance(instance.input.strip(), instance.output.strip())
        if not item.output or item.input == item.output:
            continue
        if item.input.endswith(":") or item.output.endswith(":"):
            continue
        cleaned.append(item)

    # One non-empty input paired with conflicting labels makes the task batch
    # ambiguous. The released implementation discards the entire batch.
    outputs_by_input: dict[str, set[str]] = {}
    for item in cleaned:
        if item.input:
            outputs_by_input.setdefault(item.input, set()).add(item.output)
    if any(len(outputs) > 1 for outputs in outputs_by_input.values()):
        return ()

    unique: list[Instance] = []
    seen: set[tuple[str, str]] = set()
    for item in cleaned:
        key = (item.input, item.output)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return tuple(unique[:5])  # The released pipeline keeps at most five/task.


def sample_prompt_tasks(
    pool: Sequence[Task],
    *,
    rng: random.Random,
    prompt_size: int = 8,
    generated_quota: int = 2,
) -> tuple[Task, ...]:
    """Sample up to two generated tasks and fill the prompt with seed tasks."""

    seeds = [task for task in pool if task.source == "seed"]
    generated = [task for task in pool if task.source == "generated"]
    selected_generated = rng.sample(
        generated,
        min(generated_quota, len(generated)),
    )
    seed_count = prompt_size - len(selected_generated)
    if len(seeds) < seed_count:
        raise ValueError("not enough seed tasks to build the instruction prompt")
    selected = selected_generated + rng.sample(seeds, seed_count)
    rng.shuffle(selected)
    return tuple(selected)


class SelfInstructPipeline:
    def __init__(
        self,
        backend: GenerationBackend,
        *,
        instruction_filter: InstructionFilter | None = None,
        random_seed: int = 42,
    ) -> None:
        self.backend = backend
        self.instruction_filter = instruction_filter or InstructionFilter()
        self.rng = random.Random(random_seed)

    def bootstrap(
        self,
        seed_tasks: Sequence[Task],
        *,
        rounds: int,
    ) -> BootstrapResult:
        if rounds <= 0:
            raise ValueError("rounds must be positive")
        if any(task.source != "seed" for task in seed_tasks):
            raise ValueError("initial pool must contain only seed tasks")

        pool = list(seed_tasks)
        generated: list[Task] = []
        rejection_counts: Counter[str] = Counter()
        audit_log: list[FilterDecision] = []

        for round_index in range(rounds):
            prompt_tasks = sample_prompt_tasks(pool, rng=self.rng)
            candidates = self.backend.generate_instructions(prompt_tasks, round_index)
            for candidate in candidates:
                decision = self.instruction_filter.evaluate(
                    candidate,
                    [task.instruction for task in pool],
                )
                audit_log.append(decision)
                if not decision.accepted:
                    rejection_counts[decision.reason] += 1
                    continue

                is_classification = self.backend.is_classification(
                    decision.normalized_instruction
                )
                raw_instances = self.backend.generate_instances(
                    decision.normalized_instruction,
                    output_first=is_classification,
                )
                instances = filter_instances(raw_instances)
                if not instances:
                    rejection_counts["no_valid_instances"] += 1
                    continue

                task = Task(
                    instruction=decision.normalized_instruction,
                    instances=instances,
                    is_classification=is_classification,
                    source="generated",
                )
                pool.append(task)
                generated.append(task)

        return BootstrapResult(
            pool=tuple(pool),
            generated=tuple(generated),
            rejection_counts=dict(sorted(rejection_counts.items())),
            audit_log=tuple(audit_log),
        )


PROMPT_TEMPLATES_WITH_INPUT = (
    "{instruction}\nInput: {input}\nOutput:",
    "Task: {instruction}\n\n{input}\n\nOutput:",
)
PROMPT_TEMPLATES_WITHOUT_INPUT = (
    "{instruction}\nOutput:",
    "Task: {instruction}\n\n",
)


def serialize_for_sft(task: Task) -> tuple[dict[str, str], ...]:
    """Create prompt/completion pairs using multiple textual templates."""

    rows: list[dict[str, str]] = []
    for index, instance in enumerate(task.instances):
        templates = (
            PROMPT_TEMPLATES_WITH_INPUT
            if instance.input
            else PROMPT_TEMPLATES_WITHOUT_INPUT
        )
        prompt = templates[index % len(templates)].format(
            instruction=task.instruction,
            input=instance.input,
        )
        rows.append(
            {
                "prompt": prompt,
                "completion": f" {instance.output}<|endoftext|>",
            }
        )
    return tuple(rows)


class ScriptedBackend:
    """Deterministic model substitute used by the executable demo."""

    _BATCHES = (
        (
            "Draft a polite cancellation email using the supplied reason and date.",
            "Draw a map showing the fastest route between two landmarks.",
            "Sort each product review into Positive, Neutral, or Negative.",
            "Summarize the following paragraph in one sentence.",
        ),
        (
            "Convert meeting notes into action items with owners and deadlines.",
            "Draft a polite cancellation email using the supplied reason and date.",
            "Hi.",
            "Write a program that returns the square of a number.",
        ),
    )

    def generate_instructions(
        self,
        _prompt_examples: Sequence[Task],
        round_index: int,
    ) -> Sequence[str]:
        return self._BATCHES[round_index % len(self._BATCHES)]

    def is_classification(self, instruction: str) -> bool:
        return "positive, neutral, or negative" in instruction.casefold()

    def generate_instances(
        self,
        instruction: str,
        *,
        output_first: bool,
    ) -> Sequence[Instance]:
        lowered = instruction.casefold()
        if output_first:
            # Conceptually choose each label first, then generate an input for it.
            return (
                Instance("The battery lasts all day and charges quickly.", "Positive"),
                Instance("It works as described, but nothing stands out.", "Neutral"),
                Instance("The lid broke after two uses.", "Negative"),
            )
        if lowered.startswith("draft a polite cancellation"):
            return (
                Instance(
                    "Reason: schedule conflict; Date: Friday's workshop",
                    "Hello, I need to cancel my place at Friday's workshop due "
                    "to a schedule conflict. I apologize for the inconvenience.",
                ),
            )
        if lowered.startswith("convert meeting notes"):
            return (
                Instance(
                    "Mina will send the draft by Tuesday. Lee will review it Friday.",
                    "- Mina — send draft — Tuesday\n- Lee — review draft — Friday",
                ),
            )
        return ()


def seed_tasks() -> tuple[Task, ...]:
    instructions = (
        "Summarize the following paragraph in one sentence.",
        "Translate the supplied sentence from English to French.",
        "Classify the news headline as Sports, Business, or Politics.",
        "Rewrite the message in a more professional tone.",
        "Extract every person's name from the passage.",
        "Explain the given scientific concept to a ten-year-old.",
        "Generate three interview questions for the specified role.",
        "Determine whether the premise entails the hypothesis.",
    )
    return tuple(
        Task(
            instruction=instruction,
            instances=(Instance("seed input", "seed output"),),
            is_classification="classify" in instruction.casefold()
            or "whether" in instruction.casefold(),
            source="seed",
        )
        for instruction in instructions
    )


def demo() -> BootstrapResult:
    pipeline = SelfInstructPipeline(ScriptedBackend())
    return pipeline.bootstrap(seed_tasks(), rounds=2)


def _self_check() -> None:
    assert rouge_l_f1("a b c", "a b c") == 1.0
    assert filter_instances((Instance("same", "same"),)) == ()
    assert (
        filter_instances(
            (
                Instance("same input", "Positive"),
                Instance("same input", "Negative"),
            )
        )
        == ()
    )

    result = demo()
    assert len(result.generated) == 3
    assert result.rejection_counts == {
        "ambiguous_program_request": 1,
        "too_short": 1,
        "too_similar": 2,
        "unsupported_modality": 1,
    }
    classification = next(task for task in result.generated if task.is_classification)
    assert {item.output for item in classification.instances} == {
        "Positive",
        "Neutral",
        "Negative",
    }


if __name__ == "__main__":
    _self_check()
    outcome = demo()
    print("Accepted generated tasks:")
    for task in outcome.generated:
        mode = "output-first" if task.is_classification else "input-first"
        print(f"- [{mode}] {task.instruction} ({len(task.instances)} instances)")
    print("Rejections:", outcome.rejection_counts)
    training_rows = tuple(
        row for task in outcome.generated for row in serialize_for_sft(task)
    )
    print(f"SFT rows: {len(training_rows)}")
    print("First prompt:")
    print(training_rows[0]["prompt"])
    print("First completion:")
    print(training_rows[0]["completion"])
