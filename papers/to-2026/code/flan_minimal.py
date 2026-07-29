#!/usr/bin/env python3
"""Dependency-free miniature of the FLAN 2021 data pipeline.

This script does not train a language model.  It makes the parts that were
novel in FLAN visible and testable:

1. split by task *cluster*, not merely by dataset;
2. sample datasets with examples-proportional mixing and a rate cap;
3. choose a natural-language template for every sampled example;
4. append the legal outputs to classification prompts; and
5. create causal-LM labels whose prompt positions are ignored by the loss.

Run:

    python3 papers/to-2026/code/flan_minimal.py \
        --held-out-cluster nli --samples 6 --seed 7
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence


IGNORE_INDEX = -100
MAX_EXAMPLES_PER_DATASET = 30_000
MIXING_RATE_CAP = 3_000


@dataclass(frozen=True)
class RawExample:
    """One dataset row before it is verbalized as an instruction."""

    fields: Mapping[str, str]
    target: str


@dataclass(frozen=True)
class TaskDataset:
    """The metadata FLAN needs in addition to ordinary input/target pairs."""

    name: str
    cluster: str
    declared_train_size: int
    templates: tuple[str, ...]
    examples: tuple[RawExample, ...]
    options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.declared_train_size <= 0:
            raise ValueError(f"{self.name}: declared_train_size must be positive")
        if not self.templates:
            raise ValueError(f"{self.name}: at least one template is required")
        if not self.examples:
            raise ValueError(f"{self.name}: at least one example is required")
        if self.options and any(
            example.target not in self.options for example in self.examples
        ):
            raise ValueError(f"{self.name}: every target must occur in options")


@dataclass(frozen=True)
class InstructionExample:
    """A row after task metadata has been converted into natural language."""

    dataset: str
    cluster: str
    prompt: str
    target: str


def split_by_cluster(
    datasets: Sequence[TaskDataset],
    held_out_cluster: str,
) -> tuple[list[TaskDataset], list[TaskDataset]]:
    """Return strict train/evaluation splits with no task-family leakage."""

    train = [task for task in datasets if task.cluster != held_out_cluster]
    evaluation = [task for task in datasets if task.cluster == held_out_cluster]
    if not train:
        raise ValueError("the training split is empty")
    if not evaluation:
        raise ValueError(f"unknown held-out cluster: {held_out_cluster!r}")

    train_clusters = {task.cluster for task in train}
    evaluation_clusters = {task.cluster for task in evaluation}
    overlap = train_clusters & evaluation_clusters
    if overlap:
        raise AssertionError(f"task-cluster leakage detected: {sorted(overlap)}")
    return train, evaluation


def mixing_weight(
    task: TaskDataset,
    *,
    dataset_cap: int = MAX_EXAMPLES_PER_DATASET,
    rate_cap: int = MIXING_RATE_CAP,
) -> int:
    """FLAN-style examples-proportional weight for one dataset.

    FLAN kept at most 30k training rows per dataset.  The mixture rate stopped
    growing at 3k examples, so a million-row dataset did not dominate a
    thousand-row dataset by three orders of magnitude.
    """

    usable_examples = min(task.declared_train_size, dataset_cap)
    return min(usable_examples, rate_cap)


def mixture_probabilities(
    datasets: Sequence[TaskDataset],
) -> dict[str, float]:
    """Normalize FLAN-style dataset weights into sampling probabilities."""

    weights = [mixing_weight(task) for task in datasets]
    total = sum(weights)
    if total <= 0:
        raise ValueError("mixture has no positive sampling weight")
    return {
        task.name: weight / total
        for task, weight in zip(datasets, weights)
    }


def render_instruction(
    task: TaskDataset,
    example: RawExample,
    template: str,
) -> InstructionExample:
    """Render a dataset row and optionally expose its legal class strings."""

    try:
        prompt = template.format_map(example.fields).strip()
    except KeyError as error:
        missing = error.args[0]
        raise ValueError(
            f"{task.name}: template references missing field {missing!r}"
        ) from error

    if task.options:
        option_lines = "\n".join(f"- {option}" for option in task.options)
        prompt = f"{prompt}\n\nOPTIONS:\n{option_lines}"

    return InstructionExample(
        dataset=task.name,
        cluster=task.cluster,
        prompt=prompt,
        target=example.target,
    )


def sample_instruction_stream(
    datasets: Sequence[TaskDataset],
    *,
    rng: random.Random,
) -> Iterator[InstructionExample]:
    """Yield an infinite, reproducible multi-task instruction stream."""

    weights = [mixing_weight(task) for task in datasets]
    while True:
        task = rng.choices(datasets, weights=weights, k=1)[0]
        example = rng.choice(task.examples)
        template = rng.choice(task.templates)
        yield render_instruction(task, example, template)


def make_causal_lm_features(
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    separator_id: int,
    eos_id: int,
) -> dict[str, list[int]]:
    """Concatenate prompt/target and mask prompt tokens in the LM loss."""

    input_ids = [
        *prompt_ids,
        separator_id,
        *target_ids,
        eos_id,
    ]
    labels = [
        *([IGNORE_INDEX] * (len(prompt_ids) + 1)),
        *target_ids,
        eos_id,
    ]
    if len(input_ids) != len(labels):
        raise AssertionError("input_ids and labels must have the same length")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def toy_datasets() -> tuple[TaskDataset, ...]:
    """Return tiny rows representing four different task clusters."""

    return (
        TaskDataset(
            name="movie_sentiment",
            cluster="sentiment",
            declared_train_size=1_200,
            templates=(
                "Is this review positive or negative?\nReview: {text}",
                "Classify the sentiment of the following text: {text}",
            ),
            examples=(
                RawExample(
                    {"text": "The pacing is tight and the acting is superb."},
                    "positive",
                ),
                RawExample(
                    {"text": "A dull script wastes a talented cast."},
                    "negative",
                ),
            ),
            options=("positive", "negative"),
        ),
        TaskDataset(
            name="mini_translation",
            cluster="translation",
            declared_train_size=1_000_000,
            templates=(
                "Translate from English to French: {source}",
                "Write this sentence in French.\nEnglish: {source}",
            ),
            examples=(
                RawExample({"source": "The dog runs."}, "Le chien court."),
                RawExample({"source": "Good morning."}, "Bonjour."),
            ),
        ),
        TaskDataset(
            name="tiny_reading_comprehension",
            cluster="reading_comprehension",
            declared_train_size=9_000,
            templates=(
                "Read the passage and answer the question.\n"
                "Passage: {passage}\nQuestion: {question}",
            ),
            examples=(
                RawExample(
                    {
                        "passage": "Ada published notes about the Analytical Engine.",
                        "question": "What did Ada publish notes about?",
                    },
                    "the Analytical Engine",
                ),
            ),
        ),
        TaskDataset(
            name="tiny_rte",
            cluster="nli",
            declared_train_size=2_500,
            templates=(
                "Does the premise entail the hypothesis?\n"
                "Premise: {premise}\nHypothesis: {hypothesis}",
                "Given {premise!r}, is {hypothesis!r} definitely true?",
            ),
            examples=(
                RawExample(
                    {
                        "premise": "A child is playing a violin.",
                        "hypothesis": "A child is making music.",
                    },
                    "yes",
                ),
                RawExample(
                    {
                        "premise": "The shop closes at six.",
                        "hypothesis": "The shop is open at eight.",
                    },
                    "no",
                ),
            ),
            options=("yes", "no"),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out-cluster", default="nli")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    train, evaluation = split_by_cluster(
        toy_datasets(),
        args.held_out_cluster,
    )

    print(f"held-out cluster: {args.held_out_cluster}")
    print("train datasets:   " + ", ".join(task.name for task in train))
    print("evaluation only:  " + ", ".join(task.name for task in evaluation))
    print("\nmixture probabilities:")
    for name, probability in mixture_probabilities(train).items():
        print(f"  {name:30s} {probability:6.2%}")

    print("\nsampled instruction-tuning rows:")
    stream = sample_instruction_stream(train, rng=random.Random(args.seed))
    for index in range(1, args.samples + 1):
        row = next(stream)
        compact_prompt = " ".join(row.prompt.split())
        print(
            f"  {index}. [{row.cluster}/{row.dataset}] "
            f"{compact_prompt} -> {row.target}"
        )

    features = make_causal_lm_features(
        prompt_ids=[101, 102, 103],
        target_ids=[201, 202],
        separator_id=1,
        eos_id=2,
    )
    print("\ncausal-LM target-only masking:")
    print(f"  input_ids = {features['input_ids']}")
    print(f"  labels    = {features['labels']}")


if __name__ == "__main__":
    main()
