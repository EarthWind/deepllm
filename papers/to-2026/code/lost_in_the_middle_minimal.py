#!/usr/bin/env python3
"""A zero-dependency teaching harness for *Lost in the Middle*.

This file does not reproduce the paper's API calls or ship its datasets.  It
implements the experimental skeleton that matters:

1. keep a question and its distractors fixed;
2. move the one relevant document / key-value pair through the context;
3. build the original-style and query-aware prompts;
4. score every position separately instead of reporting one average;
5. diagnose the best-to-worst gap and the edge-to-middle penalty.

The ``EdgeBiasedMockModel`` at the bottom is deliberately synthetic.  It only
makes the harness runnable without a model/API and must never be reported as a
paper result.  Replace it with a real ``Callable[[str], str]`` for experiments.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import string
import uuid
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Document:
    title: str
    text: str
    score: float = 0.0
    is_gold: bool = False


@dataclass(frozen=True)
class QACase:
    question: str
    answers: tuple[str, ...]
    gold: Document
    distractors: tuple[Document, ...]


@dataclass(frozen=True)
class KVCase:
    records: tuple[tuple[str, str], ...]
    key: str
    value: str
    gold_index: int


@dataclass(frozen=True)
class Prediction:
    position: int
    expected: tuple[str, ...]
    output: str
    correct: bool


def place_gold_document(case: QACase, total_documents: int, gold_index: int) -> list[Document]:
    """Return the same distractors with the gold document inserted at one index."""
    if total_documents < 2:
        raise ValueError("total_documents must be at least 2")
    if not 0 <= gold_index < total_documents:
        raise ValueError("gold_index is outside the context")
    if len(case.distractors) < total_documents - 1:
        raise ValueError("not enough distractors")

    # A production replica should preserve the retriever's deterministic order.
    documents = list(case.distractors[: total_documents - 1])
    documents.insert(gold_index, case.gold)
    return documents


def format_documents(documents: Sequence[Document]) -> str:
    return "\n".join(
        f"Document [{index}](Title: {document.title}) {document.text}"
        for index, document in enumerate(documents, start=1)
    )


def build_qa_prompt(
    question: str,
    documents: Sequence[Document],
    *,
    query_aware: bool = False,
    mention_random_order: bool = False,
) -> str:
    """Build a paper-style multi-document QA prompt.

    Query-aware contextualization repeats the question before and after the
    documents.  The paper found this highly effective for synthetic KV lookup,
    but not a general cure for multi-document QA.
    """
    if not question.strip() or not documents:
        raise ValueError("question and documents must be non-empty")
    order_note = " The search results are ordered randomly." if mention_random_order else ""
    instruction = (
        "Write a high-quality answer for the given question using only the "
        "provided search results (some of which might be irrelevant)."
        f"{order_note}"
    )
    packed = format_documents(documents)
    if query_aware:
        return f"{instruction}\nQuestion: {question}\n{packed}\nQuestion: {question}\nAnswer:"
    return f"{instruction}\n\n{packed}\n\nQuestion: {question}\nAnswer:"


def normalize_answer(text: str) -> str:
    """SQuAD-like normalization used before substring exact match."""
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_any_answer(output: str, answers: Iterable[str]) -> bool:
    """Approximate the paper's 'answer appears in output' accuracy criterion."""
    normalized_output = normalize_answer(output)
    return any(
        normalized_answer and normalized_answer in normalized_output
        for answer in answers
        if (normalized_answer := normalize_answer(answer))
    )


def evaluate_qa_positions(
    case: QACase,
    total_documents: int,
    positions: Sequence[int],
    predict: Callable[[str], str],
    *,
    query_aware: bool = False,
) -> list[Prediction]:
    results: list[Prediction] = []
    for position in positions:
        documents = place_gold_document(case, total_documents, position)
        prompt = build_qa_prompt(case.question, documents, query_aware=query_aware)
        output = predict(prompt)
        results.append(
            Prediction(
                position=position,
                expected=case.answers,
                output=output,
                correct=contains_any_answer(output, case.answers),
            )
        )
    return results


def _uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128)))


def make_kv_case(num_pairs: int, gold_index: int, seed: int) -> KVCase:
    """Create paired KV data: the records stay fixed; only the target moves."""
    if num_pairs < 2 or not 0 <= gold_index < num_pairs:
        raise ValueError("invalid num_pairs or gold_index")
    rng = random.Random(seed)
    records = [(_uuid(rng), _uuid(rng)) for _ in range(num_pairs)]
    target = records.pop(0)
    records.insert(gold_index, target)
    return KVCase(tuple(records), key=target[0], value=target[1], gold_index=gold_index)


def build_kv_prompt(case: KVCase, *, query_aware: bool = False) -> str:
    serialized = json.dumps(dict(case.records), indent=2)
    instruction = "Extract the value corresponding to the specified key in the JSON object below."
    if query_aware:
        return (
            f'{instruction}\nKey: "{case.key}"\nJSON data:\n{serialized}'
            f'\nKey: "{case.key}"\nCorresponding value:'
        )
    return f'{instruction}\nJSON data:\n{serialized}\nKey: "{case.key}"\nCorresponding value:'


def evaluate_kv_sweep(
    num_pairs: int,
    positions: Sequence[int],
    num_examples: int,
    predict: Callable[[str], str],
    *,
    query_aware: bool = False,
    seed: int = 17,
) -> dict[int, float]:
    """Measure accuracy at every target position over paired synthetic records."""
    correct = {position: 0 for position in positions}
    for example_index in range(num_examples):
        example_seed = seed + example_index
        for position in positions:
            case = make_kv_case(num_pairs, position, example_seed)
            output = predict(build_kv_prompt(case, query_aware=query_aware))
            correct[position] += contains_any_answer(output, (case.value,))
    return {position: count / num_examples for position, count in correct.items()}


def curve_diagnostics(curve: dict[int, float]) -> dict[str, float | int]:
    """Position diagnostics; these names are pedagogical, not paper-defined metrics."""
    if len(curve) < 3:
        raise ValueError("a position curve needs at least three points")
    ordered = sorted(curve)
    middle_position = min(ordered, key=lambda item: abs(item - (ordered[0] + ordered[-1]) / 2))
    best_position = max(ordered, key=curve.__getitem__)
    worst_position = min(ordered, key=curve.__getitem__)
    best = curve[best_position]
    worst = curve[worst_position]
    edge_mean = mean((curve[ordered[0]], curve[ordered[-1]]))
    return {
        "best_position": best_position,
        "worst_position": worst_position,
        "best_accuracy": best,
        "worst_accuracy": worst,
        "best_worst_gap": best - worst,
        "edge_mean": edge_mean,
        "middle_accuracy": curve[middle_position],
        "edge_middle_penalty": edge_mean - curve[middle_position],
        "worst_over_best": worst / best if best else math.nan,
    }


def paired_position_delta(
    outputs_a: Sequence[Prediction], outputs_b: Sequence[Prediction]
) -> tuple[float, int]:
    """Paired accuracy delta A-B; pairing avoids changing the question distribution."""
    by_expected_a = {(item.expected, item.position): item for item in outputs_a}
    by_expected_b = {(item.expected, item.position): item for item in outputs_b}
    shared = sorted(set(by_expected_a) & set(by_expected_b), key=str)
    if not shared:
        raise ValueError("no paired predictions")
    deltas = [int(by_expected_a[key].correct) - int(by_expected_b[key].correct) for key in shared]
    return mean(deltas), len(deltas)


def truncate_and_rerank(documents: Sequence[Document], top_k: int) -> list[Document]:
    """A simple RAG mitigation: deduplicate, sort by score, then truncate."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    seen: set[tuple[str, str]] = set()
    unique: list[Document] = []
    for document in documents:
        identity = (document.title, document.text)
        if identity not in seen:
            unique.append(document)
            seen.add(identity)
    return sorted(unique, key=lambda document: document.score, reverse=True)[:top_k]


def print_curve(label: str, curve: dict[int, float]) -> None:
    print(label)
    for position, accuracy in sorted(curve.items()):
        bar = "█" * round(accuracy * 24)
        print(f"  index={position:>3}  accuracy={accuracy:6.1%}  {bar}")


class EdgeBiasedMockModel:
    """Synthetic, deterministic model used only to demonstrate the harness.

    It parses the KV prompt correctly, then uses a hand-written U-shaped success
    probability.  Repeating the key before and after the JSON raises that
    probability.  This behavior is inspired by the paper, not fitted to it.
    """

    KEY_PATTERN = re.compile(r'Key: "([^"]+)"')

    def __call__(self, prompt: str) -> str:
        keys = self.KEY_PATTERN.findall(prompt)
        if not keys or "JSON data:\n" not in prompt:
            return ""
        key = keys[-1]
        serialized = prompt.split("JSON data:\n", 1)[1].rsplit("\nKey:", 1)[0]
        records = json.loads(serialized)
        ordered_keys = list(records)
        index = ordered_keys.index(key)
        normalized_position = index / max(len(ordered_keys) - 1, 1)

        # 0 at the center, 1 at either edge.
        edge_proximity = 2 * abs(normalized_position - 0.5)
        success_probability = 0.54 + 0.43 * edge_proximity**1.3
        query_aware = len(keys) >= 2 and prompt.index("Key:") < prompt.index("JSON data:")
        if query_aware:
            success_probability = max(success_probability, 0.995)

        # Same key gets the same threshold at all positions, making the sweep paired.
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        unit_interval = int.from_bytes(digest[:8], "big") / 2**64
        return records[key] if unit_interval < success_probability else "not found"


# Exact values from the paper's 20-document appendix table.  These are paper
# results, unlike the synthetic model above.
PAPER_GPT35_20_DOCS = {0: 0.758, 4: 0.572, 9: 0.538, 14: 0.554, 19: 0.632}
PAPER_CLAUDE13_20_DOCS = {0: 0.599, 4: 0.559, 9: 0.568, 14: 0.572, 19: 0.601}


def demo() -> None:
    print_curve("PAPER_GPT35_20_DOCS", PAPER_GPT35_20_DOCS)
    print("DIAGNOSTICS", curve_diagnostics(PAPER_GPT35_20_DOCS))

    model = EdgeBiasedMockModel()
    positions = (0, 18, 37, 56, 74)
    ordinary = evaluate_kv_sweep(75, positions, 200, model, query_aware=False)
    query_aware = evaluate_kv_sweep(75, positions, 200, model, query_aware=True)
    print_curve("SYNTHETIC_ORDINARY_PROMPT", ordinary)
    print_curve("SYNTHETIC_QUERY_AWARE_PROMPT", query_aware)
    print("SYNTHETIC_DIAGNOSTICS", curve_diagnostics(ordinary))

    sample = make_kv_case(num_pairs=5, gold_index=2, seed=7)
    print("PROMPT_PREVIEW")
    print(build_kv_prompt(sample, query_aware=True)[:500], "...")

    # Sanity checks for the teaching implementation.
    assert sample.records[sample.gold_index] == (sample.key, sample.value)
    assert model(build_kv_prompt(sample, query_aware=True)) == sample.value
    assert math.isclose(
        float(curve_diagnostics(PAPER_GPT35_20_DOCS)["best_worst_gap"]),
        0.22,
    )


if __name__ == "__main__":
    demo()
