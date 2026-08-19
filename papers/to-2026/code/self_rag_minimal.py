#!/usr/bin/env python3
"""Dependency-free teaching implementation of the core SELF-RAG control loop.

The real SELF-RAG system uses a trained language model and Contriever.  This
script replaces both with deterministic fixtures so that the following ideas
can be inspected without a GPU, model download, or API key:

1. trigger retrieval from normalized reflection-token probabilities;
2. score relevance, support, and utility tokens exactly as in the paper;
3. expand one candidate per retrieved passage and run segment-level beam search;
4. apply an optional hard support constraint; and
5. serialize an offline critic-labeled generator-training example.

This is an executable algorithm sketch, not an implementation of the official
SELF-RAG checkpoints.  The official project is https://github.com/AkariAsai/self-rag.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence


ProbabilityMap = Mapping[str, float]


def normalized_probability(
    probabilities: ProbabilityMap,
    target: str,
    alternatives: Iterable[str],
) -> float:
    """Normalize one token probability within its reflection-token group."""

    labels = tuple(alternatives)
    denominator = sum(probabilities.get(label, 0.0) for label in labels)
    if denominator <= 0.0:
        raise ValueError(f"probability mass for {labels!r} must be positive")
    return probabilities.get(target, 0.0) / denominator


def should_retrieve(retrieve_probabilities: ProbabilityMap, threshold: float) -> bool:
    """Paper Appendix A.4: p(Yes) / (p(Yes) + p(No)) > delta.

    ``Continue`` is deliberately excluded from this initial Yes/No gate.  In
    the paper it means that the current passage can ground another segment.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    retrieve_score = normalized_probability(
        retrieve_probabilities,
        "yes",
        ("yes", "no"),
    )
    return retrieve_score > threshold


def relevance_score(probabilities: ProbabilityMap) -> float:
    """s(ISREL) = p(Relevant) / sum p(relevance labels)."""

    return normalized_probability(
        probabilities,
        "relevant",
        ("relevant", "irrelevant"),
    )


def support_score(probabilities: ProbabilityMap) -> float:
    """s(ISSUP) = [p(Fully) + 0.5 p(Partially)] / sum p(labels)."""

    labels = ("fully", "partially", "none")
    denominator = sum(probabilities.get(label, 0.0) for label in labels)
    if denominator <= 0.0:
        raise ValueError("support probability mass must be positive")
    numerator = probabilities.get("fully", 0.0) + 0.5 * probabilities.get(
        "partially",
        0.0,
    )
    return numerator / denominator


def utility_score(probabilities: ProbabilityMap) -> float:
    """Expected five-level utility using weights {-1, -.5, 0, .5, 1}."""

    weights = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}
    denominator = sum(probabilities.get(str(level), 0.0) for level in weights)
    if denominator <= 0.0:
        raise ValueError("utility probability mass must be positive")
    return sum(
        weight * probabilities.get(str(level), 0.0)
        for level, weight in weights.items()
    ) / denominator


@dataclass(frozen=True)
class Passage:
    passage_id: str
    text: str


@dataclass(frozen=True)
class SegmentCandidate:
    """One continuation generated while conditioning on one passage."""

    text: str
    passage: Passage | None
    # A length-normalized model score is used in this teaching demo.  Raw
    # sequence probabilities underflow quickly in ordinary Python floats.
    model_score: float
    is_rel: ProbabilityMap = field(default_factory=dict)
    is_sup: ProbabilityMap = field(default_factory=dict)
    is_use: ProbabilityMap = field(default_factory=dict)

    @property
    def support_label(self) -> str:
        return max(self.is_sup, key=self.is_sup.get) if self.is_sup else "none"


@dataclass(frozen=True)
class ScoringWeights:
    relevance: float = 1.0
    support: float = 1.0
    utility: float = 0.5


def critique_score(
    candidate: SegmentCandidate,
    weights: ScoringWeights,
) -> tuple[float, dict[str, float]]:
    """Add the model score and weighted normalized critique scores."""

    components = {
        "model": candidate.model_score,
        "relevance": relevance_score(candidate.is_rel) if candidate.is_rel else 0.0,
        "support": support_score(candidate.is_sup) if candidate.is_sup else 0.0,
        "utility": utility_score(candidate.is_use) if candidate.is_use else 0.0,
    }
    total = (
        components["model"]
        + weights.relevance * components["relevance"]
        + weights.support * components["support"]
        + weights.utility * components["utility"]
    )
    return total, components


@dataclass(frozen=True)
class Beam:
    segments: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    score: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(self.segments)


Retriever = Callable[[str], Sequence[Passage]]
Generator = Callable[[str, Beam, Passage | None], SegmentCandidate]
RetrieveGate = Callable[[str, Beam], ProbabilityMap]


def segment_beam_search(
    query: str,
    *,
    retrieve_gate: RetrieveGate,
    retriever: Retriever,
    generator: Generator,
    threshold: float = 0.2,
    beam_width: int = 2,
    max_segments: int = 2,
    weights: ScoringWeights = ScoringWeights(),
    hard_support_constraint: bool = False,
) -> list[Beam]:
    """A compact version of paper Algorithm 1.

    Each active beam independently decides whether retrieval is needed.  When
    it is, every passage creates a branch.  The best ``beam_width`` accumulated
    paths survive to the next sentence-level step.
    """

    if beam_width <= 0 or max_segments <= 0:
        raise ValueError("beam_width and max_segments must be positive")

    beams = [Beam()]
    for _step in range(max_segments):
        expanded: list[Beam] = []
        for beam in beams:
            retrieve = should_retrieve(retrieve_gate(query, beam), threshold)
            passages: Sequence[Passage | None]
            passages = retriever(query) if retrieve else (None,)

            for passage in passages:
                candidate = generator(query, beam, passage)
                if hard_support_constraint and passage is not None:
                    if candidate.support_label == "none":
                        continue

                increment, _ = critique_score(candidate, weights)
                citations = beam.citations
                if passage is not None:
                    citations += (passage.passage_id,)
                expanded.append(
                    Beam(
                        segments=beam.segments + (candidate.text,),
                        citations=citations,
                        score=beam.score + increment,
                    )
                )

        if not expanded:
            raise RuntimeError("all candidates were removed by hard constraints")
        beams = sorted(expanded, key=lambda item: item.score, reverse=True)[:beam_width]
    return beams


def serialize_generator_example(
    instruction: str,
    passage: Passage,
    answer: str,
    *,
    retrieve: str,
    relevance: str,
    support: str,
    utility: int,
) -> tuple[str, tuple[bool, ...]]:
    """Show how offline critic labels become ordinary next-token targets.

    The returned boolean tuple is a conceptual loss mask at whitespace-token
    granularity.  Tokens inside ``<paragraph>`` are context and therefore False,
    mirroring the paper's decision to mask retrieved chunks from the LM loss.
    """

    target = (
        f"[Retrieve:{retrieve}] "
        f"<paragraph> {passage.text} </paragraph> "
        f"[IsRel:{relevance}] {answer} "
        f"[IsSup:{support}] [Utility:{utility}]"
    )
    tokens = target.split()
    in_passage = False
    mask: list[bool] = []
    for token in tokens:
        if token == "<paragraph>":
            in_passage = True
        mask.append(not in_passage)
        if token == "</paragraph>":
            in_passage = False
            mask[-1] = False
    return f"Instruction: {instruction}\nTarget: {target}", tuple(mask)


def demo_components() -> None:
    """Print the exact score decomposition for three passage branches."""

    candidates = build_demo_candidates()
    weights = ScoringWeights()
    print("Critique-guided candidate ranking")
    for candidate in candidates:
        total, parts = critique_score(candidate, weights)
        print(
            f"  {candidate.passage.passage_id}: total={total:.3f} "
            f"model={parts['model']:.3f} rel={parts['relevance']:.3f} "
            f"sup={parts['support']:.3f} use={parts['utility']:.3f}"
        )


def build_demo_candidates() -> list[SegmentCandidate]:
    passages = build_demo_passages()
    return [
        SegmentCandidate(
            text="Eleven of the fifty U.S. states are named after individual people. [D1]",
            passage=passages[0],
            model_score=0.72,
            is_rel={"relevant": 0.96, "irrelevant": 0.04},
            is_sup={"fully": 0.90, "partially": 0.08, "none": 0.02},
            is_use={"1": 0.01, "2": 0.01, "3": 0.04, "4": 0.20, "5": 0.74},
        ),
        SegmentCandidate(
            text="Texas is named after a Native American tribe. [D2]",
            passage=passages[1],
            model_score=0.81,
            is_rel={"relevant": 0.10, "irrelevant": 0.90},
            is_sup={"fully": 0.04, "partially": 0.10, "none": 0.86},
            is_use={"1": 0.10, "2": 0.25, "3": 0.35, "4": 0.20, "5": 0.10},
        ),
        SegmentCandidate(
            text="California's name came from a fictional island in a Spanish novel. [D3]",
            passage=passages[2],
            model_score=0.68,
            is_rel={"relevant": 0.88, "irrelevant": 0.12},
            is_sup={"fully": 0.42, "partially": 0.48, "none": 0.10},
            is_use={"1": 0.02, "2": 0.04, "3": 0.14, "4": 0.44, "5": 0.36},
        ),
    ]


def build_demo_passages() -> list[Passage]:
    return [
        Passage("D1", "Of the fifty states, eleven are named after an individual person."),
        Passage("D2", "Emma was among the most popular baby names in Texas."),
        Passage(
            "D3",
            "California's name has its origins in the fictional island in the "
            "16th-century novel Las sergas de Esplandian.",
        ),
    ]


def run_demo() -> None:
    passages = build_demo_passages()
    candidates_by_id = {
        item.passage.passage_id: item for item in build_demo_candidates()
    }

    def gate(_query: str, beam: Beam) -> ProbabilityMap:
        # Retrieve for the first factual segment; skip for the short synthesis.
        return {"yes": 0.86, "no": 0.14} if not beam.segments else {
            "yes": 0.08,
            "no": 0.92,
        }

    def retrieve(_query: str) -> Sequence[Passage]:
        return passages

    def generate(_query: str, beam: Beam, passage: Passage | None) -> SegmentCandidate:
        if passage is not None:
            return candidates_by_id[passage.passage_id]
        return SegmentCandidate(
            text="This illustrates how state names came from several kinds of sources.",
            passage=None,
            model_score=0.70,
            is_use={"1": 0.01, "2": 0.02, "3": 0.08, "4": 0.34, "5": 0.55},
        )

    demo_components()
    print("\nAdaptive segment-level beam search")
    beams = segment_beam_search(
        "How did U.S. states get their names?",
        retrieve_gate=gate,
        retriever=retrieve,
        generator=generate,
        threshold=0.2,
        beam_width=2,
        max_segments=2,
        hard_support_constraint=True,
    )
    for rank, beam in enumerate(beams, start=1):
        print(
            f"  #{rank} score={beam.score:.3f} citations={beam.citations}\n"
            f"     {beam.text}"
        )

    print("\nOffline generator-training target (paragraph tokens are loss-masked)")
    serialized, mask = serialize_generator_example(
        "How many U.S. states are named after people?",
        passages[0],
        "Eleven states are named after individual people.",
        retrieve="Yes",
        relevance="Relevant",
        support="Fully",
        utility=5,
    )
    print(serialized)
    print(f"Loss mask: {mask}")


def self_test() -> None:
    assert should_retrieve({"yes": 0.6, "no": 0.4}, 0.5)
    assert not should_retrieve({"yes": 0.2, "no": 0.8}, 0.5)
    assert math.isclose(relevance_score({"relevant": 3, "irrelevant": 1}), 0.75)
    assert math.isclose(
        support_score({"fully": 0.6, "partially": 0.2, "none": 0.2}),
        0.7,
    )
    assert math.isclose(
        utility_score({"1": 0, "2": 0, "3": 0, "4": 0, "5": 1}),
        1.0,
    )
    _, mask = serialize_generator_example(
        "q",
        Passage("D", "retrieved context"),
        "answer",
        retrieve="Yes",
        relevance="Relevant",
        support="Fully",
        utility=5,
    )
    assert False in mask and mask[-1]
    print("All SELF-RAG teaching-demo checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run deterministic checks")
    args = parser.parse_args()
    if args.test:
        self_test()
    else:
        run_demo()


if __name__ == "__main__":
    main()
