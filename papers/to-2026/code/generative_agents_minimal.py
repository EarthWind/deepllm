#!/usr/bin/env python3
"""A zero-dependency teaching implementation of the Generative Agents core.

This is not the official Smallville code.  It isolates four ideas from the
paper: a memory stream, three-factor retrieval, reflection with evidence, and
hierarchical planning with observation-triggered replanning.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> Counter[str]:
    """A tiny bag-of-words stand-in for the paper's embedding model."""
    return Counter(TOKEN_RE.findall(text.lower()))


def cosine(left: Counter[str], right: Counter[str]) -> float:
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def minmax(values: list[float]) -> list[float]:
    """Match the paper's per-component [0, 1] normalization."""
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


@dataclass
class Memory:
    ident: int
    kind: str  # observation | reflection | plan
    text: str
    created_hour: int
    last_access_hour: int
    importance: int
    evidence: tuple[int, ...] = ()
    vector: Counter[str] = field(default_factory=Counter, repr=False)


@dataclass(frozen=True)
class Retrieval:
    memory: Memory
    recency: float
    importance: float
    relevance: float
    score: float


class MemoryStream:
    def __init__(self, decay: float = 0.995) -> None:
        self.decay = decay
        self.items: list[Memory] = []
        self._unreflected_observation_ids: list[int] = []

    def add(
        self,
        text: str,
        *,
        now: int,
        importance: int,
        kind: str = "observation",
        evidence: Iterable[int] = (),
    ) -> Memory:
        if kind not in {"observation", "reflection", "plan"}:
            raise ValueError(f"unsupported memory kind: {kind}")
        if not 1 <= importance <= 10:
            raise ValueError("importance must be between 1 and 10")
        memory = Memory(
            ident=len(self.items) + 1,
            kind=kind,
            text=text,
            created_hour=now,
            last_access_hour=now,
            importance=importance,
            evidence=tuple(evidence),
            vector=tokens(text),
        )
        self.items.append(memory)
        if kind == "observation":
            self._unreflected_observation_ids.append(memory.ident)
        return memory

    def retrieve(self, query: str, *, now: int, top_k: int = 5) -> list[Retrieval]:
        if not self.items or top_k <= 0:
            return []

        query_vector = tokens(query)
        recency_raw = [
            self.decay ** max(0, now - item.last_access_hour)
            for item in self.items
        ]
        importance_raw = [float(item.importance) for item in self.items]
        relevance_raw = [cosine(query_vector, item.vector) for item in self.items]

        recency = minmax(recency_raw)
        importance = minmax(importance_raw)
        relevance = minmax(relevance_raw)
        ranked = [
            Retrieval(item, r, i, rel, r + i + rel)
            for item, r, i, rel in zip(
                self.items, recency, importance, relevance, strict=True
            )
        ]
        ranked.sort(key=lambda hit: (hit.score, hit.memory.ident), reverse=True)
        selected = ranked[:top_k]
        for hit in selected:
            hit.memory.last_access_hour = now
        return selected

    def importance_since_reflection(self) -> int:
        by_id = {item.ident: item for item in self.items}
        return sum(by_id[ident].importance for ident in self._unreflected_observation_ids)

    def mark_reflected(self) -> None:
        self._unreflected_observation_ids.clear()


class RuleBasedReflector:
    """A deterministic substitute for the paper's prompt chain."""

    def maybe_reflect(
        self,
        memory: MemoryStream,
        *,
        now: int,
        threshold: int = 15,
    ) -> list[Memory]:
        # The paper uses 150.  The demo uses 15 so four observations can fire it.
        if memory.importance_since_reflection() <= threshold:
            return []

        recent = [item for item in memory.items if item.kind == "observation"][-100:]
        recent_text = " ".join(item.text.lower() for item in recent)
        reflections: list[Memory] = []

        if "party" in recent_text or "invited" in recent_text:
            evidence = tuple(
                item.ident
                for item in recent
                if "party" in item.text.lower() or "invited" in item.text.lower()
            )
            reflections.append(
                memory.add(
                    "The Valentine's Day party matters to me; I should prepare, "
                    "tell relevant friends, and reserve time to attend.",
                    now=now,
                    importance=9,
                    kind="reflection",
                    evidence=evidence,
                )
            )

        if "research" in recent_text or "composition" in recent_text:
            evidence = tuple(
                item.ident
                for item in recent
                if {"research", "composition"} & set(tokens(item.text))
            )
            reflections.append(
                memory.add(
                    "Long, uninterrupted work blocks are important for my current project.",
                    now=now,
                    importance=8,
                    kind="reflection",
                    evidence=evidence,
                )
            )

        memory.mark_reflected()
        return reflections


@dataclass(frozen=True)
class PlanItem:
    start: str
    minutes: int
    activity: str
    location: str


class Planner:
    """Top-down plan construction plus a small reaction/replanning rule."""

    def daily_plan(self, memory: MemoryStream, *, now: int) -> list[PlanItem]:
        context = " ".join(
            hit.memory.text
            for hit in memory.retrieve(
                "important commitments party research plans", now=now, top_k=6
            )
        ).lower()
        plan = [
            PlanItem("08:00", 60, "morning routine and breakfast", "home"),
            PlanItem("09:00", 180, "work", "workplace"),
            PlanItem("12:00", 60, "lunch", "Hobbs Cafe"),
        ]
        if "party" in context:
            plan.extend(
                [
                    PlanItem("13:00", 120, "prepare for the party", "Hobbs Cafe"),
                    PlanItem("17:00", 120, "attend the party", "Hobbs Cafe"),
                ]
            )
        elif "research" in context or "project" in context:
            plan.append(PlanItem("13:00", 240, "focused project work", "library"))
        plan.append(PlanItem("19:00", 120, "dinner and wind down", "home"))
        return plan

    @staticmethod
    def decompose(item: PlanItem, chunk_minutes: int = 30) -> list[str]:
        hour, minute = (int(part) for part in item.start.split(":"))
        start = hour * 60 + minute
        result = []
        for offset in range(0, item.minutes, chunk_minutes):
            timestamp = start + offset
            result.append(
                f"{timestamp // 60:02d}:{timestamp % 60:02d} "
                f"{item.activity} @ {item.location}"
            )
        return result

    @staticmethod
    def react(observation: str, current_plan: PlanItem) -> str:
        lowered = observation.lower()
        if "burning" in lowered:
            return "react: turn off the stove; pause and regenerate the remaining plan"
        if "starts a conversation" in lowered or "asks" in lowered:
            return "react: hold a context-aware conversation, then resume or replan"
        return f"continue: {current_plan.activity}"


def print_retrieval(hits: list[Retrieval]) -> None:
    print("\nretrieval ranking")
    print("rank  id  kind        recency  importance  relevance  total  memory")
    for rank, hit in enumerate(hits, 1):
        print(
            f"{rank:>4}  {hit.memory.ident:>2}  {hit.memory.kind:<10}  "
            f"{hit.recency:>7.3f}  {hit.importance:>10.3f}  "
            f"{hit.relevance:>9.3f}  {hit.score:>5.3f}  {hit.memory.text}"
        )


def main() -> None:
    stream = MemoryStream()
    stream.add(
        "Klaus is a college student working on a research paper.",
        now=0,
        importance=6,
    )
    stream.add("Klaus ate breakfast in his room.", now=8, importance=1)
    stream.add(
        "Isabella invited Klaus to the Valentine's Day party at Hobbs Cafe at 5 pm.",
        now=10,
        importance=10,
    )
    stream.add(
        "Klaus discussed his research deadline with Maria.", now=12, importance=7
    )
    stream.add("The library desk is unoccupied.", now=13, importance=2)

    reflections = RuleBasedReflector().maybe_reflect(stream, now=14)
    print("reflections")
    for reflection in reflections:
        print(f"- {reflection.text}  evidence={reflection.evidence}")

    hits = stream.retrieve(
        "What am I looking forward to at Hobbs Cafe?", now=30, top_k=5
    )
    print_retrieval(hits)

    planner = Planner()
    plan = planner.daily_plan(stream, now=30)
    print("\ndaily plan")
    for item in plan:
        print(f"- {item.start}  {item.minutes:>3} min  {item.activity} @ {item.location}")

    party_block = next(item for item in plan if item.activity == "prepare for the party")
    print("\nrecursive decomposition")
    for action in planner.decompose(party_block):
        print(f"- {action}")

    print("\nreaction")
    print(planner.react("Maria starts a conversation about the party.", party_block))


if __name__ == "__main__":
    main()
