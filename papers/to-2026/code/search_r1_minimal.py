"""A tiny, dependency-free sketch of Search-R1.

This is not an LLM trainer.  It makes four boundaries executable:

1. a policy alternates generated reasoning/search actions with environment text;
2. a search engine injects passages into the rollout;
3. retrieved tokens are masked out of the policy loss; and
4. a final exact-match reward can be converted to group-relative advantages.

Run ``python search_r1_minimal.py --test`` for the self-checks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
import re
import string
from typing import Callable, Iterable, Sequence


TOKEN_PATTERN = re.compile(r"</?[^>]+>|[\w.'-]+|[^\s]", re.UNICODE)
SEARCH_PATTERN = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


@dataclass(frozen=True)
class Passage:
    title: str
    text: str


@dataclass(frozen=True)
class RolloutToken:
    text: str
    optimize: bool
    source: str


@dataclass
class Rollout:
    question: str
    tokens: list[RolloutToken] = field(default_factory=list)
    events: list[tuple[str, str]] = field(default_factory=list)
    search_calls: list[str] = field(default_factory=list)
    answer: str | None = None

    @property
    def text(self) -> str:
        return " ".join(token.text for token in self.tokens)

    @property
    def loss_mask(self) -> list[int]:
        return [int(token.optimize) for token in self.tokens]

    def append(self, text: str, *, optimize: bool, source: str) -> None:
        self.events.append((source, text))
        self.tokens.extend(
            RolloutToken(token, optimize=optimize, source=source)
            for token in TOKEN_PATTERN.findall(text)
        )


class ToyRetriever:
    """A lexical retriever small enough to understand at a glance."""

    def __init__(self, passages: Iterable[Passage]) -> None:
        self.passages = tuple(passages)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token.casefold().strip(string.punctuation)
            for token in text.split()
            if token.strip(string.punctuation)
        }

    def search(self, query: str, top_k: int = 1) -> list[Passage]:
        query_terms = self._terms(query)
        ranked = sorted(
            self.passages,
            key=lambda passage: (
                len(query_terms & self._terms(f"{passage.title} {passage.text}")),
                passage.title,
            ),
            reverse=True,
        )
        return ranked[:top_k]


Policy = Callable[[str, str], str]


def scripted_policy(question: str, context: str) -> str:
    """A deterministic stand-in for a learned search-and-reason policy."""
    del question
    if "Britney Spears" not in context:
        return (
            "<think>I need to identify the singer behind the fragrance.</think> "
            "<search>Curious fragrance singer</search>"
        )
    if "McComb" not in context:
        return (
            "<think>The evidence names Britney Spears; now I need her birthplace.</think> "
            "<search>Britney Spears birthplace</search>"
        )
    return "<think>The city and state are now supported.</think> <answer>McComb, Mississippi</answer>"


def format_passages(passages: Sequence[Passage]) -> str:
    body = "\n".join(
        f"Doc {index} (Title: {passage.title}) {passage.text}"
        for index, passage in enumerate(passages, start=1)
    )
    return f"<information>{body}</information>"


def run_rollout(
    question: str,
    policy: Policy,
    retriever: ToyRetriever,
    *,
    action_budget: int = 4,
    top_k: int = 1,
) -> Rollout:
    """Interleave policy actions and search-environment observations."""
    rollout = Rollout(question=question)
    for _ in range(action_budget):
        action = policy(question, rollout.text)
        rollout.append(action, optimize=True, source="policy")

        search_match = SEARCH_PATTERN.search(action)
        answer_match = ANSWER_PATTERN.search(action)
        if search_match:
            query = search_match.group(1).strip()
            rollout.search_calls.append(query)
            observation = format_passages(retriever.search(query, top_k=top_k))
            # Search output is context supplied by the environment, not an action
            # sampled from the policy.  Its tokens therefore receive mask value 0.
            rollout.append(observation, optimize=False, source="environment")
            continue
        if answer_match:
            rollout.answer = answer_match.group(1).strip()
            break
        rollout.append(
            "My action is not correct. Let me rethink.",
            optimize=False,
            source="environment",
        )
    return rollout


def normalize_answer(text: str) -> str:
    """A compact QA-style normalization used before exact match."""
    table = str.maketrans("", "", string.punctuation)
    return " ".join(text.casefold().translate(table).split())


def exact_match_reward(prediction: str | None, gold: str) -> float:
    return float(prediction is not None and normalize_answer(prediction) == normalize_answer(gold))


def masked_mean(values: Sequence[float], mask: Sequence[int]) -> float:
    if len(values) != len(mask):
        raise ValueError("values and mask must have the same length")
    denominator = sum(mask)
    if denominator == 0:
        raise ValueError("mask must select at least one policy token")
    return sum(value * selected for value, selected in zip(values, mask)) / denominator


def group_relative_advantages(rewards: Sequence[float], epsilon: float = 1e-8) -> list[float]:
    """Standardize rewards inside one GRPO response group."""
    if not rewards:
        raise ValueError("rewards must not be empty")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    scale = math.sqrt(variance)
    if scale < epsilon:
        return [0.0 for _ in rewards]
    return [(reward - mean) / (scale + epsilon) for reward in rewards]


def clipped_policy_objective(
    log_ratios: Sequence[float],
    advantages: Sequence[float],
    mask: Sequence[int],
    clip_epsilon: float = 0.2,
) -> float:
    """Compute the masked PPO/GRPO clipped surrogate for illustration."""
    if not (len(log_ratios) == len(advantages) == len(mask)):
        raise ValueError("log_ratios, advantages and mask must align")
    terms = []
    for log_ratio, advantage in zip(log_ratios, advantages):
        ratio = math.exp(log_ratio)
        clipped = min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
        terms.append(min(ratio * advantage, clipped * advantage))
    return masked_mean(terms, mask)


def build_demo_retriever() -> ToyRetriever:
    return ToyRetriever(
        [
            Passage(
                "Curious (fragrance)",
                "Curious is a fragrance released by American singer Britney Spears.",
            ),
            Passage(
                "Britney Spears",
                "Britney Spears' birthplace is McComb, Mississippi, United States.",
            ),
            Passage(
                "McComb, Mississippi",
                "McComb is a city in Pike County, Mississippi.",
            ),
        ]
    )


def demo() -> None:
    question = "In which city and state was the singer behind Curious born?"
    rollout = run_rollout(question, scripted_policy, build_demo_retriever())
    for source, text in rollout.events:
        print(f"[{source.upper()}] {text}")
    reward = exact_match_reward(rollout.answer, "McComb, Mississippi")
    print("search calls:", len(rollout.search_calls))
    print("policy/retrieved tokens:", sum(rollout.loss_mask), "/", rollout.loss_mask.count(0))
    print("exact-match reward:", reward)


def run_tests() -> None:
    rollout = run_rollout(
        "In which city and state was the singer behind Curious born?",
        scripted_policy,
        build_demo_retriever(),
    )
    assert rollout.answer == "McComb, Mississippi"
    assert rollout.search_calls == ["Curious fragrance singer", "Britney Spears birthplace"]
    assert exact_match_reward(rollout.answer, "mccomb mississippi") == 1.0
    assert any(token.source == "environment" and not token.optimize for token in rollout.tokens)
    assert all(token.optimize for token in rollout.tokens if token.source == "policy")

    # Large environment-token losses must not change the policy-only mean.
    nlls = [0.2 if token.optimize else 100.0 for token in rollout.tokens]
    assert math.isclose(masked_mean(nlls, rollout.loss_mask), 0.2)
    assert sum(nlls) / len(nlls) > 0.2

    advantages = group_relative_advantages([0.0, 1.0, 1.0])
    assert advantages[0] < 0.0 < advantages[1]
    assert math.isclose(sum(advantages), 0.0, abs_tol=1e-7)

    objective = clipped_policy_objective(
        [math.log(1.5), math.log(0.7), math.log(10.0)],
        [1.0, -1.0, 99.0],
        [1, 1, 0],
    )
    assert math.isclose(objective, (1.2 - 0.8) / 2)
    print("all tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_tests() if args.test else demo()
