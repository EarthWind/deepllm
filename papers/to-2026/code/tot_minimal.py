#!/usr/bin/env python3
"""A dependency-free, model-agnostic Tree of Thoughts controller.

The search code knows nothing about prompts or arithmetic.  A task supplies five
small callbacks: generate thoughts, apply a thought, evaluate a state, test the
goal, and build a stable state key.  The included Game of 24 adapter uses exact
``fractions.Fraction`` arithmetic, so the example is deterministic and easy to
verify.

Examples:
    python3 papers/to-2026/code/tot_minimal.py
    python3 papers/to-2026/code/tot_minimal.py --algorithm dfs 4 5 6 10
    python3 papers/to-2026/code/tot_minimal.py --beam-width 1 1 2 3 4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Callable, Generic, Hashable, Iterable, Optional, Sequence, TypeVar


StateT = TypeVar("StateT")
ThoughtT = TypeVar("ThoughtT")

Generator = Callable[[StateT, int], Iterable[ThoughtT]]
Transition = Callable[[StateT, ThoughtT], StateT]
Evaluator = Callable[[StateT], float]
GoalTest = Callable[[StateT], bool]
KeyFn = Callable[[StateT], Hashable]
Formatter = Callable[[StateT], str]


@dataclass(frozen=True)
class Node(Generic[StateT, ThoughtT]):
    """One search-tree node and the thoughts used to reach it."""

    state: StateT
    thoughts: tuple[ThoughtT, ...] = ()
    score: float = 0.0

    @property
    def depth(self) -> int:
        return len(self.thoughts)


@dataclass
class SearchStats:
    """Budget counters make test-time compute explicit."""

    expanded: int = 0
    generated: int = 0
    evaluated: int = 0
    pruned: int = 0
    duplicates: int = 0
    max_frontier: int = 1


@dataclass
class SearchResult(Generic[StateT, ThoughtT]):
    found: bool
    best: Optional[Node[StateT, ThoughtT]]
    stats: SearchStats
    trace: list[str] = field(default_factory=list)


def _rank_key(node: Node[StateT, ThoughtT], key_fn: KeyFn[StateT]) -> tuple:
    """Highest score first; repr(key) gives deterministic tie-breaking."""

    return (-node.score, repr(key_fn(node.state)))


def beam_search(
    initial_state: StateT,
    *,
    generate: Generator[StateT, ThoughtT],
    transition: Transition[StateT, ThoughtT],
    evaluate: Evaluator[StateT],
    is_goal: GoalTest[StateT],
    state_key: KeyFn[StateT],
    format_state: Formatter[StateT] = repr,
    max_depth: int,
    beam_width: int = 5,
    branch_limit: int = 100,
    max_expansions: int = 1_000,
) -> SearchResult[StateT, ThoughtT]:
    """Breadth-first Tree of Thoughts with a top-``beam_width`` frontier.

    This is the paper's BFS pattern: generate candidate thoughts for every
    retained state, evaluate the resulting states, then keep the best ``b``.
    A definitive goal checker may stop the search as soon as a valid solution is
    generated; the evaluator is only a heuristic, never the correctness oracle.
    """

    if beam_width < 1 or branch_limit < 1 or max_expansions < 1:
        raise ValueError("beam_width, branch_limit and max_expansions must be positive")

    stats = SearchStats()
    trace: list[str] = []
    start = Node(initial_state, score=evaluate(initial_state))
    stats.evaluated += 1
    if is_goal(initial_state):
        return SearchResult(True, start, stats, trace)

    frontier = [start]
    seen = {state_key(initial_state)}

    for depth in range(1, max_depth + 1):
        candidates: dict[Hashable, Node[StateT, ThoughtT]] = {}
        budget_exhausted = False

        for parent in frontier:
            if stats.expanded >= max_expansions:
                budget_exhausted = True
                break
            stats.expanded += 1

            thoughts = list(generate(parent.state, branch_limit))[:branch_limit]
            stats.generated += len(thoughts)
            for thought in thoughts:
                state = transition(parent.state, thought)
                key = state_key(state)
                if key in seen or key in candidates:
                    stats.duplicates += 1
                    continue

                score = evaluate(state)
                stats.evaluated += 1
                node = Node(state, parent.thoughts + (thought,), score)
                candidates[key] = node

                if is_goal(state):
                    trace.append(
                        f"depth={depth}: generated={len(candidates)}, goal={format_state(state)}"
                    )
                    return SearchResult(True, node, stats, trace)

        if not candidates:
            break

        ordered = sorted(candidates.values(), key=lambda n: _rank_key(n, state_key))
        frontier = ordered[:beam_width]
        stats.pruned += max(0, len(ordered) - len(frontier))
        stats.max_frontier = max(stats.max_frontier, len(frontier))
        seen.update(candidates)
        trace.append(
            f"depth={depth}: candidates={len(ordered)}, keep={len(frontier)}, "
            f"best={frontier[0].score:.3f} {format_state(frontier[0].state)}"
        )

        if budget_exhausted:
            break

    best = min(frontier, key=lambda n: _rank_key(n, state_key)) if frontier else None
    return SearchResult(False, best, stats, trace)


def depth_first_search(
    initial_state: StateT,
    *,
    generate: Generator[StateT, ThoughtT],
    transition: Transition[StateT, ThoughtT],
    evaluate: Evaluator[StateT],
    is_goal: GoalTest[StateT],
    state_key: KeyFn[StateT],
    format_state: Formatter[StateT] = repr,
    max_depth: int,
    value_threshold: float = 0.0,
    branch_limit: int = 100,
    max_expansions: int = 1_000,
) -> SearchResult[StateT, ThoughtT]:
    """Best-first ordered DFS with pruning and implicit stack backtracking."""

    if branch_limit < 1 or max_expansions < 1:
        raise ValueError("branch_limit and max_expansions must be positive")

    stats = SearchStats()
    trace: list[str] = []
    start = Node(initial_state, score=evaluate(initial_state))
    stats.evaluated += 1
    stack = [start]
    seen: set[Hashable] = set()
    best = start

    while stack and stats.expanded < max_expansions:
        stats.max_frontier = max(stats.max_frontier, len(stack))
        node = stack.pop()
        key = state_key(node.state)
        if key in seen:
            stats.duplicates += 1
            continue
        seen.add(key)

        if node.score > best.score:
            best = node
        if is_goal(node.state):
            trace.append(f"depth={node.depth}: goal={format_state(node.state)}")
            return SearchResult(True, node, stats, trace)
        if node.depth >= max_depth:
            continue

        stats.expanded += 1
        successors: dict[Hashable, Node[StateT, ThoughtT]] = {}
        thoughts = list(generate(node.state, branch_limit))[:branch_limit]
        stats.generated += len(thoughts)

        for thought in thoughts:
            state = transition(node.state, thought)
            child_key = state_key(state)
            if child_key in seen or child_key in successors:
                stats.duplicates += 1
                continue
            score = evaluate(state)
            stats.evaluated += 1
            if score < value_threshold:
                stats.pruned += 1
                continue
            successors[child_key] = Node(state, node.thoughts + (thought,), score)

        ordered = sorted(successors.values(), key=lambda n: _rank_key(n, state_key))
        # LIFO: push low-value nodes first so the highest-value child is visited next.
        stack.extend(reversed(ordered))
        trace.append(
            f"depth={node.depth}: expand={format_state(node.state)}, "
            f"push={len(ordered)}, stack={len(stack)}"
        )

    return SearchResult(False, best, stats, trace)


# ---------------------------------------------------------------------------
# An exact Game of 24 task adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Atom:
    value: Fraction
    expression: str


ArithmeticState = tuple[Atom, ...]


@dataclass(frozen=True)
class ArithmeticThought:
    left_index: int
    right_index: int
    result: Atom
    description: str

    def __str__(self) -> str:
        return self.description


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def make_initial_state(numbers: Sequence[int]) -> ArithmeticState:
    return tuple(
        Atom(Fraction(number), str(number)) for number in sorted(numbers)
    )


def arithmetic_key(state: ArithmeticState) -> tuple[tuple[int, int], ...]:
    """Expressions do not affect future legal values, so deduplicate by values."""

    return tuple(sorted((atom.value.numerator, atom.value.denominator) for atom in state))


def format_arithmetic_state(state: ArithmeticState) -> str:
    return "[" + ", ".join(_fraction_text(atom.value) for atom in state) + "]"


def generate_arithmetic_thoughts(
    state: ArithmeticState, limit: int
) -> Iterable[ArithmeticThought]:
    """Enumerate legal pairwise operations without floating-point error."""

    moves: list[ArithmeticThought] = []
    seen_results: set[tuple[int, int, Fraction]] = set()

    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            left, right = state[i], state[j]
            candidates = [
                (left.value + right.value, f"({left.expression}+{right.expression})"),
                (left.value * right.value, f"({left.expression}*{right.expression})"),
                (left.value - right.value, f"({left.expression}-{right.expression})"),
                (right.value - left.value, f"({right.expression}-{left.expression})"),
            ]
            if right.value != 0:
                candidates.append(
                    (left.value / right.value, f"({left.expression}/{right.expression})")
                )
            if left.value != 0:
                candidates.append(
                    (right.value / left.value, f"({right.expression}/{left.expression})")
                )

            for value, expression in candidates:
                signature = (i, j, value)
                if signature in seen_results:
                    continue
                seen_results.add(signature)
                description = (
                    f"{expression}={_fraction_text(value)}; "
                    f"combine positions {i} and {j}"
                )
                moves.append(
                    ArithmeticThought(i, j, Atom(value, expression), description)
                )

    moves.sort(key=lambda move: (abs(move.result.value - 24), move.result.expression))
    return moves[:limit]


def apply_arithmetic_thought(
    state: ArithmeticState, thought: ArithmeticThought
) -> ArithmeticState:
    remaining = [
        atom
        for index, atom in enumerate(state)
        if index not in (thought.left_index, thought.right_index)
    ]
    remaining.append(thought.result)
    return tuple(sorted(remaining, key=lambda atom: (atom.value, atom.expression)))


def _reachable_results(values: tuple[Fraction, ...]) -> frozenset[Fraction]:
    """Return every exact value reachable by combining all input values."""

    canonical = tuple(sorted(values))
    return _reachable_results_cached(canonical)


@lru_cache(maxsize=None)
def _reachable_results_cached(values: tuple[Fraction, ...]) -> frozenset[Fraction]:
    if len(values) == 1:
        return frozenset(values)

    results: set[Fraction] = set()
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            a, b = values[i], values[j]
            remaining = [v for index, v in enumerate(values) if index not in (i, j)]
            pair_results = {a + b, a * b, a - b, b - a}
            if b != 0:
                pair_results.add(a / b)
            if a != 0:
                pair_results.add(b / a)
            for result in pair_results:
                next_values = tuple(sorted((*remaining, result)))
                results.update(_reachable_results_cached(next_values))
    return frozenset(results)


def make_arithmetic_evaluator(target: Fraction) -> Evaluator[ArithmeticState]:
    """A deterministic stand-in for an LLM's sure/likely/impossible judgement."""

    def evaluate(state: ArithmeticState) -> float:
        if len(state) == 1 and state[0].value == target:
            return 100.0
        values = tuple(atom.value for atom in state)
        if target in _reachable_results(values):
            # "sure": a valid continuation exists.  Prefer deeper states slightly.
            return 20.0 + 1.0 / len(state)
        # "impossible/uncertain": retain a smooth ordering for weak search modes.
        distance = min(abs(value - target) for value in values)
        return float(Fraction(1, 1) / (Fraction(1, 1) + distance))

    return evaluate


def run_game24(
    numbers: Sequence[int],
    *,
    algorithm: str,
    target: int,
    beam_width: int,
    branch_limit: int,
    max_expansions: int,
) -> SearchResult[ArithmeticState, ArithmeticThought]:
    initial = make_initial_state(numbers)
    target_fraction = Fraction(target)
    common = dict(
        generate=generate_arithmetic_thoughts,
        transition=apply_arithmetic_thought,
        evaluate=make_arithmetic_evaluator(target_fraction),
        is_goal=lambda state: len(state) == 1 and state[0].value == target_fraction,
        state_key=arithmetic_key,
        format_state=format_arithmetic_state,
        max_depth=max(0, len(numbers) - 1),
        branch_limit=branch_limit,
        max_expansions=max_expansions,
    )
    if algorithm == "bfs":
        return beam_search(initial, beam_width=beam_width, **common)
    if algorithm == "dfs":
        return depth_first_search(initial, value_threshold=1.0, **common)
    raise ValueError(f"unknown algorithm: {algorithm}")


def print_result(
    algorithm: str,
    numbers: Sequence[int],
    target: int,
    result: SearchResult[ArithmeticState, ArithmeticThought],
) -> None:
    print(f"\n[{algorithm.upper()}] numbers={list(numbers)}, target={target}")
    for line in result.trace:
        print("  ", line)
    if result.found and result.best is not None:
        print("  path:")
        for index, thought in enumerate(result.best.thoughts, start=1):
            print(f"    {index}. {thought}")
        atom = result.best.state[0]
        print(f"  solution: {atom.expression} = {_fraction_text(atom.value)}")
        assert atom.value == target
        assert len(result.best.thoughts) == len(numbers) - 1
    else:
        print("  no solution found within the budget")
    print("  stats:", result.stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("numbers", nargs="*", type=int, default=[4, 5, 6, 10])
    parser.add_argument("--algorithm", choices=("bfs", "dfs", "both"), default="both")
    parser.add_argument("--target", type=int, default=24)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--branch-limit", type=int, default=100)
    parser.add_argument("--max-expansions", type=int, default=1_000)
    args = parser.parse_args()
    if len(args.numbers) < 1:
        parser.error("provide at least one number")
    return args


def main() -> None:
    args = parse_args()
    algorithms = ("bfs", "dfs") if args.algorithm == "both" else (args.algorithm,)
    for algorithm in algorithms:
        result = run_game24(
            args.numbers,
            algorithm=algorithm,
            target=args.target,
            beam_width=args.beam_width,
            branch_limit=args.branch_limit,
            max_expansions=args.max_expansions,
        )
        print_result(algorithm, args.numbers, args.target, result)


if __name__ == "__main__":
    main()
