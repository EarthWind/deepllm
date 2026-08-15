#!/usr/bin/env python3
"""Dependency-free miniature of the GSM8K verifier pipeline.

This is not a replacement for the paper's transformer training code.  It keeps
the parts that are easiest to audit locally:

1. GSM8K-compatible final-answer extraction and outcome labels;
2. solution-level and token-level verifier losses;
3. best-of-N ranking and top-ranked answer voting; and
4. a safe arithmetic evaluator for calculator-style annotations.

The examples deliberately include a high-scoring wrong answer.  They show why
more search helps only while the learned verifier remains reliable.

Run:
    python3 papers/to-2026/code/training_verifiers_minimal.py
"""

from __future__ import annotations

import ast
import operator
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


# The official repository extracts the number after the first ``####`` marker.
ANSWER_RE = re.compile(r"####\s*(-?[0-9.,]+)")
INVALID_ANSWER = "[invalid]"


def extract_final_answer(solution: str) -> str:
    """Extract a GSM8K-style answer, preserving official string semantics."""

    match = ANSWER_RE.search(solution)
    if match is None:
        return INVALID_ANSWER
    return match.group(1).replace(",", "").strip()


def outcome_label(solution: str, reference_solution: str) -> int:
    """Return the paper's automatically generated final-answer label."""

    reference = extract_final_answer(reference_solution)
    if reference == INVALID_ANSWER:
        raise ValueError("reference solution has no valid final answer")
    return int(extract_final_answer(solution) == reference)


def mean_squared_error(predictions: Sequence[float], targets: Sequence[int]) -> float:
    """MSE is the verifier loss used in the paper's main experiments."""

    if not predictions or len(predictions) != len(targets):
        raise ValueError("predictions and targets must be non-empty and equally sized")
    if any(not 0.0 <= prediction <= 1.0 for prediction in predictions):
        raise ValueError("predictions must be probabilities in [0, 1]")
    if any(target not in (0, 1) for target in targets):
        raise ValueError("targets must be binary")
    return sum((prediction - target) ** 2 for prediction, target in zip(predictions, targets)) / len(targets)


def token_level_targets(outcome: int, num_solution_tokens: int) -> tuple[int, ...]:
    """Repeat one final-answer label at every unmasked solution-token position."""

    if outcome not in (0, 1):
        raise ValueError("outcome must be binary")
    if num_solution_tokens < 1:
        raise ValueError("a solution must contain at least one token")
    return (outcome,) * num_solution_tokens


def joint_objective(verifier_loss: float, language_model_loss: float) -> float:
    """Paper's unweighted verifier plus language-model auxiliary objective."""

    if verifier_loss < 0.0 or language_model_loss < 0.0:
        raise ValueError("losses must be non-negative")
    return verifier_loss + language_model_loss


@dataclass(frozen=True)
class Candidate:
    solution: str
    verifier_score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.verifier_score <= 1.0:
            raise ValueError("verifier_score must be in [0, 1]")

    @property
    def answer(self) -> str:
        return extract_final_answer(self.solution)


@dataclass(frozen=True)
class ProblemCandidates:
    task_id: str
    reference_solution: str
    candidates: tuple[Candidate, ...]

    @property
    def reference_answer(self) -> str:
        return extract_final_answer(self.reference_solution)


def select_best(candidates: Sequence[Candidate]) -> Candidate:
    """Return the single completion with the highest learned verifier score."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    return max(candidates, key=lambda candidate: candidate.verifier_score)


def vote_among_top(candidates: Sequence[Candidate], num_voters: int) -> str:
    """Vote on final answers among the highest-ranked completions.

    The paper reports that a small vote among top verifier-ranked solutions can
    beat taking only the top solution.  For deterministic tie-breaking, this
    implementation returns the tied answer that appears first in verifier rank.
    """

    if not 1 <= num_voters <= len(candidates):
        raise ValueError("num_voters must be in [1, len(candidates)]")
    ranked = sorted(candidates, key=lambda candidate: candidate.verifier_score, reverse=True)
    answers = [candidate.answer for candidate in ranked[:num_voters]]
    counts = Counter(answers)
    largest_count = max(counts.values())
    return next(answer for answer in answers if counts[answer] == largest_count)


Selector = Callable[[Sequence[Candidate]], str]


def solve_rate(problems: Iterable[ProblemCandidates], selector: Selector) -> float:
    """Compute problem-level exact-answer accuracy for a selection policy."""

    materialized = tuple(problems)
    if not materialized:
        raise ValueError("at least one problem is required")
    solved = 0
    for problem in materialized:
        if not problem.candidates:
            raise ValueError(f"{problem.task_id} has no candidates")
        solved += selector(problem.candidates) == problem.reference_answer
    return solved / len(materialized)


def oracle_coverage(problems: Iterable[ProblemCandidates]) -> float:
    """Fraction of problems with at least one correct sampled completion."""

    materialized = tuple(problems)
    if not materialized:
        raise ValueError("at least one problem is required")
    covered = sum(
        any(candidate.answer == problem.reference_answer for candidate in problem.candidates)
        for problem in materialized
    )
    return covered / len(materialized)


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_arithmetic(expression: str) -> int | float:
    """Evaluate basic GSM8K arithmetic without Python ``eval``.

    Only numeric literals, parentheses, +, -, *, and / are accepted.  This is
    an educational replacement for the old repository's tightly filtered
    ``eval`` example, not a general untrusted-code sandbox.
    """

    if len(expression) > 200:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid arithmetic expression") from error

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if abs(node.value) > 1e12:
                raise ValueError("numeric literal is too large")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            value = _BINARY_OPERATORS[type(node.op)](visit(node.left), visit(node.right))
            if abs(value) > 1e12:
                raise ValueError("result is too large")
            return value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](visit(node.operand))
        raise ValueError(f"unsupported arithmetic syntax: {type(node).__name__}")

    return visit(tree)


def _demo() -> None:
    reference_72 = "48 in April and 24 in May gives 48+24=72.\n#### 72"
    assert extract_final_answer("work\n#### 1,024") == "1024"
    assert outcome_label("different reasoning\n#### 72", reference_72) == 1
    assert outcome_label("plausible but wrong\n#### 96", reference_72) == 0
    assert safe_arithmetic("48 + 48 / 2") == 72
    try:
        safe_arithmetic("__import__('os').system('echo unsafe')")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe syntax was not rejected")

    # All token positions receive the same outcome label; question positions
    # would be masked by the training pipeline.
    predictions = (0.20, 0.40, 0.80)
    targets = token_level_targets(outcome=1, num_solution_tokens=len(predictions))
    verifier_loss = mean_squared_error(predictions, targets)
    assert targets == (1, 1, 1)
    assert abs(joint_objective(verifier_loss, 1.25) - (verifier_loss + 1.25)) < 1e-12

    problem_a = ProblemCandidates(
        task_id="toy/0",
        reference_solution=reference_72,
        candidates=(
            Candidate("correct route A\n#### 72", 0.70),
            Candidate("convincing mistake\n#### 96", 0.90),
            Candidate("correct route B\n#### 72", 0.65),
            Candidate("arithmetic slip\n#### 70", 0.30),
        ),
    )
    problem_b = ProblemCandidates(
        task_id="toy/1",
        reference_solution="seven groups of six\n#### 42",
        candidates=(
            Candidate("correct\n#### 42", 0.85),
            Candidate("wrong\n#### 36", 0.40),
            Candidate("also correct\n#### 42", 0.75),
            Candidate("wrong\n#### 48", 0.20),
        ),
    )
    problems = (problem_a, problem_b)

    best_rate = solve_rate(problems, lambda candidates: select_best(candidates).answer)
    top3_vote_rate = solve_rate(problems, lambda candidates: vote_among_top(candidates, 3))
    coverage = oracle_coverage(problems)
    assert best_rate == 0.5
    assert top3_vote_rate == 1.0
    assert coverage == 1.0

    # Search can fail non-monotonically: the fourth candidate is an adversarial
    # false positive whose verifier score exceeds the correct candidate's score.
    search_order = (
        Candidate("correct\n#### 42", 0.85),
        Candidate("wrong\n#### 40", 0.30),
        Candidate("wrong\n#### 44", 0.20),
        Candidate("deceptive false positive\n#### 999", 0.99),
    )
    selected_by_budget = {
        budget: select_best(search_order[:budget]).answer for budget in range(1, 5)
    }
    assert selected_by_budget[1] == "42"
    assert selected_by_budget[4] == "999"

    print(f"token-level verifier MSE     : {verifier_loss:.6f}")
    print(f"oracle candidate coverage    : {coverage:.1%}")
    print(f"best-of-N solve rate         : {best_rate:.1%}")
    print(f"top-3 answer-vote solve rate : {top3_vote_rate:.1%}")
    print(f"selected answers by N=1..4   : {tuple(selected_by_budget.values())}")
    print(f"safe calculator 48 + 48 / 2  : {safe_arithmetic('48 + 48 / 2')}")
    print("all checks passed; no model training or arbitrary code execution occurred")


if __name__ == "__main__":
    _demo()
