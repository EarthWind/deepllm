"""A tiny, dependency-free sketch of AlphaGeometry's neuro-symbolic loop.

It is deliberately not a Euclidean geometry solver.  It demonstrates the
division of labour: a symbolic engine closes Horn-style rules, while a small
"neural proposal" function suggests auxiliary constructions and beam search
tries them.  Run ``python alpha_geometry_minimal.py --test``.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from typing import Callable, Iterable


Statement = str


@dataclass(frozen=True)
class Construction:
    name: str
    adds: tuple[Statement, ...]
    score: float


@dataclass(frozen=True)
class ProofState:
    facts: frozenset[Statement]
    proof: tuple[str, ...]


RULES: tuple[tuple[tuple[Statement, ...], Statement, str], ...] = (
    (("midpoint(D,B,C)",), "equal(B,D,D,C)", "midpoint implies equal segments"),
    (("midpoint(D,B,C)",), "collinear(B,D,C)", "midpoint lies on the segment"),
    (("equal(B,D,D,C)", "collinear(B,D,C)"), "D_is_center_of_BC", "combine midpoint facts"),
    (("perpendicular(A,D,B,C)", "equal(A,B,A,C)"), "collinear(A,D,midpoint(B,C))", "equal legs + perpendicular bisector"),
)


def deductive_closure(initial: Iterable[Statement]) -> tuple[frozenset[Statement], tuple[str, ...]]:
    """Forward-chain definite rules until a fixed point."""
    facts = set(initial)
    proof: list[str] = []
    changed = True
    while changed:
        changed = False
        for premises, conclusion, explanation in RULES:
            if conclusion not in facts and all(p in facts for p in premises):
                facts.add(conclusion)
                proof.append(f"{explanation}: {conclusion}")
                changed = True
    return frozenset(facts), tuple(proof)


def neural_proposals(state: ProofState) -> tuple[Construction, ...]:
    """A deterministic stand-in for LM auxiliary-point proposals."""
    proposals = []
    if "collinear(B,D,C)" not in state.facts:
        proposals.append(Construction("construct D as midpoint of BC", ("midpoint(D,B,C)",), 0.92))
    proposals.append(Construction("construct E on perpendicular through A", ("perpendicular(A,E,B,C)",), 0.31))
    proposals.append(Construction("construct F with equal distances", ("equal(A,F,A,B)",), 0.18))
    return tuple(sorted(proposals, key=lambda item: item.score, reverse=True))


def prove(initial: Iterable[Statement], goal: Statement, beam_size: int = 2, rounds: int = 3) -> ProofState | None:
    """Alternate symbolic closure and learned construction proposals."""
    facts, deduction_proof = deductive_closure(initial)
    beam = [ProofState(facts, deduction_proof)]
    for _ in range(rounds + 1):
        next_beam: list[ProofState] = []
        for state in beam:
            if goal in state.facts:
                return state
            for proposal in neural_proposals(state)[:beam_size]:
                new_facts, new_proof = deductive_closure(set(state.facts) | set(proposal.adds))
                trace = state.proof + (proposal.name,) + new_proof
                next_beam.append(ProofState(new_facts, trace))
        beam = next_beam[:beam_size]
    return next((state for state in beam if goal in state.facts), None)


def demo() -> None:
    initial = {"triangle(A,B,C)", "equal(A,B,A,C)"}
    goal = "collinear(B,D,C)"
    result = prove(initial, goal)
    print("goal:", goal)
    if result is None:
        print("not proved")
        return
    print("proved:", goal)
    for step in result.proof:
        print("  -", step)


def run_tests() -> None:
    facts, proof = deductive_closure({"midpoint(D,B,C)"})
    assert "equal(B,D,D,C)" in facts and "collinear(B,D,C)" in facts
    result = prove({"triangle(A,B,C)"}, "collinear(B,D,C)")
    assert result is not None
    assert any("midpoint" in step for step in result.proof)
    assert prove({"triangle(A,B,C)"}, "unknown_goal", rounds=1) is None
    print("all tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_tests() if args.test else demo()
