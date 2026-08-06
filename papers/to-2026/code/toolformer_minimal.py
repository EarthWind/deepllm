#!/usr/bin/env python3
"""Dependency-free miniature of Toolformer's data and inference pipelines.

The original paper trains GPT-J and uses real neural/retrieval tools.  This
module isolates the algorithm-specific mechanics so they can be inspected and
tested without downloading a model:

1. select candidate insertion positions from the probability of ``[``;
2. linearize and execute a generated API call;
3. compare future-token loss with the result against two baselines;
4. keep calls whose loss reduction reaches ``tau_f``;
5. interleave accepted calls with the original text; and
6. pause decoding at ``->``, execute the pending call, and inject its result.

Replace ``TokenNLL`` and the candidate generator with adapters around an actual
causal LM to build a small-scale reproduction.  Run the deterministic demo:

    python3 papers/to-2026/code/toolformer_minimal.py
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


TokenNLL = Callable[[str, Sequence[str], str], float]
Tool = Callable[[str], str]


@dataclass(frozen=True)
class APICall:
    """A textual tool call proposed before ``tokens[position]``."""

    position: int
    name: str
    argument: str


@dataclass(frozen=True)
class ScoredCall:
    """One executed call and the paper's three loss quantities."""

    call: APICall
    result: str
    loss_with_result: float
    loss_without_call: float
    loss_call_without_result: float

    @property
    def baseline_loss(self) -> float:
        return min(self.loss_without_call, self.loss_call_without_result)

    @property
    def gain(self) -> float:
        return self.baseline_loss - self.loss_with_result


def linearize_call(call: APICall) -> str:
    """Return e(c), the call without an API response."""

    return f"[{call.name}({call.argument})]"


def linearize_call_result(call: APICall, result: str) -> str:
    """Return e(c, r), using the paper's vocabulary-free delimiters."""

    clean_result = " ".join(result.split())
    return f"[{call.name}({call.argument}) -> {clean_result}]"


def select_candidate_positions(
    api_start_probabilities: Sequence[float],
    *,
    tau_s: float = 0.05,
    top_k: int = 5,
) -> tuple[int, ...]:
    """Keep at most ``top_k`` positions where p("[") exceeds ``tau_s``.

    The probabilities must be computed with the tool-specific few-shot prompt
    prepended.  Returned positions are sorted in document order so subsequent
    annotation is deterministic.
    """

    if not 0.0 <= tau_s <= 1.0:
        raise ValueError("tau_s must be in [0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ranked: list[tuple[float, int]] = []
    for position, probability in enumerate(api_start_probabilities):
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid probability at {position}: {probability}")
        if probability > tau_s:  # The paper uses a strict inequality.
            ranked.append((probability, position))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(sorted(position for _, position in ranked[:top_k]))


def local_weights(horizon: int = 5, decay: float = 0.2) -> tuple[float, ...]:
    """Return normalized w_t proportional to max(0, 1 - decay * t).

    ``horizon=5`` and ``decay=0.2`` reproduce the non-zero support of the
    weighting function used in the paper: nearby future tokens matter most.
    """

    if horizon <= 0 or not 0.0 < decay <= 1.0:
        raise ValueError("expected horizon > 0 and decay in (0, 1]")
    raw = [max(0.0, 1.0 - decay * offset) for offset in range(horizon)]
    total = sum(raw)
    if total == 0.0:
        raise ValueError("weight sequence has no positive values")
    return tuple(value / total for value in raw)


def weighted_future_loss(
    token_nll: TokenNLL,
    tokens: Sequence[str],
    *,
    start: int,
    api_prefix: str,
    weights: Sequence[float],
) -> float:
    """Compute L_i(z) over original tokens at and after insertion position i.

    ``token_nll(z, x[:j], x[j])`` should return
    ``-log p_M(x_j | z, x_1,...,x_{j-1})``.  As in the paper's filtering
    procedure, the serialized call is supplied as a prefix rather than being
    inserted into the middle of an unfamiliar pretraining sequence.
    """

    if not 0 <= start <= len(tokens):
        raise IndexError("start is outside the token sequence")
    if any(weight < 0.0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("weights must be finite and non-negative")
    loss = 0.0
    for offset, weight in enumerate(weights):
        index = start + offset
        if index >= len(tokens):
            break
        nll = token_nll(api_prefix, tokens[:index], tokens[index])
        if not math.isfinite(nll) or nll < 0.0:
            raise ValueError(f"invalid token NLL at {index}: {nll}")
        loss += weight * nll
    return loss


def score_call(
    call: APICall,
    result: str,
    tokens: Sequence[str],
    token_nll: TokenNLL,
    *,
    weights: Sequence[float] | None = None,
) -> ScoredCall:
    """Compute L_i+, L_i(empty), and L_i(call-with-empty-result)."""

    active_weights = tuple(weights) if weights is not None else local_weights()

    def loss(prefix: str) -> float:
        return weighted_future_loss(
            token_nll,
            tokens,
            start=call.position,
            api_prefix=prefix,
            weights=active_weights,
        )

    return ScoredCall(
        call=call,
        result=result,
        loss_with_result=loss(linearize_call_result(call, result)),
        loss_without_call=loss(""),
        loss_call_without_result=loss(linearize_call_result(call, "")),
    )


def filter_calls(
    scored_calls: Sequence[ScoredCall],
    *,
    tau_f: float = 1.0,
) -> tuple[ScoredCall, ...]:
    """Keep calls satisfying min(L_i(empty), L_i(c, empty)) - L_i(c, r)."""

    if not math.isfinite(tau_f):
        raise ValueError("tau_f must be finite")
    return tuple(call for call in scored_calls if call.gain >= tau_f)


def interleave_best_calls(
    tokens: Sequence[str],
    accepted: Sequence[ScoredCall],
) -> tuple[str, ...]:
    """Insert the highest-gain accepted call at each position.

    Choosing one call when several survive at the same position is an explicit
    engineering policy of this miniature, not a new claim about the paper.
    """

    best_by_position: dict[int, ScoredCall] = {}
    for candidate in accepted:
        position = candidate.call.position
        if not 0 <= position <= len(tokens):
            raise IndexError(f"call position {position} is outside the sequence")
        current = best_by_position.get(position)
        if current is None or candidate.gain > current.gain:
            best_by_position[position] = candidate

    output: list[str] = []
    for position in range(len(tokens) + 1):
        if position in best_by_position:
            selected = best_by_position[position]
            output.append(linearize_call_result(selected.call, selected.result))
        if position < len(tokens):
            output.append(tokens[position])
    return tuple(output)


_BINARY_OPERATORS: Mapping[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS: Mapping[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculator(expression: str) -> str:
    """Evaluate the paper's four arithmetic operations without using eval()."""

    if len(expression) > 200:
        raise ValueError("calculator expression is too long")
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("calculator accepts only numeric constants")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](visit(node.operand))
        raise ValueError(f"unsupported calculator syntax: {ast.dump(node)}")

    value = visit(tree)
    if not math.isfinite(value):
        raise ValueError("calculator result must be finite")
    rounded = round(value, 2)
    return str(int(rounded)) if rounded.is_integer() else f"{rounded:.2f}"


_PENDING_CALL = re.compile(
    r"\[(?P<name>[A-Za-z][A-Za-z0-9_]*)\((?P<argument>.*?)\)\s*->\s*$",
    re.DOTALL,
)


def execute_pending_call(generated_text: str, tools: Mapping[str, Tool]) -> str:
    """Execute a call after decoding pauses at ``->`` and return injected text."""

    match = _PENDING_CALL.search(generated_text)
    if match is None:
        raise ValueError("generation does not end with a parseable pending API call")
    name = match.group("name")
    if name not in tools:
        raise KeyError(f"unknown tool: {name}")
    argument = match.group("argument").strip()
    result = " ".join(tools[name](argument).split())
    if not result:
        raise ValueError(f"tool {name} returned an empty result")
    separator = "" if generated_text[-1].isspace() else " "
    return f"{generated_text}{separator}{result}]"


def _demo_token_nll(
    api_prefix: str,
    original_prefix: Sequence[str],
    target: str,
) -> float:
    """A deterministic stand-in whose next-token uncertainty reacts to 0.29."""

    del original_prefix
    if target == "29%":
        return 0.05 if "0.29" in api_prefix else 4.0
    return 1.0


def main() -> None:
    """Exercise sampling, filtering, annotation, and runtime injection."""

    tokens = (
        "Out",
        "of",
        "1400",
        "participants,",
        "400",
        "(or",
        "29%",
        ")",
        "passed",
        "the",
        "test.",
    )
    probabilities = (0.01, 0.01, 0.02, 0.01, 0.03, 0.14, 0.01, 0.01, 0.01, 0.01, 0.01)
    positions = select_candidate_positions(probabilities, tau_s=0.05, top_k=5)
    assert positions == (5,)

    call = APICall(position=5, name="Calculator", argument="400 / 1400")
    result = safe_calculator(call.argument)
    scored = score_call(call, result, tokens, _demo_token_nll)
    accepted = filter_calls((scored,), tau_f=0.5)
    assert accepted and scored.gain > 0.5
    annotated = " ".join(interleave_best_calls(tokens, accepted))

    pending = "Out of 1400 participants, 400 (or [Calculator(400 / 1400) ->"
    resumed = execute_pending_call(pending, {"Calculator": safe_calculator})
    assert resumed.endswith("0.29]")
    try:
        safe_calculator("__import__('os').system('echo unsafe')")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe calculator input was not rejected")

    print(f"candidate positions: {positions}")
    print(f"tool result: {result}")
    print(f"filter gain: {scored.gain:.3f} (accepted={bool(accepted)})")
    print(f"annotated text: {annotated}")
    print(f"runtime injection: {resumed}")


if __name__ == "__main__":
    main()
