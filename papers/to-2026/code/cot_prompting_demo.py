"""A dependency-free Chain-of-Thought prompting and evaluation demo.

This module deliberately does not call a specific model API. Pass any local or
remote text generator as ``Callable[[str], str]`` to ``run_prompting``.

Run:
    python3 papers/to-2026/code/cot_prompting_demo.py
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Callable, Literal, Sequence


PromptMode = Literal["standard", "cot"]
TaskKind = Literal["numeric", "choice", "text"]
Generator = Callable[[str], str]

ANSWER_PATTERN = re.compile(
    r"(?:The answer is|Final answer\s*:|答案(?:是|为)\s*[:：]?)"
    r"\s*(?P<answer>[^\n]+)",
    flags=re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"[-+]?\s*\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)
MAX_EXPRESSION_NODES = 64
MAX_ABSOLUTE_EXPONENT = 10


@dataclass(frozen=True)
class Exemplar:
    question: str
    rationale: str
    answer: str


@dataclass(frozen=True)
class ParsedCompletion:
    rationale: str
    answer: str
    raw_text: str


@dataclass(frozen=True)
class RunResult:
    mode: PromptMode
    prompt: str
    completion: str
    parsed: ParsedCompletion


def _clean_field(value: str, field_name: str) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def build_standard_prompt(
    exemplars: Sequence[Exemplar],
    question: str,
) -> str:
    """Build the ordinary few-shot baseline: input -> answer."""
    blocks = [
        f"Q: {_clean_field(ex.question, 'question')}\n"
        f"A: The answer is {_clean_field(ex.answer, 'answer')}."
        for ex in exemplars
    ]
    blocks.append(f"Q: {_clean_field(question, 'question')}\nA:")
    return "\n\n".join(blocks)


def build_cot_prompt(
    exemplars: Sequence[Exemplar],
    question: str,
) -> str:
    """Build the paper's few-shot CoT format: input -> rationale -> answer."""
    blocks = [
        f"Q: {_clean_field(ex.question, 'question')}\n"
        f"A: {_clean_field(ex.rationale, 'rationale')} "
        f"The answer is {_clean_field(ex.answer, 'answer')}."
        for ex in exemplars
    ]
    blocks.append(f"Q: {_clean_field(question, 'question')}\nA:")
    return "\n\n".join(blocks)


def build_prompt(
    mode: PromptMode,
    exemplars: Sequence[Exemplar],
    question: str,
) -> str:
    if mode == "standard":
        return build_standard_prompt(exemplars, question)
    if mode == "cot":
        return build_cot_prompt(exemplars, question)
    raise ValueError(f"unsupported mode: {mode!r}")


def fit_exemplars_to_budget(
    mode: PromptMode,
    exemplars: Sequence[Exemplar],
    question: str,
    max_chars: int,
) -> tuple[tuple[Exemplar, ...], str]:
    """Keep the largest in-order exemplar prefix that fits a character budget.

    Production systems should replace this character approximation with the
    deployed model's tokenizer. Returning the selected examples makes any
    truncation observable instead of silently cutting a rationale in half.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    selected: list[Exemplar] = []
    prompt = build_prompt(mode, selected, question)
    if len(prompt) > max_chars:
        raise ValueError("the query alone exceeds max_chars")

    for exemplar in exemplars:
        candidate = build_prompt(mode, [*selected, exemplar], question)
        if len(candidate) > max_chars:
            break
        selected.append(exemplar)
        prompt = candidate
    return tuple(selected), prompt


def parse_completion(text: str) -> ParsedCompletion:
    """Split a completion at the last explicit final-answer marker.

    Choosing the last marker avoids treating numbers inside the rationale as the
    final prediction. Missing markers fail loudly because silent guessing makes
    evaluation results difficult to audit.
    """
    raw = text.strip()
    matches = list(ANSWER_PATTERN.finditer(raw))
    if not matches:
        raise ValueError(
            "completion has no explicit final-answer marker; expected "
            "'The answer is', 'Final answer:', or '答案是'"
        )

    match = matches[-1]
    rationale = raw[: match.start()].strip()
    if rationale.lower().startswith("a:"):
        rationale = rationale[2:].strip()
    answer = match.group("answer").strip().rstrip("。.")
    if not answer:
        raise ValueError("final answer is empty")
    return ParsedCompletion(rationale=rationale, answer=answer, raw_text=raw)


def normalize_numeric_answer(value: str) -> str:
    """Extract and canonicalize the last number in an answer span."""
    matches = NUMBER_PATTERN.findall(value)
    if not matches:
        raise ValueError(f"no numeric answer found in {value!r}")

    token = re.sub(r"[\s$,]", "", matches[-1])
    is_percent = token.endswith("%")
    if is_percent:
        token = token[:-1]
    try:
        number = Decimal(token)
    except InvalidOperation as error:
        raise ValueError(f"invalid numeric answer: {token!r}") from error

    if number == 0:
        number = Decimal(0)
    canonical = format(number.normalize(), "f")
    return f"{canonical}%" if is_percent else canonical


def normalize_choice_answer(value: str) -> str:
    """Normalize answers such as ``B``, ``(b)``, and ``option B``."""
    match = re.fullmatch(
        r"\s*(?:option\s*)?[\(\[]?([A-Ea-e])[\)\].]?\s*",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"no A-E choice found in {value!r}")
    return match.group(1).upper()


def normalize_text_answer(value: str) -> str:
    text = value.casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def answers_match(prediction: str, reference: str, kind: TaskKind) -> bool:
    if kind == "numeric":
        normalize = normalize_numeric_answer
    elif kind == "choice":
        normalize = normalize_choice_answer
    elif kind == "text":
        normalize = normalize_text_answer
    else:
        raise ValueError(f"unsupported task kind: {kind!r}")
    return normalize(prediction) == normalize(reference)


def safe_eval_arithmetic(expression: str) -> Fraction:
    """Evaluate a small arithmetic expression without calling Python ``eval``.

    This is a minimal illustration of an external calculator. It accepts only
    numeric constants, parentheses, +, -, *, /, and bounded integer powers.
    It intentionally rejects names, function calls, attributes, and containers.
    """
    if len(expression) > 256:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid arithmetic expression") from error
    if sum(1 for _ in ast.walk(tree)) > MAX_EXPRESSION_NODES:
        raise ValueError("expression is too complex")

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("division by zero")
                return left / right
            if isinstance(node.op, ast.Pow):
                if right.denominator != 1:
                    raise ValueError("exponent must be an integer")
                exponent = right.numerator
                if abs(exponent) > MAX_ABSOLUTE_EXPONENT:
                    raise ValueError("exponent is too large")
                if left == 0 and exponent < 0:
                    raise ValueError("division by zero")
                return left**exponent
        raise ValueError(f"unsupported syntax: {type(node).__name__}")

    return evaluate(tree)


def run_prompting(
    generator: Generator,
    mode: PromptMode,
    exemplars: Sequence[Exemplar],
    question: str,
) -> RunResult:
    """Run a model-agnostic prompting pipeline."""
    prompt = build_prompt(mode, exemplars, question)
    completion = generator(prompt)
    return RunResult(
        mode=mode,
        prompt=prompt,
        completion=completion,
        parsed=parse_completion(completion),
    )


def _demo_generator(prompt: str) -> str:
    """Deterministic stand-in used only to exercise the pipeline."""
    if "Roger starts with 5 balls" in prompt:
        return (
            "Lina has 23 apples after buying them. Selling 20 leaves 3, "
            "and receiving 6 more gives 9. The answer is 9."
        )
    return "The answer is 9."


def _self_test() -> None:
    exemplars = (
        Exemplar(
            question="Roger starts with 5 balls and buys 3. How many now?",
            rationale="He starts with 5 and adds 3, so 5 + 3 = 8.",
            answer="8",
        ),
        Exemplar(
            question="A box has 12 pens and 4 are removed. How many remain?",
            rationale="Removing 4 from 12 gives 12 - 4 = 8.",
            answer="8",
        ),
    )
    query = (
        "Lina buys 23 apples, sells 20, then receives 6. "
        "How many apples does she have?"
    )

    standard = build_standard_prompt(exemplars, query)
    cot = build_cot_prompt(exemplars, query)
    assert "5 + 3 = 8" not in standard
    assert "5 + 3 = 8" in cot

    parsed = parse_completion(
        "First compute 23 - 20 = 3. Then 3 + 6 = 9. The answer is 9."
    )
    assert parsed.answer == "9"
    assert "3 + 6 = 9" in parsed.rationale
    assert answers_match(parsed.answer, "9.0", "numeric")
    assert normalize_numeric_answer("The total is $1,392.00.") == "1392"
    assert normalize_numeric_answer("Accuracy is 25%.") == "25%"
    assert normalize_choice_answer("option (b)") == "B"
    assert answers_match("New York.", "new york", "text")
    assert safe_eval_arithmetic("(23 - 20) + 6") == 9
    assert safe_eval_arithmetic("1 / 4 + 0.75") == 1
    try:
        safe_eval_arithmetic("__import__('os').system('echo unsafe')")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe calculator input was not rejected")

    one_example_budget = len(build_cot_prompt(exemplars[:1], query))
    selected, budgeted_prompt = fit_exemplars_to_budget(
        "cot", exemplars, query, max_chars=one_example_budget
    )
    assert len(selected) == 1
    assert len(budgeted_prompt) <= one_example_budget

    result = run_prompting(_demo_generator, "cot", exemplars, query)
    assert answers_match(result.parsed.answer, "9", "numeric")


def main() -> None:
    _self_test()
    exemplar = Exemplar(
        question="Roger starts with 5 balls and buys 3. How many now?",
        rationale="He starts with 5 and adds 3, so 5 + 3 = 8.",
        answer="8",
    )
    question = (
        "Lina buys 23 apples, sells 20, then receives 6. "
        "How many apples does she have?"
    )
    result = run_prompting(_demo_generator, "cot", [exemplar], question)

    print("=== CoT prompt ===")
    print(result.prompt)
    print("\n=== Parsed completion ===")
    print(f"rationale: {result.parsed.rationale}")
    print(f"answer:    {result.parsed.answer}")
    print("\nAll self-tests passed.")


if __name__ == "__main__":
    main()
