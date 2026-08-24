#!/usr/bin/env python3
"""DeepSeek-Prover-V2 子目标课程与 GRPO 信号的零依赖教学实现。

这个脚本演示论文最关键的数据流：

1. 从带 ``have ... := by sorry`` 的 Lean 风格 proof sketch 中抽取子目标；
2. 为每个子目标构造「独立目标」与「带前置引理」两类课程题；
3. 用一个确定性的 toy prover 递归补全子目标并拼回完整证明；
4. 用 verifier 二值结果和早期结构一致性信号给候选 proof 打分；
5. 计算 GRPO 的组内标准化 advantage 与无偏 pass@k。

它不是 Lean 解析器、Lean kernel 或模型训练代码。toy prover 只识别 demo 中的
三个命题；``lean_like_sanity_check`` 只做结构检查，绝不能代替真实 Lean 验证。
运行 ``python deepseek_prover_v2_minimal.py --test`` 执行自测。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import re
from typing import Callable, Sequence


SKETCH = """import Mathlib

theorem square_sum_nonneg (x y : ℝ) : 0 ≤ x ^ 2 + y ^ 2 := by
  have hx : 0 ≤ x ^ 2 := by
    sorry
  have hy : 0 ≤ y ^ 2 := by
    sorry
  have hsum : 0 ≤ x ^ 2 + y ^ 2 := by
    sorry
  exact hsum
"""


HOLE_RE = re.compile(
    r"^(?P<indent>[ \t]*)have\s+(?P<name>[A-Za-z_][A-Za-z0-9_']*)"
    r"\s*:\s*(?P<proposition>.+?)\s*:=\s*by\s*\n"
    r"(?P<body_indent>[ \t]*)sorry\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Subgoal:
    """一个从 ``have`` proof sketch 中抽出的局部引理。"""

    name: str
    proposition: str
    predecessors: tuple[str, ...]
    start: int
    end: int
    indent: str
    body_indent: str


@dataclass(frozen=True)
class CurriculumProblem:
    """由子目标派生出的 Lean 风格课程题。"""

    subgoal: str
    kind: str
    statement: str


@dataclass(frozen=True)
class ProofCandidate:
    """一个 RL rollout；accepted 模拟真实 Lean verifier 的返回值。"""

    text: str
    accepted: bool


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: ProofCandidate
    correctness: float
    consistency: float
    reward: float
    advantage: float


def extract_subgoals(sketch: str) -> list[Subgoal]:
    """抽取教学模板中的单层 ``have`` 洞，并记录顺序依赖。"""

    subgoals: list[Subgoal] = []
    predecessors: list[str] = []
    for match in HOLE_RE.finditer(sketch):
        name = match.group("name")
        subgoals.append(
            Subgoal(
                name=name,
                proposition=match.group("proposition").strip(),
                predecessors=tuple(predecessors),
                start=match.start(),
                end=match.end(),
                indent=match.group("indent"),
                body_indent=match.group("body_indent"),
            )
        )
        predecessors.append(name)
    return subgoals


def make_curriculum(
    theorem_name: str, binders: str, subgoals: Sequence[Subgoal]
) -> list[CurriculumProblem]:
    """构造论文 Figure 3 对应的两类子目标定理。

    ``standalone`` 只替换原目标；``with_context`` 还把前序子目标加入前提。
    """

    by_name = {subgoal.name: subgoal for subgoal in subgoals}
    problems: list[CurriculumProblem] = []
    for index, subgoal in enumerate(subgoals, start=1):
        prefix = f"theorem {theorem_name}__sg{index} {binders}"
        standalone = f"{prefix} : {subgoal.proposition} := by\n  sorry"
        problems.append(CurriculumProblem(subgoal.name, "standalone", standalone))

        if not subgoal.predecessors:
            # 对第一个子目标，两种变换完全相同，不重复写入数据集。
            continue
        assumptions = " ".join(
            f"({name} : {by_name[name].proposition})"
            for name in subgoal.predecessors
        )
        with_context = " ".join(part for part in (prefix, assumptions) if part)
        with_context += f" : {subgoal.proposition} := by\n  sorry"
        problems.append(CurriculumProblem(subgoal.name, "with_context", with_context))
    return problems


ToyProver = Callable[[Subgoal, Sequence[Subgoal]], str | None]


def toy_prover(subgoal: Subgoal, available: Sequence[Subgoal]) -> str | None:
    """demo 专用的确定性 7B-prover 替身。"""

    if subgoal.name == "hx":
        return "exact sq_nonneg x"
    if subgoal.name == "hy":
        return "exact sq_nonneg y"
    available_names = {item.name for item in available}
    if subgoal.name == "hsum" and {"hx", "hy"} <= available_names:
        return "nlinarith [hx, hy]"
    return None


def recursively_solve(
    subgoals: Sequence[Subgoal], prover: ToyProver
) -> dict[str, str] | None:
    """按依赖顺序求解；任何子目标失败，原题都不能组成完整 proof。"""

    solved: dict[str, str] = {}
    available: list[Subgoal] = []
    for subgoal in subgoals:
        proof = prover(subgoal, available)
        if proof is None:
            return None
        solved[subgoal.name] = proof
        available.append(subgoal)
    return solved


def compose_proof(
    sketch: str, subgoals: Sequence[Subgoal], solutions: dict[str, str]
) -> str:
    """把已验证的局部 proof 反向写回原 sketch，避免偏移量变化。"""

    result = sketch
    for subgoal in reversed(subgoals):
        proof = solutions[subgoal.name]
        indented_proof = proof.replace("\n", "\n" + subgoal.body_indent)
        replacement = (
            f"{subgoal.indent}have {subgoal.name} : "
            f"{subgoal.proposition} := by\n"
            f"{subgoal.body_indent}{indented_proof}"
        )
        result = result[: subgoal.start] + replacement + result[subgoal.end :]
    return result


def lean_like_sanity_check(proof: str, expected_subgoals: Sequence[str]) -> bool:
    """仅用于教学的结构检查；真实系统必须调用 Lean kernel。"""

    banned = re.search(r"\b(sorry|admit)\b", proof)
    has_structure = all(f"have {name} :" in proof for name in expected_subgoals)
    return banned is None and has_structure and "exact hsum" in proof


def structure_consistency(text: str, expected_subgoals: Sequence[str]) -> float:
    """检查最终 proof 是否保留 proof plan 指定的全部 ``have`` 引理。"""

    return float(all(f"have {name} :" in text for name in expected_subgoals))


def group_relative_advantages(
    rewards: Sequence[float], epsilon: float = 1e-8
) -> list[float]:
    """GRPO 的组内标准化优势；全对或全错组没有相对学习信号。"""

    if not rewards:
        raise ValueError("rewards must not be empty")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std < epsilon:
        return [0.0] * len(rewards)
    return [(reward - mean) / std for reward in rewards]


def score_group(
    candidates: Sequence[ProofCandidate],
    expected_subgoals: Sequence[str],
    consistency_weight: float = 0.2,
) -> list[ScoredCandidate]:
    """模拟论文早期训练的二值正确性 + 结构一致性信号。

    ``0.2`` 只是教学参数；论文没有披露一致性项的具体公式或权重。
    """

    components: list[tuple[float, float, float]] = []
    for candidate in candidates:
        correctness = float(candidate.accepted)
        consistency = structure_consistency(candidate.text, expected_subgoals)
        reward = correctness + consistency_weight * consistency
        components.append((correctness, consistency, reward))
    advantages = group_relative_advantages([item[2] for item in components])
    return [
        ScoredCandidate(candidate, correctness, consistency, reward, advantage)
        for candidate, (correctness, consistency, reward), advantage in zip(
            candidates, components, advantages
        )
    ]


def unbiased_pass_at_k(total_samples: int, correct_samples: int, k: int) -> float:
    """从 n 条样本、c 条正确中估计 pass@k。"""

    if not 1 <= k <= total_samples:
        raise ValueError("require 1 <= k <= total_samples")
    if not 0 <= correct_samples <= total_samples:
        raise ValueError("correct_samples must be between 0 and total_samples")
    if total_samples - correct_samples < k:
        return 1.0
    return 1.0 - math.comb(total_samples - correct_samples, k) / math.comb(
        total_samples, k
    )


def demo(show_curriculum: bool = False) -> None:
    subgoals = extract_subgoals(SKETCH)
    curriculum = make_curriculum("square_sum_nonneg", "(x y : ℝ)", subgoals)
    solutions = recursively_solve(subgoals, toy_prover)
    if solutions is None:
        raise RuntimeError("toy prover failed")
    composed = compose_proof(SKETCH, subgoals, solutions)

    print("subgoals:")
    for subgoal in subgoals:
        dependency = ", ".join(subgoal.predecessors) or "<none>"
        print(f"  {subgoal.name}: {subgoal.proposition}  <- {dependency}")

    if show_curriculum:
        print("\ncurriculum problems:")
        for problem in curriculum:
            print(f"\n[{problem.kind}] {problem.subgoal}\n{problem.statement}")

    print("\ncomposed proof:")
    print(composed.rstrip())
    print("sanity check:", lean_like_sanity_check(composed, [s.name for s in subgoals]))

    candidates = [
        ProofCandidate(composed, accepted=True),
        ProofCandidate(
            "theorem square_sum_nonneg (x y : ℝ) : 0 ≤ x ^ 2 + y ^ 2 := by positivity",
            accepted=True,
        ),
        ProofCandidate(SKETCH, accepted=False),
        ProofCandidate(composed.replace("exact hsum", "exact hx"), accepted=False),
    ]
    scored = score_group(candidates, [subgoal.name for subgoal in subgoals])
    print("\nidx  lean  consistent  reward  advantage")
    for index, item in enumerate(scored):
        print(
            f"{index:>3}  {item.correctness:>4.0f}  {item.consistency:>10.0f}"
            f"  {item.reward:>6.2f}  {item.advantage:>9.4f}"
        )
    print("pass@32 from 5 correct / 128 samples:", f"{unbiased_pass_at_k(128, 5, 32):.4f}")


def run_tests() -> None:
    subgoals = extract_subgoals(SKETCH)
    assert [subgoal.name for subgoal in subgoals] == ["hx", "hy", "hsum"]
    assert subgoals[2].predecessors == ("hx", "hy")

    curriculum = make_curriculum("square_sum_nonneg", "(x y : ℝ)", subgoals)
    assert len(curriculum) == 5
    hsum_context = next(
        problem.statement
        for problem in curriculum
        if problem.subgoal == "hsum" and problem.kind == "with_context"
    )
    assert "(hx : 0 ≤ x ^ 2)" in hsum_context
    assert "(hy : 0 ≤ y ^ 2)" in hsum_context

    solutions = recursively_solve(subgoals, toy_prover)
    assert solutions is not None
    composed = compose_proof(SKETCH, subgoals, solutions)
    assert lean_like_sanity_check(composed, ["hx", "hy", "hsum"])
    assert "sorry" not in composed

    advantages = group_relative_advantages([1.2, 1.0, 0.2, 0.2])
    assert abs(sum(advantages)) < 1e-9
    assert group_relative_advantages([1.0, 1.0]) == [0.0, 0.0]
    assert unbiased_pass_at_k(10, 0, 3) == 0.0
    assert unbiased_pass_at_k(10, 10, 3) == 1.0
    print("all tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--show-curriculum", action="store_true")
    args = parser.parse_args()
    run_tests() if args.test else demo(show_curriculum=args.show_curriculum)
