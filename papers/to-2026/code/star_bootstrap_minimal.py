"""STaR（Self-Taught Reasoner）的离线最小实现。

这个脚本不调用真实语言模型，而用一个确定性的 ToyReasoner 展示论文最关键的
数据流不变量：

1. 先无提示地产生 rationale + answer，仅按最终答案筛选；
2. 对失败题提供正确答案 hint，尝试 rationalization；
3. 保存 rationalization 样本时移除 hint；
4. 每个 outer loop 都从同一个原始 base model 重新训练，而非继续微调上一轮；
5. “最终答案正确”并不保证 rationale 正确或 faithful。

运行：python3 papers/to-2026/code/star_bootstrap_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Problem:
    problem_id: str
    question: str
    answer: str
    difficulty: int
    gold_rationale_for_demo: str


@dataclass(frozen=True)
class Generation:
    rationale: str
    answer: str
    mode: str  # "forward" or "rationalized"
    hint_seen: str | None = None


@dataclass(frozen=True)
class TrainingExample:
    problem_id: str
    question: str
    rationale: str
    answer: str
    source: str
    difficulty: int

    def serialized(self) -> str:
        """训练输入不包含 rationalization 阶段曾经看到的答案 hint。"""
        return f"Q: {self.question}\nA: {self.rationale} Therefore: {self.answer}"


@dataclass(frozen=True)
class IterationStats:
    iteration: int
    forward_kept: int
    rationalized_kept: int
    coverage: int
    mastered_difficulty: int
    trained_from_base_version: str


class ToyReasoner:
    """用“已掌握难度”模拟可逐轮扩展的推理能力。"""

    def __init__(
        self,
        *,
        base_version: str,
        mastered_difficulty: int,
        trained_example_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.base_version = base_version
        self.mastered_difficulty = mastered_difficulty
        self.trained_example_ids = trained_example_ids

    def generate(self, problem: Problem, answer_hint: str | None = None) -> Generation:
        # Forward generation samples from the analogue of p(r, y | x).
        if answer_hint is None:
            if problem.difficulty <= self.mastered_difficulty:
                return Generation(
                    rationale=problem.gold_rationale_for_demo,
                    answer=problem.answer,
                    mode="forward",
                )
            return Generation(
                rationale="The pattern looks familiar, but the intermediate steps are unclear.",
                answer="unknown",
                mode="forward",
            )

        # Rationalization samples from the analogue of p(r, y | x, y*).
        # 这个玩具设定中，答案提示一次只能帮助跨越一个难度层级。
        if problem.difficulty <= self.mastered_difficulty + 1:
            return Generation(
                rationale=(
                    f"Working backward from the target, {problem.gold_rationale_for_demo}"
                ),
                answer=answer_hint,
                mode="rationalized",
                hint_seen=answer_hint,
            )
        return Generation(
            rationale="The target is visible, but I cannot connect it to the question.",
            answer="unknown",
            mode="rationalized",
            hint_seen=answer_hint,
        )


def normalize_answer(answer: str) -> str:
    return " ".join(answer.strip().lower().split())


def answer_matches(predicted: str, expected: str) -> bool:
    """原始 STaR 的核心过滤信号：只检查最终答案。"""
    return normalize_answer(predicted) == normalize_answer(expected)


def collect_iteration(
    model: ToyReasoner,
    problems: Sequence[Problem],
    *,
    use_rationalization: bool,
    rationale_filter: Callable[[Problem, Generation], bool] | None = None,
) -> tuple[list[TrainingExample], int, int]:
    """构造一轮 STaR 数据集，不累计上一轮的文本样本。"""
    kept: list[TrainingExample] = []
    failed: list[Problem] = []
    forward_kept = 0
    rationalized_kept = 0

    for problem in problems:
        generation = model.generate(problem)
        if answer_matches(generation.answer, problem.answer):
            if rationale_filter is None or rationale_filter(problem, generation):
                kept.append(to_training_example(problem, generation))
                forward_kept += 1
        else:
            failed.append(problem)

    if use_rationalization:
        for problem in failed:
            generation = model.generate(problem, answer_hint=problem.answer)
            if answer_matches(generation.answer, problem.answer):
                if rationale_filter is None or rationale_filter(problem, generation):
                    example = to_training_example(problem, generation)
                    # 论文的关键做法：hint 只用于生成，不进入微调输入。
                    assert generation.hint_seen is not None
                    assert "HINT:" not in example.serialized()
                    kept.append(example)
                    rationalized_kept += 1

    return kept, forward_kept, rationalized_kept


def to_training_example(problem: Problem, generation: Generation) -> TrainingExample:
    return TrainingExample(
        problem_id=problem.problem_id,
        question=problem.question,
        rationale=generation.rationale,
        answer=problem.answer,
        source=generation.mode,
        difficulty=problem.difficulty,
    )


def finetune_fresh_from_base(
    base_model: ToyReasoner, training_data: Sequence[TrainingExample]
) -> ToyReasoner:
    """模拟论文中的 M_n = train(M, D_n)：每轮都从原始 M 开始。"""
    mastered = base_model.mastered_difficulty
    if training_data:
        mastered = max(mastered, max(item.difficulty for item in training_data))
    return ToyReasoner(
        base_version=base_model.base_version,
        mastered_difficulty=mastered,
        trained_example_ids=frozenset(item.problem_id for item in training_data),
    )


def run_star(
    base_model: ToyReasoner,
    problems: Sequence[Problem],
    *,
    outer_loops: int,
    use_rationalization: bool,
) -> tuple[ToyReasoner, list[IterationStats]]:
    model = base_model
    history: list[IterationStats] = []

    for iteration in range(1, outer_loops + 1):
        data, forward_kept, rationalized_kept = collect_iteration(
            model,
            problems,
            use_rationalization=use_rationalization,
        )
        # 注意传入的是 base_model，而不是 model。
        model = finetune_fresh_from_base(base_model, data)
        history.append(
            IterationStats(
                iteration=iteration,
                forward_kept=forward_kept,
                rationalized_kept=rationalized_kept,
                coverage=len(data),
                mastered_difficulty=model.mastered_difficulty,
                trained_from_base_version=model.base_version,
            )
        )

    return model, history


def final_answer_filter_can_accept_bad_reasoning(problem: Problem) -> None:
    """展示“答案正确 ⇒ 理由正确”这一假设为何危险。"""
    lucky_guess = Generation(
        rationale="The answer is correct because the answer must be correct.",
        answer=problem.answer,
        mode="forward",
    )
    assert answer_matches(lucky_guess.answer, problem.answer)
    contains_demo_reasoning = problem.gold_rationale_for_demo in lucky_guess.rationale
    print("answer filter accepts lucky guess:", True)
    print("rationale contains the worked reasoning:", contains_demo_reasoning)


def make_curriculum() -> list[Problem]:
    return [
        Problem("p0", "What is 1 + 1?", "2", 0, "1 plus 1 equals 2."),
        Problem("p1", "What is 7 + 5?", "12", 1, "7 plus 5 equals 12."),
        Problem(
            "p2",
            "A box has 4 rows of 6 marbles. How many marbles?",
            "24",
            2,
            "Four equal rows of six give 4 times 6, which equals 24.",
        ),
        Problem(
            "p3",
            "Thirty stickers are shared equally by 5 children. How many each?",
            "6",
            3,
            "Equal sharing means 30 divided by 5, which equals 6.",
        ),
        Problem(
            "p4",
            "A book costs 8 coins after a 20% discount. What was its price?",
            "10",
            4,
            "Eight coins is 80 percent of the original, so 8 divided by 0.8 equals 10.",
        ),
    ]


def print_history(name: str, history: Iterable[IterationStats]) -> None:
    print(f"\n== {name} ==")
    print("iter | forward | rationalized | dataset | mastered | fresh base")
    for item in history:
        print(
            f"{item.iteration:>4} | {item.forward_kept:>7} | "
            f"{item.rationalized_kept:>12} | {item.coverage:>7} | "
            f"{item.mastered_difficulty:>8} | {item.trained_from_base_version}"
        )


def main() -> None:
    problems = make_curriculum()
    base = ToyReasoner(base_version="same-pretrained-M", mastered_difficulty=0)

    _, plain_history = run_star(
        base, problems, outer_loops=4, use_rationalization=False
    )
    _, rationalized_history = run_star(
        base, problems, outer_loops=4, use_rationalization=True
    )

    print_history("STaR without rationalization: plateau", plain_history)
    print_history("STaR with rationalization: expands one frontier per loop", rationalized_history)

    print("\n== Hint removal invariant ==")
    model = replace_for_demo(base, mastered_difficulty=0)
    data, _, _ = collect_iteration(model, problems[:2], use_rationalization=True)
    rationalized = next(item for item in data if item.source == "rationalized")
    print(rationalized.serialized())
    print("serialized training example contains HINT:", "HINT:" in rationalized.serialized())

    print("\n== Final-answer filtering blind spot ==")
    final_answer_filter_can_accept_bad_reasoning(problems[2])


def replace_for_demo(model: ToyReasoner, *, mastered_difficulty: int) -> ToyReasoner:
    """dataclasses.replace 不适用于普通类；保留明确构造以便审计。"""
    return ToyReasoner(
        base_version=model.base_version,
        mastered_difficulty=mastered_difficulty,
        trained_example_ids=model.trained_example_ids,
    )


if __name__ == "__main__":
    main()
