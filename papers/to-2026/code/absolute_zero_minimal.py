#!/usr/bin/env python3
"""Absolute Zero Reasoner (AZR) 的零依赖教学实现。

这个脚本把论文的关键闭环压缩成一个可以在 CPU 上直接运行的玩具系统：

1. 同一个 ``ToyPolicy`` 既是 proposer，也是 solver；
2. proposer 从程序语法空间生成 deduction / abduction / induction 任务；
3. ``SafeExpression`` 充当可验证环境，构造标签并检查答案；
4. proposer 用多次 solver rollout 估计 learnability reward；
5. 六个 task-role 组分别标准化奖励，演示 TRR++ 的核心思想；
6. solver 的能力参数随自博弈更新，形成一个极小的自动课程。

它不是官方训练代码，也不训练语言模型。为了避免把论文原型中执行任意 Python
的风险带进教学脚本，这里不使用 ``eval`` / ``exec``，而是解释一个严格白名单的
算术 AST。运行 ``python absolute_zero_minimal.py --test`` 执行自测。
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from enum import Enum
import math
import operator
import random
from typing import Iterable, Sequence


class UnsafeExpression(ValueError):
    """表达式超出教学执行器白名单。"""


class SafeExpression:
    """只解释整数算术的微型、确定性执行环境。

    支持变量 ``x``、整数常量、``+ - * // % **`` 和一元正负号。没有函数调用、
    属性访问、下标、导入或 I/O；幂和中间结果也有限制，以免资源失控。
    """

    _binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def __init__(self, source: str, result_limit: int = 10**9) -> None:
        self.source = source
        self.result_limit = result_limit
        try:
            self.tree = ast.parse(source, mode="eval")
        except SyntaxError as error:
            raise UnsafeExpression(str(error)) from error
        self._validate(self.tree)

    def _validate(self, node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            self._validate(node.body)
            return
        if isinstance(node, ast.Constant):
            if type(node.value) is not int:  # bool 也是 int 子类，所以用 type 精确判断
                raise UnsafeExpression("only integer constants are allowed")
            return
        if isinstance(node, ast.Name):
            if node.id != "x":
                raise UnsafeExpression("only the variable x is allowed")
            return
        if isinstance(node, ast.BinOp):
            if type(node.op) not in self._binary_ops:
                raise UnsafeExpression(f"operator {type(node.op).__name__} is forbidden")
            self._validate(node.left)
            self._validate(node.right)
            return
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in self._unary_ops:
                raise UnsafeExpression(f"operator {type(node.op).__name__} is forbidden")
            self._validate(node.operand)
            return
        raise UnsafeExpression(f"node {type(node).__name__} is forbidden")

    def __call__(self, x: int) -> int:
        if type(x) is not int:
            raise TypeError("x must be an integer")
        return self._evaluate(self.tree.body, x)

    def _evaluate(self, node: ast.AST, x: int) -> int:
        if isinstance(node, ast.Constant):
            result = node.value
        elif isinstance(node, ast.Name):
            result = x
        elif isinstance(node, ast.UnaryOp):
            result = self._unary_ops[type(node.op)](self._evaluate(node.operand, x))
        elif isinstance(node, ast.BinOp):
            left = self._evaluate(node.left, x)
            right = self._evaluate(node.right, x)
            if isinstance(node.op, ast.Pow) and (right < 0 or right > 6):
                raise UnsafeExpression("power must be between 0 and 6")
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and right == 0:
                raise ZeroDivisionError("division by zero")
            result = self._binary_ops[type(node.op)](left, right)
        else:  # pragma: no cover - 构造时已经拦截
            raise UnsafeExpression(type(node).__name__)
        if type(result) is not int or abs(result) > self.result_limit:
            raise UnsafeExpression("result exceeds the bounded integer environment")
        return result


class Mode(str, Enum):
    DEDUCTION = "deduction"  # (program, input) -> output
    ABDUCTION = "abduction"  # (program, output) -> any matching input
    INDUCTION = "induction"  # examples -> program


@dataclass(frozen=True)
class Program:
    expression: str
    level: int

    def run(self, x: int) -> int:
        return SafeExpression(self.expression)(x)


@dataclass(frozen=True)
class Task:
    mode: Mode
    program: Program
    gold_input: int
    gold_output: int
    visible_examples: tuple[tuple[int, int], ...] = ()
    hidden_examples: tuple[tuple[int, int], ...] = ()
    message: str = "Infer the deterministic integer transformation."

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.mode.value, self.program.expression, self.gold_input)


@dataclass(frozen=True)
class Answer:
    value: int | str | None


@dataclass(frozen=True)
class RolloutRecord:
    mode: Mode
    role: str
    reward: float
    advantage: float = 0.0


PROGRAMS = (
    Program("x", 0),
    Program("x + 1", 1),
    Program("2 * x - 3", 1),
    Program("x * x + 1", 2),
    Program("(x + 2) * (x - 1)", 2),
    Program("x * x - 3 * x + 2", 3),
    Program("(x * x + 3 * x) % 11", 3),
    Program("(x ** 3 - 2 * x + 5) % 17", 4),
)


def make_task(mode: Mode, program: Program, seed_input: int) -> Task:
    """由执行环境补齐标签，构造论文中的 program/input/output 三元组。"""

    output = program.run(seed_input)
    if mode is not Mode.INDUCTION:
        return Task(mode, program, seed_input, output)

    inputs = (-6, -4, -2, 0, 2, 4, 6, 8)
    pairs = tuple((value, program.run(value)) for value in inputs)
    midpoint = len(pairs) // 2
    return Task(
        mode,
        program,
        seed_input,
        output,
        visible_examples=pairs[:midpoint],
        hidden_examples=pairs[midpoint:],
    )


def validate_task(task: Task) -> bool:
    """检查可执行性和确定性；论文实现用独立执行两次作近似检查。"""

    try:
        first = task.program.run(task.gold_input)
        second = task.program.run(task.gold_input)
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    if first != second or first != task.gold_output:
        return False
    if task.mode is Mode.INDUCTION:
        return all(task.program.run(x) == y for x, y in task.visible_examples + task.hidden_examples)
    return True


def verify(task: Task, answer: Answer) -> bool:
    """按三种 reasoning mode 的不同等价关系验证 solver 输出。"""

    try:
        if task.mode is Mode.DEDUCTION:
            return type(answer.value) is int and answer.value == task.gold_output
        if task.mode is Mode.ABDUCTION:
            # program 可能不是单射，因此不要求复原 proposer 的那个输入。
            return type(answer.value) is int and task.program.run(answer.value) == task.gold_output
        if task.mode is Mode.INDUCTION:
            if not isinstance(answer.value, str):
                return False
            candidate = SafeExpression(answer.value)
            return all(candidate(x) == y for x, y in task.hidden_examples)
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    raise AssertionError(task.mode)


def learnability_reward(binary_rewards: Sequence[float]) -> float:
    """论文 Equation (4)：完全不可解与完全可解都不给 proposer 奖励。"""

    if not binary_rewards:
        raise ValueError("at least one solver rollout is required")
    success_rate = sum(binary_rewards) / len(binary_rewards)
    return 0.0 if success_rate == 0.0 else 1.0 - success_rate


def task_relative_advantages(records: Sequence[RolloutRecord]) -> list[RolloutRecord]:
    """按 (task, role) 六组分别标准化，保留 TRR++ 最关键的 baseline 设计。"""

    grouped: dict[tuple[Mode, str], list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault((record.mode, record.role), []).append(index)

    advantages = [0.0] * len(records)
    for indices in grouped.values():
        rewards = [records[index].reward for index in indices]
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        std = math.sqrt(variance)
        if std == 0.0:
            continue
        for index in indices:
            advantages[index] = (records[index].reward - mean) / std
    return [
        RolloutRecord(item.mode, item.role, item.reward, advantage)
        for item, advantage in zip(records, advantages)
    ]


class ToyPolicy:
    """同一策略的 proposer / solver 两个角色。

    ``skill`` 不是神经网络参数，只是让 demo 展示“当前能力决定合适难度”的标量。
    """

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.skill = {mode: 0.8 for mode in Mode}
        self.history: set[tuple[str, str, int]] = set()

    def propose(self, mode: Mode) -> Task:
        target_level = max(0, min(4, round(self.skill[mode] + self.rng.uniform(0.2, 1.8))))
        candidates = sorted(PROGRAMS, key=lambda program: abs(program.level - target_level))
        for program in candidates:
            seed_input = self.rng.randint(-8, 8)
            task = make_task(mode, program, seed_input)
            if task.key not in self.history and validate_task(task):
                self.history.add(task.key)
                return task
        # 语法空间很小，全部见过后允许复用，但换一个输入。
        program = self.rng.choice(PROGRAMS)
        return make_task(mode, program, self.rng.randint(-12, 12))

    def solve(self, task: Task) -> Answer:
        margin = self.skill[task.mode] - task.program.level
        success_probability = 1.0 / (1.0 + math.exp(-1.35 * margin))
        if self.rng.random() > success_probability:
            return Answer(None)

        if task.mode is Mode.DEDUCTION:
            return Answer(task.program.run(task.gold_input))
        if task.mode is Mode.ABDUCTION:
            candidates = list(range(-16, 17))
            self.rng.shuffle(candidates)
            for candidate in candidates:
                if task.program.run(candidate) == task.gold_output:
                    return Answer(candidate)
            return Answer(None)
        if task.mode is Mode.INDUCTION:
            fitting = [
                program
                for program in PROGRAMS
                if all(program.run(x) == y for x, y in task.visible_examples)
            ]
            return Answer(fitting[0].expression if fitting else None)
        raise AssertionError(task.mode)

    def learn(self, task: Task, solver_rewards: Sequence[float], rate: float = 0.22) -> None:
        """用已验证成功样本推动能力；只是可视化自举，不是策略梯度。"""

        success_rate = sum(solver_rewards) / len(solver_rewards)
        if 0.0 < success_rate < 1.0:
            gap = task.program.level + 0.7 - self.skill[task.mode]
            self.skill[task.mode] += rate * max(0.05, gap) * success_rate
            self.skill[task.mode] = min(4.5, self.skill[task.mode])


def self_play_round(policy: ToyPolicy, tasks_per_mode: int, rollouts: int) -> list[RolloutRecord]:
    """执行一次 propose → validate → solve → reward → update 闭环。"""

    records: list[RolloutRecord] = []
    for mode in Mode:
        for _ in range(tasks_per_mode):
            task = policy.propose(mode)
            solver_rewards = [float(verify(task, policy.solve(task))) for _ in range(rollouts)]
            proposal_reward = learnability_reward(solver_rewards)
            records.append(RolloutRecord(mode, "propose", proposal_reward))
            records.extend(RolloutRecord(mode, "solve", reward) for reward in solver_rewards)
            policy.learn(task, solver_rewards)
    return task_relative_advantages(records)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def run_demo(rounds: int, tasks_per_mode: int, rollouts: int, seed: int) -> None:
    policy = ToyPolicy(seed)
    print("Absolute Zero toy loop (safe AST environment, no external dataset)")
    print("round | deduction | abduction | induction | proposer-r | solver-r")
    print("------+-----------+-----------+-----------+------------+---------")
    for round_index in range(1, rounds + 1):
        records = self_play_round(policy, tasks_per_mode, rollouts)
        proposer_mean = mean(item.reward for item in records if item.role == "propose")
        solver_mean = mean(item.reward for item in records if item.role == "solve")
        print(
            f"{round_index:>5} | "
            f"{policy.skill[Mode.DEDUCTION]:>9.2f} | "
            f"{policy.skill[Mode.ABDUCTION]:>9.2f} | "
            f"{policy.skill[Mode.INDUCTION]:>9.2f} | "
            f"{proposer_mean:>10.3f} | {solver_mean:>7.3f}"
        )


def run_tests() -> None:
    assert SafeExpression("(x + 2) * (x - 1)")(4) == 18
    for forbidden in ("__import__('os')", "x.__class__", "[x][0]", "open('x')"):
        try:
            SafeExpression(forbidden)
        except UnsafeExpression:
            pass
        else:
            raise AssertionError(f"unsafe expression accepted: {forbidden}")

    deduction = make_task(Mode.DEDUCTION, Program("2 * x - 3", 1), 5)
    assert verify(deduction, Answer(7))
    assert not verify(deduction, Answer(8))

    abduction = make_task(Mode.ABDUCTION, Program("x * x", 2), -4)
    assert verify(abduction, Answer(4))  # 不要求与 gold input -4 字面相同

    induction = make_task(Mode.INDUCTION, Program("x * x + 1", 2), 3)
    assert verify(induction, Answer("x * x + 1"))
    assert not verify(induction, Answer("x + 1"))

    assert learnability_reward([0.0] * 8) == 0.0
    assert learnability_reward([1.0] * 8) == 0.0
    assert learnability_reward([1.0, 0.0] * 4) == 0.5

    records = [
        RolloutRecord(Mode.DEDUCTION, "solve", 0.0),
        RolloutRecord(Mode.DEDUCTION, "solve", 1.0),
        RolloutRecord(Mode.ABDUCTION, "solve", 1.0),
        RolloutRecord(Mode.ABDUCTION, "solve", 1.0),
    ]
    normalized = task_relative_advantages(records)
    assert [item.advantage for item in normalized[:2]] == [-1.0, 1.0]
    assert [item.advantage for item in normalized[2:]] == [0.0, 0.0]

    policy = ToyPolicy(seed=3)
    before = dict(policy.skill)
    result = self_play_round(policy, tasks_per_mode=3, rollouts=8)
    assert len(result) == 3 * 3 * 9  # 每题 1 个 propose + 8 个 solve record
    assert all(policy.skill[mode] >= before[mode] for mode in Mode)
    print("all tests passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--tasks-per-mode", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test:
        run_tests()
        return
    if min(args.rounds, args.tasks_per_mode, args.rollouts) <= 0:
        raise SystemExit("rounds, tasks-per-mode and rollouts must be positive")
    run_demo(args.rounds, args.tasks_per_mode, args.rollouts, args.seed)


if __name__ == "__main__":
    main()
