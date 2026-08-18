#!/usr/bin/env python3
"""Voyager 的零依赖教学实现。

它保留论文中的四条主线：自动课程、技能检索、执行反馈/自验证、
以及“验证成功后才写入”的技能库。真实 Voyager 让 GPT-4 生成
Mineflayer JavaScript；这里改用白名单结构化动作，便于安全运行与测试。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Iterable


@dataclass(frozen=True)
class Action:
    """受控动作 IR；生产系统可把它编译为沙盒中的工具调用。"""

    kind: str
    item: str
    count: int


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    goal_item: str
    goal_count: int
    prerequisites: tuple[str, ...] = ()


@dataclass
class Skill:
    name: str
    description: str
    program: tuple[Action, ...]
    learned_from: str
    uses: int = 0


@dataclass
class WorldState:
    inventory: Counter[str] = field(default_factory=Counter)
    biome: str = "forest"
    nearby_blocks: tuple[str, ...] = ("oak_log", "stone", "iron_ore")
    completed_tasks: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)

    def summary(self) -> str:
        items = ", ".join(
            f"{name}={count}" for name, count in sorted(self.inventory.items()) if count
        )
        return f"biome={self.biome}; nearby={self.nearby_blocks}; inventory=[{items}]"


class ExecutionError(RuntimeError):
    pass


class VoxelWorld:
    """只模拟论文演示所需的资源、合成与工具前置条件。"""

    RECIPES: dict[str, tuple[dict[str, int], dict[str, int], str | None]] = {
        "oak_planks": ({"oak_log": 1}, {"oak_planks": 4}, None),
        "crafting_table": ({"oak_planks": 4}, {"crafting_table": 1}, None),
        "stick": ({"oak_planks": 2}, {"stick": 4}, None),
        "wooden_pickaxe": (
            {"oak_planks": 3, "stick": 2},
            {"wooden_pickaxe": 1},
            "crafting_table",
        ),
        "stone_pickaxe": (
            {"cobblestone": 3, "stick": 2},
            {"stone_pickaxe": 1},
            "crafting_table",
        ),
    }
    MINING: dict[str, tuple[str, str]] = {
        "cobblestone": ("stone", "wooden_pickaxe"),
        "raw_iron": ("iron_ore", "stone_pickaxe"),
    }

    def __init__(self, state: WorldState | None = None) -> None:
        self.state = state or WorldState()

    def execute(self, program: Iterable[Action]) -> list[str]:
        feedback: list[str] = []
        for action in program:
            if action.count <= 0:
                raise ExecutionError("动作数量必须为正数")
            if action.kind == "gather":
                if action.item not in self.state.nearby_blocks:
                    raise ExecutionError(f"附近没有可直接采集的 {action.item}")
                self.state.inventory[action.item] += action.count
                feedback.append(f"采集 {action.count} x {action.item}")
            elif action.kind == "craft":
                self._craft(action.item, action.count)
                feedback.append(f"合成 {action.count} 批 {action.item}")
            elif action.kind == "mine":
                self._mine(action.item, action.count)
                feedback.append(f"开采 {action.count} x {action.item}")
            else:
                raise ExecutionError(f"动作 {action.kind!r} 不在白名单中")
        return feedback

    def _craft(self, item: str, batches: int) -> None:
        if item not in self.RECIPES:
            raise ExecutionError(f"未知配方：{item}")
        inputs, outputs, station = self.RECIPES[item]
        if station and self.state.inventory[station] < 1:
            raise ExecutionError(f"合成 {item} 需要 {station}")
        missing = {
            name: amount * batches - self.state.inventory[name]
            for name, amount in inputs.items()
            if self.state.inventory[name] < amount * batches
        }
        if missing:
            raise ExecutionError(f"合成 {item} 缺少材料：{missing}")
        for name, amount in inputs.items():
            self.state.inventory[name] -= amount * batches
        for name, amount in outputs.items():
            self.state.inventory[name] += amount * batches

    def _mine(self, drop: str, count: int) -> None:
        if drop not in self.MINING:
            raise ExecutionError(f"不知道怎样开采并获得 {drop}")
        block, tool = self.MINING[drop]
        if block not in self.state.nearby_blocks:
            raise ExecutionError(f"附近没有 {block}")
        if self.state.inventory[tool] < 1:
            raise ExecutionError(f"开采 {block} 需要 {tool}")
        self.state.inventory[drop] += count


def tokens(text: str) -> set[str]:
    """小型词袋；真实系统使用 text-embedding-ada-002 向量。"""

    return set(re.findall(r"[a-z0-9_]+", text.lower()))


class SkillLibrary:
    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}

    def add(self, skill: Skill) -> None:
        self.skills[skill.name] = skill

    def retrieve(self, query: str, top_k: int = 5) -> list[Skill]:
        query_tokens = tokens(query)

        def score(skill: Skill) -> float:
            skill_tokens = tokens(skill.description)
            union = query_tokens | skill_tokens
            return len(query_tokens & skill_tokens) / len(union) if union else 0.0

        ranked = sorted(self.skills.values(), key=score, reverse=True)
        return [skill for skill in ranked if score(skill) > 0][:top_k]


class AutomaticCurriculum:
    """选择能力前沿上第一个可达且未完成的任务。"""

    def __init__(self) -> None:
        self.tasks = (
            Task("collect_logs", "collect 3 oak logs", "oak_log", 3),
            Task("make_planks", "craft 12 oak planks", "oak_planks", 12, ("collect_logs",)),
            Task("make_table", "craft a crafting table", "crafting_table", 1, ("make_planks",)),
            Task("make_sticks", "craft 4 sticks", "stick", 4, ("make_table",)),
            Task("wooden_pickaxe", "craft a wooden pickaxe", "wooden_pickaxe", 1, ("make_sticks",)),
            Task("mine_stone", "mine 3 cobblestone", "cobblestone", 3, ("wooden_pickaxe",)),
            Task("more_sticks", "craft 2 more sticks", "stick", 2, ("mine_stone",)),
            Task("stone_pickaxe", "craft a stone pickaxe", "stone_pickaxe", 1, ("more_sticks",)),
            Task("mine_iron", "mine 3 raw iron", "raw_iron", 3, ("stone_pickaxe",)),
        )

    def propose(self, state: WorldState) -> Task | None:
        done = set(state.completed_tasks)
        candidates = [
            task
            for task in self.tasks
            if task.name not in done and set(task.prerequisites) <= done
        ]
        return candidates[0] if candidates else None


class ActionAgent:
    """用规则代替 GPT-4，但保留“代码生成—反馈后修复”的接口。"""

    def generate(
        self,
        task: Task,
        state: WorldState,
        retrieved: list[Skill],
        error: str | None,
        critique: str | None,
        attempt: int,
    ) -> tuple[Action, ...]:
        # 同目标技能可直接复用；程序是可组合的长时动作，而非一帧键鼠操作。
        for skill in retrieved:
            if tokens(task.goal_item.replace("_", " ")) <= tokens(skill.description):
                skill.uses += 1
                return skill.program

        # 故意保留两种首轮错误，展示“环境进度反馈”和“解释器错误”。
        if task.name == "collect_logs" and attempt == 1:
            return (Action("gather", "oak_log", 1),)
        if task.name == "mine_iron" and attempt == 1:
            return (Action("mine", "iron_ore", task.goal_count),)

        if task.name == "collect_logs":
            missing = max(1, task.goal_count - state.inventory[task.goal_item])
            return (Action("gather", "oak_log", missing),)
        if task.name == "make_planks":
            # 一根原木生成四块木板；按仍缺的目标数量计算批次。
            missing = max(1, task.goal_count - state.inventory[task.goal_item])
            batches = (missing + 3) // 4
            return (Action("craft", "oak_planks", batches),)
        if task.name == "make_table":
            return (Action("craft", "crafting_table", 1),)
        if task.name in {"make_sticks", "more_sticks"}:
            return (Action("craft", "stick", 1),)
        if task.name == "wooden_pickaxe":
            return (Action("craft", "wooden_pickaxe", 1),)
        if task.name == "mine_stone":
            return (Action("mine", "cobblestone", task.goal_count),)
        if task.name == "stone_pickaxe":
            return (Action("craft", "stone_pickaxe", 1),)
        if task.name == "mine_iron":
            return (Action("mine", "raw_iron", task.goal_count),)
        raise KeyError(task.name)

    def canonical_program(self, task: Task) -> tuple[Action, ...]:
        """把最终修补结果泛化成可从零复用的技能。"""

        canonical = {
            "collect_logs": (Action("gather", "oak_log", 3),),
            "make_planks": (Action("craft", "oak_planks", 2),),
            "make_table": (Action("craft", "crafting_table", 1),),
            "make_sticks": (Action("craft", "stick", 1),),
            "more_sticks": (Action("craft", "stick", 1),),
            "wooden_pickaxe": (Action("craft", "wooden_pickaxe", 1),),
            "mine_stone": (Action("mine", "cobblestone", 3),),
            "stone_pickaxe": (Action("craft", "stone_pickaxe", 1),),
            "mine_iron": (Action("mine", "raw_iron", 3),),
        }
        return canonical[task.name]


class CriticAgent:
    def verify(self, task: Task, state: WorldState) -> tuple[bool, str]:
        actual = state.inventory[task.goal_item]
        if actual >= task.goal_count:
            return True, ""
        return False, f"还需要 {task.goal_count - actual} x {task.goal_item}"


class VoyagerAgent:
    def __init__(self, world: VoxelWorld, library: SkillLibrary | None = None) -> None:
        self.world = world
        self.library = library or SkillLibrary()
        self.curriculum = AutomaticCurriculum()
        self.action_agent = ActionAgent()
        self.critic = CriticAgent()

    def learn_task(self, task: Task, max_rounds: int = 4) -> bool:
        error: str | None = None
        critique: str | None = None
        print(f"\nTASK  {task.description}")
        for attempt in range(1, max_rounds + 1):
            query = f"{task.description}; {self.world.state.summary()}; {critique or ''}"
            skills = self.library.retrieve(query, top_k=5)
            program = self.action_agent.generate(
                task, self.world.state, skills, error, critique, attempt
            )
            print(f"  round {attempt}: retrieved={[s.name for s in skills]} program={program}")
            try:
                feedback = self.world.execute(program)
                error = None
                print(f"           environment={feedback}")
            except ExecutionError as exc:
                error = str(exc)
                print(f"           execution_error={error}")

            success, critique = self.critic.verify(task, self.world.state)
            print(f"           verified={success} critique={critique!r}")
            if success:
                self.world.state.completed_tasks.append(task.name)
                try:
                    reusable_program = self.action_agent.canonical_program(task)
                except KeyError:
                    reusable_program = program
                self.library.add(
                    Skill(
                        name=task.name,
                        description=task.description,
                        program=reusable_program,
                        learned_from=task.name,
                    )
                )
                return True

        self.world.state.failed_tasks.append(task.name)
        return False

    def learn(self) -> None:
        while (task := self.curriculum.propose(self.world.state)) is not None:
            if not self.learn_task(task):
                # 与论文一样：达到重试上限后把目标记为 failed，再让课程改题。
                # 这个线性教学课程没有旁支，因此直接结束，避免无限重试。
                break


def main() -> None:
    world = VoxelWorld()
    agent = VoyagerAgent(world)
    agent.learn()

    print("\nLEARNED", list(agent.library.skills))
    print("FINAL  ", world.state.summary())

    # 新世界清空物品，但保留技能库；先放入满足前置条件的装备，展示检索复用。
    transfer_state = WorldState(inventory=Counter({"stone_pickaxe": 1}))
    transfer = VoyagerAgent(VoxelWorld(transfer_state), library=agent.library)
    novel_task = Task("transfer_iron", "mine 2 raw iron in a new world", "raw_iron", 2)
    ok = transfer.learn_task(novel_task)
    print(f"\nTRANSFER success={ok}; inventory={dict(transfer_state.inventory)}")


if __name__ == "__main__":
    main()
