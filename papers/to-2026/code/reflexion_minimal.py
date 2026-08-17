#!/usr/bin/env python3
"""Dependency-free teaching implementation of the Reflexion outer loop.

Reflexion does not update language-model weights.  It runs an Actor for one
complete trial, evaluates the trajectory, asks a Self-Reflection component to
turn the failure into an actionable language lesson, stores that lesson in a
bounded episodic memory, resets the environment, and tries again.

This file uses deterministic stand-ins so the full information flow can be run
without downloading a model or using an API key.  Replace ``ScriptedActor`` and
``RuleReflector`` with LLM-backed adapters to build a real language agent; keep
the trial budget, environment reset, evaluation, and memory bounds in the
controller.

Run:

    python3 papers/to-2026/code/reflexion_minimal.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Transition:
    """One environment transition inside a trial."""

    observation: str
    action: str
    next_observation: str
    reward: int
    done: bool


@dataclass(frozen=True)
class Evaluation:
    """Trial-level result produced by the Evaluator."""

    passed: bool
    score: int
    feedback: str


@dataclass(frozen=True)
class Trial:
    """A complete attempt plus its evaluation and optional reflection."""

    index: int
    trajectory: tuple[Transition, ...]
    evaluation: Evaluation
    reflection: str | None


@dataclass(frozen=True)
class RunResult:
    success: bool
    trials: tuple[Trial, ...]
    memory: tuple[str, ...]


class Environment(Protocol):
    """Resettable environment used for episodic trial-and-error."""

    def reset(self) -> str: ...

    def step(self, action: str) -> tuple[str, int, bool]: ...


class Actor(Protocol):
    """Policy role: choose an action from state, trajectory, and memory."""

    def act(
        self,
        task: str,
        observation: str,
        trajectory: Sequence[Transition],
        memory: Sequence[str],
    ) -> str: ...


class Evaluator(Protocol):
    """Evaluator role: turn a completed trajectory into trial feedback."""

    def evaluate(
        self, task: str, trajectory: Sequence[Transition]
    ) -> Evaluation: ...


class SelfReflector(Protocol):
    """Self-Reflection role: convert feedback into a verbal lesson."""

    def reflect(
        self,
        task: str,
        trajectory: Sequence[Transition],
        evaluation: Evaluation,
        memory: Sequence[str],
    ) -> str: ...


class EpisodicMemory:
    """A sliding window matching the paper's bounded reflection memory."""

    def __init__(self, capacity: int = 3) -> None:
        if capacity <= 0:
            raise ValueError("memory capacity must be positive")
        self._items: deque[str] = deque(maxlen=capacity)

    def append(self, reflection: str) -> None:
        compact = " ".join(reflection.split())
        if not compact:
            raise ValueError("reflection cannot be empty")
        self._items.append(compact)

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self._items)


class ReflexionAgent:
    """Bounded Actor -> Evaluator -> Reflector -> Memory controller."""

    def __init__(
        self,
        actor: Actor,
        evaluator: Evaluator,
        reflector: SelfReflector,
        *,
        max_trials: int = 3,
        max_steps_per_trial: int = 2,
        memory_capacity: int = 3,
        use_memory: bool = True,
    ) -> None:
        if max_trials <= 0 or max_steps_per_trial <= 0:
            raise ValueError("trial and step budgets must be positive")
        self.actor = actor
        self.evaluator = evaluator
        self.reflector = reflector
        self.max_trials = max_trials
        self.max_steps_per_trial = max_steps_per_trial
        self.memory = EpisodicMemory(memory_capacity)
        self.use_memory = use_memory

    def run(self, task: str, environment: Environment) -> RunResult:
        if not task.strip():
            raise ValueError("task cannot be empty")

        trials: list[Trial] = []
        for trial_index in range(self.max_trials):
            observation = environment.reset()
            trajectory: list[Transition] = []

            for _ in range(self.max_steps_per_trial):
                action = self.actor.act(
                    task,
                    observation,
                    trajectory,
                    self.memory.snapshot() if self.use_memory else (),
                )
                next_observation, reward, done = environment.step(action)
                trajectory.append(
                    Transition(
                        observation=observation,
                        action=action,
                        next_observation=next_observation,
                        reward=reward,
                        done=done,
                    )
                )
                observation = next_observation
                if done:
                    break

            evaluation = self.evaluator.evaluate(task, trajectory)
            if evaluation.passed:
                trials.append(
                    Trial(trial_index, tuple(trajectory), evaluation, None)
                )
                return RunResult(True, tuple(trials), self.memory.snapshot())

            reflection: str | None = None
            if self.use_memory:
                reflection = self.reflector.reflect(
                    task,
                    trajectory,
                    evaluation,
                    self.memory.snapshot(),
                )
                self.memory.append(reflection)
            trials.append(
                Trial(trial_index, tuple(trajectory), evaluation, reflection)
            )

        return RunResult(False, tuple(trials), self.memory.snapshot())


class KeyDoorEnvironment:
    """Tiny deterministic task: take a key, then open a locked door."""

    def reset(self) -> str:
        self.has_key = False
        self.is_open = False
        return "You are in a room with a brass key and a locked door."

    def step(self, action: str) -> tuple[str, int, bool]:
        if action == "take_key":
            if self.has_key:
                return "You already have the key.", 0, False
            self.has_key = True
            return "You take the brass key.", 0, False
        if action == "open_door":
            if not self.has_key:
                return "The door is locked; you do not have the key.", 0, False
            self.is_open = True
            return "The key turns and the door opens.", 1, True
        return f"Unknown action: {action!r}.", 0, False


class ScriptedActor:
    """Deterministic stand-in showing exactly how memory changes a policy."""

    def act(
        self,
        task: str,
        observation: str,
        trajectory: Sequence[Transition],
        memory: Sequence[str],
    ) -> str:
        del task, observation
        lesson = " ".join(memory).casefold()
        if "take the key first" in lesson:
            already_took_key = any(step.action == "take_key" for step in trajectory)
            return "open_door" if already_took_key else "take_key"
        # The baseline repeats the attractive but ineffective direct action.
        return "open_door"


class BinaryEvaluator:
    """Use the environment's terminal reward as a sparse trial-level signal."""

    def evaluate(
        self, task: str, trajectory: Sequence[Transition]
    ) -> Evaluation:
        del task
        passed = bool(trajectory and trajectory[-1].done)
        if passed:
            return Evaluation(True, 1, "The door opened; the task is complete.")
        return Evaluation(
            False,
            0,
            "The trial ended without opening the door.",
        )


class RuleReflector:
    """Deterministic substitute for the paper's LLM Self-Reflection role."""

    def reflect(
        self,
        task: str,
        trajectory: Sequence[Transition],
        evaluation: Evaluation,
        memory: Sequence[str],
    ) -> str:
        del task, memory
        repeated_open = sum(step.action == "open_door" for step in trajectory) > 1
        saw_locked = any("locked" in step.next_observation for step in trajectory)
        if evaluation.score == 0 and saw_locked:
            repetition = " I repeated an ineffective action." if repeated_open else ""
            return (
                "I tried to open the locked door without the key."
                f"{repetition} In the next trial, take the key first, then open "
                "the door; do not repeat an action when the observation does not change."
            )
        return (
            "The previous trial failed. Re-read the observations, identify the first "
            "state-changing prerequisite, and try a different action sequence."
        )


def render(result: RunResult, label: str) -> None:
    print(f"\n=== {label}: success={result.success} ===")
    for trial in result.trials:
        print(f"Trial {trial.index}")
        for step in trial.trajectory:
            print(f"  Action: {step.action}")
            print(f"  Observation: {step.next_observation}")
        print(
            f"  Evaluation: score={trial.evaluation.score}; "
            f"{trial.evaluation.feedback}"
        )
        if trial.reflection is not None:
            print(f"  Reflection: {trial.reflection}")
    print(f"Final memory: {list(result.memory)}")


def main() -> None:
    task = "Open the locked door."

    baseline = ReflexionAgent(
        ScriptedActor(),
        BinaryEvaluator(),
        RuleReflector(),
        use_memory=False,
    ).run(task, KeyDoorEnvironment())
    reflexion = ReflexionAgent(
        ScriptedActor(),
        BinaryEvaluator(),
        RuleReflector(),
        use_memory=True,
    ).run(task, KeyDoorEnvironment())

    assert not baseline.success
    assert reflexion.success
    assert len(reflexion.trials) == 2
    assert reflexion.trials[-1].trajectory[-1].action == "open_door"

    render(baseline, "baseline without persistent reflection")
    render(reflexion, "Reflexion with episodic memory")


if __name__ == "__main__":
    main()
