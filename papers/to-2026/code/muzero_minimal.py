#!/usr/bin/env python3
"""Zero-dependency miniature of MuZero on a five-cell line world.

The environment knows its transition rules, but MCTS never calls those rules. It
plans only with a learned latent model:

    h(observation history) -> hidden state
    g(hidden state, action) -> predicted reward, next hidden state
    f(hidden state)         -> policy, value

The tables below replace MuZero's residual networks so that the full control loop
is readable and runs in under a second. This is an algorithm demonstration, not
a performance reproduction.

Run:
    python3 muzero_minimal.py --test
    python3 muzero_minimal.py --iterations 20 --games 12 --simulations 32
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTIONS = (ACTION_LEFT, ACTION_RIGHT)


class LineWorld:
    """Real environment. Its transition method is not available to search."""

    def __init__(self, start: int = 2, size: int = 5, max_steps: int = 6) -> None:
        self.position = start
        self.size = size
        self.max_steps = max_steps
        self.steps = 0

    def observation(self) -> int:
        return self.position

    def terminal(self) -> bool:
        return self.position in (0, self.size - 1) or self.steps >= self.max_steps

    def step(self, action: int) -> Tuple[int, float, bool]:
        if self.terminal():
            raise RuntimeError("cannot step a terminal environment")
        if action not in ACTIONS:
            raise ValueError(f"invalid action: {action}")
        self.position += -1 if action == ACTION_LEFT else 1
        self.position = min(max(self.position, 0), self.size - 1)
        self.steps += 1
        if self.position == self.size - 1:
            reward = 1.0
        elif self.position == 0:
            reward = -1.0
        else:
            reward = -0.01
        return self.observation(), reward, self.terminal()


@dataclass(frozen=True)
class HiddenState:
    """Planning state with no requirement to reconstruct an environment state."""

    root_observation: int
    imagined_actions: Tuple[int, ...] = ()


Policy = Dict[int, float]


def softmax(logits: Sequence[float]) -> Policy:
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = sum(weights)
    return {action: weights[action] / total for action in ACTIONS}


class TabularMuZeroModel:
    """Tabular stand-in for MuZero's representation, dynamics and prediction nets."""

    def __init__(self) -> None:
        self.policy_logits: Dict[HiddenState, List[float]] = {}
        self.values: Dict[HiddenState, float] = {}
        self.rewards: Dict[Tuple[HiddenState, int], float] = {}

    def representation(self, observation_history: Sequence[int]) -> HiddenState:
        # A real network consumes frames/history. The table uses the latest observation
        # only, but still creates a latent token rather than an environment simulator.
        return HiddenState(root_observation=observation_history[-1])

    def dynamics(self, hidden: HiddenState, action: int) -> Tuple[float, HiddenState]:
        reward = self.rewards.get((hidden, action), 0.0)
        next_hidden = HiddenState(
            root_observation=hidden.root_observation,
            imagined_actions=hidden.imagined_actions + (action,),
        )
        return reward, next_hidden

    def prediction(self, hidden: HiddenState) -> Tuple[Policy, float]:
        logits = self.policy_logits.setdefault(hidden, [0.0, 0.0])
        return softmax(logits), self.values.get(hidden, 0.0)

    def prediction_loss(
        self, hidden: HiddenState, target_policy: Policy, target_value: float
    ) -> float:
        policy, value = self.prediction(hidden)
        cross_entropy = -sum(
            target_policy[action] * math.log(max(policy[action], 1e-12))
            for action in ACTIONS
        )
        return cross_entropy + (target_value - value) ** 2

    def train_prediction(
        self,
        hidden: HiddenState,
        target_policy: Policy,
        target_value: float,
        learning_rate: float,
    ) -> float:
        loss = self.prediction_loss(hidden, target_policy, target_value)
        policy, value = self.prediction(hidden)
        logits = self.policy_logits[hidden]
        for action in ACTIONS:
            logits[action] -= learning_rate * (
                policy[action] - target_policy[action]
            )
        self.values[hidden] = value - learning_rate * 2.0 * (
            value - target_value
        )
        return loss

    def train_reward(
        self,
        hidden: HiddenState,
        action: int,
        target_reward: float,
        learning_rate: float,
    ) -> float:
        key = (hidden, action)
        prediction = self.rewards.get(key, 0.0)
        error = prediction - target_reward
        self.rewards[key] = prediction - learning_rate * 2.0 * error
        return error * error


@dataclass
class Node:
    hidden: HiddenState
    reward_from_parent: float = 0.0
    prior: Policy = field(default_factory=dict)
    visits: Dict[int, int] = field(default_factory=dict)
    value_sum: Dict[int, float] = field(default_factory=dict)
    children: Dict[int, "Node"] = field(default_factory=dict)
    expanded: bool = False

    def expand(self, policy: Policy) -> None:
        self.prior = dict(policy)
        self.visits = {action: 0 for action in ACTIONS}
        self.value_sum = {action: 0.0 for action in ACTIONS}
        self.expanded = True

    @property
    def total_visits(self) -> int:
        return sum(self.visits.values())


class LatentMCTS:
    def __init__(
        self,
        model: TabularMuZeroModel,
        simulations: int = 32,
        discount: float = 0.997,
        c1: float = 1.25,
        c2: float = 19652.0,
        max_depth: int = 8,
    ) -> None:
        self.model = model
        self.simulations = simulations
        self.discount = discount
        self.c1 = c1
        self.c2 = c2
        self.max_depth = max_depth

    def _select(self, node: Node) -> int:
        total = node.total_visits
        scale = math.sqrt(max(1, total))
        prior_scale = self.c1 + math.log((total + self.c2 + 1.0) / self.c2)

        q_values = []
        for action in ACTIONS:
            count = node.visits[action]
            q_values.append(node.value_sum[action] / count if count else 0.0)
        low, high = min(q_values), max(q_values)

        def score(action: int) -> float:
            count = node.visits[action]
            q_value = q_values[action]
            if high > low:
                q_value = (q_value - low) / (high - low)
            exploration = (
                node.prior[action]
                * scale
                / (1 + count)
                * prior_scale
            )
            return q_value + exploration

        return max(ACTIONS, key=lambda action: (score(action), action))

    def search(self, observation_history: Sequence[int]) -> Tuple[Node, Policy, float]:
        root_hidden = self.model.representation(observation_history)
        root = Node(root_hidden)
        root_policy, _ = self.model.prediction(root_hidden)
        root.expand(root_policy)

        for _ in range(self.simulations):
            node = root
            # Each entry stores the parent edge and its learned immediate reward.
            path: List[Tuple[Node, int, float]] = []
            depth = 0

            while node.expanded and depth < self.max_depth:
                action = self._select(node)
                if action not in node.children:
                    reward, next_hidden = self.model.dynamics(node.hidden, action)
                    node.children[action] = Node(next_hidden, reward)
                child = node.children[action]
                path.append((node, action, child.reward_from_parent))
                node = child
                depth += 1

            leaf_policy, leaf_value = self.model.prediction(node.hidden)
            if not node.expanded:
                node.expand(leaf_policy)

            # Single-agent backup: include every predicted reward and discount it.
            backed_up = leaf_value
            for parent, action, reward in reversed(path):
                backed_up = reward + self.discount * backed_up
                parent.visits[action] += 1
                parent.value_sum[action] += backed_up

        visit_total = max(1, root.total_visits)
        visit_policy = {
            action: root.visits[action] / visit_total for action in ACTIONS
        }
        root_value = sum(
            root.value_sum[action] for action in ACTIONS
        ) / visit_total
        return root, visit_policy, root_value


@dataclass
class Episode:
    observations: List[int]
    actions: List[int]
    rewards: List[float]
    policies: List[Policy]
    root_values: List[float]


def sample_action(policy: Policy, rng: random.Random) -> int:
    return ACTION_RIGHT if rng.random() < policy[ACTION_RIGHT] else ACTION_LEFT


def play_episode(
    model: TabularMuZeroModel,
    simulations: int,
    rng: random.Random,
    explore: bool,
) -> Episode:
    environment = LineWorld()
    observations = [environment.observation()]
    actions: List[int] = []
    rewards: List[float] = []
    policies: List[Policy] = []
    root_values: List[float] = []

    while not environment.terminal():
        _, policy, root_value = LatentMCTS(
            model, simulations=simulations
        ).search(observations)
        action = sample_action(policy, rng) if explore else max(policy, key=policy.get)
        observation, reward, _ = environment.step(action)
        policies.append(policy)
        root_values.append(root_value)
        actions.append(action)
        rewards.append(reward)
        observations.append(observation)

    return Episode(observations, actions, rewards, policies, root_values)


def n_step_target(
    episode: Episode,
    index: int,
    td_steps: int,
    discount: float,
) -> float:
    target = 0.0
    for offset in range(td_steps):
        reward_index = index + offset
        if reward_index >= len(episode.rewards):
            break
        target += (discount**offset) * episode.rewards[reward_index]
    bootstrap_index = index + td_steps
    if bootstrap_index < len(episode.root_values):
        target += (discount**td_steps) * episode.root_values[bootstrap_index]
    return target


def train_sequence(
    model: TabularMuZeroModel,
    episode: Episode,
    start: int,
    unroll_steps: int = 5,
    td_steps: int = 5,
    discount: float = 0.997,
    learning_rate: float = 0.12,
) -> float:
    hidden = model.representation(episode.observations[: start + 1])
    total_loss = 0.0
    trained_steps = 0

    for step in range(unroll_steps + 1):
        index = start + step
        if index >= len(episode.actions):
            break
        value_target = n_step_target(episode, index, td_steps, discount)
        total_loss += model.train_prediction(
            hidden,
            episode.policies[index],
            value_target,
            learning_rate,
        )
        trained_steps += 1
        if step == unroll_steps:
            break
        action = episode.actions[index]
        total_loss += model.train_reward(
            hidden,
            action,
            episode.rewards[index],
            learning_rate,
        )
        _, hidden = model.dynamics(hidden, action)

    return total_loss / max(1, trained_steps)


def scalar_transform(value: float, epsilon: float = 0.001) -> float:
    """MuZero/R2D2 transform used before Atari categorical encoding."""
    sign = -1.0 if value < 0 else 1.0
    return sign * (math.sqrt(abs(value) + 1.0) - 1.0) + epsilon * value


def inverse_scalar_transform(value: float, epsilon: float = 0.001) -> float:
    sign = -1.0 if value < 0 else 1.0
    inside = math.sqrt(1.0 + 4.0 * epsilon * (abs(value) + 1.0 + epsilon))
    magnitude = ((inside - 1.0) / (2.0 * epsilon)) ** 2 - 1.0
    return sign * magnitude


def encode_support(value: float, support: int = 300) -> Dict[int, float]:
    transformed = min(max(scalar_transform(value), -support), support)
    low = math.floor(transformed)
    high = math.ceil(transformed)
    if low == high:
        return {low: 1.0}
    return {low: high - transformed, high: transformed - low}


def decode_support(distribution: Dict[int, float]) -> float:
    transformed = sum(index * probability for index, probability in distribution.items())
    return inverse_scalar_transform(transformed)


def evaluate(
    model: TabularMuZeroModel,
    games: int,
    simulations: int,
    seed: int,
) -> Tuple[int, int, float]:
    rng = random.Random(seed)
    successes = 0
    failures = 0
    total_return = 0.0
    for _ in range(games):
        episode = play_episode(model, simulations, rng, explore=False)
        episode_return = sum(episode.rewards)
        total_return += episode_return
        if episode.rewards[-1] > 0:
            successes += 1
        else:
            failures += 1
    return successes, failures, total_return / games


def train_demo(
    iterations: int,
    games_per_iteration: int,
    simulations: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    model = TabularMuZeroModel()
    replay: List[Episode] = []

    before = evaluate(model, 100, simulations, seed + 1)
    print(
        f"before training: {before[0]} goal / {before[1]} fail, "
        f"mean_return={before[2]:.3f}"
    )

    for iteration in range(1, iterations + 1):
        for _ in range(games_per_iteration):
            replay.append(play_episode(model, simulations, rng, explore=True))
        replay = replay[-1000:]

        losses = []
        for _ in range(games_per_iteration * 4):
            episode = rng.choice(replay)
            start = rng.randrange(len(episode.actions))
            losses.append(train_sequence(model, episode, start))

        if iteration == 1 or iteration % max(1, iterations // 4) == 0:
            print(
                f"iteration={iteration:02d} episodes={len(replay):3d} "
                f"loss={sum(losses) / len(losses):.4f}"
            )

    after = evaluate(model, 100, simulations, seed + 2)
    print(
        f"after training:  {after[0]} goal / {after[1]} fail, "
        f"mean_return={after[2]:.3f}"
    )


def run_tests() -> None:
    environment = LineWorld()
    observation, reward, terminal = environment.step(ACTION_RIGHT)
    assert (observation, reward, terminal) == (3, -0.01, False)
    observation, reward, terminal = environment.step(ACTION_RIGHT)
    assert (observation, reward, terminal) == (4, 1.0, True)

    model = TabularMuZeroModel()
    root = model.representation([2])
    predicted_reward, child = model.dynamics(root, ACTION_RIGHT)
    assert predicted_reward == 0.0
    assert child == HiddenState(2, (ACTION_RIGHT,))
    assert child.root_observation == 2  # It did not reconstruct real observation 3.

    model.rewards[(root, ACTION_LEFT)] = -1.0
    model.rewards[(root, ACTION_RIGHT)] = 1.0
    _, policy, _ = LatentMCTS(model, simulations=32).search([2])
    assert policy[ACTION_RIGHT] > policy[ACTION_LEFT], policy

    synthetic = Episode(
        observations=[2, 3, 4],
        actions=[ACTION_RIGHT, ACTION_RIGHT],
        rewards=[-0.01, 1.0],
        policies=[{ACTION_LEFT: 0.1, ACTION_RIGHT: 0.9}] * 2,
        root_values=[0.5, 0.8],
    )
    assert abs(n_step_target(synthetic, 0, 5, 0.997) - 0.987) < 1e-9

    fresh = TabularMuZeroModel()
    hidden = fresh.representation([2])
    before = fresh.prediction_loss(hidden, synthetic.policies[0], 0.987)
    for _ in range(30):
        train_sequence(fresh, synthetic, 0, learning_rate=0.15)
    after = fresh.prediction_loss(hidden, synthetic.policies[0], 0.987)
    assert after < before, (before, after)

    for value in (-100.0, -3.7, 0.0, 3.7, 100.0):
        recovered = decode_support(encode_support(value))
        assert abs(recovered - value) < 1e-6, (value, recovered)
    print("all tests passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.test:
        run_tests()
    else:
        train_demo(
            arguments.iterations,
            arguments.games,
            arguments.simulations,
            arguments.seed,
        )
