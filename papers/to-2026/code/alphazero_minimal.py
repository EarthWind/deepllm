#!/usr/bin/env python3
"""A zero-dependency, executable miniature of the AlphaZero control loop.

The real AlphaZero uses a deep residual network and thousands of TPUs.  This
teaching implementation keeps the algorithmic interfaces but swaps the neural
network for a tiny tabular policy/value model and chess for tic-tac-toe:

    self-play -> PUCT search targets -> policy/value update -> stronger search

Run:
    python3 alphazero_minimal.py --test
    python3 alphazero_minimal.py --iterations 12 --games 12 --simulations 40
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass(frozen=True)
class State:
    """Board in the current player's perspective: own=+1, opponent=-1."""

    board: Tuple[int, ...] = (0,) * 9

    def legal_actions(self) -> Tuple[int, ...]:
        return tuple(i for i, piece in enumerate(self.board) if piece == 0)

    def play(self, action: int) -> "State":
        if action not in self.legal_actions():
            raise ValueError(f"illegal action: {action}")
        next_board = list(self.board)
        next_board[action] = 1
        # Switch player and canonicalize the board to the new player's view.
        return State(tuple(-piece for piece in next_board))

    def terminal_value(self) -> Optional[float]:
        """Return the outcome for the player to move, or None if non-terminal."""
        for a, b, c in WIN_LINES:
            line = (self.board[a], self.board[b], self.board[c])
            if line == (1, 1, 1):
                return 1.0
            if line == (-1, -1, -1):
                return -1.0
        if all(self.board):
            return 0.0
        return None


Policy = Dict[int, float]
Example = Tuple[State, Policy, float]


class TabularPolicyValue:
    """A small stand-in for f_theta(s) -> (policy, value)."""

    def __init__(self) -> None:
        self.policy_logits: Dict[Tuple[int, ...], List[float]] = {}
        self.value_logits: Dict[Tuple[int, ...], float] = {}

    def _logits(self, state: State) -> List[float]:
        return self.policy_logits.setdefault(state.board, [0.0] * 9)

    def predict(self, state: State) -> Tuple[Policy, float]:
        legal = state.legal_actions()
        logits = self._logits(state)
        maximum = max(logits[action] for action in legal)
        weights = {action: math.exp(logits[action] - maximum) for action in legal}
        total = sum(weights.values())
        policy = {action: weight / total for action, weight in weights.items()}
        value = math.tanh(self.value_logits.get(state.board, 0.0))
        return policy, value

    def loss(self, examples: Sequence[Example], l2: float = 1e-4) -> float:
        if not examples:
            return 0.0
        total = 0.0
        for state, target_policy, target_value in examples:
            policy, value = self.predict(state)
            cross_entropy = -sum(
                probability * math.log(max(policy.get(action, 1e-12), 1e-12))
                for action, probability in target_policy.items()
            )
            logits = self._logits(state)
            regularizer = l2 * (
                sum(logit * logit for logit in logits)
                + self.value_logits.get(state.board, 0.0) ** 2
            )
            total += (target_value - value) ** 2 + cross_entropy + regularizer
        return total / len(examples)

    def train_batch(self, examples: Sequence[Example], learning_rate: float = 0.15) -> float:
        """One SGD pass on (z-v)^2 - pi^T log(p), plus a small L2 term."""
        loss_before = self.loss(examples)
        for state, target_policy, target_value in examples:
            policy, value = self.predict(state)
            logits = self._logits(state)
            for action in state.legal_actions():
                gradient = policy[action] - target_policy.get(action, 0.0)
                logits[action] -= learning_rate * (gradient + 2e-4 * logits[action])

            raw_value = self.value_logits.get(state.board, 0.0)
            value_gradient = 2.0 * (value - target_value) * (1.0 - value * value)
            self.value_logits[state.board] = raw_value - learning_rate * (
                value_gradient + 2e-4 * raw_value
            )
        return loss_before


@dataclass
class Node:
    state: State
    prior: Policy = field(default_factory=dict)
    visits: Dict[int, int] = field(default_factory=dict)
    value_sum: Dict[int, float] = field(default_factory=dict)
    children: Dict[int, "Node"] = field(default_factory=dict)
    expanded: bool = False

    @property
    def total_visits(self) -> int:
        return sum(self.visits.values())

    def expand(self, policy: Policy) -> None:
        self.prior = dict(policy)
        self.visits = {action: 0 for action in policy}
        self.value_sum = {action: 0.0 for action in policy}
        self.expanded = True


class MCTS:
    def __init__(
        self,
        model: TabularPolicyValue,
        simulations: int = 40,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        noise_fraction: float = 0.25,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.rng = rng or random.Random()

    def _add_root_noise(self, node: Node) -> None:
        actions = list(node.prior)
        noise = [self.rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in actions]
        scale = sum(noise)
        for action, sample in zip(actions, noise):
            mixed = (1.0 - self.noise_fraction) * node.prior[action]
            mixed += self.noise_fraction * sample / scale
            node.prior[action] = mixed

    def _select(self, node: Node) -> int:
        parent_scale = math.sqrt(max(1, node.total_visits))

        def score(action: int) -> float:
            count = node.visits[action]
            q_value = node.value_sum[action] / count if count else 0.0
            exploration = (
                self.c_puct * node.prior[action] * parent_scale / (1 + count)
            )
            return q_value + exploration

        return max(node.prior, key=lambda action: (score(action), -action))

    def search(self, state: State, add_root_noise: bool) -> Node:
        root = Node(state)
        policy, _ = self.model.predict(state)
        root.expand(policy)
        if add_root_noise:
            self._add_root_noise(root)

        for _ in range(self.simulations):
            node = root
            path: List[Tuple[Node, int]] = []

            while True:
                outcome = node.state.terminal_value()
                if outcome is not None:
                    value = outcome
                    break

                if not node.expanded:
                    leaf_policy, value = self.model.predict(node.state)
                    node.expand(leaf_policy)
                    break

                action = self._select(node)
                path.append((node, action))
                if action not in node.children:
                    node.children[action] = Node(node.state.play(action))
                node = node.children[action]

            # `value` is from the leaf player's view. Each edge changes player.
            for parent, action in reversed(path):
                value = -value
                parent.visits[action] += 1
                parent.value_sum[action] += value

        return root

    @staticmethod
    def visit_policy(root: Node, temperature: float) -> Policy:
        actions = list(root.visits)
        if temperature <= 1e-8:
            best = max(actions, key=lambda action: (root.visits[action], -action))
            return {action: float(action == best) for action in actions}
        exponent = 1.0 / temperature
        weights = {action: root.visits[action] ** exponent for action in actions}
        if sum(weights.values()) == 0:
            return {action: 1.0 / len(actions) for action in actions}
        total = sum(weights.values())
        return {action: weight / total for action, weight in weights.items()}


def sample_action(policy: Policy, rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for action, probability in sorted(policy.items()):
        cumulative += probability
        if threshold <= cumulative:
            return action
    return next(reversed(policy))


def self_play_game(
    model: TabularPolicyValue,
    simulations: int,
    rng: random.Random,
) -> List[Example]:
    state = State()
    history: List[Tuple[State, Policy]] = []

    while state.terminal_value() is None:
        search = MCTS(model, simulations=simulations, rng=rng)
        root = search.search(state, add_root_noise=True)
        # The paper uses tau=1 for the opening, then near-greedy play.
        temperature = 1.0 if len(history) < 3 else 0.1
        policy = search.visit_policy(root, temperature)
        history.append((state, policy))
        state = state.play(sample_action(policy, rng))

    # Convert the terminal result to every historical player's perspective.
    outcome = state.terminal_value()
    assert outcome is not None
    reversed_examples: List[Example] = []
    for old_state, policy in reversed(history):
        outcome = -outcome
        reversed_examples.append((old_state, policy, outcome))
    return list(reversed(reversed_examples))


def evaluate_vs_random(
    model: TabularPolicyValue,
    games: int,
    simulations: int,
    rng: random.Random,
) -> Tuple[int, int, int]:
    wins = draws = losses = 0
    for game in range(games):
        state = State()
        model_to_move = game % 2 == 0
        while state.terminal_value() is None:
            if model_to_move:
                root = MCTS(model, simulations=simulations, rng=rng).search(
                    state, add_root_noise=False
                )
                policy = MCTS.visit_policy(root, temperature=0.0)
                action = max(policy, key=policy.get)
            else:
                action = rng.choice(state.legal_actions())
            state = state.play(action)
            model_to_move = not model_to_move

        current_outcome = state.terminal_value()
        assert current_outcome is not None
        model_outcome = current_outcome if model_to_move else -current_outcome
        if model_outcome > 0:
            wins += 1
        elif model_outcome < 0:
            losses += 1
        else:
            draws += 1
    return wins, draws, losses


def train_demo(iterations: int, games: int, simulations: int, seed: int) -> None:
    rng = random.Random(seed)
    model = TabularPolicyValue()
    replay: List[Example] = []

    for iteration in range(1, iterations + 1):
        fresh = []
        for _ in range(games):
            fresh.extend(self_play_game(model, simulations, rng))
        replay.extend(fresh)
        replay = replay[-5000:]
        batch = rng.sample(replay, min(256, len(replay)))
        loss = model.train_batch(batch)
        if iteration == 1 or iteration % max(1, iterations // 4) == 0:
            print(
                f"iteration={iteration:02d} positions={len(replay):4d} "
                f"loss={loss:.4f}"
            )

    wins, draws, losses = evaluate_vs_random(model, 100, simulations, rng)
    print(f"vs random (100 games): {wins} wins / {draws} draws / {losses} losses")


def run_tests() -> None:
    state = State((1, 1, 0, -1, -1, 0, 0, 0, 0))
    assert state.play(2).terminal_value() == -1.0
    assert len(state.legal_actions()) == 5

    model = TabularPolicyValue()
    policy, value = model.predict(State())
    assert abs(sum(policy.values()) - 1.0) < 1e-12 and value == 0.0

    root = MCTS(model, simulations=24, rng=random.Random(7)).search(
        State(), add_root_noise=False
    )
    assert root.total_visits == 24
    visit_policy = MCTS.visit_policy(root, temperature=1.0)
    assert abs(sum(visit_policy.values()) - 1.0) < 1e-12

    target = [(State(), {0: 1.0}, 1.0)]
    before = model.loss(target)
    for _ in range(30):
        model.train_batch(target, learning_rate=0.2)
    after = model.loss(target)
    assert after < before, (before, after)
    print("all tests passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run deterministic checks")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--games", type=int, default=12, help="self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=40, help="MCTS simulations per move")
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
