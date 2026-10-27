"""A small discrete JitRL-inspired selector for Harness skill candidates.

JitRL adjusts the logits of candidate actions at test time using advantages
estimated from retrieved state/action/return memories.  The implementation is
deliberately limited to discrete skill or action-chunk candidates: it does not
pretend to update π0.7's continuous action expert or model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


@dataclass(frozen=True)
class Transition:
    state: str
    action: str
    return_value: float


def _tokens(value: str) -> set[str]:
    return {token for token in value.lower().replace("_", " ").split() if token}


def jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


class JitRLMemory:
    def __init__(self) -> None:
        self.entries: list[Transition] = []

    def add_episode(
        self,
        states: Iterable[str],
        actions: Iterable[str],
        rewards: Iterable[float],
        gamma: float = 0.99,
    ) -> None:
        states, actions, rewards = list(states), list(actions), list(rewards)
        returns = [0.0] * len(rewards)
        running = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + gamma * running
            returns[index] = running
        self.entries.extend(
            Transition(state, action, value)
            for state, action, value in zip(states, actions, returns)
        )

    def retrieve(self, state: str, k: int = 8) -> list[Transition]:
        return sorted(
            self.entries,
            key=lambda item: jaccard(state, item.state),
            reverse=True,
        )[:k]


class JitRLSelector:
    def __init__(
        self,
        memory: JitRLMemory,
        beta: float = 1.0,
        exploration_probability: float = 0.2,
        exploration_bonus: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.memory = memory
        self.beta = beta
        self.exploration_probability = exploration_probability
        self.exploration_bonus = exploration_bonus
        self.random = random.Random(seed)

    def select(
        self,
        state: str,
        candidates: dict[str, float],
        k: int = 8,
    ) -> tuple[str, dict[str, float]]:
        neighbors = self.memory.retrieve(state, k=k)
        baseline = (
            sum(item.return_value for item in neighbors) / len(neighbors)
            if neighbors
            else 0.0
        )
        adjusted: dict[str, float] = {}
        for action, base_logit in candidates.items():
            matching = [item.return_value for item in neighbors if item.action == action]
            if matching:
                q_value = sum(matching) / len(matching)
            elif self.random.random() < self.exploration_probability:
                q_value = baseline + self.exploration_bonus / max(1, len(neighbors))
            else:
                q_value = 0.0
            adjusted[action] = base_logit + self.beta * (q_value - baseline)

        # Stable softmax sampling is useful in experiments, while argmax keeps
        # the default deterministic for a fixed candidate set.
        best = max(adjusted, key=adjusted.get)
        return best, adjusted
