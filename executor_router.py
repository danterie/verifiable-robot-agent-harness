"""Parallel executor interfaces for π0.7-style VLA and JitRL-style agents.

The two policies share the Harness boundary but have different roles:
π0.7-style policies emit continuous action chunks, while JitRL modifies the
selection of discrete candidate actions at test time using retrieved returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from robot_agent_harness.memory.jitrl import JitRLSelector
from robot_agent_harness.policies.pi07_adapter import ActionChunk, Pi07Policy, PolicyObservation


class SkillExecutor(Protocol):
    def propose(self, observation: PolicyObservation) -> dict[str, Any]:
        ...


@dataclass
class Pi07Executor:
    policy: Pi07Policy

    def propose(self, observation: PolicyObservation) -> dict[str, Any]:
        chunk = self.policy.predict(observation)
        return {"executor": "pi07_style", "action_chunk": chunk}


@dataclass
class JitRLExecutor:
    """JitRL-style discrete executor, independent of π0.7.

    A base LLM or rule policy supplies candidate skill/chunk IDs and logits;
    JitRL retrieves returns and selects one candidate. The selected candidate
    is then sent to the Harness, which owns continuous execution.
    """

    selector: JitRLSelector

    def propose(
        self,
        observation: PolicyObservation,
        candidates: dict[str, float],
    ) -> dict[str, Any]:
        state = " ".join(
            [
                observation.task,
                observation.subgoal or "",
                "robot_state",
            ]
        )
        action, scores = self.selector.select(state, candidates)
        return {
            "executor": "jitrl_style",
            "selected_action": action,
            "adjusted_logits": scores,
        }


@dataclass
class PolicyRouter:
    """Explicit router for parallel or hybrid experiments."""

    pi07: Pi07Executor | None = None
    jitrl: JitRLExecutor | None = None

    def run_pi07(self, observation: PolicyObservation) -> dict[str, Any]:
        if self.pi07 is None:
            raise RuntimeError("π0.7-style executor is not configured")
        return self.pi07.propose(observation)

    def run_jitrl(
        self,
        observation: PolicyObservation,
        candidates: dict[str, float],
    ) -> dict[str, Any]:
        if self.jitrl is None:
            raise RuntimeError("JitRL-style executor is not configured")
        return self.jitrl.propose(observation, candidates)

