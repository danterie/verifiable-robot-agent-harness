from __future__ import annotations

from typing import Any

from robot_agent_harness.agent_client import LLMClient
from robot_agent_harness.runtime import HarnessRuntime
from robot_agent_harness.toolkit import Toolkit


class PlannerLoop:
    """Small RPent-style planner/tool/runtime orchestration loop."""

    def __init__(
        self,
        client: LLMClient,
        toolkit: Toolkit,
        runtime: HarnessRuntime,
        max_steps: int = 8,
    ) -> None:
        self.client = client
        self.toolkit = toolkit
        self.runtime = runtime
        self.max_steps = max_steps

    def run(self, task: str) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        observation = self.runtime.world.evidence()
        source = "unknown"

        for step in range(self.max_steps):
            decision = self.client.plan_next(
                task,
                self.toolkit.openai_tools(),
                observation,
                history,
            )
            source = decision["source"]
            call = decision.get("call")
            if not call:
                break

            result = self.runtime.execute_skill(
                call["name"],
                call.get("arguments", {}),
            )
            history.append(
                {
                    "step": step,
                    "call": call,
                    "result": result,
                }
            )
            observation = self.runtime.world.evidence()

            if result.get("status") == "committed":
                # A placement commit is the terminal success condition for
                # this execution. Other skills remain eligible for later rounds.
                if call["name"] == "place_object":
                    break
            if result.get("status") == "failed":
                break

        return {
            "source": source,
            "steps": history,
            "final_evidence": self.runtime.world.evidence(),
            "trace": self.runtime.trace,
        }
