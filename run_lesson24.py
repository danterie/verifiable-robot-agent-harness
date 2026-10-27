"""Run the offline pick-and-place harness loop."""

from __future__ import annotations

import os
from pathlib import Path

from robot_agent_harness.agent_client import LLMClient
from robot_agent_harness.planner import PlannerLoop
from robot_agent_harness.runtime import HarnessRuntime
from robot_agent_harness.simulator import DesktopWorld
from robot_agent_harness.toolkit import load_toolkit
from robot_agent_harness.verifier import Verifier


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scene_path = root / "configs" / "scene.yaml"
    toolkit_path = root / "configs" / "skills.yaml"
    world = DesktopWorld(scene_path, seed=int(os.getenv("SIM_SEED", "42")))
    toolkit = load_toolkit(toolkit_path)
    runtime = HarnessRuntime(toolkit, world, Verifier())

    task = "把 red_box_1 放入 target_bin_1"
    result = PlannerLoop(
        client=LLMClient(),
        toolkit=toolkit,
        runtime=runtime,
        max_steps=int(os.getenv("MAX_PLANNER_STEPS", "8")),
    ).run(task)

    import json

    print(json.dumps({
        "plan_source": result["source"],
        "final_evidence": result["final_evidence"],
        "trace": result["trace"],
        "steps": result["steps"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
