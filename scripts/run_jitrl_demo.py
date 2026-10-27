"""Demonstrate test-time JitRL-style skill selection without weight updates."""

from __future__ import annotations

from robot_agent_harness.memory.jitrl import JitRLMemory, JitRLSelector


def main() -> None:
    memory = JitRLMemory()
    memory.add_episode(
        states=["visible red box path clear", "holding red box path clear"],
        actions=["pick_object", "place_object"],
        rewards=[0.0, 1.0],
    )
    selector = JitRLSelector(memory, beta=1.0, seed=42)
    action, scores = selector.select(
        "visible red box path clear",
        {"pick_object": 0.0, "recover": -0.2},
    )
    print({"selected_action": action, "adjusted_logits": scores})


if __name__ == "__main__":
    main()
