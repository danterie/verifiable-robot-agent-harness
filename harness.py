from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable

from robot_agent_harness.simulator.desktop_world import DesktopWorld
from robot_agent_harness.toolkit.registry import Toolkit
from robot_agent_harness.verifier.gates import Verifier

SkillHandler = Callable[[dict[str, Any]], dict[str, Any]]


class HarnessRuntime:
    def __init__(self, toolkit: Toolkit, world: DesktopWorld, verifier: Verifier) -> None:
        self.toolkit = toolkit
        self.world = world
        self.verifier = verifier
        self.episode_id = str(uuid.uuid4())
        self.trace: list[dict[str, Any]] = []

    def preconditions_ok(self, names: list[str], args: dict[str, Any]) -> tuple[bool, str | None]:
        state = self.world.state
        checks = {
            "robot_ready": bool(state.robot.get("ready")),
            "object_visible": state.observed and args.get("object_id") in state.objects,
            "path_is_clear": not state.path_blocked,
            "object_held": state.robot.get("holding") == args.get("object_id"),
        }
        for name in names:
            if name not in checks:
                return False, f"unknown_precondition:{name}"
            if not checks[name]:
                return False, name
        return True, None

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "observe_scene":
            observation = self.world.observe()
            return {"ok": True, "observation": observation}
        if name == "pick_object":
            result = self.world.pick(args["object_id"])
            if result.get("ok"):
                result["verifier"] = self.verifier.verify_grasp(
                    self.world.evidence(), args["object_id"]
                )
                result["ok"] = result["verifier"]["ok"]
            return result
        if name == "place_object":
            result = self.world.place(args["object_id"], args["bin_id"])
            if result.get("ok"):
                result["verifier"] = self.verifier.verify_placement(
                    self.world.evidence(), args["object_id"], args["bin_id"]
                )
                result["ok"] = result["verifier"]["ok"]
            return result
        if name == "verify_grasp":
            return self.verifier.verify_grasp(self.world.evidence(), args["object_id"])
        if name == "verify_placement":
            return self.verifier.verify_placement(
                self.world.evidence(), args["object_id"], args["bin_id"]
            )
        if name == "recover":
            self.world.state.observed = False
            return {"ok": True, "action": "reobserve_required", "failure_code": args.get("failure_code")}
        raise KeyError(name)

    def execute_skill(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        record: dict[str, Any] = {
            "episode_id": self.episode_id,
            "skill": name,
            "args": args,
            "status": "proposed",
        }
        started = time.time()
        try:
            spec = self.toolkit.get(name)
            spec.validate_args(args)
            record["status"] = "validated"
            ok, failed_pre = self.preconditions_ok(spec.preconditions, args)
            if not ok:
                record.update(
                    {
                        "status": "failed",
                        "failure_code": "precondition_failed",
                        "detail": failed_pre,
                    }
                )
                return record
            record["status"] = "running"
            result = self.dispatch(name, args)
            elapsed = time.time() - started
            if elapsed > spec.timeout_s:
                record.update({"status": "failed", "failure_code": "timeout", "result": result})
                return record
            if result.get("ok") is False:
                record.update(
                    {
                        "status": "failed",
                        "failure_code": result.get("failure_code", "unknown"),
                        "result": result,
                    }
                )
                return record
            verifier = result.get("verifier")
            if verifier:
                record["verifier"] = verifier
                record["status"] = "verified" if verifier.get("ok") else "failed"
                if record["status"] == "verified":
                    record["status"] = "committed"
                else:
                    record["failure_code"] = verifier.get("failure_code")
            else:
                record["status"] = "succeeded"
            record["result"] = result
            return record
        finally:
            record["elapsed_s"] = round(time.time() - started, 4)
            self.trace.append(record)

    def dump_trace(self) -> str:
        return json.dumps(self.trace, ensure_ascii=False, indent=2)
