from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WorldState:
    objects: dict[str, dict[str, Any]]
    robot: dict[str, Any]
    observed: bool = False
    path_blocked: bool = False
    log: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "objects": deepcopy(self.objects),
            "robot": deepcopy(self.robot),
            "observed": self.observed,
            "path_blocked": self.path_blocked,
        }


class DesktopWorld:
    """Deterministic tabletop world used until ROS2/MoveIt is wired in."""

    def __init__(self, scene_path: str | Path, seed: int = 42) -> None:
        self.seed = seed
        raw = yaml.safe_load(Path(scene_path).read_text(encoding="utf-8"))
        self.state = WorldState(
            objects=deepcopy(raw["objects"]),
            robot=deepcopy(raw["robot"]),
        )

    def observe(self) -> dict[str, Any]:
        self.state.observed = True
        self.state.log.append("observe_scene")
        visible = {
            name: {
                "position": obj["position"],
                "in_bin": obj.get("in_bin", False),
            }
            for name, obj in self.state.objects.items()
        }
        return {
            "visible_objects": visible,
            "holding": self.state.robot.get("holding"),
            "ee_position": self.state.robot.get("ee_position"),
            "path_blocked": self.state.path_blocked,
        }

    def pick(self, object_id: str) -> dict[str, Any]:
        if object_id not in self.state.objects:
            return {"ok": False, "failure_code": "object_not_found"}
        if self.state.path_blocked:
            return {"ok": False, "failure_code": "path_blocked"}
        if self.state.robot.get("holding"):
            return {"ok": False, "failure_code": "grasp_failed"}
        obj = self.state.objects[object_id]
        if obj.get("in_bin"):
            return {"ok": False, "failure_code": "object_not_found"}
        self.state.robot["holding"] = object_id
        self.state.robot["ee_position"] = list(obj["position"])
        self.state.log.append(f"pick:{object_id}")
        return {"ok": True, "holding": object_id}

    def place(self, object_id: str, bin_id: str) -> dict[str, Any]:
        if self.state.robot.get("holding") != object_id:
            return {"ok": False, "failure_code": "object_lost"}
        if bin_id not in self.state.objects:
            return {"ok": False, "failure_code": "object_not_found"}
        if self.state.path_blocked:
            return {"ok": False, "failure_code": "path_blocked"}
        bin_pos = self.state.objects[bin_id]["position"]
        self.state.objects[object_id]["position"] = list(bin_pos)
        self.state.objects[object_id]["in_bin"] = True
        self.state.robot["holding"] = None
        self.state.robot["ee_position"] = list(bin_pos)
        self.state.log.append(f"place:{object_id}->{bin_id}")
        return {"ok": True, "object_id": object_id, "bin_id": bin_id}

    def evidence(self) -> dict[str, Any]:
        return self.state.snapshot()
