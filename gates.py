from __future__ import annotations

from typing import Any


class Verifier:
    def verify_grasp(self, evidence: dict[str, Any], object_id: str) -> dict[str, Any]:
        holding = evidence["robot"].get("holding")
        ok = holding == object_id
        return {
            "name": "verify_grasp",
            "ok": ok,
            "status": "verified" if ok else "failed",
            "failure_code": None if ok else "grasp_failed",
            "evidence": {"holding": holding, "expected": object_id},
        }

    def verify_placement(
        self, evidence: dict[str, Any], object_id: str, bin_id: str
    ) -> dict[str, Any]:
        obj = evidence["objects"].get(object_id, {})
        bin_obj = evidence["objects"].get(bin_id, {})
        holding = evidence["robot"].get("holding")
        ok = bool(obj.get("in_bin")) and holding is None and bool(bin_obj)
        return {
            "name": "verify_placement",
            "ok": ok,
            "status": "verified" if ok else "failed",
            "failure_code": None if ok else "placement_failed",
            "evidence": {
                "in_bin": obj.get("in_bin"),
                "holding": holding,
                "object_position": obj.get("position"),
                "bin_position": bin_obj.get("position"),
            },
        }
