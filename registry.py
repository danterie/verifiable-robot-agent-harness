from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    parameters: dict[str, str]
    preconditions: list[str]
    effects: list[str]
    timeout_s: float
    risk: str
    failure_codes: list[str]
    verifier: str | None

    def validate_args(self, args: dict[str, Any]) -> None:
        missing = [key for key in self.parameters if key not in args]
        extra = [key for key in args if key not in self.parameters]
        if missing or extra:
            raise ValueError(
                f"schema_error for {self.name}: missing={missing} extra={extra}"
            )
        for key, typ in self.parameters.items():
            if typ == "string" and not isinstance(args[key], str):
                raise ValueError(f"{self.name}.{key} must be a string")


class Toolkit:
    def __init__(self, specs: dict[str, SkillSpec]) -> None:
        self.specs = specs

    def get(self, name: str) -> SkillSpec:
        if name not in self.specs:
            raise KeyError(f"unknown_skill: {name}")
        return self.specs[name]

    def openai_tools(self) -> list[dict[str, Any]]:
        tools = []
        for spec in self.specs.values():
            properties = {
                name: {"type": "string"} for name in spec.parameters
            }
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": list(spec.parameters),
                        },
                    },
                }
            )
        return tools


def load_toolkit(path: str | Path) -> Toolkit:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    specs = {}
    for item in raw["skills"]:
        spec = SkillSpec(
            name=item["name"],
            description=item["description"],
            parameters=item.get("parameters") or {},
            preconditions=item.get("preconditions") or [],
            effects=item.get("effects") or [],
            timeout_s=float(item["timeout_s"]),
            risk=item["risk"],
            failure_codes=item.get("failure_codes") or [],
            verifier=item.get("verifier"),
        )
        specs[spec.name] = spec
    return Toolkit(specs)
