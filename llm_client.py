from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv


class LLMClient:
    def __init__(self) -> None:
        load_dotenv()
        provider = os.getenv("AGENT_MODEL_PROVIDER", "deepseek").lower()
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        else:
            api_key = os.getenv("CODEX_API_KEY") or os.getenv("OPENAI_API_KEY", "")
            base_url = os.getenv("OPENAI_BASE_URL")
            self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.enabled = bool(api_key)
        self.client = OpenAI(api_key=api_key or "missing", base_url=base_url) if self.enabled else None

    def plan(self, task: str, tools: list[dict[str, Any]], observation: dict[str, Any]) -> dict[str, Any]:
        """Legacy batch planner kept for comparison experiments.

        Lesson 24 uses :meth:`plan_next` so that each tool result is returned to
        the planner before the next action is selected.
        """
        if not self.enabled or self.client is None:
            return {
                "source": "offline_fallback",
                "calls": [
                    {"name": "observe_scene", "arguments": {}},
                    {"name": "pick_object", "arguments": {"object_id": "red_box_1"}},
                    {"name": "place_object", "arguments": {"object_id": "red_box_1", "bin_id": "target_bin_1"}},
                    {
                        "name": "verify_placement",
                        "arguments": {"object_id": "red_box_1", "bin_id": "target_bin_1"},
                    },
                ],
            }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a robot task planner. Call skills only. "
                        "Never emit raw joint commands. Complete the desktop pick-and-place task."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"task": task, "observation": observation}, ensure_ascii=False),
                },
            ],
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        calls = []
        for tool_call in message.tool_calls or []:
            calls.append(
                {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments or "{}"),
                }
            )
        return {"source": "cloud", "calls": calls, "content": message.content}

    def plan_next(
        self,
        task: str,
        tools: list[dict[str, Any]],
        observation: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Select exactly one next skill from the current observation and trace.

        This is the RPent-style planner/runtime boundary used by the harness:
        the planner proposes one tool call, the runtime executes it, and the
        resulting evidence is fed into the next planning round.
        """
        if not self.enabled or self.client is None:
            calls = [
                {"name": "observe_scene", "arguments": {}},
                {"name": "pick_object", "arguments": {"object_id": "red_box_1"}},
                {
                    "name": "place_object",
                    "arguments": {"object_id": "red_box_1", "bin_id": "target_bin_1"},
                },
                {
                    "name": "verify_placement",
                    "arguments": {"object_id": "red_box_1", "bin_id": "target_bin_1"},
                },
            ]
            index = len(history)
            if index < len(calls):
                return {"source": "offline_fallback", "call": calls[index]}
            return {"source": "offline_fallback", "call": None}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a robot task planner. Select exactly one skill call. "
                        "Use only the provided tools. Never emit raw joint commands. "
                        "Stop by returning no tool call when the task is verified."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": task,
                            "observation": observation,
                            "previous_steps": history,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return {"source": "cloud", "call": None, "content": message.content}
        tool_call = tool_calls[0]
        return {
            "source": "cloud",
            "call": {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments or "{}"),
            },
            "content": message.content,
        }
