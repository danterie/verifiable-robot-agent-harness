"""Minimal Transformer policy for the Lesson 25 skill interface.

The model predicts a discrete skill name. It deliberately does not emit joint
angles or motor commands; the decoded call still has to pass through Toolkit
and Harness validation before anything can execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SkillCall:
    name: str
    arguments: dict[str, Any]


class MiniTransformerPolicy(nn.Module):
    """Fuse scene, task and robot-state tokens and classify the next skill."""

    def __init__(
        self,
        feature_dim: int = 8,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        num_skills: int = 5,
        max_tokens: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.scene_projection = nn.Linear(feature_dim, d_model)
        self.task_projection = nn.Linear(feature_dim, d_model)
        self.robot_projection = nn.Linear(feature_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.skill_head = nn.Linear(d_model, num_skills)

    @staticmethod
    def causal_mask(length: int, device: torch.device | None = None) -> Tensor:
        """Return a boolean mask where True means attention is disallowed."""
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self,
        scene_tokens: Tensor,
        task_tokens: Tensor,
        robot_tokens: Tensor,
        *,
        causal: bool = True,
    ) -> Tensor:
        if scene_tokens.ndim != 3 or task_tokens.ndim != 3 or robot_tokens.ndim != 3:
            raise ValueError("all token groups must have shape [batch, tokens, feature_dim]")
        tokens = torch.cat(
            [
                self.scene_projection(scene_tokens),
                self.task_projection(task_tokens),
                self.robot_projection(robot_tokens),
            ],
            dim=1,
        )
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("input sequence is longer than max_tokens")
        tokens = tokens + self.position[:, : tokens.shape[1]]
        mask = self.causal_mask(tokens.shape[1], tokens.device) if causal else None
        encoded = self.encoder(tokens, mask=mask)
        # The final token is the robot-state query token. With a causal mask it
        # can read earlier context, but later tokens cannot leak backward.
        return self.skill_head(encoded[:, -1, :])

    @staticmethod
    def decode_tool_call(
        skill_index: int,
        skill_names: Sequence[str],
        *,
        object_id: str = "red_box_1",
        bin_id: str = "target_bin_1",
    ) -> SkillCall:
        name = skill_names[skill_index]
        arguments: dict[str, Any] = {}
        if name in {"pick_object", "verify_grasp"}:
            arguments = {"object_id": object_id}
        elif name in {"place_object", "verify_placement"}:
            arguments = {"object_id": object_id, "bin_id": bin_id}
        elif name == "recover":
            arguments = {"failure_code": "unknown"}
        return SkillCall(name=name, arguments=arguments)
