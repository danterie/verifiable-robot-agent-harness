"""Run the attention, mask, and structured skill experiments."""

from __future__ import annotations

import json

import torch

from robot_agent_harness.controllers import MiniTransformerPolicy
from robot_agent_harness.toolkit import load_toolkit


def scaled_dot_product_demo() -> dict[str, list[int]]:
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 5, 8)
    value = torch.randn(2, 5, 8)
    weights = torch.softmax(query @ key.transpose(-2, -1) / (8**0.5), dim=-1)
    output = weights @ value
    return {
        "query": list(query.shape),
        "key": list(key.shape),
        "value": list(value.shape),
        "attention_weights": list(weights.shape),
        "attention_output": list(output.shape),
    }


def main() -> None:
    torch.manual_seed(25)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = MiniTransformerPolicy().to(device).eval()

    scene_tokens = torch.tensor(
        [[[0.35, 0.10, 0.05, 1, 0, 0, 0, 0], [0.55, -0.10, 0.02, 0, 1, 0, 0, 0]]],
        dtype=torch.float32,
        device=device,
    )
    task_tokens = torch.tensor(
        [[[1, 0, 0, 0, 0, 1, 0, 0]]], dtype=torch.float32, device=device
    )
    robot_tokens = torch.tensor(
        [[[0.20, 0.00, 0.25, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]]],
        dtype=torch.float32,
        device=device,
    )
    skill_names = [
        "observe_scene",
        "pick_object",
        "place_object",
        "verify_placement",
        "recover",
    ]

    with torch.inference_mode():
        causal_logits = policy(scene_tokens, task_tokens, robot_tokens, causal=True)
        full_logits = policy(scene_tokens, task_tokens, robot_tokens, causal=False)

    skill_index = int(causal_logits.argmax(dim=-1).item())
    call = policy.decode_tool_call(skill_index, skill_names)
    toolkit = load_toolkit("configs/skills.yaml")
    toolkit.get(call.name).validate_args(call.arguments)

    mask = policy.causal_mask(5, device=device)
    payload = {
        "device": str(device),
        "qkv_attention": scaled_dot_product_demo(),
        "sequence_layout": ["scene_0", "scene_1", "task_0", "robot_0", "robot_1"],
        "causal_mask_true_count": int(mask.sum().item()),
        "causal_logits_shape": list(causal_logits.shape),
        "full_logits_shape": list(full_logits.shape),
        "mask_max_abs_logit_difference": float((causal_logits - full_logits).abs().max().item()),
        "predicted_skill": skill_names[skill_index],
        "structured_tool_call": {"name": call.name, "arguments": call.arguments},
        "raw_joint_command_emitted": False,
        "toolkit_schema_validation": "passed",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
