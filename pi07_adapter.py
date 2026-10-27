"""π0.7-compatible action-chunk boundary.

The public π0.7 paper describes a 5B VLA, but no public π0.7 checkpoint or
local inference package is assumed here.  This module keeps the input/output
contract stable so a permitted remote or future local backend can be plugged
in without bypassing the Harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PolicyObservation:
    task: str
    image_refs: tuple[str, ...]
    proprioception: tuple[float, ...]
    subgoal: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ActionChunk:
    actions: tuple[tuple[float, ...], ...]
    policy_name: str
    policy_version: str
    confidence: float
    source: str

    def validate(self, expected_dim: int, max_steps: int) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("action_chunk confidence must be in [0, 1]")
        if not self.actions:
            raise ValueError("action_chunk must contain at least one action")
        if len(self.actions) > max_steps:
            raise ValueError("action_chunk exceeds the configured horizon")
        if any(len(action) != expected_dim for action in self.actions):
            raise ValueError("action_chunk contains an action with the wrong dimension")


class Pi07Policy(Protocol):
    """Backend contract consumed by the local Harness."""

    def predict(self, observation: PolicyObservation) -> ActionChunk:
        ...


class UnavailablePi07Policy:
    """Explicit placeholder until an authorized π0.7 backend is available."""

    def predict(self, observation: PolicyObservation) -> ActionChunk:
        raise RuntimeError(
            "π0.7 backend is unavailable: configure an authorized Pi API or "
            "a locally licensed checkpoint before enabling this adapter"
        )


class DeterministicChunkPolicy:
    """Local test policy with the same action-chunk contract.

    This is useful for testing interruption, boundary checks and Verifier
    integration.  It must be reported as a local fallback, not as π0.7.
    """

    def __init__(self, action_dim: int = 6, horizon: int = 4) -> None:
        self.action_dim = action_dim
        self.horizon = horizon

    def predict(self, observation: PolicyObservation) -> ActionChunk:
        del observation
        zero = tuple(0.0 for _ in range(self.action_dim))
        chunk = ActionChunk(
            actions=tuple(zero for _ in range(self.horizon)),
            policy_name="deterministic_chunk_fallback",
            policy_version="0.1",
            confidence=0.0,
            source="local_fallback",
        )
        chunk.validate(self.action_dim, self.horizon)
        return chunk
