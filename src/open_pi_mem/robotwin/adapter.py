from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image


class PredictsActions(Protocol):
    def predict(
        self,
        *,
        frames: list[Image.Image],
        proprio_state: list[float],
        goal: str,
        current_subtask: str | None = None,
        memory_context: str | None = None,
    ) -> dict[str, Any]:
        ...

    def reset(self) -> None:
        ...


@dataclass(frozen=True)
class EncodedRoboTwinObservation:
    frames: list[Image.Image]
    proprio_state: list[float]


def _to_rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            "RoboTwin camera rgb values must have shape (height, width, 3)"
        )
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _joint_vector(joint_action: dict[str, Any]) -> list[float]:
    if "vector" in joint_action:
        return np.asarray(joint_action["vector"], dtype=float).tolist()

    required = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    missing = [key for key in required if key not in joint_action]
    if missing:
        raise KeyError(f"RoboTwin joint_action is missing keys: {missing}")

    return (
        list(joint_action["left_arm"])
        + [joint_action["left_gripper"]]
        + list(joint_action["right_arm"])
        + [joint_action["right_gripper"]]
    )


def encode_robotwin_observation(observation: dict[str, Any]) -> EncodedRoboTwinObservation:
    """Convert a RoboTwin simulator observation into open-pi-mem inputs."""
    cameras = observation["observation"]
    frames = [
        _to_rgb_image(cameras["head_camera"]["rgb"]),
        _to_rgb_image(cameras["right_camera"]["rgb"]),
        _to_rgb_image(cameras["left_camera"]["rgb"]),
    ]
    return EncodedRoboTwinObservation(
        frames=frames,
        proprio_state=_joint_vector(observation["joint_action"]),
    )


def select_action_steps(
    action_chunk: Any,
    *,
    max_steps: int | None,
    action_clip: tuple[float, float] | None,
) -> np.ndarray:
    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2:
        raise ValueError(
            "open-pi-mem action_chunk must have shape (steps, action_dim)"
        )
    if max_steps is not None:
        actions = actions[:max_steps]
    if action_clip is not None:
        actions = np.clip(actions, action_clip[0], action_clip[1])
    return actions


class RoboTwinAdapter:
    """Adapter exposing open-pi-mem low-level predictions as RoboTwin actions."""

    def __init__(
        self,
        *,
        policy: PredictsActions | None = None,
        model_path: str | Path | None = None,
        device: str = "cuda",
        action_scale: float = 1.0,
        action_clip: tuple[float, float] | None = (-1.0, 1.0),
        max_action_steps: int | None = None,
        memory_enabled: bool = False,
    ) -> None:
        if policy is None:
            if model_path is None:
                raise ValueError("Provide either policy or model_path")
            from open_pi_mem.rmbench.adapter import RMBenchAdapter

            policy = RMBenchAdapter(
                model_path=model_path,
                device=device,
                action_scale=action_scale,
                action_clip=action_clip or (-float("inf"), float("inf")),
                memory_enabled=memory_enabled,
            )

        self.policy = policy
        self.action_clip = action_clip
        self.max_action_steps = max_action_steps
        self.memory_context = ""

    def reset(self) -> None:
        self.memory_context = ""
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def get_action(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        encoded = encode_robotwin_observation(observation)
        result = self.policy.predict(
            frames=encoded.frames,
            proprio_state=encoded.proprio_state,
            goal=instruction,
            current_subtask=instruction,
            memory_context=self.memory_context or None,
        )
        memory_update = result.get("memory_update")
        if memory_update:
            self.memory_context = str(memory_update)
        return select_action_steps(
            result["action_chunk"],
            max_steps=self.max_action_steps,
            action_clip=self.action_clip,
        )
