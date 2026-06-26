from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class MEMPlan:
    subtask: str
    memory: str
    raw_output: str = ""
    planner_name: str = "unknown"


class RoboTwinMEMPlanner(Protocol):
    def plan(
        self,
        *,
        goal: str,
        observation: dict[str, Any],
        prev_memory: str,
        step_index: int,
    ) -> MEMPlan:
        ...


class PassthroughMEMPlanner:
    """MEM-compatible planner that preserves the task instruction.

    This is the safe baseline for pi06: it exercises the open-pi-mem MEM control
    path while leaving pi05 behavior equivalent to the original task instruction.
    A trained or API-backed planner can replace it behind the same interface.
    """

    planner_name = "passthrough"

    def plan(
        self,
        *,
        goal: str,
        observation: dict[str, Any],
        prev_memory: str,
        step_index: int,
    ) -> MEMPlan:
        return MEMPlan(
            subtask=goal,
            memory=prev_memory,
            raw_output=goal,
            planner_name=self.planner_name,
        )


def encode_pi05_observation(observation: dict[str, Any]) -> tuple[list[np.ndarray], np.ndarray]:
    cameras = observation["observation"]
    images = [
        np.asarray(cameras["head_camera"]["rgb"]),
        np.asarray(cameras["right_camera"]["rgb"]),
        np.asarray(cameras["left_camera"]["rgb"]),
    ]
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    return images, state


class Pi05MEMController:
    """Run a pi05-style policy with open-pi-mem planning around its language input."""

    def __init__(
        self,
        pi05_policy: Any,
        *,
        planner: RoboTwinMEMPlanner | None = None,
        plan_interval_steps: int = 1,
        action_steps: int | None = None,
    ) -> None:
        self.pi05_policy = pi05_policy
        self.planner = planner or PassthroughMEMPlanner()
        self.plan_interval_steps = max(int(plan_interval_steps), 1)
        self.action_steps = action_steps
        self.memory = ""
        self.current_subtask: str | None = None
        self.step_index = 0

    def reset(self) -> None:
        self.memory = ""
        self.current_subtask = None
        self.step_index = 0
        reset = getattr(self.pi05_policy, "reset_obsrvationwindows", None)
        if callable(reset):
            reset()

    def reset_model(self) -> None:
        """Server-compatible reset method for RoboTwin policy_model_server."""
        self.reset()

    def act_request(self, request: dict[str, Any]) -> np.ndarray:
        """Server-compatible action method accepting one JSON payload."""
        return self.act(str(request["goal"]), request["observation"])

    def act(self, goal: str, observation: dict[str, Any]) -> np.ndarray:
        if self._needs_plan():
            plan = self.planner.plan(
                goal=goal,
                observation=observation,
                prev_memory=self.memory,
                step_index=self.step_index,
            )
            self.current_subtask = plan.subtask or goal
            self.memory = plan.memory
            self.pi05_policy.set_language(self.current_subtask)

        images, state = encode_pi05_observation(observation)
        self.pi05_policy.update_observation_window(images, state)
        actions = np.asarray(self.pi05_policy.get_action(), dtype=np.float32)
        if self.action_steps is not None:
            actions = actions[: self.action_steps]
        self.step_index += 1
        return actions

    def _needs_plan(self) -> bool:
        return (
            self.current_subtask is None
            or self.step_index % self.plan_interval_steps == 0
        )
