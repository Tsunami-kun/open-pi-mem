from __future__ import annotations

import numpy as np

from open_pi_mem.robotwin.mem_controller import (
    MEMPlan,
    PassthroughMEMPlanner,
    Pi05MEMController,
)


def _observation(value: int = 0) -> dict:
    rgb = np.zeros((3, 4, 3), dtype=np.uint8) + value
    return {
        "observation": {
            "head_camera": {"rgb": rgb},
            "right_camera": {"rgb": rgb + 1},
            "left_camera": {"rgb": rgb + 2},
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
    }


class _FakePi05Policy:
    def __init__(self) -> None:
        self.languages: list[str] = []
        self.updates: list[tuple[list[np.ndarray], np.ndarray]] = []
        self.reset_count = 0

    def set_language(self, instruction: str) -> None:
        self.languages.append(instruction)

    def update_observation_window(self, images, state) -> None:
        self.updates.append((images, state))

    def get_action(self):
        return np.array([[0.1] * 14, [0.2] * 14], dtype=np.float32)

    def reset_obsrvationwindows(self) -> None:
        self.reset_count += 1


class _FakePlanner:
    def __init__(self) -> None:
        self.calls = []

    def plan(self, *, goal, observation, prev_memory, step_index):
        self.calls.append(
            {
                "goal": goal,
                "prev_memory": prev_memory,
                "step_index": step_index,
            }
        )
        return MEMPlan(
            subtask=f"{goal} / subtask {len(self.calls)}",
            memory=f"memory {len(self.calls)}",
            raw_output="raw",
            planner_name="fake",
        )


def test_passthrough_mem_planner_preserves_goal_as_subtask() -> None:
    planner = PassthroughMEMPlanner()

    plan = planner.plan(
        goal="place the can",
        observation=_observation(),
        prev_memory="",
        step_index=0,
    )

    assert plan.subtask == "place the can"
    assert plan.memory == ""
    assert plan.planner_name == "passthrough"


def test_pi05_mem_controller_replans_and_updates_pi05_language() -> None:
    pi05 = _FakePi05Policy()
    planner = _FakePlanner()
    controller = Pi05MEMController(
        pi05,
        planner=planner,
        plan_interval_steps=1,
        action_steps=1,
    )

    first_actions = controller.act("global goal", _observation(1))
    second_actions = controller.act("global goal", _observation(2))

    np.testing.assert_allclose(first_actions, np.array([[0.1] * 14], dtype=np.float32))
    assert second_actions.shape == (1, 14)
    assert pi05.languages == [
        "global goal / subtask 1",
        "global goal / subtask 2",
    ]
    assert [call["prev_memory"] for call in planner.calls] == ["", "memory 1"]
    assert len(pi05.updates) == 2
    assert pi05.updates[0][0][0].shape == (3, 4, 3)
    np.testing.assert_allclose(pi05.updates[0][1], np.arange(14, dtype=np.float32))


def test_pi05_mem_controller_reset_clears_memory_and_low_level_state() -> None:
    pi05 = _FakePi05Policy()
    planner = _FakePlanner()
    controller = Pi05MEMController(pi05, planner=planner)
    controller.act("goal", _observation())

    controller.reset()
    controller.act("goal", _observation())

    assert pi05.reset_count == 1
    assert planner.calls[-1]["prev_memory"] == ""


def test_pi05_mem_controller_exposes_single_argument_server_methods() -> None:
    pi05 = _FakePi05Policy()
    planner = _FakePlanner()
    controller = Pi05MEMController(pi05, planner=planner, action_steps=1)

    actions = controller.act_request(
        {
            "goal": "server goal",
            "observation": _observation(),
        }
    )
    controller.reset_model()

    assert actions.shape == (1, 14)
    assert pi05.languages == ["server goal / subtask 1"]
    assert pi05.reset_count == 1
