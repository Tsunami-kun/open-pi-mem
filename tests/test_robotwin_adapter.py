from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from open_pi_mem.robotwin.adapter import (
    RoboTwinAdapter,
    encode_robotwin_observation,
    select_action_steps,
)


def _sample_observation() -> dict:
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[:, :, 0] = 32
    return {
        "observation": {
            "head_camera": {"rgb": rgb},
            "left_camera": {"rgb": rgb + 1},
            "right_camera": {"rgb": rgb + 2},
        },
        "joint_action": {
            "left_arm": [0.1, 0.2],
            "left_gripper": 0.3,
            "right_arm": [0.4, 0.5],
            "right_gripper": 0.6,
            "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        },
    }


def test_encode_robotwin_observation_uses_camera_images_and_joint_vector() -> None:
    encoded = encode_robotwin_observation(_sample_observation())

    assert [frame.size for frame in encoded.frames] == [(5, 4), (5, 4), (5, 4)]
    assert all(isinstance(frame, Image.Image) for frame in encoded.frames)
    assert encoded.proprio_state == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_select_action_steps_limits_and_clips_model_output() -> None:
    actions = np.array(
        [
            [-2.0, 0.0, 2.0],
            [0.25, 0.5, 0.75],
            [9.0, 9.0, 9.0],
        ],
        dtype=np.float32,
    )

    selected = select_action_steps(actions, max_steps=2, action_clip=(-1.0, 1.0))

    np.testing.assert_allclose(
        selected,
        np.array([[-1.0, 0.0, 1.0], [0.25, 0.5, 0.75]], dtype=np.float32),
    )


def test_robotwin_adapter_calls_policy_with_instruction_context() -> None:
    class FakePolicy:
        def __init__(self) -> None:
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            return {"action_chunk": np.array([[0.1, 0.2], [0.3, 0.4]])}

        def reset(self) -> None:
            self.calls.append({"reset": True})

    policy = FakePolicy()
    adapter = RoboTwinAdapter(policy=policy, max_action_steps=1)

    actions = adapter.get_action(_sample_observation(), instruction="place can")

    np.testing.assert_allclose(actions, np.array([[0.1, 0.2]], dtype=np.float32))
    assert policy.calls[0]["goal"] == "place can"
    assert policy.calls[0]["current_subtask"] == "place can"
    assert policy.calls[0]["proprio_state"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert len(policy.calls[0]["frames"]) == 3


def test_robotwin_policy_module_exports_expected_robo_twin_hooks() -> None:
    policy_root = Path("integrations/robotwin").resolve()
    sys.path.insert(0, str(policy_root))
    try:
        module = importlib.import_module("open_pi_mem_robotwin")
    finally:
        sys.path.remove(str(policy_root))

    assert callable(module.get_model)
    assert callable(module.eval)
    assert callable(module.reset_model)


def test_robotwin_policy_module_parses_yaml_null_and_numeric_overrides() -> None:
    policy_root = Path("integrations/robotwin").resolve()
    sys.path.insert(0, str(policy_root))
    try:
        deploy_policy = importlib.import_module("open_pi_mem_robotwin.deploy_policy")
    finally:
        sys.path.remove(str(policy_root))

    assert deploy_policy._optional_int(None) is None
    assert deploy_policy._optional_int("null") is None
    assert deploy_policy._optional_int("8") == 8
    assert deploy_policy._action_clip(
        {"action_clip_min": "-0.5", "action_clip_max": "0.75"}
    ) == (-0.5, 0.75)
    assert deploy_policy._action_clip({"disable_action_clip": "true"}) is None
