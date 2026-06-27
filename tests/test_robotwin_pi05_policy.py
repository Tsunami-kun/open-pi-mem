from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


def test_pi05_policy_module_exports_robo_twin_hooks() -> None:
    policy_root = Path("integrations/robotwin").resolve()
    sys.path.insert(0, str(policy_root))
    try:
        module = importlib.import_module("open_pi_mem_pi05")
    finally:
        sys.path.remove(str(policy_root))

    assert callable(module.get_model)
    assert callable(module.eval)
    assert callable(module.reset_model)


def test_pi05_eval_uses_openpi_websocket_client_when_available() -> None:
    policy_root = Path("integrations/robotwin").resolve()
    sys.path.insert(0, str(policy_root))
    try:
        deploy_policy = importlib.import_module("open_pi_mem_pi05.deploy_policy")
    finally:
        sys.path.remove(str(policy_root))

    class FakeTaskEnv:
        def __init__(self) -> None:
            self.actions = []

        def get_instruction(self) -> str:
            return "baseline goal"

        def take_action(self, action) -> None:
            self.actions.append(action)

        def get_obs(self) -> dict:
            return {"next": len(self.actions)}

    class FakeOpenPIClient:
        def __init__(self) -> None:
            self.requests = []

        def infer(self, request):
            self.requests.append(request)
            return {"actions": [[0.5] * 14, [0.6] * 14]}

    env = FakeTaskEnv()
    client = FakeOpenPIClient()

    result = deploy_policy.eval(env, client, {"observation": "value"})

    assert result == {"next": 2}
    assert client.requests == [
        {
            "goal": "baseline goal",
            "observation": {"observation": "value"},
        }
    ]
    np.testing.assert_allclose(env.actions, [[0.5] * 14, [0.6] * 14])
