from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_pi06_policy_module_exports_robo_twin_hooks() -> None:
    policy_root = Path("integrations/robotwin").resolve()
    sys.path.insert(0, str(policy_root))
    try:
        module = importlib.import_module("open_pi_mem_pi06")
    finally:
        sys.path.remove(str(policy_root))

    assert callable(module.get_model)
    assert callable(module.eval)
    assert callable(module.reset_model)


def test_pi06_policy_module_parses_mem_arguments() -> None:
    policy_root = Path("integrations/robotwin").resolve()
    sys.path.insert(0, str(policy_root))
    try:
        deploy_policy = importlib.import_module("open_pi_mem_pi06.deploy_policy")
    finally:
        sys.path.remove(str(policy_root))

    assert deploy_policy._optional_int(None) is None
    assert deploy_policy._optional_int("null") is None
    assert deploy_policy._optional_int("3") == 3
    assert deploy_policy._planner_kwargs(
        {
            "mem_planner_mode": "passthrough",
            "mem_plan_interval_steps": "5",
            "pi0_step": "10",
        }
    ) == {
        "planner_mode": "passthrough",
        "plan_interval_steps": 5,
        "action_steps": 10,
    }


def test_pi06_loads_pi05_model_without_importing_pi05_package_init(
    tmp_path: Path,
) -> None:
    policy_root = Path("integrations/robotwin").resolve()
    sys.path.insert(0, str(policy_root))
    try:
        deploy_policy = importlib.import_module("open_pi_mem_pi06.deploy_policy")
    finally:
        sys.path.remove(str(policy_root))

    pi05_dir = tmp_path / "policy" / "pi05"
    pi05_dir.mkdir(parents=True)
    (pi05_dir / "__init__.py").write_text(
        "raise RuntimeError('package init should not run')\n",
        encoding="utf-8",
    )
    (pi05_dir / "pi_model.py").write_text(
        "class PI0:\n"
        "    pass\n",
        encoding="utf-8",
    )

    cls = deploy_policy._load_pi05_class(tmp_path)

    assert cls.__name__ == "PI0"


def test_pi06_eval_uses_remote_model_client_when_available() -> None:
    policy_root = Path("integrations/robotwin").resolve()
    sys.path.insert(0, str(policy_root))
    try:
        deploy_policy = importlib.import_module("open_pi_mem_pi06.deploy_policy")
    finally:
        sys.path.remove(str(policy_root))

    class FakeTaskEnv:
        def __init__(self) -> None:
            self.actions = []

        def get_instruction(self) -> str:
            return "remote goal"

        def take_action(self, action) -> None:
            self.actions.append(action)

        def get_obs(self) -> dict:
            return {"next": True}

    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        def call(self, *, func_name, obs=None):
            self.calls.append({"func_name": func_name, "obs": obs})
            return [[0.1] * 14, [0.2] * 14]

    env = FakeTaskEnv()
    client = FakeClient()

    result = deploy_policy.eval(env, client, {"observation": "value"})

    assert result == {"next": True}
    assert client.calls == [
        {
            "func_name": "act_request",
            "obs": {
                "goal": "remote goal",
                "observation": {"observation": "value"},
            },
        }
    ]
    assert env.actions == [[0.1] * 14, [0.2] * 14]
