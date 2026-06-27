from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from open_pi_mem.robotwin.mem_controller import encode_pi05_observation
from open_pi_mem.robotwin.remote_policy import (
    RobotwinRemotePolicyClient,
    extract_actions,
)


def _is_null(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "none", "null"}
    )


def _optional_int(value: Any) -> int | None:
    if _is_null(value):
        return None
    return int(value)


def _remote_policy_args(usr_args: dict[str, Any]) -> tuple[str, int] | None:
    host = usr_args.get("policy_host")
    if _is_null(host):
        return None
    return str(host), _optional_int(usr_args.get("policy_port")) or 8000


def _load_pi05_class(robotwin_root: str | Path = "."):
    pi_model_path = Path(robotwin_root).resolve() / "policy" / "pi05" / "pi_model.py"
    if not pi_model_path.is_file():
        raise FileNotFoundError(f"missing pi05 model wrapper: {pi_model_path}")
    spec = importlib.util.spec_from_file_location(
        "_open_pi_mem_robotwin_pi05_model",
        pi_model_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load pi05 model wrapper: {pi_model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PI0


def get_model(usr_args: dict[str, Any]):
    remote_args = _remote_policy_args(usr_args)
    if remote_args is not None:
        return RobotwinRemotePolicyClient(*remote_args)

    pi05_class = _load_pi05_class(Path.cwd())
    return pi05_class(
        usr_args["train_config_name"],
        usr_args["model_name"],
        usr_args.get("checkpoint_id", 30000),
        usr_args["pi0_step"],
    )


def reset_model(model) -> None:
    reset = getattr(model, "reset", None)
    if callable(reset):
        reset()
        return
    model.reset_obsrvationwindows()


def eval(TASK_ENV, model, observation: dict):
    if hasattr(model, "infer"):
        response = model.infer(
            {
                "goal": TASK_ENV.get_instruction(),
                "observation": observation,
            }
        )
        actions = extract_actions(response)
    else:
        if model.observation_window is None:
            model.set_language(TASK_ENV.get_instruction())
        input_rgb_arr, input_state = encode_pi05_observation(observation)
        model.update_observation_window(input_rgb_arr, input_state)
        actions = model.get_action()[: model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        if not hasattr(model, "infer"):
            input_rgb_arr, input_state = encode_pi05_observation(observation)
            model.update_observation_window(input_rgb_arr, input_state)
    return observation
