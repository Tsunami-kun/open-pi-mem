from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from open_pi_mem.robotwin.mem_controller import (
    PassthroughMEMPlanner,
    Pi05MEMController,
)


def _is_null(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "none", "null"}
    )


def _optional_int(value: Any) -> int | None:
    if _is_null(value):
        return None
    return int(value)


def _planner_kwargs(usr_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "planner_mode": str(usr_args.get("mem_planner_mode", "passthrough")),
        "plan_interval_steps": _optional_int(
            usr_args.get("mem_plan_interval_steps", 1)
        )
        or 1,
        "action_steps": _optional_int(usr_args.get("pi0_step")),
    }


def _build_planner(mode: str):
    if mode != "passthrough":
        raise ValueError(
            "open_pi_mem_pi06 currently supports mem_planner_mode=passthrough. "
            "Use this as the pi05-equivalent baseline before enabling a trained "
            "MEM planner."
        )
    return PassthroughMEMPlanner()


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


def get_model(usr_args: dict[str, Any]) -> Pi05MEMController:
    pi05_class = _load_pi05_class(Path.cwd())
    pi05_policy = pi05_class(
        usr_args["train_config_name"],
        usr_args["model_name"],
        usr_args.get("checkpoint_id", 30000),
        usr_args["pi0_step"],
    )
    planner_args = _planner_kwargs(usr_args)
    return Pi05MEMController(
        pi05_policy,
        planner=_build_planner(planner_args["planner_mode"]),
        plan_interval_steps=planner_args["plan_interval_steps"],
        action_steps=planner_args["action_steps"],
    )


def reset_model(model: Pi05MEMController) -> None:
    if hasattr(model, "call"):
        model.call(func_name="reset_model")
        return
    model.reset()


def eval(TASK_ENV, model: Pi05MEMController, observation: dict):
    instruction = TASK_ENV.get_instruction()
    if hasattr(model, "call"):
        actions = model.call(
            func_name="act_request",
            obs={"goal": instruction, "observation": observation},
        )
    else:
        actions = model.act(instruction, observation)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation
