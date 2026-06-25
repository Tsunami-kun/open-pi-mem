from __future__ import annotations

from typing import Any

from open_pi_mem.robotwin.adapter import RoboTwinAdapter


def _is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _is_null(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "none", "null"}
    )


def _optional_int(value: Any) -> int | None:
    if _is_null(value):
        return None
    return int(value)


def _action_clip(usr_args: dict[str, Any]) -> tuple[float, float] | None:
    if _is_truthy(usr_args.get("disable_action_clip", False)):
        return None
    return (
        float(usr_args.get("action_clip_min", -1.0)),
        float(usr_args.get("action_clip_max", 1.0)),
    )


def get_model(usr_args: dict[str, Any]) -> RoboTwinAdapter:
    checkpoint_path = usr_args.get("checkpoint_path") or usr_args.get("model_path")
    if not checkpoint_path:
        raise ValueError(
            "open_pi_mem_robotwin requires checkpoint_path or model_path"
        )

    return RoboTwinAdapter(
        model_path=checkpoint_path,
        device=usr_args.get("device", "cuda"),
        action_scale=float(usr_args.get("action_scale", 1.0)),
        action_clip=_action_clip(usr_args),
        max_action_steps=_optional_int(usr_args.get("max_action_steps")),
        memory_enabled=_is_truthy(usr_args.get("memory_enabled", False)),
    )


def reset_model(model: RoboTwinAdapter) -> None:
    model.reset()


def eval(TASK_ENV, model: RoboTwinAdapter, observation: dict):
    instruction = TASK_ENV.get_instruction()
    actions = model.get_action(observation, instruction=instruction)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation
