from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

from open_pi_mem.robotwin.remote_policy import (
    RobotwinPi05RemotePolicy,
    RobotwinPi06RemotePolicy,
    serve_openpi_policy_forever,
)


def _load_pi05_class(robotwin_root: Path):
    pi_model_path = robotwin_root / "policy" / "pi05" / "pi_model.py"
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


def _build_pi05_policy(args: argparse.Namespace) -> Any:
    pi05_class = _load_pi05_class(args.robotwin_root)
    return pi05_class(
        args.train_config_name,
        args.model_name,
        args.checkpoint_id,
        args.pi0_step,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", default="benchmarks/RoboTwin", type=Path)
    parser.add_argument("--mode", choices=["pi05", "pi06"], required=True)
    parser.add_argument("--train-config-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-id", type=int, required=True)
    parser.add_argument("--pi0-step", type=int, default=50)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mem-plan-interval-steps", type=int, default=1)
    args = parser.parse_args()
    args.robotwin_root = args.robotwin_root.resolve()
    os.chdir(args.robotwin_root)

    pi05_policy = _build_pi05_policy(args)
    if args.mode == "pi05":
        policy = RobotwinPi05RemotePolicy(pi05_policy, action_steps=args.pi0_step)
    else:
        policy = RobotwinPi06RemotePolicy(
            pi05_policy,
            plan_interval_steps=args.mem_plan_interval_steps,
            action_steps=args.pi0_step,
        )

    serve_openpi_policy_forever(
        policy,
        host="0.0.0.0",
        port=args.port,
        metadata={
            "mode": args.mode,
            "model_name": args.model_name,
            "checkpoint_id": args.checkpoint_id,
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
