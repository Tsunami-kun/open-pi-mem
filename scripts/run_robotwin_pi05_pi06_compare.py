from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MissingArtifact:
    name: str
    path: Path


@dataclass(frozen=True)
class RobotwinEvalCommand:
    robotwin_root: Path
    policy_name: str
    task_name: str
    task_config: str
    train_config_name: str
    model_name: str
    checkpoint_id: int
    seed: int
    gpu_id: int

    def argv(self) -> list[str]:
        return [
            "python",
            "script/eval_policy.py",
            "--config",
            f"policy/{self.policy_name}/deploy_policy.yml",
            "--overrides",
            "--task_name",
            self.task_name,
            "--task_config",
            self.task_config,
            "--train_config_name",
            self.train_config_name,
            "--model_name",
            self.model_name,
            "--checkpoint_id",
            str(self.checkpoint_id),
            "--ckpt_setting",
            self.model_name,
            "--seed",
            str(self.seed),
            "--policy_name",
            self.policy_name,
        ]

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["PYTHONWARNINGS"] = "ignore::UserWarning"
        return env

    def display(self) -> str:
        prefix = f"CUDA_VISIBLE_DEVICES={self.gpu_id} PYTHONWARNINGS=ignore::UserWarning"
        return f"{prefix} {shlex.join(self.argv())}"


def parse_result_file(path: Path) -> float:
    values = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"No success-rate value found in {path}")
    return values[0]


def required_pi05_artifacts(
    robotwin_root: Path,
    *,
    train_config_name: str,
    model_name: str,
    checkpoint_id: int,
) -> list[MissingArtifact]:
    assets_dir = (
        robotwin_root
        / "policy"
        / "pi05"
        / "checkpoints"
        / train_config_name
        / model_name
        / str(checkpoint_id)
        / "assets"
    )
    checkpoint_dir = assets_dir.parent
    missing = []
    if not assets_dir.is_dir():
        missing.append(MissingArtifact("pi05 checkpoint assets", assets_dir))
    if not (
        (checkpoint_dir / "params").exists()
        or (checkpoint_dir / "model.safetensors").is_file()
    ):
        missing.append(MissingArtifact("pi05 checkpoint weights", checkpoint_dir))
    return missing


def latest_result_file(robotwin_root: Path, command: RobotwinEvalCommand) -> Path:
    root = (
        robotwin_root
        / "eval_result"
        / command.task_name
        / command.policy_name
        / command.task_config
        / command.model_name
    )
    candidates = sorted(root.glob("*/_result.txt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No RoboTwin result file found under {root}")
    return candidates[-1]


def run_eval(command: RobotwinEvalCommand, *, dry_run: bool) -> float | None:
    if dry_run:
        print(command.display())
        return None
    subprocess.run(
        command.argv(),
        cwd=command.robotwin_root,
        check=True,
        env=command.env(),
    )
    result_file = latest_result_file(command.robotwin_root, command)
    return parse_result_file(result_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", default="benchmarks/RoboTwin")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--train-config-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-id", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-artifact-check", action="store_true")
    args = parser.parse_args()

    robotwin_root = Path(args.robotwin_root).resolve()
    if not args.skip_artifact_check:
        missing = required_pi05_artifacts(
            robotwin_root,
            train_config_name=args.train_config_name,
            model_name=args.model_name,
            checkpoint_id=args.checkpoint_id,
        )
        if missing:
            for artifact in missing:
                print(f"missing {artifact.name}: {artifact.path}")
            raise SystemExit(2)

    commands = [
        RobotwinEvalCommand(
            robotwin_root=robotwin_root,
            policy_name=policy_name,
            task_name=args.task_name,
            task_config=args.task_config,
            train_config_name=args.train_config_name,
            model_name=args.model_name,
            checkpoint_id=args.checkpoint_id,
            seed=args.seed,
            gpu_id=args.gpu_id,
        )
        for policy_name in ("pi05", "open_pi_mem_pi06")
    ]
    results: dict[str, float | None] = {}
    for command in commands:
        results[command.policy_name] = run_eval(command, dry_run=args.dry_run)

    if not args.dry_run:
        for policy_name, success_rate in results.items():
            print(f"{policy_name}: success_rate={success_rate:.4f}")


if __name__ == "__main__":
    main()
