from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def install_policy_wrapper(robotwin_root: Path, *, overwrite: bool = True) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "integrations" / "robotwin" / "open_pi_mem_robotwin"
    destination = robotwin_root.resolve() / "policy" / "open_pi_mem_robotwin"

    if not source.is_dir():
        raise FileNotFoundError(f"missing wrapper source: {source}")
    if not (robotwin_root / "policy").is_dir():
        raise FileNotFoundError(f"missing RoboTwin policy directory: {robotwin_root / 'policy'}")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"wrapper already exists: {destination}")
        shutil.rmtree(destination)

    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    eval_script = destination / "eval.sh"
    eval_script.chmod(eval_script.stat().st_mode | 0o111)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", default="benchmarks/RoboTwin")
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    destination = install_policy_wrapper(
        Path(args.robotwin_root),
        overwrite=not args.no_overwrite,
    )
    print(f"Installed RoboTwin policy wrapper: {destination}")


if __name__ == "__main__":
    main()
