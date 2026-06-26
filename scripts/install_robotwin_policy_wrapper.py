from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_WRAPPERS = ("open_pi_mem_robotwin", "open_pi_mem_pi06")


def install_policy_wrapper(
    robotwin_root: Path,
    *,
    wrapper_name: str = "open_pi_mem_robotwin",
    overwrite: bool = True,
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "integrations" / "robotwin" / wrapper_name
    destination = robotwin_root.resolve() / "policy" / wrapper_name

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
    parser.add_argument(
        "--wrapper",
        choices=[*DEFAULT_WRAPPERS, "all"],
        default="open_pi_mem_robotwin",
    )
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    wrapper_names = DEFAULT_WRAPPERS if args.wrapper == "all" else (args.wrapper,)
    for wrapper_name in wrapper_names:
        destination = install_policy_wrapper(
            Path(args.robotwin_root),
            wrapper_name=wrapper_name,
            overwrite=not args.no_overwrite,
        )
        print(f"Installed RoboTwin policy wrapper: {destination}")


if __name__ == "__main__":
    main()
