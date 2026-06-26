from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HfFile:
    path: str
    size: int = 0


@dataclass(frozen=True)
class CheckpointPlan:
    repo_id: str
    source_subdir: str
    train_config_name: str
    model_name: str
    checkpoint_id: int
    files: tuple[HfFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(file.size for file in self.files)

    def destination(self, robotwin_root: Path) -> Path:
        return (
            robotwin_root
            / "policy"
            / "pi05"
            / "checkpoints"
            / self.train_config_name
            / self.model_name
            / str(self.checkpoint_id)
        )


def _top_level(path: str) -> str:
    return path.split("/", 1)[0]


def infer_source_subdir(paths: list[str]) -> str:
    numeric_dirs = sorted(
        {
            _top_level(path)
            for path in paths
            if "/" in path and _top_level(path).isdigit()
        }
    )
    if len(numeric_dirs) == 1:
        return numeric_dirs[0]
    return ""


def resolve_checkpoint_id(value: str, source_subdir: str) -> int:
    if value != "auto":
        return int(value)
    if source_subdir.isdigit():
        return int(source_subdir)
    return 30000


def target_relative_path(source_path: str, source_subdir: str) -> Path:
    if not source_subdir:
        return Path(source_path)
    prefix = f"{source_subdir}/"
    if not source_path.startswith(prefix):
        raise ValueError(f"{source_path} is outside source subdir {source_subdir}")
    return Path(source_path[len(prefix) :])


def selected_checkpoint_files(
    files: list[HfFile],
    *,
    source_subdir: str,
    include_optimizer: bool = False,
) -> tuple[HfFile, ...]:
    selected = []
    prefix = f"{source_subdir}/" if source_subdir else ""
    for file in files:
        path = file.path
        if prefix and not path.startswith(prefix):
            continue
        rel = target_relative_path(path, source_subdir).as_posix()
        if rel == "model.safetensors" or rel == "metadata.pt":
            selected.append(file)
        elif rel.startswith("assets/") or rel.startswith("params/"):
            selected.append(file)
        elif rel == "_CHECKPOINT_METADATA":
            selected.append(file)
        elif include_optimizer and (
            rel == "optimizer.pt" or rel.startswith("train_state/")
        ):
            selected.append(file)
    return tuple(selected)


def build_plan(
    *,
    repo_id: str,
    paths: list[HfFile],
    train_config_name: str,
    model_name: str,
    checkpoint_id: str,
    source_subdir: str | None,
    include_optimizer: bool,
) -> CheckpointPlan:
    raw_paths = [path.path for path in paths]
    resolved_source_subdir = source_subdir
    if resolved_source_subdir is None:
        resolved_source_subdir = infer_source_subdir(raw_paths)
    resolved_checkpoint_id = resolve_checkpoint_id(
        checkpoint_id,
        resolved_source_subdir,
    )
    selected = selected_checkpoint_files(
        paths,
        source_subdir=resolved_source_subdir,
        include_optimizer=include_optimizer,
    )
    if not selected:
        raise ValueError(
            "No OpenPI checkpoint files selected. Expected assets/ plus "
            "params/ or model.safetensors."
        )
    return CheckpointPlan(
        repo_id=repo_id,
        source_subdir=resolved_source_subdir,
        train_config_name=train_config_name,
        model_name=model_name,
        checkpoint_id=resolved_checkpoint_id,
        files=selected,
    )


def _format_size(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _link_or_copy(source: Path, destination: Path, *, link_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if link_mode == "copy":
        shutil.copy2(source, destination)
    elif link_mode == "symlink":
        os.symlink(source, destination)
    else:
        raise ValueError(f"unsupported link mode: {link_mode}")


def install_snapshot(
    *,
    snapshot_root: Path,
    plan: CheckpointPlan,
    destination: Path,
    link_mode: str,
    force: bool,
) -> None:
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(
            f"{destination} already exists; pass --force to replace selected files"
        )
    for file in plan.files:
        source = snapshot_root / file.path
        relative_destination = target_relative_path(file.path, plan.source_subdir)
        _link_or_copy(
            source,
            destination / relative_destination,
            link_mode=link_mode,
        )


def _load_hf_files(repo_id: str, revision: str | None) -> list[HfFile]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with "
            "`python -m pip install huggingface_hub`."
        ) from exc

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    return tuple(
        HfFile(path=sibling.rfilename, size=getattr(sibling, "size", None) or 0)
        for sibling in info.siblings
    )


def _download_snapshot(
    *,
    repo_id: str,
    revision: str | None,
    files: tuple[HfFile, ...],
    max_workers: int,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with "
            "`python -m pip install huggingface_hub`."
        ) from exc

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[file.path for file in files],
            max_workers=max_workers,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", default="benchmarks/RoboTwin")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--source-subdir")
    parser.add_argument("--train-config-name", default="pi05_base_aloha_lora")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-id", default="auto")
    parser.add_argument("--include-optimizer", action="store_true")
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--hf-transfer", action="store_true")
    parser.add_argument("--disable-xet", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    robotwin_root = Path(args.robotwin_root).resolve()
    hf_files = list(_load_hf_files(args.repo_id, args.revision))
    plan = build_plan(
        repo_id=args.repo_id,
        paths=hf_files,
        train_config_name=args.train_config_name,
        model_name=args.model_name,
        checkpoint_id=args.checkpoint_id,
        source_subdir=args.source_subdir,
        include_optimizer=args.include_optimizer,
    )
    destination = plan.destination(robotwin_root)

    print(f"repo: {plan.repo_id}")
    print(f"source subdir: {plan.source_subdir or '<repo root>'}")
    print(f"destination: {destination}")
    print(f"selected files: {len(plan.files)}")
    print(f"selected size: {_format_size(plan.total_bytes)}")
    for file in plan.files:
        print(f"  {file.path} ({_format_size(file.size)})")

    if args.dry_run:
        return

    snapshot_root = _download_snapshot(
        repo_id=plan.repo_id,
        revision=args.revision,
        files=plan.files,
        max_workers=args.max_workers,
    )
    install_snapshot(
        snapshot_root=snapshot_root,
        plan=plan,
        destination=destination,
        link_mode=args.link_mode,
        force=args.force,
    )
    print(f"installed checkpoint: {destination}")


if __name__ == "__main__":
    main()
