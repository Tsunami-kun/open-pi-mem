from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _read_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_manifest_entry(report_path: Path, report: dict[str, Any], reports_root: Path) -> dict[str, Any]:
    task_dir = report_path.parent
    model_dir = task_dir.parent
    relative_report = Path("reports") / report_path.relative_to(reports_root)
    task_name = task_dir.name.replace("_", " ")
    model_label = report.get("model_path") or model_dir.name.replace("_", " ")
    title = f"{task_name} · {model_label}"
    return {
        "title": title,
        "task": task_name,
        "model_label": model_label,
        "goal": report.get("goal", ""),
        "path": relative_report.as_posix(),
        "records": len(report.get("records", [])),
        "provider": report.get("provider", ""),
    }


def _copy_report_bundle(report_path: Path, reports_root: Path, output_reports_root: Path) -> None:
    relative_task_dir = report_path.parent.relative_to(reports_root)
    target_task_dir = output_reports_root / relative_task_dir
    target_task_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, target_task_dir / "report.json")

    preview_src = report_path.parent / "preview_frames"
    if preview_src.exists():
        _copy_tree(preview_src, target_task_dir / "preview_frames")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static GitHub Pages site for the open-pi-mem viewer.")
    parser.add_argument(
        "--output",
        default="dist/github_pages",
        help="Output directory for the staged static site.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    viewer_root = repo_root / "web" / "viewer"
    reports_root = repo_root / "data" / "eval_results"
    output_root = (repo_root / args.output).resolve()
    output_reports_root = output_root / "reports"

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_reports_root.mkdir(parents=True, exist_ok=True)

    for item in viewer_root.iterdir():
        destination = output_root / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    manifest: list[dict[str, Any]] = []
    for report_path in sorted(reports_root.glob("*/*/report.json")):
        report = _read_report(report_path)
        _copy_report_bundle(report_path, reports_root, output_reports_root)
        manifest.append(_build_manifest_entry(report_path, report, reports_root))

    with (output_reports_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    (output_root / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built GitHub Pages site at {output_root}")
    print(f"Bundled {len(manifest)} report(s)")


if __name__ == "__main__":
    main()
