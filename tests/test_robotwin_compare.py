from __future__ import annotations

from pathlib import Path

from scripts.run_robotwin_pi05_pi06_compare import (
    MissingArtifact,
    RobotwinEvalCommand,
    parse_result_file,
    required_pi05_artifacts,
)
from scripts.prepare_robotwin_pi05_checkpoint import (
    HfFile,
    build_plan,
    infer_source_subdir,
    selected_checkpoint_files,
    target_relative_path,
)


def test_parse_result_file_reads_first_success_rate(tmp_path: Path) -> None:
    result_file = tmp_path / "_result.txt"
    result_file.write_text(
        "Timestamp: 2026-06-26 12:00:00\n\n"
        "Instruction Type: unseen\n\n"
        "0.42\n",
        encoding="utf-8",
    )

    assert parse_result_file(result_file) == 0.42


def test_robotwin_eval_command_builds_policy_specific_command() -> None:
    command = RobotwinEvalCommand(
        robotwin_root=Path("/repo/benchmarks/RoboTwin"),
        policy_name="open_pi_mem_pi06",
        task_name="place_can_basket",
        task_config="demo_clean",
        train_config_name="pi05_base_aloha_lora",
        model_name="demo_ckpt",
        checkpoint_id=30000,
        seed=0,
        gpu_id=0,
    )

    argv = command.argv()

    assert argv[:4] == [
        "python",
        "script/eval_policy.py",
        "--config",
        "policy/open_pi_mem_pi06/deploy_policy.yml",
    ]
    assert "--checkpoint_id" in argv
    assert argv[argv.index("--checkpoint_id") + 1] == "30000"
    assert command.env()["CUDA_VISIBLE_DEVICES"] == "0"


def test_required_pi05_artifacts_reports_missing_checkpoint_assets(
    tmp_path: Path,
) -> None:
    missing = required_pi05_artifacts(
        tmp_path,
        train_config_name="pi05_base_aloha_lora",
        model_name="missing_model",
        checkpoint_id=30000,
    )

    assert missing == [
        MissingArtifact(
            name="pi05 checkpoint assets",
            path=tmp_path
            / "policy"
            / "pi05"
            / "checkpoints"
            / "pi05_base_aloha_lora"
            / "missing_model"
            / "30000"
            / "assets",
        ),
        MissingArtifact(
            name="pi05 checkpoint weights",
            path=tmp_path
            / "policy"
            / "pi05"
            / "checkpoints"
            / "pi05_base_aloha_lora"
            / "missing_model"
            / "30000",
        ),
    ]


def test_required_pi05_artifacts_accepts_assets_and_params(tmp_path: Path) -> None:
    checkpoint_dir = (
        tmp_path
        / "policy"
        / "pi05"
        / "checkpoints"
        / "pi05_base_aloha_lora"
        / "demo_ckpt"
        / "30000"
    )
    (checkpoint_dir / "assets").mkdir(parents=True)
    (checkpoint_dir / "params").mkdir()

    missing = required_pi05_artifacts(
        tmp_path,
        train_config_name="pi05_base_aloha_lora",
        model_name="demo_ckpt",
        checkpoint_id=30000,
    )

    assert missing == []


def test_prepare_pi05_checkpoint_plan_selects_inference_files_only() -> None:
    files = [
        HfFile("35000/assets/demo/norm_stats.json", 10),
        HfFile("35000/model.safetensors", 100),
        HfFile("35000/metadata.pt", 20),
        HfFile("35000/optimizer.pt", 1000),
        HfFile("README.md", 1),
    ]

    plan = build_plan(
        repo_id="org/model",
        paths=files,
        train_config_name="pi05_base_aloha_lora",
        model_name="demo_model",
        checkpoint_id="auto",
        source_subdir=None,
        include_optimizer=False,
    )

    assert infer_source_subdir([file.path for file in files]) == "35000"
    assert plan.checkpoint_id == 35000
    assert [file.path for file in plan.files] == [
        "35000/assets/demo/norm_stats.json",
        "35000/model.safetensors",
        "35000/metadata.pt",
    ]
    assert target_relative_path(
        "35000/assets/demo/norm_stats.json",
        "35000",
    ) == Path("assets/demo/norm_stats.json")


def test_prepare_pi05_checkpoint_plan_can_include_jax_params() -> None:
    files = [
        HfFile("assets/demo/norm_stats.json", 10),
        HfFile("params/_METADATA", 100),
        HfFile("params/manifest.ocdbt", 100),
        HfFile("train_state/_METADATA", 1000),
    ]

    selected = selected_checkpoint_files(
        files,
        source_subdir="",
        include_optimizer=False,
    )

    assert [file.path for file in selected] == [
        "assets/demo/norm_stats.json",
        "params/_METADATA",
        "params/manifest.ocdbt",
    ]
