from __future__ import annotations

from pathlib import Path

from scripts.run_robotwin_pi05_pi06_compare import (
    MissingArtifact,
    RobotwinEvalCommand,
    parse_result_file,
    required_pi05_artifacts,
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
