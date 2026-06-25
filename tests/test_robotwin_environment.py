from __future__ import annotations

import sys
import types
from pathlib import Path

from open_pi_mem.robotwin.environment import check_robotwin_integration


def test_check_robotwin_integration_reports_policy_wrapper(tmp_path: Path) -> None:
    root = tmp_path / "RoboTwin"
    wrapper = root / "policy" / "open_pi_mem_robotwin"
    wrapper.mkdir(parents=True)
    (wrapper / "__init__.py").write_text(
        "def get_model(args): return None\n"
        "def eval(env, model, observation): return observation\n"
        "def reset_model(model): return None\n",
        encoding="utf-8",
    )
    (wrapper / "deploy_policy.py").write_text("", encoding="utf-8")

    result = check_robotwin_integration(root)

    assert result.robotwin_root.name == "RoboTwin"
    assert result.has_policy_wrapper is True
    assert result.policy_import_ok is True
    assert result.open_pi_mem_import_ok is True


def test_check_robotwin_integration_reports_missing_root() -> None:
    result = check_robotwin_integration(Path("benchmarks/does-not-exist"))

    assert result.robotwin_root_exists is False
    assert result.has_policy_wrapper is False
    assert result.policy_import_ok is False


def test_check_robotwin_integration_reports_asset_readiness(tmp_path: Path) -> None:
    root = tmp_path / "RoboTwin"
    wrapper = root / "policy" / "open_pi_mem_robotwin"
    assets = root / "assets" / "objects" / "objaverse"
    wrapper.mkdir(parents=True)
    assets.mkdir(parents=True)
    (wrapper / "deploy_policy.py").write_text("", encoding="utf-8")

    missing_assets = check_robotwin_integration(root)
    assert missing_assets.assets_ready is False

    (assets / "list.json").write_text("[]", encoding="utf-8")
    ready_assets = check_robotwin_integration(root)
    assert ready_assets.assets_ready is True


def test_check_robotwin_integration_restores_existing_policy_module(
    tmp_path: Path,
) -> None:
    module_name = "open_pi_mem_robotwin"
    submodule_name = f"{module_name}.deploy_policy"
    original_package = sys.modules.get(module_name)
    original_submodule = sys.modules.get(submodule_name)
    previous_package = types.ModuleType(module_name)
    previous_submodule = types.ModuleType(submodule_name)
    try:
        sys.modules[module_name] = previous_package
        sys.modules[submodule_name] = previous_submodule

        root = tmp_path / "RoboTwin"
        wrapper = root / "policy" / module_name
        wrapper.mkdir(parents=True)
        (wrapper / "__init__.py").write_text(
            "def get_model(args): return None\n"
            "def eval(env, model, observation): return observation\n"
            "def reset_model(model): return None\n",
            encoding="utf-8",
        )
        (wrapper / "deploy_policy.py").write_text("", encoding="utf-8")

        result = check_robotwin_integration(root)

        assert result.policy_import_ok is True
        assert sys.modules[module_name] is previous_package
        assert sys.modules[submodule_name] is previous_submodule
    finally:
        for name, module in (
            (module_name, original_package),
            (submodule_name, original_submodule),
        ):
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
