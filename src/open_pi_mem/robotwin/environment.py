from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path


SIMULATOR_MODULES = ("sapien", "mplib", "transforms3d", "gymnasium")
REQUIRED_ASSET_FILES = ("assets/objects/objaverse/list.json",)


@dataclass(frozen=True)
class RoboTwinIntegrationCheck:
    robotwin_root: Path
    robotwin_root_exists: bool
    has_policy_wrapper: bool
    policy_import_ok: bool
    open_pi_mem_import_ok: bool
    missing_simulator_modules: tuple[str, ...]
    missing_asset_files: tuple[str, ...]

    @property
    def core_ready(self) -> bool:
        return (
            self.robotwin_root_exists
            and self.has_policy_wrapper
            and self.policy_import_ok
            and self.open_pi_mem_import_ok
        )

    @property
    def simulator_ready(self) -> bool:
        return self.core_ready and not self.missing_simulator_modules

    @property
    def assets_ready(self) -> bool:
        return self.robotwin_root_exists and not self.missing_asset_files

    @property
    def evaluation_ready(self) -> bool:
        return self.simulator_ready and self.assets_ready


def _import_policy_wrapper(policy_root: Path) -> bool:
    module_name = "open_pi_mem_robotwin"
    module_prefix = f"{module_name}."
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == module_name or name.startswith(module_prefix)
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(policy_root))
    try:
        module = importlib.import_module(module_name)
        return all(
            callable(getattr(module, attr, None))
            for attr in ("get_model", "eval", "reset_model")
        )
    except Exception:
        return False
    finally:
        try:
            sys.path.remove(str(policy_root))
        except ValueError:
            pass
        for name in list(sys.modules):
            if name == module_name or name.startswith(module_prefix):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


def check_robotwin_integration(
    robotwin_root: str | Path = "benchmarks/RoboTwin",
) -> RoboTwinIntegrationCheck:
    root = Path(robotwin_root).resolve()
    policy_root = root / "policy"
    wrapper = policy_root / "open_pi_mem_robotwin"
    root_exists = root.is_dir()
    has_policy_wrapper = (wrapper / "deploy_policy.py").is_file()

    policy_import_ok = False
    if root_exists and has_policy_wrapper:
        policy_import_ok = _import_policy_wrapper(policy_root)

    open_pi_mem_import_ok = find_spec("open_pi_mem") is not None
    missing_simulator_modules = tuple(
        module for module in SIMULATOR_MODULES if find_spec(module) is None
    )
    missing_asset_files = tuple(
        asset for asset in REQUIRED_ASSET_FILES if not (root / asset).is_file()
    )
    return RoboTwinIntegrationCheck(
        robotwin_root=root,
        robotwin_root_exists=root_exists,
        has_policy_wrapper=has_policy_wrapper,
        policy_import_ok=policy_import_ok,
        open_pi_mem_import_ok=open_pi_mem_import_ok,
        missing_simulator_modules=missing_simulator_modules,
        missing_asset_files=missing_asset_files,
    )
