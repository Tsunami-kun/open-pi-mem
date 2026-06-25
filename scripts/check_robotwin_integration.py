from __future__ import annotations

import argparse

from open_pi_mem.robotwin.environment import check_robotwin_integration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", default="benchmarks/RoboTwin")
    args = parser.parse_args()

    result = check_robotwin_integration(args.robotwin_root)
    print(f"RoboTwin root: {result.robotwin_root}")
    print(f"RoboTwin root exists: {result.robotwin_root_exists}")
    print(f"open-pi-mem import: {result.open_pi_mem_import_ok}")
    print(f"policy wrapper present: {result.has_policy_wrapper}")
    print(f"policy wrapper import: {result.policy_import_ok}")
    print(f"core integration ready: {result.core_ready}")
    print(f"simulator ready: {result.simulator_ready}")
    print(f"assets ready: {result.assets_ready}")
    print(f"evaluation ready: {result.evaluation_ready}")
    if result.missing_simulator_modules:
        print(
            "missing simulator modules: "
            + ", ".join(result.missing_simulator_modules)
        )
    else:
        print("missing simulator modules: none")
    if result.missing_asset_files:
        print("missing asset files: " + ", ".join(result.missing_asset_files))
    else:
        print("missing asset files: none")

    if not result.core_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
