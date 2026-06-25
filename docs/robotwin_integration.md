# RoboTwin Integration

This repository includes a thin RoboTwin policy wrapper source at:

```text
integrations/robotwin/open_pi_mem_robotwin/
```

Install it into a local RoboTwin checkout so RoboTwin can import open-pi-mem as
a policy with:

```yaml
policy_name: open_pi_mem_robotwin
```

## Install open-pi-mem

From the `open-pi-mem` repository root:

```bash
python3 -m pip install -e ".[eval,dev]"
```

This installs the open-pi-mem package, model/runtime dependencies, and the test runner.

## Install RoboTwin

RoboTwin is checked out under:

```text
benchmarks/RoboTwin
```

Install RoboTwin's simulator dependencies from that directory, following the upstream RoboTwin 2.0 documentation. The local integration does not vendor those heavy simulator assets or robot embodiment assets.

The RoboTwin script requirements are listed at:

```text
benchmarks/RoboTwin/script/requirements.txt
```

RoboTwin's simulator stack should be installed in a Python 3.10-compatible
environment. The active open-pi-mem development environment may be newer than
RoboTwin's pinned simulator dependencies.

This workspace uses a separate conda environment for RoboTwin:

```bash
conda activate open-pi-mem-robotwin
```

## Install The Policy Wrapper

From the `open-pi-mem` repository root:

```bash
python3 scripts/install_robotwin_policy_wrapper.py --robotwin-root benchmarks/RoboTwin
```

This copies `integrations/robotwin/open_pi_mem_robotwin` into
`benchmarks/RoboTwin/policy/open_pi_mem_robotwin`. The benchmark checkout stays
ignored by Git because it is external code plus large local assets.

## Check Integration

From the `open-pi-mem` repository root:

```bash
python3 scripts/check_robotwin_integration.py --robotwin-root benchmarks/RoboTwin
```

Expected core-ready output includes:

```text
core integration ready: True
```

For a full rollout, the output also needs:

```text
evaluation ready: True
```

If `simulator ready` is `False`, install the missing RoboTwin simulator modules
in the RoboTwin environment before running a full rollout. If `assets ready` is
`False`, download and unpack RoboTwin assets:

```bash
cd benchmarks/RoboTwin
conda run -n open-pi-mem-robotwin bash script/_download_assets.sh
```

## Run Evaluation

After the RoboTwin simulator environment and assets are ready, run from:

```bash
cd benchmarks/RoboTwin/policy/open_pi_mem_robotwin
```

Then launch:

```bash
bash eval.sh <task_name> <task_config> <checkpoint_path> <seed> <gpu_id>
```

Example:

```bash
bash eval.sh place_can_basket demo_randomized /path/to/open_pi_mem_low_level.pt 0 0
```

The wrapper passes RoboTwin observations into `open_pi_mem.robotwin.RoboTwinAdapter`,
then executes the returned action chunk through `TASK_ENV.take_action`.

## Direct RoboTwin Command

The helper script expands to RoboTwin's standard evaluator:

```bash
cd benchmarks/RoboTwin
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/open_pi_mem_robotwin/deploy_policy.yml \
    --overrides \
    --task_name place_can_basket \
    --task_config demo_randomized \
    --checkpoint_path /path/to/open_pi_mem_low_level.pt \
    --ckpt_setting /path/to/open_pi_mem_low_level.pt \
    --seed 0 \
    --policy_name open_pi_mem_robotwin
```

Useful optional overrides:

```bash
--device cuda
--max_action_steps 8
--action_scale 1.0
--action_clip_min -1.0
--action_clip_max 1.0
--memory_enabled false
```

## Smoke Tests

From the `open-pi-mem` repository root:

```bash
python3 -m pytest tests/test_robotwin_adapter.py tests/test_rmbench_adapter.py
python3 scripts/check_robotwin_integration.py --robotwin-root benchmarks/RoboTwin
```
