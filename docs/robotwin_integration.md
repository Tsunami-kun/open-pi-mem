# RoboTwin Integration

This repository includes a thin RoboTwin policy wrapper source at:

```text
integrations/robotwin/open_pi_mem_robotwin/
integrations/robotwin/open_pi_mem_pi05/
integrations/robotwin/open_pi_mem_pi06/
```

Install the needed wrapper into a local RoboTwin checkout so RoboTwin can import
open-pi-mem as a policy with:

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

This copies the selected wrapper into `benchmarks/RoboTwin/policy/`. The
benchmark checkout stays ignored by Git because it is external code plus large
local assets.

To also install the pi05+MEM policy variant:

```bash
python3 scripts/install_robotwin_policy_wrapper.py --robotwin-root benchmarks/RoboTwin --wrapper all
```

See [RoboTwin pi05/pi06 Architecture](robotwin_pi06_architecture.md) for the
recommended OpenPI/pi05 client-server runtime.

## Prepare A pi05 Checkpoint

For the public RoboTwin pi05 checkpoint used in this workspace:

```bash
python scripts/prepare_robotwin_pi05_checkpoint.py \
  --robotwin-root benchmarks/RoboTwin \
  --repo-id Crelf/C3I_pi05_Robotwin_50tasks_model_democlean \
  --model-name C3I_pi05_Robotwin_50tasks_model_democlean \
  --checkpoint-id auto \
  --disable-xet \
  --max-workers 1
```

The installed checkpoint should resolve to:

```text
benchmarks/RoboTwin/policy/pi05/checkpoints/pi05_base_aloha_lora/C3I_pi05_Robotwin_50tasks_model_democlean/35000
```

Expected inference files include `model.safetensors` and
`assets/Robotwin_50tasks_lerobot_data_clean/norm_stats.json`.

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

## Remote pi05/pi06 Runtime

The recommended pi05/pi06 path runs OpenPI in the pi05 virtual environment and
RoboTwin in the simulator conda environment.

Start pi05 baseline policy serving:

```bash
PYTHONPATH=/path/to/open-pi-mem/src \
CUDA_VISIBLE_DEVICES= \
OPENPI_PYTORCH_DEVICE=cpu \
TORCH_COMPILE_DISABLE=1 \
TORCHDYNAMO_DISABLE=1 \
/path/to/open-pi-mem/benchmarks/RoboTwin/policy/pi05/.venv/bin/python \
  /path/to/open-pi-mem/scripts/serve_robotwin_pi05_policy.py \
  --robotwin-root /path/to/open-pi-mem/benchmarks/RoboTwin \
  --mode pi05 \
  --train-config-name pi05_base_aloha_lora \
  --model-name C3I_pi05_Robotwin_50tasks_model_democlean \
  --checkpoint-id 35000 \
  --pi0-step 50 \
  --port 8105
```

Run a RoboTwin client smoke from `benchmarks/RoboTwin`:

```bash
conda run -n open-pi-mem-robotwin python script/eval_policy.py \
  --config policy/open_pi_mem_pi05/deploy_policy.yml \
  --overrides \
  --task_name place_can_basket \
  --task_config demo_clean \
  --train_config_name pi05_base_aloha_lora \
  --model_name C3I_pi05_Robotwin_50tasks_model_democlean \
  --checkpoint_id 35000 \
  --ckpt_setting C3I_pi05_Robotwin_50tasks_model_democlean \
  --seed 0 \
  --policy_name open_pi_mem_pi05 \
  --policy_host 127.0.0.1 \
  --policy_port 8105 \
  --eval_video_log False
```

Use `--mode pi06` on the server and `policy/open_pi_mem_pi06/deploy_policy.yml`
with `--policy_name open_pi_mem_pi06` for the current MEM-wrapped baseline.

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
python3 -m pytest \
  tests/test_robotwin_adapter.py \
  tests/test_robotwin_remote_policy.py \
  tests/test_robotwin_pi05_policy.py \
  tests/test_robotwin_pi06_policy.py \
  tests/test_rmbench_adapter.py
python3 scripts/check_robotwin_integration.py --robotwin-root benchmarks/RoboTwin
```

## Current Local Runtime Limit

This workspace can verify checkpoint download, checkpoint layout, simulator
imports, wrapper installation, websocket transport, and that RoboTwin reaches
the real pi05 policy call.

It cannot produce useful pi05/pi06 success rates on the current local hardware:

- the RTX 5060 Laptop GPU has 8 GB VRAM and is also used by the desktop,
  remote-session processes, and RoboTwin rendering;
- the installed pi05 PyTorch build reports no native support for `sm_120`;
- loading pi05 on GPU has OOMed before a simulator rollout can run;
- CPU fallback reaches pi05 action sampling, but one-action RoboTwin smoke runs
  timed out at 900 seconds while the server remained CPU-bound.

For benchmark numbers, run the same server/client commands on a larger GPU or
serve pi05/pi06 from a separate GPU host while the RoboTwin simulator runs
locally.
