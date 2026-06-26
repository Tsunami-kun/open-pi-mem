# RoboTwin pi05/pi06 Architecture

`pi06` is the open-pi-mem RoboTwin policy variant:

```text
pi06 = pi05 low-level OpenPI actor + open-pi-mem MEM planning wrapper
```

It does not reimplement pi05. The low-level model, action normalization,
LeRobot data format, checkpoint layout, and training loop should come from
official OpenPI / RoboTwin pi05 code. `open-pi-mem` owns only the memory-planning
layer, trace/evaluation tooling, and wrapper scripts.

This boundary is deliberate. Physical Intelligence OpenPI already ships
pi05 model code, base checkpoints, LeRobot data utilities, policy construction,
and remote inference support. The RoboTwin checkout adds task-specific pi05
configs and a `PI0` policy wrapper. Reusing those pieces keeps pi06 comparable
to pi05 and avoids creating a second low-level policy stack.

## Reused Components

Use these directly from OpenPI / RoboTwin pi05:

- pi05 model architecture and policy creation
- pi05 base checkpoint loading and fine-tuned checkpoint layout
- ALOHA/RoboTwin image, state, action, and normalization transforms
- LeRobot conversion and normalization-stat computation
- qpos-only 14D action chunks and `pi0_step` execution
- remote policy-server pattern for separating simulator and OpenPI runtimes

`open-pi-mem` should add only:

- a planner API that maps `goal + observation + prev_memory` to
  `subtask + next_memory`
- a controller that feeds the selected subtask into the existing pi05 policy
- logging/evaluation hooks for memory, subtask, and success-rate comparison
- install and compare scripts that do not vendor model checkpoints or datasets

For the longer-running self-evolution loop, see
[RoboTwin Self-Evolving MEM Agent](robotwin_self_evolving_agent.md).

## Runtime Split

Use separate runtimes:

- RoboTwin simulator runtime: Python 3.10 with SAPIEN/MPLib simulator deps.
- OpenPI/pi05 runtime: Python 3.11+ with JAX/OpenPI deps and pi05 checkpoints.

This split is intentional. Official OpenPI has a heavier JAX stack and should
not be forced into the simulator process.

From the local RoboTwin pi05 checkout, create or verify the OpenPI runtime with:

```bash
cd benchmarks/RoboTwin/policy/pi05
uv run --python 3.11 python -c \
  "import jax, torch, openpi; print(jax.__version__, torch.cuda.is_available())"
```

For server-side pi06 imports from the RoboTwin root:

```bash
cd benchmarks/RoboTwin
PYTHONPATH=/path/to/open-pi-mem/src:./policy:./description/utils:./policy/pi05/src:./policy/pi05/packages/openpi-client/src \
  policy/pi05/.venv/bin/python -c \
  "import open_pi_mem_pi06; print(callable(open_pi_mem_pi06.get_model))"
```

## Policies

`pi05` baseline:

```text
RoboTwin instruction -> pi05/OpenPI policy -> qpos action chunk
```

`open_pi_mem_pi06`:

```text
RoboTwin instruction + observation + previous memory
  -> open-pi-mem MEM planner
  -> subtask + updated memory
  -> pi05/OpenPI policy
  -> qpos action chunk
```

The current committed `pi06` planner mode is `passthrough`, which preserves the
original instruction and exercises the MEM control path without changing pi05
behavior. This is the required baseline before enabling a trained or API-backed
planner.

## Installing Wrappers

From the repository root:

```bash
python scripts/install_robotwin_policy_wrapper.py \
  --robotwin-root benchmarks/RoboTwin \
  --wrapper all
```

This installs:

- `policy/open_pi_mem_robotwin`
- `policy/open_pi_mem_pi06`

## Direct In-Process Mode

Direct mode works only when the RoboTwin Python environment also has OpenPI/JAX
and pi05 checkpoint dependencies installed:

```bash
cd benchmarks/RoboTwin
bash policy/open_pi_mem_pi06/eval.sh \
  place_can_basket demo_clean pi05_base_aloha_lora <model_name> 0 0
```

This mode is useful for debugging but is not the recommended long-term runtime.

## Client/Server Mode

Recommended production shape:

1. Start a model server in the OpenPI/pi05 runtime.
2. Run RoboTwin simulator evaluation in the simulator runtime.
3. The simulator calls `open_pi_mem_pi06.eval(...)`.
4. `pi06` calls the remote server method `act_request`.

The server model should be `open_pi_mem_pi06`, not raw pi05, so MEM planning
and pi05 inference run together in the OpenPI runtime.

Server command shape:

```bash
cd benchmarks/RoboTwin
python script/policy_model_server.py \
  --port 9999 \
  --config policy/open_pi_mem_pi06/deploy_policy.yml \
  --overrides \
  --task_name place_can_basket \
  --task_config demo_clean \
  --train_config_name pi05_base_aloha_lora \
  --model_name <model_name> \
  --ckpt_setting <model_name> \
  --seed 0 \
  --policy_name open_pi_mem_pi06
```

Client evaluation command shape:

```bash
cd benchmarks/RoboTwin
python script/eval_policy_client.py \
  --port 9999 \
  --config policy/open_pi_mem_pi06/deploy_policy.yml \
  --overrides \
  --task_name place_can_basket \
  --task_config demo_clean \
  --train_config_name pi05_base_aloha_lora \
  --model_name <model_name> \
  --ckpt_setting <model_name> \
  --seed 0 \
  --policy_name open_pi_mem_pi06
```

## Success-Rate Compare

Once pi05 checkpoint assets and weights exist, run:

```bash
python scripts/run_robotwin_pi05_pi06_compare.py \
  --robotwin-root benchmarks/RoboTwin \
  --task-name place_can_basket \
  --task-config demo_clean \
  --train-config-name pi05_base_aloha_lora \
  --model-name <model_name> \
  --seed 0 \
  --gpu-id 0
```

The script checks for the expected pi05 checkpoint assets and model weights
before launching RoboTwin. If they are missing, it exits before the simulator
run.

## Preparing A Public Checkpoint

If a public Hugging Face model already uses the OpenPI checkpoint layout, prepare
it for RoboTwin with:

```bash
python scripts/prepare_robotwin_pi05_checkpoint.py \
  --robotwin-root benchmarks/RoboTwin \
  --repo-id Crelf/C3I_pi05_Robotwin_50tasks_model_democlean \
  --model-name C3I_pi05_Robotwin_50tasks_model_democlean \
  --checkpoint-id auto \
  --dry-run
```

Remove `--dry-run` to download only inference files and symlink them into:

```text
benchmarks/RoboTwin/policy/pi05/checkpoints/<train_config>/<model>/<step>
```

The downloader skips optimizer and train-state files by default. It supports
both OpenPI JAX checkpoints (`params/` plus `assets/`) and single-file PyTorch
checkpoints (`model.safetensors` plus `assets/`).

If the Hugging Face transfer path is slow, retry with one of:

```bash
--hf-transfer
--disable-xet
--max-workers 1
```

These only change the download transport; they do not change the checkpoint
layout.

## Current Local Blockers

The local workspace currently has RoboTwin simulator assets and dependencies,
but it does not include:

- OpenPI/JAX runtime dependencies in `open-pi-mem-robotwin`
- pi05 checkpoints under `benchmarks/RoboTwin/policy/pi05/checkpoints`
- local RoboTwin demonstration data under `benchmarks/RoboTwin/data`

Therefore this workspace can verify wrapper import, command construction, and
simulator readiness, but cannot produce real pi05/pi06 success rates until a
trained pi05 checkpoint is available.
