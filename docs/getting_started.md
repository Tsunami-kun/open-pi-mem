# Getting Started

This page collects the command-line workflows that are useful once the main repository README has already oriented you.

## Installation

Base package:

```bash
python3 -m pip install -e .
```

Evaluation-related extras:

```bash
python3 -m pip install -e ".[eval]"
```

Dataset-export extras:

```bash
python3 -m pip install -e ".[data]"
```

Developer extras:

```bash
python3 -m pip install -e ".[dev]"
```

System tools required for video evaluation:

- `ffmpeg`
- `ffprobe`

## Run High-Level Inference With A Local VLM

```bash
python3 scripts/run_high_level_inference.py \
  --config configs/high_level_vlm.yaml \
  --model-path /absolute/path/to/Qwen3-VL-8B-Instruct \
  --local-files-only \
  --image examples/frames/frame_00.png \
  --goal "pick up the target object" \
  --prev-memory "" \
  --planner-hz 1
```

If you already have a fine-tuned high-level checkpoint:

```bash
python3 scripts/run_high_level_inference.py \
  --config configs/high_level_vlm.yaml \
  --checkpoint /path/to/high_level_checkpoint.pt \
  --image examples/frames/frame_00.png \
  --goal "pick up the target object"
```

Example local config:

- [`configs/high_level_qwen3_vl_8b_local.example.yaml`](../configs/high_level_qwen3_vl_8b_local.example.yaml)

## Run Gemini-Based RMBench Video Evaluation

Set your API key first:

```bash
export GEMINI_API_KEY=your_key
```

Then run:

```bash
python3 scripts/run_rmbench_high_level_episode_gemini_sdk_video.py \
  --video data/rmbench_local/observe_and_pickup_demo_clean/episode0.mp4 \
  --instruction-json data/rmbench_local/observe_and_pickup_demo_clean/episode0.json \
  --hz 1 \
  --model gemini-3.1-pro-preview \
  --max-output-tokens 10000 \
  --update-memory \
  --report-dir data/eval_results/demo_observe_and_pickup
```

To switch models:

```bash
--model gemini-robotics-er-1.5-preview
```

## Launch The Viewer Locally

```bash
python3 scripts/run_test_viewer_app.py --host 127.0.0.1 --port 8766 --rebuild
```

Open the viewer:

```text
http://127.0.0.1:8766/
```

Deep-link to a specific bundled report:

```text
http://127.0.0.1:8766/?report=reports/Gemini_3_1_Pro/press_button/report.json
```

The viewer shows:

- input frames and clip boundaries
- goal and previous memory
- predicted next subtask and next memory
- full prompt text
- raw model output

## Build The Static GitHub Pages Site

```bash
python3 scripts/build_github_pages_site.py --output dist/github_pages
```

This stages:

- the viewer app at the site root
- a curated subset of bundled demo reports under `reports/`
- a generated `reports/index.json` manifest for the demo picker

If you explicitly want the full archive in the static bundle:

```bash
python3 scripts/build_github_pages_site.py --output dist/github_pages --include-all-reports
```

GitHub Pages deployment workflow:

- [`.github/workflows/pages.yml`](../.github/workflows/pages.yml)

## Main Entry Points

- [`scripts/run_high_level_inference.py`](../scripts/run_high_level_inference.py): single-image high-level inference
- [`scripts/run_rmbench_high_level_episode_gemini_sdk_video.py`](../scripts/run_rmbench_high_level_episode_gemini_sdk_video.py): video-based high-level evaluation with Gemini
- [`scripts/run_test_viewer_app.py`](../scripts/run_test_viewer_app.py): local static viewer server
- [`scripts/build_github_pages_site.py`](../scripts/build_github_pages_site.py): static site builder for GitHub Pages
- [`scripts/generate_memory_data.py`](../scripts/generate_memory_data.py): build memory supervision from episode annotations
- [`scripts/generate_manual_high_level_data.py`](../scripts/generate_manual_high_level_data.py): build high-level data from manual annotations
- [`scripts/prepare_open_dataset.py`](../scripts/prepare_open_dataset.py): export RLDS/TFDS data to low-level JSONL windows
- [`scripts/train_high_level.py`](../scripts/train_high_level.py): high-level trainer
- [`scripts/train_low_level.py`](../scripts/train_low_level.py): low-level trainer
- [`scripts/run_rmbench_eval.py`](../scripts/run_rmbench_eval.py): low-level evaluation scaffold

Older experiment variants are preserved in [`scripts/archive/`](../scripts/archive).
