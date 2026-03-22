# Contributing to open-pi-mem

Thanks for contributing. This repository is a research codebase, so clarity, reproducibility, and artifact hygiene matter as much as raw feature count.

## Scope

Good contributions include:

- improving high-level memory-planning experiments
- making the low-level scaffold more complete or better documented
- tightening data adapters and report generation
- improving viewer usability and result inspection
- adding tests, CI coverage, and reproducibility tooling

## Development Setup

Install the package in editable mode:

```bash
python3 -m pip install -e .
```

For evaluation-related scripts:

```bash
python3 -m pip install -e ".[eval]"
```

For dataset export utilities:

```bash
python3 -m pip install -e ".[data]"
```

For local development helpers:

```bash
python3 -m pip install -e ".[dev]"
```

## Before Opening a PR

Please run the lightweight checks that match your change:

```bash
python3 -m compileall src scripts
python3 -m open_pi_mem.cli --version
python3 scripts/run_high_level_inference.py --help
python3 scripts/run_rmbench_eval.py --help
python3 scripts/run_rmbench_high_level_episode_gemini_sdk_video.py --help
python3 scripts/run_test_viewer_app.py --help
python3 scripts/prepare_open_dataset.py --help
```

If your change touches training or evaluation logic, include:

- what you changed
- how you validated it
- whether any saved report artifacts changed intentionally

## Artifact Policy

This repository intentionally tracks some generated artifacts when they are useful for public inspection:

- `data/eval_results/` viewer-ready reports
- extracted preview frames that are needed by the local HTML viewer

This repository should not include:

- private raw datasets
- large temporary caches
- machine-specific runtime leftovers
- credentials or API keys

Raw local RMBench-style source data under `data/rmbench_local/` should remain untracked.

## Pull Request Style

- Keep PRs focused.
- Prefer small, reviewable commits.
- Document assumptions when reproducing paper details that are underspecified.
- Update the README or docs when changing public-facing workflows.
- Avoid reverting unrelated user work in a dirty tree.

## Reporting Issues

When filing a bug, please include:

- the command you ran
- the config and model path you used
- the platform and Python version
- the stack trace or failing output
- whether the failure involves local data, public sample data, or saved report artifacts
