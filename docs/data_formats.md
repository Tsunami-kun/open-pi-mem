# Data Formats And Results

This page collects the data structures and saved result artifacts used throughout `open-pi-mem`.

## Saved Results Included In The Repository

The repository intentionally keeps viewer-ready qualitative artifacts in-repo under [`data/eval_results/`](../data/eval_results), including:

- [`data/eval_results/Gemini_3_1_Pro`](../data/eval_results/Gemini_3_1_Pro)
- [`data/eval_results/Gemini_Robotics_ER_1_5`](../data/eval_results/Gemini_Robotics_ER_1_5)

These reports are useful for qualitative comparison:

- how memory is updated over time
- whether subtasks stay atomic
- where models advance too early or remain conservative
- how different models behave on the same episode

Unlike raw local source videos, these report artifacts are intentionally kept in the repository because the local HTML viewer depends on them for inspection and sharing.

## High-Level Memory Supervision JSONL

Each row looks like:

```json
{
  "episode_id": "ep_stageA_001",
  "goal": "put the red mug into the upper cabinet",
  "observation_ref": "examples/frames/frame_00.png",
  "prev_memory": "",
  "next_subtask": "continue current task",
  "next_memory": "No stable memory yet.",
  "history": [
    {
      "text": "scan counter for red mug",
      "status": "unknown",
      "start_index": 0,
      "end_index": 0
    }
  ]
}
```

Sample file:

- [`examples/memory_supervision.sample.jsonl`](../examples/memory_supervision.sample.jsonl)

## Low-Level Rollout JSONL

Each row contains:

- `goal`
- `subtask`
- `frame_paths`
- `proprio`
- `action_chunk`
- optional `fast_tokens`

Sample file:

- [`examples/low_level_rollouts.sample.jsonl`](../examples/low_level_rollouts.sample.jsonl)

## RMBench Report JSON

Saved report files contain:

- run metadata such as source video and planner frequency
- a `records` array with one planning step per record
- prompt text, raw output, predicted subtask, and updated memory
- extracted frame paths for viewer playback
- clip timing fields such as `input_clip_start_sec` and `input_clip_end_sec`

## Upstream Projects And Dependencies

This repository depends on or is inspired by:

- MEM and hierarchical policy ideas from Physical Intelligence
- RMBench-style memory-sensitive robot evaluation tasks
- open VLM backbones such as Qwen-VL, Gemma, and SigLIP
- open robot datasets distributed through RLDS and TFDS, especially DROID and Bridge V2
- Gemini APIs for report-oriented qualitative evaluation

Primary reference for the hierarchical memory-style setup in this repository:

- Torne et al., [MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://www.pi.website/download/Mem.pdf)
- Physical Intelligence, [VLAs with Long and Short-Term Memory](https://www.pi.website/research/memory)

## Known Limitations

- The repository currently emphasizes qualitative inspection more than headline benchmark numbers.
- The low-level evaluation path is still a scaffold and does not yet provide a finished end-to-end benchmark story.
- No pretrained checkpoints are released yet.
- Saved qualitative reports are stronger than the current quantitative evaluation story.
- Some viewer-facing artifacts are intentionally tracked, so the repository includes non-source assets by design.

## Roadmap

- complete the high-level adapter path inside low-level RMBench evaluation
- add benchmark summary tables and reproducible experiment summaries
- expand automated tests beyond import and CLI smoke checks
- publish checkpoints once the training recipe stabilizes
- keep refining which generated artifacts belong in-repo versus release assets
