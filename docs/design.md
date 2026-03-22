# Design Notes

## Primary References

This repository is primarily informed by:

- Torne et al., [MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://www.pi.website/download/Mem.pdf)
- Physical Intelligence, [VLAs with Long and Short-Term Memory](https://www.pi.website/research/memory)

This scaffold makes explicit choices where the MEM paper is underspecified.

## Parameter Sharing

Default: `share_backbone_between_policies = false`.

Reason: Figure 2 looks more like separate high-level and low-level VLM blocks than a single shared backbone with two heads. The repo keeps that as the default, but the configuration surface remains open for experiments.

## Initialization Path

Default assumption:

1. initialize text and vision towers from a pretrained VLM stack
2. add MEM temporal video encoding
3. train high-level and low-level policies separately first
4. add joint finetuning later if needed

The repo therefore exposes:

- HF model names or local paths for the text and vision towers
- optional text-only / vision-only checkpoints
- optional combined VLM checkpoint loading with prefix filtering

## High-Level Supervision

Default implementation trains the high-level policy as a causal LM on structured targets rather than as two pooled-token classification heads.

Reason:

- it matches the paper's language-generation framing better
- it is easier to keep aligned with Gemma checkpoints
- it avoids inventing head designs the paper does not specify

## Episode Annotation

Open data is messy. The preprocessing path therefore supports:

- explicit subtask events when they exist
- metadata-based segmentation when subtasks are already present in episode annotations
- per-step language change heuristics when only instruction streams exist
- success/failure annotation from indices, rewards, or terminal flags

This is not PI's exact labeling pipeline. It is the minimum viable route for producing high-level supervision at scale.

## Low-Level Data Adapter

Default minimal experiment target: DROID or Bridge V2 exported locally in RLDS / TFDS format.

Reason:

- these datasets are realistic open replacements for PI's private robot data
- RLDS gives a tractable route to windowing recent frames, proprio, and action chunks
- the adaptation step can be made explicit and cached as JSONL instead of hiding dataset-specific assumptions inside the trainer

## Loss Composition

The paper does not publish exact loss weights. This repo exposes:

- action chunk regression
- optional FAST token cross entropy
- optional flow-style regression loss

The action expert path is detached from the fused VLM representation by default to mirror the paper's statement that gradients from the action expert do not flow into the VLM backbone.
