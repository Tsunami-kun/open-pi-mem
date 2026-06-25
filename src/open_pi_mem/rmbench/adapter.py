"""RMBench integration adapter for open-pi-mem low-level policy.

This module provides utilities to run open-pi-mem's low-level policy
within the RMBench framework, enabling evaluation on memory-dependent
robotic manipulation tasks.

Data Flow:
  RMBench Simulator
    ↓ (frames, proprioception, goal)
  RMBenchAdapter
    ↓ (tokenize instruction, prepare video)
  LowLevelPolicy.forward()
    ↓ (action predictions)
  apply_action() → Simulator
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from open_pi_mem.models.low_level_policy import LowLevelPolicy


class RMBenchAdapter:
    """Wraps open-pi-mem's low-level policy for RMBench evaluation.

    Handles:
    1. Data format conversion (RMBench → open-pi-mem)
    2. Tokenization of goals/subtasks
    3. Video frame processing (SigLIP-compatible)
    4. Action post-processing (scaling, clipping)
    5. Optional memory integration
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        action_scale: float = 1.0,
        action_clip: tuple[float, float] = (-1.0, 1.0),
        memory_enabled: bool = False,
    ) -> None:
        """Initialize RMBench adapter.

        Args:
            model_path: Path to trained low-level policy checkpoint
            device: Device for inference (cuda/cpu)
            action_scale: Scale factor for predicted actions
            action_clip: (min, max) clipping range for actions
            memory_enabled: Whether to use high-level memory for task context
        """
        self.device = device
        self.action_scale = action_scale
        self.action_clip = action_clip
        self.memory_enabled = memory_enabled
        self.history: list[dict[str, Any]] = []

        # Load model config and checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model_cfg = checkpoint.get("config", {})

        # Initialize policy
        self.policy = LowLevelPolicy(model_cfg).to(device).eval()
        self.policy.load_state_dict(checkpoint["model_state_dict"])

        # Initialize tokenizer (Gemma-2-2b)
        tokenizer_name = model_cfg.get("backbone_name", "google/gemma-2-2b-it")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize image processor (SigLIP)
        vision_model_name = model_cfg.get(
            "vision_tower_name", "google/siglip-base-patch16-224"
        )
        self.image_processor = AutoProcessor.from_pretrained(vision_model_name)

    @torch.no_grad()
    def predict(
        self,
        frames: list[Image.Image] | list[str],
        proprio_state: list[float],
        goal: str,
        current_subtask: str | None = None,
        memory_context: str | None = None,
    ) -> dict[str, Any]:
        """Predict action chunk given observation and task context.

        Args:
            frames: List of PIL images or frame paths
            proprio_state: Current proprioceptive state [joint_angles]
            goal: High-level goal text
            current_subtask: Optional subtask from high-level policy
            memory_context: Optional memory from previous steps (M(n) tasks)

        Returns:
            dict with:
                action_chunk: (action_horizon, action_dim) predicted actions
                logits: (hidden_size,) for debugging
                confidence: float [0, 1] - prediction confidence
                memory_update: Optional memory text for M(n) tasks
        """
        # Load and process frames
        if isinstance(frames[0], str):
            frames = [Image.open(f).convert("RGB") for f in frames]

        # Tokenize instruction
        subtask_text = current_subtask or goal
        if memory_context:
            instruction = f"Goal: {goal}\nSubtask: {subtask_text}\nMemory: {memory_context}"
        else:
            instruction = f"Goal: {goal}\nSubtask: {subtask_text}"

        input_ids = self.tokenizer(
            instruction,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        # Process video frames
        image_inputs = self.image_processor(images=frames, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].unsqueeze(0)  # (1, T, C, H, W)

        # Prepare proprioception (current state, broadcast to sequence length)
        T = pixel_values.shape[1]
        proprio = torch.tensor(
            [proprio_state] * T, dtype=torch.float32
        ).unsqueeze(0)  # (1, T, action_dim)

        # Move to device
        input_ids = {k: v.to(self.device) for k, v in input_ids.items()}
        pixel_values = pixel_values.to(self.device)
        proprio = proprio.to(self.device)

        # Forward pass
        outputs = self.policy(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            video=pixel_values,
            proprio=proprio,
        )

        # Extract predictions
        action_chunk = outputs["action_chunk"].squeeze(0)  # (horizon, action_dim)
        pooled = outputs["pooled_hidden"].squeeze(0)  # (hidden_size,)

        # Post-process actions
        action_chunk = action_chunk * self.action_scale
        action_chunk = torch.clamp(action_chunk, *self.action_clip)

        # Compute confidence via fast logits if available
        confidence = 1.0
        if "fast_logits" in outputs:
            fast_logits = outputs["fast_logits"].squeeze(0)  # (vocab_size,)
            confidence = float(F.softmax(fast_logits, dim=0).max().item())

        result = {
            "action_chunk": action_chunk.cpu().numpy(),
            "pooled_hidden": pooled.cpu().numpy(),
            "confidence": confidence,
        }

        # Optional: Generate memory update for M(n) tasks
        if self.memory_enabled:
            memory_update = self._generate_memory_update(
                instruction, action_chunk, confidence
            )
            result["memory_update"] = memory_update

        return result

    def _generate_memory_update(
        self, instruction: str, actions: torch.Tensor, confidence: float
    ) -> str:
        """Generate memory update text for M(n) multi-step tasks.

        Simple rule-based approach; can be replaced with VLM call.
        """
        if confidence < 0.5:
            return "ACTION UNCERTAIN: need additional observation"
        action_magnitude = float(actions.abs().mean().item())
        if action_magnitude > 0.8:
            return f"EXECUTED: large movement detected, confidence={confidence:.2f}"
        return f"EXECUTED: small adjustment, confidence={confidence:.2f}"

    def reset(self) -> None:
        """Reset memory history for new episode."""
        self.history = []

    def add_history(self, subtask: str, success: bool, memory: str = "") -> None:
        """Record subtask execution result in history."""
        self.history.append(
            {"subtask": subtask, "success": success, "memory": memory}
        )

    def get_history_context(self) -> str:
        """Format history as prompt context for next subtask."""
        if not self.history:
            return ""
        lines = ["Previous actions:"]
        for item in self.history[-3:]:  # Last 3 steps
            status = "success" if item["success"] else "failed"
            lines.append(f"  - {item['subtask']} [{status}]")
            if item["memory"]:
                lines.append(f"    Memory: {item['memory']}")
        return "\n".join(lines)
