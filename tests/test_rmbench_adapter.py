from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from open_pi_mem.rmbench.adapter import RMBenchAdapter


class _Tokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"

    def __call__(self, *args, **kwargs):
        return {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }


class _ImageProcessor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.ones((len(images), 3, 8, 8), dtype=torch.float32)
        }


class _Policy:
    def __call__(self, *, input_ids, attention_mask, video, proprio):
        assert video.shape == (1, 2, 3, 8, 8)
        assert proprio.shape == (1, 2, 3)
        return {
            "action_chunk": torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32),
            "pooled_hidden": torch.zeros((1, 5), dtype=torch.float32),
        }


def test_rmbench_adapter_batches_video_frames_for_low_level_policy() -> None:
    adapter = RMBenchAdapter.__new__(RMBenchAdapter)
    adapter.device = "cpu"
    adapter.action_scale = 1.0
    adapter.action_clip = (-1.0, 1.0)
    adapter.memory_enabled = False
    adapter.policy = _Policy()
    adapter.tokenizer = _Tokenizer()
    adapter.image_processor = _ImageProcessor()

    frames = [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(2)]
    result = adapter.predict(
        frames=frames,
        proprio_state=[0.0, 0.1, 0.2],
        goal="test goal",
    )

    np.testing.assert_allclose(result["action_chunk"], [[0.1, 0.2, 0.3]])
