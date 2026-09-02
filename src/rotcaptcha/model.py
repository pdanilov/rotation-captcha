"""The shared, task-agnostic model: a pretrained `transformers` image backbone
with a fresh linear head of `output_dim` outputs.

Pretrained weights are mandatory. What the `output_dim` outputs *mean* (bin logits,
a (sin, cos) vector, ...) is decided by the `AngleHead` in heads.py — the backbone
does not care. The forward pass returns a `.logits` tensor of shape [B, output_dim].
"""

from __future__ import annotations

import torch
from transformers import AutoModelForImageClassification


def build_model(model_name: str, output_dim: int) -> torch.nn.Module:
    """Pretrained backbone from the transformers hub + a fresh `output_dim` head."""
    return AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=output_dim,
        ignore_mismatched_sizes=True,  # replace the pretrained classifier head
    )
