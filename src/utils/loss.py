"""Loss/label helpers shared across training stages."""
from __future__ import annotations

import torch


def make_next_token_labels(
    input_ids: torch.Tensor,
    pad_id: int,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Right-shift inputs for next-token prediction with PAD masking."""

    if input_ids.dim() != 2:
        raise ValueError(f"Expected 2D input_ids, got {input_ids.shape}")
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = ignore_index
    if pad_id >= 0:
        labels = labels.masked_fill(input_ids == pad_id, ignore_index)
    return labels
