"""Predictive coding auxiliaries: LM head + next-state predictor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .modules import RMSNorm


@dataclass
class PredictiveOutput:
    logits: torch.Tensor
    pc_loss: torch.Tensor
    error_trace: torch.Tensor


class PredictiveCodingHead(nn.Module):
    def __init__(
        self, d_model: int, vocab_size: int, embedding: nn.Embedding, enabled: bool = True
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.decoder.weight = embedding.weight
        self.state_pred = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.register_buffer("ema_error", torch.tensor(1.0), persistent=False)
        self.enabled = enabled

    def forward(self, hidden: torch.Tensor) -> PredictiveOutput:
        """Returns logits and predictive coding loss."""

        logits = self.decoder(self.norm(hidden))
        if not self.enabled or hidden.size(1) < 2:
            zeros = hidden.new_zeros(hidden.size(0), max(hidden.size(1) - 1, 1))
            return PredictiveOutput(logits=logits, pc_loss=zeros.mean(), error_trace=zeros)

        pred = self.state_pred(hidden[:, :-1])
        target = hidden[:, 1:].detach()
        pc_loss = torch.mean((pred - target) ** 2)
        error = torch.mean((pred - target) ** 2, dim=-1)  # per-token scalar
        return PredictiveOutput(logits=logits, pc_loss=pc_loss, error_trace=error)

    def curiosity_bonus(self, error: torch.Tensor, clip: float = 1.0, alpha: float = 0.1) -> float:
        """Computes an intrinsic reward proportional to error improvement."""

        if not self.enabled:
            return 0.0
        error_mean = error.mean().detach()
        ema = (1 - alpha) * self.ema_error + alpha * error_mean
        improvement = torch.clamp(self.ema_error - error_mean, min=0.0)
        bonus = float(torch.clamp(improvement / (ema + 1e-6), max=clip).item())
        self.ema_error.copy_(ema)
        return bonus
