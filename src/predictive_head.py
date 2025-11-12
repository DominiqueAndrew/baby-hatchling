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
        self.enabled = enabled
        self.register_buffer("ema_bonus_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("ema_bonus_var", torch.tensor(1.0), persistent=False)
        self._prev_error: float | None = None

    def forward(self, hidden: torch.Tensor) -> PredictiveOutput:
        """Returns logits and predictive coding loss."""

        logits = self.decoder(self.norm(hidden))
        if not self.enabled or hidden.size(1) < 2:
            zeros = hidden.new_zeros(hidden.size(0), max(hidden.size(1) - 1, 1))
            padded = torch.nn.functional.pad(zeros, (0, 1), value=0.0)
            return PredictiveOutput(logits=logits, pc_loss=zeros.mean(), error_trace=padded)

        pred = self.state_pred(hidden[:, :-1])
        target = hidden[:, 1:].detach()
        pc_loss = torch.mean((pred - target) ** 2)
        error = torch.mean((pred - target) ** 2, dim=-1)
        error = torch.nn.functional.pad(error, (0, 1), value=0.0)
        return PredictiveOutput(logits=logits, pc_loss=pc_loss, error_trace=error)

    def curiosity_bonus(self, error: torch.Tensor, clip: float = 0.05, alpha: float = 0.01) -> float:
        """EMA-normalized prediction progress bonus."""

        if not self.enabled:
            return 0.0
        scalar = float(error.mean().detach().cpu())
        if not torch.isfinite(torch.tensor(scalar)):
            return 0.0
        prev = self._prev_error if self._prev_error is not None else scalar
        delta = max(-clip, min(clip, prev - scalar))
        self._prev_error = scalar
        mean = (1 - alpha) * float(self.ema_bonus_mean.item()) + alpha * delta
        diff = delta - mean
        var = (1 - alpha) * float(self.ema_bonus_var.item()) + alpha * diff * diff
        self.ema_bonus_mean.copy_(torch.tensor(mean, device=self.ema_bonus_mean.device))
        self.ema_bonus_var.copy_(torch.tensor(max(var, 1e-6), device=self.ema_bonus_var.device))
        std = (self.ema_bonus_var.sqrt() + 1e-6).item()
        return float(diff / std)
