from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root mean square normalization with optional learned epsilon."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * x * norm


class HeadwiseRMSNorm(nn.Module):
    """RMSNorm applied per head for tensors shaped [B,T,H,D]."""

    def __init__(self, num_heads: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, num_heads, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * x * norm


class SwiGLU(nn.Module):
    """SwiGLU feed-forward (Shazeer, 2020)."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = torch.nn.functional.silu(self.w1(x)) * self.w2(x)
        return self.w3(self.dropout(gated))


def depthwise_short_conv(dim: int, kernel_size: int = 3) -> nn.Conv1d:
    """Creates a depthwise Conv1d used before the attention projections."""

    padding = kernel_size // 2
    conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
    nn.init.zeros_(conv.bias)
    return conv
