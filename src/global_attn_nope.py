"""Global NoPE attention layer used every 4th block."""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .modules import HeadwiseRMSNorm, RMSNorm, SwiGLU


class GlobalNoPEBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dk: int,
        dv: int,
        d_ff: int,
        group_kv: int = 1,
        gw_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv
        self.group_kv = max(1, group_kv)
        self.heads_per_group = num_heads // self.group_kv

        self.norm_in = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, num_heads * dk, bias=False)
        self.k_proj = nn.Linear(d_model, self.group_kv * dk, bias=False)
        self.v_proj = nn.Linear(d_model, self.group_kv * dv, bias=False)
        self.out_proj = nn.Linear(num_heads * dv, d_model, bias=False)

        self.head_norm = HeadwiseRMSNorm(num_heads, dv)
        self.ff_norm = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

        if gw_tokens > 0:
            self.gw_tokens = nn.Parameter(torch.zeros(gw_tokens, d_model))
        else:
            self.register_parameter("gw_tokens", None)
        
        # Cache for causal mask to avoid recreating every forward pass
        self.register_buffer("_causal_mask_cache", None, persistent=False)
        self._cached_mask_size = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        h = self.norm_in(x)
        if self.gw_tokens is not None:
            gw = self.gw_tokens.unsqueeze(0).expand(bsz, -1, -1)
            h = torch.cat([gw, h], dim=1)
        t = h.shape[1]

        q = self.q_proj(h).view(bsz, t, self.num_heads, self.dk)
        k = self._expand_groups(self.k_proj(h), self.dk)
        v = self._expand_groups(self.v_proj(h), self.dv)

        context = self._attention(q, k, v)
        context = self.head_norm(context).reshape(bsz, t, -1)
        y = x
        if self.gw_tokens is not None:
            context = context[:, self.gw_tokens.shape[0] :, :]
        y = y + self.out_proj(context)
        y = y + self.ff(self.ff_norm(y))
        return y

    def _expand_groups(self, proj: torch.Tensor, width: int) -> torch.Tensor:
        bsz, seq, _ = proj.shape
        kv = proj.view(bsz, seq, self.group_kv, width)
        repeat = self.heads_per_group
        kv = kv.unsqueeze(3).repeat(1, 1, 1, repeat, 1)
        kv = kv.view(bsz, seq, self.num_heads, width)
        return kv

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bsz, t, heads, _ = q.shape
        q_flat = q.permute(0, 2, 1, 3).reshape(bsz * heads, t, self.dk)
        k_flat = k.permute(0, 2, 1, 3).reshape(bsz * heads, t, self.dk)
        v_flat = v.permute(0, 2, 1, 3).reshape(bsz * heads, t, self.dv)

        scores = torch.bmm(q_flat, k_flat.transpose(1, 2)) / math.sqrt(self.dk)
        
        # Use cached causal mask to avoid recreating every forward pass
        if self._causal_mask_cache is None or self._cached_mask_size < t:
            self._causal_mask_cache = torch.tril(torch.ones(t, t, device=q.device, dtype=torch.bool))
            self._cached_mask_size = t
        causal_mask = self._causal_mask_cache[:t, :t]
        
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(weights, v_flat)
        ctx = ctx.view(bsz, heads, t, self.dv).permute(0, 2, 1, 3)
        return ctx
