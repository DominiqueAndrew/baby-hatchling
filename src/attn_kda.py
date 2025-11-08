"""Kimi Delta Attention block (linear attention with constant state)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .modules import HeadwiseRMSNorm, RMSNorm, SwiGLU, depthwise_short_conv


@dataclass
class KDAState:
    """Container for the per-head fast state S (dk x dv)."""

    tensor: torch.Tensor  # [B,H,dk,dv]

    @classmethod
    def zeros(
        cls,
        batch: int,
        heads: int,
        dk: int,
        dv: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KDAState":
        tensor = torch.zeros(batch, heads, dk, dv, device=device, dtype=dtype)
        return cls(tensor=tensor)


class KDABlock(nn.Module):
    """KDA block following the Baby-Hatchling specification."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dk: int,
        dv: int,
        d_ff: int,
        rank_gate: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv

        self.norm_in = RMSNorm(d_model)
        self.short_conv = depthwise_short_conv(d_model)
        self.act = nn.SiLU()

        self.q_proj = nn.Linear(d_model, num_heads * dk, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * dk, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * dv, bias=False)
        self.out_proj = nn.Linear(num_heads * dv, d_model, bias=False)

        self.alpha_proj = nn.Linear(d_model, num_heads * dk)
        self.beta_proj = nn.Linear(d_model, num_heads)

        self.head_norm = HeadwiseRMSNorm(num_heads, dv)
        self.low_rank_u1 = nn.Linear(d_model, rank_gate)
        self.low_rank_u2 = nn.Linear(rank_gate, d_model)
        self.ff_norm = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[KDAState] = None,
    ) -> Tuple[torch.Tensor, KDAState]:
        """Runs the block over a full sequence.

        Parameters
        ----------
        x: Tensor shaped [B,T,D].
        state: Optional previous fast state of shape [B,H,dk,dv]. If omitted a
            zero state is used (useful for teacher forcing during training).
        """

        bsz, seq, _ = x.shape
        device, dtype = x.device, x.dtype
        if state is None:
            state = KDAState.zeros(bsz, self.num_heads, self.dk, self.dv, device=device, dtype=dtype)
        s = state.tensor

        h = self.norm_in(x)
        conv = self.short_conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(conv)

        q = self._reshape_heads(self.q_proj(h), self.dk)  # [B,T,H,dk]
        k = self._reshape_heads(self.k_proj(h), self.dk)
        v = self._reshape_heads(self.v_proj(h), self.dv)

        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        alpha = torch.sigmoid(self._reshape_heads(self.alpha_proj(h), self.dk))
        beta = torch.sigmoid(self.beta_proj(h)).unsqueeze(-1)  # [B,T,H,1]

        outputs = []
        for t in range(seq):
            s, out = self._step(s, q[:, t], k[:, t], v[:, t], alpha[:, t], beta[:, t])
            outputs.append(out)
        y = torch.stack(outputs, dim=1)  # [B,T,H,dv]
        y = self.head_norm(y).reshape(bsz, seq, -1)

        gate = torch.sigmoid(self.low_rank_u2(torch.nn.functional.silu(self.low_rank_u1(x))))
        y = x + self.out_proj(y) * gate
        y = y + self.ff(self.ff_norm(y))

        return y, KDAState(tensor=s)

    def _step(
        self,
        s: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a single recurrent step (vectorized over batch+heads)."""

        # Apply channel-wise decay
        s = s * alpha.unsqueeze(-1)
        # Rank-1 forget via k^T S term
        u = torch.einsum("bhd,bhdv->bhv", k, s)
        forget = torch.einsum("bhd,bhv->bhdv", k, u)
        s = s - beta.unsqueeze(-1) * forget
        # Delta rule write
        write = torch.einsum("bhd,bhv->bhdv", k, v)
        s = s + beta.unsqueeze(-1) * write
        # Output
        out = torch.einsum("bhdv,bhd->bhv", s, q)
        return s, out

    def _reshape_heads(self, proj: torch.Tensor, width: int) -> torch.Tensor:
        bsz, seq, _ = proj.shape
        return proj.view(bsz, seq, self.num_heads, width)
