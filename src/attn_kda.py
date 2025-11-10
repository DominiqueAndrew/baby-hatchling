"""Kimi Delta Attention block (linear attention with constant state)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .modules import HeadwiseRMSNorm, LIFSpike, RMSNorm, SwiGLU, depthwise_short_conv


@dataclass
class KDAState:
    """Container for the fast weight and spiking membrane state."""

    tensor: torch.Tensor  # [B,H,dk,dv]
    spike_mem: torch.Tensor  # [B,H,dk]

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
        spike_mem = torch.zeros(batch, heads, dk, device=device, dtype=dtype)
        return cls(tensor=tensor, spike_mem=spike_mem)


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
        spike_decay: float = 0.9,
        spike_threshold: float = 0.5,
        spike_surrogate_beta: float = 10.0,
        token_drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv
        self.token_drop_prob = token_drop_prob
        self.last_drop_fraction = 0.0

        self.norm_in = RMSNorm(d_model)
        self.short_conv = depthwise_short_conv(d_model)
        self.act = nn.SiLU()

        self.q_proj = nn.Linear(d_model, num_heads * dk, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * dk, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * dv, bias=False)
        self.out_proj = nn.Linear(num_heads * dv, d_model, bias=False)

        self.spike_proj = nn.Linear(d_model, num_heads * dk, bias=False)
        self.lif = LIFSpike(decay=spike_decay, threshold=spike_threshold, surrogate_beta=spike_surrogate_beta)

        self.alpha_up = nn.Linear(d_model, rank_gate)
        self.alpha_down = nn.Linear(rank_gate, num_heads * dk)
        self.alpha_spike = nn.Parameter(torch.zeros(1, num_heads, dk))
        self.beta_proj = nn.Linear(d_model, num_heads)
        self.beta_spike = nn.Parameter(torch.zeros(1, num_heads, dk))

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
        spike_mem = getattr(state, "spike_mem", None)
        if spike_mem is None:
            spike_mem = torch.zeros(bsz, self.num_heads, self.dk, device=device, dtype=dtype)

        h = self.norm_in(x)
        conv = self.short_conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(conv)

        q = self._reshape_heads(self.q_proj(h), self.dk)  # [B,T,H,dk]
        k = self._reshape_heads(self.k_proj(h), self.dk)
        v = self._reshape_heads(self.v_proj(h), self.dv)

        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        spike_drive = self._reshape_heads(self.spike_proj(h), self.dk)
        alpha_base = self._reshape_heads(self.alpha_down(torch.nn.functional.silu(self.alpha_up(h))), self.dk)
        beta_base = self.beta_proj(h)  # [B,T,H]

        outputs = []
        drop_mask = None
        if self.training and self.token_drop_prob > 0.0:
            drop_mask = torch.rand(bsz, seq, device=device) < self.token_drop_prob
            guard = min(2, seq)
            drop_mask[:, :guard] = False
            drop_mask[:, -guard:] = False
            self.last_drop_fraction = float(drop_mask.float().mean().item())
        else:
            self.last_drop_fraction = 0.0
        prev_out = None
        for t in range(seq):
            spike_mem, spike = self.lif(spike_mem, spike_drive[:, t])
            alpha = torch.sigmoid(alpha_base[:, t] + self.alpha_spike * spike)
            beta = beta_base[:, t].unsqueeze(-1)
            beta = torch.sigmoid(beta + torch.sum(spike * self.beta_spike, dim=-1, keepdim=True))
            active = (spike.abs().sum(dim=-1, keepdim=True) > 0).to(x.dtype)
            if drop_mask is not None:
                token_keep = (~drop_mask[:, t]).to(x.dtype).unsqueeze(-1).unsqueeze(-1)
                active = active * token_keep
            s, out = self._step(s, q[:, t], k[:, t], v[:, t], alpha, beta, active)
            if drop_mask is not None:
                if prev_out is None:
                    prev_out = torch.zeros_like(out)
                mask_t = drop_mask[:, t].to(out.dtype).unsqueeze(-1).unsqueeze(-1)
                out = out * (1 - mask_t) + prev_out * mask_t
                prev_out = out.detach()
            else:
                prev_out = out.detach()
            outputs.append(out)
        y = torch.stack(outputs, dim=1)  # [B,T,H,dv]
        y = self.head_norm(y).reshape(bsz, seq, -1)

        gate = torch.sigmoid(self.low_rank_u2(torch.nn.functional.silu(self.low_rank_u1(x))))
        y = x + self.out_proj(y) * gate
        y = y + self.ff(self.ff_norm(y))

        return y, KDAState(tensor=s, spike_mem=spike_mem)

    def _step(
        self,
        s: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        active: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a single recurrent step (vectorized over batch+heads)."""

        if active is not None:
            alpha = torch.where(active.expand_as(alpha) > 0, alpha, torch.ones_like(alpha))
            beta = beta * active
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

    def stream(self, x_t: torch.Tensor, state: Optional[KDAState] = None) -> Tuple[torch.Tensor, KDAState]:
        """Processes a single timestep (useful for autoregressive decoding)."""

        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        y, new_state = self.forward(x_t, state)
        return y[:, 0], new_state
