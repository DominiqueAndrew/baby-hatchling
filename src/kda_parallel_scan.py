"""Parallel scan helpers for Spike-KDA fast weights."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class KDAParallelUpdates:
    """Container for per-token transition/write tensors used by the scan."""

    transitions: torch.Tensor  # [B,T,H,dk,dk]
    writes: torch.Tensor  # [B,T,H,dk,dv]
    drop_mask: Optional[torch.Tensor] = None  # [B,T]


def precompute_updates(
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    active: torch.Tensor,
    drop_mask: Optional[torch.Tensor] = None,
    memory_chunk_size: int = 64,
) -> KDAParallelUpdates:
    """Builds per-token transition and write matrices.

    Parameters
    ----------
    k, v: Projections with shape [B,T,H,dk/dv].
    alpha: Decay factors [B,T,H,dk].
    beta: Write factors [B,T,H,1].
    active: Binary indicator per head [B,T,H,1].
    drop_mask: Optional bool tensor [B,T] for token dropping.
    memory_chunk_size: Number of tokens to process at once (lower = less memory, slower).
    """

    alpha_eff = torch.where(active > 0, alpha, torch.ones_like(alpha))
    beta_eff = beta * active

    if drop_mask is not None:
        token_keep = (~drop_mask).to(alpha.dtype)
        token_keep_alpha = token_keep[:, :, None, None]
        token_keep_beta = token_keep[:, :, None, None]
        alpha_eff = torch.where(token_keep_alpha > 0, alpha_eff, torch.ones_like(alpha_eff))
        beta_eff = beta_eff * token_keep_beta

    # Memory-efficient computation: process in chunks to avoid OOM
    # Instead of materializing full [B,T,H,dk,dk] tensors at once
    bsz, seq, heads, dk = alpha_eff.shape
    dv = v.shape[-1]
    chunk_size = min(memory_chunk_size, seq)  # Process in configurable chunks
    transitions_list = []
    writes_list = []
    
    alpha_k = alpha_eff * k  # This is [B,T,H,dk], relatively small
    
    for start in range(0, seq, chunk_size):
        end = min(start + chunk_size, seq)
        alpha_chunk = alpha_eff[:, start:end]
        alpha_k_chunk = alpha_k[:, start:end]
        beta_chunk = beta_eff[:, start:end]
        k_chunk = k[:, start:end]
        v_chunk = v[:, start:end]
        
        # Compute writes for this chunk
        writes_chunk = torch.matmul(k_chunk.unsqueeze(-1), v_chunk.unsqueeze(-2))
        writes_chunk = beta_chunk.unsqueeze(-1) * writes_chunk
        writes_list.append(writes_chunk)
        
        # Compute transitions for this chunk
        decay_diag = torch.diag_embed(alpha_chunk)
        forget_rank1 = torch.matmul(k_chunk.unsqueeze(-1), alpha_k_chunk.unsqueeze(-2))
        forget_rank1 = beta_chunk.unsqueeze(-1) * forget_rank1
        transitions_chunk = decay_diag - forget_rank1
        transitions_list.append(transitions_chunk)
    
    transitions = torch.cat(transitions_list, dim=1)
    writes = torch.cat(writes_list, dim=1)

    return KDAParallelUpdates(transitions=transitions, writes=writes, drop_mask=drop_mask)


def scan_emit_outputs(
    updates: KDAParallelUpdates,
    initial_state: torch.Tensor,
    q: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs scan and emits outputs without materializing all states at once.

    Parameters
    ----------
    updates: Precomputed transition/write tensors.
    initial_state: Fast weight state before the sequence, [B,H,dk,dv].
    q: Query tensor [B,T,H,dk].
    chunk_size: Number of tokens to materialize at a time while emitting.
    """

    prefix_m, prefix_b = _blelloch_exclusive_scan(updates.transitions, updates.writes)
    bsz, seq, heads, dk, dv = updates.writes.shape

    if chunk_size <= 0:
        chunk_size = seq

    outputs = []
    final_state = None
    s0 = initial_state.unsqueeze(1)  # [B,1,H,dk,dv]

    for start in range(0, seq, chunk_size):
        end = min(start + chunk_size, seq)
        m_chunk = prefix_m[:, start:end]  # [B,C,H,dk,dk]
        b_chunk = prefix_b[:, start:end]
        prior = torch.matmul(m_chunk, s0) + b_chunk
        state_chunk = torch.matmul(updates.transitions[:, start:end], prior) + updates.writes[:, start:end]

        q_chunk = q[:, start:end].unsqueeze(-2)
        outputs.append((q_chunk * state_chunk).sum(dim=-2))
        final_state = state_chunk[:, -1]

    outputs_tensor = torch.cat(outputs, dim=1)
    return outputs_tensor, final_state


def _combine_transforms(
    left_m: torch.Tensor,
    left_b: torch.Tensor,
    right_m: torch.Tensor,
    right_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combines two (A,B) operators where right is applied after left."""

    combined_m = torch.matmul(right_m, left_m)
    combined_b = torch.matmul(right_m, left_b) + right_b
    return combined_m, combined_b


def _blelloch_exclusive_scan(
    transitions: torch.Tensor,
    writes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes exclusive prefix operators for all tokens."""

    bsz, seq, heads, dk, _ = transitions.shape
    dv = writes.shape[-1]
    if seq == 0:
        raise ValueError("Sequence length must be > 0 for Spike-KDA scan.")

    pow2 = 1 << (seq - 1).bit_length()
    pad_len = pow2 - seq
    device = transitions.device
    dtype = transitions.dtype

    # Avoid unnecessary clones - work in-place when possible
    if pad_len > 0:
        eye = torch.eye(dk, device=device, dtype=dtype).view(1, 1, 1, dk, dk)
        eye = eye.expand(bsz, pad_len, heads, dk, dk).contiguous()
        zero = torch.zeros(bsz, pad_len, heads, dk, dv, device=device, dtype=writes.dtype)
        arr_m = torch.cat([transitions, eye], dim=1)
        arr_b = torch.cat([writes, zero], dim=1)
    else:
        # No padding needed - clone only once
        arr_m = transitions.clone()
        arr_b = writes.clone()
    levels = int(math.log2(pow2)) if pow2 > 1 else 0

    for level in range(levels):
        step = 1 << (level + 1)
        half = step >> 1
        new_shape = (bsz, pow2 // step, step, heads, dk, dk)
        arr_m = arr_m.view(new_shape)
        arr_b = arr_b.view(bsz, pow2 // step, step, heads, dk, dv)

        left_m = arr_m[:, :, half - 1]
        right_m = arr_m[:, :, step - 1]
        left_b = arr_b[:, :, half - 1]
        right_b = arr_b[:, :, step - 1]
        comb_m, comb_b = _combine_transforms(left_m, left_b, right_m, right_b)
        arr_m[:, :, step - 1] = comb_m
        arr_b[:, :, step - 1] = comb_b

        arr_m = arr_m.view(bsz, pow2, heads, dk, dk)
        arr_b = arr_b.view(bsz, pow2, heads, dk, dv)

    identity = torch.eye(dk, device=device, dtype=dtype).view(1, 1, 1, dk, dk)
    zero_b = torch.zeros(1, 1, 1, dk, dv, device=device, dtype=writes.dtype)
    arr_m[:, -1] = identity
    arr_b[:, -1] = zero_b

    for level in reversed(range(levels)):
        step = 1 << (level + 1)
        half = step >> 1
        arr_m = arr_m.view(bsz, pow2 // step, step, heads, dk, dk)
        arr_b = arr_b.view(bsz, pow2 // step, step, heads, dk, dv)

        left_m = arr_m[:, :, half - 1].clone()
        right_m = arr_m[:, :, step - 1].clone()
        left_b = arr_b[:, :, half - 1].clone()
        right_b = arr_b[:, :, step - 1].clone()

        arr_m[:, :, half - 1] = right_m
        arr_b[:, :, half - 1] = right_b
        comb_m, comb_b = _combine_transforms(right_m, right_b, left_m, left_b)
        arr_m[:, :, step - 1] = comb_m
        arr_b[:, :, step - 1] = comb_b

        arr_m = arr_m.view(bsz, pow2, heads, dk, dk)
        arr_b = arr_b.view(bsz, pow2, heads, dk, dv)

    return arr_m[:, :seq], arr_b[:, :seq]
