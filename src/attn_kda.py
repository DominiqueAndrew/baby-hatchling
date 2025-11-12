"""Kimi Delta Attention block (linear attention with constant state)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .modules import HeadwiseRMSNorm, LIFSpike, RMSNorm, SwiGLU, depthwise_short_conv
from .kda_parallel_scan import KDAParallelUpdates, precompute_updates, scan_emit_outputs


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
    """KDA block following the Baby-Hatchling specification with chunk-wise parallel processing."""

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
        chunk_size: int = 16,
        kda_mode: str = "sequential",
        scan_min_len: int = 64,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv
        self.token_drop_prob = token_drop_prob
        self.last_drop_fraction = 0.0
        self.chunk_size = chunk_size
        self.kda_mode = kda_mode.lower()
        self.scan_min_len = max(1, scan_min_len)
        self._force_sequential_after_oom = False
        if self.kda_mode not in {"sequential", "chunked", "scan", "auto"}:
            raise ValueError(f"Unsupported kda_mode '{kda_mode}'.")
        threshold_gb = os.environ.get("KDA_SCAN_GPU_THRESHOLD_GB")
        try:
            threshold_gb = int(threshold_gb) if threshold_gb is not None else 30
        except ValueError:
            threshold_gb = 30
        self.scan_memory_threshold_bytes = max(1, threshold_gb) * 1024**3
        block_override = os.environ.get("KDA_SCAN_BLOCK_TOKENS")
        if block_override is not None:
            try:
                self.scan_block_override = max(1, int(block_override))
            except ValueError:
                self.scan_block_override = None
        else:
            self.scan_block_override = None
        batch_override = os.environ.get("KDA_SCAN_BATCH_BLOCK")
        if batch_override is not None:
            try:
                self.scan_batch_block_override = max(1, int(batch_override))
            except ValueError:
                self.scan_batch_block_override = None
        else:
            self.scan_batch_block_override = None

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
        self.q_norm = HeadwiseRMSNorm(num_heads, dk)
        self.k_norm = HeadwiseRMSNorm(num_heads, dk)
        self.v_norm = HeadwiseRMSNorm(num_heads, dv)
        self.low_rank_u1 = nn.Linear(d_model, rank_gate)
        self.low_rank_u2 = nn.Linear(rank_gate, d_model)
        self.ff_norm = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)
        self.alpha_floor = 1e-4
        self.beta_floor = 0.0

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

        q = self.q_norm(self._reshape_heads(self.q_proj(h), self.dk))
        k = self.k_norm(self._reshape_heads(self.k_proj(h), self.dk))
        v = self.v_norm(self._reshape_heads(self.v_proj(h), self.dv))

        spike_drive = self._reshape_heads(self.spike_proj(h), self.dk)
        alpha_base = self._reshape_heads(self.alpha_down(torch.nn.functional.silu(self.alpha_up(h))), self.dk)
        beta_base = self.beta_proj(h)  # [B,T,H]

        drop_prob = self.token_drop_prob if self.training else 0.0
        if self._use_scan(seq, device):
            y, s, spike_mem = self._forward_parallel_scan(
                q, k, v, spike_drive, alpha_base, beta_base, s, spike_mem, drop_prob
            )
        elif self._use_chunked(seq):
            y, s, spike_mem = self._forward_parallel_scan(
                q,
                k,
                v,
                spike_drive,
                alpha_base,
                beta_base,
                s,
                spike_mem,
                drop_prob,
                block_override=self.chunk_size,
            )
        else:
            y, s, spike_mem = self._forward_sequential(
                q, k, v, spike_drive, alpha_base, beta_base, s, spike_mem, drop_prob
            )
        y = self.head_norm(y).reshape(bsz, seq, -1)

        gate = torch.sigmoid(self.low_rank_u2(torch.nn.functional.silu(self.low_rank_u1(x))))
        y = x + self.out_proj(y) * gate
        y = y + self.ff(self.ff_norm(y))

        return y, KDAState(tensor=s, spike_mem=spike_mem)
    
    def _forward_sequential(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        spike_drive: torch.Tensor,
        alpha_base: torch.Tensor,
        beta_base: torch.Tensor,
        s: torch.Tensor,
        spike_mem: torch.Tensor,
        drop_prob: float,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sequential token-by-token processing with automatic batch slicing on OOM."""
        bsz, seq, _, _ = q.shape
        device = q.device
        drop_mask = self._prepare_drop_mask(drop_mask, bsz, seq, drop_prob, device)
        try:
            return self._forward_sequential_inner(
                q,
                k,
                v,
                spike_drive,
                alpha_base,
                beta_base,
                s,
                spike_mem,
                drop_mask,
            )
        except RuntimeError as exc:
            if not self._is_cuda_oom(exc) or bsz == 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._forward_sequential_batch_sliced(
                q,
                k,
                v,
                spike_drive,
                alpha_base,
                beta_base,
                s,
                spike_mem,
                drop_mask,
            )

    def _forward_sequential_inner(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        spike_drive: torch.Tensor,
        alpha_base: torch.Tensor,
        beta_base: torch.Tensor,
        s: torch.Tensor,
        spike_mem: torch.Tensor,
        drop_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq, _, _ = q.shape

        outputs = []
        prev_out = None
        for t in range(seq):
            spike_mem, spike = self.lif(spike_mem, spike_drive[:, t])
            alpha = torch.sigmoid(alpha_base[:, t] + self.alpha_spike * spike)
            alpha = torch.clamp(alpha, self.alpha_floor, 1.0)
            beta = beta_base[:, t].unsqueeze(-1)
            beta = torch.sigmoid(beta + torch.sum(spike * self.beta_spike, dim=-1, keepdim=True))
            beta = torch.clamp(beta, self.beta_floor, 1.0)
            active = (spike.abs().sum(dim=-1, keepdim=True) > 0).to(q.dtype)
            if drop_mask is not None:
                token_keep = (~drop_mask[:, t]).to(q.dtype).unsqueeze(-1).unsqueeze(-1)
                active = active * token_keep
            s, out = self._step_optimized(s, q[:, t], k[:, t], v[:, t], alpha, beta, active)
            if drop_mask is not None:
                if prev_out is None:
                    prev_out = torch.zeros_like(out)
                mask_t = drop_mask[:, t].to(out.dtype).unsqueeze(-1).unsqueeze(-1)
                out = out * (1 - mask_t) + prev_out * mask_t
                prev_out = out.detach()
            else:
                prev_out = out.detach()
            outputs.append(out)

        output_tensor = torch.stack(outputs, dim=1)
        return output_tensor, s, spike_mem

    def _forward_sequential_batch_sliced(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        spike_drive: torch.Tensor,
        alpha_base: torch.Tensor,
        beta_base: torch.Tensor,
        s: torch.Tensor,
        spike_mem: torch.Tensor,
        drop_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq, _, _ = q.shape
        outputs = torch.empty(
            bsz,
            seq,
            self.num_heads,
            self.dv,
            device=q.device,
            dtype=q.dtype,
        )
        state_out = s.clone()
        spike_out = spike_mem.clone()

        batch_block = max(1, min(bsz, self._select_scan_batch_block(bsz, q.device)))
        adaptive_block = batch_block
        start = 0

        while start < bsz:
            current = min(adaptive_block, bsz - start)
            end = start + current
            drop_slice = drop_mask[start:end] if drop_mask is not None else None
            try:
                out_slice, new_state, new_spike = self._forward_sequential_inner(
                    q[start:end],
                    k[start:end],
                    v[start:end],
                    spike_drive[start:end],
                    alpha_base[start:end],
                    beta_base[start:end],
                    state_out[start:end].clone(),
                    spike_out[start:end].clone(),
                    drop_slice,
                )
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc) or current <= 1:
                    raise
                adaptive_block = max(1, current // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            outputs[start:end] = out_slice
            state_out[start:end] = new_state
            spike_out[start:end] = new_spike
            start = end

        return outputs, state_out, spike_out
    
    def _forward_parallel_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        spike_drive: torch.Tensor,
        alpha_base: torch.Tensor,
        beta_base: torch.Tensor,
        s: torch.Tensor,
        spike_mem: torch.Tensor,
        drop_prob: float,
        block_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq, _, _ = q.shape
        device = q.device

        drop_mask = self._sample_drop_mask(bsz, seq, drop_prob, device)
        if self._force_sequential_after_oom:
            return self._forward_sequential(
                q,
                k,
                v,
                spike_drive,
                alpha_base,
                beta_base,
                s,
                spike_mem,
                drop_prob,
                drop_mask=drop_mask,
            )

        state_backup = s.clone()
        spike_backup = spike_mem.clone()
        planned_block = (
            max(1, block_override)
            if block_override is not None
            else self._select_scan_block_tokens(seq, bsz, device, q.element_size())
        )
        emit_chunk = max(1, min(self.chunk_size, planned_block))

        try:
            outputs_chunks = []
            state = s
            current_spike_mem = spike_mem
            adaptive_block = max(1, planned_block)
            start = 0

            while start < seq:
                block_len = min(adaptive_block, seq - start)
                state_snapshot = state.clone()
                spike_snapshot = current_spike_mem.clone()
                try:
                    current_spike_mem, spike_block = self._compute_spike_block(
                        current_spike_mem, spike_drive[:, start : start + block_len]
                    )
                    alpha_block = torch.sigmoid(
                        alpha_base[:, start : start + block_len]
                        + self.alpha_spike.unsqueeze(1) * spike_block
                    )
                    alpha_block = torch.clamp(alpha_block, self.alpha_floor, 1.0)
                    beta_block = torch.sigmoid(
                        beta_base[:, start : start + block_len].unsqueeze(-1)
                        + torch.sum(spike_block * self.beta_spike.unsqueeze(1), dim=-1, keepdim=True)
                    )
                    beta_block = torch.clamp(beta_block, self.beta_floor, 1.0)
                    active_block = (spike_block.abs().sum(dim=-1, keepdim=True) > 0).to(q.dtype)
                    updates: KDAParallelUpdates = precompute_updates(
                        k[:, start : start + block_len],
                        v[:, start : start + block_len],
                        alpha_block,
                        beta_block,
                        active_block,
                        drop_mask[:, start : start + block_len] if drop_mask is not None else None,
                    )
                    block_outputs, state = scan_emit_outputs(
                        updates,
                        state,
                        q[:, start : start + block_len],
                        chunk_size=min(emit_chunk, block_len),
                    )
                    outputs_chunks.append(block_outputs)
                    start += block_len
                except RuntimeError as exc:
                    if not self._is_cuda_oom(exc) or adaptive_block == 1:
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    adaptive_block = max(1, adaptive_block // 2)
                    emit_chunk = max(1, min(self.chunk_size, adaptive_block))
                    state = state_snapshot
                    current_spike_mem = spike_snapshot
                    continue

            outputs = torch.cat(outputs_chunks, dim=1)
            outputs = self._apply_drop_outputs(outputs, drop_mask)
            return outputs, state, current_spike_mem
        except RuntimeError as exc:
            if not self._is_cuda_oom(exc):
                raise
            self._force_sequential_after_oom = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._forward_sequential(
                q,
                k,
                v,
                spike_drive,
                alpha_base,
                beta_base,
                state_backup,
                spike_backup,
                drop_prob,
                drop_mask=drop_mask,
            )

    def _compute_spike_block(
        self, spike_mem: torch.Tensor, drive_block: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spikes for a block without materializing the whole sequence.
        
        Note: This loop is sequential due to LIF dynamics but will be optimized
        by torch.compile when the model is compiled.
        """

        bsz, block_len, _, _ = drive_block.shape
        spikes = torch.empty(
            bsz,
            block_len,
            self.num_heads,
            self.dk,
            device=drive_block.device,
            dtype=drive_block.dtype,
        )
        current = spike_mem
        # Sequential loop required for LIF dynamics - compilation will optimize
        for idx in range(block_len):
            current, spike = self.lif(current, drive_block[:, idx])
            spikes[:, idx] = spike
        return current, spikes

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

        return self._step_optimized(s, q, k, v, alpha, beta, active)
    
    def _step_optimized(
        self,
        s: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        active: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Numerically stable fast-weight update."""

        decay = torch.clamp(alpha, self.alpha_floor, 1.0)
        write_gate = torch.clamp(beta, self.beta_floor, 1.0)
        if active is not None:
            decay = torch.where(active.expand_as(decay) > 0, decay, torch.ones_like(decay))
            write_gate = write_gate * active

        s = s * decay.unsqueeze(-1)
        s = s + write_gate.unsqueeze(-1) * self._kv_outer(k, v)
        out = torch.einsum("bhdv,bhd->bhv", s, q)
        return s, out

    def _kv_outer(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))

    def _reshape_heads(self, proj: torch.Tensor, width: int) -> torch.Tensor:
        bsz, seq, _ = proj.shape
        return proj.view(bsz, seq, self.num_heads, width)

    def _sample_drop_mask(self, bsz: int, seq: int, drop_prob: float, device: torch.device) -> Optional[torch.Tensor]:
        if drop_prob <= 0.0 or not self.training:
            self.last_drop_fraction = 0.0
            return None
        mask = torch.rand(bsz, seq, device=device) < drop_prob
        guard = min(2, seq)
        if guard > 0:
            mask[:, :guard] = False
            mask[:, -guard:] = False
        self.last_drop_fraction = float(mask.float().mean())
        return mask

    def _prepare_drop_mask(
        self,
        drop_mask: Optional[torch.Tensor],
        bsz: int,
        seq: int,
        drop_prob: float,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if drop_mask is not None:
            if drop_mask.numel() > 0:
                self.last_drop_fraction = float(drop_mask.float().mean())
            else:
                self.last_drop_fraction = 0.0
            return drop_mask
        return self._sample_drop_mask(bsz, seq, drop_prob, device)

    def _apply_drop_outputs(self, outputs: torch.Tensor, drop_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if drop_mask is None:
            return outputs
        bsz, seq, heads, dv = outputs.shape
        keep = (~drop_mask).to(outputs.device)
        time_idx = torch.arange(seq, device=outputs.device).view(1, seq).expand(bsz, -1)
        masked_time = torch.where(keep, time_idx, torch.full_like(time_idx, -1))
        last_keep = torch.cummax(masked_time, dim=1)[0]
        safe_idx = last_keep.clamp(min=0)
        gather_idx = safe_idx.view(bsz, seq, 1, 1).expand(-1, -1, heads, dv)
        gathered = torch.gather(outputs, dim=1, index=gather_idx)
        gathered = torch.where(
            (last_keep >= 0).view(bsz, seq, 1, 1),
            gathered,
            torch.zeros_like(gathered),
        )
        keep_mask = keep.view(bsz, seq, 1, 1).to(outputs.dtype)
        return outputs * keep_mask + gathered * (1 - keep_mask)

    def _select_scan_block_tokens(
        self,
        seq: int,
        batch: int,
        device: torch.device,
        element_size: int,
    ) -> int:
        min_tokens = 1
        chunk_floor = max(1, self.chunk_size)
        preferred = max(chunk_floor, self.scan_min_len)
        if self.scan_block_override is not None:
            return min(seq, max(min_tokens, self.scan_block_override))

        block = preferred
        lowmem_cap = False
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                idx = device.index if device.index is not None else torch.cuda.current_device()
            except RuntimeError:
                idx = None
            if idx is not None:
                total_mem = None
                try:
                    total_mem = torch.cuda.get_device_properties(idx).total_memory
                except (AssertionError, RuntimeError):
                    total_mem = None
                if total_mem is not None and total_mem <= self.scan_memory_threshold_bytes:
                    lowmem_cap = True
                    block = 1
                free_mem = None
                mem_getter = getattr(torch.cuda, "mem_get_info", None)
                if mem_getter is not None:
                    try:
                        free_mem, _ = mem_getter(idx)
                    except (AssertionError, RuntimeError):
                        free_mem = None
                if free_mem:
                    per_token = self._estimate_scan_token_bytes(batch, element_size)
                    if per_token > 0:
                        frac = 0.05 if lowmem_cap else 0.2
                        safe_bytes = max(1, int(free_mem * frac))
                        est = safe_bytes // per_token
                        if est > 0:
                            block = max(1, min(block, est))
                        else:
                            block = 1

        return min(seq, max(min_tokens, block))

    def _select_scan_batch_block(self, batch: int, device: torch.device) -> int:
        if batch <= 1:
            return 1
        if self.scan_batch_block_override is not None:
            return min(batch, max(1, self.scan_batch_block_override))

        block = batch
        lowmem_cap = False
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                idx = device.index if device.index is not None else torch.cuda.current_device()
            except RuntimeError:
                idx = None
            if idx is not None:
                total_mem = None
                try:
                    total_mem = torch.cuda.get_device_properties(idx).total_memory
                except (AssertionError, RuntimeError):
                    total_mem = None
                if total_mem is not None and total_mem <= self.scan_memory_threshold_bytes:
                    lowmem_cap = True
                    block = 1
        if lowmem_cap and block > 1:
            block = max(1, min(block, batch // 2 or 1))
        return block

    def _estimate_scan_token_bytes(self, batch: int, element_size: int) -> int:
        mat_terms = 3 * self.dk * self.dk + 2 * self.dk * self.dv
        per_token = batch * self.num_heads * mat_terms * max(1, element_size)
        return max(1, per_token)

    @staticmethod
    def _is_cuda_oom(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "out of memory" in message and ("cuda" in message or "cublas" in message or "cudnn" in message)

    def _use_scan(self, seq: int, device: torch.device) -> bool:
        if self.kda_mode == "scan":
            device_ok = True
        elif self.kda_mode == "auto":
            device_ok = device.type in {"cuda", "xpu", "mps"}
        else:
            return False
        return device_ok and seq >= self.scan_min_len

    def _use_chunked(self, seq: int) -> bool:
        if self.kda_mode == "chunked":
            return True
        return self.kda_mode == "auto" and self.training and seq > self.chunk_size

    def stream(self, x_t: torch.Tensor, state: Optional[KDAState] = None) -> Tuple[torch.Tensor, KDAState]:
        """Processes a single timestep (useful for autoregressive decoding)."""

        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        y, new_state = self.forward(x_t, state)
        return y[:, 0], new_state
