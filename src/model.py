"""Model composition for Baby-Hatchling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attn_kda import KDAState, KDABlock
from .episodic_mem import EpisodicMemory
from .global_attn_nope import GlobalNoPEBlock
from .predictive_head import PredictiveCodingHead, PredictiveOutput


@dataclass
class ModelConfig:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    dk: int
    dv: int
    d_ff: int
    rank_gate: int
    group_kv: int
    gw_tokens: int
    max_seq: int
    spike_decay: float = 0.9
    spike_threshold: float = 0.5
    spike_surrogate_beta: float = 10.0
    vocab_size: int = 32000
    episodic_bytes: int = 32 * 1024 * 1024
    use_predictive_head: bool = True
    use_episodic_memory: bool = True
    use_curiosity_bonus: bool = True
    use_gradient_checkpointing: bool = False
    kda_chunk_size: int = 16
    token_drop: Optional[dict] = None


class BabyHatchlingModel(nn.Module):
    def __init__(self, cfg: ModelConfig, episodic_path: str = "data/episodic.db") -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList()
        self.kda_positions: List[int] = []
        drop_cfg = cfg.token_drop or {}
        drop_enabled = drop_cfg.get("enabled", False)
        drop_prob = drop_cfg.get("prob", 0.0)
        drop_min = drop_cfg.get("min_layer", 0)
        drop_max = drop_cfg.get("max_layer", cfg.n_layers)
        for idx in range(cfg.n_layers):
            if (idx + 1) % 4 == 0:
                block = GlobalNoPEBlock(
                    d_model=cfg.d_model,
                    num_heads=cfg.n_heads,
                    dk=cfg.dk,
                    dv=cfg.dv,
                    d_ff=cfg.d_ff,
                    group_kv=cfg.group_kv,
                    gw_tokens=cfg.gw_tokens,
                )
            else:
                layer_drop_prob = drop_prob if (drop_enabled and drop_min <= idx < drop_max) else 0.0
                block = KDABlock(
                    d_model=cfg.d_model,
                    num_heads=cfg.n_heads,
                    dk=cfg.dk,
                    dv=cfg.dv,
                    d_ff=cfg.d_ff,
                    rank_gate=cfg.rank_gate,
                    spike_decay=cfg.spike_decay,
                    spike_threshold=cfg.spike_threshold,
                    spike_surrogate_beta=cfg.spike_surrogate_beta,
                    token_drop_prob=layer_drop_prob,
                    chunk_size=getattr(cfg, 'kda_chunk_size', 16),
                )
                self.kda_positions.append(idx)
            self.layers.append(block)

        self.use_memory = cfg.use_episodic_memory
        self.use_curiosity = cfg.use_curiosity_bonus
        self.use_checkpoint = cfg.use_gradient_checkpointing
        self.pred_head = PredictiveCodingHead(
            cfg.d_model, cfg.vocab_size, self.embedding, enabled=cfg.use_predictive_head
        )
        self.memory = (
            EpisodicMemory(dim=cfg.d_model, path=episodic_path, max_bytes=cfg.episodic_bytes)
            if self.use_memory
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[KDAState]] = None,
        use_memory: bool = True,
    ) -> tuple[PredictiveOutput, List[KDAState]]:
        hidden = self.embedding(input_ids)
        new_states: List[KDAState] = []
        state_iter = iter(states or [])
        for layer in self.layers:
            if isinstance(layer, KDABlock):
                layer_state = next(state_iter, None)
                hidden, updated_state = layer(hidden, layer_state)
                new_states.append(updated_state)
            else:
                if self.use_checkpoint and self.training:
                    hidden = checkpoint(layer, hidden, use_reentrant=False)
                else:
                    hidden = layer(hidden)

        if use_memory and self.use_memory:
            hidden = self._memory_read(hidden)

        pred_out = self.pred_head(hidden)
        if use_memory and self.use_memory:
            self._memory_write(hidden, pred_out)
        return pred_out, new_states

    def _memory_read(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.use_memory or self.memory is None:
            return hidden
        last = hidden[:, -1, :]
        with torch.no_grad():
            logits = last @ self.pred_head.decoder.weight.t()
            probs = torch.softmax(logits, dim=-1)
            entropy = (-probs * torch.log(torch.clamp(probs, min=1e-6))).sum(dim=-1)
            enriched = []
            for idx in range(last.shape[0]):
                enriched.append(self.memory.maybe_read(last[idx].detach(), float(entropy[idx])))
            enriched_tensor = torch.stack(enriched, dim=0).to(hidden.device)
        delta = torch.zeros_like(hidden)
        delta[:, -1, :] = (enriched_tensor - last).detach()
        return hidden + delta

    def _memory_write(self, hidden: torch.Tensor, pred: PredictiveOutput) -> None:
        if not self.use_memory or self.memory is None:
            return
        last = hidden[:, -1, :]
        logits = pred.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        entropy = (-probs * torch.log(torch.clamp(probs, min=1e-6))).sum(dim=-1)
        error = pred.error_trace[:, -1]
        with torch.no_grad():
            for idx in range(last.shape[0]):
                key = hidden[idx, -4:, :].mean(dim=0).detach()
                value = last[idx].detach()
                self.memory.maybe_write(key, value, float(entropy[idx]), float(error[idx]))

    def episodic_gate_penalty(self) -> float:
        if not self.use_memory or self.memory is None:
            return 0.0
        return self.memory.gate_penalty()


def build_model(cfg_dict: dict) -> BabyHatchlingModel:
    cfg = ModelConfig(**cfg_dict)
    return BabyHatchlingModel(cfg)
