"""Model composition for Baby-Hatchling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

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
    vocab_size: int = 32000
    episodic_bytes: int = 32 * 1024 * 1024
    use_predictive_head: bool = True
    use_episodic_memory: bool = True
    use_curiosity_bonus: bool = True


class BabyHatchlingModel(nn.Module):
    def __init__(self, cfg: ModelConfig, episodic_path: str = "data/episodic.db") -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList()
        self.kda_positions: List[int] = []
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
                block = KDABlock(
                    d_model=cfg.d_model,
                    num_heads=cfg.n_heads,
                    dk=cfg.dk,
                    dv=cfg.dv,
                    d_ff=cfg.d_ff,
                    rank_gate=cfg.rank_gate,
                )
                self.kda_positions.append(idx)
            self.layers.append(block)

        self.use_memory = cfg.use_episodic_memory
        self.use_curiosity = cfg.use_curiosity_bonus
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
