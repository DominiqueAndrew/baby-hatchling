"""Mixed Sparsity Training utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


@dataclass
class SparsitySchedule:
    warmup: int
    prune: int
    restore: int
    target: float
    update_every: int


class MSTSparsifier:
    """Magnitude-based sparsifier with warmup/prune/restore phases."""

    def __init__(self, modules: Iterable[nn.Linear], schedule: SparsitySchedule) -> None:
        self.modules: List[nn.Linear] = list(modules)
        self.schedule = schedule
        self.masks: List[torch.Tensor] = []
        self.current_sparsity = 0.0
        for module in self.modules:
            mask = torch.ones_like(module.weight)
            self.masks.append(mask)
            module.register_buffer("_mst_mask", mask, persistent=False)

    def _desired_sparsity(self, step: int) -> float:
        warm = self.schedule.warmup
        prune = self.schedule.prune
        restore = self.schedule.restore
        target = self.schedule.target

        if step <= warm:
            return 0.0
        elif step <= warm + prune:
            frac = (step - warm) / max(1, prune)
            return target * frac
        elif step <= warm + prune + restore:
            frac = 1 - (step - warm - prune) / max(1, restore)
            return max(0.0, target * frac)
        else:
            return 0.0

    def apply_masks(self) -> None:
        for module, mask in zip(self.modules, self.masks):
            module.weight.data.mul_(mask)

    @torch.no_grad()
    def maybe_update(self, step: int) -> None:
        if not self.modules:
            return
        if self.schedule.update_every <= 0:
            return
        if step % self.schedule.update_every != 0:
            return
        desired = self._desired_sparsity(step)
        if abs(desired - self.current_sparsity) <= 1e-3:
            return
        self.current_sparsity = desired
        for module, mask in zip(self.modules, self.masks):
            weight = module.weight.data
            total = weight.numel()
            active = max(1, int(total * (1 - desired)))
            scores = weight.abs().reshape(-1)
            if active >= scores.numel():
                new_mask = torch.ones_like(weight)
            else:
                topk = torch.topk(scores, active, largest=True)
                flat_mask = torch.zeros_like(scores)
                flat_mask.scatter_(0, topk.indices, 1.0)
                new_mask = flat_mask.view_as(weight)
            mask.copy_(new_mask)
            module.weight.data.mul_(new_mask)


def collect_sparse_modules(model: nn.Module, min_dim: int = 64) -> List[nn.Linear]:
    modules: List[nn.Linear] = []
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.weight.requires_grad:
            if module.weight.size(0) >= min_dim and module.weight.size(1) >= min_dim:
                modules.append(module)
    return modules
