"""Adaptive Partial Momentum optimizer (AdaPM)."""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class AdaPM(Optimizer):
    """Optimizer that keeps low-rank momentum buffers to cut memory usage.

    For matrices, momentum is stored per output channel (row). For vectors,
    element-wise momentum is kept. This approximates full momentum while
    shrinking the state footprint by >90% on typical transformer layers.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta: float = 0.9,
        gamma: float = 0.3,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, beta=beta, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            gamma = group["gamma"]
            wd = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if wd != 0.0:
                    grad = grad.add(param, alpha=wd)

                state = self.state[param]

                if grad.ndim >= 2:
                    rows = grad.size(0)
                    if "row_momentum" not in state:
                        state["row_momentum"] = torch.zeros(
                            rows, 1, device=grad.device, dtype=grad.dtype
                        )
                    buf = state["row_momentum"]
                    grad_mean = grad.mean(dim=1, keepdim=True)
                    buf.mul_(beta).add_(grad_mean, alpha=1 - beta)
                    update = grad + gamma * buf
                else:
                    if "scalar_momentum" not in state:
                        state["scalar_momentum"] = torch.zeros_like(grad)
                    buf = state["scalar_momentum"]
                    buf.mul_(beta).add_(grad, alpha=1 - beta)
                    update = grad + gamma * buf

                param.add_(update, alpha=-lr)

        return loss
