import torch
from torch import nn

from src.predictive_head import PredictiveCodingHead


def test_predictive_head_computes_losses_and_bonus():
    torch.manual_seed(0)
    embedding = nn.Embedding(32, 16)
    head = PredictiveCodingHead(16, 32, embedding, enabled=True)
    hidden = torch.randn(3, 6, 16, requires_grad=True)
    out = head(hidden)
    assert out.logits.shape[:2] == hidden.shape[:2]
    assert out.pc_loss.item() >= 0.0
    out.pc_loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads, "Predictive head parameters should receive gradients"
    bonus = head.curiosity_bonus(out.error_trace)
    assert isinstance(bonus, float)
