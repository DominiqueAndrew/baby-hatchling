import torch

from src.attn_kda import KDABlock


def test_kda_state_stays_bounded_and_trains():
    torch.manual_seed(0)
    block = KDABlock(
        d_model=32,
        num_heads=2,
        dk=8,
        dv=8,
        d_ff=64,
        rank_gate=16,
        token_drop_prob=0.0,
        kda_mode="sequential",
    )
    x = torch.randn(2, 256, 32, requires_grad=True)
    out, state = block(x)
    assert torch.isfinite(state.tensor).all()
    assert state.tensor.norm().item() < 1e3
    loss = out.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert grads, "KDABlock parameters received no gradients"
