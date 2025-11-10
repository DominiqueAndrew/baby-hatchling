import torch

from src.attn_kda import KDABlock


def test_token_drop_reuses_previous_output():
    torch.manual_seed(0)
    base = KDABlock(d_model=16, num_heads=2, dk=4, dv=4, d_ff=32, rank_gate=8, token_drop_prob=0.0)
    dropped = KDABlock(d_model=16, num_heads=2, dk=4, dv=4, d_ff=32, rank_gate=8, token_drop_prob=1.0)
    dropped.load_state_dict(base.state_dict())
    base.train()
    dropped.train()
    x = torch.randn(1, 6, 16)
    base(x)
    dropped(x)
    assert dropped.last_drop_fraction > 0.2
