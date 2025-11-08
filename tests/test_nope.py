import torch

from src.global_attn_nope import GlobalNoPEBlock


def test_nope_block_is_causal():
    torch.manual_seed(0)
    block = GlobalNoPEBlock(d_model=12, num_heads=3, dk=4, dv=4, d_ff=24, group_kv=1, gw_tokens=0)
    x = torch.randn(1, 5, 12)
    y = block(x)
    x_mask = x.clone()
    x_mask[:, -1, :] = 0.0
    y_mask = block(x_mask)
    assert torch.allclose(y[:, :-1, :], y_mask[:, :-1, :], atol=1e-5)
