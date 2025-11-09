import torch

from src.attn_kda import KDABlock


def test_kda_stream_matches_full_sequence():
    torch.manual_seed(0)
    block = KDABlock(
        d_model=16,
        num_heads=2,
        dk=4,
        dv=4,
        d_ff=32,
        rank_gate=8,
        spike_threshold=10.0,  # effectively disable spikes for determinism
    )
    x = torch.randn(1, 6, 16)
    full, full_state = block(x)
    first, state = block(x[:, :3])
    second, state = block(x[:, 3:], state)
    stitched = torch.cat([first, second], dim=1)
    assert torch.allclose(full, stitched, atol=1e-5)
    assert torch.allclose(full_state.tensor, state.tensor, atol=1e-5)
