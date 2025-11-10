import types

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


def test_kda_scan_matches_sequential():
    torch.manual_seed(1)
    seq_block = KDABlock(
        d_model=32,
        num_heads=2,
        dk=8,
        dv=8,
        d_ff=64,
        rank_gate=16,
        token_drop_prob=0.0,
        kda_mode="sequential",
    )
    scan_block = KDABlock(
        d_model=32,
        num_heads=2,
        dk=8,
        dv=8,
        d_ff=64,
        rank_gate=16,
        token_drop_prob=0.0,
        kda_mode="scan",
        scan_min_len=1,
    )
    scan_block.load_state_dict(seq_block.state_dict())

    x = torch.randn(2, 64, 32)
    out_seq, state_seq = seq_block(x)
    out_scan, state_scan = scan_block(x)
    assert torch.allclose(out_seq, out_scan, atol=1e-5)
    assert torch.allclose(state_seq.tensor, state_scan.tensor, atol=1e-5)


def test_kda_scan_respects_token_drop_mask():
    torch.manual_seed(2)
    seq_block = KDABlock(
        d_model=32,
        num_heads=2,
        dk=8,
        dv=8,
        d_ff=64,
        rank_gate=16,
        token_drop_prob=0.0,
        kda_mode="sequential",
    )
    scan_block = KDABlock(
        d_model=32,
        num_heads=2,
        dk=8,
        dv=8,
        d_ff=64,
        rank_gate=16,
        token_drop_prob=0.0,
        kda_mode="scan",
        scan_min_len=1,
    )
    scan_block.load_state_dict(seq_block.state_dict())
    seq_block.train()
    scan_block.train()

    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[:, 3] = True
    mask[:, 7] = True

    def _inject_mask(block):
        def sampler(self, bsz, seq, drop_prob, device):
            self.last_drop_fraction = float(mask.float().mean())
            return mask.to(device)

        block._sample_drop_mask = types.MethodType(sampler, block)

    _inject_mask(seq_block)
    _inject_mask(scan_block)

    x = torch.randn(1, 10, 32)
    out_seq, _ = seq_block(x)
    out_scan, _ = scan_block(x)
    assert torch.allclose(out_seq, out_scan, atol=1e-5)
