import torch

from src.attn_kda import KDAState, KDABlock


def test_kda_step_matches_reference():
    torch.manual_seed(0)
    block = KDABlock(d_model=8, num_heads=2, dk=4, dv=4, d_ff=16, rank_gate=4)
    x = torch.randn(1, 3, 8)
    state = KDAState.zeros(1, 2, 4, 4, device=x.device, dtype=x.dtype)
    _, new_state = block(x, state)

    # Manually apply recurrence
    s = state.tensor.clone()
    h = block.norm_in(x)
    conv = block.short_conv(h.transpose(1, 2)).transpose(1, 2)
    h = block.act(conv)
    q = block._reshape_heads(block.q_proj(h), block.dk)
    k = block._reshape_heads(block.k_proj(h), block.dk)
    v = block._reshape_heads(block.v_proj(h), block.dv)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    alpha = torch.sigmoid(block._reshape_heads(block.alpha_proj(h), block.dk))
    beta = torch.sigmoid(block.beta_proj(h)).unsqueeze(-1)

    eye = torch.eye(block.dk, device=x.device).view(1, 1, block.dk, block.dk)
    for t in range(x.size(1)):
        diag = torch.diag_embed(alpha[:, t])
        kkT = k[:, t].unsqueeze(-1) @ k[:, t].unsqueeze(-2)
        term = eye - beta[:, t].unsqueeze(-1) * kkT
        s = torch.matmul(term, torch.matmul(diag, s))
        delta = beta[:, t].unsqueeze(-1) * (k[:, t].unsqueeze(-1) @ v[:, t].unsqueeze(-2))
        s = s + delta
    assert torch.allclose(s, new_state.tensor, atol=1e-4)
