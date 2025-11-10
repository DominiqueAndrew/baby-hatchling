import torch

from src.optim.adapm import AdaPM


def test_adapm_maintains_row_momentum():
    lin = torch.nn.Linear(8, 4)
    opt = AdaPM(lin.parameters(), lr=0.1, beta=0.5, gamma=0.2)
    data = torch.randn(2, 8)
    target = torch.randn(2, 4)
    loss_fn = torch.nn.MSELoss()
    for _ in range(2):
        opt.zero_grad()
        out = lin(data)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
    state = next(iter(opt.state.values()))
    assert "row_momentum" in state
    assert state["row_momentum"].shape == (4, 1)
