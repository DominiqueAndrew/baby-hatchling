import torch

from src.policy_rlvr import ppo_objective


def test_ppo_objective_clips_ratio():
    logp = torch.tensor([0.0, 2.0])
    ref = torch.tensor([0.0, 0.0])
    rewards = torch.tensor([0.5, 1.0])
    loss, ratio = ppo_objective(logp, ref, rewards, epsilon=0.2, rho_max=1.5)
    assert torch.all(ratio <= 1.5 + 1e-6)
    # When rewards favor the high log-prob sample, objective should encourage it (negative loss)
    assert loss.item() < 0
