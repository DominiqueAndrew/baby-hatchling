import torch
from torch import nn

from src.utils.sparsity import MSTSparsifier, SparsitySchedule


def test_mst_sparsifier_reduces_weights():
    linear = nn.Linear(64, 64, bias=False)
    schedule = SparsitySchedule(warmup=0, prune=1, restore=0, target=0.5, update_every=1)
    sparsifier = MSTSparsifier([linear], schedule)
    sparsifier.maybe_update(step=1)
    nonzero = linear.weight.data.count_nonzero()
    assert nonzero < linear.weight.data.numel()
