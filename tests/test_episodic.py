import tempfile

import torch

from src.episodic_mem import EpisodicMemory


def test_memory_write_and_read():
    with tempfile.TemporaryDirectory() as tmp:
        mem = EpisodicMemory(dim=4, path=f"{tmp}/mem.db", write_threshold=0.1, read_threshold=0.1)
        key = torch.ones(4)
        value = torch.zeros(4)
        mem.add(key, value)
        vec, gate = mem.query(key)
        assert gate > 0.0
        fused = mem.maybe_read(key, entropy=0.5)
        assert torch.allclose(fused, key, atol=1e-6)
        assert mem.maybe_write(key, value, entropy=0.2, error=0.5)
