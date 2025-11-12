import torch
import pytest

pytest.importorskip("sqlite3")

from src.episodic_mem import EpisodicMemory


def test_episode_memory_knn_gate_varies(tmp_path):
    mem = EpisodicMemory(dim=4, path=tmp_path / "episodic.db", max_bytes=1024)
    key = torch.randn(4)
    value = torch.randn(4)
    mem.add(key, value)

    close_query = key + 0.01
    far_query = torch.randn(4) + 5.0
    _, close_gate = mem.query(close_query)
    _, far_gate = mem.query(far_query)
    assert close_gate >= far_gate

    mem.maybe_write(key, value, entropy=1.0, error=1.0)
    read_vec = mem.maybe_read(key, entropy=1.0)
    assert read_vec.shape == key.shape
    assert mem.gate_penalty() >= 0.0
