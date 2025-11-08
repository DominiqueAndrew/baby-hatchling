"""Tiny episodic memory backed by SQLite + FAISS."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import faiss
import numpy as np
import torch


class EpisodicMemory:
    def __init__(
        self,
        dim: int,
        path: str | Path,
        max_bytes: int = 32 * 1024 * 1024,
        write_threshold: float = 0.5,
        read_threshold: float = 0.2,
    ) -> None:
        self.dim = dim
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS entries (id INTEGER PRIMARY KEY AUTOINCREMENT, key_vec BLOB, value_vec BLOB)"
        )
        self.conn.commit()
        self.write_threshold = write_threshold
        self.read_threshold = read_threshold
        self.max_entries = max(1, max_bytes // (dim * 4 * 2))
        self.keys = np.zeros((0, dim), dtype="float32")
        self.values = np.zeros((0, dim), dtype="float32")
        self.index = faiss.IndexFlatL2(dim)
        self.last_gate = 0.0

    def should_write(self, entropy: float, error: float) -> bool:
        return error > self.write_threshold or entropy > self.write_threshold

    def should_read(self, entropy: float) -> bool:
        return entropy > self.read_threshold and len(self.keys) > 0

    def add(self, key: torch.Tensor, value: torch.Tensor) -> None:
        key_np = key.detach().cpu().to(torch.float32).numpy().reshape(1, self.dim)
        value_np = value.detach().cpu().to(torch.float32).numpy().reshape(1, self.dim)
        self.keys = np.concatenate([self.keys, key_np], axis=0)
        self.values = np.concatenate([self.values, value_np], axis=0)
        self._rebuild_index()
        self.conn.execute(
            "INSERT INTO entries (key_vec, value_vec) VALUES (?, ?)",
            (key_np.tobytes(), value_np.tobytes()),
        )
        self.conn.commit()
        if len(self.keys) > self.max_entries:
            self.keys = self.keys[-self.max_entries :]
            self.values = self.values[-self.max_entries :]
            self._rebuild_index()

    def query(self, key: torch.Tensor, k: int = 4) -> Tuple[torch.Tensor, float]:
        """Returns the weighted sum of the top-k retrieved values and a gate."""

        if len(self.keys) == 0:
            return torch.zeros_like(key), 0.0
        key_np = key.detach().cpu().to(torch.float32).numpy().reshape(1, self.dim)
        distances, indices = self.index.search(key_np, min(k, len(self.keys)))
        weights = torch.softmax(torch.from_numpy(-distances[0]).float(), dim=0)
        retrieved = torch.from_numpy(self.values[indices[0]]).float()
        combined = torch.sum(weights.unsqueeze(-1) * retrieved, dim=0).to(key.device)
        gate = float(torch.clamp(weights.mean(), 0.0, 1.0).item())
        return combined, gate

    def maybe_write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        entropy: float,
        error: float,
    ) -> bool:
        if self.should_write(entropy, error):
            self.add(key, value)
            return True
        return False

    def maybe_read(
        self,
        key: torch.Tensor,
        entropy: float,
    ) -> torch.Tensor:
        if self.should_read(entropy):
            vec, gate = self.query(key)
            self.last_gate = gate
            return key + gate * vec.to(key.device)
        self.last_gate = 0.0
        return key

    def _rebuild_index(self) -> None:
        self.index.reset()
        if len(self.keys) > 0:
            self.index.add(self.keys)

    def gate_penalty(self) -> float:
        return self.last_gate
