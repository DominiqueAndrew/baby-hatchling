"""Simple CSV + stdout logging utilities."""
from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from rich.console import Console


@dataclass
class CSVLogger:
    """Writes scalar metrics to both stdout and CSV."""

    path: pathlib.Path
    fieldnames: Iterable[str]
    console: Console = field(default_factory=Console)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, float | int | str]) -> None:
        self.console.log(row)
        with self.path.open("a", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)
