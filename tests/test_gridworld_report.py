import io
from pathlib import Path

import pandas as pd

from src.tools.gridworld_report import summarize


def test_gridworld_report_summarize(tmp_path: Path, capsys):
    data = pd.DataFrame(
        {
            "episode": [0, 0, 1, 1],
            "step": [0, 1, 0, 1],
            "intrinsic": [0.1, 0.2, 0.05, 0.15],
            "pred_error": [0.4, 0.3, 0.5, 0.25],
            "reward": [0.0, 0.1, -0.05, 0.2],
        }
    )
    log_path = tmp_path / "grid.csv"
    data.to_csv(log_path, index=False)

    summarize(log_path, window=2)
    captured = capsys.readouterr().out
    assert "Episode summary" in captured
    assert "corr" in captured
