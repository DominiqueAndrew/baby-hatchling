"""CLI utility to summarize gridworld curiosity logs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def summarize(path: Path, window: int = 10) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Log file {path} does not exist")
    df = pd.read_csv(path)
    required = {"episode", "step", "intrinsic", "pred_error", "reward"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    grouped = df.groupby("episode")
    stats = grouped[["intrinsic", "pred_error", "reward"]].mean()
    stats["steps"] = grouped.size()

    print("Episode summary (mean per step):")
    print(stats.round(4))

    corr = df["intrinsic"].corr(df["pred_error"])
    print(f"\nPearson corr(intrinsic, pred_error) = {corr:.4f}")

    if len(df) >= window:
        rolling = df[["intrinsic", "pred_error"]].rolling(window, min_periods=1).mean()
        delta = rolling["pred_error"].diff()
        improved = (delta < 0).mean()
        print(f"Rolling-window ({window}) predictive error decreases {improved*100:.1f}% of the time")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize gridworld curiosity logs")
    parser.add_argument("--log", required=True, help="Path to gridworld CSV log")
    parser.add_argument("--window", type=int, default=10, help="Rolling window for improvement stats")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summarize(Path(args.log), window=args.window)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
