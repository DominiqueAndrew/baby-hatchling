#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/hn_xs.yaml}
CKPT=${2:-out/hn_xs_sft.pt}

python -m src.trainer --config "$CONFIG" --stage gridworld --load "$CKPT"

# Auto-summarize curiosity vs. predictive error if logs exist.
LOG_PATH=$(CONFIG="$CONFIG" PYTHONPATH=. python - <<'PY'
import os
from src.utils.config import load_config

cfg = load_config(os.environ["CONFIG"])
print(cfg.get("gridworld", {}).get("log_path", "logs/gridworld.csv"))
PY
)
LOG_PATH=${LOG_PATH//$'\n'/}
if [ -f "$LOG_PATH" ]; then
  PYTHONPATH=. python -m src.tools.gridworld_report --log "$LOG_PATH"
else
  echo "Gridworld log $LOG_PATH not found; skipping summary" >&2
fi
