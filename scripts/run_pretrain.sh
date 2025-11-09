#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/hn_xs.yaml}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"
python -m src.trainer \
  --config "$CONFIG" \
  --stage pretrain \
  --save out/hn_xs_pretrain.pt
