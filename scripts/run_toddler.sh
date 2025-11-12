#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/hn_toddler.yaml}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python -m src.trainer \
  --config "$CONFIG" \
  --stage pretrain \
  --save out/hn_toddler.pt
