#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/hatchling_xs.yaml}
python -m src.trainer \
  --config "$CONFIG" \
  --stage pretrain \
  --save out/xs_pretrain.pt
