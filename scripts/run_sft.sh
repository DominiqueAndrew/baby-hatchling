#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/hatchling_xs.yaml}
CKPT=${2:-out/xs_pretrain.pt}
python -m src.trainer \
  --config "$CONFIG" \
  --stage sft \
  --load "$CKPT" \
  --save out/xs_sft.pt
