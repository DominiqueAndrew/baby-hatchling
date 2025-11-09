#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/hn_xs.yaml}
CKPT=${2:-out/hn_xs_pretrain.pt}
python -m src.trainer \
  --config "$CONFIG" \
  --stage sft \
  --load "$CKPT" \
  --save out/hn_xs_sft.pt
