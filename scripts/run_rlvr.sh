#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/hatchling_xs.yaml}
CKPT=${2:-out/xs_sft.pt}
python -m src.policy_rlvr \
  --config "$CONFIG" \
  --load "$CKPT" \
  --save out/xs_rlvr.pt
