#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/hn_xs.yaml}
CKPT=${2:-out/hn_xs_sft.pt}
python -m src.policy_rlvr \
  --config "$CONFIG" \
  --load "$CKPT" \
  --save out/hn_xs_rlvr.pt
