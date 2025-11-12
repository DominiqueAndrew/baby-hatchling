#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/hn_toddler_rlvr.yaml}
INPUT_CKPT=${2:-out/hn_toddler_sft.pt}
OUTPUT_CKPT=${3:-out/hn_toddler_rlvr.pt}

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python -m src.policy_rlvr \
  --config "$CONFIG" \
  --load "$INPUT_CKPT" \
  --save "$OUTPUT_CKPT"
