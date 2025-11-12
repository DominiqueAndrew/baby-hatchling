#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/hn_toddler_sft.yaml}
INPUT_CKPT=${2:-out/hn_toddler.pt}
OUTPUT_CKPT=${3:-out/hn_toddler_sft.pt}

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python -m src.trainer \
  --config "$CONFIG" \
  --stage sft \
  --load "$INPUT_CKPT" \
  --save "$OUTPUT_CKPT"
