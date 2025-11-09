#!/usr/bin/env bash
set -euo pipefail
CKPT=${1:-out/hn_xs_rlvr.pt}
python -m src.eval_harness.arc_easy --load "$CKPT"
python -m src.eval_harness.winogrande --load "$CKPT"
python -m src.eval_harness.hellaswag --load "$CKPT"
python -m src.eval_harness.gsm8k --load "$CKPT"
python -m src.eval_harness.humaneval --load "$CKPT"
python -m src.eval_harness.probes --load "$CKPT"
