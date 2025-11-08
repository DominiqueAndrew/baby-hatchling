#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/hatchling_xs.yaml}
CKPT=${2:-out/xs_rlvr.pt}
LABEL=${3:-baseline}

python -m src.eval_harness.arc_easy --config "$CONFIG" --load "$CKPT" > logs/ablation_${LABEL}_arc_easy.txt
python -m src.eval_harness.winogrande --config "$CONFIG" --load "$CKPT" > logs/ablation_${LABEL}_winogrande.txt
python -m src.eval_harness.hellaswag --config "$CONFIG" --load "$CKPT" > logs/ablation_${LABEL}_hellaswag.txt
python -m src.eval_harness.gsm8k --config "$CONFIG" --load "$CKPT" > logs/ablation_${LABEL}_gsm8k.txt
python -m src.eval_harness.humaneval --config "$CONFIG" --load "$CKPT" > logs/ablation_${LABEL}_humaneval.txt
python -m src.eval_harness.probes --config "$CONFIG" --load "$CKPT" > logs/ablation_${LABEL}_probes.txt
