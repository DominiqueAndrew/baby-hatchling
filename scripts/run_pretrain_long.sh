#!/usr/bin/env bash
# Long-running pretrain script with background execution and monitoring
set -euo pipefail

CONFIG=${1:-configs/hn_xs.yaml}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

# Create logs directory
mkdir -p logs

# Run training in background with nohup
echo "🚀 Starting long pretrain training..."
echo "📝 Logs will be written to: logs/pretrain_runpod.txt"
echo "💾 Checkpoints will be saved to: out/hn_xs_pretrain.pt"
echo ""
echo "To monitor progress:"
echo "  tail -f logs/pretrain_runpod.txt"
echo "  tail -f logs/train_pretrain.csv"
echo ""
echo "To check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""

nohup python -m src.trainer \
  --config "$CONFIG" \
  --stage pretrain \
  --save out/hn_xs_pretrain.pt \
  > logs/pretrain_runpod.txt 2>&1 &

PID=$!
echo "✅ Training started in background (PID: $PID)"
echo "📊 Monitor with: tail -f logs/pretrain_runpod.txt"
echo "🛑 Stop with: kill $PID"

