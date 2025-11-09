#!/usr/bin/env bash
# Auto-sync script that runs in the background
# Continuously syncs logs and checkpoints from cloud to local
# Run this in a separate terminal or as a background process

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_INTERVAL=${1:-30}  # Default: sync every 30 seconds

echo "Starting auto-sync (every ${SYNC_INTERVAL} seconds)..."
echo "Press Ctrl+C to stop"
echo ""

# Trap to cleanup on exit
trap 'echo ""; echo "Auto-sync stopped."; exit 0' INT TERM

while true; do
    bash "$SCRIPT_DIR/cloud_sync.sh" 2>&1 | grep -E "\[SYNC\]|Error|complete" || true
    sleep "$SYNC_INTERVAL"
done

