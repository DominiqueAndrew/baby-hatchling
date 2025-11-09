#!/usr/bin/env bash
# Run training on cloud instance
# This script SSHes into the cloud instance and starts training

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$PROJECT_ROOT/.cloud_env" ]; then
    echo "Error: .cloud_env not found. Run scripts/cloud_setup.sh first."
    exit 1
fi

source "$PROJECT_ROOT/.cloud_env"

# Parse arguments
STAGE=${1:-pretrain}
CONFIG=${2:-configs/hatchling_xs.yaml}
SAVE_PATH=${3:-out/xs_${STAGE}.pt}
LOAD_PATH=${4:-""}

# Build SSH options
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
[ -n "${CLOUD_PORT:-}" ] && SSH_OPTS="${SSH_OPTS} -p ${CLOUD_PORT}"
[ -n "${CLOUD_KEY:-}" ] && SSH_OPTS="${SSH_OPTS} -i ${CLOUD_KEY}"

# Build training command
if [ "$STAGE" = "rlvr" ]; then
    # RLVR uses a different module
    TRAIN_CMD="cd ~/baby-hatchling && source .venv/bin/activate && python -m src.policy_rlvr"
    TRAIN_CMD="${TRAIN_CMD} --config ${CONFIG} --save ${SAVE_PATH}"
    [ -n "$LOAD_PATH" ] && TRAIN_CMD="${TRAIN_CMD} --load ${LOAD_PATH}"
else
    # Pretrain and SFT use the trainer module
    TRAIN_CMD="cd ~/baby-hatchling && source .venv/bin/activate && python -m src.trainer"
    TRAIN_CMD="${TRAIN_CMD} --config ${CONFIG} --stage ${STAGE} --save ${SAVE_PATH}"
    [ -n "$LOAD_PATH" ] && TRAIN_CMD="${TRAIN_CMD} --load ${LOAD_PATH}"
fi

echo "Starting training on ${CLOUD_HOST}..."
echo "Command: ${TRAIN_CMD}"
echo ""
echo "Training will run in the background. Use 'tail -f' to monitor logs."
echo "Run 'scripts/cloud_sync.sh' in another terminal to sync results."
echo ""

# Run training in background with nohup
ssh ${SSH_OPTS} "${CLOUD_USER}@${CLOUD_HOST}" << ENDSSH
    cd ~/baby-hatchling
    nohup bash -c "${TRAIN_CMD}" > training.log 2>&1 &
    echo \$! > training.pid
    echo "Training started with PID: \$(cat training.pid)"
    echo "Monitor with: tail -f training.log"
ENDSSH

echo ""
echo "Training started! To monitor:"
echo "  ssh ${CLOUD_USER}@${CLOUD_HOST} 'tail -f ~/baby-hatchling/training.log'"
echo ""
echo "To sync results:"
echo "  bash scripts/cloud_sync.sh"

