#!/usr/bin/env bash
# Sync logs and checkpoints from cloud instance to local machine
# Run this script periodically or in the background to keep results up to date

set -euo pipefail

# Load cloud configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$PROJECT_ROOT/.cloud_env" ]; then
    echo "Error: .cloud_env not found. Run scripts/cloud_setup.sh first."
    exit 1
fi

source "$PROJECT_ROOT/.cloud_env"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[SYNC]${NC} $1"
}

# Build SSH and RSYNC options
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
RSYNC_SSH_OPTS=""
[ -n "${CLOUD_PORT:-}" ] && {
    SSH_OPTS="${SSH_OPTS} -p ${CLOUD_PORT}"
    RSYNC_SSH_OPTS="-e 'ssh -p ${CLOUD_PORT} -i ${CLOUD_KEY}'"
}
[ -n "${CLOUD_KEY:-}" ] && {
    SSH_OPTS="${SSH_OPTS} -i ${CLOUD_KEY}"
    [ -z "${CLOUD_PORT:-}" ] && RSYNC_SSH_OPTS="-e 'ssh -i ${CLOUD_KEY}'"
}

# Remote paths
REMOTE_DIR="~/baby-hatchling"
REMOTE_LOGS="${REMOTE_DIR}/logs"
REMOTE_OUT="${REMOTE_DIR}/out"

# Local paths
LOCAL_LOGS="${PROJECT_ROOT}/logs"
LOCAL_OUT="${PROJECT_ROOT}/out"

# Create local directories if they don't exist
mkdir -p "$LOCAL_LOGS" "$LOCAL_OUT"

# Sync function
sync_directory() {
    local remote_path=$1
    local local_path=$2
    local name=$3
    
    log_info "Syncing ${name}..."
    
    if [ -n "$RSYNC_SSH_OPTS" ]; then
        eval rsync -avz --progress ${RSYNC_SSH_OPTS} \
            "${CLOUD_USER}@${CLOUD_HOST}:${remote_path}/" \
            "${local_path}/" 2>/dev/null || {
            log_info "${name} not found on remote or sync failed (this is OK if training hasn't started)"
            return 0
        }
    else
        rsync -avz --progress \
            "${CLOUD_USER}@${CLOUD_HOST}:${remote_path}/" \
            "${local_path}/" 2>/dev/null || {
            log_info "${name} not found on remote or sync failed (this is OK if training hasn't started)"
            return 0
        }
    fi
}

# Check if remote is accessible
log_info "Checking connection to ${CLOUD_HOST}..."
ssh ${SSH_OPTS} "${CLOUD_USER}@${CLOUD_HOST}" "echo 'Connection successful'" >/dev/null 2>&1 || {
    echo "Error: Cannot connect to ${CLOUD_HOST}"
    echo "Make sure the instance is running and SSH is accessible"
    exit 1
}

# Sync logs
sync_directory "$REMOTE_LOGS" "$LOCAL_LOGS" "logs"

# Sync checkpoints (only new/modified files to save bandwidth)
sync_directory "$REMOTE_OUT" "$LOCAL_OUT" "checkpoints"

log_info "Sync complete!"
log_info "Latest logs:"
ls -lht "$LOCAL_LOGS"/*.csv 2>/dev/null | head -5 || echo "  (no log files yet)"
log_info "Latest checkpoints:"
ls -lht "$LOCAL_OUT"/*.pt 2>/dev/null | head -5 || echo "  (no checkpoints yet)"

