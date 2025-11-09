#!/usr/bin/env bash
# Real-time monitoring of cloud training
# Streams logs and metrics from the remote instance

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$PROJECT_ROOT/.cloud_env" ]; then
    echo "Error: .cloud_env not found. Run scripts/cloud_setup.sh first."
    exit 1
fi

source "$PROJECT_ROOT/.cloud_env"

# Build SSH options
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
[ -n "${CLOUD_PORT:-}" ] && SSH_OPTS="${SSH_OPTS} -p ${CLOUD_PORT}"
[ -n "${CLOUD_KEY:-}" ] && SSH_OPTS="${SSH_OPTS} -i ${CLOUD_KEY}"

# Function to display latest metrics
show_metrics() {
    local log_file=$1
    if [ -f "$log_file" ]; then
        echo ""
        echo "=== Latest Training Metrics ==="
        tail -1 "$log_file" 2>/dev/null | while IFS=',' read -r line; do
            echo "$line"
        done
        echo ""
    fi
}

# Function to watch training log
watch_training() {
    echo "Monitoring training on ${CLOUD_HOST}..."
    echo "Press Ctrl+C to stop"
    echo ""
    
    ssh ${SSH_OPTS} "${CLOUD_USER}@${CLOUD_HOST}" "tail -f ~/baby-hatchling/training.log" &
    SSH_PID=$!
    
    # Also sync periodically in background
    (
        while kill -0 $SSH_PID 2>/dev/null; do
            sleep 30
            bash "$SCRIPT_DIR/cloud_sync.sh" >/dev/null 2>&1
        done
    ) &
    SYNC_PID=$!
    
    # Cleanup on exit
    trap "kill $SSH_PID $SYNC_PID 2>/dev/null; exit" INT TERM
    
    wait $SSH_PID
}

# Function to show GPU usage
show_gpu() {
    echo "=== GPU Status ==="
    ssh ${SSH_OPTS} "${CLOUD_USER}@${CLOUD_HOST}" "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits" 2>/dev/null || {
        echo "GPU not available or nvidia-smi not found"
    }
    echo ""
}

# Function to show latest CSV metrics
show_csv_metrics() {
    echo "=== Latest CSV Metrics ==="
    ssh ${SSH_OPTS} "${CLOUD_USER}@${CLOUD_HOST}" << 'ENDSSH'
        cd ~/baby-hatchling/logs
        for csv in *.csv; do
            if [ -f "$csv" ]; then
                echo "--- $csv ---"
                tail -3 "$csv" 2>/dev/null || echo "  (empty)"
                echo ""
            fi
        done
ENDSSH
}

# Main menu
case "${1:-watch}" in
    watch)
        watch_training
        ;;
    gpu)
        show_gpu
        ;;
    metrics)
        show_csv_metrics
        ;;
    all)
        while true; do
            clear
            show_gpu
            show_csv_metrics
            sleep 5
        done
        ;;
    *)
        echo "Usage: $0 [watch|gpu|metrics|all]"
        echo "  watch    - Stream training log (default)"
        echo "  gpu      - Show GPU status"
        echo "  metrics  - Show latest CSV metrics"
        echo "  all      - Show all info in a loop"
        exit 1
        ;;
esac

