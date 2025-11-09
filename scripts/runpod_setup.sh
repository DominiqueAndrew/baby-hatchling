#!/usr/bin/env bash
# Streamlined RunPod setup script for Baby-Hatchling
# This script helps you configure RunPod quickly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

echo "=========================================="
echo "  RunPod Setup for Baby-Hatchling"
echo "=========================================="
echo ""

# Step 1: Instructions
log_step "Step 1: Create a RunPod Pod"
echo ""
echo "1. Go to https://www.runpod.io/ and sign up/login"
echo "2. Navigate to 'Pods' → 'Deploy Pod'"
echo "3. Select a GPU template:"
echo "   - For budget: RTX 3090 (Community Cloud, ~$0.29/hr)"
echo "   - For speed: A100 40GB (~$1.19/hr) or A100 80GB (~$1.29/hr)"
echo "   - For latest: H100 80GB (~$1.99/hr)"
echo "4. Choose template: 'RunPod PyTorch' or 'RunPod PyTorch 2.0'"
echo "5. Configure:"
echo "   - Container Disk: 20GB+ (for your code and data)"
echo "   - Volume Disk: Optional (for persistent storage)"
echo "6. Deploy the pod"
echo ""
read -p "Press Enter when your pod is deployed..."

# Step 2: Get connection details
log_step "Step 2: Get SSH Connection Details"
echo ""
echo "In RunPod dashboard, find your pod and click on it."
echo "Look for 'SSH' or 'Connect' section to get:"
echo "  - Host/IP address"
echo "  - Port (usually 22, but can be different)"
echo "  - Username (usually 'root')"
echo ""

read -p "Enter RunPod host/IP: " RUNPOD_HOST
read -p "Enter SSH port (default 22): " RUNPOD_PORT
RUNPOD_PORT=${RUNPOD_PORT:-22}
read -p "Enter username (default root): " RUNPOD_USER
RUNPOD_USER=${RUNPOD_USER:-root}

# Step 3: Test SSH connection
log_step "Step 3: Testing SSH Connection"
echo ""
echo "You'll need to set up SSH access. RunPod typically provides:"
echo "  - A web-based terminal (no SSH key needed)"
echo "  - Or SSH with a key they provide"
echo ""

read -p "Do you have an SSH key from RunPod? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter path to SSH key (e.g., ~/.ssh/runpod_key): " SSH_KEY
    SSH_KEY=${SSH_KEY:-~/.ssh/runpod_key}
    
    # Test connection
    log_info "Testing SSH connection..."
    SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    [ "$RUNPOD_PORT" != "22" ] && SSH_OPTS="${SSH_OPTS} -p ${RUNPOD_PORT}"
    SSH_OPTS="${SSH_OPTS} -i ${SSH_KEY}"
    
    if ssh ${SSH_OPTS} "${RUNPOD_USER}@${RUNPOD_HOST}" "echo 'Connection successful'" 2>/dev/null; then
        log_info "✓ SSH connection works!"
    else
        log_warn "SSH connection failed. You can still use RunPod's web terminal."
        log_warn "We'll set up the config, but you'll need to manually deploy code."
        SSH_KEY=""
    fi
else
    log_info "No SSH key - you can use RunPod's web terminal instead"
    SSH_KEY=""
    read -p "Enter path to SSH key if you want to set it up later (or press Enter): " SSH_KEY
    SSH_KEY=${SSH_KEY:-""}
fi

# Step 4: Save configuration
log_step "Step 4: Saving Configuration"
echo "export CLOUD_HOST=${RUNPOD_HOST}" > "$PROJECT_ROOT/.cloud_env"
echo "export CLOUD_USER=${RUNPOD_USER}" >> "$PROJECT_ROOT/.cloud_env"
[ -n "$SSH_KEY" ] && echo "export CLOUD_KEY=${SSH_KEY}" >> "$PROJECT_ROOT/.cloud_env"
[ "$RUNPOD_PORT" != "22" ] && echo "export CLOUD_PORT=${RUNPOD_PORT}" >> "$PROJECT_ROOT/.cloud_env"
echo "export CLOUD_PROVIDER=runpod" >> "$PROJECT_ROOT/.cloud_env"

log_info "Configuration saved to .cloud_env"

# Step 5: Deploy code
log_step "Step 5: Deploy Code to RunPod"
echo ""
read -p "Deploy code to RunPod now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -z "$SSH_KEY" ]; then
        log_warn "No SSH key configured. Manual deployment instructions:"
        echo ""
        echo "1. Open RunPod web terminal"
        echo "2. Run: git clone <your-repo-url> baby-hatchling"
        echo "   OR upload files via RunPod's file manager"
        echo "3. cd baby-hatchling"
        echo "4. python3 -m venv .venv && source .venv/bin/activate"
        echo "5. pip install -r requirements.txt"
        echo "6. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        echo "7. pip install faiss-gpu  # Replace faiss-cpu"
        echo ""
    else
        log_info "Deploying code via SSH..."
        
        SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        [ "$RUNPOD_PORT" != "22" ] && SSH_OPTS="${SSH_OPTS} -p ${RUNPOD_PORT}"
        SSH_OPTS="${SSH_OPTS} -i ${SSH_KEY}"
        
        RSYNC_OPTS="-avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='out/*.pt' --exclude='logs/*.csv'"
        [ "$RUNPOD_PORT" != "22" ] && RSYNC_OPTS="${RSYNC_OPTS} -e 'ssh -p ${RUNPOD_PORT} -i ${SSH_KEY}'"
        [ -n "$SSH_KEY" ] && [ "$RUNPOD_PORT" = "22" ] && RSYNC_OPTS="${RSYNC_OPTS} -e 'ssh -i ${SSH_KEY}'"
        
        cd "$PROJECT_ROOT"
        eval rsync ${RSYNC_OPTS} ./ "${RUNPOD_USER}@${RUNPOD_HOST}:~/baby-hatchling/"
        
        log_info "Code deployed! Setting up environment..."
        
        ssh ${SSH_OPTS} "${RUNPOD_USER}@${RUNPOD_HOST}" << 'ENDSSH'
            cd ~/baby-hatchling
            if [ ! -d ".venv" ]; then
                python3 -m venv .venv
            fi
            source .venv/bin/activate
            pip install --upgrade pip
            pip install -r requirements.txt
            # Install GPU PyTorch
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            # Replace faiss-cpu with GPU version
            pip uninstall -y faiss-cpu 2>/dev/null || true
            pip install faiss-gpu
            echo "✓ Environment ready!"
ENDSSH
        
        log_info "Deployment complete!"
    fi
else
    log_info "Skipping deployment. You can deploy later with:"
    echo "  bash scripts/cloud_setup.sh runpod"
fi

# Step 6: Next steps
echo ""
log_step "Next Steps"
echo ""
echo "1. Start training:"
echo "   bash scripts/cloud_train.sh pretrain configs/hatchling_xs.yaml out/xs_pretrain.pt"
echo ""
echo "2. Monitor training (in another terminal):"
echo "   bash scripts/cloud_monitor.sh watch"
echo ""
echo "3. Auto-sync results (in another terminal):"
echo "   bash scripts/cloud_autosync.sh"
echo ""
echo "4. Check GPU usage:"
echo "   bash scripts/cloud_monitor.sh gpu"
echo ""
log_info "Setup complete! Happy training! 🚀"

