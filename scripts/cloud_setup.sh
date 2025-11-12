#!/usr/bin/env bash
# Cloud computing setup script for Baby-Hatchling
# Supports: AWS EC2, GCP Compute Engine, RunPod, Lambda Labs, and generic SSH

set -euo pipefail

PROVIDER=${1:-""}
INSTANCE_NAME=${2:-"baby-hatchling-train"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    local missing=()
    command -v ssh >/dev/null 2>&1 || missing+=("ssh")
    command -v rsync >/dev/null 2>&1 || missing+=("rsync")
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_info "Install with: brew install openssh rsync (macOS) or apt-get install openssh-client rsync (Linux)"
        exit 1
    fi
}

# Setup AWS EC2 instance
setup_aws() {
    log_info "Setting up AWS EC2 instance..."
    
    if ! command -v aws >/dev/null 2>&1; then
        log_error "AWS CLI not found. Install from: https://aws.amazon.com/cli/"
        exit 1
    fi
    
    # Check for existing instance
    INSTANCE_ID=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=${INSTANCE_NAME}" "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "")
    
    if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
        log_warn "Instance ${INSTANCE_NAME} already exists: ${INSTANCE_ID}"
        read -p "Use existing instance? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_info "Creating new EC2 instance..."
        # You'll need to customize these parameters
        INSTANCE_ID=$(aws ec2 run-instances \
            --image-id ami-0c55b159cbfafe1f0 \
            --instance-type g4dn.xlarge \
            --key-name your-key-name \
            --security-group-ids sg-xxxxxxxxx \
            --subnet-id subnet-xxxxxxxxx \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
            --query 'Instances[0].InstanceId' --output text)
        
        log_info "Waiting for instance to be running..."
        aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    fi
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    
    log_info "Instance ready at: ${PUBLIC_IP}"
    echo "export CLOUD_HOST=${PUBLIC_IP}" > .cloud_env
    echo "export CLOUD_USER=ec2-user" >> .cloud_env
    echo "export CLOUD_KEY=~/.ssh/your-key.pem" >> .cloud_env
    echo "export CLOUD_PROVIDER=aws" >> .cloud_env
    
    log_info "Configuration saved to .cloud_env"
    log_warn "Please update CLOUD_KEY in .cloud_env with your actual key path"
}

# Setup GCP Compute Engine
setup_gcp() {
    log_info "Setting up GCP Compute Engine instance..."
    
    if ! command -v gcloud >/dev/null 2>&1; then
        log_error "gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check for existing instance
    if gcloud compute instances describe "${INSTANCE_NAME}" --zone=us-central1-a >/dev/null 2>&1; then
        log_warn "Instance ${INSTANCE_NAME} already exists"
        read -p "Use existing instance? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_info "Creating new GCP instance..."
        gcloud compute instances create "${INSTANCE_NAME}" \
            --zone=us-central1-a \
            --machine-type=n1-standard-4 \
            --accelerator=type=nvidia-tesla-t4,count=1 \
            --image-family=ubuntu-2204-lts \
            --image-project=ubuntu-os-cloud \
            --maintenance-policy=TERMINATE \
            --boot-disk-size=100GB
    fi
    
    PUBLIC_IP=$(gcloud compute instances describe "${INSTANCE_NAME}" \
        --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    
    log_info "Instance ready at: ${PUBLIC_IP}"
    echo "export CLOUD_HOST=${PUBLIC_IP}" > .cloud_env
    echo "export CLOUD_USER=ubuntu" >> .cloud_env
    echo "export CLOUD_KEY=~/.ssh/google_compute_engine" >> .cloud_env
    echo "export CLOUD_PROVIDER=gcp" >> .cloud_env
    
    log_info "Configuration saved to .cloud_env"
}

# Setup RunPod
setup_runpod() {
    log_info "RunPod setup instructions:"
    echo ""
    echo "1. Go to https://www.runpod.io/"
    echo "2. Create a pod with GPU (e.g., RTX 3090, A100)"
    echo "3. Use template: RunPod PyTorch"
    echo "4. Note the SSH connection details"
    echo ""
    read -p "Enter RunPod host/IP: " RUNPOD_HOST
    read -p "Enter RunPod SSH port (default 22): " RUNPOD_PORT
    RUNPOD_PORT=${RUNPOD_PORT:-22}
    
    echo "export CLOUD_HOST=${RUNPOD_HOST}" > .cloud_env
    echo "export CLOUD_USER=root" >> .cloud_env
    echo "export CLOUD_KEY=~/.ssh/id_rsa" >> .cloud_env
    echo "export CLOUD_PORT=${RUNPOD_PORT}" >> .cloud_env
    echo "export CLOUD_PROVIDER=runpod" >> .cloud_env
    
    log_info "Configuration saved to .cloud_env"
}

# Setup Lambda Labs
setup_lambda() {
    log_info "Lambda Labs setup instructions:"
    echo ""
    echo "1. Go to https://lambdalabs.com/"
    echo "2. Launch an instance (e.g., GPU Cloud)"
    echo "3. Note the SSH connection details"
    echo ""
    read -p "Enter Lambda Labs host/IP: " LAMBDA_HOST
    read -p "Enter Lambda Labs user (default ubuntu): " LAMBDA_USER
    LAMBDA_USER=${LAMBDA_USER:-ubuntu}
    
    echo "export CLOUD_HOST=${LAMBDA_HOST}" > .cloud_env
    echo "export CLOUD_USER=${LAMBDA_USER}" >> .cloud_env
    echo "export CLOUD_KEY=~/.ssh/lambda_key" >> .cloud_env
    echo "export CLOUD_PROVIDER=lambda" >> .cloud_env
    
    log_info "Configuration saved to .cloud_env"
    log_warn "Please update CLOUD_KEY in .cloud_env with your actual key path"
}

# Generic SSH setup
setup_ssh() {
    log_info "Setting up generic SSH connection..."
    read -p "Enter host/IP: " SSH_HOST
    read -p "Enter username (default ubuntu): " SSH_USER
    SSH_USER=${SSH_USER:-ubuntu}
    read -p "Enter SSH port (default 22): " SSH_PORT
    SSH_PORT=${SSH_PORT:-22}
    read -p "Enter SSH key path (default ~/.ssh/id_rsa): " SSH_KEY
    SSH_KEY=${SSH_KEY:-~/.ssh/id_rsa}
    
    echo "export CLOUD_HOST=${SSH_HOST}" > .cloud_env
    echo "export CLOUD_USER=${SSH_USER}" >> .cloud_env
    echo "export CLOUD_KEY=${SSH_KEY}" >> .cloud_env
    echo "export CLOUD_PORT=${SSH_PORT}" >> .cloud_env
    echo "export CLOUD_PROVIDER=ssh" >> .cloud_env
    
    log_info "Configuration saved to .cloud_env"
}

# Install dependencies on remote
install_remote() {
    source .cloud_env 2>/dev/null || {
        log_error ".cloud_env not found. Run setup first."
        exit 1
    }
    
    SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    [ -n "${CLOUD_PORT:-}" ] && SSH_OPTS="${SSH_OPTS} -p ${CLOUD_PORT}"
    [ -n "${CLOUD_KEY:-}" ] && SSH_OPTS="${SSH_OPTS} -i ${CLOUD_KEY}"
    
    log_info "Installing dependencies on remote machine..."
    
    ssh ${SSH_OPTS} "${CLOUD_USER}@${CLOUD_HOST}" << 'ENDSSH'
        set -e
        # Update system
        sudo apt-get update -qq
        
        # Install Python and pip if needed
        if ! command -v python3 &> /dev/null; then
            sudo apt-get install -y python3 python3-pip python3-venv
        fi
        
        # Install CUDA if GPU instance (check if nvidia-smi exists)
        if ! command -v nvidia-smi &> /dev/null; then
            echo "GPU not detected or CUDA not installed. Installing CUDA..."
            # Add CUDA installation commands here if needed
        fi
        
        # Install git if needed
        if ! command -v git &> /dev/null; then
            sudo apt-get install -y git
        fi
        
        echo "Dependencies installed successfully"
ENDSSH
    
    log_info "Remote setup complete!"
}

# Deploy code to remote
deploy_code() {
    source .cloud_env 2>/dev/null || {
        log_error ".cloud_env not found. Run setup first."
        exit 1
    }
    
    SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    RSYNC_OPTS="-avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='out/*.pt' --exclude='logs/*.csv'"
    [ -n "${CLOUD_PORT:-}" ] && RSYNC_OPTS="${RSYNC_OPTS} -e 'ssh -p ${CLOUD_PORT} -i ${CLOUD_KEY}'"
    [ -n "${CLOUD_KEY:-}" ] && [ -z "${CLOUD_PORT:-}" ] && RSYNC_OPTS="${RSYNC_OPTS} -e 'ssh -i ${CLOUD_KEY}'"
    
    log_info "Deploying code to remote machine..."
    
    cd "$(dirname "$0")/.."
    eval rsync ${RSYNC_OPTS} ./ "${CLOUD_USER}@${CLOUD_HOST}:~/baby-hatchling/"
    
    log_info "Code deployed! Setting up remote environment..."
    
    [ -n "${CLOUD_PORT:-}" ] && SSH_OPTS="${SSH_OPTS} -p ${CLOUD_PORT}"
    [ -n "${CLOUD_KEY:-}" ] && SSH_OPTS="${SSH_OPTS} -i ${CLOUD_KEY}"
    
    ssh ${SSH_OPTS} "${CLOUD_USER}@${CLOUD_HOST}" << 'ENDSSH'
        cd ~/baby-hatchling
        if [ ! -d ".venv" ]; then
            python3 -m venv .venv
        fi
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        # Install GPU version of PyTorch if CUDA is available
        if command -v nvidia-smi &> /dev/null; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            pip install faiss-gpu  # Replace faiss-cpu with GPU version
        fi
        echo "Remote environment ready!"
ENDSSH
    
    log_info "Deployment complete!"
}

# Main menu
main() {
    check_dependencies
    
    if [ -z "$PROVIDER" ]; then
        echo "Baby-Hatchling Cloud Setup"
        echo ""
        echo "Available providers:"
        echo "  1) aws      - AWS EC2"
        echo "  2) gcp      - Google Cloud Platform"
        echo "  3) runpod   - RunPod"
        echo "  4) lambda  - Lambda Labs"
        echo "  5) ssh      - Generic SSH"
        echo ""
        read -p "Select provider (1-5): " choice
        
        case $choice in
            1) PROVIDER=aws ;;
            2) PROVIDER=gcp ;;
            3) PROVIDER=runpod ;;
            4) PROVIDER=lambda ;;
            5) PROVIDER=ssh ;;
            *) log_error "Invalid choice"; exit 1 ;;
        esac
    fi
    
    case $PROVIDER in
        aws) setup_aws ;;
        gcp) setup_gcp ;;
        runpod) setup_runpod ;;
        lambda) setup_lambda ;;
        ssh) setup_ssh ;;
        *) log_error "Unknown provider: $PROVIDER"; exit 1 ;;
    esac
    
    read -p "Install dependencies on remote? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_remote
    fi
    
    read -p "Deploy code to remote? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        deploy_code
    fi
    
    log_info "Setup complete! Use scripts/cloud_sync.sh to sync results back."
}

main "$@"


