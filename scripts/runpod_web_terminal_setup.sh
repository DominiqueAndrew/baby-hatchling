#!/usr/bin/env bash
# Copy-paste this entire script into RunPod's Web Terminal
# This sets up Baby-Hatchling on your RunPod instance

set -e

echo "=========================================="
echo "  Baby-Hatchling RunPod Setup"
echo "=========================================="
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi || echo "Warning: GPU not detected"

# Create project directory
echo ""
echo "Setting up project directory..."
cd ~
if [ -d "baby-hatchling" ]; then
    echo "Directory exists, updating..."
    cd baby-hatchling
    git pull 2>/dev/null || echo "Not a git repo, continuing..."
else
    echo "Creating directory..."
    mkdir -p baby-hatchling
    cd baby-hatchling
fi

# You'll need to upload your code or clone from git
# For now, we'll set up the environment
echo ""
echo "Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas datasets evaluate transformers sentencepiece faiss-gpu sqlite-utils pyyaml datasketch tiktoken tqdm pytest rich tabulate

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your code files to ~/baby-hatchling/"
echo "   (Use RunPod's file manager or git clone)"
echo ""
echo "2. Start training:"
echo "   cd ~/baby-hatchling"
echo "   source .venv/bin/activate"
echo "   python -m src.trainer --config configs/hn_xs.yaml --stage pretrain --save out/hn_xs_pretrain.pt"
echo ""
