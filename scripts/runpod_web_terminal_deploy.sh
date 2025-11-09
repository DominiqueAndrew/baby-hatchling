#!/usr/bin/env bash
# Instructions for deploying via RunPod Web Terminal
# Copy-paste these commands into RunPod's Web Terminal

cat << 'EOF'
==========================================
  Baby-Hatchling Deployment via Web Terminal
==========================================

STEP 1: Set up environment
---------------------------
cd ~
mkdir -p baby-hatchling
cd baby-hatchling

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas datasets evaluate transformers sentencepiece faiss-gpu sqlite-utils pyyaml datasketch tiktoken tqdm pytest rich tabulate

# Check GPU
nvidia-smi

STEP 2: Upload your code
-------------------------
Option A: Using RunPod File Manager
1. Click on your pod → "Files" tab
2. Navigate to /root/baby-hatchling
3. Upload your project files

Option B: Using git (if your repo is public)
git clone <your-repo-url> .

Option C: Using wget/curl (if you have a zip file)
wget <your-zip-url>
unzip <file.zip> -d .

STEP 3: Verify setup
---------------------
cd ~/baby-hatchling
source .venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

STEP 4: Start training
-----------------------
cd ~/baby-hatchling
source .venv/bin/activate

# Pretrain
python -m src.trainer --config configs/hn_xs.yaml --stage pretrain --save out/hn_xs_pretrain.pt

# Or run in background
nohup python -m src.trainer --config configs/hn_xs.yaml --stage pretrain --save out/hn_xs_pretrain.pt > training.log 2>&1 &
echo $! > training.pid

# Monitor training
tail -f training.log

# Check GPU usage
watch -n 1 nvidia-smi

==========================================
EOF
