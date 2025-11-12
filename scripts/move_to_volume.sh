#!/usr/bin/env bash
# Move project to RunPod volume to avoid disk space issues
set -euo pipefail

# Find the volume mount point (RunPod typically uses /workspace)
VOLUME_DIR="/workspace"
if [ ! -d "$VOLUME_DIR" ]; then
    # Try alternative locations
    if [ -d "/runpod-volume" ]; then
        VOLUME_DIR="/runpod-volume"
    elif [ -d "/mnt/workspace" ]; then
        VOLUME_DIR="/mnt/workspace"
    else
        echo "❌ Could not find volume mount point!"
        echo "   Checked: /workspace, /runpod-volume, /mnt/workspace"
        echo "   Please find your volume mount point and update this script"
        exit 1
    fi
fi

echo "📦 Found volume at: $VOLUME_DIR"
echo "💾 Current disk usage:"
df -h / | tail -1
echo "💾 Volume usage:"
df -h "$VOLUME_DIR" | tail -1
echo ""

# Check if already on volume
CURRENT_DIR=$(pwd)
if [[ "$CURRENT_DIR" == "$VOLUME_DIR"* ]]; then
    echo "✅ Already on volume! No need to move."
    exit 0
fi

PROJECT_NAME="baby-hatchling"
VOLUME_PROJECT="$VOLUME_DIR/$PROJECT_NAME"

echo "🚀 Moving project to volume..."
echo "   From: $CURRENT_DIR"
echo "   To:   $VOLUME_PROJECT"
echo ""

# Create volume project directory
mkdir -p "$VOLUME_PROJECT"

# Copy project (excluding large files that can be regenerated)
echo "📋 Copying project files..."
rsync -av --progress \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='out/' \
    --exclude='logs/' \
    --exclude='data/episodic.db*' \
    --exclude='.git/' \
    "$CURRENT_DIR/" "$VOLUME_PROJECT/"

# Move large directories that need to be on volume
echo "📦 Moving large directories to volume..."
[ -d "$CURRENT_DIR/.venv" ] && mv "$CURRENT_DIR/.venv" "$VOLUME_PROJECT/" 2>/dev/null || true
[ -d "$CURRENT_DIR/out" ] && mv "$CURRENT_DIR/out" "$VOLUME_PROJECT/" 2>/dev/null || true
[ -d "$CURRENT_DIR/logs" ] && mv "$CURRENT_DIR/logs" "$VOLUME_PROJECT/" 2>/dev/null || true
[ -d "$CURRENT_DIR/data" ] && mv "$CURRENT_DIR/data" "$VOLUME_PROJECT/" 2>/dev/null || true

# Create symlinks back to original location for convenience
echo "🔗 Creating symlinks..."
cd "$(dirname "$CURRENT_DIR")"
ln -sfn "$VOLUME_PROJECT" "$PROJECT_NAME" 2>/dev/null || true

echo ""
echo "✅ Migration complete!"
echo ""
echo "📝 Next steps:"
echo "   1. cd $VOLUME_PROJECT"
echo "   2. Recreate virtual environment:"
echo "      python -m venv .venv && source .venv/bin/activate"
echo "      pip install -r requirements.txt"
echo "   3. Continue training from the volume location"
echo ""
echo "💾 New disk usage:"
df -h / | tail -1
df -h "$VOLUME_DIR" | tail -1


