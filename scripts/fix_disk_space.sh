#!/usr/bin/env bash
# Emergency disk space fix for RunPod
set -euo pipefail

echo "🚨 Emergency Disk Space Cleanup"
echo "================================"
echo ""

# Check current usage
echo "📊 Current disk usage:"
df -h
echo ""

# 1. Clean pip cache (can free 1-5GB)
echo "🧹 Step 1: Cleaning pip cache..."
pip cache purge 2>/dev/null || true
echo "   ✓ Pip cache cleared"
echo ""

# 2. Remove Python bytecode
echo "🧹 Step 2: Removing Python cache..."
find /baby-hatchling -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find /baby-hatchling -type f -name "*.pyc" -delete 2>/dev/null || true
echo "   ✓ Python cache cleared"
echo ""

# 3. Clean system temp
echo "🧹 Step 3: Cleaning system temp..."
rm -rf /tmp/* 2>/dev/null || true
rm -rf /var/tmp/* 2>/dev/null || true
echo "   ✓ Temp files cleared"
echo ""

# 4. Clean apt cache (if root)
if [ "$EUID" -eq 0 ]; then
    echo "🧹 Step 4: Cleaning apt cache..."
    apt-get clean 2>/dev/null || true
    apt-get autoclean 2>/dev/null || true
    echo "   ✓ Apt cache cleared"
    echo ""
fi

# 5. Check for large files
echo "🔍 Step 5: Finding large files (>100MB)..."
find /baby-hatchling -type f -size +100M 2>/dev/null | head -10 || echo "   No large files found"
echo ""

# Final check
echo "📊 Updated disk usage:"
df -h
echo ""

echo "✅ Cleanup complete!"
echo ""
echo "💡 If still low on space, try:"
echo "   1. Remove .venv and reinstall: rm -rf .venv && python -m venv .venv"
echo "   2. Install minimal requirements: pip install -r requirements_minimal.txt"
echo "   3. Use --no-cache-dir: pip install --no-cache-dir -r requirements.txt"

