#!/usr/bin/env bash
# Check disk and volume usage on RunPod
set -euo pipefail

echo "💾 Disk Usage Analysis"
echo "======================"
echo ""
echo "Root disk (/):"
df -h / | tail -1
echo ""
echo "Checking for volumes..."
echo ""

# Check common volume mount points
for vol_dir in /workspace /runpod-volume /mnt/workspace /mnt/volume; do
    if [ -d "$vol_dir" ]; then
        echo "✅ Found volume: $vol_dir"
        df -h "$vol_dir" | tail -1
        echo "   Contents:"
        ls -lh "$vol_dir" 2>/dev/null | head -10 || echo "   (empty or not accessible)"
        echo ""
    fi
done

echo "📁 Current directory: $(pwd)"
echo "📊 Size of current directory:"
du -sh . 2>/dev/null || echo "   (cannot calculate)"
echo ""
echo "📦 Largest directories in current location:"
du -h --max-depth=1 . 2>/dev/null | sort -hr | head -10


