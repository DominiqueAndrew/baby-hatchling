#!/usr/bin/env bash
# Disk cleanup script for RunPod instances
set -euo pipefail

echo "🔍 Checking disk usage..."
df -h

echo ""
echo "🧹 Cleaning up..."

# Clean pip cache
echo "  • Cleaning pip cache..."
pip cache purge 2>/dev/null || true

# Clean Python cache
echo "  • Cleaning Python __pycache__..."
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Clean old checkpoints (keep only latest)
echo "  • Cleaning old checkpoints..."
if [ -d "out" ]; then
    # Keep only the 3 most recent checkpoints
    ls -t out/*.pt 2>/dev/null | tail -n +4 | xargs rm -f 2>/dev/null || true
fi

# Clean logs (keep only recent)
echo "  • Cleaning old logs..."
if [ -d "logs" ]; then
    # Keep last 1000 lines of each log
    for log in logs/*.txt logs/*.csv; do
        if [ -f "$log" ] && [ $(wc -l < "$log") -gt 1000 ]; then
            tail -1000 "$log" > "${log}.tmp" && mv "${log}.tmp" "$log"
        fi
    done
fi

# Clean temporary files
echo "  • Cleaning temp files..."
rm -rf /tmp/* 2>/dev/null || true
rm -rf /var/tmp/* 2>/dev/null || true

# Clean apt cache if root
if [ "$EUID" -eq 0 ]; then
    echo "  • Cleaning apt cache..."
    apt-get clean 2>/dev/null || true
    apt-get autoclean 2>/dev/null || true
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 Updated disk usage:"
df -h

