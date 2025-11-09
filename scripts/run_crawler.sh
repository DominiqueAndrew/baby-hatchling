#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/crawler_english.yaml}
echo "Starting crawler with config: $CONFIG"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"
echo "Running: python -m src.crawler.pipeline --config \"$CONFIG\""
python -m src.crawler.pipeline --config "$CONFIG" || {
    echo "Error: Crawler failed with exit code $?"
    exit 1
}
echo "Crawler completed successfully"
