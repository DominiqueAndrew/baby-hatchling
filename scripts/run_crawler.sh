#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/crawler_english.yaml}
python -m src.crawler.pipeline --config "$CONFIG"
