#!/usr/bin/env bash
# Test SSH connection to RunPod
# Run this after adding your SSH key to RunPod

source "$(dirname "$0")/../.cloud_env" 2>/dev/null || {
    echo "Error: .cloud_env not found"
    exit 1
}

echo "Testing SSH connection to ${CLOUD_HOST}:${CLOUD_PORT}..."
echo ""
echo "If it asks for a password, the SSH key may not be set up correctly."
echo "Options:"
echo "  1. Make sure you added the public key in RunPod's SSH Key Setup section"
echo "  2. Use RunPod's Web Terminal instead (no SSH needed)"
echo "  3. Check if RunPod provided a password for this pod"
echo ""

# Try with verbose output to see what's happening
ssh -v -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -p "${CLOUD_PORT}" -i "${CLOUD_KEY}" \
    "${CLOUD_USER}@${CLOUD_HOST}" "echo '✓ SSH connection successful!' && nvidia-smi" 2>&1 | head -20

