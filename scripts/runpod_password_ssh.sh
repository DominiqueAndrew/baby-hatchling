#!/usr/bin/env bash
# Connect to RunPod using password authentication
# RunPod may provide a password in the pod details or template readme

source "$(dirname "$0")/../.cloud_env" 2>/dev/null || {
    echo "Error: .cloud_env not found"
    exit 1
}

echo "Connecting to RunPod with password authentication..."
echo ""
echo "If you don't know the password, check:"
echo "  1. RunPod dashboard → Your pod → 'Template Readme' tab"
echo "  2. RunPod dashboard → Your pod → 'Details' tab"
echo "  3. The email/notification when you created the pod"
echo ""

# Try to connect - it will prompt for password
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -p "${CLOUD_PORT}" \
    "${CLOUD_USER}@${CLOUD_HOST}"


