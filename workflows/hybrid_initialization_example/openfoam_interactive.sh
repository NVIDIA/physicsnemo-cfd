#!/bin/bash

set -e  # Exit on error

IMAGE_NAME="openfoam-python:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_DIR="${SCRIPT_DIR}/container"

# Check if the image exists locally
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "Docker image '${IMAGE_NAME}' not found locally."
    read -p "Would you like to build it now? (this may take several hours) [y/N]: " RESP
    if [[ "$RESP" =~ ^[Yy]$ ]]; then
        echo "Building the container..."
        cd "${CONTAINER_DIR}"
        make container
        cd "${SCRIPT_DIR}"
        echo "Container build complete."
    else
        echo "Aborting. Please build the container before running this script."
        exit 1
    fi
fi

docker run \
    --rm \
    --runtime=nvidia \
    --gpus 1 \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${PWD}:/workspace" \
    -w /workspace \
    -it \
    ${IMAGE_NAME}
