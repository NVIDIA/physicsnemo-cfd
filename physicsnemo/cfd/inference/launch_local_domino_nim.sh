#!/bin/bash

# For more info on this NIM or how to obtain an NGC API key, see:
# https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/domino-automotive-aero

# Check that NGC_API_KEY is set and non-empty
if [[ -z "${NGC_API_KEY}" ]]; then
  echo "ERROR: The environment variable NGC_API_KEY is not set, or it was not exported to this shell."
  echo ""
  echo "To use this script, you must obtain an NGC API key from NVIDIA GPU Cloud (NGC)."
  echo "Visit: https://ngc.nvidia.com/setup/api-key"
  echo ""
  echo "A NGC account is required to obtain an API key; the link above will allow you to register for one."
  echo ""
  echo "Once you have your API key, set and export it in your shell, and re-run this script:"
  echo '  export NGC_API_KEY="YOUR_NGC_API_KEY_HERE"'
  echo '  ./launch_local_domino_nim.sh'
  echo ""
  exit 1
fi

DOCKER_CONFIG="${HOME}/.docker/config.json"
if ! grep -q '"nvcr.io"' "$DOCKER_CONFIG" 2>/dev/null; then
  echo "Not yet authenticated to nvcr.io, attempting login using NGC API key..."
  echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
else
  echo "Already authenticated to nvcr.io (found in ${DOCKER_CONFIG})."
fi

# Check if the port is already in use by a running Docker container
PORT_TO_USE=8000
CONTAINER_ID_USING_PORT=$(docker ps --filter "publish=${PORT_TO_USE}" --format "{{.ID}}")

if [[ -n "$CONTAINER_ID_USING_PORT" ]]; then
  echo "WARNING: TCP port ${PORT_TO_USE} is already in use by the following Docker container(s):"
  docker ps --filter "publish=${PORT_TO_USE}" --format "  ID: {{.ID}}  Image: {{.Image}}  Status: {{.Status}}"
  echo ""
  read -p "Do you want to stop the container(s) using port ${PORT_TO_USE}, to allow launch? [y/N]: " RESP
  if [[ "$RESP" =~ ^[Yy]$ ]]; then
    echo "Stopping container(s) using port ${PORT_TO_USE}..."
    docker stop $CONTAINER_ID_USING_PORT
    echo "Container(s) stopped."
  else
    echo "Aborting launch. Please free up port ${PORT_TO_USE} and try again."
    exit 3
  fi
fi

echo "Launching Domino NIM container..."

docker run \
    --rm \
    --runtime=nvidia \
    --gpus 1 \
    --shm-size 2g \
    -p ${PORT_TO_USE}:${PORT_TO_USE} \
    -e NGC_API_KEY \
    -t nvcr.io/nim/nvidia/domino-automotive-aero:1.0.0
