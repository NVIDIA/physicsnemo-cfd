# For more info on this NIM or how to obtain an NGC API key, see:
# https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/domino-automotive-aero
export NGC_API_KEY  # Note: You must assign your API key to this environment variable (not shown here)
# Modify this to something like: `export NGC_API_KEY="..."`
docker run \
    --rm \
    --runtime=nvidia \
    --gpus 1 \
    --shm-size 2g \
    -e NGC_API_KEY \
    -t nvcr.io/nim/nvidia/domino-automotive-aero:2.0.0
