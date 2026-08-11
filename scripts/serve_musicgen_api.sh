#!/bin/bash
# Serve the production music-edit model as an HTTP API.
# Usage: bash src/scripts/serve_musicgen_api.sh   (env: API_GPU=1 API_PORT=9004)
set -u
cd /home/mamba/ML_project/Testing/Huy/diffusion
export CUDA_VISIBLE_DEVICES=${API_GPU:-1}
export HF_HUB_CACHE=$PWD/weights
export API_PORT=${API_PORT:-9004}
exec conda run --no-capture-output -n llama python -u src/edit_agent/musicgen_api.py
