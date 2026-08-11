#!/bin/bash
# Download and precompute Suno dataset embeddings
# Appends to existing cached_latents
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src/DiffRhythm:$(pwd):$PYTHONPATH"

CUDA_VISIBLE_DEVICES=1 python src/data_utils/prepare_suno.py \
    --config src/configs/image_cond.yaml \
    --device cuda \
    "$@"
