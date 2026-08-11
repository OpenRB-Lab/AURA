#!/bin/bash
# Precompute cached latents for image-conditioned training
cd "$(dirname "$0")/../.."
conda run -n llama --no-banner python src/data_utils/prepare_data.py \
    --config src/configs/image_cond.yaml \
    --device cuda
