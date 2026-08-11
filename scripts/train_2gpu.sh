#!/bin/bash
# Train image-conditioned DiffRhythm with LoRA on 2 GPUs via torchrun DDP
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$(pwd)/src/DiffRhythm:$(pwd):$PYTHONPATH"

torchrun --nproc_per_node=2 \
    src/image_cond/train.py \
    --config src/configs/image_cond.yaml
