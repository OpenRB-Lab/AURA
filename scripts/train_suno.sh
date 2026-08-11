#!/bin/bash
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$(pwd)/src/DiffRhythm:$(pwd):$PYTHONPATH"

torchrun --nproc_per_node=2 \
    src/image_cond/train.py \
    --config src/configs/image_cond.yaml \
    --manifest data/image_music/cached_latents/index_suno.jsonl \
    --checkpoint-dir ckpts/it_d_suno
