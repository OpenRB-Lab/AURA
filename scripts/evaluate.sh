#!/bin/bash
# Run FAD + CLAP score evaluation
# Usage: bash src/scripts/evaluate.sh --generated results/cross_attn_mod/latest/ --reference data/image_music/music_dataset/
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src/DiffRhythm:$(pwd):$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python src/evaluation/run_benchmark.py "$@"
