#!/bin/bash
# Generate music prompts for all dataset entries using Gemma
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src/DiffRhythm:$(pwd):$PYTHONPATH"

python src/data_utils/generate_prompts.py \
    --config src/configs/image_cond.yaml \
    "$@"
