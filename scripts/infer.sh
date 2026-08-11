#!/bin/bash
# Generate music from an image (+ optional text prompt)
#
# Usage examples:
#   # Image only (auto-detects latest checkpoint)
#   bash src/scripts/infer.sh --image photo.jpg
#
#   # Image + text prompt
#   bash src/scripts/infer.sh --image photo.jpg --text "calm ambient piano with soft strings"
#
#   # Image + text + duration + specific checkpoint
#   bash src/scripts/infer.sh --image photo.jpg \
#       --text "epic orchestral celebration" \
#       --duration 60 \
#       --checkpoint ckpts/it_a/step_3000
#
#   # Full example
#   bash src/scripts/infer.sh --image ronaldo.jpg \
#       --text "triumphant celebration with brass and drums" \
#       --duration 30 \
#       --output victory.wav
#
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$(pwd)/src/DiffRhythm:$(pwd):$PYTHONPATH"

python src/image_cond/infer_image.py "$@"
