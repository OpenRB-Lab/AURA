#!/bin/bash
# Stage 3: MelodyFlow editing (GPU). Pick GPU via MELODYFLOW_DEVICE.
set -e
cd "$(dirname "$0")/../.."
export MELODYFLOW_DEVICE="${MELODYFLOW_DEVICE:-cuda:0}"
# MelodyFlow weights are pre-downloaded into weights/ (HF hub cache layout)
export AUDIOCRAFT_CACHE_DIR="$(pwd)/weights"
export HF_HUB_CACHE="$(pwd)/weights"
conda run -n melodyflow python src/data_utils/run_melodyflow_edit.py "$@"
