#!/bin/bash
# Stage 1: beat-aligned chunking of suno + painting audio (CPU)
set -e
cd "$(dirname "$0")/../.."
conda run -n llama python src/data_utils/chunk_audio.py "$@"
