#!/bin/bash
# Stage B.2: LoRA SFT of Qwen2.5-Omni thinker on edit dialogues.
# Pick GPU via SFT_GPU (default 2).
set -e
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${SFT_GPU:-2}"
export HF_HUB_CACHE="$(pwd)/weights"
conda run --no-capture-output -n llama python -u src/edit_agent/train_sft.py "$@"
