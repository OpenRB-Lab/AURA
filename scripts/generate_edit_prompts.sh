#!/bin/bash
# Stage 2: Qwen (vLLM :9003) chunk descriptions + edit prompts
set -e
cd "$(dirname "$0")/../.."
conda run -n llama python src/data_utils/generate_edit_prompts.py --workers 16 "$@"
