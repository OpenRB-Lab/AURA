#!/bin/bash
# Stage 4: join manifests + QA into dataset.jsonl
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:$(pwd):$PYTHONPATH"
conda run -n llama python src/data_utils/build_edit_dataset.py "$@"
