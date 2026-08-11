#!/bin/bash
# Edit-agent web demo: MelodyFlow worker (:7861) + Gradio app (:7860), both on GPU 2.
set -e
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${WEBAPP_GPU:-2}"
export HF_HUB_CACHE="$(pwd)/weights"
export AUDIOCRAFT_CACHE_DIR="$(pwd)/weights"

conda run --no-capture-output -n melodyflow python -u src/edit_agent/melodyflow_worker.py \
  > logs/melodyflow_worker.log 2>&1 &
WORKER_PID=$!
trap "kill $WORKER_PID 2>/dev/null" EXIT

echo "waiting for melodyflow worker..."
for i in $(seq 1 60); do
  curl -fsS -m 2 http://127.0.0.1:7861/health >/dev/null 2>&1 && break
  sleep 5
done
curl -fsS http://127.0.0.1:7861/health || { echo "worker failed"; exit 1; }
echo "worker ready — starting gradio"

conda run --no-capture-output -n llama python -u src/edit_agent/webapp.py
