#!/bin/bash
# Full baseline run: chunk -> Qwen prompts -> MelodyFlow edits (GPU 2) -> dataset build.
# Each stage is resumable; stages 2 and 3 are invoked twice so transient failures retry.
set -e
cd "$(dirname "$0")/../.."
LOGDIR="logs/baseline_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOGDIR"
echo "[baseline] logs in $LOGDIR"

echo "[baseline] stage 1: chunking all songs"
bash src/scripts/chunk_audio.sh --procs 8 > "$LOGDIR/stage1_chunk.log" 2>&1
echo "[baseline] stage 1 done: $(wc -l < data/edit_dataset/manifests/chunks.jsonl) chunks"

echo "[baseline] waiting for vLLM on :9003"
ok=""
for i in $(seq 1 240); do
  if curl -fsS -m 4 -o /dev/null http://127.0.0.1:9003/v1/models; then ok=1; break; fi
  sleep 15
done
[ -n "$ok" ] || { echo "[baseline] ERROR: vLLM never became healthy"; exit 1; }

echo "[baseline] stage 2: edit prompts (Qwen + verify)"
bash src/scripts/generate_edit_prompts.sh --verify > "$LOGDIR/stage2_prompts.log" 2>&1 || true
bash src/scripts/generate_edit_prompts.sh --verify > "$LOGDIR/stage2_prompts_retry.log" 2>&1
echo "[baseline] stage 2 done: $(wc -l < data/edit_dataset/manifests/edit_prompts.jsonl) edit prompts"

echo "[baseline] stage 3: MelodyFlow editing on GPU 2"
MELODYFLOW_DEVICE=cuda:2 bash src/scripts/run_melodyflow_edit.sh --batch-size 8 > "$LOGDIR/stage3_melodyflow.log" 2>&1 || true
MELODYFLOW_DEVICE=cuda:2 bash src/scripts/run_melodyflow_edit.sh --batch-size 8 > "$LOGDIR/stage3_melodyflow_retry.log" 2>&1
echo "[baseline] stage 3 done: $(grep -c '"status": "ok"' data/edit_dataset/manifests/edited.jsonl) edits rendered"

echo "[baseline] stage 4: dataset build + CLAP QA on GPU 2"
CUDA_VISIBLE_DEVICES=2 bash src/scripts/build_edit_dataset.sh --clap > "$LOGDIR/stage4_build.log" 2>&1
echo "[baseline] stage 4 done: $(wc -l < data/edit_dataset/manifests/dataset.jsonl) final records"
echo "[baseline] COMPLETE"
