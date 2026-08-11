#!/bin/bash
# After a joint training run: emission verification -> live-hidden benchmark -> FAD.
# Usage: final_eval_chain.sh <joint_log> <out_dir>
set -u
cd /home/mamba/ML_project/Testing/Huy/diffusion
LOG=${1:?joint log}
OUT=${2:?out dir}
until grep -q "done:" "$LOG"; do sleep 30; done
sleep 15

echo "=== verify emission (joint qwen) ==="
conda run --no-capture-output -n llama python -u src/edit_agent/verify_sft.py \
    --adapter ckpts/edit_agent/joint/final/qwen

echo "=== benchmark (live joint hidden -> ${ARCH:-kv} bridge, anchored) ==="
conda run --no-capture-output -n llama python -u src/edit_agent/eval_musicgen.py \
    --arch "${ARCH:-kv}" --qwen ckpts/edit_agent/joint/final/qwen \
    --mg ckpts/edit_agent/joint/final/musicgen \
    --n-seg 20 --n-global 40 --n-triplets 999 --n-ablate 8 --guidance 2.0 \
    --out "$OUT"

echo "=== FAD ==="
cd "results_tmp_unused" 2>/dev/null; cd /home/mamba/ML_project/Testing/Huy/diffusion
mkdir -p "$OUT/GEN" "$OUT/TARGET"
for f in "$OUT"/triplets/*_GEN.wav; do cp "$f" "$OUT/GEN/"; done
for f in "$OUT"/triplets/*_TARGET.wav; do cp "$f" "$OUT/TARGET/"; done
conda run --no-capture-output -n llama python -u - <<EOF
import sys, torch
sys.path.insert(0,'src')
torch.backends.cudnn.enabled = False
from evaluation.fad import compute_fad
print(f"FAD(GEN,TARGET) = {compute_fad('$OUT/GEN', '$OUT/TARGET', verbose=False):.3f}")
EOF
echo "=== final chain done ==="
