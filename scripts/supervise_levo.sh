#!/bin/bash
# Supervisor for one LeVo generation shard: relaunches on external kills.
# Usage: GPU=0 SHARD=0 bash src/scripts/supervise_levo.sh
set -u
cd /home/mamba/ML_project/Testing/Huy/diffusion/baselines/LeVo
G=${GPU:?}; S=${SHARD:?}; R=${RUN:-}   # RUN e.g. "2" -> out_impg2_s0 dirs

count() { find "$1/audios" -name "*.flac" 2>/dev/null | grep -v "_vocal\|_bgm" | wc -l; }

for attempt in $(seq 1 30); do
  n_impg=$(count out_impg${R}_s$S)
  want_impg=$(wc -l < ../../results/impg_bench/levo_input_s$S.jsonl)
  if [ "$n_impg" -lt "$want_impg" ]; then
    echo "[sup g$G] attempt $attempt impg $n_impg/$want_impg $(date '+%F %H:%M')" >> ../../logs/levo_sup_g$G.log
    CUDA_VISIBLE_DEVICES=$G conda run --no-capture-output -n levo \
      bash generate.sh songgeneration_v2_medium \
      ../../results/impg_bench/levo_input_s$S.jsonl out_impg${R}_s$S --bgm \
      >> ../../logs/levo_impg_s$S.log 2>&1
    sleep 20; continue
  fi
  n_moi=$(count out_moises${R}_s$S)
  want_moi=$(wc -l < ../../results/moises_bench/levo_input_s$S.jsonl)
  if [ "$n_moi" -lt "$want_moi" ]; then
    echo "[sup g$G] attempt $attempt moises $n_moi/$want_moi $(date '+%F %H:%M')" >> ../../logs/levo_sup_g$G.log
    CUDA_VISIBLE_DEVICES=$G conda run --no-capture-output -n levo \
      bash generate.sh songgeneration_v2_medium \
      ../../results/moises_bench/levo_input_s$S.jsonl out_moises${R}_s$S --bgm \
      >> ../../logs/levo_moises_s$S.log 2>&1
    sleep 20; continue
  fi
  echo "SHARD_${S}_COMPLETE" > ../../logs/levo_g$G.status
  break
done
