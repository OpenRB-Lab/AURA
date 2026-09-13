#!/bin/bash
# Supervisor for the 20k joint+classifier run: relaunches from the latest
# checkpoint if the trainer is killed (the box SIGTERMs long GPU jobs).
set -u
cd /home/mamba/ML_project/Testing/Huy/diffusion
CKPT=ckpts/edit_agent/joint_cls20k
LOG=logs/joint_cls20k.log
GPUS=${GPUS:-0}
STEPS=${STEPS:-20000}

for attempt in $(seq 1 40); do
  if grep -q "^done:" "$LOG" 2>/dev/null; then
    echo "[supervisor] training finished"; break
  fi
  echo "[supervisor] attempt $attempt $(date +%H:%M:%S)" >> logs/supervisor.log
  CUDA_VISIBLE_DEVICES=$GPUS conda run --no-capture-output -n llama \
    python -u src/edit_agent/train_joint_ddp.py \
      --arch fusion --mg-init ckpts/edit_agent/musicgen_fusion/step_30000 \
      --steps "$STEPS" --batch 2 --accum 4 --lam 0.5 --cls-weight 0.1 \
      --ckpt-dir "$CKPT" --resume >> "$LOG" 2>&1
  rc=$?
  echo "[supervisor] exited rc=$rc $(date +%H:%M:%S)" >> logs/supervisor.log
  if grep -q "^done:" "$LOG" 2>/dev/null; then break; fi
  sleep 60   # let GPU memory settle before retrying
done
