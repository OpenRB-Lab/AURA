#!/bin/bash
# Supervisor for the 20k joint+classifier DDP run on the NVLink pair (2,3):
# relaunches from the latest checkpoint whenever the trainer dies.
set -u
cd /home/mamba/ML_project/Testing/Huy/diffusion
LOG=logs/joint_cls20k.log
PORT=29560

for attempt in $(seq 1 60); do
  if grep -q "^done:" "$LOG" 2>/dev/null; then
    echo "[supervisor] finished at attempt $attempt" >> logs/supervisor.log
    break
  fi
  echo "[supervisor] attempt $attempt $(date '+%F %H:%M:%S')" >> logs/supervisor.log
  CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n llama \
    torchrun --nproc_per_node=2 --master_port=$((PORT + attempt)) \
    src/edit_agent/train_joint_ddp.py \
      --arch fusion --mg-init ckpts/edit_agent/musicgen_fusion/step_30000 \
      --steps 20000 --batch 2 --accum 2 --lam 0.5 --cls-weight 0.1 \
      --ckpt-dir ckpts/edit_agent/joint_cls20k --resume >> "$LOG" 2>&1
  echo "[supervisor] exited rc=$? $(date '+%F %H:%M:%S')" >> logs/supervisor.log
  pkill -9 -f train_joint_ddp 2>/dev/null
  sleep 90   # let GPU memory + NCCL ports settle
done
