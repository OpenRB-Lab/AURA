#!/bin/bash
# Wait for the 40k MusicGen bridge run (pid $1) to finish, then launch the
# joint thinker+bridge training on GPU 0, warm-started from its final adapter.
set -u
cd /home/mamba/ML_project/Testing/Huy/diffusion
WAIT_PID=${1:?usage: joint_after_mg.sh <mg_train_pid>}
while [ -d "/proc/$WAIT_PID" ]; do sleep 60; done
if ! grep -q "done:" logs/mg_train.log; then
  echo "mg_train did not finish cleanly — joint launch aborted"
  exit 1
fi
echo "=== mg 40k done, launching joint training $(date) ==="
exec conda run --no-capture-output -n llama python -u src/edit_agent/train_joint.py \
    --mg-init ckpts/edit_agent/musicgen_bridge/final --steps 8000
