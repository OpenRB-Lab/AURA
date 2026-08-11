#!/bin/bash
# Wait for the running mulan precompute (old manifest) to finish, then
# rebuild the bridge manifest (picks up new Slakh pairs + tpl_ examples)
# and top-up latents + mulan for the new audio. Hidden phase deliberately
# NOT run here — it must wait for the SFT v5 checkpoint.
set -u
cd /home/mamba/ML_project/Testing/Huy/diffusion

WAIT_PID=${1:?usage: bridge_topup.sh <mulan_pid>}
while [ -d "/proc/$WAIT_PID" ]; do sleep 60; done
echo "=== mulan phase (pid $WAIT_PID) finished, starting top-up $(date) ==="

conda run --no-capture-output -n llama python -u src/edit_agent/precompute_bridge.py --phase manifest
conda run --no-capture-output -n llama python -u src/edit_agent/precompute_bridge.py --phase latents
conda run --no-capture-output -n llama python -u src/edit_agent/precompute_bridge.py --phase mulan
echo "=== bridge top-up done $(date) ==="
