#!/bin/bash
# Synthesize edit-agent training conversations via the :9003 vLLM server
set -e
cd "$(dirname "$0")/../.."
conda run -n llama python src/edit_agent/synth_dialogues.py "$@"
