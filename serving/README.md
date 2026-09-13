# Serving AURA

Two ways to serve the model, both loading the thinker + fusion bridge + stem
executor and running the full add / remove / extract editing pipeline:

- `api.py` — FastAPI service (default port **9005**)
- `gradio_app.py` — Gradio web app (default port **7862**)

> Ports 9000–9003 belong to other services (the dataset-generation vLLM server);
> AURA serving uses 9005 / 7862 to avoid collisions.

## Requirements

- Conda env **`llama`** and a GPU.
- Trained checkpoints (not shipped in the repo), by default:
  - thinker: `ckpts/edit_agent/joint_fusion_r64/final/qwen`
  - fusion bridge: `ckpts/edit_agent/joint_fusion_r64/final/musicgen`
  - classifier: `ckpts/edit_agent/joint_cls/final/classifier.pt`

  Override via the `AURA_QWEN`, `AURA_MG`, `AURA_CLASSIFIER` environment
  variables. (Train your own with `src/aura/training/`.)

## Run the API

```bash
CUDA_VISIBLE_DEVICES=0 API_PORT=9005 conda run -n llama python -u serving/api.py
```

Endpoints:
- `GET  /health` — liveness + loaded checkpoint paths.
- `POST /edit` — JSON `{audio_b64, instruction, history?, executor?, guidance?, seed?}`
  → `{audio_b64, gen_time_s}` (32 kHz mono edited wav, base64). `executor` is
  `hybrid` (default) / `pure` / `full`.

Example:
```bash
curl -s localhost:9005/edit -H 'Content-Type: application/json' -d '{
  "audio_b64": "'$(base64 -w0 input.wav)'",
  "instruction": "remove the drums"
}' | python -c "import sys,json,base64; open('edited.wav','wb').write(base64.b64decode(json.load(sys.stdin)['audio_b64']))"
```

## Run the web app

```bash
CUDA_VISIBLE_DEVICES=0 GRADIO_PORT=7862 conda run -n llama python -u serving/gradio_app.py
```

Upload a track, type an instruction (optionally attach an image or continue a
multi-turn conversation), and download the edited audio.
