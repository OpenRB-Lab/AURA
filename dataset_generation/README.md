# AURA dataset generation

Code to build the AURA edit-training dataset. Two independent source tracks
(MelodyFlow-regenerated targets and exact Slakh stem-mixes) feed a single
multi-turn conversation file, which is then augmented with chain-of-thought
stem plans and turned into EnCodec / classifier caches for bridge training.

## Prerequisites

- **Conda envs:** `llama` (default, all scripts unless noted), `melodyflow`
  (MelodyFlow rendering), `sa3` (optional Stable-Audio-3 inpaint track).
- **Local LLM server (required):** a vLLM OpenAI-compatible **Qwen** server on
  `http://localhost:9003/v1` — used for instruction and dialogue generation.
  Configure via `LLM_URL` / `LLM_MODEL` (see `data_utils/llm_client.py`). Launch
  the server with thinking disabled (`--reasoning-parser qwen3`).
- Source corpora on disk (Suno/painting audio, Slakh, MoisesDB) under `data/`.

## Pipeline (run order)

### A. MelodyFlow track  (`data_utils/`)
```bash
conda run -n llama      python -u data_utils/chunk_audio.py          # Stage 1: beat-aligned <30s chunks -> manifests/chunks.jsonl
conda run -n llama      python -u data_utils/generate_edit_prompts.py # Stage 2: Qwen edit prompts (needs :9003) -> edit_prompts.jsonl
conda run -n melodyflow python -u data_utils/run_melodyflow_edit.py   # Stage 3: render edited targets -> edited/ + edited.jsonl
conda run -n llama      python -u data_utils/build_edit_dataset.py    # Stage 4: join + QA/CLAP filter -> manifests/dataset.jsonl
conda run -n llama      python -u synth_dialogues.py                  # multi-turn dialogues (needs :9003)
```

### B. Slakh track  (independent)
```bash
conda run -n llama python -u slakh_edits.py            # exact stem-mix edit pairs -> slakh/slakh_pairs.jsonl
conda run -n llama python -u synth_dialogues_slakh.py  # dialogues for Slakh pairs (needs :9003)
conda run -n llama python -u fix_typed_blocks.py       # normalize [EDIT] -> typed blocks
```

### C. Dialogue augmentation & caches
```bash
conda run -n llama python -u synth_dialogues_audioedit.py   # optional: general-audio triplets (needs :9003)
conda run -n llama python -u build_cot_dialogues.py         # prepend <plan>...</plan> -> dialogues_cot.jsonl
conda run -n llama python -u validate_dialogues.py          # QA report

conda run -n llama python -u build_semantic_labels.py       # classifier labels -> bridge_cache/semantic_labels.json
conda run -n llama python -u precompute_encodec.py          # MusicGen-EnCodec code cache
conda run -n llama python -u precompute_latent.py           # pre-quantizer latent cache
conda run -n llama python -u build_stem_targets.py          # per-stem targets for stem-mode training
conda run -n llama python -u filter_silence.py              # silence ratios -> bridge_cache/silence_ratio.json
```

### D. Optional: Stable-Audio-3 segment-inpaint track
```bash
conda run -n llama python -u plan_inpaint_edits.py   # plan segment edits (needs :9003)
conda run -n sa3   python -u render_inpaint.py       # render targets (Stable Audio 3)
conda run -n llama python -u synth_dialogues_inpaint.py
```

All outputs land under `data/edit_dataset/` (chunks, edited, slakh, dialogues,
bridge_cache). Every script carries a `Usage:` docstring with its exact I/O.
Held-out evaluation benchmarks are built separately (kept in `benchmarks/`).
