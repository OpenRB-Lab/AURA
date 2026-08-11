"""Stage-A bridge precompute: VAE latents + [EDIT] hidden states + MuLan targets.

Builds the training cache for the bridge (edit-token hidden states -> DiffRhythm).
Examples come from the edit dialogues (46.6k): one per single-edit conversation, one
per step of trajectory/session conversations. Three resumable phases:

  --phase latents : DiffRhythm-VAE encode every unique audio file (src + tgt) @44.1k
  --phase hidden  : teacher-forced v4 thinker forward per example; grab last-layer
                    hidden states at the [EDIT_0..7] positions of the relevant
                    assistant turn -> [8, 3584] fp16
  --phase mulan   : MuLan audio embedding of each unique TARGET file (align loss)

Outputs under data/edit_dataset/bridge_cache/:
  latents/<md5>.pt        {"latent": fp16 [T,64], "n_frames": int}
  hidden/<example_id>.pt  {"h": fp16 [8,3584]}
  mulan/<md5>.pt          fp16 [512]
  examples.jsonl          manifest linking everything

Usage:
  python src/edit_agent/precompute_bridge.py --phase manifest
  python src/edit_agent/precompute_bridge.py --phase latents [--shard i/n]
  python src/edit_agent/precompute_bridge.py --phase hidden  [--shard i/n]
  python src/edit_agent/precompute_bridge.py --phase mulan
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "DiffRhythm"))

DIALOGUES = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"
CACHE = PROJECT_ROOT / "data/edit_dataset/bridge_cache"
EXAMPLES = CACHE / "examples.jsonl"

SAMPLE_RATE = 44100
DOWNSAMPLE = 2048
FPS = SAMPLE_RATE / DOWNSAMPLE
MAX_FRAMES = 2048


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


# ──────────────────────────────────────────────────────────────
# Phase 0: manifest of examples
# ──────────────────────────────────────────────────────────────

def build_manifest() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(DIALOGUES)]
    n = 0
    with open(EXAMPLES, "w") as f:
        for r in rows:
            if not r["has_edit"]:
                continue
            if r.get("steps"):
                for i, s in enumerate(r["steps"]):
                    tgt = s.get("target_path")
                    src = s.get("input_path", r["chunk_path"])
                    if not tgt:
                        continue
                    f.write(json.dumps({
                        "example_id": f"{r['id']}__s{i}",
                        "dialogue_id": r["id"], "step_index": i,
                        "conv_type": r["conv_type"], "split": r["split"],
                        "src_path": src, "tgt_path": tgt,
                        "chunk_path": r["chunk_path"],
                        "image_path": r.get("image_path"),
                    }) + "\n")
                    n += 1
            elif r.get("edited_path"):
                f.write(json.dumps({
                    "example_id": r["id"], "dialogue_id": r["id"], "step_index": None,
                    "conv_type": r["conv_type"], "split": r["split"],
                    "src_path": r["chunk_path"], "tgt_path": r["edited_path"],
                    "chunk_path": r["chunk_path"],
                    "image_path": r.get("image_path"),
                }) + "\n")
                n += 1
        # pair-only examples (no dialogue): template contexts, e.g. Slakh expansion
        slakh_pairs = PROJECT_ROOT / "data/edit_dataset/slakh/slakh_pairs.jsonl"
        have = {json.loads(l)["id"] for l in open(DIALOGUES)}
        n_tpl = 0
        if slakh_pairs.exists():
            for l in open(slakh_pairs):
                p = json.loads(l)
                if p["trajectory_id"] or f"slakh_{p['pair_id']}" in have:
                    continue
                f.write(json.dumps({
                    "example_id": f"tpl_slakh_{p['pair_id']}",
                    "dialogue_id": None, "step_index": None,
                    "conv_type": f"tpl_slakh_{p['op']}", "split": p["split"],
                    "src_path": p["input_path"], "tgt_path": p["target_path"],
                    "chunk_path": p["input_path"], "image_path": None,
                    "template": {"instruction": p["instruction_seed"], "op": p["op"]},
                }) + "\n")
                n += 1
                n_tpl += 1
        print(f"template examples added: {n_tpl}")
    uniq = set()
    for l in open(EXAMPLES):
        e = json.loads(l)
        uniq.add(e["src_path"]); uniq.add(e["tgt_path"])
    print(f"manifest: {n} examples, {len(uniq)} unique audio files")


# ──────────────────────────────────────────────────────────────
# Phase 1: VAE latents
# ──────────────────────────────────────────────────────────────

def load_wav_any(path: Path) -> tuple[torch.Tensor, int]:
    """torchaudio 2.10 needs torchcodec for decoding; use librosa instead."""
    import librosa
    import numpy as np
    y, sr = librosa.load(path, sr=44100, mono=False)  # resample here: torchaudio 2.10 lacks functional.Resample
    y = np.atleast_2d(y)
    return torch.from_numpy(y.astype("float32")), int(sr)


def run_latents(shard: str | None) -> None:
    torch.backends.cudnn.enabled = False
    from huggingface_hub import hf_hub_download
    from infer.infer_utils import encode_audio, normalize_audio, prepare_audio, vae_sample

    out_dir = CACHE / "latents"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted({p for l in open(EXAMPLES)
                    for p in (json.loads(l)["src_path"], json.loads(l)["tgt_path"])})
    pending = [p for p in paths if not (out_dir / f"{md5(p)}.pt").exists()]
    if shard:
        i, k = (int(x) for x in shard.split("/"))
        pending = [p for j, p in enumerate(pending) if j % k == i]
    print(f"latents: {len(pending)} of {len(paths)} files pending", flush=True)
    if not pending:
        return

    vae = torch.jit.load(hf_hub_download("ASLP-lab/DiffRhythm-vae", "vae_model.pt",
                                         cache_dir=str(PROJECT_ROOT / "weights")),
                         map_location="cpu").to("cuda")
    for k_i, rel in enumerate(pending):
        try:
            wav, sr = load_wav_any(PROJECT_ROOT / rel)
            n_frames = min(int(wav.shape[-1] / sr * FPS), MAX_FRAMES)
            audio = prepare_audio(wav, in_sr=sr, target_sr=SAMPLE_RATE,
                                  target_length=int(n_frames * DOWNSAMPLE),
                                  target_channels=2, device=torch.device("cuda"))
            audio = normalize_audio(audio, -6)
            with torch.no_grad():
                lat = encode_audio(audio.float(), vae, chunked=True)
                mean, scale = lat.chunk(2, dim=1)
                z, _kl = vae_sample(mean, scale)
                latent = z.transpose(1, 2)[0].half().cpu()
            torch.save({"latent": latent, "n_frames": n_frames}, out_dir / f"{md5(rel)}.pt")
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {rel}: {exc}", flush=True)
        if (k_i + 1) % 500 == 0:
            print(f"latents {k_i + 1}/{len(pending)}", flush=True)
    print("latents done", flush=True)


# ──────────────────────────────────────────────────────────────
# Phase 2: [EDIT] hidden states
# ──────────────────────────────────────────────────────────────

def run_hidden(shard: str | None, only_split: str | None = None) -> None:
    from edit_agent.qwen_wrapper import find_edit_positions, load_audio_16k, load_thinker
    from edit_agent.sft_data import SYSTEM_PROMPT

    out_dir = CACHE / "hidden"
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogues = {json.loads(l)["id"]: json.loads(l) for l in open(DIALOGUES)}
    examples = [json.loads(l) for l in open(EXAMPLES)]
    pending = [e for e in examples if not (out_dir / f"{e['example_id']}.pt").exists()]
    if only_split:
        pending = [e for e in pending if e["split"] == only_split]
    if shard:
        i, k = (int(x) for x in shard.split("/"))
        pending = [e for j, e in enumerate(pending) if j % k == i]
    print(f"hidden: {len(pending)} of {len(examples)} examples pending", flush=True)
    if not pending:
        return

    model, proc, edit_ids = load_thinker(
        lora_dir=str(PROJECT_ROOT / "ckpts/edit_agent/sft/final"))
    model.eval()

    from edit_agent.tokens import typed_block
    from PIL import Image
    for k_i, e in enumerate(pending):
        try:
            if e.get("template"):
                tpl = e["template"]
                msgs = [
                    {"role": "user", "content": [
                        {"type": "audio", "audio": e["chunk_path"]},
                        {"type": "text", "text": tpl["instruction"]}]},
                    {"role": "assistant", "content": [
                        {"type": "text",
                         "text": f"I've applied that edit. {typed_block(tpl['op'])}"}]},
                ]
                d = {"chunk_path": e["chunk_path"], "messages": msgs}
            else:
                d = dialogues[e["dialogue_id"]]
                msgs = d["messages"]
                if e["step_index"] is not None:
                    msgs = msgs[: 2 * e["step_index"] + 2]  # through assistant turn i
            full = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}] + msgs
            text = proc.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
            images = None
            if e.get("image_path"):
                images = [Image.open(PROJECT_ROOT / e["image_path"]).convert("RGB")]
            inputs = proc(text=[text], audio=[load_audio_16k(d["chunk_path"])],
                          images=images, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            ids = inputs["input_ids"][0]
            # the LAST edit block in the (possibly truncated) context is this step's;
            # include the preceding [EDIT_<KIND>] type token -> [9, 3584]
            starts = (ids == edit_ids[0]).nonzero(as_tuple=True)[0]
            start = int(starts[-1])
            h = out.hidden_states[-1][0, start - 1:start + len(edit_ids)]
            assert h.shape[0] == len(edit_ids) + 1
            torch.save({"h": h.half().cpu()}, out_dir / f"{e['example_id']}.pt")
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {e['example_id']}: {exc}", flush=True)
        if (k_i + 1) % 500 == 0:
            print(f"hidden {k_i + 1}/{len(pending)}", flush=True)
    print("hidden done", flush=True)


# ──────────────────────────────────────────────────────────────
# Phase 3: MuLan target embeddings
# ──────────────────────────────────────────────────────────────

def run_mulan() -> None:
    torch.backends.cudnn.enabled = False
    import librosa
    from muq import MuQMuLan
    out_dir = CACHE / "mulan"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted({json.loads(l)["tgt_path"] for l in open(EXAMPLES)})
    pending = [p for p in paths if not (out_dir / f"{md5(p)}.pt").exists()]
    print(f"mulan: {len(pending)} of {len(paths)} targets pending", flush=True)
    if not pending:
        return
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large",
                                     cache_dir=str(PROJECT_ROOT / "weights")).to("cuda").eval()
    for k_i, rel in enumerate(pending):
        try:
            dur = librosa.get_duration(path=str(PROJECT_ROOT / rel))
            off = max(dur / 2 - 5, 0)
            y, _ = librosa.load(PROJECT_ROOT / rel, sr=24000, mono=True,
                                offset=off, duration=min(10, dur))
            with torch.no_grad():
                emb = mulan(wavs=torch.from_numpy(y).unsqueeze(0).to("cuda"))[0]
            torch.save(emb.half().cpu(), out_dir / f"{md5(rel)}.pt")
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {rel}: {exc}", flush=True)
        if (k_i + 1) % 1000 == 0:
            print(f"mulan {k_i + 1}/{len(pending)}", flush=True)
    print("mulan done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True,
                        choices=["manifest", "latents", "hidden", "mulan"])
    parser.add_argument("--shard", type=str, default=None)
    parser.add_argument("--only-split", type=str, default=None)
    args = parser.parse_args()
    if args.phase == "manifest":
        build_manifest()
    elif args.phase == "latents":
        run_latents(args.shard)
    elif args.phase == "hidden":
        run_hidden(args.shard, args.only_split)
    else:
        run_mulan()


if __name__ == "__main__":
    main()
