"""Stage 3 of the edit-dataset pipeline: apply edits with MelodyFlow (GPU).

Reads edit_prompts.jsonl, loads each source chunk once, encodes it with the
MelodyFlow EnCodec-VAE, and runs regularized flow-inversion editing toward each
edit's melodyflow_prompt. Writes edited audio + edited.jsonl manifest.

Runs in the dedicated `melodyflow` conda env (torch 2.8, audiocraft fork from the
HF Space facebook/MelodyFlow). Device via MELODYFLOW_DEVICE (default cuda:0).

Usage:
  MELODYFLOW_DEVICE=cuda:2 python src/data_utils/run_melodyflow_edit.py --batch-size 8
  python src/data_utils/run_melodyflow_edit.py --dry-run          # no GPU, no weights
  python src/data_utils/run_melodyflow_edit.py --shard 0/2        # multi-GPU sharding
"""

import argparse
import functools
import json
import os
import time
from pathlib import Path

import torch

# MelodyFlow checkpoints store omegaconf objects; torch>=2.6 defaults to
# weights_only=True which rejects them. Same patch as the official demo app.
torch.load = functools.partial(torch.load, weights_only=False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = PROJECT_ROOT / "data/edit_dataset/manifests"
CHUNKS_MANIFEST = MANIFEST_DIR / "chunks.jsonl"
PROMPTS_MANIFEST = MANIFEST_DIR / "edit_prompts.jsonl"
EDITED_MANIFEST = MANIFEST_DIR / "edited.jsonl"
EDITED_DIR = PROJECT_ROOT / "data/edit_dataset/edited"

MODEL_NAME = os.environ.get("MELODYFLOW_MODEL", "facebook/melodyflow-t24-30secs")
DEVICE = os.environ.get("MELODYFLOW_DEVICE", "cuda:0")

EDIT_PARAMS = {
    "solver": "euler",
    "steps": 25,
    "target_flowstep": 0.0,
    "regularize": True,
    "regularize_iters": 4,
    "keep_last_k_iters": 2,
    "lambda_kl": 0.2,
}


def load_pending(shard: str | None, limit: int | None, include_flagged: bool):
    with open(CHUNKS_MANIFEST) as f:
        chunks = {r["chunk_id"]: r for r in map(json.loads, f)}
    with open(PROMPTS_MANIFEST) as f:
        edits = [json.loads(l) for l in f]

    done = set()
    if EDITED_MANIFEST.exists():
        with open(EDITED_MANIFEST) as f:
            done = {r["edit_id"] for r in map(json.loads, f) if r.get("status") == "ok"}

    pending = []
    for e in edits:
        if e["edit_id"] in done:
            continue
        if not include_flagged and e["feasibility"] != "ok":
            continue
        chunk = chunks.get(e["chunk_id"])
        if chunk is None:
            continue
        pending.append((e, chunk))

    if shard:
        i, n = (int(x) for x in shard.split("/"))
        pending = [p for k, p in enumerate(pending) if k % n == i]
    if limit:
        pending = pending[:limit]
    return pending, len(done)


def group_batches(pending, batch_size: int):
    """Group edits by identical chunk sample-length so latents can be concatenated.

    Edits of the same chunk land in the same group, so each chunk is encoded once
    per batch it appears in.
    """
    import soundfile as sf
    by_len: dict[int, list] = {}
    for e, chunk in pending:
        # exact frame count — rounded duration can differ from the file by a frame,
        # which breaks torch.cat on the latents
        n_samples = sf.info(PROJECT_ROOT / chunk["chunk_path"]).frames
        by_len.setdefault(n_samples, []).append((e, chunk))
    batches = []
    for _, items in sorted(by_len.items()):
        items.sort(key=lambda ec: ec[1]["chunk_id"])  # same-chunk edits adjacent
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=str, default=None, help="i/n round-robin shard")
    parser.add_argument("--include-flagged", action="store_true",
                        help="also process edits whose feasibility check flagged them")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan batches and verify imports only; no model load")
    args = parser.parse_args()

    pending, n_done = load_pending(args.shard, args.limit, args.include_flagged)
    batches = group_batches(pending, args.batch_size)
    print(f"{len(pending)} edits pending ({n_done} already done), "
          f"{len(batches)} batches of <= {args.batch_size}")

    if args.dry_run:
        from audiocraft.models import MelodyFlow  # noqa: F401 — import check only
        for b in batches[:5]:
            ids = [e["edit_id"] for e, _ in b]
            print(f"  batch of {len(b)} @ {b[0][1]['duration_s']:.1f}s: {ids[:4]}{'...' if len(ids) > 4 else ''}")
        print("dry run OK (audiocraft imports, manifests consistent)")
        return

    import torchaudio
    from audiocraft.data.audio import audio_write
    from audiocraft.data.audio_utils import convert_audio
    from audiocraft.models import MelodyFlow

    print(f"loading {MODEL_NAME} on {DEVICE} ...")
    model = MelodyFlow.get_pretrained(MODEL_NAME, device=DEVICE)
    model.set_editing_params(**EDIT_PARAMS)
    sr, ac = model.sample_rate, model.audio_channels

    token_cache: dict[str, torch.Tensor] = {}

    def encode_chunk(chunk: dict) -> torch.Tensor:
        cid = chunk["chunk_id"]
        if cid not in token_cache:
            wav, in_sr = torchaudio.load(PROJECT_ROOT / chunk["chunk_path"])
            wav = convert_audio(wav, in_sr, sr, ac)
            wav = wav[..., : int(sr * model.duration)]
            with torch.no_grad():
                token_cache[cid] = model.encode_audio(wav.unsqueeze(0).to(DEVICE)).cpu()
            if len(token_cache) > 64:  # keep memory bounded
                token_cache.pop(next(iter(token_cache)))
        return token_cache[cid]

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(EDITED_MANIFEST, "a") as mf:
        for bi, batch in enumerate(batches):
            t0 = time.time()
            try:
                tokens = torch.cat([encode_chunk(c) for _, c in batch]).to(DEVICE)
                targets = [e["melodyflow_prompt"] for e, _ in batch]
                sources = [e.get("src_prompt", "") for e, _ in batch]
                with torch.no_grad():
                    out = model.edit(prompt_tokens=tokens, descriptions=targets,
                                     src_descriptions=sources, progress=False,
                                     return_tokens=False)
                dt = time.time() - t0
                for (e, chunk), wav in zip(batch, out):
                    out_dir = EDITED_DIR / e["chunk_id"]
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{e['edit_id']}.wav"
                    audio_write(str(out_path.with_suffix("")), wav.cpu().float(), sr,
                                strategy="loudness", loudness_headroom_db=16,
                                loudness_compressor=True, add_suffix=True)
                    mf.write(json.dumps({
                        "edit_id": e["edit_id"],
                        "edited_path": str(out_path.relative_to(PROJECT_ROOT)),
                        "melodyflow_params": {"model": MODEL_NAME, **EDIT_PARAMS},
                        "gen_time_s": round(dt / len(batch), 2),
                        "status": "ok",
                    }) + "\n")
                mf.flush()
                print(f"batch {bi + 1}/{len(batches)}: {len(batch)} edits in {dt:.1f}s")
            except Exception as exc:  # noqa: BLE001 — record failures, keep going
                for e, _ in batch:
                    mf.write(json.dumps({"edit_id": e["edit_id"], "status": f"error: {exc}"}) + "\n")
                mf.flush()
                print(f"batch {bi + 1} FAILED: {exc}")
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
