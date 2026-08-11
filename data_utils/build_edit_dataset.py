"""Stage 4 of the edit-dataset pipeline: join manifests + QA into dataset.jsonl.

Joins chunks.jsonl x edit_prompts.jsonl x edited.jsonl on edit_id/chunk_id, drops
(or keeps, with --keep-flagged) infeasible edits, optionally scores edited audio
against the target prompt with CLAP, and writes the final training manifest plus a
summary.

Usage:
  python src/data_utils/build_edit_dataset.py                 # join + format QA
  python src/data_utils/build_edit_dataset.py --clap          # + CLAP scores (GPU best)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = PROJECT_ROOT / "data/edit_dataset/manifests"
CHUNKS_MANIFEST = MANIFEST_DIR / "chunks.jsonl"
PROMPTS_MANIFEST = MANIFEST_DIR / "edit_prompts.jsonl"
EDITED_MANIFEST = MANIFEST_DIR / "edited.jsonl"
DATASET_MANIFEST = MANIFEST_DIR / "dataset.jsonl"

CLAP_PASS_MIN = 0.25   # min sim(edited, target prompt); tune on the smoke batch


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clap", action="store_true", help="compute CLAP QA scores")
    parser.add_argument("--keep-flagged", action="store_true",
                        help="keep feasibility-flagged edits (marked, not dropped)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    chunks = {r["chunk_id"]: r for r in load_jsonl(CHUNKS_MANIFEST)}
    prompts = load_jsonl(PROMPTS_MANIFEST)
    edited = {r["edit_id"]: r for r in load_jsonl(EDITED_MANIFEST) if r.get("status") == "ok"}
    print(f"{len(chunks)} chunks, {len(prompts)} edit prompts, {len(edited)} edited audios")

    records, dropped = [], Counter()
    for p in prompts:
        chunk = chunks.get(p["chunk_id"])
        if chunk is None:
            dropped["missing_chunk"] += 1
            continue
        if p["feasibility"] != "ok" and not args.keep_flagged:
            dropped[f"feasibility:{p['feasibility'].split(':')[0]}"] += 1
            continue
        ed = edited.get(p["edit_id"])
        if ed is None:
            dropped["not_edited_yet"] += 1
            continue
        if not (PROJECT_ROOT / ed["edited_path"]).exists():
            dropped["edited_audio_missing"] += 1
            continue

        records.append({
            **{k: chunk[k] for k in ("chunk_id", "song_id", "source", "chunk_path",
                                     "start_s", "end_s", "duration_s", "chunk_index",
                                     "n_chunks", "position", "features")},
            "edit_id": p["edit_id"],
            "description": p["description"],
            "edit_type": p["edit_type"],
            "edit_instruction": p["edit_instruction"],
            "melodyflow_prompt": p["melodyflow_prompt"],
            "src_prompt": p["src_prompt"],
            "feasibility": p["feasibility"],
            "edited_path": ed["edited_path"],
            "melodyflow_params": ed["melodyflow_params"],
            "qa": {"json_valid": True},
        })

    if args.clap and records:
        # reuse scores from a previous build so incremental rebuilds only score new records
        prev_qa = {r["edit_id"]: r["qa"] for r in load_jsonl(DATASET_MANIFEST)
                   if "clap_edit_target" in r.get("qa", {})}
        cached = [r for r in records if r["edit_id"] in prev_qa]
        todo = [r for r in records if r["edit_id"] not in prev_qa]
        for r in cached:
            r["qa"] = prev_qa[r["edit_id"]]
        if cached:
            print(f"reusing CLAP scores for {len(cached)} records, computing {len(todo)} new")
        if todo:
            import torch
            # cuDNN init is broken in this env (CUDNN_STATUS_NOT_INITIALIZED on any conv);
            # the plain GPU conv path works, so bypass cuDNN for CLAP's STFT convs
            torch.backends.cudnn.enabled = False
            from evaluation.clap_score import compute_clap_score  # src/ on PYTHONPATH

            def batched_scores(paths, texts, bs=128):
                # compute_clap_score loads every file into RAM at once; batch it
                out = []
                for i in range(0, len(paths), bs):
                    out += compute_clap_score(paths[i:i + bs], texts[i:i + bs],
                                              device=args.device, verbose=False)["scores"]
                    if (i // bs) % 10 == 0:
                        print(f"  CLAP {min(i + bs, len(paths))}/{len(paths)}", flush=True)
                return out

            edited_paths = [str(PROJECT_ROOT / r["edited_path"]) for r in todo]
            src_paths = [str(PROJECT_ROOT / r["chunk_path"]) for r in todo]
            targets = [r["melodyflow_prompt"] for r in todo]
            print("computing CLAP(edited, target) ...")
            s_edit = batched_scores(edited_paths, targets)
            print("computing CLAP(source, target) ...")
            s_src = batched_scores(src_paths, targets)
            for r, a, b in zip(todo, s_edit, s_src):
                margin = float(a) - float(b)
                r["qa"].update({
                    "clap_edit_target": round(float(a), 4),
                    "clap_src_target": round(float(b), 4),
                    "clap_margin": round(margin, 4),
                    "passed": bool(a > CLAP_PASS_MIN and margin > 0),
                })

    with open(DATASET_MANIFEST, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(records)} records -> {DATASET_MANIFEST.relative_to(PROJECT_ROOT)}")
    if dropped:
        print("dropped:", dict(dropped))
    print("by edit_type:", dict(Counter(r["edit_type"] for r in records)))
    print("by source:", dict(Counter(r["source"] for r in records)))
    if args.clap and records and "passed" in records[0]["qa"]:
        n_pass = sum(r["qa"]["passed"] for r in records)
        print(f"CLAP pass rate: {n_pass}/{len(records)} ({n_pass / len(records):.1%})")


if __name__ == "__main__":
    main()
