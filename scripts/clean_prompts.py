"""Clean up broken text_prompt fields in cached latent manifests.

Issues fixed:
  1. Chain-of-thought leaks (LLM reasoning instead of prompt) → use tags
  2. ".PROMPT: prefix → strip it
  3. Garbage (<10 chars) → use tags
  4. Meta-prompts (about making prompts) → use tags
  5. Truncated (<100 chars) → use tags if tags are longer

Backs up originals to *.bak before overwriting.
"""

import json
import shutil
from pathlib import Path


def classify_prompt(text_prompt, tags=""):
    """Return (issue_type, cleaned_prompt)."""
    p = text_prompt.strip()

    # Garbage: empty or < 10 chars
    if len(p) < 10:
        return "garbage", tags

    # Chain-of-thought leak
    if (p.startswith("Here's a thinking") or
        "thinking process" in p[:150] or
        p.startswith("The user wants me to") or
        ("Analyze the" in p[:50] and "Tags" in p[:80]) or
        p.startswith("1.  **Analyze")):
        return "thinking_leak", tags

    # ".PROMPT: prefix
    if p.startswith('".PROMPT:') or p.startswith(".PROMPT:"):
        cleaned = p.lstrip('"').lstrip(".PROMPT:").lstrip(": ").strip()
        if cleaned:
            return "dot_prompt", cleaned
        return "dot_prompt", tags

    # Meta-prompt (talks about prompt creation)
    if ("prompt for a music generation" in p.lower() or
        "create a single, cohesive prompt" in p.lower()[:60] or
        "convert a highly detailed" in p.lower()[:60]):
        return "meta_prompt", tags

    # Truncated: < 100 chars but looks real — use tags if longer
    if len(p) < 100 and tags and len(tags) > len(p):
        return "truncated", tags

    return "ok", p


def clean_manifest(manifest_path: str):
    path = Path(manifest_path)
    if not path.exists():
        print(f"  Skipping {path} (not found)")
        return

    # Backup
    backup = path.with_suffix(".jsonl.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  Backed up to {backup.name}")

    with open(path) as f:
        entries = [json.loads(line) for line in f]

    stats = {"ok": 0, "thinking_leak": 0, "dot_prompt": 0,
             "garbage": 0, "meta_prompt": 0, "truncated": 0}

    for entry in entries:
        text_prompt = entry.get("text_prompt", "")
        tags = entry.get("tags", "")
        issue, cleaned = classify_prompt(text_prompt, tags)
        stats[issue] += 1
        if issue != "ok":
            entry["text_prompt"] = cleaned
            entry["_prompt_cleaned"] = issue

    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_fixed = sum(v for k, v in stats.items() if k != "ok")
    print(f"  {path.name}: {len(entries)} entries, {total_fixed} fixed")
    for k, v in stats.items():
        if v > 0:
            print(f"    {k}: {v}")


def main():
    base = Path("data/image_music/cached_latents")

    print("Cleaning index.jsonl...")
    clean_manifest(base / "index.jsonl")

    print("\nCleaning index_suno.jsonl...")
    clean_manifest(base / "index_suno.jsonl")

    # Verify a few cleaned entries
    print("\n--- Verification (5 cleaned samples) ---")
    with open(base / "index_suno.jsonl") as f:
        entries = [json.loads(l) for l in f]

    count = 0
    for e in entries:
        if "_prompt_cleaned" in e:
            print(f"  idx={e['idx']} ({e['_prompt_cleaned']}):")
            print(f"    prompt: {e['text_prompt'][:150]}...")
            count += 1
            if count >= 5:
                break


if __name__ == "__main__":
    main()
