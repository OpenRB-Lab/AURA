"""Comprehensive benchmark for the image-conditioned DiffRhythm model.

Runs:
  1. Retrieval eval (R@K) — projector quality on cached painting dataset
  2. Batch generation — generate samples from test images (5 per category)
  3. FAD — Fréchet Audio Distance vs reference music
  4. CLAP score — text-audio similarity using text prompts from the manifest

Usage:
  CUDA_VISIBLE_DEVICES=1 python src/scripts/benchmark.py \
      --adapter-dir ckpts/it_d_suno/final \
      --samples-per-category 5
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFRHYTHM_ROOT = PROJECT_ROOT / "src" / "DiffRhythm"
sys.path.insert(0, str(DIFFRHYTHM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        raw = f.read()
    raw = raw.replace("${dataset_root}", str(PROJECT_ROOT / "data" / "image_music"))
    raw = raw.replace("${diffrhythm_root}", str(DIFFRHYTHM_ROOT))
    raw = raw.replace("${image_encoder.embed_dim}", "768")
    return yaml.safe_load(raw)


def run_retrieval_eval(cfg, adapter_dir, device):
    """Run retrieval R@K evaluation on cached painting latents."""
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION (R@K)")
    print("=" * 60)

    from src.image_cond.eval import evaluate
    results = evaluate(cfg, adapter_dir=adapter_dir, device_str=device)
    return results


def select_test_images(painting_dir, categories, n_per_cat, seed=42):
    """Select random test images from each category."""
    random.seed(seed)
    selected = []
    for cat in categories:
        cat_dir = Path(painting_dir) / cat
        if not cat_dir.exists():
            print(f"Warning: category dir {cat_dir} not found")
            continue
        images = sorted([p for p in cat_dir.iterdir()
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')])
        if len(images) <= n_per_cat:
            chosen = images
        else:
            chosen = random.sample(images, n_per_cat)
        for img in chosen:
            selected.append({"image_path": str(img), "category": cat})
    return selected


@torch.no_grad()
def batch_generate(cfg, adapter_dir, test_images, output_dir, device, duration=30.0):
    """Generate music from a batch of test images."""
    print("\n" + "=" * 60)
    print(f"BATCH GENERATION ({len(test_images)} samples)")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    infer_cfg = cfg["inference"]

    SAMPLE_RATE = 44100
    DOWNSAMPLE_RATIO = 2048
    max_frames = cfg["model"]["max_frames"]
    target_frames = int(duration * SAMPLE_RATE / DOWNSAMPLE_RATIO)
    target_frames = min(target_frames, max_frames)
    actual_dur = target_frames * DOWNSAMPLE_RATIO / SAMPLE_RATE

    # Load models once
    print("Loading SigLIP image encoder...")
    from transformers import SiglipVisionModel, SiglipImageProcessor
    enc_cfg = cfg["image_encoder"]
    vision_model = SiglipVisionModel.from_pretrained(enc_cfg["model_name"]).to(device).eval()
    processor = SiglipImageProcessor.from_pretrained(enc_cfg["model_name"])

    print("Loading DiffRhythm + LoRA + projector...")
    from src.image_cond.lora_dit import build_image_cond_model, load_adapter
    model = build_image_cond_model(cfg, device)
    model = load_adapter(model, adapter_dir, device)
    model.eval()
    model.cfm.half()
    model.projector.float()

    print("Loading VAE decoder...")
    from huggingface_hub import hf_hub_download
    pretrained_cache = cfg.get("pretrained_cache", str(PROJECT_ROOT / "weights"))
    vae_path = hf_hub_download(repo_id="ASLP-lab/DiffRhythm-vae",
                               filename="vae_model.pt", cache_dir=pretrained_cache)
    vae = torch.jit.load(vae_path, map_location="cpu").to(device)

    import numpy as np
    neg_style_path = DIFFRHYTHM_ROOT / "infer" / "example" / "vocal.npy"
    neg_style = torch.from_numpy(np.load(neg_style_path)).to(device).half()

    from infer.infer_utils import decode_audio
    from PIL import Image
    import soundfile as sf

    # Pre-encode all images, then free the vision model to save GPU memory
    print("  Encoding all images...")
    encoded_images = []
    for item in test_images:
        img = Image.open(item["image_path"]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        vision_out = vision_model(**inputs)
        z_img = vision_out.pooler_output
        z_img = z_img / z_img.norm(dim=-1, keepdim=True)
        z_img_tokens = vision_out.last_hidden_state
        encoded_images.append((z_img.cpu(), z_img_tokens.cpu()))
    del vision_model, processor, inputs, vision_out
    torch.cuda.empty_cache()
    print(f"  Encoded {len(encoded_images)} images, freed vision model")

    manifest = []
    for i, item in enumerate(test_images):
        img_path = item["image_path"]
        category = item["category"]
        stem = f"{category}_{Path(img_path).stem}"
        wav_path = output_dir / f"{stem}.wav"

        if wav_path.exists():
            print(f"  [{i+1}/{len(test_images)}] Skipping (exists): {stem}")
            manifest.append({
                "image_path": img_path, "category": category,
                "wav_path": str(wav_path), "stem": stem,
            })
            continue

        print(f"  [{i+1}/{len(test_images)}] Generating: {stem} ({actual_dur:.0f}s)")
        t0 = time.time()

        z_img, z_img_tokens = encoded_images[i]
        z_img = z_img.to(device)
        z_img_tokens = z_img_tokens.to(device)

        style = model.projector(z_img.float()).half()
        model._set_img_tokens(z_img_tokens)

        cond = torch.zeros(1, max_frames, 64, device=device, dtype=torch.half)
        lrc = torch.zeros(1, max_frames, dtype=torch.long, device=device)
        start_time = torch.zeros(1, device=device, dtype=torch.half)
        norm_duration = torch.tensor([target_frames / max_frames],
                                     device=device, dtype=torch.half)

        out, _ = model.cfm.sample(
            cond=cond, text=lrc, duration=max_frames,
            style_prompt=style, negative_style_prompt=neg_style,
            steps=infer_cfg["steps"], cfg_strength=infer_cfg["cfg_strength"],
            seed=infer_cfg.get("seed", 42) + i,
            start_time=start_time,
            latent_pred_segments=[(0, target_frames)],
            song_duration=norm_duration,
        )

        generated = out[0][:, :target_frames, :]
        latent = generated.float().permute(0, 2, 1)
        audio = decode_audio(latent, vae, chunked=True).squeeze(0).float().cpu()
        target_samples = int(actual_dur * SAMPLE_RATE)
        audio = audio[:, :target_samples]

        max_amp = audio.abs().max()
        if max_amp > 0:
            audio = audio / max_amp * 0.95
        audio = audio.clamp(-1, 1)
        audio_int16 = (audio * 32767).detach().numpy().astype("int16")
        sf.write(str(wav_path), audio_int16.T, SAMPLE_RATE, subtype="PCM_16")

        elapsed = time.time() - t0
        print(f"    Saved {wav_path.name} ({elapsed:.1f}s)")

        manifest.append({
            "image_path": img_path, "category": category,
            "wav_path": str(wav_path), "stem": stem,
        })

        del out, generated, latent, audio, cond, lrc, style, z_img, z_img_tokens
        torch.cuda.empty_cache()

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Clean up GPU memory
    del model, vae
    torch.cuda.empty_cache()

    return manifest


def run_fad(generated_dir, reference_dir):
    """Compute FAD between generated and reference audio."""
    print("\n" + "=" * 60)
    print("FAD (Fréchet Audio Distance)")
    print("=" * 60)
    from src.evaluation.fad import compute_fad
    score = compute_fad(generated_dir=generated_dir,
                        reference_dir=reference_dir,
                        model_name="vggish")
    return score


def run_clap(generated_dir, manifest, categories):
    """Compute CLAP score using category-based text prompts."""
    print("\n" + "=" * 60)
    print("CLAP Score (Text-Audio Similarity)")
    print("=" * 60)

    category_prompts = {
        "安静平和": "serene peaceful calm ambient traditional East Asian instrumental music with soft strings and nature sounds",
        "悲伤孤傲": "melancholic sorrowful lonely instrumental music with solo strings and deep emotional atmosphere",
        "活泼欢快": "lively cheerful upbeat festive instrumental music with bright melodies and energetic rhythm",
        "激昂肆意": "intense passionate powerful dramatic orchestral music with bold brass and driving percussion",
    }

    audio_paths = []
    text_prompts = []
    for item in manifest:
        wav_path = item["wav_path"]
        if Path(wav_path).exists():
            audio_paths.append(wav_path)
            text_prompts.append(category_prompts.get(item["category"],
                                                      "instrumental music"))
            # Also write the .txt file for the benchmark script
            txt_path = Path(wav_path).with_suffix(".txt")
            txt_path.write_text(category_prompts.get(item["category"], "instrumental music"))

    if not audio_paths:
        print("No generated audio files found!")
        return None

    from src.evaluation.clap_score import compute_clap_score
    results = compute_clap_score(audio_paths, text_prompts)

    # Per-category breakdown
    cat_scores = {}
    for item in manifest:
        cat = item["category"]
        stem = item["stem"]
        if stem in results.get("per_sample", {}):
            cat_scores.setdefault(cat, []).append(results["per_sample"][stem])

    print(f"\nPer-category CLAP scores:")
    for cat in categories:
        if cat in cat_scores:
            scores = cat_scores[cat]
            import numpy as np
            print(f"  {cat}: {np.mean(scores):.4f} ± {np.std(scores):.4f} (n={len(scores)})")

    return results


def print_summary(retrieval_results, fad_score, clap_results, adapter_dir):
    """Print final summary table."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print(f"Model: {adapter_dir}")
    print("=" * 60)

    print("\n--- Retrieval (Projector Quality) ---")
    if "trained_projector" in retrieval_results:
        for k, v in retrieval_results["trained_projector"].items():
            print(f"  {k}: {v:.4f}")
    if "mean_cosine_sim" in retrieval_results:
        print(f"  Mean cosine sim: {retrieval_results['mean_cosine_sim']:.4f}")

    if "raw_clip_vs_mulan" in retrieval_results:
        print("\n  (Baseline: raw CLIP → MuLan, no projector)")
        for k, v in retrieval_results["raw_clip_vs_mulan"].items():
            print(f"    {k}: {v:.4f}")

    print(f"\n--- FAD (lower is better) ---")
    if fad_score is not None:
        print(f"  FAD (vggish): {fad_score:.4f}")
    else:
        print(f"  FAD: skipped")

    print(f"\n--- CLAP Score (higher is better, max 1.0) ---")
    if clap_results:
        print(f"  Mean: {clap_results['mean']:.4f} ± {clap_results['std']:.4f}")
    else:
        print(f"  CLAP: skipped")

    # Save full results
    all_results = {
        "adapter_dir": adapter_dir,
        "retrieval": retrieval_results,
        "fad": {"score": fad_score, "model": "vggish"} if fad_score else None,
        "clap": clap_results,
    }
    results_path = PROJECT_ROOT / "results" / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark image-conditioned DiffRhythm model")
    parser.add_argument("--adapter-dir", type=str, default=None,
                        help="Path to adapter checkpoint (default: auto-detect latest in it_d_suno)")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "src/configs" / "image_cond.yaml"))
    parser.add_argument("--samples-per-category", type=int, default=5,
                        help="Number of test images per category for generation")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duration of generated samples in seconds")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation, only run retrieval + eval on existing samples")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-fad", action="store_true")
    parser.add_argument("--skip-clap", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)

    adapter_dir = args.adapter_dir
    if adapter_dir is None:
        # Auto-detect latest in it_d_suno
        ckpt_base = PROJECT_ROOT / "ckpts" / "it_d_suno"
        if (ckpt_base / "final" / "projector.pt").exists():
            adapter_dir = str(ckpt_base / "final")
        else:
            step_dirs = sorted(
                [d for d in ckpt_base.iterdir() if d.is_dir() and d.name.startswith("step_")],
                key=lambda d: int(d.name.split("_")[1]))
            adapter_dir = str(step_dirs[-1]) if step_dirs else None
        if adapter_dir is None:
            print("Error: no adapter checkpoint found")
            return

    print(f"Benchmarking model: {adapter_dir}")
    categories = cfg["data"]["categories"]
    painting_dir = cfg["data"]["painting_dir"]
    music_dir = cfg["data"]["music_dir"]

    # Output directory for this benchmark run
    adapter_name = Path(adapter_dir).parent.name + "_" + Path(adapter_dir).name
    output_dir = PROJECT_ROOT / "results" / "benchmark" / adapter_name

    # 1. Retrieval eval
    retrieval_results = {}
    if not args.skip_retrieval:
        retrieval_results = run_retrieval_eval(cfg, adapter_dir, args.device)

    # 2. Batch generation
    manifest = []
    if not args.skip_generation:
        test_images = select_test_images(painting_dir, categories,
                                         args.samples_per_category, seed=args.seed)
        print(f"\nSelected {len(test_images)} test images across {len(categories)} categories")
        manifest = batch_generate(cfg, adapter_dir, test_images,
                                  str(output_dir), args.device, duration=args.duration)
    else:
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            print(f"Loaded {len(manifest)} samples from existing manifest")

    # 3. FAD — use a clean dir with only .wav files (FAD lib scans all files)
    fad_score = None
    if not args.skip_fad and manifest:
        gen_wavs = [m["wav_path"] for m in manifest if Path(m["wav_path"]).exists()]
        if len(gen_wavs) >= 2:
            fad_wav_dir = output_dir / "wav_only"
            fad_wav_dir.mkdir(exist_ok=True)
            import shutil
            for wav in gen_wavs:
                dst = fad_wav_dir / Path(wav).name
                if not dst.exists():
                    shutil.copy2(wav, dst)
            # FAD lib may choke on subdirs — create flat symlink dir for reference
            ref_wav_dir = output_dir / "ref_wav_flat"
            if not ref_wav_dir.exists():
                ref_wav_dir.mkdir(exist_ok=True)
                for wav in Path(music_dir).rglob("*.wav"):
                    dst = ref_wav_dir / wav.name
                    if not dst.exists():
                        os.symlink(wav.resolve(), dst)
            fad_score = run_fad(str(fad_wav_dir), str(ref_wav_dir))
        else:
            print("Not enough generated samples for FAD")

    # 4. CLAP
    clap_results = None
    if not args.skip_clap and manifest:
        clap_results = run_clap(str(output_dir), manifest, categories)

    # Summary
    print_summary(retrieval_results, fad_score, clap_results, adapter_dir)


if __name__ == "__main__":
    main()
