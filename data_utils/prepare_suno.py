"""Precompute cached latents from Kukedlc/suno-ai-music-dataset.

Downloads cover images + audio, computes:
  1. z_img: SigLIP image embedding [768]
  2. latent: VAE-encoded music [64, T]
  3. z_mulan_audio: MuQ-MuLan audio embedding [512]

Appends to the existing cached_latents/index.jsonl manifest.
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False

import requests
import soundfile as sf
import numpy as np
from PIL import Image
from tqdm import tqdm

DIFFRHYTHM_ROOT = Path(__file__).resolve().parents[2] / "src" / "DiffRhythm"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DIFFRHYTHM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from infer.infer_utils import prepare_audio, encode_audio, vae_sample, normalize_audio


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        raw = f.read()
    raw = raw.replace("${dataset_root}", str(PROJECT_ROOT / "data" / "image_music"))
    raw = raw.replace("${diffrhythm_root}", str(DIFFRHYTHM_ROOT))
    raw = raw.replace("${image_encoder.embed_dim}", "768")
    return yaml.safe_load(raw)


def build_models(cfg, device):
    from transformers import SiglipVisionModel, SiglipImageProcessor
    enc_cfg = cfg["image_encoder"]
    vision_model = SiglipVisionModel.from_pretrained(enc_cfg["model_name"]).to(device).eval()
    processor = SiglipImageProcessor.from_pretrained(enc_cfg["model_name"])
    for p in vision_model.parameters():
        p.requires_grad_(False)

    from huggingface_hub import hf_hub_download
    pretrained_cache = cfg.get("pretrained_cache", str(PROJECT_ROOT / "weights"))
    vae_path = hf_hub_download(repo_id="ASLP-lab/DiffRhythm-vae",
                               filename="vae_model.pt", cache_dir=pretrained_cache)
    vae = torch.jit.load(vae_path, map_location="cpu").to(device).eval()

    from muq import MuQMuLan
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large",
                                      cache_dir=pretrained_cache).to(device).eval()

    return vision_model, processor, vae, mulan


@torch.no_grad()
def process_image_from_url(url: str, vision_model, processor, device):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    z = vision_model(**inputs).pooler_output
    z = z / z.norm(dim=-1, keepdim=True)
    return z.squeeze(0).cpu()


@torch.no_grad()
def process_audio_file(audio_path: str, vae, mulan, device,
                       max_duration_sec: float = 95.0):
    import librosa
    y, sr = librosa.load(audio_path, sr=44100, mono=False, duration=max_duration_sec)
    if y.ndim == 1:
        y = np.stack([y, y])  # mono to stereo
    audio_t = torch.from_numpy(y).float()
    audio_t = prepare_audio(audio_t, in_sr=44100, target_sr=44100,
                            target_length=None, target_channels=2, device=device)
    audio_t = normalize_audio(audio_t, -6)
    latent = encode_audio(audio_t, vae, chunked=False)
    mean, scale = latent.chunk(2, dim=1)
    z_vae, _ = vae_sample(mean, scale)
    vae_out = z_vae.squeeze(0).cpu()

    y_mono, _ = librosa.load(audio_path, sr=24000, mono=True, duration=max_duration_sec)
    dur = len(y_mono) / 24000
    if dur >= 10:
        mid = dur / 2
        start = int((mid - 5) * 24000)
        y_mono = y_mono[start:start + 10 * 24000]
    wav_t = torch.tensor(y_mono).unsqueeze(0).to(device)
    z_mulan = mulan(wavs=wav_t).squeeze(0).cpu()

    return vae_out, z_mulan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/image_cond.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples to process")
    parser.add_argument("--max-duration", type=float, default=95.0,
                        help="Max audio duration in seconds (default: 95)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_count = len(list(cache_dir.glob("sample_*.pt")))
    print(f"Existing cached samples: {existing_count}")

    existing_manifest = []
    manifest_path = cache_dir / "index.jsonl"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    existing_manifest.append(json.loads(line))

    print("Loading models...")
    vision_model, processor, vae, mulan = build_models(cfg, device)

    print("Loading Suno dataset...")
    from datasets import load_dataset
    ds = load_dataset("Kukedlc/suno-ai-music-dataset", split="train")

    suno_audio_dir = cache_dir / "suno_audio"
    suno_audio_dir.mkdir(exist_ok=True)

    new_entries = []
    idx = existing_count
    limit = args.max_samples or len(ds)

    for i, sample in enumerate(tqdm(ds, desc="Processing Suno", total=min(limit, len(ds)))):
        if i >= limit:
            break

        try:
            image_url = sample.get("image_url") or sample.get("image_large_url")
            if not image_url:
                continue

            audio_file = cache_dir.parent.parent / sample["file_name"]
            if not audio_file.exists():
                audio_file = suno_audio_dir / f"{sample['id']}.mp3"
                if not audio_file.exists():
                    audio_url = sample.get("audio_url")
                    if audio_url:
                        resp = requests.get(audio_url, timeout=60)
                        resp.raise_for_status()
                        audio_file.write_bytes(resp.content)
                    else:
                        continue

            z_img = process_image_from_url(image_url, vision_model, processor, device)
            latent, z_mulan = process_audio_file(
                str(audio_file), vae, mulan, device,
                max_duration_sec=args.max_duration,
            )

            out_path = cache_dir / f"sample_{idx:05d}.pt"
            torch.save({
                "z_img": z_img.half(),
                "latent": latent.half(),
                "z_mulan_audio": z_mulan.half(),
            }, out_path)

            entry = {
                "idx": idx,
                "cache_path": str(out_path),
                "source": "suno",
                "suno_id": sample["id"],
                "tags": sample.get("tags", ""),
                "latent_frames": latent.shape[-1],
                "duration": sample.get("duration", 0),
            }
            new_entries.append(entry)
            idx += 1

        except Exception as e:
            print(f"Error [{i}] {sample.get('id','?')}: {e}")
            continue

    all_entries = existing_manifest + new_entries
    with open(manifest_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Added {len(new_entries)} Suno samples (total: {len(all_entries)})")


if __name__ == "__main__":
    main()
