"""Precompute and cache all embeddings for image-conditioned DiffRhythm training.

For each (image, music) pair:
  1. z_img: frozen image encoder pooled embedding → [d_img]
  2. latent: VAE-encode music_wav → [d_vae, T] (flow-matching target)
  3. z_mulan_audio: MuQ-MuLan audio embedding → [512] (projector target)

Outputs cached .pt files + index.jsonl manifest.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False
import soundfile as sf
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFRHYTHM_ROOT = PROJECT_ROOT / "src" / "DiffRhythm"
sys.path.insert(0, str(DIFFRHYTHM_ROOT))

from infer.infer_utils import (
    prepare_audio, encode_audio, vae_sample, normalize_audio
)


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        raw = f.read()
    raw = raw.replace("${dataset_root}", str(Path(__file__).resolve().parents[2] / "data" / "image_music"))
    raw = raw.replace("${diffrhythm_root}", str(DIFFRHYTHM_ROOT))
    raw = raw.replace("${image_encoder.embed_dim}", "768")
    cfg = yaml.safe_load(raw)
    return cfg


def build_image_encoder(cfg: dict, device: torch.device):
    from transformers import SiglipVisionModel, SiglipImageProcessor
    enc_cfg = cfg["image_encoder"]
    model = SiglipVisionModel.from_pretrained(enc_cfg["model_name"]).to(device).eval()
    processor = SiglipImageProcessor.from_pretrained(enc_cfg["model_name"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def build_vae(device: torch.device, cache_dir: str = "./weights"):
    from huggingface_hub import hf_hub_download
    vae_path = hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-vae",
        filename="vae_model.pt",
        cache_dir=cache_dir,
    )
    vae = torch.jit.load(vae_path, map_location="cpu").to(device)
    vae.eval()
    return vae


def build_mulan(device: torch.device, cache_dir: str = "./weights"):
    from muq import MuQMuLan
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=cache_dir)
    mulan = mulan.to(device).eval()
    return mulan


def collect_pairs(painting_dir: str, music_dir: str, categories: list[str]):
    """Pair images and music by category. Since counts differ, cycle music within each category."""
    pairs = []
    for cat in categories:
        img_dir = Path(painting_dir) / cat
        mus_dir = Path(music_dir) / cat
        if not img_dir.exists() or not mus_dir.exists():
            print(f"Warning: skipping category {cat}")
            continue

        imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        wavs = sorted(mus_dir.glob("*.wav"))
        if not wavs:
            continue

        for i, img_path in enumerate(imgs):
            wav_path = wavs[i % len(wavs)]
            pairs.append({
                "image_path": str(img_path),
                "music_path": str(wav_path),
                "category": cat,
            })
    return pairs


@torch.no_grad()
def process_image(img_path: str, vision_model, processor, device: torch.device):
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = vision_model(**inputs)
    z_img = out.pooler_output  # [1, d_img]
    z_img = z_img / z_img.norm(dim=-1, keepdim=True)
    return z_img.squeeze(0).cpu()  # [d_img]


@torch.no_grad()
def process_audio_vae(wav_path: str, vae, device: torch.device,
                      target_sr: int = 44100, channels: int = 2):
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim == 1:
        data = data[:, None]
    audio = torch.from_numpy(data.T)  # [channels, samples]
    audio = prepare_audio(audio, in_sr=sr, target_sr=target_sr,
                          target_length=None, target_channels=channels, device=device)
    audio = normalize_audio(audio, -6)
    latent = encode_audio(audio, vae, chunked=False)  # [1, 128, T]
    mean, scale = latent.chunk(2, dim=1)  # [1, 64, T] each
    z, _ = vae_sample(mean, scale)
    return z.squeeze(0).cpu()  # [64, T]


@torch.no_grad()
def process_audio_mulan(wav_path: str, mulan, device: torch.device):
    import librosa
    y, _ = librosa.load(wav_path, sr=24000, mono=True)
    dur = len(y) / 24000
    if dur >= 10:
        mid = dur / 2
        start = int((mid - 5) * 24000)
        y = y[start:start + 10 * 24000]
    wav_t = torch.tensor(y).unsqueeze(0).to(device)
    emb = mulan(wavs=wav_t)  # [1, 512]
    return emb.squeeze(0).cpu()  # [512]


def main():
    parser = argparse.ArgumentParser(description="Precompute cached latents for image-cond training")
    parser.add_argument("--config", type=str, default="src/configs/image_cond.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    cache_dir = Path(cfg.get("cache_dir", "data/image_music/cached_latents"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    pretrained_cache = cfg.get("pretrained_cache", str(PROJECT_ROOT / "weights"))

    print("Loading image encoder...")
    clip_model, preprocess = build_image_encoder(cfg, device)

    print("Loading VAE...")
    vae = build_vae(device, cache_dir=pretrained_cache)

    print("Loading MuQ-MuLan...")
    mulan = build_mulan(device, cache_dir=pretrained_cache)

    data_cfg = cfg["data"]
    pairs = collect_pairs(
        data_cfg["painting_dir"], data_cfg["music_dir"], data_cfg["categories"]
    )
    print(f"Found {len(pairs)} image-music pairs")

    manifest = []
    for idx, pair in enumerate(tqdm(pairs, desc="Processing")):
        try:
            z_img = process_image(pair["image_path"], clip_model, preprocess, device)
            latent = process_audio_vae(pair["music_path"], vae, device,
                                       target_sr=data_cfg["target_sr"],
                                       channels=data_cfg["audio_channels"])
            z_mulan = process_audio_mulan(pair["music_path"], mulan, device)

            out_path = cache_dir / f"sample_{idx:05d}.pt"
            torch.save({
                "z_img": z_img.half(),           # [d_img]
                "latent": latent.half(),          # [64, T]
                "z_mulan_audio": z_mulan.half(),  # [512]
            }, out_path)

            entry = {
                "idx": idx,
                "cache_path": str(out_path),
                "image_path": pair["image_path"],
                "music_path": pair["music_path"],
                "category": pair["category"],
                "latent_frames": latent.shape[-1],
            }
            manifest.append(entry)

        except Exception as e:
            print(f"Error processing pair {idx}: {e}")
            continue

    manifest_path = cache_dir / "index.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Done. Cached {len(manifest)} samples to {cache_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
