"""Inference: image → instrumental music via trained LoRA + projector.

Usage:
  python src/image_cond/infer_image.py --image path/to/image.jpg
  python src/image_cond/infer_image.py --image path/to/image.jpg --output my_song.wav
  python src/image_cond/infer_image.py --image path/to/image.jpg --adapter-dir ckpts/step_1000
"""

import argparse
import os
import sys
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False

from PIL import Image

DIFFRHYTHM_ROOT = Path(__file__).resolve().parents[2] / "src" / "DiffRhythm"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DIFFRHYTHM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from infer.infer_utils import decode_audio

import numpy as np
def get_negative_style_prompt(device):
    file_path = DIFFRHYTHM_ROOT / "infer" / "example" / "vocal.npy"
    vocal_style = np.load(file_path)
    vocal_style = torch.from_numpy(vocal_style).to(device).half()
    return vocal_style


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        raw = f.read()
    raw = raw.replace("${dataset_root}", str(PROJECT_ROOT / "data" / "image_music"))
    raw = raw.replace("${diffrhythm_root}", str(DIFFRHYTHM_ROOT))
    raw = raw.replace("${image_encoder.embed_dim}", "768")
    return yaml.safe_load(raw)


def find_latest_adapter(checkpoint_dir: str) -> str:
    ckpt = Path(checkpoint_dir)
    if (ckpt / "final" / "projector.pt").exists():
        return str(ckpt / "final")
    step_dirs = sorted(
        [d for d in ckpt.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if step_dirs:
        return str(step_dirs[-1])
    raise FileNotFoundError(f"No adapter checkpoints found in {checkpoint_dir}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Image → Music: generate instrumental music from an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None,
                        help="Output wav path (default: <image_stem>_music.wav)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to adapter checkpoint dir (default: auto-detect latest)")
    parser.add_argument("--adapter-dir", type=str, default=None,
                        help="Alias for --checkpoint")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "src/configs" / "image_cond.yaml"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None, help="Override ODE solver steps (default: 32)")
    parser.add_argument("--cfg-strength", type=float, default=None, help="Override CFG strength (default: 4.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--text", type=str, default=None,
                        help="Text prompt to guide music generation (e.g., 'calm piano ambient')")
    parser.add_argument("--duration", type=float, default=None,
                        help="Song duration in seconds (default: 95, max: 285)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    infer_cfg = cfg["inference"]

    if args.steps is not None:
        infer_cfg["steps"] = args.steps
    if args.cfg_strength is not None:
        infer_cfg["cfg_strength"] = args.cfg_strength
    if args.seed is not None:
        infer_cfg["seed"] = args.seed

    SAMPLE_RATE = 44100
    DOWNSAMPLE_RATIO = 2048
    if args.duration is not None:
        duration_sec = min(max(args.duration, 5.0), 285.0)
        target_frames = int(duration_sec * SAMPLE_RATE / DOWNSAMPLE_RATIO)
        if target_frames <= 2048:
            cfg["model"]["max_frames"] = 2048
        else:
            cfg["model"]["max_frames"] = 6144
    else:
        duration_sec = None
        target_frames = None

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        args.output = str(results_dir / (Path(args.image).stem + "_music.wav"))
    elif not os.path.isabs(args.output):
        args.output = str(results_dir / args.output)

    adapter_dir = args.checkpoint or args.adapter_dir
    if adapter_dir is None:
        adapter_dir = find_latest_adapter(cfg["checkpoint_dir"])
    print(f"Using adapter: {adapter_dir}")

    # --- Load image encoder ---
    print("Loading SigLIP image encoder...")
    from transformers import SiglipVisionModel, SiglipImageProcessor
    enc_cfg = cfg["image_encoder"]
    vision_model = SiglipVisionModel.from_pretrained(enc_cfg["model_name"]).to(device).eval()
    processor = SiglipImageProcessor.from_pretrained(enc_cfg["model_name"])

    # --- Load model with LoRA + projector ---
    print("Loading DiffRhythm + LoRA + projector...")
    from src.image_cond.lora_dit import build_image_cond_model, load_adapter
    model = build_image_cond_model(cfg, device)
    model = load_adapter(model, adapter_dir, device)
    model.eval()
    model.cfm.half()
    model.projector.float()

    # --- Load VAE ---
    print("Loading VAE decoder...")
    from huggingface_hub import hf_hub_download
    pretrained_cache = cfg.get("pretrained_cache", str(PROJECT_ROOT / "weights"))
    vae_path = hf_hub_download(repo_id="ASLP-lab/DiffRhythm-vae",
                               filename="vae_model.pt", cache_dir=pretrained_cache)
    vae = torch.jit.load(vae_path, map_location="cpu").to(device)

    # --- Encode image ---
    print(f"Encoding image: {args.image}")
    img = Image.open(args.image).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    vision_out = vision_model(**inputs)
    z_img = vision_out.pooler_output  # [1, 768]
    z_img = z_img / z_img.norm(dim=-1, keepdim=True)
    z_img_tokens = vision_out.last_hidden_state  # [1, 256, 768]

    # --- Encode text prompt (optional) ---
    # Use real MuLan text encoder for style (matches base DiT's training distribution)
    # Image conditioning goes through cross-attention on patch tokens
    from muq import MuQMuLan
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large",
                                      cache_dir=pretrained_cache).to(device).eval()

    if args.text:
        print(f"Encoding text via MuLan: {args.text}")
        with torch.no_grad():
            style = mulan(texts=args.text).half()  # [1, 512] — real MuLan text embedding
    else:
        # No text: use projector(image) as fallback style
        style = model.projector(z_img.float()).half()

    del mulan; torch.cuda.empty_cache()

    # --- Generate ---
    max_frames = cfg["model"]["max_frames"]
    gen_frames = target_frames if target_frames else max_frames
    gen_frames = min(gen_frames, max_frames)
    actual_dur = gen_frames * DOWNSAMPLE_RATIO / SAMPLE_RATE
    print(f"Generating music ({actual_dur:.1f}s, steps={infer_cfg['steps']}, cfg={infer_cfg['cfg_strength']})...")

    model._set_img_tokens(z_img_tokens)
    neg_style = get_negative_style_prompt(device)  # [1, 512]

    cond = torch.zeros(1, max_frames, 64, device=device, dtype=torch.half)
    lrc = torch.zeros(1, max_frames, dtype=torch.long, device=device)
    start_time = torch.zeros(1, device=device, dtype=torch.half)
    norm_duration = torch.tensor([gen_frames / max_frames], device=device, dtype=torch.half)

    out, _ = model.cfm.sample(
        cond=cond,
        text=lrc,
        duration=max_frames,
        style_prompt=style,
        negative_style_prompt=neg_style,
        steps=infer_cfg["steps"],
        cfg_strength=infer_cfg["cfg_strength"],
        seed=infer_cfg.get("seed"),
        start_time=start_time,
        latent_pred_segments=[(0, gen_frames)],
        song_duration=norm_duration,
    )

    # --- Decode to audio ---
    print("Decoding to audio...")
    generated = out[0]  # [1, T, 64]
    generated = generated[:, :gen_frames, :]
    latent = generated.float().permute(0, 2, 1)  # [1, 64, T]
    audio = decode_audio(latent, vae, chunked=True)  # [1, 2, samples]
    audio = audio.squeeze(0).float().cpu()  # [2, samples]
    target_samples = int(actual_dur * SAMPLE_RATE)
    audio = audio[:, :target_samples]

    max_amp = audio.abs().max()
    if max_amp > 0:
        audio = audio / max_amp * 0.95
    audio = audio.clamp(-1, 1)
    audio_int16 = (audio * 32767).numpy().astype("int16")

    import soundfile as sf
    sf.write(args.output, audio_int16.T, 44100, subtype="PCM_16")

    duration_sec = audio.shape[-1] / 44100
    print(f"Saved: {args.output} ({duration_sec:.1f}s, stereo, 44100 Hz)")


if __name__ == "__main__":
    main()
