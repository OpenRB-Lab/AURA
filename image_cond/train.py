"""Training loop for image-conditioned DiffRhythm with LoRA.

Uses PyTorch DDP for multi-GPU training.

Usage:
  # Single GPU
  python src/image_cond/train.py --config src/configs/image_cond.yaml

  # Multi-GPU (2 GPUs)
  torchrun --nproc_per_node=2 src/image_cond/train.py --config src/configs/image_cond.yaml
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False

import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

DIFFRHYTHM_ROOT = Path(__file__).resolve().parents[2] / "src" / "DiffRhythm"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DIFFRHYTHM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        raw = f.read()
    raw = raw.replace("${dataset_root}", str(PROJECT_ROOT / "data" / "image_music"))
    raw = raw.replace("${diffrhythm_root}", str(DIFFRHYTHM_ROOT))
    raw = raw.replace("${image_encoder.embed_dim}", "768")
    cfg = yaml.safe_load(raw)
    return cfg


class CachedImageMusicDataset(Dataset):
    def __init__(self, cache_dir: str, max_frames: int = 2048, manifest_file: str = None):
        self.cache_dir = Path(cache_dir)
        self.max_frames = max_frames
        self.manifest = []

        manifest_path = Path(manifest_file) if manifest_file else self.cache_dir / "index.jsonl"
        with open(manifest_path) as f:
            for line in f:
                self.manifest.append(json.loads(line))

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        data = torch.load(entry["cache_path"], map_location="cpu", weights_only=True)

        z_img = data["z_img"].float()
        latent = data["latent"].float()
        z_mulan = data["z_mulan_audio"].float()
        z_text = data.get("z_mulan_text", torch.zeros(512)).float()
        z_img_tokens = data.get("z_img_tokens", z_img.unsqueeze(0)).float()  # [S, 768]

        T = latent.shape[-1]
        if T > self.max_frames:
            start = torch.randint(0, T - self.max_frames, (1,)).item()
            latent = latent[:, start:start + self.max_frames]
            T = self.max_frames

        lrc = torch.zeros(self.max_frames, dtype=torch.long)
        return {
            "z_img": z_img,
            "z_img_tokens": z_img_tokens,
            "latent": latent,
            "z_mulan_audio": z_mulan,
            "z_mulan_text": z_text,
            "lrc": lrc,
            "latent_frames": T,
            "start_time": 0.0,
        }


def collate_fn(batch, max_frames=2048):
    z_imgs = torch.stack([b["z_img"] for b in batch])
    z_texts = torch.stack([b["z_mulan_text"] for b in batch])
    # Pad image tokens to same length (256 for paintings, 1 for suno fallback)
    max_img_tokens = max(b["z_img_tokens"].shape[0] for b in batch)
    z_img_tokens_list = []
    for b in batch:
        tokens = b["z_img_tokens"]
        pad_len = max_img_tokens - tokens.shape[0]
        if pad_len > 0:
            tokens = F.pad(tokens, (0, 0, 0, pad_len))
        z_img_tokens_list.append(tokens)
    z_img_tokens = torch.stack(z_img_tokens_list)
    lrcs = torch.stack([b["lrc"] for b in batch])
    z_mulans = torch.stack([b["z_mulan_audio"] for b in batch])
    start_times = torch.tensor([b["start_time"] for b in batch])
    latent_lengths = torch.tensor([b["latent_frames"] for b in batch], dtype=torch.long)

    padded_latents = []
    for b in batch:
        lat = b["latent"]
        pad_len = max_frames - lat.shape[-1]
        if pad_len > 0:
            lat = F.pad(lat, (0, pad_len))
        padded_latents.append(lat)
    latents = torch.stack(padded_latents)

    return {
        "z_img": z_imgs,
        "z_img_tokens": z_img_tokens,
        "z_mulan_text": z_texts,
        "latent": latents,
        "z_mulan_audio": z_mulans,
        "lrc": lrcs,
        "latent_lengths": latent_lengths,
        "start_time": start_times,
    }


def build_optimizer(model, lr_lora: float, lr_proj: float, weight_decay: float):
    lora_params, proj_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "projector" in name:
            proj_params.append(p)
        else:
            lora_params.append(p)
    return AdamW([
        {"params": lora_params, "lr": lr_lora},
        {"params": proj_params, "lr": lr_proj},
    ], weight_decay=weight_decay)


def get_cosine_schedule(optimizer, num_warmup_steps, num_training_steps):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def setup_ddp():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(cfg: dict):
    train_cfg = cfg["training"]
    torch.manual_seed(train_cfg["seed"])

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        print(f"Running on {world_size} GPU(s), precision: {train_cfg['precision']}")

    from src.image_cond.lora_dit import build_image_cond_model
    model = build_image_cond_model(cfg, device)

    model.cfm.audio_drop_prob = train_cfg["audio_drop_prob"]
    model.cfm.style_drop_prob = train_cfg["style_drop_prob"]
    model.cfm.lrc_drop_prob = train_cfg["lrc_drop_prob"]
    model.cfm.cond_drop_prob = train_cfg["cond_drop_prob"]

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if world_size > 1 else model

    dataset = CachedImageMusicDataset(
        cache_dir=cfg["cache_dir"],
        max_frames=cfg["model"]["max_frames"],
        manifest_file=cfg.get("manifest_file"),
    )
    max_frames = cfg["model"]["max_frames"]

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, max_frames=max_frames),
    )

    optimizer = build_optimizer(raw_model, train_cfg["learning_rate_lora"],
                                train_cfg["learning_rate_proj"], train_cfg["weight_decay"])

    total_steps = len(dataloader) * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    use_amp = train_cfg["precision"] in ("fp16", "bf16")
    amp_dtype = torch.float16 if train_cfg["precision"] == "fp16" else torch.bfloat16
    scaler = GradScaler(enabled=(train_cfg["precision"] == "fp16"))

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    grad_accum = train_cfg["grad_accumulation_steps"]

    for epoch in range(train_cfg["epochs"]):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}",
                        disable=not is_main)

        align_weight = train_cfg.get("align_weight", 1.0)
        accum_loss = accum_fm = accum_al = accum_cl = 0.0

        for batch_idx, batch in enumerate(progress):
            z_img = batch["z_img"].to(device)
            z_img_tokens = batch["z_img_tokens"].to(device)
            z_text = batch["z_mulan_text"].to(device)
            z_mulan = batch["z_mulan_audio"].to(device)
            mel_spec = batch["latent"].permute(0, 2, 1).to(device)
            text = batch["lrc"].to(device)
            lens = batch["latent_lengths"].to(device)
            start_time = batch["start_time"].to(device)

            contrastive_weight = train_cfg.get("contrastive_weight", 0.5)
            contrastive_temp = train_cfg.get("contrastive_temp", 0.07)

            with autocast(dtype=amp_dtype, enabled=use_amp):
                loss, fm_loss, align_loss, contra_loss = model(
                    inp=mel_spec, text=text, z_img=z_img, z_text=z_text,
                    z_img_tokens=z_img_tokens,
                    lens=lens, start_time=start_time,
                    z_mulan_target=z_mulan, align_weight=align_weight,
                    contrastive_weight=contrastive_weight,
                    contrastive_temp=contrastive_temp,
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            # Accumulate losses for logging
            accum_loss = accum_loss + loss.item() * grad_accum
            accum_fm = accum_fm + fm_loss.item()
            accum_al = accum_al + align_loss.item()
            accum_cl = accum_cl + contra_loss.item()

            if (batch_idx + 1) % grad_accum == 0:
                if train_cfg["max_grad_norm"] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        train_cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main:
                    avg_loss = accum_loss / grad_accum
                    avg_fm = accum_fm / grad_accum
                    avg_al = accum_al / grad_accum
                    avg_cl = accum_cl / grad_accum
                    progress.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        fm=f"{avg_fm:.3f}",
                        al=f"{avg_al:.3f}",
                        cl=f"{avg_cl:.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        step=global_step)

                    if global_step % train_cfg["save_every_steps"] == 0:
                        from src.image_cond.lora_dit import save_adapter
                        save_adapter(raw_model, str(checkpoint_dir / f"step_{global_step}"))

                accum_loss = accum_fm = accum_al = accum_cl = 0.0

    if is_main:
        from src.image_cond.lora_dit import save_adapter
        save_adapter(raw_model, str(checkpoint_dir / "final"))
        print(f"Training complete. Final checkpoint: {checkpoint_dir / 'final'}")

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/image_cond.yaml")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Override manifest file (e.g. index_suno.jsonl)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint save directory")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.manifest:
        cfg["manifest_file"] = args.manifest
    if args.checkpoint_dir:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    train(cfg)


if __name__ == "__main__":
    main()
