"""Train the MusicGen edit bridge (LoRA cross-attn + projectors, base frozen).

Usage:
  conda run -n llama python -u src/edit_agent/train_musicgen_bridge.py
  ... --limit 16 --max-steps 500          # overfit smoke
  ... --init-from ckpts/... --lr 3e-5     # warm restart
DDP (2 GPUs):
  conda run -n llama torchrun --nproc_per_node=2 \
      src/edit_agent/train_musicgen_bridge.py --arch kv
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

torch.backends.cudnn.enabled = False
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from edit_agent.musicgen_bridge import build_musicgen_bridge  # noqa: E402
from edit_agent.musicgen_data import MusicGenBridgeDataset, collate  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device, max_batches=40, use_stem=False):
    model.eval()
    tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        sm = b["stem_mode"].to(device) if use_stem and "stem_mode" in b else None
        loss = model(b["h"].to(device), b["src"].to(device), b["tgt"].to(device),
                     n_frames=b["n_frames"].to(device),
                     z_tgt=b["z_tgt"].to(device) if "z_tgt" in b else None,
                     stem_mode=sm)
        tot += loss.item(); n += 1
    model.train()
    return tot / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--arch", choices=["mem", "kv", "fusion"], default="mem",
                   help="mem: projected-embedding memory; kv: per-layer K/V source fusion")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-mlp", action="store_true",
                   help="also LoRA the decoder fc1/fc2 layers (kv arch only)")
    p.add_argument("--ckpt-dir", type=str, default=None,
                   help="override checkpoint dir (protects prior runs)")
    p.add_argument("--data-filter", type=str, default=None,
                   help="keep only examples whose id contains this substring "
                        "(e.g. 'slakh')")
    p.add_argument("--ssm-weight", type=float, default=0.0,
                   help="weight of the structural self-similarity loss "
                        "(fusion arch only)")
    p.add_argument("--latent-weight", type=float, default=0.0,
                   help="weight of the hybrid L2 loss in EnCodec latent space "
                        "(fusion arch; needs latent32 cache)")
    p.add_argument("--silence-filter", action="store_true",
                   help="drop Slakh rows with >30%% silent source windows")
    p.add_argument("--stem-mode-mix", type=float, default=0.0,
                   help="fraction of stem-mode rows (isolated-stem targets, "
                        "fusion arch only; needs stem_encodec cache)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--freeze-film", action="store_true",
                   help="RQ4 ablation: zero+freeze FiLM+gate (no dual-stream "
                        "modulation; only cross-attn concept conditioning)")
    p.add_argument("--freeze-bifam", action="store_true",
                   help="RQ4 ablation: zero+freeze alpha1/alpha2 (no reference "
                        "signal into FiLM)")
    p.add_argument("--ref-fusion", default="bifam",
                   choices=["bifam", "none", "crossattn", "concat", "both"],
                   help="reference-fusion mechanism (fusion arch only)")
    args = p.parse_args()

    import os
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world > 1
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        args.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    is_main = rank == 0

    default_dir = {"kv": "ckpts/edit_agent/musicgen_kv",
                   "fusion": "ckpts/edit_agent/musicgen_fusion",
                   "mem": "ckpts/edit_agent/musicgen_bridge"}[args.arch]
    ckpt_dir = PROJECT_ROOT / (args.ckpt_dir or default_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.arch == "fusion":
        from edit_agent.musicgen_fusion import build_fusion_bridge
        raw_model = build_fusion_bridge(torch.device(args.device), lora_r=args.lora_r,
                                        lora_alpha=args.lora_alpha)
    elif args.arch == "kv":
        from edit_agent.musicgen_kv import build_kv_bridge
        raw_model = build_kv_bridge(torch.device(args.device), lora_r=args.lora_r,
                                    lora_alpha=args.lora_alpha, lora_mlp=args.lora_mlp)
    else:
        raw_model = build_musicgen_bridge(torch.device(args.device))
    if args.ssm_weight > 0 and hasattr(raw_model, "ssm_weight"):
        raw_model.ssm_weight = args.ssm_weight
        if is_main:
            print(f"ssm loss on, weight {args.ssm_weight}", flush=True)
    if args.latent_weight > 0 and hasattr(raw_model, "latent_weight"):
        raw_model.latent_weight = args.latent_weight
        if is_main:
            print(f"latent L2 loss on, weight {args.latent_weight}", flush=True)
    if args.init_from:
        raw_model.load_adapter(PROJECT_ROOT / args.init_from, args.device)
        raw_model.decoder.to(args.device, dtype=torch.bfloat16)
        if is_main:
            print(f"warm-started from {args.init_from}", flush=True)
    if args.arch == "fusion":
        # set AFTER warm-start so it overrides the loaded ckpt's ref_fusion
        raw_model.ref_fusion = args.ref_fusion
        if is_main:
            print(f"[ref-fusion] {raw_model.ref_fusion}", flush=True)
    if args.freeze_film:
        raw_model.gate.data.zero_(); raw_model.gate.requires_grad_(False)
        for m in raw_model.film:
            for p in m.parameters():
                p.data.zero_(); p.requires_grad_(False)
        if is_main:
            print("[ablate] FiLM+gate zeroed & frozen", flush=True)
    if args.freeze_bifam:
        raw_model.alpha1.data.zero_(); raw_model.alpha1.requires_grad_(False)
        raw_model.alpha2.data.zero_(); raw_model.alpha2.requires_grad_(False)
        if is_main:
            print("[ablate] alpha1/alpha2 zeroed & frozen", flush=True)
    model = raw_model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(raw_model, device_ids=[rank], find_unused_parameters=True)
    trainable = raw_model.trainable_parameters()
    if is_main:
        print(f"trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M "
              f"(musicgen-medium frozen), world={world}", flush=True)

    train_ds = MusicGenBridgeDataset("train", limit=args.limit,
                                     id_filter=args.data_filter,
                                     latent=args.latent_weight > 0,
                                     silence_filter=args.silence_filter,
                                     stem_mode_mix=args.stem_mode_mix)
    val_ds = MusicGenBridgeDataset("val", limit=200, id_filter=args.data_filter,
                                   latent=args.latent_weight > 0,
                                   silence_filter=args.silence_filter)
    val_stem_loader = None
    if args.stem_mode_mix > 0:
        val_stem_ds = MusicGenBridgeDataset("val", limit=200,
                                            id_filter=args.data_filter,
                                            stem_mode_mix=-1.0)
        val_stem_loader = DataLoader(val_stem_ds, batch_size=8, num_workers=2,
                                     collate_fn=collate)
    if is_main:
        print(f"train {len(train_ds)} / val {len(val_ds)} examples", flush=True)
    sampler = None
    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                     shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=6, collate_fn=collate, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=2, collate_fn=collate)

    total_steps = args.max_steps or args.steps
    warmup = min(500, total_steps // 10)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        pr = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(pr, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    model.train()

    step, t0, running = 0, time.time(), []
    done = False
    epoch = 0
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for i, b in enumerate(train_loader):
            loss = model(b["h"].to(args.device), b["src"].to(args.device),
                         b["tgt"].to(args.device),
                         n_frames=b["n_frames"].to(args.device),
                         z_tgt=b["z_tgt"].to(args.device) if "z_tgt" in b else None,
                         stem_mode=b["stem_mode"].to(args.device)
                         if args.stem_mode_mix > 0 else None)
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at step {step}, batch skipped", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            (loss / args.accum).backward()
            running.append(loss.item())
            if (i + 1) % args.accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if torch.isfinite(gn):
                    opt.step()
                else:
                    print(f"[warn] non-finite grad norm at step {step}", flush=True)
                sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0 and is_main:
                    print(f"step {step}/{total_steps} ce "
                          f"{sum(running)/len(running):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} "
                          f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
                    running = []
                if step % 1000 == 0 and is_main:
                    vl = evaluate(raw_model, val_loader, args.device)
                    msg = f"[val] step {step}: ce {vl:.4f}"
                    if val_stem_loader is not None:
                        vs = evaluate(raw_model, val_stem_loader, args.device,
                                      use_stem=True)
                        msg += f" stem_ce {vs:.4f}"
                    print(msg, flush=True)
                if step % 2000 == 0 and is_main:
                    raw_model.save_adapter(ckpt_dir / f"step_{step}")
                if step >= total_steps:
                    done = True
                    break

    if is_main:
        raw_model.save_adapter(ckpt_dir / "final")
        vl = evaluate(raw_model, val_loader, args.device)
        print(f"done: {step} steps, final val ce {vl:.4f}, saved to {ckpt_dir/'final'}",
              flush=True)
    if ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
