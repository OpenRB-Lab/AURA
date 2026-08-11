"""Bridge training: [EDIT] hidden states -> frozen DiffRhythm (Stage A).

Trainable: projector + per-layer cross-attn adapters + h_norm (~110M params).
DiffRhythm DiT/CFM fully frozen. Loss = flow + 1.0*align + 0.5*InfoNCE.

Usage:
  python src/edit_agent/train_bridge.py --config src/configs/edit_agent.yaml
  python src/edit_agent/train_bridge.py --limit 16 --max-steps 2000   # overfit smoke
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "DiffRhythm"))

torch.backends.cudnn.enabled = False
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from edit_agent.bridge import build_bridge, save_adapter  # noqa: E402
from edit_agent.bridge_data import BridgeDataset, collate  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device, max_batches=40):
    model.projector.eval(); model.cross_attn.eval()
    tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        flow, align, nce = model(
            b["src"].to(device), b["tgt"].to(device), b["h"].to(device),
            b["mulan"].to(device), b["lens"].to(device),
            torch.zeros(b["src"].shape[0], device=device, dtype=torch.half),
            pred_mask=b["pred_mask"].to(device))
        tot += flow.item(); n += 1
    model.projector.train(); model.cross_attn.train()
    return tot / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="src/configs/edit_agent.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--init-from", type=str, default=None,
                        help="adapter checkpoint dir to warm-start from")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--w-align", type=float, default=None)
    parser.add_argument("--w-nce", type=float, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(PROJECT_ROOT / args.config))
    bcfg = cfg["bridge"]
    ckpt_dir = PROJECT_ROOT / cfg["paths"]["ckpt_root"] / "bridge"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_bridge(torch.device(args.device))
    if args.init_from:
        from edit_agent.bridge import load_adapter
        load_adapter(model, PROJECT_ROOT / args.init_from, args.device)
        print(f"warm-started from {args.init_from}", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_tr/1e6:.1f}M (DiffRhythm frozen)", flush=True)

    train_ds = BridgeDataset("train", limit=args.limit)
    val_ds = BridgeDataset("val", limit=200)
    print(f"train {len(train_ds)} / val {len(val_ds)} examples", flush=True)
    train_loader = DataLoader(train_ds, batch_size=bcfg.get("batch", 6), shuffle=True,
                              num_workers=4, collate_fn=collate, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=2, collate_fn=collate)

    accum = bcfg.get("grad_accum", 2)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = args.max_steps or bcfg.get("steps", 25000)
    warmup = min(500, total_steps // 10)
    w_align = args.w_align if args.w_align is not None else bcfg["loss_w"].get("align", 1.0)
    w_nce = args.w_nce if args.w_nce is not None else bcfg["loss_w"].get("nce", 0.5)

    lr = args.lr if args.lr is not None else bcfg.get("lr", 1e-4)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    step, t0, running = 0, time.time(), []
    done = False
    while not done:
        for i, b in enumerate(train_loader):
            flow, align, nce = model(
                b["src"].to(args.device), b["tgt"].to(args.device),
                b["h"].to(args.device), b["mulan"].to(args.device),
                b["lens"].to(args.device),
                torch.zeros(b["src"].shape[0], device=args.device, dtype=torch.half),
                pred_mask=b["pred_mask"].to(args.device))
            loss = flow + w_align * align + w_nce * nce
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at step {step}, batch skipped", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            (loss / accum).backward()
            running.append((flow.item(), align.item()))
            if (i + 1) % accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if torch.isfinite(gn):
                    opt.step()
                else:
                    print(f"[warn] non-finite grad norm at step {step}, "
                          f"update skipped", flush=True)
                sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    f = sum(r[0] for r in running) / len(running)
                    a = sum(r[1] for r in running) / len(running)
                    print(f"step {step}/{total_steps} flow {f:.4f} align {a:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} "
                          f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
                    running = []
                if step % 1000 == 0:
                    vl = evaluate(model, val_loader, args.device)
                    print(f"[val] step {step}: flow {vl:.4f}", flush=True)
                if step % 2000 == 0:
                    save_adapter(model, ckpt_dir / f"step_{step}")
                if step >= total_steps:
                    done = True
                    break

    save_adapter(model, ckpt_dir / "final")
    vl = evaluate(model, val_loader, args.device)
    print(f"done: {step} steps, final val flow {vl:.4f}, saved to {ckpt_dir/'final'}",
          flush=True)


if __name__ == "__main__":
    main()
