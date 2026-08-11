"""Train the text-instruction MusicGen editor (Instruct-MusicGen baseline).

Usage:
  conda run -n llama python -u src/edit_agent/train_musicgen_text.py [--steps 40000]
  ... --limit 16 --max-steps 400 --lr 2e-4     # overfit smoke
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

from edit_agent.musicgen_text import build_text_editor  # noqa: E402
from edit_agent.musicgen_text_data import TextEditDataset, make_collate  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device, max_batches=25):
    model.eval()
    tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        loss = model(b["text_ids"].to(device), b["text_mask"].to(device),
                     b["src"].to(device), b["tgt"].to(device),
                     n_frames=b["n_frames"].to(device))
        tot += loss.item(); n += 1
    model.train()
    return tot / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    ckpt_dir = PROJECT_ROOT / "ckpts/edit_agent/musicgen_text"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/musicgen-medium", cache_dir=str(PROJECT_ROOT / "weights"))

    model = build_text_editor(torch.device(args.device))
    if args.init_from:
        model.load_adapter(PROJECT_ROOT / args.init_from, args.device)
        model.decoder.to(args.device, dtype=torch.bfloat16)
        print(f"warm-started from {args.init_from}", flush=True)
    trainable = model.trainable_parameters()
    print(f"trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M "
          f"(musicgen-medium + T5 frozen)", flush=True)

    collate = make_collate(tokenizer)
    train_ds = TextEditDataset("train", limit=args.limit)
    val_ds = TextEditDataset("val", limit=200)
    print(f"train {len(train_ds)} / val {len(val_ds)} examples", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
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
    while not done:
        for i, b in enumerate(train_loader):
            loss = model(b["text_ids"].to(args.device), b["text_mask"].to(args.device),
                         b["src"].to(args.device), b["tgt"].to(args.device),
                         n_frames=b["n_frames"].to(args.device))
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
                if step % 25 == 0:
                    print(f"step {step}/{total_steps} ce "
                          f"{sum(running)/len(running):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} "
                          f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
                    running = []
                if step % 1000 == 0:
                    vl = evaluate(model, val_loader, args.device)
                    print(f"[val] step {step}: ce {vl:.4f}", flush=True)
                if step % 2000 == 0:
                    model.save_adapter(ckpt_dir / f"step_{step}")
                if step >= total_steps:
                    done = True
                    break

    model.save_adapter(ckpt_dir / "final")
    vl = evaluate(model, val_loader, args.device)
    print(f"done: {step} steps, final val ce {vl:.4f}, saved to {ckpt_dir/'final'}",
          flush=True)


if __name__ == "__main__":
    main()
