"""LoRA SFT of the Qwen2.5-Omni-7B thinker on the edit dialogues (Stage B.2).

Trains the model to chat about music chunks and emit the [EDIT_0..7] block when an
edit is agreed. CE loss on assistant tokens only. Audio tower and vision tower frozen;
LoRA on the text decoder; embed_tokens/lm_head trained so the new [EDIT] token rows
become meaningful.

Usage:
  python src/edit_agent/train_sft.py --config src/configs/edit_agent.yaml
  python src/edit_agent/train_sft.py --config ... --limit 64 --max-steps 50  # smoke
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.models.qwen_wrapper import load_thinker  # noqa: E402
from edit_agent.dataloaders.sft_data import DialogueDataset, make_collate  # noqa: E402

torch.backends.cudnn.enabled = False


def build_lora(model, cfg):
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"].get("dropout", 0.05),
        bias="none",
        # anchor to the text decoder: audio_tower/visual use different prefixes
        target_modules=r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                       r"|mlp\.(gate_proj|up_proj|down_proj))",
        modules_to_save=["embed_tokens", "lm_head"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="src/configs/edit_agent.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dialogues", type=str, default=None,
                        help="override paths.dialogues_jsonl")
    parser.add_argument("--init-lora", type=str, default=None,
                        help="warm-start: continue training this adapter "
                             "instead of creating a fresh LoRA")
    parser.add_argument("--ckpt-name", type=str, default="sft",
                        help="checkpoint subdir under ckpt_root")
    parser.add_argument("--lr", type=float, default=None, help="override sft.lr")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override sft.epochs")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sft = cfg["sft"]
    if args.lr:
        sft["lr"] = args.lr
    if args.epochs:
        sft["epochs"] = args.epochs
    if args.dialogues:
        cfg["paths"]["dialogues_jsonl"] = args.dialogues
    ckpt_dir = PROJECT_ROOT / cfg["paths"]["ckpt_root"] / args.ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model, processor, edit_ids = load_thinker(cfg["qwen"]["model_id"], device=args.device)
    print("edit token ids:", edit_ids)
    model.audio_tower.requires_grad_(False)
    if hasattr(model, "visual"):
        model.visual.requires_grad_(False)
    if args.init_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, PROJECT_ROOT / args.init_lora,
                                          is_trainable=True)
        model.print_trainable_parameters()
        print(f"warm-started LoRA from {args.init_lora}", flush=True)
    else:
        model = build_lora(model, sft)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    dialogues = PROJECT_ROOT / cfg["paths"]["dialogues_jsonl"]
    train_ds = DialogueDataset(dialogues, "train", limit=args.limit)
    val_ds = DialogueDataset(dialogues, "val", limit=200)
    collate = make_collate(processor)
    train_loader = DataLoader(train_ds, batch_size=sft["batch"], shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, collate_fn=collate)
    print(f"train {len(train_ds)} / val {len(val_ds)} dialogues")

    accum = sft["grad_accum"]
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = args.max_steps or steps_per_epoch * sft["epochs"]
    warmup = min(sft.get("warmup", 500), total_steps // 10)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=sft["lr"], weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    step = 0
    t0 = time.time()
    running = []
    done = False
    for epoch in range(sft["epochs"]):
        if done:
            break
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / accum).backward()
            running.append(out.loss.item())
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % sft.get("log_every", 10) == 0:
                    lr = sched.get_last_lr()[0]
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss {sum(running)/len(running):.4f} lr {lr:.2e} "
                          f"({(time.time()-t0)/step:.1f}s/step)", flush=True)
                    running = []
                if step % sft.get("save_every", 500) == 0:
                    sd = ckpt_dir / f"step_{step}"
                    model.save_pretrained(sd)
                    processor.save_pretrained(sd)
                    vl = evaluate(model, val_loader, args.device)
                    print(f"[val] step {step}: CE {vl:.4f}", flush=True)
                if step >= total_steps:
                    done = True
                    break

    final = ckpt_dir / "final"
    model.save_pretrained(final)
    processor.save_pretrained(final)
    vl = evaluate(model, val_loader, args.device)
    print(f"done: {step} steps, final val CE {vl:.4f}, saved to {final}", flush=True)


if __name__ == "__main__":
    main()
