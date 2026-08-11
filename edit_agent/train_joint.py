"""JOINT training: Qwen-Omni thinker (LoRA, live) + MusicGen edit bridge.

Loss per example:
  edit rows:  L = CE_musicgen(tgt codes | [proj(h_live); proj(src codes)])
                  + w_lm * CE_lm(assistant tokens)
  qa rows:    L = w_lm * CE_lm(assistant tokens)       (music Q&A, no edit)

h_live = last-layer states at the typed [EDIT_<KIND>][EDIT_0..7] block of the
teacher-forced dialogue (same positions as the cached hidden phase) — gradients
flow from the MusicGen CE back into the thinker so the edit tokens adapt to the
renderer, while the LM loss keeps the thinker conversational about the music.

Usage:
  conda run -n llama python -u src/edit_agent/train_joint.py \
      --mg-init ckpts/edit_agent/musicgen_bridge/final [--limit 16 --max-steps 200]
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

from edit_agent.joint_data import JointDataset  # noqa: E402
from edit_agent.musicgen_bridge import build_musicgen_bridge  # noqa: E402
from edit_agent.qwen_wrapper import load_audio_16k  # noqa: E402
from edit_agent.sft_data import SYSTEM_PROMPT, _assistant_label_mask  # noqa: E402

QWEN_DTYPE = torch.bfloat16


def load_trainable_thinker(lora_dir: str, device: str):
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from peft import PeftModel
    from edit_agent.qwen_wrapper import WEIGHTS_CACHE, DEFAULT_MODEL, add_edit_tokens
    proc = Qwen2_5OmniProcessor.from_pretrained(DEFAULT_MODEL, cache_dir=WEIGHTS_CACHE)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        DEFAULT_MODEL, cache_dir=WEIGHTS_CACHE, torch_dtype=QWEN_DTYPE,
        attn_implementation="sdpa")
    edit_ids = add_edit_tokens(proc, model)
    model = PeftModel.from_pretrained(model, lora_dir, is_trainable=True)
    return model.to(device), proc, edit_ids


def prep_inputs(proc, ex, device):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    msgs += ex["messages"]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    images = None
    if ex.get("image_path"):
        from PIL import Image
        images = [Image.open(PROJECT_ROOT / ex["image_path"]).convert("RGB")]
    inputs = proc(text=[text], audio=[load_audio_16k(ex["chunk_path"])],
                  images=images, return_tensors="pt", padding=True).to(device)
    return inputs


def lm_loss_and_hidden(model, proc, inputs, edit_ids, need_hidden: bool):
    ids = inputs["input_ids"][0]
    keep = _assistant_label_mask(ids.cpu(), proc.tokenizer).to(ids.device)
    labels = ids.clone()
    labels[~keep] = -100
    out = model(**inputs, labels=labels.unsqueeze(0),
                output_hidden_states=need_hidden)
    h = None
    if need_hidden:
        starts = (ids == edit_ids[0]).nonzero(as_tuple=True)[0]
        if len(starts):
            s = int(starts[-1])
            h = out.hidden_states[-1][0, s - 1: s - 1 + 9]
            if h.shape[0] != 9:
                h = None
    return out.loss, h


@torch.no_grad()
def evaluate(model, proc, bridge, loader, edit_ids, device, max_items=60):
    model.eval(); bridge.eval()
    lm_t = mg_t = lm_n = mg_n = 0
    for i, ex in enumerate(loader):
        if i >= max_items:
            break
        ex = ex[0]
        try:
            inputs = prep_inputs(proc, ex, device)
            lm, h = lm_loss_and_hidden(model, proc, inputs, edit_ids,
                                       need_hidden=ex["kind"] == "edit")
            lm_t += lm.item(); lm_n += 1
            if h is not None and ex["enc"] is not None:
                enc = ex["enc"]
                zt = ex.get("z_tgt")
                mg = bridge(h.float().unsqueeze(0),
                            enc["src"].long().unsqueeze(0).to(device),
                            enc["tgt"].long().unsqueeze(0).to(device),
                            n_frames=torch.tensor([enc["n_frames"]], device=device),
                            z_tgt=zt.unsqueeze(0).to(device) if zt is not None
                            else None)
                mg_t += mg.item(); mg_n += 1
        except Exception:  # noqa: BLE001
            continue
    model.train(); bridge.train()
    return lm_t / max(lm_n, 1), mg_t / max(mg_n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qwen-init", default="ckpts/edit_agent/sft/final")
    p.add_argument("--mg-init", default="ckpts/edit_agent/musicgen_bridge/final")
    p.add_argument("--arch", choices=["mem", "kv", "fusion"], default="mem")
    p.add_argument("--lr-qwen", type=float, default=1e-5)
    p.add_argument("--lr-bridge", type=float, default=5e-5)
    p.add_argument("--w-lm", type=float, default=1.0,
                   help="DEPRECATED: superseded by --lam convex combination")
    p.add_argument("--lam", type=float, default=0.5,
                   help="loss = lam*L_musicgen + (1-lam)*CE_LM (Qwen built-in)")
    p.add_argument("--latent-weight", type=float, default=0.5,
                   help="L2-latent weight inside L_musicgen (hybrid loss)")
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = args.device

    ckpt_dir = PROJECT_ROOT / "ckpts/edit_agent/joint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model, proc, edit_ids = load_trainable_thinker(
        str(PROJECT_ROOT / args.qwen_init), device)
    if args.arch == "fusion":
        from edit_agent.musicgen_fusion import build_fusion_bridge
        bridge = build_fusion_bridge(torch.device(device))
    elif args.arch == "kv":
        from edit_agent.musicgen_kv import build_kv_bridge
        bridge = build_kv_bridge(torch.device(device))
    else:
        bridge = build_musicgen_bridge(torch.device(device))
    if args.mg_init and (PROJECT_ROOT / args.mg_init).exists():
        bridge.load_adapter(PROJECT_ROOT / args.mg_init, device)
        bridge.decoder.to(device, dtype=torch.bfloat16)
        print(f"bridge warm-started from {args.mg_init}", flush=True)
    if args.latent_weight > 0 and hasattr(bridge, "latent_weight"):
        bridge.latent_weight = args.latent_weight
        print(f"latent L2 on in joint, weight {args.latent_weight}; "
              f"lam {args.lam}", flush=True)

    q_params = [pp for pp in model.parameters() if pp.requires_grad]
    b_params = bridge.trainable_parameters()
    print(f"trainable: qwen {sum(x.numel() for x in q_params)/1e6:.1f}M, "
          f"bridge {sum(x.numel() for x in b_params)/1e6:.1f}M", flush=True)

    train_ds = JointDataset("train", limit=args.limit)
    val_ds = JointDataset("val", limit=200)
    print(f"train {len(train_ds)} / val {len(val_ds)} rows", flush=True)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4,
                              collate_fn=lambda b: b)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2,
                            collate_fn=lambda b: b)

    total = args.max_steps or args.steps
    warmup = min(300, total // 10)
    opt = torch.optim.AdamW([{"params": q_params, "lr": args.lr_qwen},
                             {"params": b_params, "lr": args.lr_bridge}],
                            weight_decay=0.01)

    def lam(step):
        if step < warmup:
            return step / max(warmup, 1)
        pr = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(pr, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    model.train(); bridge.train()

    step, t0 = 0, time.time()
    run_lm, run_mg = [], []
    micro = 0
    done = False
    while not done:
        for ex in train_loader:
            ex = ex[0]
            try:
                inputs = prep_inputs(proc, ex, device)
                lm, h = lm_loss_and_hidden(model, proc, inputs, edit_ids,
                                           need_hidden=ex["kind"] == "edit")
                loss = (1 - args.lam) * lm
                if h is not None and ex["enc"] is not None:
                    enc = ex["enc"]
                    zt = ex.get("z_tgt")
                    mg = bridge(h.float().unsqueeze(0),
                                enc["src"].long().unsqueeze(0).to(device),
                                enc["tgt"].long().unsqueeze(0).to(device),
                                n_frames=torch.tensor([enc["n_frames"]], device=device),
                                z_tgt=zt.unsqueeze(0).to(device) if zt is not None
                                else None)
                    loss = loss + args.lam * mg
                    run_mg.append(mg.item())
                run_lm.append(lm.item())
            except torch.cuda.OutOfMemoryError:
                print("[warn] OOM, example skipped", flush=True)
                torch.cuda.empty_cache()
                opt.zero_grad(set_to_none=True)
                continue
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at step {step}, skipped", flush=True)
                continue
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(q_params + b_params, 1.0)
                if torch.isfinite(gn):
                    opt.step()
                else:
                    print(f"[warn] non-finite grad norm at step {step}", flush=True)
                sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    lm_a = sum(run_lm) / max(len(run_lm), 1)
                    mg_a = sum(run_mg) / max(len(run_mg), 1)
                    print(f"step {step}/{total} lm {lm_a:.4f} mg {mg_a:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} "
                          f"({(time.time()-t0)/step:.1f}s/step)", flush=True)
                    run_lm, run_mg = [], []
                if step % 500 == 0:
                    vl, vm = evaluate(model, proc, bridge, val_loader, edit_ids, device)
                    print(f"[val] step {step}: lm {vl:.4f} mg {vm:.4f}", flush=True)
                if step % 1000 == 0:
                    model.save_pretrained(ckpt_dir / f"step_{step}" / "qwen")
                    bridge.save_adapter(ckpt_dir / f"step_{step}" / "musicgen")
                if step >= total:
                    done = True
                    break

    model.save_pretrained(ckpt_dir / "final" / "qwen")
    bridge.save_adapter(ckpt_dir / "final" / "musicgen")
    vl, vm = evaluate(model, proc, bridge, val_loader, edit_ids, device)
    print(f"done: {step} steps, final val lm {vl:.4f} mg {vm:.4f}, "
          f"saved to {ckpt_dir/'final'}", flush=True)


if __name__ == "__main__":
    main()
