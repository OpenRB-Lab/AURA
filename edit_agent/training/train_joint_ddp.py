"""JOINT training, optimized: micro-batching + DDP + semantic classifier.

Loss per batch:
  L = lam*CE_musicgen(live h) + (1-lam)*CE_LM(Qwen native)
      + cls_weight*(CE_kind + CE_inst)      [classifier on the live h]

Differences vs train_joint.py (batch-1, single-GPU):
- true micro-batches (--batch): one padded thinker forward per batch, one
  batched bridge forward, one batched classifier forward
- DDP over torchrun (rank-0 IO), DistributedSampler
- rows grouped so a batch is homogeneous in kind (edit vs qa) -> no wasted
  output_hidden_states on qa-only batches
- right padding so absolute edit-block positions stay valid

Usage:
  torchrun --nproc_per_node=2 src/edit_agent/train_joint_ddp.py \
      --arch fusion --mg-init ... --steps 20000 --batch 4 --cls-weight 0.1
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

torch.backends.cudnn.enabled = False
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from edit_agent.dataloaders.joint_data import JointDataset  # noqa: E402
from edit_agent.models.qwen_wrapper import load_audio_16k  # noqa: E402
from edit_agent.dataloaders.sft_data import SYSTEM_PROMPT, _assistant_label_mask  # noqa: E402
from edit_agent.training.train_joint import load_trainable_thinker  # noqa: E402

QWEN_DTYPE = torch.bfloat16


def prep_batch(proc, rows, device):
    """Padded multimodal batch. Returns (inputs, keep_mask [B,T] bool)."""
    texts, audios, images = [], [], []
    for ex in rows:
        msgs = [{"role": "system",
                 "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        msgs += ex["messages"]
        texts.append(proc.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=False))
        audios.append(load_audio_16k(ex["chunk_path"]))
        if ex.get("image_path"):
            from PIL import Image
            images.append(Image.open(PROJECT_ROOT / ex["image_path"]).convert("RGB"))
    proc.tokenizer.padding_side = "right"
    inputs = proc(text=texts, audio=audios, images=images or None,
                  return_tensors="pt", padding=True).to(device)
    return inputs


def batch_lm_and_hidden(model, proc, inputs, edit_ids, need_hidden):
    ids = inputs["input_ids"]
    labels = ids.clone()
    keep = torch.zeros_like(ids, dtype=torch.bool)
    for b in range(ids.shape[0]):
        keep[b] = _assistant_label_mask(ids[b].cpu(), proc.tokenizer).to(ids.device)
    if "attention_mask" in inputs:
        keep &= inputs["attention_mask"].bool()
    labels[~keep] = -100
    out = model(**inputs, labels=labels, output_hidden_states=need_hidden)
    hs, idx_ok = [], []
    if need_hidden:
        last = out.hidden_states[-1]
        for b in range(ids.shape[0]):
            pos = (ids[b] == edit_ids[0]).nonzero(as_tuple=True)[0]
            if not len(pos):
                continue
            s = int(pos[-1])
            h = last[b, s - 1: s - 1 + 9]
            if h.shape[0] == 9:
                hs.append(h)
                idx_ok.append(b)
    h_batch = torch.stack(hs) if hs else None
    return out.loss, h_batch, idx_ok


@torch.no_grad()
def evaluate(model, proc, bridge, rows, edit_ids, device, batch=2, max_items=48):
    model.eval(); bridge.eval()
    lm_t = mg_t = lm_n = mg_n = 0
    for i in range(0, min(len(rows), max_items), batch):
        chunk = rows[i:i + batch]
        kinds = {r["kind"] for r in chunk}
        if len(kinds) > 1:
            continue
        try:
            inputs = prep_batch(proc, chunk, device)
            lm, h, idx = batch_lm_and_hidden(model, proc, inputs, edit_ids,
                                             need_hidden=chunk[0]["kind"] == "edit")
            lm_t += lm.item(); lm_n += 1
            if h is not None:
                sel = [chunk[j] for j in idx if chunk[j]["enc"] is not None]
                if sel:
                    hsel = h[: len(sel)]
                    src = torch.stack([r["enc"]["src"].long() for r in sel]).to(device)
                    tgt = torch.stack([r["enc"]["tgt"].long() for r in sel]).to(device)
                    nf = torch.tensor([r["enc"]["n_frames"] for r in sel], device=device)
                    mg = bridge(hsel.float(), src, tgt, n_frames=nf)
                    mg_t += mg.item(); mg_n += 1
        except Exception:  # noqa: BLE001
            continue
    model.train(); bridge.train()
    return lm_t / max(lm_n, 1), mg_t / max(mg_n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qwen-init", default="ckpts/edit_agent/sft/final")
    p.add_argument("--mg-init", default="ckpts/edit_agent/musicgen_fusion/step_30000")
    p.add_argument("--arch", choices=["mem", "kv", "fusion"], default="fusion")
    p.add_argument("--lr-qwen", type=float, default=1e-5)
    p.add_argument("--lr-bridge", type=float, default=5e-5)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--cls-weight", type=float, default=0.1)
    p.add_argument("--latent-weight", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ckpt-dir", type=str, default="ckpts/edit_agent/joint_cls20k")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--resume", action="store_true",
                   help="continue from the latest step_* in --ckpt-dir")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world > 1
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = args.device
    is_main = rank == 0
    ckpt_dir = PROJECT_ROOT / args.ckpt_dir
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume and ckpt_dir.exists():
        steps_done = sorted(int(d.name.split("_")[1]) for d in ckpt_dir.glob("step_*")
                            if d.name.split("_")[1].isdigit())
        if steps_done:
            start_step = steps_done[-1]
            latest = ckpt_dir / f"step_{start_step}"
            args.qwen_init = str(latest / "qwen")
            args.mg_init = str(latest / "musicgen")
            if is_main:
                print(f"RESUMING from {latest} (step {start_step})", flush=True)

    model, proc, edit_ids = load_trainable_thinker(
        str(PROJECT_ROOT / args.qwen_init), device)
    from edit_agent.models.musicgen_fusion import build_fusion_bridge
    bridge = build_fusion_bridge(torch.device(device))
    if args.mg_init and Path(args.mg_init).exists():
        bridge.load_adapter(Path(args.mg_init), device)
        bridge.decoder.to(device, dtype=torch.bfloat16)
        if is_main:
            print(f"bridge warm-started from {args.mg_init}", flush=True)
    elif args.mg_init and (PROJECT_ROOT / args.mg_init).exists():
        bridge.load_adapter(PROJECT_ROOT / args.mg_init, device)
        bridge.decoder.to(device, dtype=torch.bfloat16)
        if is_main:
            print(f"bridge warm-started from {args.mg_init}", flush=True)
    bridge.latent_weight = args.latent_weight

    classifier = None
    if args.cls_weight > 0:
        from edit_agent.models.edit_classifier import (EditSemanticClassifier,
                                                       INST_TO_ID, KIND_TO_ID)
        classifier = EditSemanticClassifier().to(device)
        cls_p = Path(args.qwen_init).parent / "classifier.pt"
        if start_step and cls_p.exists():
            classifier.load_state_dict(torch.load(cls_p, map_location=device))
            if is_main:
                print("classifier state restored", flush=True)

    # NOTE: no DDP module wrappers. Hook-driven bucket all-reduce desyncs when
    # per-rank batch composition differs across three modules (observed:
    # mismatched bucket sizes at the same NCCL SeqNum). Gradients are synced
    # manually in a fixed parameter order at each accumulation boundary.
    raw_model, raw_bridge = model, bridge

    q_params = [pp for pp in raw_model.parameters() if pp.requires_grad]
    b_params = raw_bridge.trainable_parameters()
    if is_main:
        print(f"trainable: qwen {sum(x.numel() for x in q_params)/1e6:.1f}M, "
              f"bridge {sum(x.numel() for x in b_params)/1e6:.1f}M | "
              f"world={world} batch={args.batch} accum={args.accum} "
              f"(eff {world * args.batch * args.accum})", flush=True)

    train_ds = JointDataset("train", limit=args.limit)
    val_ds = JointDataset("val", limit=200)
    val_rows = [val_ds[i] for i in range(min(len(val_ds), 96))]
    if is_main:
        print(f"train {len(train_ds)} / val {len(val_ds)} rows", flush=True)

    sampler = None
    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                     shuffle=True)
    # batch_size*4 then split into kind-homogeneous micro-batches
    loader = DataLoader(train_ds, batch_size=args.batch * 4, sampler=sampler,
                        shuffle=(sampler is None), num_workers=2,
                        collate_fn=lambda b: b, pin_memory=False, drop_last=True)

    groups = [{"params": q_params, "lr": args.lr_qwen},
              {"params": b_params, "lr": args.lr_bridge}]
    if classifier is not None:
        groups.append({"params": list(classifier.parameters()), "lr": 1e-4})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    total = args.steps
    warmup = min(500, total // 20)

    def lam_fn(s):
        if s < warmup:
            return s / max(warmup, 1)
        pr = (s - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(pr, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam_fn)

    sync_params = q_params + b_params + \
        (list(classifier.parameters()) if classifier is not None else [])

    def sync_grads():
        if not ddp:
            return
        import torch.distributed as dist
        handles = []
        for p_ in sync_params:
            if p_.grad is None:
                p_.grad = torch.zeros_like(p_)
            handles.append(dist.all_reduce(p_.grad, op=dist.ReduceOp.AVG,
                                           async_op=True))
        for h_ in handles:
            h_.wait()
    model.train(); bridge.train()

    step, t0, micro = start_step, time.time(), 0
    for _ in range(start_step):
        sched.step()
    run_lm, run_mg, run_cls = [], [], []
    done, epoch = False, 0
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for big in loader:
            # split into kind-homogeneous micro-batches
            def _len(r):
                return sum(len(c.get("text", "")) for m in r["messages"]
                           for c in m["content"])
            # DDP-safe: mixed batches, length-sorted. Every micro-batch runs the
            # thinker AND the bridge/classifier (dummy if it holds no edit row),
            # so all ranks invoke the same modules in the same order.
            rows_all = sorted(big, key=_len)
            micro_batches = [rows_all[i:i + args.batch]
                             for i in range(0, len(rows_all), args.batch)]
            for rows in micro_batches:
                if not rows:
                    continue
                try:
                    inputs = prep_batch(proc, rows, device)
                    lm, h, idx = batch_lm_and_hidden(model, proc, inputs, edit_ids,
                                                     need_hidden=True)
                    loss = (1 - args.lam) * lm
                    keep_pos = [p_ for p_, j in enumerate(idx)
                                if rows[j].get("enc") is not None]
                    idx = [idx[p_] for p_ in keep_pos]
                    if h is not None and idx:
                        h = h[keep_pos]
                        sel = [rows[j] for j in idx]
                        src = torch.stack([r["enc"]["src"].long() for r in sel]).to(device)
                        tgt = torch.stack([r["enc"]["tgt"].long() for r in sel]).to(device)
                        nf = torch.tensor([r["enc"]["n_frames"] for r in sel],
                                          device=device)
                        mg = bridge(h.float(), src, tgt, n_frames=nf)
                        loss = loss + args.lam * mg
                        run_mg.append(mg.item())
                        if classifier is not None:
                            kl, il = classifier(h.float())
                            kt = torch.tensor([KIND_TO_ID.get(r.get("sem_kind"), -100)
                                               for r in sel], device=device)
                            it = torch.tensor([INST_TO_ID.get(r.get("sem_inst"), -100)
                                               for r in sel], device=device)
                            closs = torch.zeros((), device=device)
                            mk = kt >= 0
                            if mk.any():
                                closs = closs + torch.nn.functional.cross_entropy(
                                    kl[mk], kt[mk])
                            mi = it >= 0
                            if mi.any():
                                closs = closs + torch.nn.functional.cross_entropy(
                                    il[mi], it[mi])
                                run_cls.append(
                                    (il.argmax(-1)[mi] == it[mi]).float().mean().item())
                            loss = loss + args.cls_weight * closs
                    else:
                        # keep DDP in lockstep: invoke bridge+classifier with a
                        # zero batch, contributing exactly 0 to the loss
                        zh = torch.zeros(1, 9, 3584, device=device)
                        zc = torch.zeros(1, 4, 500, dtype=torch.long, device=device)
                        mg0 = bridge(zh, zc, zc,
                                     n_frames=torch.tensor([500], device=device))
                        loss = loss + 0.0 * mg0
                        if classifier is not None:
                            kl0, il0 = classifier(zh)
                            loss = loss + 0.0 * (kl0.sum() + il0.sum())
                    run_lm.append(lm.item())
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    if not isinstance(exc, torch.cuda.OutOfMemoryError) and \
                            "CUBLAS" not in str(exc) and "memory" not in str(exc):
                        raise
                    if is_main:
                        print("[warn] OOM micro-batch -> zero-loss pass", flush=True)
                    torch.cuda.empty_cache()
                    loss = None
                if loss is not None and not torch.isfinite(loss):
                    loss = None
                if loss is not None:
                    (loss / args.accum).backward()
                micro += 1
                if micro % args.accum:
                    continue
                sync_grads()
                gn = torch.nn.utils.clip_grad_norm_(q_params + b_params, 1.0)
                if torch.isfinite(gn):
                    opt.step()
                sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0 and is_main:
                    ca = (f" inst-acc {sum(run_cls)/len(run_cls):.2f}"
                          if run_cls else "")
                    print(f"step {step}/{total} lm {sum(run_lm)/max(len(run_lm),1):.4f} "
                          f"mg {sum(run_mg)/max(len(run_mg),1):.4f}{ca} "
                          f"lr {sched.get_last_lr()[0]:.2e} "
                          f"({(time.time()-t0)/max(step-start_step,1):.2f}s/step)", flush=True)
                    run_lm, run_mg, run_cls = [], [], []
                if step % 1000 == 0 and is_main:
                    vl, vm = evaluate(raw_model, proc, raw_bridge, val_rows, edit_ids, device,
                                      batch=args.batch)
                    print(f"[val] step {step}: lm {vl:.4f} mg {vm:.4f}", flush=True)
                if step % args.save_every == 0 and is_main:
                    d = ckpt_dir / f"step_{step}"
                    raw_model.save_pretrained(d / "qwen")
                    raw_bridge.save_adapter(d / "musicgen")
                    if classifier is not None:
                        torch.save((classifier.module if ddp else classifier).state_dict(),
                                   d / "classifier.pt")
                if step >= total:
                    done = True
                    break
            if done:
                break

    if is_main:
        raw_model.save_pretrained(ckpt_dir / "final" / "qwen")
        raw_bridge.save_adapter(ckpt_dir / "final" / "musicgen")
        if classifier is not None:
            torch.save((classifier.module if ddp else classifier).state_dict(),
                       ckpt_dir / "final" / "classifier.pt")
        vl, vm = evaluate(raw_model, proc, raw_bridge, val_rows, edit_ids, device,
                          batch=args.batch)
        print(f"done: {step} steps, final val lm {vl:.4f} mg {vm:.4f}, "
              f"saved to {ckpt_dir/'final'}", flush=True)
    if ddp:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
