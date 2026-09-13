"""Pre-RVQ EnCodec latent cache for the hybrid CE+L2 loss.

Per bridge example: encode the SAME 10 s TARGET window as the encodec code
cache (win copied from it) with MusicGen's EnCodec encoder, STOPPING BEFORE
the quantizer -> continuous latent [128, 500] fp16.

Output: data/edit_dataset/bridge_cache/latent32/<example_id>.pt
  {"tgt": fp16 [128,500], "n_frames": int}

Resumable; run: conda run -n llama python -u src/edit_agent/precompute_latent.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

torch.backends.cudnn.enabled = False

CACHE = PROJECT_ROOT / "data/edit_dataset/bridge_cache"
OUT = CACHE / "latent32"

SR = 32000
WIN_S = 10.0
N_FRAMES = 500


class ClipDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import librosa
        e = self.rows[i]
        try:
            y, _ = librosa.load(PROJECT_ROOT / e["tgt_path"], sr=SR, mono=True,
                                offset=e["win"][0], duration=WIN_S)
            if len(y) < int(WIN_S * SR):
                y = np.pad(y, (0, int(WIN_S * SR) - len(y)))
            return {"id": e["example_id"], "wav": torch.from_numpy(y.astype("float32")),
                    "n_frames": e["n_frames"], "ok": True}
        except Exception as exc:  # noqa: BLE001
            return {"id": e["example_id"], "err": str(exc), "ok": False}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in open(CACHE / "examples.jsonl"):
        e = json.loads(line)
        if (OUT / f"{e['example_id']}.pt").exists():
            continue
        enc_p = CACHE / "encodec" / f"{e['example_id']}.pt"
        if not enc_p.exists():
            continue
        enc = torch.load(enc_p, weights_only=True)
        e["win"] = enc["win"]
        e["n_frames"] = enc["n_frames"]
        rows.append(e)
    print(f"latent32: {len(rows)} examples pending", flush=True)
    if not rows:
        return

    from transformers import MusicgenForConditionalGeneration
    enc = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-medium", cache_dir=str(PROJECT_ROOT / "weights"),
        torch_dtype=torch.float32).audio_encoder.to("cuda").eval()

    loader = DataLoader(ClipDataset(rows), batch_size=24, num_workers=12,
                        collate_fn=lambda b: b)
    done = 0
    with torch.no_grad():
        for batch in loader:
            good = [b for b in batch if b["ok"]]
            for b in batch:
                if not b["ok"]:
                    print(f"[error] {b['id']}: {b['err']}", flush=True)
            if good:
                wavs = torch.stack([b["wav"] for b in good]).unsqueeze(1).to("cuda")
                # encoder only, pre-quantizer: continuous latent [B,128,T]
                lat = enc.encoder(wavs)
                for j, b in enumerate(good):
                    torch.save({"tgt": lat[j, :, :N_FRAMES].half().cpu(),
                                "n_frames": int(b["n_frames"])},
                               OUT / f"{b['id']}.pt")
            done += len(batch)
            if done % 2400 < 24:
                print(f"latent {done}/{len(rows)}", flush=True)
    print("latent32 done", flush=True)


if __name__ == "__main__":
    main()
