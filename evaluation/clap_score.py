"""CLAP Score computation.

Computes text-audio similarity using the LAION-CLAP model.
Measures how well generated audio matches its text prompt.
"""

import numpy as np
import torch
from pathlib import Path

import laion_clap


_clap_model = None


def _get_clap_model(device: str = "cuda"):
    global _clap_model
    if _clap_model is None:
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
        _clap_model.load_ckpt()
    return _clap_model


def compute_clap_score(audio_paths: list[str], text_prompts: list[str],
                       device: str = "cuda", verbose: bool = True) -> dict:
    """
    Compute CLAP score for each (audio, text) pair.

    Args:
        audio_paths: list of paths to generated .wav files
        text_prompts: list of text prompts (same length as audio_paths)
        device: torch device
        verbose: print progress

    Returns:
        dict with "per_sample" scores and "mean" score
    """
    assert len(audio_paths) == len(text_prompts), \
        f"Mismatch: {len(audio_paths)} audio files vs {len(text_prompts)} prompts"

    if verbose:
        print(f"Computing CLAP score for {len(audio_paths)} samples")

    model = _get_clap_model(device)

    audio_embeddings = model.get_audio_embedding_from_filelist(
        x=audio_paths, use_tensor=True
    )
    text_embeddings = model.get_text_embedding(text_prompts, use_tensor=True)

    # Normalize
    audio_embeddings = audio_embeddings / audio_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    # Per-sample cosine similarity
    scores = (audio_embeddings * text_embeddings).sum(dim=-1).detach().cpu().numpy()

    results = {
        "per_sample": {Path(p).stem: float(s) for p, s in zip(audio_paths, scores)},
        "scores": [float(s) for s in scores],  # ordered like audio_paths (stems may repeat)
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }

    if verbose:
        print(f"  Mean CLAP score: {results['mean']:.4f} ± {results['std']:.4f}")
        for name, score in results["per_sample"].items():
            print(f"    {name}: {score:.4f}")

    return results
