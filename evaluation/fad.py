"""Fréchet Audio Distance (FAD) computation.

Computes FAD between a set of generated audio files and reference audio files
using VGGish embeddings.
"""

import os
from pathlib import Path
from frechet_audio_distance import FrechetAudioDistance


def compute_fad(generated_dir: str, reference_dir: str,
                model_name: str = "vggish", sample_rate: int = 16000,
                use_pca: bool = False, use_activation: bool = False,
                verbose: bool = True) -> float:
    """
    Compute FAD between generated and reference audio directories.

    Args:
        generated_dir: path to directory with generated .wav files
        reference_dir: path to directory with reference .wav files
        model_name: embedding model ("vggish", "pann", "clap", "encodec")
        sample_rate: audio sample rate for processing
        use_pca: whether to use PCA on embeddings
        use_activation: whether to use activation layer
        verbose: print progress

    Returns:
        FAD score (lower is better)
    """
    if verbose:
        gen_count = len(list(Path(generated_dir).rglob("*.wav")))
        ref_count = len(list(Path(reference_dir).rglob("*.wav")))
        print(f"Computing FAD ({model_name}): {gen_count} generated vs {ref_count} reference")

    fad = FrechetAudioDistance(
        model_name=model_name,
        sample_rate=sample_rate,
        use_pca=use_pca,
        use_activation=use_activation,
        verbose=verbose,
    )

    score = fad.score(generated_dir, reference_dir)
    return score
