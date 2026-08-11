"""Run FAD and CLAP score benchmark on generated music samples.

Usage:
  # With text prompts as companion .txt files (same name as .wav)
  python src/evaluation/run_benchmark.py \
      --generated results/cross_attn_mod/latest/ \
      --reference data/image_music/music_dataset/

  # With explicit prompts via manifest
  python src/evaluation/run_benchmark.py \
      --generated results/cross_attn_mod/latest/ \
      --reference data/image_music/music_dataset/ \
      --prompts "epic orchestral|calm piano|driving techno|sad violin"

  # FAD only (no text prompts needed)
  python src/evaluation/run_benchmark.py \
      --generated results/cross_attn_mod/latest/ \
      --reference data/image_music/music_dataset/ \
      --fad-only
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def collect_wav_files(directory: str) -> list[str]:
    """Recursively collect all .wav files."""
    return sorted([str(p) for p in Path(directory).rglob("*.wav")])


def load_prompts(generated_dir: str, prompts_arg: str = None) -> dict:
    """Load text prompts for each generated wav file.

    Looks for:
      1. --prompts arg (pipe-separated, matched by order)
      2. Companion .txt files (same name as .wav)
      3. prompts.json in the generated dir
    """
    wav_files = collect_wav_files(generated_dir)
    prompts = {}

    if prompts_arg:
        prompt_list = prompts_arg.split("|")
        for i, wav in enumerate(wav_files):
            if i < len(prompt_list):
                prompts[wav] = prompt_list[i].strip()
        return prompts

    # Try companion .txt files
    for wav in wav_files:
        txt = Path(wav).with_suffix(".txt")
        if txt.exists():
            prompts[wav] = txt.read_text().strip()

    # Try prompts.json
    prompts_json = Path(generated_dir) / "prompts.json"
    if prompts_json.exists():
        data = json.loads(prompts_json.read_text())
        for wav in wav_files:
            stem = Path(wav).stem
            if stem in data:
                prompts[wav] = data[stem]

    return prompts


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated music: FAD + CLAP score")
    parser.add_argument("--generated", type=str, required=True,
                        help="Directory with generated .wav files")
    parser.add_argument("--reference", type=str, required=True,
                        help="Directory with reference .wav files (for FAD)")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Pipe-separated text prompts for CLAP score")
    parser.add_argument("--fad-only", action="store_true",
                        help="Only compute FAD, skip CLAP score")
    parser.add_argument("--fad-model", type=str, default="vggish",
                        choices=["vggish", "pann", "clap", "encodec"],
                        help="Embedding model for FAD (default: vggish)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    results = {}

    # --- FAD ---
    print("=" * 50)
    print("FAD (Fréchet Audio Distance)")
    print("=" * 50)
    from src.evaluation.fad import compute_fad
    fad_score = compute_fad(
        generated_dir=args.generated,
        reference_dir=args.reference,
        model_name=args.fad_model,
    )
    results["fad"] = {"score": fad_score, "model": args.fad_model}
    print(f"FAD ({args.fad_model}): {fad_score:.4f}")
    print()

    # --- CLAP Score ---
    if not args.fad_only:
        print("=" * 50)
        print("CLAP Score (Text-Audio Similarity)")
        print("=" * 50)

        prompts = load_prompts(args.generated, args.prompts)
        if prompts:
            audio_paths = list(prompts.keys())
            text_prompts = list(prompts.values())

            from src.evaluation.clap_score import compute_clap_score
            clap_results = compute_clap_score(audio_paths, text_prompts, device=args.device)
            results["clap"] = clap_results
        else:
            print("No text prompts found. Skipping CLAP score.")
            print("Provide prompts via --prompts, companion .txt files, or prompts.json")

    # --- Summary ---
    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"FAD ({args.fad_model}): {results['fad']['score']:.4f}")
    if "clap" in results:
        print(f"CLAP score (mean): {results['clap']['mean']:.4f} ± {results['clap']['std']:.4f}")

    # --- Save ---
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
