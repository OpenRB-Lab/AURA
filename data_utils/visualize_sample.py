"""Visualize a sample from the ontocord/VALID dataset by sample index or video_id.

Supports:
  --save output.png        Static image with metadata + frames
  --save output.mp4        Video composited from frames + audio via ffmpeg
  --save output.html       Interactive HTML player with embedded video/audio
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "VALID" / "data"


def load_sample_by_index(data_dir: Path, idx: int):
    files = sorted(data_dir.glob("*.parquet"))
    cumulative = 0
    for f in files:
        table = pq.read_table(f)
        if cumulative + table.num_rows > idx:
            local_idx = idx - cumulative
            row = {col: table.column(col)[local_idx].as_py() for col in table.column_names}
            row["_shard"] = f.name
            row["_global_idx"] = idx
            return row
        cumulative += table.num_rows
    raise IndexError(f"Index {idx} out of range (total rows < {idx + 1})")


def load_samples_by_video_id(data_dir: Path, video_id: str):
    files = sorted(data_dir.glob("*.parquet"))
    rows = []
    for f in files:
        table = pq.read_table(f)
        vid_col = table.column("video_id")
        for i in range(table.num_rows):
            if vid_col[i].as_py() == video_id:
                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                row["_shard"] = f.name
                rows.append(row)
    if not rows:
        raise KeyError(f"video_id '{video_id}' not found")
    return rows


def parse_media(chunk_media_str: str):
    media = json.loads(chunk_media_str) if chunk_media_str else {}
    images, audios = {}, {}
    for tag, path in media.items():
        if "image" in tag:
            images[tag] = path
        elif "audio" in tag:
            audios[tag] = path
    return images, audios


def resolve_media(images: dict, audios: dict, media_root: Path):
    loaded_images = {}
    for tag, rel_path in sorted(images.items(), key=lambda x: _tag_sort_key(x[0])):
        img_path = media_root / rel_path
        if img_path.exists():
            loaded_images[tag] = img_path

    loaded_audios = {}
    for tag, rel_path in sorted(audios.items(), key=lambda x: _tag_sort_key(x[0])):
        audio_path = media_root / rel_path
        if audio_path.exists():
            loaded_audios[tag] = audio_path

    return loaded_images, loaded_audios


def _tag_sort_key(tag: str):
    import re
    m = re.search(r"(\d+)", tag)
    return int(m.group(1)) if m else 0


def _build_metadata_text(row: dict, images: dict, audios: dict):
    metadata = json.loads(row["video_metadata"]) if row.get("video_metadata") else {}
    video_id = row.get("video_id", "N/A")
    chunk_idx = row.get("chunk_idx", "N/A")
    language = row.get("video_language", "N/A")
    shard = row.get("_shard", "N/A")
    global_idx = row.get("_global_idx", "")

    title = f"video_id: {video_id}  |  chunk: {chunk_idx}  |  lang: {language}  |  shard: {shard}"
    if global_idx != "":
        title = f"[idx {global_idx}]  " + title

    lines = [title, ""]

    if metadata.get("params"):
        p = metadata["params"]
        lines.append(f"duration: {p.get('duration', '?')}s  |  views: {p.get('view_count', '?')}  |  channel: {p.get('channel_id', '?')}")
        lines.append("")

    if audios:
        lines.append(f"Audio: {', '.join(f'{t} -> {p}' for t, p in audios.items())}")
    if images:
        lines.append(f"Images: {', '.join(f'{t} -> {p}' for t, p in images.items())}")
    if audios or images:
        lines.append("")

    text = row.get("chunk_text", "")
    wrapped = textwrap.fill(text[:1500], width=120)
    if len(text) > 1500:
        wrapped += "\n... [truncated]"
    lines.append("Text:")
    lines.append(wrapped)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output: static PNG
# ---------------------------------------------------------------------------

def save_static(row: dict, data_dir: Path, save_path: str | None):
    images, audios = parse_media(row.get("chunk_media", "{}"))
    media_root = data_dir.parent
    loaded_images, _ = resolve_media(images, audios, media_root)

    info_text = _build_metadata_text(row, images, audios)
    n_images = len(loaded_images)

    fig_height = 4
    if n_images > 0:
        fig_height += 3 * ((n_images + 2) // 3)

    fig = plt.figure(figsize=(14, fig_height))
    gs = gridspec.GridSpec(2 if n_images > 0 else 1, 1,
                           height_ratios=[1, max(n_images, 1)] if n_images > 0 else [1])

    ax_text = fig.add_subplot(gs[0])
    ax_text.axis("off")
    ax_text.text(0.02, 0.98, info_text, transform=ax_text.transAxes,
                 fontsize=8, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    if n_images > 0:
        cols = min(n_images, 3)
        nrows = (n_images + cols - 1) // cols
        gs_img = gridspec.GridSpecFromSubplotSpec(nrows, cols, subplot_spec=gs[1],
                                                  wspace=0.05, hspace=0.15)
        for i, (tag, img_path) in enumerate(loaded_images.items()):
            r, c = divmod(i, cols)
            ax = fig.add_subplot(gs_img[r, c])
            ax.imshow(Image.open(img_path))
            ax.set_title(tag, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved static image to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output: MP4 video (frames + audio merged via ffmpeg)
# ---------------------------------------------------------------------------

def save_video(row: dict, data_dir: Path, save_path: str):
    images, audios = parse_media(row.get("chunk_media", "{}"))
    media_root = data_dir.parent
    loaded_images, loaded_audios = resolve_media(images, audios, media_root)

    if not loaded_images and not loaded_audios:
        print("No media files found on disk — cannot create video. Falling back to static image.")
        save_static(row, data_dir, save_path.replace(".mp4", ".png"))
        return

    tmpdir = tempfile.mkdtemp(prefix="valid_viz_")
    try:
        _compose_video(row, loaded_images, loaded_audios, tmpdir, save_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _compose_video(row: dict, loaded_images: dict, loaded_audios: dict,
                   tmpdir: str, save_path: str):
    audio_path = None
    if loaded_audios:
        audio_path = _concat_audios(loaded_audios, tmpdir)

    if loaded_audios and not loaded_images:
        audio_duration = _get_duration(audio_path)
        _make_audio_only_video(row, audio_path, audio_duration, save_path)
        return

    if not loaded_audios and loaded_images:
        _make_slideshow(loaded_images, None, 3.0, save_path)
        return

    audio_duration = _get_duration(audio_path)
    n_frames = len(loaded_images)
    sec_per_frame = max(audio_duration / n_frames, 0.5)
    _make_slideshow(loaded_images, audio_path, sec_per_frame, save_path)


def _concat_audios(loaded_audios: dict, tmpdir: str) -> str:
    if len(loaded_audios) == 1:
        return str(next(iter(loaded_audios.values())))

    list_file = os.path.join(tmpdir, "audio_list.txt")
    with open(list_file, "w") as f:
        for tag, path in loaded_audios.items():
            f.write(f"file '{path}'\n")

    out = os.path.join(tmpdir, "concat_audio.ogg")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out],
        capture_output=True, check=True,
    )
    return out


def _get_duration(audio_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _make_slideshow(loaded_images: dict, audio_path: str | None,
                    sec_per_frame: float, save_path: str):
    tmpdir = tempfile.mkdtemp(prefix="valid_frames_")
    try:
        frame_paths = []
        target_size = None
        for i, (tag, img_path) in enumerate(loaded_images.items()):
            img = Image.open(img_path).convert("RGB")
            if target_size is None:
                w, h = img.size
                w = w + (w % 2)
                h = h + (h % 2)
                target_size = (w, h)
            img = img.resize(target_size, Image.LANCZOS)
            frame_file = os.path.join(tmpdir, f"frame_{i:04d}.png")
            img.save(frame_file)
            frame_paths.append(frame_file)

        fps = 1.0 / sec_per_frame

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%04d.png"),
        ]
        if audio_path:
            cmd += ["-i", audio_path, "-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", f"fps={max(fps, 1)},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            save_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"Saved video to {save_path}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _make_audio_only_video(row: dict, audio_path: str, duration: float, save_path: str):
    metadata = json.loads(row["video_metadata"]) if row.get("video_metadata") else {}
    video_id = row.get("video_id", "N/A")
    chunk_idx = row.get("chunk_idx", "N/A")
    label = f"video_id: {video_id} | chunk: {chunk_idx}"
    text = row.get("chunk_text", "")[:300].replace("'", "\\'").replace('"', '\\"')

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        f"color=c=0x1a1a2e:s=640x360:d={duration}",
        "-i", audio_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-vf", (
            f"drawtext=text='{label}':fontcolor=white:fontsize=18:"
            f"x=(w-text_w)/2:y=40,"
            f"drawtext=text='Audio-only chunk':fontcolor=gray:fontsize=14:"
            f"x=(w-text_w)/2:y=80"
        ),
        "-shortest", save_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    print(f"Saved audio-only video to {save_path}")


# ---------------------------------------------------------------------------
# Output: HTML player (embedded video/audio + metadata)
# ---------------------------------------------------------------------------

def save_html(row: dict, data_dir: Path, save_path: str):
    images, audios = parse_media(row.get("chunk_media", "{}"))
    media_root = data_dir.parent
    loaded_images, loaded_audios = resolve_media(images, audios, media_root)
    metadata = json.loads(row["video_metadata"]) if row.get("video_metadata") else {}
    video_id = row.get("video_id", "N/A")
    chunk_idx = row.get("chunk_idx", "N/A")
    language = row.get("video_language", "N/A")
    shard = row.get("_shard", "N/A")
    text = row.get("chunk_text", "")

    image_tags = ""
    for tag, img_path in loaded_images.items():
        data = base64.b64encode(img_path.read_bytes()).decode()
        suffix = img_path.suffix.lstrip(".")
        mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
        image_tags += f'<div class="frame"><img src="data:{mime};base64,{data}"/><span>{tag}</span></div>\n'

    audio_tags = ""
    for tag, audio_path in loaded_audios.items():
        data = base64.b64encode(audio_path.read_bytes()).decode()
        suffix = audio_path.suffix.lstrip(".")
        mime = f"audio/{suffix}" if suffix != "ogg" else "audio/ogg"
        audio_tags += (
            f'<div class="audio-item"><label>{tag} &mdash; {audios[tag]}</label>'
            f'<audio controls preload="metadata"><source src="data:{mime};base64,{data}" type="{mime}"></audio></div>\n'
        )

    params = metadata.get("params", {})
    meta_html = (
        f"<b>Duration:</b> {params.get('duration', '?')}s &nbsp;|&nbsp; "
        f"<b>Views:</b> {params.get('view_count', '?')} &nbsp;|&nbsp; "
        f"<b>Channel:</b> {params.get('channel_id', '?')}"
    )

    import html as html_mod
    escaped_text = html_mod.escape(text)

    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>VALID Sample: {video_id} chunk {chunk_idx}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2em auto; background: #fafafa; color: #222; }}
  h1 {{ font-size: 1.3em; }}
  .meta {{ background: #fff8e1; padding: 12px; border-radius: 6px; margin-bottom: 16px; font-size: 0.9em; }}
  .frames {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0; }}
  .frame {{ text-align: center; }}
  .frame img {{ max-width: 400px; max-height: 300px; border: 1px solid #ddd; border-radius: 4px; }}
  .frame span {{ display: block; font-size: 0.8em; color: #666; margin-top: 4px; }}
  .audio-item {{ margin: 8px 0; }}
  .audio-item label {{ display: block; font-size: 0.85em; color: #555; margin-bottom: 4px; }}
  .text-box {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 14px; white-space: pre-wrap;
               font-family: monospace; font-size: 0.85em; max-height: 400px; overflow-y: auto; }}
  .play-all {{ margin: 12px 0; }}
  .play-all button {{ padding: 8px 20px; font-size: 1em; cursor: pointer; border-radius: 4px; border: 1px solid #888; }}
</style></head><body>
<h1>VALID &mdash; {video_id} &nbsp; chunk {chunk_idx}</h1>
<div class="meta">
  <b>Language:</b> {language} &nbsp;|&nbsp; <b>Shard:</b> {shard}<br>
  {meta_html}
</div>

<h3>Frames ({len(loaded_images)} found on disk / {len(images)} referenced)</h3>
<div class="frames">{image_tags if image_tags else '<em>No image files found on disk</em>'}</div>

<h3>Audio ({len(loaded_audios)} found on disk / {len(audios)} referenced)</h3>
{audio_tags if audio_tags else '<em>No audio files found on disk</em>'}

{"<div class='play-all'><button onclick='playAll()'>Play All Audio Sequentially</button></div>" if loaded_audios else ""}

<h3>Chunk Text</h3>
<div class="text-box">{escaped_text}</div>

<script>
function playAll() {{
  const players = document.querySelectorAll('audio');
  let i = 0;
  function next() {{
    if (i < players.length) {{ players[i].play(); players[i].onended = () => {{ i++; next(); }}; }}
  }}
  next();
}}
</script>
</body></html>"""

    Path(save_path).write_text(html_content, encoding="utf-8")
    print(f"Saved HTML player to {save_path}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def visualize_sample(row: dict, data_dir: Path, save_path: str | None = None):
    if save_path is None:
        save_static(row, data_dir, None)
    elif save_path.endswith(".mp4"):
        save_video(row, data_dir, save_path)
    elif save_path.endswith(".html"):
        save_html(row, data_dir, save_path)
    else:
        save_static(row, data_dir, save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a sample from ontocord/VALID",
        epilog="Output format is inferred from --save extension: .png (static), .mp4 (video), .html (interactive player)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--idx", type=int, help="Global sample index")
    group.add_argument("--video-id", type=str, help="YouTube video ID to look up")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to VALID/data/")
    parser.add_argument("--save", type=str, default=None,
                        help="Save output (.png, .mp4, or .html). Omit to display with matplotlib.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if args.idx is not None:
        print(f"Loading sample at index {args.idx}...")
        row = load_sample_by_index(data_dir, args.idx)
        visualize_sample(row, data_dir, save_path=args.save)
    else:
        print(f"Searching for video_id '{args.video_id}'...")
        rows = load_samples_by_video_id(data_dir, args.video_id)
        print(f"Found {len(rows)} chunks for video_id '{args.video_id}'")
        for i, row in enumerate(rows):
            save = None
            if args.save:
                base, ext = os.path.splitext(args.save)
                save = f"{base}_chunk{i}{ext}"
            visualize_sample(row, data_dir, save_path=save)


if __name__ == "__main__":
    main()
