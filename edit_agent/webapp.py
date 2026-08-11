"""Gradio chat demo for the music-editing agent (runs in the `llama` env, GPU 2).

Upload a song -> beat-aligned chunks -> chat with the SFT'd Qwen2.5-Omni thinker about
one chunk (optionally with a reference image). When the model confirms an edit it emits
the [EDIT_0..7] block; the confirmation sentence is rendered to audio by the MelodyFlow
worker (:7861) as a baseline until the Stage-A bridge is trained.

Launch via src/scripts/webapp.sh (starts the worker first).
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gradio as gr  # noqa: E402
import librosa  # noqa: E402

from data_utils.chunk_audio import plan_chunks, save_chunk  # noqa: E402
from edit_agent.qwen_wrapper import (  # noqa: E402
    generate_with_edit_capture, load_audio_16k, load_thinker,
)
from edit_agent.sft_data import SYSTEM_PROMPT  # noqa: E402
from edit_agent.tokens import EDIT_BLOCK  # noqa: E402

import os as _os
WORKER_URL = _os.environ.get("WORKER_URL", "http://127.0.0.1:7861")
SFT_DIR = _os.environ.get("SFT_ADAPTER", str(PROJECT_ROOT / "ckpts/edit_agent/sft/final"))
SESSION_ROOT = Path(tempfile.mkdtemp(prefix="edit_agent_web_"))
SAMPLE_SONG = next((PROJECT_ROOT / "data/image_music/cached_latents/suno_audio").glob("*.mp3"))

print("loading SFT thinker ...", flush=True)
MODEL, PROC, EDIT_IDS = load_thinker(lora_dir=SFT_DIR)
MODEL.eval()
IM_END = PROC.tokenizer.convert_tokens_to_ids("<|im_end|>")
print("thinker ready", flush=True)


# ── chunking ─────────────────────────────────────────────────

def chunk_song(audio_path):
    if not audio_path:
        return gr.update(choices=[], value=None), None, "upload a song first", []
    y_mono, sr_a = librosa.load(audio_path, sr=22050, mono=True)
    dur = len(y_mono) / sr_a
    _, beats = librosa.beat.beat_track(y=y_mono, sr=sr_a)
    beat_times = librosa.frames_to_time(beats, sr=sr_a)
    spans = plan_chunks(np.asarray(beat_times), dur)

    y_full, sr_full = librosa.load(audio_path, sr=None, mono=False)
    y_full = np.atleast_2d(y_full)
    out_dir = SESSION_ROOT / f"song_{int(time.time())}"
    paths = []
    for k, (s, e) in enumerate(spans):
        p = out_dir / f"chunk_{k:02d}.wav"
        save_chunk(y_full, sr_full, s, e, p)
        paths.append(str(p))
    choices = [f"chunk {k} · {s:.0f}–{e:.0f}s" for k, (s, e) in enumerate(spans)]
    status = f"{len(spans)} chunks from {dur:.0f}s of audio — pick one and start chatting"
    return gr.update(choices=choices, value=choices[0]), paths[0], status, paths


def select_chunk(choice, chunk_paths):
    if not choice or not chunk_paths:
        return None, []
    idx = int(choice.split()[1])
    return chunk_paths[idx], []  # reset chat on chunk switch


# ── chat ─────────────────────────────────────────────────────

def build_messages(history, chunk_path, image_path):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    for i, (role, text) in enumerate(history):
        if i == 0 and role == "user":
            content = [{"type": "audio", "audio": chunk_path}]
            if image_path:
                content.append({"type": "image", "image": image_path})
            content.append({"type": "text", "text": text})
        else:
            content = [{"type": "text", "text": text}]
        msgs.append({"role": role, "content": content})
    return msgs


def generate_reply(history, chunk_path, image_path):
    msgs = build_messages(history, chunk_path, image_path)
    text = PROC.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    audio = load_audio_16k(chunk_path)
    images = None
    if image_path:
        from PIL import Image
        images = [Image.open(image_path).convert("RGB")]
    inputs = PROC(text=[text], audio=[audio], images=images,
                  return_tensors="pt", padding=True).to("cuda")
    reply, h = generate_with_edit_capture(MODEL, PROC, inputs, EDIT_IDS,
                                          max_new_tokens=256, eos_token_id=IM_END)
    return reply, h


import re as _re

_SEG_RE = _re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:s\b|sec\w*)?\s*(?:to|-|–|through|until)\s*"
    r"(\d+(?:\.\d+)?)\s*(?:s\b|sec\w*)", _re.IGNORECASE)


def parse_segment(*texts) -> list | None:
    """Extract 'X to Y seconds' style spans so the renderer can edit locally."""
    for t in texts:
        m = _SEG_RE.search(t or "")
        if m:
            s, e = float(m.group(1)), float(m.group(2))
            if 0 <= s < e:
                return [s, e]
    return None


def render_edit(chunk_path, prompt, segment=None):
    r = requests.post(f"{WORKER_URL}/edit",
                      json={"wav_path": chunk_path, "prompt": prompt,
                            "segment": segment}, timeout=600)
    r.raise_for_status()
    return r.json()


def worker_has_bridge() -> bool:
    try:
        return bool(requests.get(f"{WORKER_URL}/health", timeout=3).json().get("bridge"))
    except Exception:  # noqa: BLE001
        return False


def render_edit_bridge(chunk_path, h, segment=None):
    import base64
    payload = base64.b64encode(h.half().cpu().numpy().tobytes()).decode()
    r = requests.post(f"{WORKER_URL}/edit_bridge",
                      json={"wav_path": chunk_path, "hidden_b64": payload,
                            "segment": segment}, timeout=600)
    r.raise_for_status()
    return r.json()


def chat(user_msg, display_history, raw_history, chunk_path, image_path):
    if not chunk_path:
        return display_history + [{"role": "assistant",
                                   "content": "Please load a song and pick a chunk first."}], \
               raw_history, None, ""
    if not user_msg.strip():
        return display_history, raw_history, None, ""

    raw_history = raw_history + [("user", user_msg)]
    reply, h = generate_reply(raw_history, chunk_path, image_path)
    raw_history = raw_history + [("assistant", reply)]

    edited_audio = None
    shown = reply
    if EDIT_BLOCK[:8] in reply:  # "[EDIT_0]" present
        confirmation = _re.sub(r"\[EDIT_[A-Z0-9]+\]", "", reply).strip()
        shown = confirmation + "\n\n🎛️ **edit captured** — rendering…"
        try:
            segment = parse_segment(confirmation, user_msg)
            use_bridge = h is not None and worker_has_bridge()
            if use_bridge:
                result = render_edit_bridge(chunk_path, h, segment)
                mode = "bridge (hidden-state conditioned DiffRhythm)"
            else:
                result = render_edit(chunk_path, confirmation, segment)
                mode = "DiffRhythm text baseline"
            edited_audio = result["edited_path"]
            seg_note = (f", segment {segment[0]:.0f}–{segment[1]:.0f}s preserved-outside"
                        if segment else "")
            shown = (confirmation + f"\n\n🎛️ **edit captured** — rendered in "
                     f"{result['gen_time_s']}s{seg_note} *({mode})*")
        except Exception as exc:  # noqa: BLE001 — surface render errors in chat
            shown = confirmation + f"\n\n⚠️ edit captured but render failed: {exc}"

    display_history = display_history + [{"role": "user", "content": user_msg},
                                         {"role": "assistant", "content": shown}]
    return display_history, raw_history, edited_audio, ""


def reset_chat():
    return [], [], None


# ── UI ───────────────────────────────────────────────────────

with gr.Blocks(title="Music Edit Agent") as demo:
    gr.Markdown("# 🎵 Music Edit Agent\nChat with the model about a chunk of your song; "
                "when an edit is agreed it emits `[EDIT]` tokens and renders the result.")
    chunk_paths_state = gr.State([])
    raw_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            song_in = gr.Audio(label="Song", type="filepath")
            sample_btn = gr.Button("Load sample song", size="sm")
            chunk_btn = gr.Button("Chunk song", variant="primary")
            status = gr.Markdown("")
            chunk_sel = gr.Radio(label="Chunk", choices=[])
            chunk_player = gr.Audio(label="Selected chunk", type="filepath", interactive=False)
            image_in = gr.Image(label="Reference image (optional)", type="filepath")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=420)
            msg_box = gr.Textbox(label="Message", placeholder="e.g. make the drums softer and add some reverb")
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                reset_btn = gr.Button("Reset chat")
            edited_player = gr.Audio(label="Edited chunk", type="filepath", interactive=False)

    sample_btn.click(lambda: str(SAMPLE_SONG), outputs=song_in)
    chunk_btn.click(chunk_song, inputs=song_in,
                    outputs=[chunk_sel, chunk_player, status, chunk_paths_state])
    chunk_sel.change(select_chunk, inputs=[chunk_sel, chunk_paths_state],
                     outputs=[chunk_player, raw_history])
    for trigger in (send_btn.click, msg_box.submit):
        trigger(chat, inputs=[msg_box, chatbot, raw_history, chunk_player, image_in],
                outputs=[chatbot, raw_history, edited_player, msg_box])
    reset_btn.click(reset_chat, outputs=[chatbot, raw_history, edited_player])

if __name__ == "__main__":
    import os
    port = int(os.environ.get("WEBAPP_PORT", "7860"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=False)
