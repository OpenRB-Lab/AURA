"""AURA Gradio web app — upload a track, type an instruction, get edited audio.

  CUDA_VISIBLE_DEVICES=0 GRADIO_PORT=7862 conda run -n llama python -u src/serving/gradio_app.py
"""
import os

import gradio as gr

from engine import AuraEngine

ENGINE = AuraEngine(device="cuda")


def run_edit(audio_path, instruction, executor, guidance, seed):
    if not audio_path:
        return None, "Please upload an audio file."
    if not (instruction or "").strip():
        return None, "Please type an edit instruction."
    r = ENGINE.edit(audio_path, instruction, executor=executor,
                    guidance=float(guidance), seed=int(seed))
    status = r["reply"]
    if r["wav"] is None:
        return None, f"(no edit emitted)\n{status}"
    return (r["sr"], r["wav"]), f"{status}\n\n[{executor}] {r['gen_time_s']}s  plan={r['plan']}"


with gr.Blocks(title="AURA — Conversational Music Editing") as demo:
    gr.Markdown("# AURA — Conversational Music Editing\n"
                "Upload a track, describe the edit (e.g. *remove the drums*, "
                "*add a soft pad*, *keep only the guitar*), and listen to the result.")
    with gr.Row():
        with gr.Column():
            in_audio = gr.Audio(label="Input track", type="filepath")
            instruction = gr.Textbox(label="Edit instruction",
                                     placeholder="remove the drums")
            with gr.Accordion("Options", open=False):
                executor = gr.Radio(["hybrid", "pure", "full"], value="hybrid",
                                    label="Executor")
                guidance = gr.Slider(1.0, 5.0, value=2.0, step=0.5, label="CFG scale")
                seed = gr.Number(value=1234, label="Seed", precision=0)
            btn = gr.Button("Edit", variant="primary")
        with gr.Column():
            out_audio = gr.Audio(label="Edited track")
            out_text = gr.Textbox(label="Assistant reply / details", lines=6)
    btn.click(run_edit, [in_audio, instruction, executor, guidance, seed],
              [out_audio, out_text])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("GRADIO_PORT", 7862)))
