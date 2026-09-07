from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.graphs.agent import run_agent
from app.jobs.store import asset_path, init_db, new_id, register_asset
from app.settings import settings

init_db()


def _save_numpy(image: np.ndarray) -> str:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    asset_id = new_id("a")
    path = settings.assets_dir / f"{asset_id}.jpg"
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    path.write_bytes(buf.tobytes())
    register_asset(path, kind="upload", asset_id=asset_id)
    return asset_id


def run_ui(image: np.ndarray | None, message: str):
    if image is None:
        state = run_agent(message or "介绍一下", None)
        return None, None, state.get("reply") or ""
    asset_id = _save_numpy(image)
    state = run_agent(message or "胶片暖调", asset_id)
    out_img = None
    if state.get("job_id"):
        from app.jobs.store import get_job

        job = get_job(state["job_id"])
        if job and job.outputs:
            p = asset_path(job.outputs[0].asset_id)
            out_img = np.array(Image.open(p).convert("RGB"))
    src = image if image.ndim == 3 else image
    return src, out_img, state.get("reply") or ""


def launch():
    import os

    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    import gradio as gr

    with gr.Blocks(title="StyleForge 一天闭环") as demo:
        gr.Markdown(
            "## StyleForge · 一天闭环\n"
            "上传照片，说「胶片暖调 / 电影青橙 / 港风夜景」。"
            "LangGraph：`route → execute`。调色：OpenCV，0 元。"
            "思路参考 [PhotoAgent](https://mdyao.github.io/PhotoAgent/)。"
        )
        with gr.Row():
            inp = gr.Image(label="原图", type="numpy")
            out_src = gr.Image(label="输入预览")
            out = gr.Image(label="Look 结果")
        msg = gr.Textbox(label="指令", value="做成胶片暖调，脸别磨皮")
        btn = gr.Button("运行 Agent", variant="primary")
        reply = gr.Textbox(label="Agent 回复", lines=4)
        btn.click(run_ui, inputs=[inp, msg], outputs=[out_src, out, reply])
    print("starting gradio on http://127.0.0.1:7860", flush=True)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    launch()
