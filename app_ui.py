from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

from app.graphs.agent import run_agent
from app.jobs.store import asset_path, get_job, init_db, new_id, register_asset
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


def run_ui(image: np.ndarray | None, message: str, thread_id: str | None):
    thread_id = thread_id or None
    asset_id = _save_numpy(image) if image is not None else None
    state = run_agent(message or "介绍一下", asset_id, thread_id)
    out_img = None
    compare = None
    if state.get("job_id"):
        job = get_job(state["job_id"])
        if job:
            for item in job.outputs:
                p = asset_path(item.asset_id)
                arr = np.array(Image.open(p).convert("RGB"))
                if item.role == "compare":
                    compare = arr
                elif item.role == "result":
                    out_img = arr
    src = image
    return src, out_img, compare, state.get("reply") or "", state.get("thread_id")


def launch():
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    import gradio as gr

    with gr.Blocks(title="StyleForge LUT 闭环") as demo:
        gr.Markdown(
            "## StyleForge · LUT 摄影 Look\n"
            "上传照片，说「胶片暖调 / 电影青橙 / 港风夜景」，再试「再暗一点」。"
            "调色使用 **.cube 3D LUT**（格式对齐 [CubeLUT](https://cubelut.cn/index.php) / Premiere Lumetri），"
            "LangGraph：`route → execute`。无图可问「水彩和水墨差在哪」。"
            "说「改成水墨画」走 image.2d（需硅基流动 Key，超日预算会熔断）。Look 路径 0 元。"
        )
        thread = gr.State(value=None)
        with gr.Row():
            inp = gr.Image(label="原图", type="numpy")
            out = gr.Image(label="Look 结果")
            cmp = gr.Image(label="原图 | 结果")
        msg = gr.Textbox(label="指令", value="做成胶片暖调")
        btn = gr.Button("运行 Agent", variant="primary")
        reply = gr.Textbox(label="Agent 回复", lines=3)
        tid = gr.Textbox(label="thread_id（多轮自动复用）", interactive=False)
        preview = gr.Image(label="输入预览", visible=False)

        def _wrap(image, message, thread_id):
            src, out_img, compare, text, new_tid = run_ui(image, message, thread_id)
            return src, out_img, compare, text, new_tid, new_tid

        btn.click(
            _wrap,
            inputs=[inp, msg, thread],
            outputs=[preview, out, cmp, reply, tid, thread],
        )
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
