"""Look critique: luminance SSIM + geometry. Free; at most one retry."""

from __future__ import annotations

import cv2
import numpy as np

from app.jobs.store import asset_path, get_job, save_job
from app.settings import settings


def luma_ssim(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float64)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float64)
    mu_a, mu_b = ga.mean(), gb.mean()
    var_a, var_b = ga.var(), gb.var()
    cov = ((ga - mu_a) * (gb - mu_b)).mean()
    c1, c2 = 6.5025, 58.5225
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if den <= 1e-9:
        return 1.0
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / den)


def _read_bgr(asset_id: str) -> np.ndarray | None:
    path = asset_path(asset_id)
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def critique_look(source: np.ndarray, result: np.ndarray) -> dict:
    if source is None or result is None:
        return {"pass": False, "reason": "missing_image"}
    dh = abs(source.shape[0] - result.shape[0])
    dw = abs(source.shape[1] - result.shape[1])
    ssim = luma_ssim(source, result)
    passed = dh <= 2 and dw <= 2 and 0.35 <= ssim <= 0.997
    reason = "ok"
    if dh > 2 or dw > 2:
        reason = "geometry"
    elif ssim > 0.997:
        reason = "no_change"
    elif ssim < 0.35:
        reason = "too_far"
    return {"pass": passed, "ssim": round(ssim, 4), "reason": reason}


def critique_node(state: dict) -> dict:
    if not settings.enable_critique or state.get("intent") != "look":
        return {**state, "need_retry": False}
    job = get_job(state.get("job_id") or "")
    source_id = state.get("asset_id")
    result_id = None
    if job:
        for item in job.outputs:
            if item.role == "result":
                result_id = item.asset_id
                break
    src = _read_bgr(source_id) if source_id else None
    dst = _read_bgr(result_id) if result_id else None
    report = critique_look(src, dst)
    step = int(state.get("step") or 1)
    need_retry = (not report["pass"]) and step < int(settings.max_critique_step)
    params = dict(state.get("params") or {})
    if need_retry:
        params["lut_strength"] = 1.0
        params["exposure"] = float(params.get("exposure") or 0.0) + 0.05
    if job:
        job.trace.critique = report
        job.trace.retry_count = 0 if not need_retry else step
        if "critique" not in job.trace.actions:
            job.trace.actions.append("critique")
        if job.trace.params is None:
            job.trace.params = {}
        job.trace.params["critique"] = report
        save_job(job)
    extra = f" 评价 SSIM={report['ssim']} ({report['reason']})"
    if need_retry:
        extra += "，重试一次。"
    reply = (state.get("reply") or "") + extra
    return {
        **state,
        "params": params,
        "need_retry": need_retry,
        "step": step,
        "critique": report,
        "reply": reply,
    }
