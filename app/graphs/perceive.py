"""Cheap scene cues for Look routing. No VLM, no extra weights."""

from __future__ import annotations

import cv2
import numpy as np

from app.jobs.store import asset_path
from app.renderers.photo_look import LOOKS


def _has_face(bgr: np.ndarray) -> bool:
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            return False
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(24, 24))
        return len(faces) > 0
    except Exception:
        return False


def classify_scene(bgr: np.ndarray) -> dict:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_v = float(hsv[..., 2].mean())
    mean_s = float(hsv[..., 1].mean())
    mean_h = float(hsv[..., 0].mean())
    b_mean, g_mean, r_mean = [float(x) for x in cv2.mean(bgr)[:3]]
    face = _has_face(bgr)
    if mean_v < 55:
        scene = "night"
        suggest = "hk_night"
    elif face and r_mean >= b_mean:
        scene = "portrait"
        suggest = "film_portra"
    elif mean_h < 25 and mean_v > 90 and r_mean > b_mean + 15:
        scene = "golden"
        suggest = "golden_hour"
    elif mean_s < 45:
        scene = "muted"
        suggest = "cool_steel"
    elif b_mean > r_mean + 10:
        scene = "cool"
        suggest = "cinematic_teal_orange"
    else:
        scene = "landscape"
        suggest = "film_portra"
    return {
        "scene": scene,
        "suggest_style_id": suggest if suggest in LOOKS else "film_portra",
        "mean_v": round(mean_v, 2),
        "mean_s": round(mean_s, 2),
        "has_face": face,
        "actions": [f"apply_look:{suggest}", "keep_geometry"],
    }


def perceive_node(state: dict) -> dict:
    asset_id = state.get("asset_id")
    if not asset_id:
        return state
    bgr = cv2.imdecode(np.fromfile(str(asset_path(asset_id)), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {**state, "scene": "unknown"}
    info = classify_scene(bgr)
    style_id = state.get("style_id")
    reply = state.get("reply") or ""
    if not state.get("user_named_style") and not state.get("is_patch"):
        style_id = info["suggest_style_id"]
        name = LOOKS[style_id]["name"]
        reply = f"感知场景={info['scene']}，建议 Look {name}（{style_id}）。" + (
            "" if "将使用" in reply else f" {reply}"
        )
    return {
        **state,
        "style_id": style_id,
        "scene": info["scene"],
        "perceive": info,
        "reply": reply.strip(),
        "params": state.get("params") or {},
    }
