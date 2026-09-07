"""Traditional photo looks (OpenCV). Cost = 0. Inspired by PhotoAgent executor tools."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 2048

LOOKS = {
    "film_portra": {
        "name": "胶片暖调",
        "keywords": ("胶片", "暖调", "portra", "暖一点", "日系"),
        "params": {"exposure": 0.06, "warm": 0.08, "grain": 0.03},
    },
    "cinematic_teal_orange": {
        "name": "电影青橙",
        "keywords": ("电影", "青橙", "cinematic", "teal", "橙"),
        "params": {"shadow_teal": 0.08, "highlight_orange": 0.07, "contrast": 0.18},
    },
    "hk_night": {
        "name": "港风夜景",
        "keywords": ("港风", "夜景", "霓虹", "hk", "赛博夜"),
        "params": {"exposure": -0.08, "magenta": 0.05, "grain": 0.05},
    },
}


def load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def encode_jpeg(bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _s_curve(x: np.ndarray, amount: float) -> np.ndarray:
    # amount 0..0.4
    return np.clip(x + amount * (x - 0.5) * (1.0 - x) * 4.0, 0, 1)


def _grain(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    noise = np.random.default_rng(42).normal(0, amount, img.shape[:2]).astype(np.float32)
    return np.clip(img + noise[:, :, None], 0, 1)


def apply_look(bgr: np.ndarray, style_id: str) -> np.ndarray:
    if style_id not in LOOKS:
        raise ValueError(f"unknown look: {style_id}")
    img = np.clip(bgr.astype(np.float32) / 255.0, 0, 1)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    luma = 0.114 * b + 0.587 * g + 0.299 * r

    if style_id == "film_portra":
        r = np.clip(r * 1.10 + 0.02, 0, 1)
        g = np.clip(g * 1.04, 0, 1)
        b = np.clip(b * 0.92, 0, 1)
        stacked = np.stack([b, g, r], axis=-1)
        stacked = np.clip(stacked * 1.06, 0, 1)
        stacked = _s_curve(stacked, 0.12)
        stacked = _grain(stacked, 0.028)
    elif style_id == "cinematic_teal_orange":
        shadow = np.clip(1.0 - luma * 1.4, 0, 1)[:, :, None]
        highlight = np.clip((luma - 0.45) * 2.0, 0, 1)[:, :, None]
        stacked = np.stack([b, g, r], axis=-1)
        teal = np.array([0.12, 0.02, -0.06], dtype=np.float32)
        orange = np.array([-0.06, -0.01, 0.10], dtype=np.float32)
        stacked = stacked + shadow * teal + highlight * orange
        stacked = _s_curve(np.clip(stacked, 0, 1), 0.18)
        stacked = _grain(stacked, 0.02)
    else:  # hk_night
        stacked = np.stack([b, g, r], axis=-1) * 0.88
        stacked[:, :, 0] = np.clip(stacked[:, :, 0] * 1.08 + 0.03, 0, 1)  # B
        stacked[:, :, 2] = np.clip(stacked[:, :, 2] * 1.10 + 0.02, 0, 1)  # R
        stacked = _s_curve(stacked, 0.22)
        stacked = _grain(stacked, 0.045)

    out = np.clip(stacked * 255.0, 0, 255).astype(np.uint8)
    return out


class PhotoLookRenderer:
    modality = "image.photo_look"

    def estimate_cost(self) -> float:
        return 0.0

    def run(self, source: Path, style_id: str) -> tuple[np.ndarray, dict]:
        bgr = load_bgr(source)
        out = apply_look(bgr, style_id)
        return out, dict(LOOKS[style_id]["params"])
