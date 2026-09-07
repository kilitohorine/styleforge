"""Traditional photo looks via 3D LUT (.cube) + param patches. Cost = 0."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from app.renderers.lut import apply_cube_rgb, parse_cube

MAX_SIDE = 2048
LUT_DIR = Path(__file__).resolve().parent.parent / "luts"

LOOKS = {
    "film_portra": {
        "name": "胶片暖调",
        "keywords": ("胶片", "暖调", "portra", "日系", "柯达", "fuji", "富士"),
        "lut": "film_portra.cube",
        "params": {"exposure": 0.0, "warm": 0.0, "grain": 0.03, "lut_strength": 1.0},
    },
    "cinematic_teal_orange": {
        "name": "电影青橙",
        "keywords": ("电影", "青橙", "cinematic", "teal", "阿莱", "质感"),
        "lut": "cinematic_teal_orange.cube",
        "params": {"exposure": 0.0, "warm": 0.0, "grain": 0.02, "lut_strength": 1.0},
    },
    "hk_night": {
        "name": "港风夜景",
        "keywords": ("港风", "夜景", "霓虹", "hk", "赛博夜", "暗青"),
        "lut": "hk_night.cube",
        "params": {"exposure": 0.0, "warm": 0.0, "grain": 0.05, "lut_strength": 1.0},
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
    return np.clip(x + amount * (x - 0.5) * (1.0 - x) * 4.0, 0, 1)


def color_grade_rgb(rgb: np.ndarray, style_id: str) -> np.ndarray:
    """Look colour (no grain). Used to bake .cube files."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    if style_id == "film_portra":
        r = np.clip(r * 1.10 + 0.02, 0, 1)
        g = np.clip(g * 1.04, 0, 1)
        b = np.clip(b * 0.92, 0, 1)
        stacked = np.stack([r, g, b], axis=-1)
        stacked = np.clip(stacked * 1.06, 0, 1)
        return _s_curve(stacked, 0.12)
    if style_id == "cinematic_teal_orange":
        shadow = np.clip(1.0 - luma * 1.4, 0, 1)[..., None]
        highlight = np.clip((luma - 0.45) * 2.0, 0, 1)[..., None]
        stacked = np.stack([r, g, b], axis=-1)
        teal = np.array([-0.06, 0.02, 0.12], dtype=np.float32)
        orange = np.array([0.10, -0.01, -0.06], dtype=np.float32)
        stacked = stacked + shadow * teal + highlight * orange
        return _s_curve(np.clip(stacked, 0, 1), 0.18)
    stacked = np.stack([r, g, b], axis=-1) * 0.88
    stacked[..., 2] = np.clip(stacked[..., 2] * 1.08 + 0.03, 0, 1)
    stacked[..., 0] = np.clip(stacked[..., 0] * 1.10 + 0.02, 0, 1)
    return _s_curve(stacked, 0.22)


def _grain(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    noise = np.random.default_rng(42).normal(0, amount, img.shape[:2]).astype(np.float32)
    return np.clip(img + noise[:, :, None], 0, 1)


@lru_cache(maxsize=8)
def _load_table(style_id: str) -> np.ndarray | None:
    name = LOOKS[style_id]["lut"]
    path = LUT_DIR / name
    if not path.exists():
        return None
    _, table = parse_cube(path)
    return table


def merge_params(style_id: str, overrides: dict | None = None) -> dict:
    params = dict(LOOKS[style_id]["params"])
    if overrides:
        for key, value in overrides.items():
            if key in params and isinstance(value, (int, float)):
                params[key] = float(value)
    return params


def apply_look(bgr: np.ndarray, style_id: str, overrides: dict | None = None) -> np.ndarray:
    if style_id not in LOOKS:
        raise ValueError(f"unknown look: {style_id}")
    params = merge_params(style_id, overrides)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    table = _load_table(style_id)
    graded = apply_cube_rgb(rgb, table) if table is not None else color_grade_rgb(rgb, style_id)
    strength = float(np.clip(params.get("lut_strength", 1.0), 0.0, 1.0))
    rgb = rgb * (1.0 - strength) + graded * strength
    rgb = np.clip(rgb * (2.0 ** params.get("exposure", 0.0)), 0, 1)
    warm = params.get("warm", 0.0)
    rgb[..., 0] = np.clip(rgb[..., 0] + warm * 0.5, 0, 1)
    rgb[..., 2] = np.clip(rgb[..., 2] - warm * 0.35, 0, 1)
    rgb = _grain(rgb, params.get("grain", 0.0))
    out = (np.clip(rgb, 0, 1) * 255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def make_comparison(src_bgr: np.ndarray, dst_bgr: np.ndarray) -> np.ndarray:
    h = min(src_bgr.shape[0], dst_bgr.shape[0])
    w = min(src_bgr.shape[1], dst_bgr.shape[1])
    left = cv2.resize(src_bgr, (w, h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(dst_bgr, (w, h), interpolation=cv2.INTER_AREA)
    gap = np.full((h, 8, 3), 240, dtype=np.uint8)
    return np.hstack([left, gap, right])


class PhotoLookRenderer:
    modality = "image.photo_look"

    def estimate_cost(self) -> float:
        return 0.0

    def run(
        self, source: Path, style_id: str, overrides: dict | None = None
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        bgr = load_bgr(source)
        out = apply_look(bgr, style_id, overrides)
        params = merge_params(style_id, overrides)
        params["lut"] = LOOKS[style_id]["lut"]
        compare = make_comparison(bgr, out)
        return out, compare, params
