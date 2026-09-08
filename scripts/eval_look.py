"""Look direction heuristics for eval (Lab). Thresholds frozen for CI fixtures."""

from __future__ import annotations

import cv2
import numpy as np

from app.graphs.critique import luma_ssim


def lab_ab(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[:, :, 0], lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0


def size_ok(src: np.ndarray, dst: np.ndarray) -> bool:
    return abs(src.shape[0] - dst.shape[0]) <= 2 and abs(src.shape[1] - dst.shape[1]) <= 2


def chroma(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(a * a + b * b).mean())


def look_direction_hit(style_id: str, src: np.ndarray, dst: np.ndarray) -> bool:
    sl, sa, sb = lab_ab(src)
    dl, da, db = lab_ab(dst)
    if style_id == "film_portra":
        return db.mean() > sb.mean() + 0.2 and chroma(da, db) < chroma(sa, sb) * 2.8 + 20
    if style_id == "cinematic_teal_orange":
        shadow = sl < np.percentile(sl, 45)
        highlight = sl > np.percentile(sl, 55)
        if int(shadow.sum()) < 20 or int(highlight.sum()) < 20:
            return db.mean() != sb.mean() or da.mean() != sa.mean()
        teal = (da[shadow].mean() <= sa[shadow].mean() + 2.0) or (db[shadow].mean() <= sb[shadow].mean() + 2.0)
        orange = (da[highlight].mean() >= sa[highlight].mean() - 2.0) or (
            db[highlight].mean() >= sb[highlight].mean() - 2.0
        )
        return bool(teal and orange)
    if style_id == "hk_night":
        return dl.mean() <= sl.mean() + 1.0
    if style_id == "mono_bw":
        return chroma(da, db) < chroma(sa, sb) * 0.65 + 8.0
    if style_id == "vintage_fade":
        return float(dl.std()) <= float(sl.std()) + 4.0
    if style_id == "golden_hour":
        return db.mean() > sb.mean() + 0.4
    if style_id == "cool_steel":
        return db.mean() < sb.mean() + 0.5
    if style_id == "matte_film":
        return float(dl.std()) <= float(sl.std()) + 1.0
    return True


def score_pair(style_id: str, src: np.ndarray, dst: np.ndarray) -> dict:
    ssim = luma_ssim(src, dst)
    return {
        "style_id": style_id,
        "size_ok": size_ok(src, dst),
        "ssim": round(float(ssim), 4),
        "direction_hit": look_direction_hit(style_id, src, dst),
    }
