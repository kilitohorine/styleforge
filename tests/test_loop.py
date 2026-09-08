from pathlib import Path

import cv2
import numpy as np

from app.graphs.route import keyword_route
from app.renderers.photo_look import apply_look


def test_looks_change_pixels():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 160)
    for sid in ("film_portra", "cinematic_teal_orange", "hk_night", "mono_bw", "golden_hour"):
        out = apply_look(img, sid)
        assert out.shape == img.shape
        assert not np.array_equal(out, img)


def test_keyword_route():
    intent, style, _ = keyword_route("做成胶片暖调", True)
    assert intent == "look"
    assert style == "film_portra"
    intent, style, reply = keyword_route("水彩和水墨差在哪", False)
    assert intent == "qa"
    assert style is None
    intent, _, _ = keyword_route("生成3D模型", False)
    assert intent == "unsupported"
