from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api.main import app
from app.graphs.route import apply_param_delta, parse_patch
from app.renderers.lut import parse_cube
from app.renderers.photo_look import LOOKS, apply_look, merge_params


def _jpg() -> bytes:
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[:, :32] = (30, 70, 140)
    img[:, 32:] = (160, 90, 40)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_cube_files_exist():
    for meta in LOOKS.values():
        path = Path("app/luts") / meta["lut"]
        assert path.exists(), path
        size, table = parse_cube(path)
        assert size == 17
        assert table.shape == (17, 17, 17, 3)


def test_looks_change_pixels():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 160)
    for sid in LOOKS:
        out = apply_look(img, sid)
        assert out.shape == img.shape
        assert not np.array_equal(out, img)


def test_patch_keeps_style_and_comparison():
    c = TestClient(app)
    up = c.post("/v1/assets", files={"file": ("t.jpg", _jpg(), "image/jpeg")})
    aid = up.json()["asset_id"]
    first = c.post("/v1/chat", json={"message": "做成胶片暖调", "asset_ids": [aid]})
    assert first.status_code == 200
    tid = first.json()["thread_id"]
    assert first.json()["style_id"] == "film_portra"
    job1 = c.get(f"/v1/jobs/{first.json()['job_id']}").json()
    assert job1["trace"]["comparison_asset_id"]
    assert job1["trace"]["lut"] == "film_portra.cube"
    roles = {o["role"] for o in job1["outputs"]}
    assert "result" in roles and "compare" in roles
    exp0 = first.json()["params"]["exposure"]

    second = c.post("/v1/chat", json={"thread_id": tid, "message": "再暗一点"})
    assert second.status_code == 200
    assert second.json()["style_id"] == "film_portra"
    assert second.json()["thread_id"] == tid
    assert second.json()["params"]["exposure"] < exp0

    thread = c.get(f"/v1/chat/{tid}")
    assert thread.status_code == 200
    assert thread.json()["style_id"] == "film_portra"


def test_parse_patch():
    assert parse_patch("再暗一点")["exposure"] < 0
    merged = apply_param_delta(merge_params("hk_night"), parse_patch("再暗一点"))
    assert merged["exposure"] < merge_params("hk_night")["exposure"]
