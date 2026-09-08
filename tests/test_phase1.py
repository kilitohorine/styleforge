from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api.main import app
from app.graphs.perceive import classify_scene
from app.renderers.photo_look import LOOKS
from app.settings import settings


def _jpg(color=(12, 10, 18)) -> bytes:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = color
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_eight_photo_looks():
    assert len(LOOKS) == 8
    for sid in LOOKS:
        assert (Path("app/luts") / LOOKS[sid]["lut"]).exists()


def test_keyword_new_looks():
    from app.graphs.route import keyword_route

    intent, style, _ = keyword_route("做成黑白胶片", True)
    assert intent == "look" and style == "mono_bw"
    intent, style, _ = keyword_route("黄金时刻", True)
    assert style == "golden_hour"


def test_perceive_night_prefers_hk(monkeypatch):
    info = classify_scene(np.full((80, 80, 3), 18, dtype=np.uint8))
    assert info["scene"] == "night"
    assert info["suggest_style_id"] == "hk_night"
    c = TestClient(app)
    up = c.post("/v1/assets", files={"file": ("n.jpg", _jpg(), "image/jpeg")})
    chat = c.post("/v1/chat", json={"message": "修好", "asset_ids": [up.json()["asset_id"]]})
    assert chat.status_code == 200
    job = c.get(f"/v1/jobs/{chat.json()['job_id']}").json()
    assert job["trace"]["scene"] == "night"
    assert "critique" in job["trace"]["actions"] or job["trace"].get("critique") is not None
    assert chat.json()["style_id"] == "hk_night"


def test_llm_three_dialogues(monkeypatch):
    def fake(message, has_image):
        table = {
            "做成胶片暖调": ("look", "film_portra", "llm-look"),
            "水彩和水墨差在哪": ("qa", None, "llm-qa"),
            "改成水墨画": ("image2d", "ink_wash", "llm-2d"),
        }
        return table[message]

    monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
    monkeypatch.setattr("app.graphs.route._llm_route", fake)
    monkeypatch.setattr(settings, "image_2d_backend", "mock")
    c = TestClient(app)
    look = c.post("/v1/chat", json={"message": "做成胶片暖调", "asset_ids": []})
    assert look.json()["style_id"] in (None, "film_portra")
    qa = c.post("/v1/chat", json={"message": "水彩和水墨差在哪"})
    assert not qa.json().get("job_id")
    gen = c.post("/v1/chat", json={"message": "改成水墨画"})
    assert gen.json()["style_id"] == "ink_wash"
    job = c.get(f"/v1/jobs/{gen.json()['job_id']}").json()
    assert job["modality"] == "image.2d"


def test_docker_files_exist():
    assert Path("Dockerfile").exists()
    assert Path("compose.yaml").exists()
    assert Path(".dockerignore").exists()
