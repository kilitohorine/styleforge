from fastapi.testclient import TestClient
import cv2
import numpy as np

from app.api.main import app
from app.graphs.route import keyword_route
from app.providers import siliconflow
from app.settings import settings


def _jpg() -> bytes:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :] = (20, 80, 160)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_keyword_generate_vs_ask():
    intent, style, _ = keyword_route("水彩和水墨差在哪", False)
    assert intent == "qa"
    intent, style, _ = keyword_route("改成水墨画", False)
    assert intent == "image2d"
    assert style == "ink_wash"
    intent, style, _ = keyword_route("做成胶片暖调", True)
    assert intent == "look"
    assert style == "film_portra"


def test_look_still_free_when_daily_budget_zero(monkeypatch):
    monkeypatch.setattr(settings, "daily_budget_cny", 0.0)
    c = TestClient(app)
    up = c.post("/v1/assets", files={"file": ("t.jpg", _jpg(), "image/jpeg")})
    aid = up.json()["asset_id"]
    job = c.post(
        "/v1/jobs",
        json={
            "modality": "image.photo_look",
            "style_id": "film_portra",
            "input_assets": [{"asset_id": aid, "role": "source"}],
        },
    )
    assert job.status_code == 200
    assert job.json()["status"] == "succeeded"
    assert job.json()["actual_cost_cny"] == 0


def test_image_2d_budget_exceeded_does_not_call_provider(monkeypatch):
    siliconflow.generate_calls.clear()
    monkeypatch.setattr(settings, "daily_budget_cny", 0.0)
    monkeypatch.setattr(settings, "image_2d_backend", "siliconflow")
    monkeypatch.setattr(settings, "siliconflow_api_key", "sk-test-not-used")
    c = TestClient(app)
    job = c.post(
        "/v1/jobs",
        json={"modality": "image.2d", "style_id": "ink_wash", "prompt": "改成水墨画", "budget_cny_max": 1},
    )
    assert job.status_code == 200
    body = job.json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "BUDGET_EXCEEDED"
    assert siliconflow.generate_calls == []


def test_image_2d_mock_succeeds(monkeypatch):
    monkeypatch.setattr(settings, "image_2d_backend", "mock")
    c = TestClient(app)
    job = c.post(
        "/v1/jobs",
        json={"modality": "image.2d", "style_id": "watercolor", "prompt": "生成水彩"},
    )
    assert job.status_code == 200
    body = job.json()
    assert body["status"] == "succeeded"
    assert body["modality"] == "image.2d"
    assert body["trace"]["renderer"] == "image_2d.mock"
    aid = body["outputs"][0]["asset_id"]
    f = c.get(f"/v1/assets/{aid}/file")
    assert f.status_code == 200
    assert len(f.content) > 50


def test_chat_generate_ink_with_mock(monkeypatch):
    monkeypatch.setattr(settings, "image_2d_backend", "mock")
    c = TestClient(app)
    chat = c.post("/v1/chat", json={"message": "改成水墨画"})
    assert chat.status_code == 200
    body = chat.json()
    assert body["style_id"] == "ink_wash"
    assert body["job_id"]
    job = c.get(f"/v1/jobs/{body['job_id']}").json()
    assert job["modality"] == "image.2d"
    assert job["status"] == "succeeded"


def test_budget_endpoint():
    c = TestClient(app)
    r = c.get("/v1/budget")
    assert r.status_code == 200
    assert "daily_budget_cny" in r.json()
    assert "project_budget_cny" in r.json()
