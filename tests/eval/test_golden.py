import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.jobs.store import IllegalJobTransition, save_job
from app.schemas.job import JobOut, JobTrace
from app.settings import settings
from scripts.eval_report import run_report


@pytest.mark.golden
def test_golden_mock_report():
    payload = run_report(out=None)
    assert payload["gates"]["G1"]
    assert payload["gates"]["G2"]
    assert payload["gates"]["G3"]
    assert payload["gates"]["G4"]
    assert payload["E1"]["intent_acc"] >= 0.9
    assert payload["E1"]["style_top1"] >= 0.85
    assert payload["E1"]["modality_acc"] >= 0.9
    assert payload["E2"]["size_ok"] == 1.0
    assert payload["E2"]["ssim_mean"] >= 0.75
    assert payload["E4"]["recall_at_3"] >= 0.8
    assert payload["E5"]["style_kept"] == 5


def test_openapi_core_paths():
    spec = TestClient(app).get("/openapi.json").json()
    paths = spec["paths"]
    for prefix in ("/v1/jobs", "/v1/chat", "/v1/styles", "/v1/assets", "/v1/capabilities"):
        assert any(prefix in p for p in paths)


def test_capabilities_reserved():
    body = TestClient(app).get("/v1/capabilities").json()
    assert body["modalities"]["asset.3d"]["status"] == "reserved"
    assert body["modalities"]["video.long"]["status"] == "reserved"
    assert "siliconflow" in body["modalities"]["image.2d"]["providers"]
    assert "dashscope_wanxiang" in body["modalities"]["image.2d"]["providers"]
    assert len(body["modalities"]["image.photo_look"]["styles"]) == 8


def test_upload_rejects_txt():
    r = TestClient(app).post("/v1/assets", files={"file": ("x.txt", b"nope", "text/plain")})
    assert r.status_code == 415


def test_illegal_job_transition():
    job = JobOut(
        job_id="j_illegal_test",
        status="succeeded",
        modality="image.photo_look",
        trace=JobTrace(style_id="film_portra"),
    )
    save_job(job)
    job.status = "queued"
    with pytest.raises(IllegalJobTransition):
        save_job(job)


def test_dashscope_backend_fail_closed(monkeypatch):
    from app.providers import dashscope, siliconflow

    siliconflow.generate_calls.clear()
    dashscope.generate_calls.clear()
    monkeypatch.setattr(settings, "image_2d_backend", "dashscope_wanxiang")
    monkeypatch.setattr(settings, "dashscope_api_key", "")
    monkeypatch.setattr(settings, "siliconflow_api_key", "sk-should-not-use")
    job = TestClient(app).post(
        "/v1/jobs",
        json={"modality": "image.2d", "style_id": "ink_wash", "prompt": "改成水墨画"},
    )
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert job.json()["error"]["code"] == "PROVIDER_NOT_CONFIGURED"
    assert siliconflow.generate_calls == []
    assert dashscope.generate_calls == []


def test_pastoral_prompt_has_no_ghibli():
    from app.renderers.image_2d import build_prompt

    prompt, negative = build_prompt("pastoral_anime", "改成田园动画风")
    blob = (prompt + negative).lower()
    assert "ghibli" not in blob
    assert "吉卜力" not in blob
    assert "官方吉卜力授权" not in blob
