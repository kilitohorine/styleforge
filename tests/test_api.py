from fastapi.testclient import TestClient
import cv2
import numpy as np

from app.api.main import app


def test_health():
    c = TestClient(app)
    r = c.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_reserved_3d():
    c = TestClient(app)
    r = c.post("/v1/jobs", json={"modality": "asset.3d", "style_id": "ink_wash", "prompt": "x"})
    assert r.status_code == 501


def test_upload_look_chat():
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[:, :] = (30, 70, 140)
    raw = cv2.imencode(".jpg", img)[1].tobytes()
    c = TestClient(app)
    up = c.post("/v1/assets", files={"file": ("t.jpg", raw, "image/jpeg")})
    assert up.status_code == 200
    aid = up.json()["asset_id"]
    chat = c.post("/v1/chat", json={"message": "做成胶片暖调", "asset_ids": [aid]})
    assert chat.status_code == 200
    body = chat.json()
    assert body["job_id"]
    assert body["style_id"] == "film_portra"
    job = c.get(f"/v1/jobs/{body['job_id']}")
    assert job.json()["actual_cost_cny"] == 0
    outs = job.json()["outputs"]
    result = next(o for o in outs if o["role"] in ("result", "source"))
    f = c.get(f"/v1/assets/{result['asset_id']}/file")
    assert f.status_code == 200
    assert len(f.content) > 100
