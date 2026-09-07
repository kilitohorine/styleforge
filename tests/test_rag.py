from fastapi.testclient import TestClient

from app.api.main import app
from app.rag import load_packs


def test_ten_style_packs():
    packs = load_packs()
    ids = {p["style_id"] for p in packs}
    assert len(packs) >= 8
    assert {"film_portra", "hk_night", "cyberpunk", "watercolor", "ink_wash"} <= ids


def test_styles_endpoint_lists_packs():
    c = TestClient(app)
    r = c.get("/v1/styles")
    assert r.status_code == 200
    ids = {row["style_id"] for row in r.json()}
    assert len(ids) >= 8
    assert "watercolor" in ids
    assert "film_portra" in ids


def test_rag_query_neon_hits_night_styles():
    c = TestClient(app)
    ingested = c.post("/v1/styles/ingest")
    assert ingested.status_code == 200
    assert ingested.json()["packs"] >= 8
    r = c.post("/v1/rag/query", json={"query": "夜景霓虹", "k": 3})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert hits
    ids = {h["style_id"] for h in hits}
    assert "cyberpunk" in ids or "hk_night" in ids


def test_qa_chat_uses_rag_citations_no_job():
    c = TestClient(app)
    r = c.post("/v1/chat", json={"message": "水彩和水墨差在哪"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("job_id") in (None, "")
    cites = " ".join(body.get("citations") or [])
    assert "watercolor" in cites
    assert "ink_wash" in cites
    assert "水彩" in body["reply"] and "水墨" in body["reply"]
