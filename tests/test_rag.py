from fastapi.testclient import TestClient

from app.api.main import app
from app.rag import load_packs


def test_ten_style_packs():
    packs = load_packs()
    ids = {p["style_id"] for p in packs}
    assert len(packs) >= 15
    assert {"film_portra", "hk_night", "golden_hour", "cyberpunk", "watercolor", "ink_wash"} <= ids


def test_styles_endpoint_lists_packs():
    c = TestClient(app)
    r = c.get("/v1/styles")
    assert r.status_code == 200
    ids = {row["style_id"] for row in r.json()}
    assert len(ids) >= 8
    assert "watercolor" in ids
    assert "film_portra" in ids
    assert "matte_film" in ids


def test_rag_placeholder_docs_hit():
    c = TestClient(app)
    c.post("/v1/styles/ingest")
    shot = c.post("/v1/rag/query", json={"query": "分镜语法慢推", "k": 3})
    mat = c.post("/v1/rag/query", json={"query": "PBR 金属度材质", "k": 3})
    shot_ids = {h["style_id"] for h in shot.json()["hits"]}
    mat_ids = {h["style_id"] for h in mat.json()["hits"]}
    assert "shot_grammar" in shot_ids
    assert "material_3d" in mat_ids


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
