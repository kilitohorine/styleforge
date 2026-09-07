from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.graphs.agent import run_agent
from app.jobs.store import (
    asset_path,
    dump_output,
    get_job,
    init_db,
    new_id,
    register_asset,
    save_job,
)
from app.jobs.threads import load_thread
from app.rag import load_packs
from app.rag.retriever import ingest as ingest_styles
from app.rag.retriever import query as rag_query
from app.renderers.photo_look import LOOKS, PhotoLookRenderer, encode_jpeg
from app.schemas.job import (
    ChatIn,
    ChatOut,
    ErrorBody,
    InputAsset,
    JobCreate,
    JobOut,
    JobTrace,
    RagHit,
    RagQueryIn,
    RagQueryOut,
)
from app.settings import settings

app = FastAPI(title="StyleForge", version="0.1.0")
init_db()

RESERVED = {"asset.3d", "video.clip", "video.long"}


@app.get("/v1/health")
def health():
    return {"status": "ok", "loop": "langgraph:route->execute"}


@app.get("/v1/capabilities")
def capabilities():
    return {
        "modalities": {
            "image.photo_look": {
                "status": "ready",
                "providers": ["opencv_cube_lut"],
                "lut_format": "adobe_iridas_cube",
                "styles": list(LOOKS),
            },
            "image.2d": {"status": "not_ready"},
            "asset.3d": {"status": "reserved"},
            "video.clip": {"status": "reserved"},
            "video.long": {"status": "reserved"},
        },
        "rag": {
            "status": "ready",
            "store": "chroma",
            "embedding": "ngram_hash_zh",
            "note": "BGE-small-zh optional later; default hash embedder needs no download",
        },
    }


@app.get("/v1/styles")
def list_styles():
    packs = load_packs()
    if packs:
        return [
            {
                "style_id": p["style_id"],
                "name": p.get("name"),
                "domain": p.get("domain") or [],
                "ready": p["style_id"] in LOOKS,
            }
            for p in packs
        ]
    return [
        {"style_id": k, "name": v["name"], "domain": ["image.photo_look"], "ready": True}
        for k, v in LOOKS.items()
    ]


@app.post("/v1/styles/ingest")
def styles_ingest():
    return ingest_styles()


@app.post("/v1/rag/query", response_model=RagQueryOut)
def rag_query_http(body: RagQueryIn):
    hits = rag_query(body.query, k=body.k)
    return RagQueryOut(hits=[RagHit(**h) for h in hits])


@app.post("/v1/assets")
async def upload_asset(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=415, detail="only jpg/png/webp")
    asset_id = new_id("a")
    dest = settings.assets_dir / f"{asset_id}{suffix}"
    dest.write_bytes(await file.read())
    register_asset(dest, kind="upload", asset_id=asset_id)
    return {"asset_id": asset_id}


@app.get("/v1/assets/{asset_id}")
def asset_meta(asset_id: str):
    try:
        path = asset_path(asset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="asset not found")
    return {"asset_id": asset_id, "path": path.name}


@app.get("/v1/assets/{asset_id}/file")
def asset_file(asset_id: str):
    try:
        path = asset_path(asset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(path)


@app.post("/v1/jobs", response_model=JobOut)
def create_job(body: JobCreate):
    if body.modality in RESERVED:
        raise HTTPException(
            status_code=501,
            detail={"error": ErrorBody(code="MODALITY_RESERVED", message=f"{body.modality} reserved").model_dump()},
        )
    if body.modality != "image.photo_look":
        raise HTTPException(status_code=501, detail="only image.photo_look in day-1 loop")
    if not body.input_assets:
        raise HTTPException(status_code=400, detail="input_assets required")
    source_id = body.input_assets[0].asset_id
    try:
        source = asset_path(source_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="source asset not found")
    if body.style_id not in LOOKS:
        raise HTTPException(status_code=400, detail="unknown style_id")
    renderer = PhotoLookRenderer()
    bgr, compare, params = renderer.run(source, body.style_id)
    out_id = dump_output(encode_jpeg(bgr), body.style_id, kind=f"look:{body.style_id}")
    cmp_id = dump_output(encode_jpeg(compare), body.style_id, kind="compare")
    job = JobOut(
        job_id=new_id("j"),
        status="succeeded",
        modality="image.photo_look",
        actual_cost_cny=0.0,
        outputs=[
            InputAsset(asset_id=out_id, role="result"),
            InputAsset(asset_id=cmp_id, role="compare"),
        ],
        trace=JobTrace(
            style_id=body.style_id,
            renderer="photo_look.cube_lut",
            params=params,
            source_asset_id=source_id,
            comparison_asset_id=cmp_id,
            lut=LOOKS[body.style_id]["lut"],
        ),
    )
    save_job(job)
    return job


@app.get("/v1/jobs/{job_id}", response_model=JobOut)
def read_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/v1/chat/{thread_id}")
def read_thread(thread_id: str):
    row = load_thread(thread_id)
    if not row:
        raise HTTPException(status_code=404, detail="thread not found")
    return row


@app.post("/v1/chat", response_model=ChatOut)
def chat(body: ChatIn):
    asset_id = body.asset_ids[0] if body.asset_ids else None
    thread_id = body.thread_id or new_id("t")
    state = run_agent(body.message, asset_id, thread_id)
    return ChatOut(
        thread_id=state.get("thread_id") or thread_id,
        reply=state.get("reply") or "",
        job_id=state.get("job_id"),
        style_id=state.get("style_id"),
        citations=state.get("citations") or [],
        params=state.get("params") or {},
    )
