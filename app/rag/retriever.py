"""Chroma ingest + query over Style Packs. Embedding: n-gram hash (0 download)."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from app.rag import extra_documents, load_packs, pack_document
from app.rag.embedder import embed_texts
from app.settings import settings

COLLECTION = "styles"


class NgramHashEmbedding(EmbeddingFunction[Documents]):
    def name(self) -> str:
        return "ngram_hash_zh"

    def __call__(self, input: Documents) -> Embeddings:
        return embed_texts(list(input)).tolist()


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return _CLIENT


def reset_client() -> None:
    global _CLIENT
    _CLIENT = None


def _collection(create: bool = True):
    client = _client()
    kwargs = {
        "name": COLLECTION,
        "embedding_function": NgramHashEmbedding(),
        "metadata": {"hnsw:space": "cosine"},
    }
    if create:
        return client.get_or_create_collection(**kwargs)
    return client.get_collection(name=COLLECTION, embedding_function=NgramHashEmbedding())


def ingest() -> dict[str, Any]:
    client = _client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = _collection(create=True)
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict[str, str]] = []
    packs = load_packs()
    for pack in packs:
        sid = str(pack["style_id"])
        ids.append(f"pack:{sid}")
        docs.append(pack_document(pack))
        metas.append(
            {
                "style_id": sid,
                "name": str(pack.get("name") or sid),
                "kind": "pack",
                "domain": ",".join(pack.get("domain") or []),
            }
        )
    for extra in extra_documents():
        ids.append(f"doc:{extra['doc_id']}")
        docs.append(extra["text"])
        metas.append(
            {
                "style_id": extra["doc_id"],
                "name": extra["name"],
                "kind": extra["kind"],
                "domain": "reserved",
            }
        )
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    return {"count": len(ids), "packs": len(packs), "collection": COLLECTION}


def _ensure_ingested() -> None:
    col = _collection(create=True)
    if col.count() == 0:
        ingest()


def query(text: str, k: int = 3) -> list[dict[str, Any]]:
    _ensure_ingested()
    col = _collection(create=True)
    k = max(1, min(int(k or 3), 8))
    raw = col.query(query_texts=[text], n_results=k, include=["documents", "metadatas", "distances"])
    hits: list[dict[str, Any]] = []
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    ids = (raw.get("ids") or [[]])[0]
    for i, meta in enumerate(metas):
        snippet = (docs[i] if i < len(docs) else "") or ""
        snippet = " ".join(snippet.split())[:240]
        hits.append(
            {
                "id": ids[i] if i < len(ids) else "",
                "style_id": (meta or {}).get("style_id"),
                "name": (meta or {}).get("name"),
                "kind": (meta or {}).get("kind"),
                "distance": dists[i] if i < len(dists) else None,
                "snippet": snippet,
            }
        )
    return hits


def compose_qa(message: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "知识库为空。请先调用 POST /v1/styles/ingest。"
    lines = ["根据 Style Pack 检索（只引用配方，不调用生图）："]
    for hit in hits:
        name = hit.get("name") or hit.get("style_id")
        sid = hit.get("style_id")
        snippet = (hit.get("snippet") or "").replace("\n", " ")[:90]
        lines.append(f"- {name}（{sid}）：{snippet}")
    text = message or ""
    if "水彩" in text and "水墨" in text:
        lines.append(
            "对比：水彩是透明彩色叠层与纸纹湿边；水墨是焦墨、飞白与留白。二者都属于 image.2d 预留，当前不能出图。"
        )
    else:
        lines.append("当前可渲染：胶片暖调 / 电影青橙 / 港风夜景（.cube LUT）。艺术风格仅可问答检索。")
    return "\n".join(lines)
