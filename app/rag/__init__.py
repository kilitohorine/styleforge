"""Style Pack YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PACKS_DIR = Path(__file__).resolve().parent / "packs"
DOCS_DIR = Path(__file__).resolve().parent / "docs"


def load_packs() -> list[dict[str, Any]]:
    packs: list[dict[str, Any]] = []
    for path in sorted(PACKS_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["_path"] = str(path.name)
        packs.append(data)
    return packs


def pack_document(pack: dict[str, Any]) -> str:
    domain = pack.get("domain") or []
    if isinstance(domain, list):
        domain_s = ", ".join(domain)
    else:
        domain_s = str(domain)
    chunks = [
        f"style_id: {pack.get('style_id', '')}",
        f"name: {pack.get('name', '')}",
        f"domain: {domain_s}",
        pack.get("summary") or "",
        pack.get("positive") or "",
        pack.get("negative") or "",
        pack.get("use_when") or "",
        pack.get("avoid") or "",
    ]
    return "\n".join(str(c).strip() for c in chunks if str(c).strip())


def extra_documents() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    if not DOCS_DIR.exists():
        return docs
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs.append(
            {
                "doc_id": path.stem,
                "name": path.stem,
                "text": path.read_text(encoding="utf-8"),
                "kind": path.stem,
            }
        )
    return docs
