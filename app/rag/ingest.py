"""Ingest entrypoint: python -m app.rag.ingest"""

from __future__ import annotations

from app.rag.retriever import ingest


def main() -> None:
    info = ingest()
    print(info)


if __name__ == "__main__":
    main()
