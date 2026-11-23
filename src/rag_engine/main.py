"""ASGI entry point: ``uvicorn rag_engine.main:app``."""

from __future__ import annotations

from rag_engine.api.app import create_app

app = create_app()
