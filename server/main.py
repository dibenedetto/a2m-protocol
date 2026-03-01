"""
A2M — AgentToMemory Protocol
FastAPI application entry point.

Usage:
    python -m server.main [--port 8765] [--db a2m.db]

The server starts on http://0.0.0.0:8765 by default.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import init_store, router
from .store import A2MStore

# ── Store singleton ───────────────────────────────────────────────────────────
# Initialised in lifespan; accessible to all route handlers via api.init_store().
_store: A2MStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the TTL expiry background task on startup; cancel it on shutdown."""
    assert _store is not None

    async def _ttl_loop():
        while True:
            await asyncio.sleep(60)
            n = _store.expire()
            if n:
                print(f"[A2M] TTL expiry: removed {n} entries")

    task = asyncio.create_task(_ttl_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def create_app(db_path: str = "a2m.db") -> FastAPI:
    global _store
    _store = A2MStore(db_path=db_path)
    init_store(_store)

    app = FastAPI(
        title="A2M — AgentToMemory",
        version="0.1.0",
        description=(
            "A shared memory protocol for AI agents across frameworks. "
            "See https://github.com/numel-ai/a2m for the full specification."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/ping")
    async def ping():
        return {"status": "ok", "version": "0.1.0"}

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A2M Memory Server")
    parser.add_argument("--port", type=int, default=8765, help="Port (default 8765)")
    parser.add_argument("--db",   type=str, default="a2m.db", help="SQLite DB path")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--reload", action="store_true", help="Hot reload (dev only)")
    args = parser.parse_args()

    # Pre-create the app / store before uvicorn forks
    create_app(db_path=args.db)

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


# Module-level `app` used by uvicorn when running as a module
app = create_app()

if __name__ == "__main__":
    main()
