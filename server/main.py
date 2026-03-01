"""
A2M — AgentToMemory Protocol
FastAPI application entry point.

Usage:
    python -m server.main [--port 8765] [--db a2m.db]
    python -m server.main --numpy-path ./numpy_index.npz
    python -m server.main --relational-backend postgresql --postgresql-dsn postgresql://localhost/a2m
    python -m server.main --vector-backend lancedb --lancedb-uri ./lancedb_data
    python -m server.main --vector-backend chroma --chroma-path ./chroma_data
    python -m server.main --vector-backend pgvector --pgvector-dsn postgresql://localhost/a2m

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
from .backends.base import AbstractRelationalBackend, AbstractVectorBackend
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


def create_app(
    relational: AbstractRelationalBackend | None = None,
    vector:     AbstractVectorBackend     | None = None,
    db_path:    str                              = "a2m.db",
) -> FastAPI:
    """
    Create the A2M FastAPI application.

    Args:
        relational: Relational backend instance.  Defaults to SQLiteRelationalBackend(db_path).
        vector:     Vector backend instance.      Defaults to NumpyVectorBackend().
        db_path:    SQLite path used when relational is None (default "a2m.db").
                    Pass ":memory:" for an ephemeral in-process store.

    Examples::

        # Default (SQLite + numpy, zero extra deps):
        app = create_app()
        app = create_app(db_path=":memory:")

        # Numpy with persistence (fast restarts, no extra deps):
        from server.backends.numpy_backend import NumpyVectorBackend
        app = create_app(vector=NumpyVectorBackend(path="./numpy_index.npz"))

        # LanceDB vector backend:
        from server.backends.lancedb_backend import LanceVectorBackend
        app = create_app(vector=LanceVectorBackend(uri="./lancedb_data"))

        # ChromaDB vector backend (persistent):
        from server.backends.chroma_backend import ChromaVectorBackend
        app = create_app(vector=ChromaVectorBackend(path="./chroma_data"))

        # PostgreSQL relational + pgvector:
        from server.backends.postgres import PostgreSQLRelationalBackend
        from server.backends.pgvector_backend import PgVectorBackend
        dsn = "postgresql://user:pass@localhost/a2m"
        app = create_app(
            relational=PostgreSQLRelationalBackend(dsn),
            vector=PgVectorBackend(dsn),
        )
    """
    global _store

    if relational is None:
        from .backends.sqlite_relational import SQLiteRelationalBackend
        relational = SQLiteRelationalBackend(db_path)

    if vector is None:
        from .backends.numpy_vector import NumpyVectorBackend
        vector = NumpyVectorBackend()

    _store = A2MStore(relational=relational, vector=vector)
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
    parser.add_argument("--port",   type=int, default=8765,     help="Port (default 8765)")
    parser.add_argument("--host",   type=str, default="0.0.0.0")
    parser.add_argument("--reload", action="store_true",         help="Hot reload (dev only)")

    # ── Relational backend ────────────────────────────────────────────────────
    parser.add_argument(
        "--relational-backend", type=str, default="sqlite",
        choices=["sqlite", "postgresql"],
        help="Relational backend (default: sqlite)",
    )
    parser.add_argument(
        "--db", type=str, default="a2m.db",
        help="SQLite DB path (used when --relational-backend=sqlite, default a2m.db)",
    )
    parser.add_argument(
        "--postgresql-dsn", type=str, default="postgresql://localhost/a2m",
        help="PostgreSQL DSN (used when --relational-backend=postgresql)",
    )

    # ── Vector backend ────────────────────────────────────────────────────────
    parser.add_argument(
        "--vector-backend", type=str, default="numpy",
        choices=["numpy", "lancedb", "chroma", "pgvector"],
        help=(
            "Vector backend: 'numpy' (default, in-memory or persistent via --numpy-path), "
            "'lancedb' (persistent ANN), "
            "'chroma' (ChromaDB), "
            "'pgvector' (PostgreSQL + pgvector)"
        ),
    )
    parser.add_argument(
        "--numpy-path", type=str, default=None,
        help=(
            "Numpy index persistence file (used when --vector-backend=numpy). "
            "Omit for ephemeral in-memory mode. "
            "Example: --numpy-path ./numpy_index.npz"
        ),
    )
    parser.add_argument(
        "--lancedb-uri", type=str, default="./lancedb_data",
        help="LanceDB data directory (used when --vector-backend=lancedb)",
    )
    parser.add_argument(
        "--chroma-path", type=str, default=None,
        help="ChromaDB persistence directory (omit for ephemeral in-memory mode)",
    )
    parser.add_argument(
        "--chroma-collection", type=str, default="a2m_vectors",
        help="ChromaDB collection name (default: a2m_vectors)",
    )
    parser.add_argument(
        "--pgvector-dsn", type=str, default="postgresql://localhost/a2m",
        help="PostgreSQL DSN for pgvector (used when --vector-backend=pgvector)",
    )
    parser.add_argument(
        "--pgvector-table", type=str, default="a2m_vectors",
        help="pgvector table name (default: a2m_vectors)",
    )

    args = parser.parse_args()

    # ── Build relational backend ──────────────────────────────────────────────
    relational = None
    if args.relational_backend == "postgresql":
        from .backends.postgres_relational import PostgreSQLRelationalBackend
        relational = PostgreSQLRelationalBackend(dsn=args.postgresql_dsn)
        print(f"[A2M] Relational backend: PostgreSQL ({args.postgresql_dsn})")
    elif args.relational_backend == "sqlite":
        print(f"[A2M] Relational backend: SQLite ({args.db})")
    else:
        raise ValueError(f"Unsupported relational backend: {args.relational_backend}")

    # ── Build vector backend ──────────────────────────────────────────────────
    vector = None
    if args.vector_backend == "lancedb":
        from .backends.lancedb_vector import LanceVectorBackend
        vector = LanceVectorBackend(uri=args.lancedb_uri)
        print(f"[A2M] Vector backend: LanceDB at {args.lancedb_uri}")
    elif args.vector_backend == "chroma":
        from .backends.chroma_vector import ChromaVectorBackend
        vector = ChromaVectorBackend(
            path=args.chroma_path,
            collection_name=args.chroma_collection,
        )
        mode = f"persistent at {args.chroma_path}" if args.chroma_path else "ephemeral"
        print(f"[A2M] Vector backend: ChromaDB ({mode}, collection={args.chroma_collection})")
    elif args.vector_backend == "pgvector":
        from .backends.pgvector_vector import PgVectorBackend
        vector = PgVectorBackend(dsn=args.pgvector_dsn, table_name=args.pgvector_table)
        print(
            f"[A2M] Vector backend: pgvector "
            f"({args.pgvector_dsn}, table={args.pgvector_table})"
        )
    elif args.vector_backend == "numpy":
        from .backends.numpy_vector import NumpyVectorBackend
        vector = NumpyVectorBackend(path=args.numpy_path)
        if args.numpy_path:
            print(f"[A2M] Vector backend: numpy (persistent at {args.numpy_path})")
        else:
            print("[A2M] Vector backend: numpy (ephemeral in-memory)")
    else:
        raise ValueError(f"Unsupported vector backend: {args.vector_backend}")

    create_app(db_path=args.db, relational=relational, vector=vector)

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
