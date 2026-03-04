"""
Agent2Memory (A2M) Protocol
FastAPI router — all REST endpoints + WebSocket subscription.

Namespace routing uses {namespace:path} so that namespaces containing "/" are
handled correctly. Starlette's regex engine backtracks to anchor the literal
"/entries" suffix, making slash-separated namespaces fully transparent.

Route registration order matters for correct disambiguation:
  - /{namespace:path}/entries/{key:path}  (more specific) registered first
  - /{namespace:path}/entries             (less specific) registered after
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .models import (
    Entry,
    EntryWrite,
    ErrorBody,
    ListResponse,
    MemoryType,
    QueryRequest,
    QueryResult,
)
from .store import A2MStore, _validate_namespace

router = APIRouter(prefix="/a2m/v1")

# Store instance — injected by main.py
_store: Optional[A2MStore] = None


def init_store(store: A2MStore) -> None:
    global _store
    _store = store


def _s() -> A2MStore:
    assert _store is not None, "Store not initialised"
    return _store


def _ns_error(e: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=ErrorBody(code="INVALID_NAMESPACE", message=str(e)).model_dump(),
    )


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    """
    Probe the relational and vector backends.
    Returns backend status and type names.
    """
    return _s().health()


# ── Write (upsert) ────────────────────────────────────────────────────────────

@router.post("/{namespace:path}/entries")
async def write_entry(namespace: str, body: EntryWrite, response: Response):
    """
    Write or upsert a memory entry.
    - 201 Created  on first insert.
    - 200 OK       on update (id and created_at are preserved).
    Backend: Relational (persist) + optional Vector index (if embedding provided).
    """
    try:
        _validate_namespace(namespace)
    except ValueError as e:
        return _ns_error(e)

    entry, created = _s().write(namespace, body)
    response.status_code = 201 if created else 200
    response.headers["X-Entry-Created"] = str(created).lower()
    return entry.model_dump()


# ── Read ─────────────────────────────────────────────────────────────────────

@router.get("/{namespace:path}/entries/{key:path}")
async def read_entry(namespace: str, key: str):
    """
    Read a single entry by key.
    Backend: Relational.
    """
    try:
        _validate_namespace(namespace)
    except ValueError as e:
        return _ns_error(e)

    entry = _s().read(namespace, key)
    if entry is None:
        return JSONResponse(
            status_code=404,
            content=ErrorBody(
                code="ENTRY_NOT_FOUND",
                message=f"No entry with key '{key}' in namespace '{namespace}'",
            ).model_dump(),
        )
    return entry.model_dump()


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/{namespace:path}/entries")
async def list_entries(
    namespace: str,
    type:      Optional[MemoryType] = Query(None),
    tag:       list[str]            = Query(default=[]),
    limit:     int                  = Query(50, ge=1, le=500),
    offset:    int                  = Query(0, ge=0),
    recursive: bool                 = Query(False),
):
    """
    List and filter entries within a namespace.
    Backend: Relational.
    """
    try:
        _validate_namespace(namespace)
    except ValueError as e:
        return _ns_error(e)

    entries, total = _s().list(
        namespace,
        type=type,
        tags=tag or None,
        limit=limit,
        offset=offset,
        recursive=recursive,
    )
    next_offset = offset + limit if offset + limit < total else None
    return ListResponse(
        entries=entries, total=total, next_offset=next_offset
    ).model_dump()


# ── Semantic query ────────────────────────────────────────────────────────────

@router.post("/{namespace:path}/query")
async def query_entries(namespace: str, body: QueryRequest):
    """
    Semantic nearest-neighbour search.
    Caller MUST supply `embedding` — the server never generates embeddings (spec §5.1).
    Backend:
      1. Relational — fetch candidates matching type / tag filters (embedding IS NOT NULL).
      2. Vector     — rank candidates by cosine similarity, return top_k results.
    """
    try:
        _validate_namespace(namespace)
    except ValueError as e:
        return _ns_error(e)

    if not body.embedding:
        return JSONResponse(
            status_code=400,
            content=ErrorBody(
                code="INVALID_BODY",
                message="'embedding' is required and must be a non-empty list of floats",
            ).model_dump(),
        )

    results = _s().query(
        namespace,
        embedding=body.embedding,
        type=body.type,
        top_k=body.top_k,
        min_score=body.min_score,
        tags=body.tag or None,
        recursive=body.recursive,
    )
    return [QueryResult(entry=e, score=s).model_dump() for e, s in results]


# ── Delete single ─────────────────────────────────────────────────────────────

@router.delete("/{namespace:path}/entries/{key:path}", status_code=204)
async def delete_entry(namespace: str, key: str, response: Response):
    """
    Delete a single entry by key.
    Backend: Relational.
    """
    try:
        _validate_namespace(namespace)
    except ValueError as e:
        return _ns_error(e)

    deleted = _s().delete(namespace, key)
    if not deleted:
        return JSONResponse(
            status_code=404,
            content=ErrorBody(
                code="ENTRY_NOT_FOUND",
                message=f"No entry with key '{key}' in namespace '{namespace}'",
            ).model_dump(),
        )
    deleted_count = 1
    response.headers["X-Deleted-Count"] = str(deleted_count)
    return Response(status_code=204)


# ── Bulk delete ───────────────────────────────────────────────────────────────

@router.delete("/{namespace:path}/entries", status_code=204)
async def delete_bulk(
    namespace: str,
    response:  Response,
    type:      Optional[MemoryType] = Query(None),
    tag:       list[str]            = Query(default=[]),
):
    """
    Bulk delete all entries in namespace matching optional type / tag filters.
    No filters → clears the entire namespace.
    Backend: Relational.
    """
    try:
        _validate_namespace(namespace)
    except ValueError as e:
        return _ns_error(e)

    count = _s().delete_bulk(namespace, type=type, tags=tag or None)
    response.headers["X-Deleted-Count"] = str(count)
    return Response(status_code=204)


# ── WebSocket subscribe ───────────────────────────────────────────────────────

@router.websocket("/{namespace:path}/subscribe")
async def subscribe_ws(
    websocket: WebSocket,
    namespace: str,
    pattern:   str = "*",
):
    """
    Real-time event stream for a namespace.
    Sends {"event": "write"|"delete", "entry": {...}} when entries change.
    `pattern` is a glob against entry keys (e.g. "user/*").
    """
    try:
        _validate_namespace(namespace)
    except ValueError:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    q = _s().subscribe(namespace)
    try:
        while True:
            msg = await q.get()
            key = msg["entry"].get("key", "")
            if fnmatch(key, pattern):
                await websocket.send_json(msg)
    except WebSocketDisconnect:
        _s().unsubscribe(namespace, q)
    except Exception:
        _s().unsubscribe(namespace, q)
        raise
