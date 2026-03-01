"""
A2M — AgentToMemory Protocol
Storage coordinator: A2MStore.

A2MStore owns one AbstractRelationalBackend and one AbstractVectorBackend and
provides the public API used by all route handlers.

Default backends (zero external dependencies):
    relational → SQLiteRelationalBackend    (server/backends/sqlite_relational.py)
    vector     → NumpyVectorBackend         (server/backends/numpy_vector.py)

Production backends:
    relational → PostgreSQLRelationalBackend (server/backends/postgres_relational.py)
    vector     → LanceVectorBackend          (server/backends/lancedb_vector.py)
    vector     → PgVectorBackend             (server/backends/pgvector_vector.py)
    vector     → ChromaVectorBackend         (server/backends/chroma_vector.py)

On startup, A2MStore calls vector.rebuild(all_with_embeddings) so that
in-memory backends (NumpyVectorBackend) are re-populated from the relational
store.  Persistent backends (LanceVectorBackend) ignore the call.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from typing import Optional

from .backends.base import AbstractRelationalBackend, AbstractVectorBackend
from .models import Entry, EntryWrite, MemoryType


# ── helpers ───────────────────────────────────────────────────────────────────

_NS_SEGMENT = re.compile(r'^[a-z0-9_-]+$')


def _validate_namespace(namespace: str) -> None:
    if not namespace or len(namespace) > 256:
        raise ValueError(f"Namespace must be 1-256 chars, got {len(namespace)!r}")
    for seg in namespace.split("/"):
        if not seg or not _NS_SEGMENT.match(seg):
            raise ValueError(
                f"Namespace segment {seg!r} must match [a-z0-9_-]+"
            )


# ── A2MStore ──────────────────────────────────────────────────────────────────

class A2MStore:
    """
    Public A2M storage interface. Coordinates:
      - AbstractRelationalBackend  — all structured ops (write, read, list, delete, TTL)
      - AbstractVectorBackend      — embedding index + ANN search
      - In-memory pub/sub          — WebSocket event delivery

    On write:   relational.upsert()   + vector.index()   (if embedding present)
    On delete:  relational.delete_*() + vector.remove*()
    On query:   vector.query() -> [(ns, key, score)] -> relational.get() per hit
    """

    def __init__(
        self,
        relational: AbstractRelationalBackend,
        vector:     AbstractVectorBackend,
    ) -> None:
        self.relational = relational
        self.vector     = vector
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)

        # Seed in-memory backends (e.g. NumpyVectorBackend) from the relational
        # store.  Persistent backends (LanceVectorBackend) treat this as a no-op.
        seeded = self.relational.all_with_embeddings()
        self.vector.rebuild(seeded)

    # ── write / read / delete ─────────────────────────────────────────────────

    def write(self, namespace: str, ew: EntryWrite) -> tuple[Entry, bool]:
        """
        Upsert an entry.
        Relational: always.  Vector: only when embedding is present.
        Returns (Entry, created: bool).
        """
        _validate_namespace(namespace)
        entry, created = self.relational.upsert(namespace, ew)
        if entry.embedding is not None:
            self.vector.index(entry.id, namespace, entry.key, entry.embedding)
        self._notify(namespace, "write", entry)
        return entry, created

    def read(self, namespace: str, key: str) -> Optional[Entry]:
        """Fetch a single entry by key. Relational only."""
        return self.relational.get(namespace, key)

    def list(
        self,
        namespace: str,
        type:      Optional[MemoryType] = None,
        tags:      Optional[list[str]]  = None,
        limit:     int                  = 50,
        offset:    int                  = 0,
        recursive: bool                 = False,
    ) -> tuple[list[Entry], int]:
        """List and filter entries. Relational only."""
        return self.relational.select(namespace, type, tags, limit, offset, recursive)

    def delete(self, namespace: str, key: str) -> bool:
        """
        Delete a single entry.
        Relational + Vector (removes from index if the entry had an embedding).
        """
        entry   = self.relational.get(namespace, key)   # fetch before delete for id + notify
        deleted = self.relational.delete_one(namespace, key)
        if deleted and entry:
            if entry.embedding is not None:
                self.vector.remove(entry.id, namespace, key)
            self._notify(namespace, "delete", entry)
        return deleted

    def delete_bulk(
        self,
        namespace: str,
        type:      Optional[MemoryType] = None,
        tags:      Optional[list[str]]  = None,
    ) -> int:
        """
        Bulk-delete entries matching optional type/tag filters.
        Removes matched entries from both the relational store and the vector index.
        """
        # Fetch before delete so we have entry ids for vector cleanup.
        to_delete, _ = self.relational.select(
            namespace, type=type, tags=tags, limit=100_000, offset=0
        )
        count = self.relational.delete_many(namespace, type, tags)
        for e in to_delete:
            if e.embedding is not None:
                self.vector.remove(e.id, namespace, e.key)
        return count

    # ── semantic query ────────────────────────────────────────────────────────

    def query(
        self,
        namespace:  str,
        embedding:  list[float],
        type:       Optional[MemoryType] = None,
        top_k:      int                  = 5,
        min_score:  Optional[float]      = None,
        tags:       Optional[list[str]]  = None,
        recursive:  bool                 = False,
    ) -> list[tuple[Entry, float]]:
        """
        Semantic search.
        1. Vector:     ANN search -> list of (namespace, key, score).
        2. Relational: hydrate each Entry from (namespace, key); apply type/tag filters.
        """
        hits = self.vector.query(embedding, namespace, top_k, min_score, recursive)

        results: list[tuple[Entry, float]] = []
        for ns, key, score in hits:
            entry = self.relational.get(ns, key)
            if entry is None:
                continue  # stale vector entry (already deleted from relational)
            if type is not None and entry.type != type:
                continue
            if tags:
                entry_tag_set = set(entry.meta.tags)
                if not all(t in entry_tag_set for t in tags):
                    continue
            results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ── TTL expiry ────────────────────────────────────────────────────────────

    def expire(self) -> int:
        """Remove TTL-expired entries from the relational store. Called by background loop."""
        return self.relational.expire()

    # ── health ─────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """
        Probe both backends and return status info.
        Used by the /a2m/v1/health endpoint.
        """
        rel_ok = True
        try:
            self.relational.select("__health__", limit=1)
        except Exception:
            rel_ok = False
        return {
            "status":     "ok" if rel_ok else "degraded",
            "relational": {
                "ok":   rel_ok,
                "type": type(self.relational).__name__,
            },
            "vector": {
                "type": type(self.vector).__name__,
            },
        }

    # ── pub/sub for WebSocket subscriptions ──────────────────────────────────

    def subscribe(self, namespace: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs[namespace].append(q)
        return q

    def unsubscribe(self, namespace: str, q: asyncio.Queue) -> None:
        try:
            self._subs[namespace].remove(q)
        except ValueError:
            pass

    def _notify(self, namespace: str, event: str, entry: Entry) -> None:
        """Push a write/delete event to all matching WebSocket subscribers."""
        msg = {"event": event, "entry": entry.model_dump()}
        for sub_ns, queues in list(self._subs.items()):
            if namespace == sub_ns or namespace.startswith(sub_ns + "/"):
                for q in queues:
                    try:
                        q.put_nowait(msg)
                    except asyncio.QueueFull:
                        pass
