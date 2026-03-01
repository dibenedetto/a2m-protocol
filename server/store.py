"""
A2M — AgentToMemory Protocol
Storage layer: RelationalBackend (SQLite) + VectorBackend (numpy cosine) + A2MStore coordinator.

Architecture:
  - RelationalBackend  — all structured operations (CRUD, filtering, TTL, tags).
  - VectorBackend      — stateless cosine ranking over embeddings fetched from relational.
  - A2MStore           — public API; coordinates both backends + in-memory pub/sub for WS.

The embedding is stored in the relational backend (as JSON) so that it is the single source
of truth. The vector backend is a pure compute layer — it holds no persistent state.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from fnmatch import fnmatch
from typing import Optional

import numpy as np

from .models import Entry, EntryMeta, EntryWrite, MemoryType


# ── helpers ─────────────────────────────────────────────────────────────────

_NS_SEGMENT = re.compile(r'^[a-z0-9_-]+$')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_namespace(namespace: str) -> None:
    if not namespace or len(namespace) > 256:
        raise ValueError(f"Namespace must be 1–256 chars, got {len(namespace)!r}")
    for seg in namespace.split("/"):
        if not seg or not _NS_SEGMENT.match(seg):
            raise ValueError(
                f"Namespace segment {seg!r} must match [a-z0-9_-]+"
            )


def _row_to_entry(row: sqlite3.Row, tags: list[str]) -> Entry:
    embedding = json.loads(row["embedding"]) if row["embedding"] else None
    meta = EntryMeta(
        source_agent=row["source_agent"],
        source_framework=row["source_fw"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        ttl_s=row["ttl_s"],
        tags=tags,
        confidence=row["confidence"],
    )
    return Entry(
        id=row["id"],
        key=row["key"],
        namespace=row["namespace"],
        type=MemoryType(row["type"]),
        value=json.loads(row["value"]),
        embedding=embedding,
        meta=meta,
    )


# ── RelationalBackend ────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS entries (
    id           TEXT PRIMARY KEY,
    namespace    TEXT NOT NULL,
    key          TEXT NOT NULL,
    type         TEXT NOT NULL,
    value        TEXT NOT NULL,
    embedding    TEXT,
    source_agent TEXT,
    source_fw    TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    ttl_s        INTEGER,
    expires_at   TEXT,
    confidence   REAL,
    UNIQUE(namespace, key)
);

CREATE TABLE IF NOT EXISTS entry_tags (
    entry_id TEXT NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    tag      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ns      ON entries(namespace);
CREATE INDEX IF NOT EXISTS idx_type    ON entries(type);
CREATE INDEX IF NOT EXISTS idx_exp     ON entries(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tag_eid ON entry_tags(entry_id);
"""


class RelationalBackend:
    """SQLite-backed relational store. Thread-safe via a single lock."""

    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── internal ────────────────────────────────────────────────────────────

    def _get_tags(self, entry_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT tag FROM entry_tags WHERE entry_id = ?", (entry_id,)
        ).fetchall()
        return [r["tag"] for r in rows]

    def _set_tags(self, entry_id: str, tags: list[str]) -> None:
        self._conn.execute("DELETE FROM entry_tags WHERE entry_id = ?", (entry_id,))
        for tag in tags:
            self._conn.execute(
                "INSERT INTO entry_tags(entry_id, tag) VALUES (?, ?)", (entry_id, tag)
            )

    def _is_expired(self, row: sqlite3.Row) -> bool:
        if not row["expires_at"]:
            return False
        return row["expires_at"] <= _now()

    # ── public ──────────────────────────────────────────────────────────────

    def upsert(self, namespace: str, ew: EntryWrite) -> tuple[Entry, bool]:
        """Upsert an entry. Returns (Entry, created: bool).
        On update: preserves id and created_at (spec §5.4).
        """
        now = _now()
        expires_at: Optional[str] = None
        if ew.meta.ttl_s is not None:
            from datetime import timedelta
            expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=ew.meta.ttl_s)
            ).isoformat()

        embedding_json = json.dumps(ew.embedding) if ew.embedding is not None else None
        value_json = json.dumps(ew.value)

        with self._lock:
            existing = self._conn.execute(
                "SELECT id, created_at FROM entries WHERE namespace = ? AND key = ?",
                (namespace, ew.key),
            ).fetchone()

            if existing:
                entry_id = existing["id"]
                created_at = existing["created_at"]
                created = False
                self._conn.execute(
                    """UPDATE entries SET
                        type=?, value=?, embedding=?, source_agent=?, source_fw=?,
                        updated_at=?, ttl_s=?, expires_at=?, confidence=?
                       WHERE id=?""",
                    (
                        ew.type.value, value_json, embedding_json,
                        ew.meta.source_agent, ew.meta.source_framework,
                        now, ew.meta.ttl_s, expires_at, ew.meta.confidence,
                        entry_id,
                    ),
                )
            else:
                entry_id = str(uuid.uuid4())
                created_at = now
                created = True
                self._conn.execute(
                    """INSERT INTO entries
                        (id, namespace, key, type, value, embedding, source_agent, source_fw,
                         created_at, updated_at, ttl_s, expires_at, confidence)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        entry_id, namespace, ew.key, ew.type.value, value_json,
                        embedding_json, ew.meta.source_agent, ew.meta.source_framework,
                        created_at, now, ew.meta.ttl_s, expires_at, ew.meta.confidence,
                    ),
                )

            self._set_tags(entry_id, ew.meta.tags)
            self._conn.commit()

        tags = ew.meta.tags
        meta = EntryMeta(
            source_agent=ew.meta.source_agent,
            source_framework=ew.meta.source_framework,
            created_at=created_at,
            updated_at=now,
            ttl_s=ew.meta.ttl_s,
            tags=tags,
            confidence=ew.meta.confidence,
        )
        return Entry(
            id=entry_id, key=ew.key, namespace=namespace,
            type=ew.type, value=ew.value,
            embedding=ew.embedding, meta=meta,
        ), created

    def get(self, namespace: str, key: str) -> Optional[Entry]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
        if row is None or self._is_expired(row):
            return None
        return _row_to_entry(row, self._get_tags(row["id"]))

    def select(
        self,
        namespace: str,
        type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
        limit: int = 50,
        offset: int = 0,
        recursive: bool = False,
        has_embedding: bool = False,
    ) -> tuple[list[Entry], int]:
        """Returns (entries, total_matching_count)."""
        now = _now()

        # Namespace filter
        if recursive:
            ns_clause = "(namespace = ? OR namespace LIKE ?)"
            ns_params: list = [namespace, namespace + "/%"]
        else:
            ns_clause = "namespace = ?"
            ns_params = [namespace]

        clauses = [ns_clause, "(expires_at IS NULL OR expires_at > ?)"]
        params: list = ns_params + [now]

        if type is not None:
            clauses.append("type = ?")
            params.append(type.value)

        if has_embedding:
            clauses.append("embedding IS NOT NULL")

        if tags:
            placeholders = ",".join("?" * len(tags))
            clauses.append(
                f"id IN (SELECT entry_id FROM entry_tags WHERE tag IN ({placeholders})"
                f" GROUP BY entry_id HAVING COUNT(DISTINCT tag) = ?)"
            )
            params.extend(tags)
            params.append(len(tags))

        where = " AND ".join(clauses)
        count_sql = f"SELECT COUNT(*) FROM entries WHERE {where}"
        data_sql  = f"SELECT * FROM entries WHERE {where} ORDER BY created_at ASC LIMIT ? OFFSET ?"

        with self._lock:
            total = self._conn.execute(count_sql, params).fetchone()[0]
            rows = self._conn.execute(data_sql, params + [limit, offset]).fetchall()

        entries = [_row_to_entry(r, self._get_tags(r["id"])) for r in rows]
        return entries, total

    def delete_one(self, namespace: str, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM entries WHERE namespace = ? AND key = ?", (namespace, key)
            )
            self._conn.commit()
        return cur.rowcount > 0

    def delete_many(
        self,
        namespace: str,
        type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
    ) -> int:
        clauses = ["namespace = ?"]
        params: list = [namespace]

        if type is not None:
            clauses.append("type = ?")
            params.append(type.value)

        if tags:
            placeholders = ",".join("?" * len(tags))
            clauses.append(
                f"id IN (SELECT entry_id FROM entry_tags WHERE tag IN ({placeholders})"
                f" GROUP BY entry_id HAVING COUNT(DISTINCT tag) = ?)"
            )
            params.extend(tags)
            params.append(len(tags))

        where = " AND ".join(clauses)
        with self._lock:
            cur = self._conn.execute(f"DELETE FROM entries WHERE {where}", params)
            self._conn.commit()
        return cur.rowcount

    def expire(self) -> int:
        """Delete entries past their expires_at. Returns count deleted."""
        now = _now()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM entries WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,)
            )
            self._conn.commit()
        return cur.rowcount


# ── VectorBackend ────────────────────────────────────────────────────────────

class VectorBackend:
    """Stateless cosine similarity ranking over caller-provided embeddings.

    Takes candidate Entry objects (fetched from the relational backend) and
    ranks them against a query vector. No persistent state — the relational
    backend is the single source of truth for all data including embeddings.
    """

    def search(
        self,
        candidates: list[Entry],
        query_vec: list[float],
        top_k: int,
        min_score: Optional[float] = None,
    ) -> list[tuple[Entry, float]]:
        """Return top_k candidates ranked by cosine similarity, highest first."""
        if not candidates or not query_vec:
            return []

        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return []
        q = q / q_norm

        results: list[tuple[Entry, float]] = []
        for entry in candidates:
            if entry.embedding is None:
                continue
            e = np.array(entry.embedding, dtype=np.float32)
            e_norm = np.linalg.norm(e)
            if e_norm < 1e-10:
                continue
            score = float(np.dot(q, e / e_norm))
            if min_score is None or score >= min_score:
                results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ── A2MStore ─────────────────────────────────────────────────────────────────

class A2MStore:
    """
    Public A2M storage interface. Coordinates:
      - RelationalBackend  for all structured operations (write, read, list, delete, TTL)
      - VectorBackend      for semantic query (cosine ranking over relational candidates)
      - In-memory pub/sub  for WebSocket subscriptions
    """

    def __init__(self, db_path: str = "a2m.db") -> None:
        self.relational = RelationalBackend(db_path)
        self.vector     = VectorBackend()
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)

    # ── write / read / delete ────────────────────────────────────────────────

    def write(self, namespace: str, ew: EntryWrite) -> tuple[Entry, bool]:
        """Upsert an entry. Uses: Relational (persist). Returns (Entry, created)."""
        _validate_namespace(namespace)
        entry, created = self.relational.upsert(namespace, ew)
        self._notify(namespace, "write", entry)
        return entry, created

    def read(self, namespace: str, key: str) -> Optional[Entry]:
        """Fetch a single entry by key. Uses: Relational."""
        return self.relational.get(namespace, key)

    def list(
        self,
        namespace: str,
        type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
        limit: int = 50,
        offset: int = 0,
        recursive: bool = False,
    ) -> tuple[list[Entry], int]:
        """List and filter entries. Uses: Relational."""
        return self.relational.select(namespace, type, tags, limit, offset, recursive)

    def delete(self, namespace: str, key: str) -> bool:
        """Delete a single entry. Uses: Relational."""
        entry = self.relational.get(namespace, key)
        deleted = self.relational.delete_one(namespace, key)
        if deleted and entry:
            self._notify(namespace, "delete", entry)
        return deleted

    def delete_bulk(
        self,
        namespace: str,
        type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
    ) -> int:
        """Bulk delete matching entries. Uses: Relational."""
        return self.relational.delete_many(namespace, type, tags)

    # ── semantic query ───────────────────────────────────────────────────────

    def query(
        self,
        namespace: str,
        embedding: list[float],
        type: Optional[MemoryType] = None,
        top_k: int = 5,
        min_score: Optional[float] = None,
        tags: Optional[list[str]] = None,
        recursive: bool = False,
    ) -> list[tuple[Entry, float]]:
        """
        Semantic search. Uses:
          1. Relational — fetch candidates with non-null embedding matching filters.
          2. Vector     — rank candidates by cosine similarity against query embedding.
        """
        candidates, _ = self.relational.select(
            namespace,
            type=type,
            tags=tags,
            limit=10_000,
            offset=0,
            recursive=recursive,
            has_embedding=True,
        )
        return self.vector.search(candidates, embedding, top_k, min_score)

    # ── TTL expiry ───────────────────────────────────────────────────────────

    def expire(self) -> int:
        """Remove expired entries. Uses: Relational. Called by background loop."""
        return self.relational.expire()

    # ── pub/sub for WebSocket subscriptions ─────────────────────────────────

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
        """Push an event to all subscribers whose namespace matches or is a prefix."""
        msg = {"event": event, "entry": entry.model_dump()}
        for sub_ns, queues in list(self._subs.items()):
            # Notify if subscription namespace equals or is a parent of the entry namespace
            if namespace == sub_ns or namespace.startswith(sub_ns + "/"):
                for q in queues:
                    try:
                        q.put_nowait(msg)
                    except asyncio.QueueFull:
                        pass
