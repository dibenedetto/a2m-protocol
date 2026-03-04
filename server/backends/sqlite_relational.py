"""
Agent2Memory (A2M) Protocol
Default SQLite relational backend: SQLiteRelationalBackend.

Requires no external dependencies beyond the Python standard library.
It is the default relational backend used when create_app() is called
without an explicit relational= argument.

Embeddings are stored as JSON in the entries table so the relational store
remains the single source of truth for all data, regardless of which vector
backend is configured.

The numpy vector backend has been moved to numpy_backend.py.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..models import Entry, EntryMeta, EntryWrite, MemoryType
from .base import AbstractRelationalBackend


# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


# ── SQLiteRelationalBackend ───────────────────────────────────────────────────

class SQLiteRelationalBackend(AbstractRelationalBackend):
    """SQLite-backed relational store. Thread-safe via a single threading.Lock."""

    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── private ──────────────────────────────────────────────────────────────

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

    # ── AbstractRelationalBackend ─────────────────────────────────────────────

    def upsert(self, namespace: str, ew: EntryWrite) -> tuple[Entry, bool]:
        now = _now()
        expires_at: Optional[str] = None
        if ew.meta.ttl_s is not None:
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
                entry_id   = existing["id"]
                created_at = existing["created_at"]
                created    = False
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
                entry_id   = str(uuid.uuid4())
                created_at = now
                created    = True
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

        meta = EntryMeta(
            source_agent=ew.meta.source_agent,
            source_framework=ew.meta.source_framework,
            created_at=created_at,
            updated_at=now,
            ttl_s=ew.meta.ttl_s,
            tags=ew.meta.tags,
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
        now = _now()

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

        where    = " AND ".join(clauses)
        count_sql = f"SELECT COUNT(*) FROM entries WHERE {where}"
        data_sql  = f"SELECT * FROM entries WHERE {where} ORDER BY created_at ASC LIMIT ? OFFSET ?"

        with self._lock:
            total = self._conn.execute(count_sql, params).fetchone()[0]
            rows  = self._conn.execute(data_sql, params + [limit, offset]).fetchall()

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
        now = _now()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM entries WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,)
            )
            self._conn.commit()
        return cur.rowcount

    def all_with_embeddings(self) -> list[Entry]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM entries WHERE embedding IS NOT NULL"
            ).fetchall()
        return [_row_to_entry(r, self._get_tags(r["id"])) for r in rows]



