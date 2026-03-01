"""
A2M — AgentToMemory Protocol
PostgreSQL relational backend: PostgreSQLRelationalBackend.

Implements AbstractRelationalBackend using PostgreSQL.
Schema is equivalent to SQLiteRelationalBackend; only SQL dialect differences apply.

Key differences from SQLite:
  - Paramstyle uses %s instead of ?
  - DOUBLE PRECISION instead of REAL for confidence
  - threading.RLock (re-entrant) so _get_tags() can be called while the
    outer lock is held (psycopg2 connections are not thread-safe on their own).

Install requirement:
    pip install psycopg2-binary

Usage:
    from server.backends.postgres import PostgreSQLRelationalBackend
    from server.main import create_app

    app = create_app(
        relational=PostgreSQLRelationalBackend(
            dsn="postgresql://user:pass@localhost/a2m"
        ),
    )
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    import psycopg2
    import psycopg2.extras
except ImportError as exc:
    raise ImportError(
        "psycopg2-binary is required for PostgreSQLRelationalBackend. "
        "Install it with: pip install psycopg2-binary"
    ) from exc

from ..models import Entry, EntryMeta, EntryWrite, MemoryType
from .base import AbstractRelationalBackend


# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_entry(row: dict, tags: list[str]) -> Entry:
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


_SCHEMA_STMTS = [
    """
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
        confidence   DOUBLE PRECISION,
        UNIQUE(namespace, key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entry_tags (
        entry_id TEXT NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
        tag      TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ns      ON entries(namespace)",
    "CREATE INDEX IF NOT EXISTS idx_type    ON entries(type)",
    "CREATE INDEX IF NOT EXISTS idx_exp     ON entries(expires_at) WHERE expires_at IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_tag_eid ON entry_tags(entry_id)",
]


# ── PostgreSQLRelationalBackend ───────────────────────────────────────────────

class PostgreSQLRelationalBackend(AbstractRelationalBackend):
    """
    PostgreSQL-backed relational store.

    Thread-safe via a threading.RLock (re-entrant so _get_tags() can be
    called from within other locked methods without deadlocking).
    A single connection is shared across threads; all access is serialised
    through the lock.
    """

    def __init__(self, dsn: str) -> None:
        """
        Args:
            dsn: PostgreSQL connection string, e.g.:
                 "postgresql://user:pass@localhost/a2m"
                 "host=localhost dbname=a2m user=postgres password=secret"
        """
        self._lock = threading.RLock()
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        with self._lock:
            with self._conn.cursor() as cur:
                for stmt in _SCHEMA_STMTS:
                    cur.execute(stmt)
            self._conn.commit()

    # ── private ──────────────────────────────────────────────────────────────

    def _cur(self):
        """Return a RealDictCursor (dict-like row access)."""
        return self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def _get_tags(self, entry_id: str) -> list[str]:
        """Fetch tags for one entry. May be called inside or outside the lock."""
        with self._lock:
            with self._cur() as cur:
                cur.execute(
                    "SELECT tag FROM entry_tags WHERE entry_id = %s", (entry_id,)
                )
                return [r["tag"] for r in cur.fetchall()]

    def _set_tags(self, cur, entry_id: str, tags: list[str]) -> None:
        cur.execute("DELETE FROM entry_tags WHERE entry_id = %s", (entry_id,))
        for tag in tags:
            cur.execute(
                "INSERT INTO entry_tags (entry_id, tag) VALUES (%s, %s)",
                (entry_id, tag),
            )

    def _is_expired(self, row: dict) -> bool:
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
            with self._cur() as cur:
                cur.execute(
                    "SELECT id, created_at FROM entries WHERE namespace = %s AND key = %s",
                    (namespace, ew.key),
                )
                existing = cur.fetchone()

                if existing:
                    entry_id   = existing["id"]
                    created_at = existing["created_at"]
                    created    = False
                    cur.execute(
                        """UPDATE entries SET
                            type=%s, value=%s, embedding=%s, source_agent=%s, source_fw=%s,
                            updated_at=%s, ttl_s=%s, expires_at=%s, confidence=%s
                           WHERE id=%s""",
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
                    cur.execute(
                        """INSERT INTO entries
                            (id, namespace, key, type, value, embedding, source_agent, source_fw,
                             created_at, updated_at, ttl_s, expires_at, confidence)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (
                            entry_id, namespace, ew.key, ew.type.value, value_json,
                            embedding_json, ew.meta.source_agent, ew.meta.source_framework,
                            created_at, now, ew.meta.ttl_s, expires_at, ew.meta.confidence,
                        ),
                    )

                self._set_tags(cur, entry_id, ew.meta.tags)
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
            with self._cur() as cur:
                cur.execute(
                    "SELECT * FROM entries WHERE namespace = %s AND key = %s",
                    (namespace, key),
                )
                row = cur.fetchone()
            if row is None or self._is_expired(row):
                return None
            tags = self._get_tags(row["id"])   # RLock allows re-entry
        return _row_to_entry(row, tags)

    def select(
        self,
        namespace: str,
        type:          Optional[MemoryType] = None,
        tags:          Optional[list[str]]  = None,
        limit:         int                  = 50,
        offset:        int                  = 0,
        recursive:     bool                 = False,
        has_embedding: bool                 = False,
    ) -> tuple[list[Entry], int]:
        now = _now()

        if recursive:
            ns_clause  = "(namespace = %s OR namespace LIKE %s)"
            ns_params: list = [namespace, namespace + "/%"]
        else:
            ns_clause  = "namespace = %s"
            ns_params  = [namespace]

        clauses = [ns_clause, "(expires_at IS NULL OR expires_at > %s)"]
        params: list = ns_params + [now]

        if type is not None:
            clauses.append("type = %s")
            params.append(type.value)

        if has_embedding:
            clauses.append("embedding IS NOT NULL")

        if tags:
            placeholders = ",".join(["%s"] * len(tags))
            clauses.append(
                f"id IN (SELECT entry_id FROM entry_tags WHERE tag IN ({placeholders})"
                f" GROUP BY entry_id HAVING COUNT(DISTINCT tag) = %s)"
            )
            params.extend(tags)
            params.append(len(tags))

        where     = " AND ".join(clauses)
        count_sql = f"SELECT COUNT(*) FROM entries WHERE {where}"
        data_sql  = (
            f"SELECT * FROM entries WHERE {where} "
            f"ORDER BY created_at ASC LIMIT %s OFFSET %s"
        )

        with self._lock:
            with self._cur() as cur:
                cur.execute(count_sql, params)
                total = cur.fetchone()["count"]
                cur.execute(data_sql, params + [limit, offset])
                rows = cur.fetchall()
            entries = [_row_to_entry(r, self._get_tags(r["id"])) for r in rows]
        return entries, total

    def delete_one(self, namespace: str, key: str) -> bool:
        with self._lock:
            with self._cur() as cur:
                cur.execute(
                    "DELETE FROM entries WHERE namespace = %s AND key = %s",
                    (namespace, key),
                )
                rowcount = cur.rowcount
            self._conn.commit()
        return rowcount > 0

    def delete_many(
        self,
        namespace: str,
        type: Optional[MemoryType] = None,
        tags: Optional[list[str]]  = None,
    ) -> int:
        clauses = ["namespace = %s"]
        params: list = [namespace]

        if type is not None:
            clauses.append("type = %s")
            params.append(type.value)

        if tags:
            placeholders = ",".join(["%s"] * len(tags))
            clauses.append(
                f"id IN (SELECT entry_id FROM entry_tags WHERE tag IN ({placeholders})"
                f" GROUP BY entry_id HAVING COUNT(DISTINCT tag) = %s)"
            )
            params.extend(tags)
            params.append(len(tags))

        where = " AND ".join(clauses)
        with self._lock:
            with self._cur() as cur:
                cur.execute(f"DELETE FROM entries WHERE {where}", params)
                rowcount = cur.rowcount
            self._conn.commit()
        return rowcount

    def expire(self) -> int:
        now = _now()
        with self._lock:
            with self._cur() as cur:
                cur.execute(
                    "DELETE FROM entries WHERE expires_at IS NOT NULL AND expires_at <= %s",
                    (now,),
                )
                rowcount = cur.rowcount
            self._conn.commit()
        return rowcount

    def all_with_embeddings(self) -> list[Entry]:
        with self._lock:
            with self._cur() as cur:
                cur.execute("SELECT * FROM entries WHERE embedding IS NOT NULL")
                rows = cur.fetchall()
            return [_row_to_entry(r, self._get_tags(r["id"])) for r in rows]
