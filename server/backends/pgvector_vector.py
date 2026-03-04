"""
Agent2Memory (A2M) Protocol
pgvector vector backend: PgVectorBackend.

Implements AbstractVectorBackend using PostgreSQL + pgvector extension.
Stores embeddings in a dedicated table alongside (not inside) the relational
entries table; can share the same PostgreSQL server.

Key properties:
  - Persistent: index survives server restarts.
  - Cosine similarity via the <=> operator (cosine distance).
  - Upsert via ON CONFLICT DO UPDATE keyed on entry_id.
  - Namespace filtering is applied in SQL for exact matches and in Python
    for recursive prefix matches.
  - Table dimension is fixed on first index() call; subsequent calls must
    use the same embedding dimension.

Install requirements:
    pip install psycopg2-binary pgvector

pgvector PostgreSQL extension must be installed on the server:
    CREATE EXTENSION IF NOT EXISTS vector;   -- run once as superuser

Usage:
    from server.backends.pgvector_backend import PgVectorBackend
    from server.backends.postgres import PostgreSQLRelationalBackend
    from server.main import create_app

    dsn = "postgresql://user:pass@localhost/a2m"
    app = create_app(
        relational=PostgreSQLRelationalBackend(dsn),
        vector=PgVectorBackend(dsn),
    )
"""

from __future__ import annotations

import threading
from typing import Optional

try:
    import psycopg2
    import psycopg2.extras
except ImportError as exc:
    raise ImportError(
        "psycopg2-binary is required for PgVectorBackend. "
        "Install it with: pip install psycopg2-binary"
    ) from exc

try:
    import numpy as np
    from pgvector.psycopg2 import register_vector
except ImportError as exc:
    raise ImportError(
        "pgvector and numpy are required for PgVectorBackend. "
        "Install them with: pip install pgvector numpy"
    ) from exc

from .base import AbstractVectorBackend, Entry

# Extra candidates fetched before Python-side namespace prefix filtering.
_OVER_FETCH = 10


class PgVectorBackend(AbstractVectorBackend):
    """
    pgvector-backed ANN vector index.

    The table is created lazily on the first index() call once the embedding
    dimension is known.  All writes are upserts keyed on entry_id.

    Cosine distance convention (same as LanceVectorBackend):
      _distance = 0   → identical vectors   (score = 1.0)
      _distance = 1   → orthogonal vectors  (score = 0.0)
      _distance = 2   → opposite vectors    (score = -1.0)
      score = 1.0 - _distance
    """

    def __init__(
        self,
        dsn: str,
        table_name: str = "a2m_vectors",
    ) -> None:
        """
        Args:
            dsn:        PostgreSQL connection string, e.g.
                        "postgresql://user:pass@localhost/a2m"
            table_name: Name of the table that stores A2M vectors.
                        Use a different name to share a database with multiple
                        A2M instances.
        """
        self._table_name = table_name
        self._dim: Optional[int] = None
        self._lock = threading.Lock()
        self._ready = False   # True once the table + index exist

        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        register_vector(self._conn)

        # Ensure the pgvector extension exists.
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.commit()

        # If the table already exists, open it and infer the dimension.
        self._try_open_existing()

    # ── private ───────────────────────────────────────────────────────────────

    def _cur(self):
        return self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def _try_open_existing(self) -> None:
        """If the table already exists, infer dimension from it."""
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = %s AND column_name = 'embedding'",
                    (self._table_name,),
                )
                if cur.fetchone() is None:
                    return   # table does not exist yet

                # Infer dimension from the first row that has an embedding.
                cur.execute(
                    f"SELECT embedding FROM {self._table_name} LIMIT 1"
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    self._dim = len(row[0])

            self._ready = True

    def _ensure_table(self, dim: int) -> None:
        """Create table + HNSW index the first time we see an embedding."""
        with self._lock:
            if self._ready:
                return
            with self._conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table_name} (
                        entry_id  TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        entry_key TEXT NOT NULL,
                        embedding vector({dim})
                    )
                """)
                # HNSW index for fast ANN (pgvector >= 0.5); falls back
                # gracefully if the index type is unavailable.
                try:
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {self._table_name}_vec_idx "
                        f"ON {self._table_name} USING hnsw (embedding vector_cosine_ops)"
                    )
                except Exception:
                    pass   # pgvector < 0.5 or index already exists
            self._conn.commit()
            self._dim = dim
            self._ready = True

    # ── AbstractVectorBackend ─────────────────────────────────────────────────

    def rebuild(self, entries: list[Entry]) -> None:
        """No-op: pgvector persists its own index on disk."""

    def index(self, entry_id: str, namespace: str, key: str, embedding: list[float]) -> None:
        """Upsert one embedding. Creates the table on the first call."""
        dim = len(embedding)
        self._ensure_table(dim)

        vec = np.array(embedding, dtype=np.float32)
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._table_name}
                        (entry_id, namespace, entry_key, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (entry_id) DO UPDATE SET
                        namespace = EXCLUDED.namespace,
                        entry_key = EXCLUDED.entry_key,
                        embedding = EXCLUDED.embedding
                    """,
                    (entry_id, namespace, key, vec),
                )
            self._conn.commit()

    def remove(self, entry_id: str, namespace: str, key: str) -> None:
        """Delete one embedding by entry_id."""
        if not self._ready:
            return
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table_name} WHERE entry_id = %s",
                    (entry_id,),
                )
            self._conn.commit()

    def remove_namespace(self, namespace: str, recursive: bool = False) -> int:
        """
        Delete all embeddings for a namespace (and children if recursive).
        Returns the number of rows deleted.
        """
        if not self._ready:
            return 0
        with self._lock:
            with self._conn.cursor() as cur:
                if recursive:
                    cur.execute(
                        f"DELETE FROM {self._table_name} "
                        f"WHERE namespace = %s OR namespace LIKE %s",
                        (namespace, namespace + "/%"),
                    )
                else:
                    cur.execute(
                        f"DELETE FROM {self._table_name} WHERE namespace = %s",
                        (namespace,),
                    )
                count = cur.rowcount
            self._conn.commit()
        return count

    def query(
        self,
        query_vec: list[float],
        namespace: str,
        top_k: int,
        min_score: Optional[float],
        recursive: bool,
    ) -> list[tuple[str, str, float]]:
        """
        ANN search using pgvector cosine distance (<=>).

        For exact namespace match, filtering is pushed into SQL.
        For recursive match, over-fetch and filter in Python.

        Returns list of (namespace, key, cosine_score) sorted by score desc.
        """
        if not self._ready:
            return []

        q = np.array(query_vec, dtype=np.float32)
        fetch_n = top_k * _OVER_FETCH

        with self._lock:
            with self._cur() as cur:
                if not recursive:
                    cur.execute(
                        f"""
                        SELECT namespace, entry_key,
                               1 - (embedding <=> %s) AS score
                        FROM {self._table_name}
                        WHERE namespace = %s
                        ORDER BY embedding <=> %s
                        LIMIT %s
                        """,
                        (q, namespace, q, fetch_n),
                    )
                else:
                    # Over-fetch; Python-side namespace prefix filter below.
                    cur.execute(
                        f"""
                        SELECT namespace, entry_key,
                               1 - (embedding <=> %s) AS score
                        FROM {self._table_name}
                        ORDER BY embedding <=> %s
                        LIMIT %s
                        """,
                        (q, q, fetch_n),
                    )
                rows = cur.fetchall()

        results: list[tuple[str, str, float]] = []
        for r in rows:
            ns    = r["namespace"]
            score = max(-1.0, float(r["score"]))
            if recursive and ns != namespace and not ns.startswith(namespace + "/"):
                continue
            if min_score is not None and score < min_score:
                continue
            results.append((ns, r["entry_key"], score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
