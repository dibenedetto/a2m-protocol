"""
Agent2Memory (A2M) Protocol
LanceDB vector backend: LanceVectorBackend

Implements AbstractVectorBackend using LanceDB for persistent ANN search.

Key properties:
  - Persistent: index survives server restarts without re-indexing.
  - Cosine similarity metric (metric="cosine").
  - Upsert via merge_insert keyed on entry_id.
  - Namespace filtering is applied in Python after the ANN search
    (over-fetch by factor OVER_FETCH to compensate for filtered-out rows).

Storage schema (LanceDB table "a2m_vectors" by default):
    entry_id  : string   — A2M Entry UUID (primary upsert key)
    namespace : string   — slash-delimited namespace
    entry_key : string   — A2M entry key within the namespace
    vector    : [float32 x dim]  — embedding (fixed dim per table)

Distance metric: cosine.
  _distance = 0   → identical vectors   (score = 1.0)
  _distance = 1   → orthogonal vectors  (score = 0.0)
  _distance = 2   → opposite vectors    (score = -1.0)
  score = 1.0 - _distance

Install requirement:
    pip install lancedb pyarrow

Usage:
    from server.backends.lancedb_backend import LanceVectorBackend
    from server.backends.sqlite import SQLiteRelationalBackend
    from server.store import A2MStore

    store = A2MStore(
        relational=SQLiteRelationalBackend("a2m.db"),
        vector=LanceVectorBackend(uri="./lancedb_data"),
    )
"""

from __future__ import annotations

import threading
from typing import Optional

from .base import AbstractVectorBackend, Entry

try:
    import lancedb
    import pyarrow as pa
except ImportError as exc:
    raise ImportError(
        "lancedb and pyarrow are required for LanceVectorBackend. "
        "Install them with: pip install lancedb pyarrow"
    ) from exc

# How many extra candidates to fetch from LanceDB before namespace filtering.
# Increase if many embeddings share the same LanceDB index across namespaces.
_OVER_FETCH = 10


class LanceVectorBackend(AbstractVectorBackend):
    """
    LanceDB-backed ANN vector index.

    All writes are upserts keyed on entry_id.
    Reads use cosine similarity search followed by Python-side namespace filtering.

    Thread-safe: lancedb uses file locking internally; the Python-side table
    handle is protected by a threading.Lock for the lazy-init sequence only.
    """

    def __init__(
        self,
        uri: str,
        table_name: str = "a2m_vectors",
    ) -> None:
        """
        Args:
            uri:        Path to the LanceDB data directory (created if absent).
                        LanceDB does not support in-memory mode on all platforms;
                        use a real filesystem path.
            table_name: Name of the LanceDB table used to store A2M vectors.
                        Override this to share a LanceDB directory across
                        multiple A2M instances.
        """
        self._uri        = uri
        self._table_name = table_name
        self._db         = lancedb.connect(uri)
        self._table      = None          # lazy: created/opened on first index()
        self._dim: Optional[int] = None  # set on first index(); fixed thereafter
        self._lock       = threading.Lock()

    # ── private ───────────────────────────────────────────────────────────────

    def _get_table(self, dim: Optional[int] = None):
        """Return the LanceDB table, creating it if necessary."""
        if self._table is not None:
            return self._table

        with self._lock:
            if self._table is not None:        # double-check after acquiring lock
                return self._table

            existing_names = self._db.table_names()
            if self._table_name in existing_names:
                self._table = self._db.open_table(self._table_name)
                # Infer dimension from existing data (first row)
                try:
                    row = self._table.to_arrow().slice(0, 1).to_pylist()
                    if row:
                        self._dim = len(row[0]["vector"])
                except Exception:
                    pass
            elif dim is not None:
                schema = pa.schema([
                    pa.field("entry_id",  pa.string()),
                    pa.field("namespace", pa.string()),
                    pa.field("entry_key", pa.string()),
                    pa.field("vector",    pa.list_(pa.float32(), dim)),
                ])
                self._table = self._db.create_table(self._table_name, schema=schema)
                self._dim = dim

        return self._table

    # ── AbstractVectorBackend ─────────────────────────────────────────────────

    def rebuild(self, entries: list[Entry]) -> None:
        """
        No-op: LanceDB persists its own index.
        All embeddings written via index() are already on disk.
        """

    def index(self, entry_id: str, namespace: str, key: str, embedding: list[float]) -> None:
        """
        Upsert one embedding into LanceDB.
        The table is created on the first call, using the embedding's dimension.
        """
        dim   = len(embedding)
        table = self._get_table(dim)
        if table is None:
            return

        table.merge_insert("entry_id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute([{
                "entry_id":  entry_id,
                "namespace": namespace,
                "entry_key": key,
                "vector":    [float(x) for x in embedding],
            }])

    def remove(self, entry_id: str, namespace: str, key: str) -> None:
        """Delete a single embedding by entry_id."""
        table = self._get_table()
        if table is None:
            return
        # entry_id is a UUID — safe to interpolate directly
        table.delete(f"entry_id = '{entry_id}'")

    def remove_namespace(self, namespace: str, recursive: bool = False) -> int:
        """
        Delete all embeddings for a namespace (and children if recursive).
        Returns an approximate count (exact count requires a pre-scan).
        """
        table = self._get_table()
        if table is None:
            return 0

        # Count matching rows before delete
        try:
            all_rows = table.to_arrow().to_pylist()
        except Exception:
            all_rows = []

        if recursive:
            matching = [
                r for r in all_rows
                if r["namespace"] == namespace
                or r["namespace"].startswith(namespace + "/")
            ]
        else:
            matching = [r for r in all_rows if r["namespace"] == namespace]

        count = len(matching)
        if count == 0:
            return 0

        # Build delete condition using the matched entry_ids (safe: UUIDs)
        ids_sql = ", ".join(f"'{r['entry_id']}'" for r in matching)
        table.delete(f"entry_id IN ({ids_sql})")
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
        ANN search using cosine metric.

        Over-fetches by _OVER_FETCH * top_k to compensate for entries
        that are filtered out by namespace after retrieval.

        Returns list of (namespace, key, cosine_score) sorted by score desc.
        """
        table = self._get_table()
        if table is None:
            return []

        fetch_n = top_k * _OVER_FETCH
        try:
            raw = (
                table.search([float(x) for x in query_vec])
                .metric("cosine")
                .limit(fetch_n)
                .select(["entry_id", "namespace", "entry_key", "_distance"])
                .to_list()
            )
        except Exception:
            return []

        results: list[tuple[str, str, float]] = []
        for r in raw:
            ns = r["namespace"]
            if ns != namespace and not (recursive and ns.startswith(namespace + "/")):
                continue
            # _distance = cosine_distance = 1 - cosine_similarity
            score = max(-1.0, 1.0 - float(r.get("_distance", 1.0)))
            if min_score is not None and score < min_score:
                continue
            results.append((ns, r["entry_key"], score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
