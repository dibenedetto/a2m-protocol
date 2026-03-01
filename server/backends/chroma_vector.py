"""
A2M — AgentToMemory Protocol
ChromaDB vector backend: ChromaVectorBackend.

Implements AbstractVectorBackend using ChromaDB for persistent or in-memory
ANN search with cosine similarity.

Key properties:
  - Persistent (PersistentClient) or ephemeral (EphemeralClient).
  - Cosine similarity via hnsw:space="cosine".
  - Upsert via collection.upsert() keyed on entry_id.
  - Namespace + recursive filtering applied in Python after retrieval
    (ChromaDB where-clauses support exact equality but not prefix matching).

ChromaDB distance convention (hnsw:space="cosine"):
  distance = 1 - cosine_similarity   → range [0, 2]
  score    = 1.0 - distance          → range [-1, 1]  (same as other backends)

Python compatibility:
    ChromaDB relies on pydantic v1 internals which are not compatible with
    Python 3.14+.  Use LanceVectorBackend or NumpyVectorBackend on Python 3.14.
    ChromaDB works on Python 3.9 - 3.12.

Install requirement:
    pip install chromadb

Usage:
    from server.backends.chroma_backend import ChromaVectorBackend
    from server.main import create_app

    # Persistent (survives restart):
    app = create_app(vector=ChromaVectorBackend(path="./chroma_data"))

    # Ephemeral in-memory (rebuilt from SQLite on startup):
    app = create_app(vector=ChromaVectorBackend())
"""

from __future__ import annotations

from typing import Optional

try:
    import chromadb
except Exception as exc:
    # ImportError when chromadb is not installed; other errors (e.g. pydantic
    # ConfigError) when chromadb is installed but incompatible with the running
    # Python version (chromadb requires Python ≤ 3.12 due to pydantic v1 use).
    raise ImportError(
        "chromadb is required for ChromaVectorBackend and must be compatible "
        "with the running Python version (Python 3.9 - 3.12 only). "
        "Install it with: pip install chromadb"
    ) from exc

from .base import AbstractVectorBackend, Entry

# Extra candidates fetched before Python-side namespace prefix filtering.
_OVER_FETCH = 10


class ChromaVectorBackend(AbstractVectorBackend):
    """
    ChromaDB-backed ANN vector index.

    A single ChromaDB collection stores all A2M embeddings.  Namespace and
    entry_key are stored as metadata fields so they can be retrieved alongside
    the cosine scores.

    For exact-namespace queries the ChromaDB where-clause filters at query
    time, avoiding a Python scan.  For recursive queries the backend
    over-fetches and filters in Python (ChromaDB has no prefix/LIKE operator).

    Thread safety: ChromaDB's Python client is thread-safe for reads; writes
    are protected by ChromaDB's own internal locking.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        collection_name: str = "a2m_vectors",
    ) -> None:
        """
        Args:
            path:            Directory for persistent storage.
                             Pass None (default) for an ephemeral in-memory
                             instance that is rebuilt from the relational
                             backend on every server startup.
            collection_name: Name of the ChromaDB collection.
        """
        self._path       = path
        self._persistent = path is not None

        if self._persistent:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.EphemeralClient()

        # hnsw:space=cosine → distance in [0, 2]; score = 1.0 - distance
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── AbstractVectorBackend ─────────────────────────────────────────────────

    def rebuild(self, entries: list[Entry]) -> None:
        """
        Re-populate the collection from Entry objects.

        For persistent instances this is a no-op (ChromaDB already holds the
        data on disk).  For ephemeral instances it re-indexes all embeddings
        that the relational backend carries so the collection is ready to serve
        queries immediately after startup.
        """
        if self._persistent:
            return

        ids, embeddings, metadatas = [], [], []
        for e in entries:
            if e.embedding:
                ids.append(e.id)
                embeddings.append(e.embedding)
                metadatas.append({"namespace": e.namespace, "entry_key": e.key})

        if ids:
            self._collection.upsert(
                ids=ids, embeddings=embeddings, metadatas=metadatas
            )

    def index(self, entry_id: str, namespace: str, key: str, embedding: list[float]) -> None:
        """Upsert one embedding into the collection."""
        self._collection.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            metadatas=[{"namespace": namespace, "entry_key": key}],
        )

    def remove(self, entry_id: str, namespace: str, key: str) -> None:
        """Delete one embedding by entry_id."""
        self._collection.delete(ids=[entry_id])

    def remove_namespace(self, namespace: str, recursive: bool = False) -> int:
        """
        Delete all embeddings for a namespace (and children if recursive).
        Returns the number of embeddings removed.
        """
        if self._collection.count() == 0:
            return 0

        if not recursive:
            # Use ChromaDB where-clause for exact namespace match.
            results = self._collection.get(
                where={"namespace": {"$eq": namespace}},
                include=[],   # ids only
            )
            ids_to_delete = results["ids"]
        else:
            # Fetch all; filter by prefix in Python.
            results = self._collection.get(include=["metadatas"])
            ids_to_delete = [
                r_id
                for r_id, meta in zip(results["ids"], results["metadatas"])
                if meta["namespace"] == namespace
                or meta["namespace"].startswith(namespace + "/")
            ]

        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def query(
        self,
        query_vec: list[float],
        namespace: str,
        top_k: int,
        min_score: Optional[float],
        recursive: bool,
    ) -> list[tuple[str, str, float]]:
        """
        ANN search using ChromaDB cosine distance.

        For exact namespace matches the where-clause is pushed into ChromaDB.
        For recursive matches all candidates are fetched and namespace-filtered
        in Python.

        Returns list of (namespace, key, cosine_score) sorted by score desc.
        """
        total = self._collection.count()
        if total == 0:
            return []

        fetch_n = min(top_k * _OVER_FETCH, total)

        try:
            if not recursive:
                raw = self._collection.query(
                    query_embeddings=[query_vec],
                    n_results=fetch_n,
                    where={"namespace": {"$eq": namespace}},
                    include=["metadatas", "distances"],
                )
            else:
                raw = self._collection.query(
                    query_embeddings=[query_vec],
                    n_results=fetch_n,
                    include=["metadatas", "distances"],
                )
        except Exception:
            return []

        # ChromaDB returns lists-of-lists (one per query vector).
        ids       = raw["ids"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        results: list[tuple[str, str, float]] = []
        for r_id, meta, dist in zip(ids, metadatas, distances):
            ns = meta["namespace"]
            if recursive and ns != namespace and not ns.startswith(namespace + "/"):
                continue
            # hnsw:space=cosine: distance = 1 - cosine_similarity → range [0, 2]
            score = max(-1.0, 1.0 - float(dist))
            if min_score is not None and score < min_score:
                continue
            results.append((ns, meta["entry_key"], score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
