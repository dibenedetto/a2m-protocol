"""
Agent2Memory (A2M) Protocol
Abstract backend interfaces.

Every A2M store consists of two pluggable backends:

  AbstractRelationalBackend
      Handles structured operations: CRUD, tag filtering, pagination, TTL.
      Default implementation: SQLiteRelationalBackend (server/backends/sqlite.py)
      Production option:      PostgreSQLRelationalBackend (future)

  AbstractVectorBackend
      Handles embedding storage and ANN search.
      Default implementation: NumpyVectorBackend (in-memory, rebuilt from relational on startup)
      Production option:      LanceVectorBackend (server/backends/lancedb_backend.py)

A2MStore (server/store.py) coordinates both backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..models import Entry, EntryWrite, MemoryType


class AbstractRelationalBackend(ABC):
    """
    Structured storage contract.

    Implementations must be thread-safe (the A2M server uses one instance
    shared across all request handlers).
    """

    @abstractmethod
    def upsert(self, namespace: str, ew: EntryWrite) -> tuple[Entry, bool]:
        """
        Insert or update an entry keyed on (namespace, ew.key).
        On update: preserve id and created_at (spec §5.4).
        Returns (Entry, created: bool).
        """

    @abstractmethod
    def get(self, namespace: str, key: str) -> Optional[Entry]:
        """Fetch a single entry by (namespace, key). Returns None if absent or expired."""

    @abstractmethod
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
        """
        List entries with optional filters.
        Returns (page, total_matching_count).
        recursive=True traverses child namespaces.
        has_embedding=True skips entries without an embedding vector.
        """

    @abstractmethod
    def delete_one(self, namespace: str, key: str) -> bool:
        """Delete a single entry. Returns True if it existed."""

    @abstractmethod
    def delete_many(
        self,
        namespace: str,
        type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
    ) -> int:
        """Bulk-delete entries matching the given filters. Returns count deleted."""

    @abstractmethod
    def expire(self) -> int:
        """Delete entries past their expires_at. Returns count deleted."""

    @abstractmethod
    def all_with_embeddings(self) -> list[Entry]:
        """
        Return every entry that has a non-null embedding, across all namespaces.
        Used by A2MStore on startup to seed in-memory vector backends.
        """


class AbstractVectorBackend(ABC):
    """
    Vector index contract.

    Implementations either maintain their own persistent index (e.g. LanceDB)
    or operate entirely from an in-memory structure (e.g. NumpyVectorBackend).

    query() returns (namespace, key, score) triples; A2MStore hydrates full
    Entry objects from the relational backend using those coordinates.
    """

    def rebuild(self, entries: list[Entry]) -> None:
        """
        Seed the index from a list of entries.
        Called once by A2MStore.__init__ to restore an in-memory index after
        restart.  Persistent backends (LanceDB) can leave this as a no-op.
        """

    @abstractmethod
    def index(self, entry_id: str, namespace: str, key: str, embedding: list[float]) -> None:
        """Upsert one embedding into the index. Called after every relational write."""

    @abstractmethod
    def remove(self, entry_id: str, namespace: str, key: str) -> None:
        """Remove one embedding by entry_id. Called after a relational delete."""

    @abstractmethod
    def remove_namespace(self, namespace: str, recursive: bool = False) -> int:
        """
        Remove all embeddings for a namespace (and optionally its children).
        Returns the number of embeddings removed.
        """

    @abstractmethod
    def query(
        self,
        query_vec: list[float],
        namespace: str,
        top_k: int,
        min_score: Optional[float],
        recursive: bool,
    ) -> list[tuple[str, str, float]]:
        """
        ANN search.
        Returns a list of (namespace, key, cosine_score) tuples,
        sorted by score descending, length <= top_k.
        """
