"""
A2M — AgentToMemory Protocol
Agno adapter: A2MAgnoMemoryDb

Duck-typed implementation of Agno's MemoryManager db interface.
Maps Agno's UserMemory objects to A2M semantic entries and delegates
all storage to the A2M REST API.

Backend usage per operation:
┌──────────────────────────────┬──────────────────────────────────┬────────────────────────────────┐
│ Operation                    │ A2M call                         │ Backend                        │
├──────────────────────────────┼──────────────────────────────────┼────────────────────────────────┤
│ upsert_user_memory           │ write(type="semantic")           │ Relational + Vector (if embed) │
│ upsert_memories (bulk)       │ write × N                        │ Relational + Vector (if embed) │
│ get_user_memories            │ list(type="semantic", tags=[uid])│ Relational only                │
│ get_user_memory              │ list + filter by memory_id       │ Relational only                │
│ delete_user_memory           │ delete(key=memory_id)            │ Relational                     │
│ delete_user_memories         │ delete × N                       │ Relational                     │
│ clear_memories               │ delete_bulk(type="semantic")     │ Relational                     │
│ get_all_memory_topics        │ list + aggregate topics          │ Relational                     │
│ get_user_memory_stats        │ list + aggregate                 │ Relational                     │
└──────────────────────────────┴──────────────────────────────────┴────────────────────────────────┘

Note on embeddings:
  Agno's UserMemory does not natively carry an embedding vector. To enable
  vector search via A2M, supply an `embed_fn` at construction. When present,
  the adapter embeds `memory.memory` before writing, making all memories
  eligible for semantic query via search_user_memories().

Install requirement:
    pip install agno

Usage:
    from adapters.agno import A2MAgnoMemoryDb
    from client import A2MClient
    from agno.memory import MemoryManager

    client = A2MClient("http://localhost:8765", namespace="myapp/agent-0")

    # Without semantic search:
    db = A2MAgnoMemoryDb(client=client)

    # With semantic search — caller supplies embed function:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    db = A2MAgnoMemoryDb(client=client, embed_fn=lambda t: model.encode(t).tolist())

    manager = MemoryManager(db=db)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from agno.memory import UserMemory
except ImportError as exc:
    raise ImportError(
        "Agno is required for A2MAgnoMemoryDb. "
        "Install it with: pip install agno"
    ) from exc

from client.client import A2MClient


def _to_a2m_value(memory: UserMemory) -> dict:
    """Serialize a UserMemory to a JSON-safe dict for A2M storage."""
    return {
        "memory":    memory.memory,
        "memory_id": memory.memory_id,
        "user_id":   memory.user_id,
        "agent_id":  memory.agent_id,
        "team_id":   memory.team_id,
        "topics":    memory.topics or [],
        "input":     memory.input,
        "feedback":  memory.feedback,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
    }


def _from_a2m_entry(entry: dict) -> UserMemory:
    """Reconstruct a UserMemory from an A2M entry dict."""
    v = entry.get("value", {})
    if isinstance(v, str):
        # value stored as plain string (should not happen, but handle gracefully)
        return UserMemory(memory=v, memory_id=entry.get("key"))
    return UserMemory(
        memory=v.get("memory", ""),
        memory_id=v.get("memory_id") or entry.get("key"),
        user_id=v.get("user_id"),
        agent_id=v.get("agent_id"),
        team_id=v.get("team_id"),
        topics=v.get("topics") or [],
        input=v.get("input"),
        feedback=v.get("feedback"),
        created_at=v.get("created_at"),
        updated_at=v.get("updated_at"),
    )


class A2MAgnoMemoryDb:
    """
    Agno MemoryManager db adapter for A2M.

    Duck-typed to implement the methods Agno's MemoryManager calls on self.db.
    Does NOT inherit from agno.db.base.BaseDb (which has ~40 abstract methods
    covering sessions, evals, traces, etc. — all out of scope for a memory adapter).

    Pass an instance of this class as the `db` argument to MemoryManager:
        manager = MemoryManager(db=A2MAgnoMemoryDb(client=client))
    """

    def __init__(
        self,
        client:   A2MClient,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
    ) -> None:
        """
        Args:
            client:   A2MClient scoped to the desired namespace.
            embed_fn: Optional function to embed memory text → float vector.
                      Required for search_user_memories() to use Vector backend.
                      When None, search falls back to Relational text pre-filter.
        """
        self.client   = client
        self.embed_fn = embed_fn

    # ── internal helpers ─────────────────────────────────────────────────────

    def _user_tags(self, user_id: Optional[str]) -> list[str]:
        tags = ["agno-memory"]
        if user_id:
            tags.append(f"user:{user_id}")
        return tags

    def _write_memory(self, memory: UserMemory) -> None:
        """
        Persist one UserMemory.
        Backend: Relational (always) + Vector (if embed_fn provided).
        """
        mid = memory.memory_id or memory.memory  # key fallback
        embedding = self.embed_fn(memory.memory) if self.embed_fn else None
        tags = self._user_tags(memory.user_id)
        if memory.topics:
            tags.extend(memory.topics)
        self.client.write(
            key=mid,
            type="semantic",
            value=_to_a2m_value(memory),
            embedding=embedding,
            meta={
                "source_framework": "agno",
                "tags": tags,
            },
        )

    # ── MemoryManager interface ──────────────────────────────────────────────

    def upsert_user_memory(
        self,
        memory:      UserMemory,
        deserialize: Optional[bool] = True,
    ) -> Optional[UserMemory]:
        """
        Upsert a single memory.
        Backend: Relational + Vector (if embed_fn set).
        """
        self._write_memory(memory)
        return memory if deserialize else None

    def upsert_memories(
        self,
        memories:             List[UserMemory],
        deserialize:          Optional[bool] = True,
        preserve_updated_at:  bool           = False,
    ) -> List[UserMemory]:
        """
        Bulk upsert multiple memories.
        Backend: Relational + Vector per entry (if embed_fn set).
        """
        for m in memories:
            self._write_memory(m)
        return memories if deserialize else []

    def get_user_memories(
        self,
        user_id:     Optional[str]  = None,
        agent_id:    Optional[str]  = None,
        team_id:     Optional[str]  = None,
        topics:      Optional[list] = None,
        search_content: Optional[str] = None,
        limit:       Optional[int]  = None,
        page:        Optional[int]  = None,
        sort_by:     Optional[str]  = None,
        sort_order:  Optional[str]  = None,
        deserialize: Optional[bool] = True,
    ) -> List[UserMemory]:
        """
        List memories with optional user / topic filters.
        Backend: Relational.
        """
        tags = self._user_tags(user_id)
        offset = ((page or 1) - 1) * (limit or 100) if page else 0
        resp = self.client.list(
            type="semantic",
            tags=tags,
            limit=limit or 100,
            offset=offset,
        )
        memories = [_from_a2m_entry(e) for e in resp["entries"]]

        # Client-side filters for agent_id / team_id / topics (not stored as A2M tags)
        if agent_id:
            memories = [m for m in memories if m.agent_id == agent_id]
        if team_id:
            memories = [m for m in memories if m.team_id == team_id]
        if topics:
            topic_set = set(topics)
            memories = [m for m in memories if topic_set & set(m.topics or [])]

        return memories

    def get_user_memory(
        self,
        memory_id:   str,
        deserialize: Optional[bool] = True,
        user_id:     Optional[str]  = None,
    ) -> Optional[UserMemory]:
        """
        Read a single memory by memory_id.
        Backend: Relational.
        """
        entry = self.client.read(memory_id)
        if entry is None:
            return None
        return _from_a2m_entry(entry)

    def delete_user_memory(
        self,
        memory_id: str,
        user_id:   Optional[str] = None,
    ) -> None:
        """
        Delete a single memory.
        Backend: Relational.
        """
        try:
            self.client.delete(memory_id)
        except Exception:
            pass  # Treat not-found as a no-op (Agno's contract)

    def delete_user_memories(
        self,
        memory_ids: List[str],
        user_id:    Optional[str] = None,
    ) -> None:
        """
        Delete multiple memories by ID.
        Backend: Relational.
        """
        for mid in memory_ids:
            self.delete_user_memory(mid, user_id=user_id)

    def clear_memories(self) -> None:
        """
        Delete all memories in this namespace.
        Backend: Relational.
        """
        self.client.delete_bulk(type="semantic", tags=["agno-memory"])

    def get_all_memory_topics(self, user_id: Optional[str] = None) -> List[str]:
        """
        Aggregate all unique topics across stored memories.
        Backend: Relational (fetches all pages, aggregates client-side).
        """
        topics: set[str] = set()
        page = 1
        batch_size = 500
        while True:
            batch = self.get_user_memories(user_id=user_id, limit=batch_size, page=page)
            for m in batch:
                topics.update(m.topics or [])
            if len(batch) < batch_size:
                break
            page += 1
        return sorted(topics)

    def get_user_memory_stats(
        self,
        limit:   Optional[int] = None,
        page:    Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Return memory stats: list of {memory_id, memory, topics} + total count.
        Backend: Relational.
        """
        memories = self.get_user_memories(user_id=user_id, limit=limit or 100, page=page)
        stats = [
            {
                "memory_id": m.memory_id,
                "memory":    m.memory,
                "topics":    m.topics or [],
                "user_id":   m.user_id,
            }
            for m in memories
        ]
        return stats, len(stats)

    def search_user_memories(
        self,
        embedding: list[float],
        user_id:   Optional[str] = None,
        limit:     int           = 5,
    ) -> List[UserMemory]:
        """
        Semantic search over stored memories.
        Backend:
          1. Relational — pre-filter candidates by user_id tag + has embedding.
          2. Vector     — cosine ranking to return top `limit` results.

        Requires embed_fn to have been set at construction to produce useful results.
        If embed_fn was not set, memories were stored without embeddings and this
        method will return an empty list.
        """
        tags = self._user_tags(user_id)
        results = self.client.query(
            embedding=embedding,
            type="semantic",
            tags=tags,
            top_k=limit,
        )
        return [_from_a2m_entry(r["entry"]) for r in results]
