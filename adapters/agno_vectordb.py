"""
A2M — AgentToMemory Protocol
Agno adapter: A2MAgnoVectorDb

Concrete implementation of Agno's VectorDb abstract base class.
Maps Agno's Document objects to A2M semantic entries and delegates
all storage to the A2M REST API.

Backend usage per operation:
┌──────────────────────────────┬──────────────────────────────────────┬──────────────────────────────────┐
│ Operation                    │ A2M call                             │ Backend                          │
├──────────────────────────────┼──────────────────────────────────────┼──────────────────────────────────┤
│ insert / upsert              │ write(type="semantic", embedding=...) │ Relational + Vector (if embed)  │
│ content_hash_exists          │ list(tags=["hash:…"])                │ Relational only                  │
│ name_exists                  │ list(tags=["name:…"])                │ Relational only                  │
│ id_exists                    │ list(tags=["docid:…"])               │ Relational only                  │
│ search                       │ query(embedding=embed_fn(query))     │ Relational → Vector ranking      │
│ drop / delete                │ delete_bulk(type="semantic")         │ Relational                       │
│ delete_by_id                 │ list(tags=["docid:…"]) + delete × N │ Relational                       │
│ delete_by_name               │ list(tags=["name:…"]) + delete × N  │ Relational                       │
│ delete_by_content_id         │ list(tags=["cid:…"]) + delete × N   │ Relational                       │
│ delete_by_metadata           │ list + client-side filter + delete   │ Relational                       │
│ exists                       │ list(limit=1)                        │ Relational                       │
└──────────────────────────────┴──────────────────────────────────────┴──────────────────────────────────┘

Storage key per document:  "{content_hash}/{doc_id}"
Tags per document:         ["agno-vector", "hash:{content_hash}", "docid:{doc_id}",
                            "name:{doc.name}",  "cid:{doc.content_id}"]   (name/cid omitted if absent)

Note on embeddings:
  Agno's Document may already carry a pre-computed embedding (doc.embedding).
  If present, it is stored verbatim. If absent and embed_fn is provided, the
  adapter embeds doc.content on the fly, making the entry eligible for
  semantic search via search().

  search() always requires embed_fn — it cannot fall back to text search
  because A2M's /query endpoint requires a caller-provided embedding (spec §5.1).

Install requirement:
    pip install agno

Usage:
    from adapters.agno_vectordb import A2MAgnoVectorDb
    from client import A2MClient
    from agno.agent import Agent
    from agno.knowledge.pdf import PDFKnowledgeBase

    client = A2MClient("http://localhost:8765", namespace="myapp/kb")

    # Without embedding (document storage only, no semantic search):
    vdb = A2MAgnoVectorDb(client=client)

    # With semantic search — caller supplies embed function:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vdb = A2MAgnoVectorDb(
        client=client,
        embed_fn=lambda t: model.encode(t).tolist(),
    )

    kb = PDFKnowledgeBase(path="docs/", vector_db=vdb)
    agent = Agent(knowledge=kb)
"""

from __future__ import annotations

import asyncio
from hashlib import md5
from typing import Any, Callable, Dict, List, Optional

try:
    from agno.knowledge.document import Document
    from agno.vectordb.base import VectorDb
    from agno.vectordb.search import SearchType
except ImportError as exc:
    raise ImportError(
        "Agno is required for A2MAgnoVectorDb. "
        "Install it with: pip install agno"
    ) from exc

from client.client import A2MClient

_TAG = "a2m:knowledge"


def _stable_id(doc: Document) -> str:
    """Derive a stable document ID from whichever field is available."""
    return doc.id or doc.name or md5(doc.content.encode()).hexdigest()


def _entry_to_doc(entry: dict) -> Document:
    """Reconstruct an agno Document from an A2M entry dict.

    Handles both Agno-originated entries (``content`` / ``meta_data``) and
    LangChain-originated entries (``page_content`` / ``metadata``) so that
    cross-framework knowledge sharing works transparently.
    """
    v = entry.get("value", {})
    if not isinstance(v, dict):
        return Document(content=str(v))
    content = v.get("content") or v.get("page_content", "")
    meta = v.get("meta_data") or v.get("metadata") or {}
    return Document(
        content=content,
        id=v.get("doc_id") or v.get("id") or entry.get("key"),
        name=v.get("name"),
        meta_data=meta,
        content_id=v.get("content_id"),
    )


class A2MAgnoVectorDb(VectorDb):
    """
    Agno VectorDb adapter for A2M.

    Implements every abstract method of agno.vectordb.base.VectorDb.
    All async methods delegate to their sync counterparts via asyncio.to_thread,
    so the adapter is safe to use in both sync and async Agno workflows.

    Pass an instance to any Agno KnowledgeBase that accepts a vector_db parameter:
        kb = PDFKnowledgeBase(path="…", vector_db=A2MAgnoVectorDb(client=client))
    """

    def __init__(
        self,
        client: A2MClient,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        name: str = "a2m-vector",
        description: Optional[str] = None,
    ) -> None:
        """
        Args:
            client:      A2MClient scoped to the desired namespace.
            embed_fn:    Optional text → float-vector function.
                         Required for search(); used to auto-embed doc.content
                         when Document.embedding is None.
            name:        Logical name for this VectorDb (used by Agno internals).
            description: Optional description.
        """
        super().__init__(name=name, description=description)
        self.client   = client
        self.embed_fn = embed_fn

    # ── internal helpers ─────────────────────────────────────────────────────

    def _doc_key(self, content_hash: str, doc_id: str) -> str:
        return f"{content_hash}/{doc_id}"

    def _doc_tags(self, content_hash: str, doc: Document, doc_id: str) -> List[str]:
        tags = [_TAG, f"hash:{content_hash}", f"docid:{doc_id}"]
        if doc.name:
            tags.append(f"name:{doc.name}")
        if doc.content_id:
            tags.append(f"cid:{doc.content_id}")
        return tags

    def _write_doc(self, content_hash: str, doc: Document) -> None:
        """
        Persist one Document to A2M.
        Backend: Relational (always) + Vector (if embedding available).
        """
        doc_id    = _stable_id(doc)
        embedding = doc.embedding
        if embedding is None and self.embed_fn:
            embedding = self.embed_fn(doc.content)
        self.client.write(
            key=self._doc_key(content_hash, doc_id),
            type="semantic",
            value={
                "content":      doc.content,
                "name":         doc.name,
                "meta_data":    doc.meta_data or {},
                "content_id":   doc.content_id,
                "content_hash": content_hash,
                "doc_id":       doc_id,
            },
            embedding=embedding,
            meta={
                "source_framework": "agno",
                "tags": self._doc_tags(content_hash, doc, doc_id),
            },
        )

    def _list_tagged(self, *extra_tags: str, limit: int = 500) -> List[dict]:
        """Fetch entries matching _TAG plus any additional tags."""
        tags = [_TAG, *extra_tags]
        resp = self.client.list(type="semantic", tags=tags, limit=limit)
        return resp.get("entries", [])

    def _list_total(self, *extra_tags: str) -> int:
        tags = [_TAG, *extra_tags]
        resp = self.client.list(type="semantic", tags=tags, limit=1)
        return resp.get("total", 0)

    def _delete_entries(self, entries: List[dict]) -> int:
        deleted = 0
        for e in entries:
            try:
                self.client.delete(e["key"])
                deleted += 1
            except Exception:
                pass
        return deleted

    # ── lifecycle ────────────────────────────────────────────────────────────

    def create(self) -> None:
        """No-op: A2M initialises automatically on first write."""

    async def async_create(self) -> None:
        pass

    def exists(self) -> bool:
        """
        True if any agno-vector entries exist in this namespace.
        Backend: Relational.
        """
        return self._list_total() > 0

    async def async_exists(self) -> bool:
        return await asyncio.to_thread(self.exists)

    # ── existence checks ─────────────────────────────────────────────────────

    def content_hash_exists(self, content_hash: str) -> bool:
        """Backend: Relational."""
        return self._list_total(f"hash:{content_hash}") > 0

    def name_exists(self, name: str) -> bool:
        """Backend: Relational."""
        return self._list_total(f"name:{name}") > 0

    async def async_name_exists(self, name: str) -> bool:
        return await asyncio.to_thread(self.name_exists, name)

    def id_exists(self, id: str) -> bool:
        """Backend: Relational."""
        return self._list_total(f"docid:{id}") > 0

    # ── write ────────────────────────────────────────────────────────────────

    def upsert_available(self) -> bool:
        return True

    def insert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Any] = None,
    ) -> None:
        """
        Write documents. A2M writes are always upserts, so insert and upsert
        behave identically. Caller is responsible for checking content_hash_exists
        before calling insert to avoid duplicate ingestion.
        Backend: Relational + Vector (if embed_fn set or doc.embedding present).
        """
        for doc in documents:
            self._write_doc(content_hash, doc)

    async def async_insert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Any] = None,
    ) -> None:
        await asyncio.to_thread(self.insert, content_hash, documents, filters)

    def upsert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Any] = None,
    ) -> None:
        """Backend: Relational + Vector (if embed_fn set or doc.embedding present)."""
        for doc in documents:
            self._write_doc(content_hash, doc)

    async def async_upsert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Any] = None,
    ) -> None:
        await asyncio.to_thread(self.upsert, content_hash, documents, filters)

    # ── search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Any] = None,
    ) -> List[Document]:
        """
        Semantic search over stored documents.
        Backend: Relational (candidate pre-filter) → Vector (cosine ranking).

        Requires embed_fn to be provided at construction.
        """
        if self.embed_fn is None:
            raise ValueError(
                "embed_fn is required for search(). "
                "Provide it at A2MAgnoVectorDb construction."
            )
        results = self.client.query(
            embedding=self.embed_fn(query),
            type="semantic",
            tags=[_TAG],
            top_k=limit,
        )
        return [_entry_to_doc(r["entry"]) for r in results]

    async def async_search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Any] = None,
    ) -> List[Document]:
        return await asyncio.to_thread(self.search, query, limit, filters)

    # ── delete ───────────────────────────────────────────────────────────────

    def drop(self) -> None:
        """Delete all agno-vector entries in this namespace. Backend: Relational."""
        self.client.delete_bulk(type="semantic", tags=[_TAG])

    async def async_drop(self) -> None:
        await asyncio.to_thread(self.drop)

    def delete(self) -> bool:
        """Delete all entries. Returns True if any were deleted. Backend: Relational."""
        count = self.client.delete_bulk(type="semantic", tags=[_TAG])
        return count > 0

    def delete_by_id(self, id: str) -> bool:
        """Backend: Relational — list by docid tag, then delete each entry."""
        entries = self._list_tagged(f"docid:{id}")
        return self._delete_entries(entries) > 0

    def delete_by_name(self, name: str) -> bool:
        """Backend: Relational — list by name tag, then delete each entry."""
        entries = self._list_tagged(f"name:{name}")
        return self._delete_entries(entries) > 0

    def delete_by_content_id(self, content_id: str) -> bool:
        """Backend: Relational — list by cid tag, then delete each entry."""
        entries = self._list_tagged(f"cid:{content_id}")
        return self._delete_entries(entries) > 0

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Delete entries whose stored meta_data matches all given key-value pairs.
        Backend: Relational — fetches all entries and filters client-side
        (A2M v1 has no metadata index).
        """
        entries = self._list_tagged()
        deleted = 0
        for e in entries:
            v = e.get("value", {})
            stored_meta: dict = v.get("meta_data", {}) if isinstance(v, dict) else {}
            if all(stored_meta.get(k) == val for k, val in metadata.items()):
                try:
                    self.client.delete(e["key"])
                    deleted += 1
                except Exception:
                    pass
        return deleted > 0

    # ── metadata ─────────────────────────────────────────────────────────────

    def get_supported_search_types(self) -> List[str]:
        return ["vector"] if self.embed_fn else []

    def optimize(self) -> None:
        """No-op: A2M backends handle their own index optimisation."""

    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> None:
        """
        Merge *metadata* into every document whose content_id matches.
        Backend: Relational (list by cid tag, read-modify-write each entry).
        """
        entries = self._list_tagged(f"cid:{content_id}")
        for e in entries:
            v = e.get("value", {})
            if not isinstance(v, dict):
                continue
            stored = v.get("meta_data") or {}
            stored.update(metadata)
            v["meta_data"] = stored
            # rebuild tags from the (possibly updated) document
            doc = _entry_to_doc(e)
            doc.meta_data = stored
            content_hash = v.get("content_hash", "")
            doc_id = v.get("doc_id") or _stable_id(doc)
            self.client.write(
                key=e["key"],
                type="semantic",
                value=v,
                embedding=e.get("embedding"),
                meta={
                    "source_framework": "agno",
                    "tags": self._doc_tags(content_hash, doc, doc_id),
                },
            )
