"""
A2M — AgentToMemory Protocol
LangChain adapter: A2MLangChainVectorStore

Implements LangChain's VectorStore abstract base class (langchain-core ≥ 0.1).
Documents are stored as A2M semantic entries. Embeddings are produced by the
caller-supplied Embeddings object (langchain_core.embeddings.Embeddings), keeping
A2M model-agnostic per spec §5.1.

Backend usage per operation:
┌────────────────────────────────┬───────────────────────────────────────┬───────────────────────────────┐
│ Operation                      │ A2M call                              │ Backend                       │
├────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ add_documents                  │ write(type="semantic", embedding=...) │ Relational + Vector           │
│ similarity_search              │ query(embedding=embed(query))         │ Relational → Vector ranking   │
│ similarity_search_with_score   │ query(embedding=embed(query))         │ Relational → Vector ranking   │
│ get_by_ids                     │ read(key) per id                      │ Relational                    │
│ delete(ids=[…])                │ delete(key) per id                    │ Relational                    │
│ delete(ids=None)               │ delete_bulk(type="semantic")          │ Relational                    │
│ mmr_search                     │ query(top_k=fetch_k) + MMR re-rank   │ Relational → Vector → client  │
│ mmr_search_by_vector           │ query(top_k=fetch_k) + MMR re-rank   │ Relational → Vector → client  │
└────────────────────────────────┴───────────────────────────────────────┴───────────────────────────────┘

Storage key per document: the document id (caller-supplied or auto-generated UUID4).
Tags per document: ["langchain-vs", "{collection_tag}", "docid:{id}"]

Install requirement:
    pip install langchain-core

Usage:
    from adapters.langchain_vectorstore import A2MLangChainVectorStore
    from client import A2MClient
    from langchain_openai import OpenAIEmbeddings

    client = A2MClient("http://localhost:8765", namespace="myapp/kb")
    embeddings = OpenAIEmbeddings()

    # Build from existing documents:
    from langchain_core.documents import Document
    docs = [Document(page_content="A2M is a shared memory protocol", metadata={"src": "spec"})]
    store = A2MLangChainVectorStore.from_documents(docs, embedding=embeddings, client=client)

    # Or build from raw texts:
    store = A2MLangChainVectorStore.from_texts(
        ["A2M is …", "Embeddings are caller-owned …"],
        embedding=embeddings,
        client=client,
    )

    # Similarity search:
    results = store.similarity_search("shared memory", k=3)

    # Use as a retriever:
    retriever = store.as_retriever(search_kwargs={"k": 5})
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Callable, Iterable, List, Optional, Sequence

try:
    from langchain_core.documents import Document as LCDoc
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as exc:
    raise ImportError(
        "langchain-core is required for A2MLangChainVectorStore. "
        "Install it with: pip install langchain-core"
    ) from exc

from client.client import A2MClient

_BASE_TAG = "a2m:knowledge"


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _mmr_select(
    query_embedding: List[float],
    doc_embeddings: List[List[float]],
    k: int,
    lambda_mult: float,
) -> List[int]:
    """Return indices selected by Maximal Marginal Relevance."""
    if not doc_embeddings:
        return []
    scores = [_cosine_sim(query_embedding, e) for e in doc_embeddings]
    selected: List[int] = []
    remaining = list(range(len(doc_embeddings)))
    for _ in range(min(k, len(doc_embeddings))):
        best_idx = -1
        best_score = -float("inf")
        for i in remaining:
            relevance = scores[i]
            redundancy = max(
                (_cosine_sim(doc_embeddings[i], doc_embeddings[s]) for s in selected),
                default=0.0,
            )
            mmr = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def _new_id() -> str:
    return str(uuid.uuid4())


def _entry_to_doc(entry: dict) -> LCDoc:
    """Convert an A2M entry to a LangChain Document.

    Handles both LangChain-originated entries (``page_content`` / ``metadata``)
    and Agno-originated entries (``content`` / ``meta_data``) so that
    cross-framework knowledge sharing works transparently.
    """
    v = entry.get("value", {})
    if not isinstance(v, dict):
        return LCDoc(page_content=str(v), id=entry.get("key"))
    text = v.get("page_content") or v.get("content", "")
    meta = v.get("metadata") or v.get("meta_data") or {}
    return LCDoc(
        page_content=text,
        metadata=meta,
        id=v.get("id") or v.get("doc_id") or entry.get("key"),
    )


class A2MLangChainVectorStore(VectorStore):
    """
    LangChain VectorStore adapter for A2M.

    Stores documents as A2M semantic entries. All embeddings are produced
    by the provided Embeddings object — A2M never generates embeddings.

    Use as_retriever() to plug into any LangChain retrieval chain:
        retriever = A2MLangChainVectorStore(client, embeddings).as_retriever()
    """

    def __init__(
        self,
        client: A2MClient,
        embeddings: Embeddings,
        collection_tag: str = "default",
    ) -> None:
        """
        Args:
            client:         A2MClient scoped to the desired namespace.
            embeddings:     LangChain Embeddings implementation (e.g. OpenAIEmbeddings).
                            Used for both indexing (embed_documents) and querying (embed_query).
            collection_tag: Logical name for this collection within the namespace.
                            Allows multiple distinct stores in the same A2M namespace.
        """
        self.client         = client
        self._embeddings    = embeddings
        self.collection_tag = collection_tag

    # ── VectorStore property ─────────────────────────────────────────────────

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    # ── write ────────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[LCDoc],
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Embed and store documents.
        Backend: Relational (always) + Vector index.
        Returns the list of stored document IDs.
        """
        texts   = [doc.page_content for doc in documents]
        vectors = self._embeddings.embed_documents(texts)

        doc_ids: List[str] = []
        for i, (doc, vec) in enumerate(zip(documents, vectors)):
            doc_id = (ids[i] if ids and i < len(ids) else None) or doc.id or _new_id()
            self.client.write(
                key=doc_id,
                type="semantic",
                value={
                    "page_content": doc.page_content,
                    "metadata":     doc.metadata or {},
                    "id":           doc_id,
                },
                embedding=vec,
                meta={
                    "source_framework": "langchain",
                    "tags": [_BASE_TAG, self.collection_tag, f"docid:{doc_id}"],
                },
            )
            doc_ids.append(doc_id)
        return doc_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Convenience wrapper — converts raw strings to Documents and calls add_documents.
        Backend: Relational + Vector.
        """
        texts_list  = list(texts)
        metadatas_  = metadatas or [{}] * len(texts_list)
        docs = [
            LCDoc(page_content=t, metadata=m)
            for t, m in zip(texts_list, metadatas_)
        ]
        return self.add_documents(docs, ids=ids)

    # ── search ───────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[LCDoc]:
        """
        Return the k most similar documents to the query.
        Backend: Relational (candidate pre-filter) → Vector (cosine ranking).
        """
        pairs = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in pairs]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[tuple[LCDoc, float]]:
        """
        Return (document, cosine_score) pairs, sorted by score descending.
        Backend: Relational → Vector.
        """
        vec = self._embeddings.embed_query(query)
        raw = self.client.query(
            embedding=vec,
            type="semantic",
            tags=[_BASE_TAG],
            top_k=k,
        )
        return [(_entry_to_doc(r["entry"]), float(r["score"])) for r in raw]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[LCDoc]:
        """
        Search using a pre-computed query embedding.
        Backend: Relational → Vector.
        """
        raw = self.client.query(
            embedding=embedding,
            type="semantic",
            tags=[_BASE_TAG],
            top_k=k,
        )
        return [_entry_to_doc(r["entry"]) for r in raw]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        A2M's /query endpoint returns cosine similarity scores directly
        (range [-1, 1]).  Normalise to [0, 1] for LangChain's
        similarity_search_with_relevance_scores().
        """
        return self._cosine_relevance_score_fn

    # ── MMR search ────────────────────────────────────────────────────────────

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[LCDoc]:
        """
        Return documents selected by Maximal Marginal Relevance.
        Balances relevance to the query with diversity among results.
        Backend: Relational → Vector (over-fetch), then client-side MMR re-rank.
        """
        vec = self._embeddings.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            vec, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[LCDoc]:
        """
        MMR search using a pre-computed query embedding.
        Over-fetches fetch_k candidates from A2M, then re-ranks client-side.
        Backend: Relational → Vector (fetch_k candidates), then client-side MMR.
        """
        raw = self.client.query(
            embedding=embedding,
            type="semantic",
            tags=[_BASE_TAG],
            top_k=fetch_k,
        )
        if not raw:
            return []
        docs = [_entry_to_doc(r["entry"]) for r in raw]
        doc_embeddings = [r["entry"].get("embedding", []) for r in raw]
        selected = _mmr_select(embedding, doc_embeddings, k, lambda_mult)
        return [docs[i] for i in selected]

    # ── read ─────────────────────────────────────────────────────────────────

    def get_by_ids(self, ids: Sequence[str], /) -> List[LCDoc]:
        """
        Retrieve documents by their IDs.
        Backend: Relational — one GET per id, missing ids silently omitted.
        """
        docs: List[LCDoc] = []
        for id_ in ids:
            entry = self.client.read(id_)
            if entry is not None:
                docs.append(_entry_to_doc(entry))
        return docs

    # ── delete ───────────────────────────────────────────────────────────────

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """
        Delete documents by ID, or the entire collection when ids=None.
        Backend: Relational.
        """
        if ids is None:
            count = self.client.delete_bulk(
                type="semantic",
                tags=[_BASE_TAG],
            )
            return count > 0
        for id_ in ids:
            try:
                self.client.delete(id_)
            except Exception:
                pass
        return True

    # ── class-method factory ──────────────────────────────────────────────────

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "A2MLangChainVectorStore":
        """
        Create a store from raw text strings and immediately index them.

        Required kwargs:
            client (A2MClient): scoped client for storage.

        Optional kwargs:
            collection_tag (str): logical collection name (default "default").
        """
        client         = kwargs.pop("client")
        collection_tag = kwargs.pop("collection_tag", "default")
        instance = cls(client=client, embeddings=embedding, collection_tag=collection_tag)
        instance.add_texts(texts, metadatas=metadatas, ids=ids)
        return instance
