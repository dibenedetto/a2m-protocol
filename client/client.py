"""
A2M — AgentToMemory Protocol
Python client library.

Wraps the A2M REST API with a clean, namespace-scoped interface.
Uses httpx (sync) with no other runtime dependencies.

Quick start:
    from client import A2MClient

    client = A2MClient("http://localhost:8765", namespace="myapp/wf-1")
    client.write("user/goal", type="semantic", value="...", embedding=[...])
    results = client.query(embedding=[...], top_k=5)

Scoped clients:
    session = client.scoped("sess-abc")        # myapp/wf-1/sess-abc
    agent   = session.scoped("agent-0")        # myapp/wf-1/sess-abc/agent-0
    agent.write("scratchpad", type="working", value={...})
"""

from __future__ import annotations

from typing import Any, Optional

import httpx


class A2MError(Exception):
    """Raised when the A2M server returns an error response."""
    def __init__(self, code: str, message: str, status: int):
        super().__init__(f"[{code}] {message} (HTTP {status})")
        self.code    = code
        self.message = message
        self.status  = status


class A2MClient:
    """
    Synchronous A2M client. Namespace is bound at construction.

    Args:
        base_url:  Server base URL, e.g. "http://localhost:8765"
        namespace: Slash-delimited scope, e.g. "myapp/wf-42/sess-abc/agent-0"
        timeout:   Request timeout in seconds (default 10).
    """

    def __init__(
        self,
        base_url: str,
        namespace: str,
        timeout: float = 10.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self.namespace = namespace.strip("/")
        self._http = httpx.Client(timeout=timeout)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _url(self, *parts: str) -> str:
        return "/".join([self._base, "a2m", "v1", self.namespace, *parts])

    def _raise(self, r: httpx.Response) -> None:
        if r.is_error:
            try:
                body = r.json()
                raise A2MError(body.get("code", "UNKNOWN"), body.get("message", r.text), r.status_code)
            except (ValueError, KeyError):
                raise A2MError("UNKNOWN", r.text, r.status_code)

    # ── write ────────────────────────────────────────────────────────────────

    def write(
        self,
        key:       str,
        type:      str,
        value:     Any,
        embedding: Optional[list[float]] = None,
        meta:      Optional[dict]        = None,
    ) -> dict:
        """
        Write or upsert a memory entry.
        Backend: Relational (persist) + Vector index (if embedding provided).

        Returns the full Entry dict. Returns HTTP 201 on create, 200 on update.
        """
        body: dict = {"key": key, "type": type, "value": value}
        if embedding is not None:
            body["embedding"] = embedding
        if meta:
            body["meta"] = meta
        r = self._http.post(self._url("entries"), json=body)
        self._raise(r)
        return r.json()

    # ── read ─────────────────────────────────────────────────────────────────

    def read(self, key: str) -> Optional[dict]:
        """
        Read a single entry by key. Returns None if not found.
        Backend: Relational.
        """
        r = self._http.get(self._url("entries", key))
        if r.status_code == 404:
            return None
        self._raise(r)
        return r.json()

    # ── list ─────────────────────────────────────────────────────────────────

    def list(
        self,
        type:      Optional[str]       = None,
        tags:      Optional[list[str]] = None,
        limit:     int                 = 50,
        offset:    int                 = 0,
        recursive: bool                = False,
    ) -> dict:
        """
        List entries with optional filtering. Returns {"entries", "total", "next_offset"}.
        Backend: Relational.
        """
        params: dict = {"limit": limit, "offset": offset, "recursive": recursive}
        if type:
            params["type"] = type
        if tags:
            params["tag"] = tags
        r = self._http.get(self._url("entries"), params=params)
        self._raise(r)
        return r.json()

    # ── semantic query ────────────────────────────────────────────────────────

    def query(
        self,
        embedding:  list[float],
        type:       Optional[str]       = None,
        top_k:      int                 = 5,
        min_score:  Optional[float]     = None,
        tags:       Optional[list[str]] = None,
        recursive:  bool                = False,
    ) -> list[dict]:
        """
        Semantic nearest-neighbour search.
        Caller MUST provide the query embedding (spec §5.1 — A2M never embeds).

        Returns list of {"entry": {...}, "score": float}, sorted by score descending.
        Backend:
          1. Relational — candidate pre-filter (type, tags, has embedding).
          2. Vector     — cosine ranking.
        """
        body: dict = {"embedding": embedding, "top_k": top_k, "recursive": recursive}
        if type:
            body["type"] = type
        if min_score is not None:
            body["min_score"] = min_score
        if tags:
            body["tag"] = tags
        r = self._http.post(self._url("query"), json=body)
        self._raise(r)
        return r.json()

    # ── delete ───────────────────────────────────────────────────────────────

    def delete(self, key: str) -> None:
        """
        Delete a single entry. Raises A2MError if not found.
        Backend: Relational.
        """
        r = self._http.delete(self._url("entries", key))
        self._raise(r)

    def delete_bulk(
        self,
        type: Optional[str]       = None,
        tags: Optional[list[str]] = None,
    ) -> int:
        """
        Bulk delete entries matching optional type / tag filters.
        No filters → clears the entire namespace.
        Returns count of deleted entries.
        Backend: Relational.
        """
        params: dict = {}
        if type:
            params["type"] = type
        if tags:
            params["tag"] = tags
        r = self._http.delete(self._url("entries"), params=params)
        self._raise(r)
        return int(r.headers.get("X-Deleted-Count", "0"))

    # ── namespace scoping ─────────────────────────────────────────────────────

    def scoped(self, *segments: str) -> "A2MClient":
        """
        Return a new client scoped to a child namespace.

        Example:
            base    = A2MClient(url, "myapp/wf-42")
            session = base.scoped("sess-abc")          # myapp/wf-42/sess-abc
            agent   = session.scoped("agent-0")        # myapp/wf-42/sess-abc/agent-0
        """
        child_ns = "/".join([self.namespace] + [s.strip("/") for s in segments])
        return A2MClient(self._base, child_ns, self._http.timeout.read or 10.0)

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "A2MClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()
