"""
A2M — AgentToMemory Protocol
LangChain adapter: A2MLangChainBaseChatMessageHistory

Implements LangChain's BaseChatMessageHistory interface (langchain-core ≥ 0.1).
Messages are serialized via langchain_core message_to_dict / messages_from_dict
and stored as A2M episodic entries, one entry per message.

Backend usage per operation:
┌───────────────────────────────┬─────────────────────────────────┬──────────────────────────────┐
│ Operation                     │ A2M call                        │ Backend                      │
├───────────────────────────────┼─────────────────────────────────┼──────────────────────────────┤
│ add_messages                  │ write(type="episodic")          │ Relational + Vector (if emb) │
│ messages  (with embed_fn)     │ query(embedding=summary_emb)    │ Relational → Vector ranking  │
│ messages  (no embed_fn ★)     │ list(type="episodic")           │ Relational only (chrono)     │
│ clear                         │ delete_bulk(type="episodic")    │ Relational                   │
└───────────────────────────────┴─────────────────────────────────┴──────────────────────────────┘
★ When no embed_fn is provided, messages are retrieved in chronological order.
  When embed_fn is provided, the last user message (or all messages joined) is
  embedded and used for semantic retrieval — useful for long conversation histories.

Install requirement:
    pip install langchain-core   (included in: pip install langchain)

Usage:
    from adapters.langchain import A2MLangChainBaseChatMessageHistory
    from client import A2MClient
    from langchain_core.runnables.history import RunnableWithMessageHistory

    client = A2MClient("http://localhost:8765", namespace="myapp/wf-1/sess-abc/agent-0")

    # Without semantic retrieval (chronological):
    history = A2MLangChainBaseChatMessageHistory(client=client)

    # With semantic retrieval — caller supplies the embedding function:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    history = A2MLangChainBaseChatMessageHistory(
        client=client,
        embed_fn=lambda t: model.encode(t).tolist(),
    )

    # Use with RunnableWithMessageHistory:
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=lambda session_id: A2MLangChainBaseChatMessageHistory(
            client=client.scoped(session_id)
        ),
    )
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict
except ImportError as exc:
    raise ImportError(
        "langchain-core is required for A2MLangChainBaseChatMessageHistory. "
        "Install it with: pip install langchain-core"
    ) from exc

from client.client import A2MClient


def _now_ms() -> str:
    """Millisecond-precision timestamp key for stable episodic ordering."""
    return f"{int(time.time() * 1000)}"


class A2MLangChainBaseChatMessageHistory(BaseChatMessageHistory):
    """
    LangChain ChatMessageHistory adapter for A2M.

    Each message is stored as one A2M episodic entry whose value is the
    dict produced by langchain_core's message_to_dict(). Keys are
    monotonically increasing millisecond timestamps so chronological
    order is preserved by the Relational backend's natural sort.

    When embed_fn is provided, the message text is embedded at write time
    and stored alongside the entry, enabling semantic retrieval via the
    Vector backend.

    Pass an instance directly to RunnableWithMessageHistory or any
    LangChain component that accepts a BaseChatMessageHistory.
    """

    def __init__(
        self,
        client: A2MClient,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        max_results: int = 50,
    ) -> None:
        """
        Args:
            client:      A2MClient scoped to the desired namespace.
            embed_fn:    Optional function to embed message text → float vector.
                         Required for semantic (vector) retrieval.
                         When None, messages are retrieved chronologically.
            max_results: Maximum number of messages to retrieve.
        """
        self.client      = client
        self.embed_fn    = embed_fn
        self.max_results = max_results

    # ── BaseChatMessageHistory interface ─────────────────────────────────────

    @property
    def messages(self) -> List[BaseMessage]:
        """
        Return stored messages.

        With embed_fn:
          Backend: Relational (candidate fetch with has_embedding filter) →
                   Vector (cosine ranking by a summary of recent context).
        Without embed_fn:
          Backend: Relational only — chronological order, last max_results messages.
        """
        if self.embed_fn:
            # Use an empty-context query; in practice callers invoke this
            # with context that can be used for ranking.  We embed a neutral
            # probe so existing messages surface in relevance order.
            probe = "conversation history"
            results = self.client.query(
                embedding=self.embed_fn(probe),
                type="episodic",
                tags=["langchain-history"],
                top_k=self.max_results,
            )
            raw = [r["entry"]["value"] for r in results]
        else:
            resp = self.client.list(
                type="episodic",
                tags=["langchain-history"],
                limit=self.max_results,
            )
            raw = [e["value"] for e in resp["entries"]]

        # Deserialise each stored dict back to a BaseMessage
        msgs: List[BaseMessage] = []
        for item in raw:
            try:
                msgs.extend(messages_from_dict([item]))
            except Exception:
                pass  # skip corrupt entries gracefully
        return msgs

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """
        Persist one or more messages.
        Backend: Relational (always) + Vector (if embed_fn provided).
        """
        for msg in messages:
            serialised = message_to_dict(msg)
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            embedding = self.embed_fn(text) if self.embed_fn else None
            self.client.write(
                key=f"msg/{_now_ms()}",
                type="episodic",
                value=serialised,
                embedding=embedding,
                meta={
                    "source_framework": "langchain",
                    "tags": ["langchain-history", f"role:{msg.type}"],
                },
            )

    def clear(self) -> None:
        """
        Delete all messages in this namespace.
        Backend: Relational.
        """
        self.client.delete_bulk(type="episodic", tags=["langchain-history"])
