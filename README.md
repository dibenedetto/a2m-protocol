# Agent2Memory (A2M) Protocol

> A shared memory protocol for AI agents across frameworks.

[![Status](https://img.shields.io/badge/status-draft_v0.1-orange)](#status)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Spec](https://img.shields.io/badge/spec-HTML-green)](a2m-spec.html)

**LangChain**, **Agno**, **n8n**, **CrewAI**, **AutoGen** — each ships its own memory model. Agents from different frameworks cannot share state, history, or knowledge, even when running inside the same workflow.

**A2M** is a thin, open protocol that lets any agent framework read and write to a shared memory store through a single REST interface — without modifying existing agents.

---

## Contents

- [The problem](#the-problem)
- [How A2M works](#how-a2m-works)
- [Memory model](#memory-model)
- [Namespace addressing](#namespace-addressing)
- [REST API at a glance](#rest-api-at-a-glance)
- [Adapters](#adapters)
- [Design decisions](#design-decisions)
- [Status](#status)
- [Contributing](#contributing)

---

## The problem

```
LangChain agent     Agno agent      n8n node        CrewAI crew
[BufferMemory]      [AgentMemory]   [$json ctx]     [EntityMemory]
      │                   │               │                │
      ▼                   ▼               ▼                ▼
  in-process          PostgreSQL      workflow ctx      ChromaDB

  ✗ No shared state   ✗ Lost across runs   ✗ No cross-framework queries
```

Each framework's memory is isolated, framework-specific, and incompatible with the others.

---

## How A2M works

```
LangChain agent     Agno agent      n8n node        CrewAI crew
      │                   │               │                │
      ▼                   ▼               ▼                ▼
  A2M adapter         A2M adapter     HTTP Request    A2M adapter
      │                   │               │                │
      └───────────────────┴───────────────┴────────────────┘
                                  │
                          A2M REST API
                                  │
                    ┌─────────────┴──────────────┐
                    │                            │
             Relational store              Vector index
          (SQLite / PostgreSQL)       (FAISS / pgvector / Chroma)

  ✓ Shared state   ✓ Persistent across runs   ✓ Semantic search built-in
```

A2M defines:

- a **wire format** (JSON over HTTP, REST baseline)
- a **data model** (5 memory types, hierarchical namespaces, optional embeddings)
- a **storage contract** (relational + vector backends)
- a **4-method adapter interface** any framework implements

---

## Memory model

Every A2M entry has a `type` that determines its lifetime and indexing strategy.

| Type | Lifetime | Purpose |
|---|---|---|
| `working` | Session | In-flight scratchpad. Ephemeral task state. |
| `episodic` | Long | Interaction history. Ordered log of events. |
| `semantic` | Long | Facts and knowledge. Vector-indexed for similarity search. |
| `procedural` | Long | Learned steps and heuristics. How to accomplish goals. |
| `external` | Long | Pointer to an external resource — file, URL, blob. |

A single entry looks like:

```json
{
  "id":        "018f2a3b-…",
  "key":       "user/goal",
  "namespace": "myapp/wf-42/sess-abc/agent-0",
  "type":      "semantic",
  "value":     "Build a real-time translation pipeline",
  "embedding": [0.12, -0.04, 0.87, "…"],
  "meta": {
    "source_framework": "langchain",
    "created_at":       "2025-09-01T14:22:11Z",
    "tags":             ["user", "goal"],
    "confidence":       0.95
  }
}
```

> **Embeddings are caller-owned.** A2M stores and indexes them verbatim. The server never generates or replaces embeddings, keeping the protocol model-agnostic.

---

## Namespace addressing

Every entry is scoped to a slash-delimited namespace:

```
{app} / {workflow} / {session} / {agent}

myapp/wf-42/sess-abc/agent-0    # single agent
myapp/wf-42/sess-abc            # all agents in a session
myapp/wf-42                     # all sessions in a workflow
myapp                           # entire app
```

Callers set the namespace explicitly on every request. Trailing segments can be omitted to broaden scope. Reads with `recursive=true` traverse child namespaces.

---

## REST API at a glance

Base path: `/a2m/v1`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/{namespace}/entries` | Write or upsert an entry |
| `GET` | `/{namespace}/entries/{key}` | Read a single entry |
| `GET` | `/{namespace}/entries` | List and filter entries |
| `POST` | `/{namespace}/query` | Semantic search (caller provides embedding) |
| `DELETE` | `/{namespace}/entries/{key}` | Delete an entry |
| `WS` | `/{namespace}/subscribe` | Real-time event stream |

**Write an entry:**

```http
POST /a2m/v1/myapp/wf-42/sess-abc/agent-0/entries
Content-Type: application/json

{
  "key":       "user/goal",
  "type":      "semantic",
  "value":     "Build a real-time translation pipeline",
  "embedding": [0.12, -0.04, 0.87],
  "meta": { "tags": ["user", "goal"] }
}
```

**Semantic query:**

```http
POST /a2m/v1/myapp/wf-42/query
Content-Type: application/json

{
  "embedding": [0.11, -0.03, 0.89],
  "type":      "semantic",
  "top_k":     5
}
```

All writes are **upserts** keyed on `(namespace, key)`. Retries are safe.

---

## Adapters

A framework adapter implements four methods and delegates to the A2M HTTP API. No changes to existing agents are needed.

```python
class A2MAdapter(ABC):

    def write(self, key, type, value, embedding=None, meta={}) -> dict: ...
    def read(self, key) -> dict | None: ...
    def query(self, embedding, type=None, top_k=5) -> list[dict]: ...
    def delete(self, key) -> None: ...
```

**LangChain** (example):

```python
from langchain.memory import BaseMemory
from a2m import A2MClient

class A2MMemory(BaseMemory):
    client: A2MClient
    namespace: str

    def save_context(self, inputs, outputs):
        self.client.write(
            self.namespace,
            key="chat_history",
            type="episodic",
            value={"in": inputs, "out": outputs}
        )

    def load_memory_variables(self, inputs):
        results = self.client.query(
            self.namespace,
            embedding=embed(str(inputs)),  # caller embeds
            top_k=5
        )
        return {"history": [r["entry"]["value"] for r in results]}
```

**n8n** requires no adapter code — use the HTTP Request node pointing at `/a2m/v1/…`.

### Adapter status

| Framework | Status |
|---|---|
| LangChain | In progress |
| Agno | In progress |
| n8n | Ready (HTTP Request node) |
| CrewAI | Planned |
| AutoGen | Planned |

---

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding ownership | Caller-provided | Keeps A2M model-agnostic; embedding quality varies by domain |
| Namespace resolution | Explicit (caller sets it) | No auth context needed; simpler, auditable |
| Backend | Relational **+** vector (both required) | SQL for exact lookup and TTL; vector index for semantic search |
| Write conflict policy | Upsert (last-write-wins on key) | Idempotent writes; safe retries |

---

## Status

**Draft v0.1** — the wire format and data model are stable enough for adapter development and feedback. Not yet recommended for production use.

The spec is hosted as a self-contained HTML document:

- [**`a2m-spec.html`**](a2m-spec.html) — full technical specification (data model, API, backend requirements, adapter contract, versioning)
- [**`a2m-protocol.html`**](a2m-protocol.html) — partner-facing overview

---

## Contributing

A2M is an open initiative. We are looking for:

- **Framework maintainers** to co-design the adapter interface for their framework
- **Infrastructure partners** to validate the storage contract against real backends
- **Early adopters** to implement and test the protocol against real workloads

Open an issue to start a conversation, or reach out directly at **marco.dibenedetto@isti.cnr.it**.

### Implementing a conformant store

A conformant A2M Memory Store must:

1. Expose the REST API at `/a2m/v1/`
2. Support all 6 endpoints (write, read, list, query, delete, subscribe)
3. Provide a **relational backend** (SQLite or PostgreSQL) for key lookup, metadata filtering, and TTL
4. Provide a **vector backend** (FAISS, pgvector, Chroma, or Weaviate) for semantic search
5. Implement upsert semantics preserving `id` and `created_at`
6. Never generate or replace caller-provided embeddings

---

## License

MIT — see [LICENSE](LICENSE).
