# Changelog

Notable changes to the A2M specification and to the implementations in this
repository. Reasoning lives in [DECISIONS.md](DECISIONS.md); this file records
*what* changed and *when*.

The specification version (`a2m/0.1`) and the package version (`0.1.0`) move
independently: a package release that only touches implementations does not
change the protocol.

---

## `a2m/0.1` — unreleased

The first version of the protocol intended for anyone else to implement.

### The protocol

- **JSON-RPC 2.0 on three bindings** — in-process, stdio, and a single HTTP
  endpoint. Deliberately the bindings MCP and A2A already use, down to the
  stdio framing, so a runtime that speaks one needs no second code path.
- **A five-method core**, with everything else in capabilities a server
  declares and a client checks: `tiers`, `salience`, `scopes`, `sessions`,
  `keys`, `embeddings`, `external`, `events`, `summarize`, `prompt`.
- **Four memory kinds** — `working`, `episodic`, `semantic`, `procedural` —
  with spilling and promotion as separate forces, and nothing spilling into
  `procedural`.
- **Addressable keys.** `id` identifies a write; `key` addresses a fact.
  Writing to an occupied key replaces what is there, keeping the id and
  advancing `revision`, so a corrected memory stops being recallable rather
  than merely being outnumbered by its successor.
- **Caller-owned embeddings**, stored verbatim and never regenerated. Two
  frameworks using different models can share one store only if neither has its
  vectors silently rewritten into the other's space. A store with no embedding
  model at all still answers vector searches.
- **External records.** A record may carry a `uri` pointing at a file, URL or
  blob instead of containing it, and the server **never dereferences it**.
- **Events**, as a cursor any transport can carry, with push as an optional
  layer where a notification can be delivered. Volume is coalesced: a
  consolidation moving ten thousand records is one event.
- **Summarize**, distilling a set of records into a durable statement on
  demand. It writes ordinary records, never deletes its sources, and replaces
  rather than accumulates when given a key.
- **Prompt rendering**, as an optional field beside the records rather than
  instead of them — so a client in any language, including an n8n HTTP node,
  gets prompt-ready text without a helper library in its own language.
- **Read-only servers are conformant** (§2.1), which is what makes exposing an
  existing corpus a checkable claim rather than an assurance.
- **An executable conformance suite** that speaks only the protocol and never
  imports the server it tests.

### Superseding the pre-0.1 draft

The repository was reset on 2026-07-28 to carry A2M 0.1. The earlier REST draft
remains in git history at commit `7ca383c` and is recoverable with
`git checkout 7ca383c`. What changed:

| | pre-0.1 draft | 0.1 |
|---|---|---|
| wire format | REST over HTTP | JSON-RPC 2.0 — in-process, stdio, HTTP |
| memory kinds | working, episodic, semantic, procedural, **external** | the first four |
| addressing | hierarchical namespaces, recursive reads | `key_prefix`, plus `owner` and `session` |
| identity | caller-set `key`, upsert by key | **both** — opaque `id` *and* addressable `key` |
| embeddings | **caller-owned**, stored verbatim | **both** — caller-owned, or server-side |
| record kinds | `external` as a fifth *type* | `external` as a record *property*, legal in any tier |
| events | `WS /subscribe` | cursor polling on every transport, push where one can carry it |
| conformance | — | executable suite, seven passing implementations |

The four memory kinds survived unchanged, having been arrived at twice
independently — which is the strongest evidence in this repository that they are
the right four.

Every idea the draft got right is now in 0.1, in most cases improved:
addressable keys and caller-owned embeddings came back after the reset dropped
them (DECISIONS 017 and 018); external records became a record *property*
rather than a fifth kind, because "points at a file" describes content while
the four kinds describe lifetime (DECISION 019); hierarchical namespaces were
folded into slash-delimited keys plus `key_prefix` rather than added as a second
addressing dimension (DECISION 017); and `WS /subscribe` became a cursor,
because no binding in the specification carries a server-initiated message on
every transport (DECISION 026).

### Implementations in this repository

- Reference library and server, plus a CLI (`python -m a2m`).
- Two independent implementations written from the specification alone:
  `server_minimal.py` and `server_minimal.ts`, importing nothing from this
  repository. The Python one found a real bug in the reference — it was
  rejecting unrecognised parameters, breaking forward compatibility.
- Two storage engines sharing all tier logic and no SQL: SQLite + sqlite-vec,
  PostgreSQL + pgvector.
- A federated topology: one A2M server per tier behind a router that reaches
  its backends over the protocol.
- `bridge_mcp.py`, exposing any A2M server as an MCP tool server.
- Adapters for LangChain, Agno, CrewAI, AutoGen and the OpenAI Agents SDK; an
  importable n8n workflow, named `<Framework>A2M<WhatItImplements>` —
  `AgnoA2MVectorDb`, `LangChainA2MRetriever`, `OpenAIAgentsA2MSession`. The
  framework leads because the second half is not distinctive: five of them
  implement something called a store, a memory or a session. The OpenAI one
  imports nothing: the SDK declares `Session` as a structural `Protocol`, so
  having the four methods is being a session.
- `LangChainA2MRetriever` takes an embedder, so LangChain retrieval can rank by
  vector in the caller's own space rather than by text only. `cross_framework.py`
  now checks the whole crossing: Agno writes a document with its vector, LangChain
  searches with the same embedder and finds it first, a query sharing no word
  with the document finds nothing without one, and the vector the store returns
  is the one Agno computed. Given an embedder on a server that never declared
  `embeddings`, the retriever refuses at construction instead of silently
  ranking by text.
- A cross-framework interop matrix: every adapter writes, every adapter reads,
  and the grid must be complete. Its first run found the CrewAI adapter could
  only read what it had written itself.
- Interoperability checked a second time, one level up, at the agent rather than
  the store: `examples/agent_interop.py` runs an OpenAI Agents SDK agent, an Agno
  agent and an AutoGen agent against one server and asserts that each one's fact
  reached the *others' model calls*. Each agent runs on a scripted model that
  records what it was handed, so it is offline, free and deterministic. It found
  the same namespace bug a third time: `AutoGenA2MMemory` scoped reads to its own
  namespace, and its `clear` with no namespace deleted every record that had
  none — including other frameworks'.
- Packaging as `a2m-protocol`, standard library only, Python 3.10+.
