# A2M — the Agent-to-Memory Protocol

> One memory store, readable and writable by agents from any framework.

[![CI](https://github.com/dibenedetto/a2m-protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/dibenedetto/a2m-protocol/actions/workflows/ci.yml)
[![Status](https://img.shields.io/badge/status-draft_v0.1-orange)](#status)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Spec](https://img.shields.io/badge/spec-a2m--0.1-green)](spec/a2m-0.1.md)

**LangChain**, **Agno**, **n8n**, **CrewAI**, **AutoGen** — each ships its own
memory model. Agents from different frameworks cannot share state, history or
knowledge, even when running inside the same workflow.

**A2M** is a small open protocol that puts a memory store behind one interface,
so any of them can read and write the same memory without being modified.

**Specification:** [spec/a2m-0.1.md](spec/a2m-0.1.md) · **Home:** <https://a2m-protocol.org>

---

## The problem

```
LangChain agent     Agno agent      n8n node        CrewAI crew
[BufferMemory]      [AgentMemory]   [$json ctx]     [EntityMemory]
      │                   │               │                │
      ▼                   ▼               ▼                ▼
  in-process          PostgreSQL      workflow ctx      ChromaDB

  ✗ No shared state   ✗ Lost across runs   ✗ No cross-framework recall
```

## Sixty seconds

```bash
pip install a2m-protocol
```

```python
from a2m import connect_local, connect_stdio, connect_http
from a2m.memory import MemoryStack

memory = connect_local(MemoryStack())              # in-process
memory = connect_stdio(["python", "-m", "a2m"])    # a child process
memory = connect_http("http://127.0.0.1:8778/")    # over the network

memory.remember("the deploy key rotates every ninety days")
memory.recall(query="how often does the key change?")
```

The transport changes; the client does not.

```
LangChain agent     Agno agent      n8n node        CrewAI crew
      │                   │               │                │
      ▼                   ▼               ▼                ▼
  A2M client          A2M client      HTTP Request    A2M client
      │                   │               │                │
      └───────────────────┴───────────────┴────────────────┘
                                  │
                     A2M — JSON-RPC 2.0, `memory/*`
                     in-process · stdio · HTTP
                                  │
                    ┌─────────────┴──────────────┐
                    │                            │
             Relational store              Vector index
          (SQLite / PostgreSQL)       (sqlite-vec / pgvector / …)

  ✓ Shared state   ✓ Persistent across runs   ✓ Semantic search built in
```

---

## A2M and MCP

**MCP is how a model calls a tool. A2M is how programs share a memory store.
Most projects want both.**

They are built to compose: A2M uses the same JSON-RPC 2.0, the same stdio
framing and the same single-endpoint HTTP binding as MCP, and its methods live
under `memory/` precisely so one endpoint can serve both. A runtime that already
speaks MCP needs no second code path.

What differs is who is on the receiving end. Ask an MCP memory tool for
something and you get text, because a tool result is written for a model to
read:

```
Found 3 memories:
1. the deploy key rotates every ninety days (score 0.72)
2. ...
```

Ask A2M and you get a record:

```json
{ "id": "01J8Z9", "content": "the deploy key rotates every ninety days",
  "created_at": "2026-07-28T10:15:30.123Z", "tier": "semantic",
  "key": "ops/deploy-key", "revision": 2, "score": 0.72 }
```

A model can work with the first. A program cannot: it cannot tell a corrected
fact from a stale one, restrict a search to one tier, replay a transcript in
order, or ask the store what it supports before calling it. Those are exactly
what a framework adapter, a router, or a second agent needs — and keeping them
is the whole reason A2M is a protocol rather than a tool schema.

**So use whichever matches your consumer:**

| Your consumer | Use |
|---|---|
| An agent, through an MCP client (Claude Code, Cursor, Claude Desktop) | the **bridge** — one command, and every MCP client can use any A2M store |
| Your own code: an adapter, a router, a pipeline, another agent | **A2M directly** |

```bash
python -m implementations.bridge_mcp --stdio python -m implementations.store_sqlite memory.db
```

The bridge is [one dependency-free file](implementations/bridge_mcp.py), and it
builds its tool list from whatever the store underneath declares.

---

## How it is shaped

**A memory store is not a database.** Agents do not query memory, they *recall*
from it: they hand over the situation they are in and expect back whatever is
worth knowing, ranked. Ranking is the primitive, not filtering.

**Most stores are not layered.** A vector database can write, search and delete
in an afternoon and has no concept of tiers, consolidation or salience. A
protocol demanding all of it would be implementable only by its own reference
implementation. So A2M defines a small mandatory **core** any store can satisfy,
and layers the rest into **capabilities** a server declares and a client checks.

| Capability | Methods | Required |
|---|---|---|
| `core` | `describe` `remember` `recall` `timeline` `forget` | **yes** |
| `tiers` | `promote` `consolidate` | no |
| `salience` | `reinforce` | no |
| `scopes` | *(adds `owner`)* | no |
| `sessions` | `session/list` `session/close` | no |
| `keys` | `fetch` *(adds `key`: addressable, upsert)* | no |
| `embeddings` | *(adds `embedding`: caller-owned, verbatim)* | no |
| `external` | *(adds `uri`: points at a file, URL or blob)* | no |
| `events` | `events` `events/subscribe` `events/unsubscribe` | no |

Which means **classic RAG is the degenerate case**: one tier, read-only,
`recall` only. An existing RAG stack becomes an A2M server by implementing two
methods and refusing writes, and every A2M client then works against it.

---

## Memory model

Every record has a `tier`, and every tier declares a `kind` that determines its
lifetime and how it is read.

| Kind | Holds | Read as | Bounded by |
|---|---|---|---|
| `working` | the live transcript | **replay**, chronological | capacity, **per conversation** |
| `episodic` | what happened | **search**, relevance + filter | capacity |
| `semantic` | what is true | **search**, relevance | unbounded |
| `procedural` | how to do things | **search**, at task start | unbounded |

Two different forces move a record, and keeping them apart is the point:

```
working ──spill──▶ episodic ──spill──▶ semantic
                       │
                       └──promote──▶ semantic        procedural
                         (recalled 3×)                ▲
                                                      └── written deliberately
```

**Spilling** is pressure: a tier is over capacity, so its weakest records are
displaced. **Promotion** is reinforcement: a record recalled often enough has
stopped being an episode and become a fact. Consolidation runs promotion
*first*, so a record that keeps proving useful is never displaced by sheer
volume of newer material.

Nothing spills into `procedural`. A fact does not decay into a procedure.

`working` is bounded **per conversation**, not per tier — otherwise two
concurrent chats compete for the same slots and the busier one evicts the
quieter one's context purely by talking more.

Where each tier should actually be *stored*, and why `working` must never go in
a vector database, is in [spec/implementing-a2m.md](spec/implementing-a2m.md).

---

## Conformance

[tools/conformance.py](tools/conformance.py) speaks only the protocol — it never
imports the server under test, so an implementation in another language is
tested exactly as a Python one is.

```bash
python -m tools.conformance --stdio python -m a2m
python -m tools.conformance --stdio node --experimental-strip-types implementations/server_minimal.ts
python -m tools.conformance --http  http://127.0.0.1:8778/
```

Six implementations ship, and the same unmodified suite passes against all of
them:

| | storage | declares | conformance |
|---|---|---|---|
| `python -m a2m` | a dict in memory | everything | 94/94 |
| [server_minimal.py](implementations/server_minimal.py) | a dict, stdlib only | `core` only | 36/36 |
| [server_minimal.ts](implementations/server_minimal.ts) | a Map, **TypeScript** | `core` + `keys` | 46/46 |
| [store_sqlite.py](implementations/store_sqlite.py) | SQLite + sqlite-vec | everything | 94/94 |
| [store_postgres.py](implementations/store_postgres.py) | **PostgreSQL + pgvector** | everything | 94/94 |
| [server_federated.py](implementations/server_federated.py) | four A2M servers | everything | 94/94 |

Checks are grouped by capability and skipped when a server does not declare one.
Declaring a capability and then not honouring it *is* a failure — a client
trusts what a server says about itself, so a server that lies there breaks
clients in ways no defensive coding on their side can fix.

[server_minimal.py](implementations/server_minimal.py) imports **nothing from
this repository**. It exists to answer a question the reference implementation
cannot: *is the specification enough on its own?* Writing it found a real bug —
the reference was rejecting unrecognised parameters, breaking the
forward-compatibility rule that lets a newer client talk to an older server.

[server_minimal.ts](implementations/server_minimal.ts) answers the next one:
*is it enough in a language that is not the reference language?* No
dependencies, no build step — `node --experimental-strip-types` runs the file as
it is. Which makes the useful demonstration a pair:
[client.py](implementations/client.py) talking to that server is a Python client
and a TypeScript server sharing not one line of code, neither written against
the other.

---

## What is in here

| | |
|---|---|
| [spec/a2m-0.1.md](spec/a2m-0.1.md) | **the normative specification** |
| [spec/implementing-a2m.md](spec/implementing-a2m.md) | what each tier means, and where it should live |
| [spec/schema/](spec/schema/) | JSON Schema for every request, response and record |
| [a2m/](a2m/) | the reference library: [protocol.py](a2m/protocol.py) · [memory.py](a2m/memory.py) · [retrieval.py](a2m/retrieval.py) · [text.py](a2m/text.py) · [jsonrpc.py](a2m/jsonrpc.py) |
| [implementations/server_minimal.py](implementations/server_minimal.py) | independent `core`-only server, standard library only |
| [implementations/server_minimal.ts](implementations/server_minimal.ts) | the same server in TypeScript, no dependencies, no build |
| [implementations/client.py](implementations/client.py) | independent client and CLI, standard library only |
| [implementations/store.py](implementations/store.py) | the tier logic both SQL backends share — no SQL in it |
| [implementations/store_sqlite.py](implementations/store_sqlite.py) | that logic on SQLite: one file, one store per tier |
| [implementations/store_postgres.py](implementations/store_postgres.py) | the same logic on PostgreSQL and pgvector |
| [implementations/server_federated.py](implementations/server_federated.py) | one A2M server per tier, one router in front |
| [implementations/bridge_mcp.py](implementations/bridge_mcp.py) | any A2M server as an MCP tool server, stdlib only |
| [implementations/adapters/](implementations/adapters/) | LangChain, Agno, CrewAI and AutoGen, talking to an A2M server unmodified |
| [tools/conformance.py](tools/conformance.py) | conformance suite for **any** A2M server |
| [tools/test_a2m.py](tools/test_a2m.py) | `python -m tools.test_a2m` — no test runner, no network |
| [tools/bench_embeddings.py](tools/bench_embeddings.py) | which embedding model backs recall, measured |
| [examples/](examples/) | [cross-framework](examples/cross_framework.py) · [embedders](examples/embedders.py) · [corpus ingestion](examples/rag_ingest.py) · [procedural memory](examples/procedural.py) · [n8n](examples/n8n_workflow.json) |
| [DECISIONS.md](DECISIONS.md) | why the non-obvious choices are what they are |

Three directories, and the split is the argument. `a2m/` is the library an
implementation is built out of. `implementations/` is the specification
implemented more than once — two storage engines, two languages, two topologies —
because one implementation only ever proves the document describes the program
that was already written. `tools/` is what judges them.

The protocol, the reference implementation and the conformance suite need
**nothing but the standard library**.

### Reading order

1. **[spec/a2m-0.1.md](spec/a2m-0.1.md)** — everything else is downstream of it.
2. **[implementations/server_minimal.py](implementations/server_minimal.py)** —
   a whole server in 470 lines, written from the specification alone. If you are
   implementing A2M, copy this.
3. **[implementations/client.py](implementations/client.py)** — the other side,
   under the same rule.

The large files — [store.py](implementations/store.py),
[store_sqlite.py](implementations/store_sqlite.py),
[server_federated.py](implementations/server_federated.py) — are reference
implementations rather than samples. Read a section when you hit the problem it
solves; reading one front to back to learn A2M is the wrong way round.

---

## Running it

Everything runs from the repository root.

```bash
python -m tools.test_a2m                       # 232 checks, offline
python -m tools.demo_stack                     # the whole stack, on disk
python -m tools.demo_stack --router            # same, federated across processes
python -m a2m                                  # the reference server, in memory
python -m implementations.store_sqlite memory.db                   # persistent, on stdio
python -m implementations.server_federated memories/ --http 8778   # federated, over HTTP
```

`server_minimal.py` and `client.py` import nothing at all, so they run as plain
files from anywhere — which is the whole claim they are making.

```bash
python implementations/client.py --stdio python -m implementations.store_sqlite memory.db -- remember "the deploy key rotates every ninety days"
python implementations/client.py --stdio python -m implementations.store_sqlite memory.db -- recall   "how often does the key change?"
```

---

## Before you implement

Two rules that are easy to miss and expensive to get wrong. Both are normative
in the specification.

**`owner` is not a security boundary.** The `scopes` capability partitions data;
it does not control access. A client asserts its own `owner`, and nothing in the
protocol stops it asserting a different one. On a local transport that is fine.
Over a network, a server **must** derive the scope from the authenticated
principal and ignore what the client claimed.

**`score` is ranking information only.** Never comparable between servers,
between calls, or against a fixed threshold. Different scorers occupy entirely
different ranges, and a model rating everything `0.9` may discriminate worse
than one spreading across `0..1`.

[CONTRIBUTING.md](CONTRIBUTING.md) has the rest, including the five rules every
implementation gets wrong.

---

## Status

Draft `a2m/0.1`. Pre-1.0, so compatibility requires an exact minor match and any
minor version may introduce breaking changes. What changed and why:
[CHANGELOG.md](CHANGELOG.md).

The most useful contribution is **an implementation this repository did not
write** — in any language, over any storage. Run the conformance suite against
it and send a pull request adding your row to the table above. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT. See [LICENSE](LICENSE).

A protocol that is expensive to implement does not get implemented.
