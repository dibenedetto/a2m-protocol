# Agent2Memory (A2M) Protocol

> A shared memory protocol for AI agents across frameworks.

[![Status](https://img.shields.io/badge/status-draft_v0.1-orange)](#status)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Spec](https://img.shields.io/badge/spec-a2m--0.1-green)](spec/a2m-0.1.md)

**LangChain**, **Agno**, **n8n**, **CrewAI**, **AutoGen** — each ships its own memory model. Agents from different frameworks cannot share state, history, or knowledge, even when running inside the same workflow.

**A2M** is a thin, open protocol that lets any agent framework read and write to a shared memory store through one interface — without modifying existing agents.

**Specification:** [spec/a2m-0.1.md](spec/a2m-0.1.md) · **Home:** <https://a2m-protocol.org>

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

## How A2M works

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

```python
memory = connect_local(MemoryStack())               # in-process
memory = connect_stdio([sys.executable, "a2m.py"])  # a child process
memory = connect_http("http://127.0.0.1:8778/")     # over the network

memory.remember("the deploy key rotates every ninety days")
memory.recall(query="how often does the key change?")
```

The transport changes; the client does not. Methods live under `memory/`, a
namespace chosen so one endpoint can serve A2M alongside MCP's `tools/`,
`resources/` and `prompts/`.

The bindings are deliberately the ones MCP and A2A already use — JSON-RPC 2.0,
newline-delimited over stdio, a single POST endpoint over HTTP, no batches, and
no server-initiated requests. A2M adds only what a memory server specifically
needs: `Origin` validation, an `A2M-Protocol-Version` header, and a profile at
`/.well-known/a2m-server.json`. See spec [§8](spec/a2m-0.1.md) and DECISION 021.

---

## Why it is shaped this way

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

A call into an undeclared capability returns `-32003 CAPABILITY_NOT_SUPPORTED` —
**not** `-32601`, which a client cannot distinguish from a typo.

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

## What is in here

| | |
|---|---|
| [spec/a2m-0.1.md](spec/a2m-0.1.md) | **the normative specification** |
| [spec/implementing-a2m.md](spec/implementing-a2m.md) | what each tier means, and where it should live |
| [spec/schema/](spec/schema/) | JSON Schema for every request, response and record |
| [a2m.py](a2m.py) | reference server and client, all capabilities |
| [a2m_minimal.py](a2m_minimal.py) | independent `core`-only server, standard library only |
| [a2m_minimal.ts](a2m_minimal.ts) | the same server in TypeScript, no dependencies, no build |
| [a2m_client.py](a2m_client.py) | independent client and CLI, standard library only |
| [a2m_conformance.py](a2m_conformance.py) | conformance suite for **any** A2M server |
| [a2m_store.py](a2m_store.py) | reference persistent store: one SQLite file, one store per tier |
| [a2m_postgres.py](a2m_postgres.py) | the same tier logic on PostgreSQL and pgvector |
| [a2m_router.py](a2m_router.py) | reference federation: one process per tier, one router |
| [memory.py](memory.py) · [retrieval.py](retrieval.py) · [text.py](text.py) · [jsonrpc.py](jsonrpc.py) | the reference stack |
| [adapters/](adapters/) | LangChain and Agno, talking to an A2M server unmodified |
| [examples/cross_framework.py](examples/cross_framework.py) | both frameworks sharing one store, as a runnable script |
| [test_a2m.py](test_a2m.py) | `python test_a2m.py` — no test runner, no network |
| [bench_embeddings.py](bench_embeddings.py) | which embedding model backs recall, measured |
| [DECISIONS.md](DECISIONS.md) | why the non-obvious choices are what they are |

The protocol, the reference implementation and the conformance suite need
**nothing but the standard library**. `sqlite-vec` and `ollama` are optional and
used only by the reference store and the benchmark.

### Reading order

The files are not equally good places to start, and the two largest are the
worst ones.

1. **[spec/a2m-0.1.md](spec/a2m-0.1.md)** — everything else is downstream of it.
2. **[a2m_minimal.py](a2m_minimal.py)** (467 lines) — a whole server, written
   from the specification alone. If you are implementing A2M, copy this.
3. **[a2m_client.py](a2m_client.py)** — the other side, under the same rule.
   Between them they show both halves of a conversation with nothing shared.
4. **[a2m_store.py](a2m_store.py)** and **[a2m_router.py](a2m_router.py)** —
   1700 and 1000 lines. These are *reference implementations*, not samples: they
   exist to prove the protocol survives a real store and a real federation, and
   to be read a section at a time when you hit the problem they solve. Reading
   either front to back to learn A2M is the wrong way round.

---

## Conformance

[a2m_conformance.py](a2m_conformance.py) speaks only the protocol — it never
imports the server under test, so an implementation in another language is
tested exactly as a Python one is.

```
python a2m_conformance.py --stdio python a2m.py
python a2m_conformance.py --stdio python a2m_minimal.py
python a2m_conformance.py --stdio node --experimental-strip-types a2m_minimal.ts
python a2m_conformance.py --http  http://127.0.0.1:8778/
```

Checks are grouped by capability and skipped when undeclared. Declaring a
capability and then not honouring it *is* a failure — a client trusts
`describe`, so a server lying there breaks clients in ways no defensive coding
on their side can fix.

Six implementations ship, and the same unmodified suite passes against all of
them:

| | storage | declares | conformance |
|---|---|---|---|
| [a2m.py](a2m.py) | a dict in memory | everything | 78/78 |
| [a2m_minimal.py](a2m_minimal.py) | a dict, stdlib only | `core` only | 34/34 |
| [a2m_minimal.ts](a2m_minimal.ts) | a Map, **TypeScript** | `core` only | 34/34 |
| [a2m_store.py](a2m_store.py) | SQLite + sqlite-vec | everything | 78/78 |
| [a2m_postgres.py](a2m_postgres.py) | **PostgreSQL + pgvector** | everything | 78/78 |
| [a2m_router.py](a2m_router.py) | four A2M servers | everything | 78/78 |

Over HTTP the suite runs five further checks that stdio cannot reach — `Origin`,
the version header, `405` on GET, the well-known profile — for **82/82**.

[a2m_minimal.py](a2m_minimal.py) imports **nothing from this repository**. It
exists to answer a question the reference implementation cannot: *is the
specification enough on its own?* An implementation sharing code with the
reference proves only that the reference agrees with itself. Writing it found a
real bug — `a2m.py` was rejecting unrecognised parameters, breaking the
forward-compatibility rule that lets a newer client talk to an older server.

[a2m_minimal.ts](a2m_minimal.ts) answers the next question: *is it enough in a
language that is not the reference language?* It needs no dependencies and no
build — `node --experimental-strip-types` runs the file as it is. The port is
where a JSON-shaped protocol earns the description, and two rules did the work:
timestamps stay RFC 3339 strings where JavaScript's instinct is an epoch integer
(§3.3), and an `id` is echoed back with its type intact where JavaScript would
happily turn `1` into `"1"` (§3.2).

Which makes the useful demonstration a pair: [a2m_client.py](a2m_client.py)
talking to [a2m_minimal.ts](a2m_minimal.ts) is a Python client and a TypeScript
server that share not one line of code, and neither was written against the
other.

---

## Two things to know before implementing

**`owner` is not a security boundary.** The `scopes` capability partitions data;
it does not control access. A client asserts its own `owner`, and nothing in the
protocol stops it asserting a different one. On a local transport that is fine.
Over a network a server **must** derive the scope from the authenticated
principal and ignore what the client claimed — spec §6.

**`score` is ranking information only.** Never comparable between servers,
between calls, or against a fixed threshold. Different scorers occupy entirely
different ranges, and a model rating everything `0.9` may discriminate worse
than one spreading across `0..1` — spec §5.3.

---

## Running it

```
python test_a2m.py                          # 219 checks, offline
python demo_a2m_stack.py                    # the whole stack, on disk
python demo_a2m_stack.py --router           # same, federated across processes
python a2m_store.py memory.db               # a persistent server on stdio
python a2m_router.py memories/ --http 8778  # federated, over HTTP
```

Talking to any of them, with a client that shares no code with them:

```
python a2m_client.py --stdio python a2m_store.py memory.db -- remember "the deploy key rotates every ninety days"
python a2m_client.py --stdio python a2m_store.py memory.db -- recall   "how often does the key change?"
python a2m_client.py --http  http://127.0.0.1:8778/         -- describe
```

---

## Status

Draft `a2m/0.1`. Pre-1.0, so compatibility requires an exact minor match and any
minor version may introduce breaking changes.

### History, and what is still open

This repository was reset on 2026-07-28 to carry A2M 0.1. The earlier REST draft
is **not gone** — it remains in this repository's history at commit
[`7ca383c`](../../commit/7ca383c), recoverable with `git checkout 7ca383c`.

| | draft | 0.1 |
|---|---|---|
| wire format | REST over HTTP | JSON-RPC 2.0 — in-process, stdio, HTTP |
| memory kinds | working, episodic, semantic, procedural, **external** | the first four |
| addressing | hierarchical namespaces, recursive reads | `key_prefix`, plus `owner` and `session` |
| identity | caller-set `key`, upsert by key | **both** — opaque `id` *and* addressable `key` |
| embeddings | **caller-owned**, stored verbatim | **both** — caller-owned, or server-side |
| record kinds | `external` as a fifth *type* | `external` as a record *property*, legal in any tier |
| events | `WS /subscribe` | deferred — name reserved for 0.2 |
| conformance | — | executable suite, four passing implementations |

The four memory kinds survived unchanged, having been arrived at twice
independently — which is the strongest evidence in this repository that they are
the right four.

Two of the draft's ideas are now **in** 0.1, and both improve on what they
replaced:

- **Addressable keys** (`keys`). `id` identifies a write; `key` addresses a
  *fact*. Writing to an occupied key **replaces** what is there, keeping the id
  and advancing `revision`. That is what makes a memory correctable rather than
  merely appendable — a superseded fact that is only *outnumbered* by its
  successor is still there to be recalled, with the same confidence as the truth.
  Slash-delimited keys plus `key_prefix` give the hierarchy and the recursive
  scope reads the draft wanted from namespaces, without a second addressing
  dimension to keep consistent with the first.

- **Caller-owned embeddings** (`embeddings`). A vector supplied by the caller is
  stored **verbatim** — never regenerated, never replaced. Two frameworks
  embedding with different models can share a store only if neither has its
  vectors silently rewritten into the other's space. A store with no model at all
  still answers vector searches. Mismatched widths are refused with `-32008`
  rather than compared: cosine between vectors of different lengths is not a
  worse answer, it is not an answer.

- **External records** (`external`). A record may point at a file, URL or blob
  instead of containing it. `content` keeps its ordinary meaning — the text that
  gets indexed, so the reference is findable — while `uri` says where the real
  thing lives. The server **never dereferences it**: fetching a caller's URI
  would make every write a request the server chose to issue to an address its
  caller supplied.

Hierarchical namespaces were **folded into keys** rather than added as a
separate dimension; see [DECISIONS.md](DECISIONS.md) 017. All four of the
draft's ideas are now either in 0.1 or accounted for.

---

## Licence

MIT. See [LICENSE](LICENSE) and spec §10.

A protocol that is expensive to implement does not get implemented.
