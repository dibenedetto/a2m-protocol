# A2M — working notes for Claude

This repository is a **protocol**, not an application. The specification is the
product; the code exists to prove the specification is implementable and to give
implementers something to copy. When code and
[spec/a2m-0.1.md](spec/a2m-0.1.md) disagree, **the specification wins** and the
code is the bug.

## Layout

Three directories, and the split is the argument: `a2m/` is the library,
`implementations/` is the spec implemented more than once, `tools/` is what
judges them.

```
spec/a2m-0.1.md            normative. RFC 2119 language. The product.
spec/implementing-a2m.md   non-normative engineering guide: where each tier lives
spec/schema/               JSON Schema for every request, response, record

a2m/jsonrpc.py             JSON-RPC 2.0 from scratch + local/stdio/HTTP transports
a2m/text.py                Unicode tokenizer, CJK segmentation, per-language stopwords
a2m/retrieval.py           Scorer seam: lexical, embedding, hybrid; LLM consolidator
a2m/memory.py              the reference stack: records, tiers, consolidation
a2m/protocol.py            reference A2M server + client (all capabilities)
a2m/__init__.py            re-exports the surface implementers reach for
a2m/__main__.py            `python -m a2m` — the reference server on the CLI

implementations/server_minimal.py    core-only server. IMPORTS NOTHING FROM HERE.
implementations/server_minimal.ts    core + keys, in TypeScript. No deps, no build.
implementations/client.py            independent client + CLI. IMPORTS NOTHING FROM HERE.
implementations/store.py             tier logic shared by both SQL backends. NO SQL.
implementations/store_sqlite.py      that logic on SQLite, one TierStore per tier
implementations/store_postgres.py    the same on PostgreSQL + pgvector. Optional dep.
implementations/server_federated.py  one A2M server per tier, one router in front
implementations/bridge_mcp.py        any A2M server as an MCP tool server. Stdlib only.
implementations/adapters/langchain.py  LangChain chat history + retriever. Optional dep.
implementations/adapters/agno.py       Agno VectorDb. Optional dep.
implementations/adapters/crewai.py     CrewAI StorageBackend. Optional dep, needs Python <= 3.13.
implementations/adapters/autogen.py    AutoGen Memory protocol. Optional dep.

tools/conformance.py       conformance suite. Speaks only the protocol.
tools/test_a2m.py          implementation tests
tools/demo_stack.py        both topologies, end to end
tools/bench_embeddings.py  which embedding model, measured
tools/check_stdlib_only.py walks every import; CI fails on a stray dependency

www/render.py              renders spec + DECISIONS to www/public/. Stdlib only.
www/index.html             the landing page, hand-written
.github/workflows/ci.yml   every conformance target, on every push

examples/cross_framework.py  both frameworks, one store, 6/6
examples/embedders.py        the scorer seam and the embed seam, offline, 11/11
examples/rag_ingest.py       corpus ingestion: group, keys, external, 16/16
examples/procedural.py       skills in, promotion in, nothing spilled in, 14/14
examples/n8n_workflow.json   n8n over stock HTTP nodes, importable
```

`implementations/adapters/` is the only place a third-party import is allowed,
and its `__init__.py` imports none of them — a checkout without langchain-core
or agno must still run every test, every conformance target and both demos.

Anything importing `a2m` runs as a module from the repository root
(`python -m implementations.store_sqlite`). `server_minimal.py`, `client.py` and
`server_minimal.ts` import nothing, so they run as plain files from anywhere —
that is the claim they exist to make, and it is why they are not `-m`.

`tools/conformance.py` imports `a2m.jsonrpc` and nothing else from this
repository. It must never import anything under `implementations/`: a suite that
imported a server could only confirm the server agrees with itself.

## How to check anything

Everything runs from the repository root.

```bash
python -m tools.check_stdlib_only     # no dependency crept in anywhere
python -m tools.test_a2m              # 232 checks, offline, no test runner
python -m doctest a2m/memory.py a2m/text.py a2m/retrieval.py a2m/jsonrpc.py a2m/protocol.py  # examples are real
python -m tools.conformance --stdio python -m a2m                            # 94/94
python -m tools.conformance --stdio python implementations/server_minimal.py # 36/36, 9 skipped
python -m tools.conformance --stdio python -m implementations.store_sqlite s.db      # 94/94
python -m tools.conformance --stdio python -m implementations.server_federated r/    # 94/94
python -m tools.conformance --stdio node --experimental-strip-types implementations/server_minimal.ts  # 46/46
python -m tools.demo_stack && python -m tools.demo_stack --router   # 18/18 each

# PostgreSQL targets, need a server with pgvector:
docker run -d --name a2m-pg -e POSTGRES_PASSWORD=a2m -e POSTGRES_USER=a2m \
  -e POSTGRES_DB=a2m -p 55432:5432 pgvector/pgvector:pg16
python -m tools.conformance --stdio python -m implementations.store_postgres \
  postgresql://a2m:a2m@127.0.0.1:55432/a2m                          # 94/94
python -m tools.conformance --stdio python -m implementations.server_federated \
  postgresql://a2m:a2m@127.0.0.1:55432/a2mfed --backend postgres    # 94/94
# (the a2mfed database must exist: docker exec a2m-pg psql -U a2m -d a2m -c "CREATE DATABASE a2mfed")

python implementations/client.py --stdio python implementations/server_minimal.py -- describe
```

Over stdio the full-capability targets include push delivery end to end. Over
HTTP the suite instead checks that push is honestly refused, plus the binding
checks stdio cannot reach — Origin, version header, 405, well-known — for
**94/94** there too. Start a server with `--http` first, then
`python -m tools.conformance --http http://127.0.0.1:8778/`.

**A change is not done until all seven conformance targets still pass.** They
share no storage code — one is not even Python, two need a database — so a change
that passes only against `python -m a2m` has probably leaked an implementation
assumption into the protocol layer. If Postgres is not running, say so rather
than reporting five of seven as a pass.

The two SQL stores **do** share code, and deliberately: `store.py` holds
`TieredMemoryStack` and the `TierStore` base and contains no SQL at all, while
`store_sqlite.py` and `store_postgres.py` hold the storage. Tier logic goes in
`store.py` or it goes in neither. If you find yourself writing SQL — a table
name, a placeholder, a dialect — in `store.py`, that is the bug the Postgres port
was written to catch. The file boundary is the enforcement; before it existed
`TieredMemoryStack` tested `isinstance(store, DurableStore)` and every PostgreSQL
tier silently failed it.

Ask the store, never its class. `TierStore.EMBEDS` is how the stack decides what
gets a vector, precisely because a backend is under no obligation to subclass
anything in `store.py`.

Docstring examples are executed by doctest. If you write one, it must be true.

## Invariants — easy to break by accident

- **`working` is replayed, never searched.** Served by `timeline`, not `recall`.
  Never embed it: it turns over before anything searches it. Ranking a transcript
  destroys it.
- **Groups move whole.** A record's `group` ties an assistant `tool_calls`
  message to the tool results answering it. Splitting them produces a transcript
  most providers reject outright.
- **Nothing spills into `procedural`.** A fact does not decay into a procedure.
  It is written deliberately, or reached by `promote`.
- **Promotion runs before spilling** in `consolidate`. Otherwise a record that
  keeps proving useful is displaced by sheer volume of newer material.
- **`owner` is not a security boundary.** It is data partitioning. Over a network
  the scope must come from the authenticated transport, never from the client.
- **`score` is ranking only.** Never comparable across servers or calls. Do not
  compare, threshold or average scores from different sources.
- **Undeclared capability → `-32003`, never `-32601`.** A client cannot tell
  `METHOD_NOT_FOUND` from a typo.
- **Unknown parameters are ignored, never rejected.** This is what lets a 0.2
  client talk to a 0.1 server. Every handler ends in `**ignored`.
- **Timestamps are RFC 3339 strings.** Never epoch numbers, at any boundary.
- **A caller's embedding is stored verbatim.** Never regenerate it, never
  replace it, in any tier. A store that re-embeds moves every record into its own
  model's space, which is the interoperability failure A2M exists to remove.
- **The server never dereferences a `uri`.** Not on write, not on recall, not
  in the background. Fetching a caller's URI is server-side request forgery in a
  component whose job is accepting arbitrary strings from agents.
- **Writing to an occupied key replaces.** It keeps the id and advances
  `revision`. Appending instead leaves the stale fact recallable.
- **`working` capacity is per session.** Otherwise a busy conversation evicts a
  quiet one's context by talking more.
- **No batches, no server-initiated requests.** Every binding carries single
  JSON-RPC objects; an array is `-32600`. Both rules exist to match MCP, so an
  agent runtime that already speaks it needs no second code path (DECISION 021).
  `MemoryServer` and `MemoryRouter` build `Dispatcher(allow_batch=False)`. The
  one server-to-client message is the `memory/event` **notification** — no
  `id`, no reply expected — and only after an explicit subscribe.
- **Push and poll are one log.** Every event goes through `EventLog` first;
  push mirrors it. Never emit a notification that is not also pollable — a
  client that misses it could never catch up. A consolidation is **one**
  coalesced event, never one per record.
- **Push is never advertised over HTTP.** The binding cannot carry a
  notification back, so `describe` must report `"push": false` there and
  subscribe must answer `-32003`. `serve_a2m_http` and `serve_a2m_stdio` set
  `push_transport`; bare `serve_stdio` without the drain would accept a
  subscription and then deliver nothing — use `serve_a2m_stdio`.
- **Event visibility is recall visibility.** A scoped caller sees its own
  events and unscoped ones, never another owner's (spec §4.12, DECISION 026).
- **Over HTTP, an unpermitted `Origin` is 403.** The interesting deployment is
  local and unauthenticated, which is exactly where a browsed page could
  otherwise drain the agent's memory. Use `serve_a2m_http`, never bare
  `serve_http` — the bare one has no version header and no well-known profile.

## Style

Tabs, and `=` aligned within a block — match the surrounding code, it is
deliberate. Imports in three groups: plain, stdlib `from`, local `from`, two
blank lines between groups and between top-level definitions.

Docstrings are Google style: summary line, then `Args:` / `Returns:` / `Raises:`
/ `Example:`. Never open a docstring with a section header.

## Editing the specification

The spec is unpublished, so amending `0.1` in place is still free — but that
stops the moment it is announced at <https://a2m-protocol.org>. After that,
additions are a new capability (invisible to clients that do not ask) and
breaking changes need a version bump.

New functionality **should** arrive as a capability rather than as a change to
an existing method.

If you change the wire format, update in the same commit: the spec, the JSON
schema, `a2m/protocol.py`, `implementations/server_minimal.py`,
`implementations/server_minimal.ts`, `implementations/client.py`,
`tools/conformance.py`, `implementations/store_sqlite.py`,
`implementations/store_postgres.py` and `implementations/server_federated.py`.
`server_minimal.py` is the one people forget — and it is the one that proves the
spec is implementable from the document alone, so letting it rot defeats its
purpose.

## Open questions

Everything carried from the pre-0.1 draft is now settled: caller-owned
embeddings, addressable keys, external records and events all landed in 0.1,
and hierarchical namespaces were folded into keys (DECISIONS 017, 018, 019,
026). `events` answered DECISION 020's four questions — scoping is recall's,
polling needs no subscription, volume is coalesced, ordering is the cursor's —
and is exercised by every conformance target. No capability names are
currently reserved (spec §9.1).

## Provenance

This work was developed in `dibenedetto/agent-playground`, which also contains
the agent framework (`agent.py`, `skills.py`) that consumes A2M, plus LoRA
fine-tuning notebooks. That framework is **deliberately not here** — keeping it
out is what lets this repository be standard-library only.
