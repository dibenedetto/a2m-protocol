# A2M — working notes for Claude

This repository is a **protocol**, not an application. The specification is the
product; the code exists to prove the specification is implementable and to give
implementers something to copy. When code and
[spec/a2m-0.1.md](spec/a2m-0.1.md) disagree, **the specification wins** and the
code is the bug.

## Layout

```
spec/a2m-0.1.md            normative. RFC 2119 language. The product.
spec/implementing-a2m.md   non-normative engineering guide: where each tier lives
spec/schema/               JSON Schema for every request, response, record

jsonrpc.py                 JSON-RPC 2.0 from scratch + local/stdio/HTTP transports
text.py                    Unicode tokenizer, CJK segmentation, per-language stopwords
retrieval.py               Scorer seam: lexical, embedding, hybrid; LLM consolidator
memory.py                  the reference stack: records, tiers, consolidation
a2m.py                     reference A2M server + client (all capabilities)

a2m_minimal.py             independent core-only server. IMPORTS NOTHING FROM HERE.
a2m_client.py              independent client + CLI. IMPORTS NOTHING FROM HERE.
a2m_conformance.py         conformance suite. Speaks only the protocol.
a2m_store.py               reference store: persistent, SQLite, one TierStore per tier
a2m_postgres.py            the same tier logic on PostgreSQL + pgvector. Optional dep.
a2m_router.py              reference federation: one A2M server per tier
demo_a2m_stack.py          both topologies, end to end
bench_embeddings.py        which embedding model, measured
test_a2m.py                implementation tests

adapters/langchain.py      LangChain chat history + retriever. Optional dep.
adapters/agno.py           Agno VectorDb. Optional dep.
examples/cross_framework.py  both frameworks, one store, 6/6
```

`adapters/` is the only place a third-party import is allowed, and
`adapters/__init__.py` imports none of them — a checkout without langchain-core
or agno must still run every test, every conformance target and both demos.

## How to check anything

```bash
python test_a2m.py                    # 219 checks, offline, no test runner
python -m doctest memory.py text.py retrieval.py jsonrpc.py    # examples are real
python a2m_conformance.py --stdio python a2m.py            # 78/78
python a2m_conformance.py --stdio python a2m_minimal.py    # 34/34, 8 skipped
python a2m_conformance.py --stdio python a2m_store.py s.db # 78/78
python a2m_conformance.py --stdio python a2m_router.py r/  # 78/78
python a2m_conformance.py --stdio node --experimental-strip-types a2m_minimal.ts  # 34/34
python demo_a2m_stack.py && python demo_a2m_stack.py --router   # 18/18 each

# PostgreSQL target, needs a server with pgvector:
docker run -d --name a2m-pg -e POSTGRES_PASSWORD=a2m -e POSTGRES_USER=a2m \
  -e POSTGRES_DB=a2m -p 55432:5432 pgvector/pgvector:pg16
python a2m_conformance.py --stdio python a2m_postgres.py \
  postgresql://a2m:a2m@127.0.0.1:55432/a2m                 # 78/78

python a2m_client.py --stdio python a2m_minimal.py -- describe  # the client, by hand
```

Over HTTP the suite runs five more checks that stdio cannot reach — Origin,
version header, 405, well-known — for **82/82**. Start a server with `--http`
first, then `python a2m_conformance.py --http http://127.0.0.1:8778/`.

**A change is not done until all six conformance targets still pass.** They share
no storage code — one is not even Python, one needs a database — so a change that
passes only against `a2m.py` has probably leaked an implementation assumption
into the protocol layer. If Postgres is not running, say so rather than reporting
five of six as a pass.

`a2m_store.py` and `a2m_postgres.py` **do** share code, and deliberately:
`TieredMemoryStack` holds every tier decision and touches no SQL, while
`TierStore` subclasses hold the storage. Tier logic goes in the former or it goes
in neither. If you find yourself writing SQL in the stack, that is the bug the
Postgres port was written to catch.

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
  `MemoryServer` and `MemoryRouter` build `Dispatcher(allow_batch=False)`.
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
schema, `a2m.py`, `a2m_minimal.py`, `a2m_minimal.ts`, `a2m_client.py`,
`a2m_conformance.py`, `a2m_store.py` and `a2m_router.py`.
`a2m_minimal.py` is the one people forget — and it is the one that proves the
spec is implementable from the document alone, so letting it rot defeats its
purpose.

## Open questions

Everything carried from the pre-0.1 draft is now settled: caller-owned
embeddings, addressable keys and external records all landed in 0.1, and
hierarchical namespaces were folded into keys (DECISIONS 017, 018, 019).

`events` is **out of 0.1** and its name reserved for 0.2 (spec §9.1, DECISION
020). No transport in §8 carries a server-initiated message, so nothing could
exercise it. **Do not re-add it to `A2M_CAPABILITIES`** — the conformance suite
now fails a server that declares it. Reviving it in 0.2 means answering the four
questions in DECISION 020 first, of which `owner` scoping is the one with teeth.

## Provenance

This work was developed in `dibenedetto/agent-playground`, which also contains
the agent framework (`agent.py`, `skills.py`) that consumes A2M, plus LoRA
fine-tuning notebooks. That framework is **deliberately not here** — keeping it
out is what lets this repository be standard-library only.
