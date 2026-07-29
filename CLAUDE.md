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
a2m_conformance.py         conformance suite. Speaks only the protocol.
a2m_store.py               sample: persistent, SQLite, one TierStore per tier
a2m_router.py              sample: federated, one A2M server per tier
demo_a2m_stack.py          both topologies, end to end
bench_embeddings.py        which embedding model, measured
test_a2m.py                implementation tests
```

## How to check anything

```bash
python test_a2m.py                    # 171 checks, offline, no test runner
python -m doctest memory.py text.py retrieval.py jsonrpc.py    # examples are real
python a2m_conformance.py --stdio python a2m.py            # 51/51
python a2m_conformance.py --stdio python a2m_minimal.py    # 32/32, 4 skipped
python a2m_conformance.py --stdio python a2m_store.py s.db # 51/51
python a2m_conformance.py --stdio python a2m_router.py r/  # 51/51
python demo_a2m_stack.py && python demo_a2m_stack.py --router   # 18/18 each
```

**A change is not done until all four conformance targets still pass.** They
share no storage code, so a change that passes only against `a2m.py` has
probably leaked an implementation assumption into the protocol layer.

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
- **`working` capacity is per session.** Otherwise a busy conversation evicts a
  quiet one's context by talking more.

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
schema, `a2m.py`, `a2m_minimal.py`, `a2m_conformance.py`, and both samples.
`a2m_minimal.py` is the one people forget — and it is the one that proves the
spec is implementable from the document alone, so letting it rot defeats its
purpose.

## Open questions

Carried from the pre-0.1 draft, still unresolved — see the README's *what is
still open*: `external` records, caller-owned embeddings, addressable keys with
upsert, hierarchical namespaces.

Also unresolved: `events` is declared as a capability but no notification
transport is specified beyond JSON-RPC notifications, and neither sample emits
any.

## Provenance

This work was developed in `dibenedetto/agent-playground`, which also contains
the agent framework (`agent.py`, `skills.py`) that consumes A2M, plus LoRA
fine-tuning notebooks. That framework is **deliberately not here** — keeping it
out is what lets this repository be standard-library only.
