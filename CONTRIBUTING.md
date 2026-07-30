# Contributing to A2M

The most valuable thing you can contribute is **an implementation this
repository did not write**. Everything else in this document is downstream of
that.

A2M is a specification. Its claim is that the document is implementable from the
document alone, and that claim is only worth something when someone other than
the author has done it. If you write an A2M server — in any language, over any
storage — and it passes the conformance suite, that is the contribution the
project most needs.

---

## Write a server, in any language

You need three things: [the specification](spec/a2m-0.1.md), a JSON-RPC 2.0
library or fifty lines of your own, and somewhere to put records.

**Start from the smallest honest thing.** Implement the five `core` methods,
declare `core` and nothing else, and answer `-32003 CAPABILITY_NOT_SUPPORTED`
for everything else. That is a conformant A2M server. It is not a lesser one —
capabilities exist precisely so a store can be honest about what it does.

Two files are worth reading before you start, and neither is the reference
implementation:

- [implementations/server_minimal.py](implementations/server_minimal.py) — a
  whole `core` server in 470 lines. It imports **nothing from this repository**,
  which is the point: it exists to prove the specification is enough on its own.
  Copy it.
- [implementations/server_minimal.ts](implementations/server_minimal.ts) — the
  same server in TypeScript, run by `node --experimental-strip-types` with no
  dependencies and no build step. Read it if your language is not Python; it
  shows which parts of the specification are actually language-independent and
  which two rules bite when you leave the reference language.

Then check it:

```bash
python -m tools.conformance --stdio <your server command>
```

The suite speaks only the protocol and never imports what it tests, so it does
not care what your server is written in. Checks are grouped by capability and
skipped when you do not declare one — **skipping is not failing.** Declaring a
capability and then not honouring it *is* failing, and that is the case worth
catching: a client trusts `describe`, so a server that lies there breaks clients
in ways no defensive coding on their side can fix.

### The five rules that catch everyone

In the order implementations get them wrong:

1. **Undeclared capability → `-32003`, never `-32601`.** A client cannot
   distinguish `METHOD_NOT_FOUND` from a typo in its own call. Register every
   A2M method, including the ones you do not implement.
2. **Unknown parameters are ignored, never rejected.** This is what lets a 0.2
   client talk to a 0.1 server. The reference implementation got this wrong
   until the minimal server was written from the spec and disagreed with it.
3. **Timestamps are RFC 3339 strings.** Never epoch numbers, at any boundary.
   JavaScript's instinct is `Date.now()`; that is exactly the ambiguity the rule
   removes.
4. **Echo the JSON-RPC `id` with its type intact.** Ids may be strings or
   numbers, and a language that coerces `1` into `"1"` produces a response the
   client waits for forever.
5. **`timeline` is not `recall` with a different sort.** Relevance order is what
   a search wants; creation order is what rebuilding a conversation requires. A
   server must not implement one in terms of the other.

[Appendix B of the specification](spec/a2m-0.1.md) is the same list as a
checklist, plus the transport rules.

### Then send a pull request adding your row

Add your implementation to the table in [README.md](README.md) with its storage,
what it declares, and its conformance count. If it lives in another repository,
link it — this repository does not need to host your code to count it.

That table is the project's evidence. Every row that Marco did not write makes
it stronger than any amount of documentation.

---

## Other useful contributions

**Report a specification bug.** If the document is ambiguous, contradictory, or
describes something you could not implement, that is the highest-value issue you
can file. Say what you were trying to do and where the document stopped helping.
Code and spec disagreeing is *always* a bug in the code — but the spec being
unclear is a bug in the spec.

**Adapters.** A framework this repository has not reached — Semantic Kernel,
Haystack, LlamaIndex, Pydantic AI, Mastra — mapped onto A2M. See
[implementations/adapters/](implementations/adapters/) for the four that exist.
Each one imports exactly one framework, and nothing imports them.

**Stores.** Redis, DuckDB, Qdrant, Chroma, S3. The interesting ones are those
whose access shape is genuinely different, because they test whether the tier
model is honest.

**Documentation and examples.** Especially anything that made you stop and
re-read.

---

## Working in this repository

```bash
python -m tools.check_stdlib_only   # no dependency crept in
python -m tools.test_a2m            # the offline suite, no test runner
python -m doctest a2m/memory.py a2m/text.py a2m/retrieval.py a2m/jsonrpc.py a2m/protocol.py
python -m tools.conformance --stdio python -m a2m
```

**A change is not done until every conformance target still passes.** They share
no storage code — one is not even Python, two need a database — so a change that
passes only against the reference has probably leaked an implementation
assumption into the protocol layer. The full list is in
[CLAUDE.md](CLAUDE.md), which is the working manual for this repository.

Three constraints are load-bearing and a pull request that breaks one will be
asked to change rather than merged:

- **Standard library only**, everywhere except `implementations/adapters/`,
  which may each import one framework. `tools/check_stdlib_only.py` enforces
  this, and it fails CI. A dependency here becomes a dependency of everyone who
  reads this as an example.
- **`tools/conformance.py` imports nothing from `implementations/`.** A suite
  that imported a server could only confirm the server agrees with itself.
- **`server_minimal.py`, `server_minimal.ts` and `client.py` import nothing from
  this repository at all.** That is the claim they exist to make, and it is
  worth more than the twenty duplicated lines it costs.

**Style.** Match the surrounding code: tabs, `=` aligned within a block, imports
in three groups, Google-style docstrings. Docstring examples are executed by
doctest — if you write one, it must be true.

**Changing the wire format** means updating, in the same commit: the
specification, the JSON schema, the reference implementation, both minimal
servers, the independent client, the conformance suite, both SQL stores and the
federated router. `server_minimal.py` is the one people forget, and it is the
one that proves the specification is implementable from the document alone.

New functionality **should** arrive as a new capability rather than as a change
to an existing method, since a capability is invisible to clients that do not
ask for it.

---

## Decisions

[DECISIONS.md](DECISIONS.md) records why the non-obvious choices are what they
are, with the evidence where evidence exists. If you disagree with something,
argue with the entry — that is what it is for. If you propose something with no
entry, expect to be asked for the reasoning, because it will become one.

---

## Licence

MIT. By contributing you agree your contribution is licensed under it.

A protocol that is expensive to implement does not get implemented — and that
applies to contributing to it, too. If something here is more friction than it
is worth, say so in an issue.
