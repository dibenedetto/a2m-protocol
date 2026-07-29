# Decisions

Why the non-obvious choices are what they are, with the evidence where evidence
exists. Recorded so that a later reader — or a later version — argues with the
reasoning rather than rediscovering it.

---

## 001 — A small core plus declared capabilities

**Decision.** Five mandatory methods; everything else is a capability a server
declares and a client checks.

**Why.** A vector database can write, search and delete in an afternoon and has
no concept of tiers, consolidation or salience. A protocol mandating all of it
would be implementable only by its own reference implementation, and a protocol
only its author satisfies is not a protocol.

**Consequence, unplanned and good.** Classic RAG turns out to be the degenerate
case: one tier, read-only, `recall` only. An existing RAG stack can be exposed
over A2M in an afternoon — implement `describe` and `recall`, return
`-32004 READ_ONLY` from `remember` — and every A2M client works against it.

---

## 002 — Undeclared capability returns `-32003`, not `-32601`

**Decision.** Every A2M method is registered even when unimplemented, so it can
answer `CAPABILITY_NOT_SUPPORTED` rather than `METHOD_NOT_FOUND`.

**Why.** A client cannot distinguish `-32601` from a typo in its own call. The
distinction is the difference between "ask for something else" and "you have a
bug".

---

## 003 — Timestamps are RFC 3339 strings, never epoch numbers

**Decision.** `"2026-07-28T10:15:30.123Z"`, UTC, millisecond precision.

**Why.** Epoch floats are ambiguous about unit and timezone, and lose precision
in any language whose only number is a double — which is most of them. This is
invisible in a single-language implementation and fatal across two.

**Cost.** Breaking change to the pre-0.1 wire format. Taken while free.

---

## 004 — `owner` is not a security boundary

**Decision.** Say so in normative language, and require network servers to
derive scope from the authenticated principal.

**Why.** The client asserts its own `owner` and the server believes it. In-process
that is fine — it is data partitioning. Over a network it means any agent can
read any other agent's private memory by claiming their name.

This was a real flaw in the implementation before it was a specification clause.
Publishing a scoping mechanism that *reads* like a security feature without
saying it is not would have been the more damaging option.

---

## 005 — `score` is ranking information only

**Decision.** Normatively: never comparable between servers, calls, or against a
fixed threshold.

**Why.** Different scorers occupy entirely different ranges. A model that rates
everything `0.9` may discriminate worse than one spreading across `0..1`. A
`min_score` tuned against one backend is meaningless against another — and A2M
explicitly permits swapping the ranker, which silently changes what a threshold
means.

**Consequence.** A federating router cannot merge ranked results by score,
because a router *is* a client. See 009.

---

## 006 — Four memory kinds, and procedural is not the fourth link

**Decision.** `working → episodic → semantic` is the spill chain. `procedural`
sits outside it.

**Why.** Appending procedural to the chain would encode "facts that get old
become procedures", which is backwards. Procedural memory is not a later stage
of decay; it is acquired by repetition and written deliberately.

So two forces move a record, and keeping them apart is the point:

- **spill** — pressure. A tier is over capacity; its weakest records are displaced.
- **promote** — reinforcement. A record recalled `promote_after` times has stopped
  being an episode and become a fact.

Consolidation runs **promotion first**, otherwise a record that keeps proving
useful is displaced by sheer volume of newer material arriving above it.

**Corroboration.** The pre-0.1 REST draft arrived at the same four kinds
independently, without the spill-chain reasoning. That is the strongest evidence
available here that the four are right.

---

## 007 — `recall` and `timeline` are separate, and neither implements the other

**Decision.** Two methods. A server **must not** implement one in terms of the
other.

**Why.** Relevance order is what a search wants; creation order is what
rebuilding a conversation requires. Reordering a transcript by relevance destroys
it, and separating an assistant `tool_calls` message from its results produces a
transcript most providers reject outright.

This is also why `working` must never live in a vector database. It is
**replayed**, not searched.

---

## 008 — Sessions, and per-conversation capacity

**Decision.** `session` is a first-class field with its own capability, `where`
is available on `timeline`, and `working` capacity applies *within* a session.

**Why.** Two gaps, found by testing rather than by design:

- `timeline` had no filter at all, so a session could be *searched* but never
  *replayed* — in the one tier that exists to be replayed. A metadata convention
  could not have fixed that; it needed a protocol change.
- With one shared working tier, two conversations competed for the same slots:
  `chat-2` evicted `chat-1`'s turns purely by talking more.

**Closing a session percolates the whole stack**, rather than one hop. A finished
conversation will never be replayed, so its records leave working memory
immediately instead of waiting for capacity pressure; every tier below then
applies its own ordinary rules to what arrives. `procedural` is untouched — a
finished conversation does not become a procedure.

---

## 009 — A federating router cannot merge by score

**Decision.** Two strategies, selectable: `RankFusion` uses each backend's
*order* and discards its numbers; `Rerank` ignores backend ordering and scores
every candidate with one model.

**Why.** Decision 005 forbids comparing scores across servers, and a router is a
client. Sorting the union of four backends' results by score is exactly the
mistake the specification tells clients not to make.

**What the implementation taught.** `Rerank` first returned nothing at all,
because the backends' own scorers dropped the right record before the router saw
it. **A re-ranker can only rank what the backends chose to return.** So `Rerank`
sets `wants_candidates`, and the router withholds the query and asks for a broad
pool instead. Measured on the same corpus:

| strategy | `"what is my name?"` | why |
|---|---|---|
| `rank-fusion` | nothing | the backends' lexical scorers dropped it first |
| `rerank` | `the user is called marco` (0.549) | the router pulled candidates and judged them |

Useful consequence: with `wants_candidates`, **only the router needs a model**.
The backends are storage.

---

## 010 — A spill across servers writes, verifies, then deletes

**Decision.** `_move` writes to the destination with the **original id**,
verifies the write, and only then deletes from the source.

**Why.** No transaction spans two servers. A crash between the two operations
therefore leaves either a duplicate or a hole, and duplicates are recoverable
while a lost memory is not. Writing with the original id makes the write
idempotent, so a retry cannot duplicate either.

---

## 011 — Client-supplied ids make writes idempotent

**Decision.** If a client supplies `id` and a record with that id exists, the
server returns the existing id and does not create a second record.

**Why.** A network failure between `memory/remember` and its response is
otherwise indistinguishable from a failure before it. This is the only
retry-safety mechanism in A2M, which is why it is normative rather than advisory.

---

## 012 — Unknown parameters are ignored, never rejected

**Decision.** Servers **must** ignore parameters they do not recognise.

**Why.** It is what lets a 0.2 client talk to a 0.1 server.

**How it was found.** The conformance suite caught the reference implementation
violating it — `a2m.py` was rejecting unrecognised parameters with `-32602`. The
bug had been there since the file was written, invisible because the only client
was its own and never sent anything unexpected.

---

## 013 — Ranking is not specified

**Decision.** §5.1 leaves ranking entirely to the implementation. Being bad at it
is conformant.

**Why.** Ranking is where implementations should compete. Fixing it in the
protocol would freeze the least stable part of the field.

**The seam.** `Scorer.relevance(query, candidates) -> {id: 0..1} | None`.
Returning `None` **abstains** — "I cannot judge this query" — which is not the
same as `{}` for "I judged it and nothing matched". The stack needs both:
abstaining falls back to recency and salience, an empty verdict correctly recalls
nothing. Collapsing them would either recall everything on a stopword-only query
or nothing at all.

---

## 014 — bge-m3 as the default embedding model, by measurement

**Decision.** `EMBEDDING_MODEL = "bge-m3"`.

**Why.** Measured, not assumed. Absolute cosine is not comparable across models
— each has its own similarity scale — so the metric is **precision@1**: ten
facts in four deliberately confusable pairs, rendered in five languages, corpus
loaded in one language and queried from all five. 250 queries per model.

| model | size | same-language | cross-lingual |
|---|---|---|---|
| lexical (no model) | — | 76.0% | 7.0% |
| `nomic-embed-text` | 0.27 GB | 82.0% | 24.5% |
| `granite-embedding:278m` | 0.56 GB | 88.0% | 84.5% |
| `paraphrase-multilingual` | 0.56 GB | 94.0% | 89.0% |
| **`bge-m3`** | 1.16 GB | **100.0%** | **99.5%** |

Two assumptions the measurement overturned. The tradeoff is **disk, not speed** —
bge-m3 is no slower than the two mid-sized models (20.2s against 20.4s and 18.6s
for identical work). And the lexical baseline's 7% cross-lingual is almost
entirely proper nouns, which survive translation unchanged.

Reproduce with `python bench_embeddings.py`.

---

## 015 — Per-language stopwords, and a Unicode tokenizer

**Decision.** `STOPWORDS` is keyed by language and detected per text. The
tokenizer is Unicode, with CJK cut into character bigrams.

**Why.** An ASCII tokenizer indexed `Köln` as `['ln']` and `città` as `['citt']`
— which then failed to match the unaccented `citta` anyone might type — while
Cyrillic, Greek, Chinese and Japanese produced **nothing at all**.

A single English stopword list is not merely incomplete elsewhere, it is
destructive: it deleted the Italian verbs `so` (*I know*) and `do` (*I give*) and
the noun `can` (*dog*), while passing all thirteen of `il lo la di che per con
non sono una del ho ha` through as if they carried meaning. That produced
confidently *wrong* answers — `"qual e la citta?"` returned a record about deploy
keys, matched on the article **`la`**.

**The dangerous part was silence.** Two Russian records both indexed to `[]`, the
query did too, the scorer abstained, and recall fell back to recency — returning
everything at plausible scores with no error. `LexicalScorer` now counts records
that hold text but yield no terms, and reports it as `unindexed`.

---

## 016 — MIT

**Decision.** Specification and reference implementation under MIT.

**Why.** A protocol that is expensive to implement does not get implemented. The
work was developed in an AGPL-3.0 repository; carrying that licence here would
have obliged anyone running a hosted memory server to publish their source, which
is the principal deployment shape for this protocol.

---

## 017 — Addressable keys, and no separate namespace

**Decision.** A caller-set `key` addresses a record; writing to an occupied key
replaces it. Hierarchy comes from slash-delimited keys and `key_prefix`, not
from a namespace field.

**Why keys.** `id` identifies a *write*. Nothing in 0.1 identified a *fact*, so
a corrected fact could only be appended alongside the stale one — and ranking
(013) has no way to prefer the newer. `implementing-a2m.md` §4 described exactly
this as unsolved and offered two mediocre answers: delete-then-write, or a
`superseded_by` marker filtered at recall. Upsert beats both — one operation, and
no window in which the fact is missing or doubled.

**Why not namespaces.** The pre-0.1 draft addressed records as
`{app}/{workflow}/{session}/{agent}`. But a slash-delimited key *is* a hierarchy
and `key_prefix` *is* the recursive scope read. A namespace field would add a
second addressing dimension that must be kept consistent with the first, for no
expressive power keys do not already have. `owner` and `session` stay separate
because they are not addressing: one is a scope boundary, the other a lifecycle.

**Cost, accepted.** Key uniqueness is per `owner`, so "unique" depends on who is
asking. That is deliberate — two agents should each be able to hold their own
`user/city`.

---

## 018 — Caller-owned embeddings

**Decision.** A vector supplied by the caller is stored **verbatim**. The server
never generates one for a record that already carries one, and never replaces one
it was given. A mismatched width is refused with `-32008`.

**Why.** This is the rule the pre-0.1 draft got right and that 0.1 initially
lost. A2M exists so agents from different frameworks can share memory, and
vectors from different models are not comparable. A server that re-embeds
whatever it is handed quietly moves every record into its own model's space —
precisely the interoperability failure the protocol is meant to remove.

Refusing a mismatched width rather than accepting it follows from 005: cosine
between a 768- and a 1024-dimensional vector is not a lower-quality answer, it is
not an answer, and a store mixing them ranks nonsense confidently.

**Consequence worth having.** A store needs no embedding model at all to serve
vector search: `EmbeddingScorer(None)` ranks entirely on what callers brought.
The smallest useful A2M server just got smaller.

**What the implementation taught.** The first version persisted vectors only in
the tier that had a vector index, so a caller-supplied embedding on a
working-tier record was silently dropped — "stored verbatim" quietly untrue for
three tiers out of four. Every table now carries the column, and a record's own
vector is searchable wherever it lives. The store still never *generates* one for
working memory; it just no longer discards what it was handed.

---

## 019 — External records, and why they are not a fifth kind

**Decision.** A record **MAY** carry a `uri` pointing at a file, URL or blob,
with an optional `media_type`. `content` keeps its ordinary meaning: the text
that gets indexed. The server **MUST NOT** dereference the `uri`.

**Why not a fifth `kind`.** The pre-0.1 draft listed `external` alongside
working, episodic, semantic and procedural, as a fifth memory *type*. But the
four kinds describe **lifetime and access pattern** — how long a record lives
and whether it is replayed or searched. "Points at a file" describes **content**,
and says nothing about either. A referenced design document is a fact and belongs
in `semantic`; a referenced runbook is a procedure and belongs in `procedural`.
Making it a kind would force the caller to choose between saying *what a record
is for* and saying *where its bytes are*, when those are independent.

So it is a record property with its own capability, and a reference is legal in
any tier.

**Why `content` still matters.** A record that is only a URI is unrecallable by
anything except its address, because there is nothing for a scorer to rank. That
is legal — `by_key` still finds it — but the useful shape is a title, a summary
or an extracted passage in `content`, with `uri` saying where to go for the rest.
This is how a citation works, and it is why the two fields are not redundant.

**Why the server must not fetch.** A memory server's entire job is accepting
arbitrary strings from agents. A server that dereferences those strings is
issuing requests of its own choosing to addresses its callers supplied — a
server-side request forgery primitive, sitting behind whatever network position
and credentials the memory service happens to have. Resolution belongs to the
client, which already has the context to know whether a URI should be fetched at
all. This is the same family of rule as 004: the dangerous default is the
convenient one.

---

## 020 — `events` is deferred to 0.2, and its name reserved

**Decision.** Drop the `events` capability from 0.1. The name is **reserved** in
§9.1 as a candidate for 0.2, and a 0.1 server **MUST NOT** declare it.

**Why.** No transport binding in §8 carries a server-initiated message. HTTP is
one POST endpoint whose connection closes with the response; in-process is a
function call returning one value. Only stdio can physically carry a
notification, and there only because the client already tolerates interleaved
messages while awaiting a response.

So the two notification methods the section defined could not be exercised by
any of the four conformant implementations. Their wire shape was unproven — the
one thing this repository's four-implementation discipline exists to prevent, and
it was true of the *only* section that had no implementation behind it.

**Why reserve rather than simply delete.** A name that is merely absent is a name
someone else's extension can take. Reserving it costs a paragraph and keeps 0.2
free to define `events` without colliding with a vendor capability that got there
first. Reserving is explicitly not a commitment to specify it.

**What 0.2 has to answer, and 0.1 never did.** The section specified payloads and
skipped everything that makes them implementable:

- **Scoping.** Does a `changed` notification respect `owner`? A broadcast tells
  agent B that agent A just wrote. 004 says `owner` is partitioning rather than
  protection — an event channel turns that from *available* into *active*, so
  this must be settled before any code, not after.
- **Subscription.** Opt-in, or does declaring the capability mean everything
  always?
- **Volume.** A consolidation moving ten thousand records: one notification or
  ten thousand? `{"tiers": [...]}` implies coalesced and never said so.
- **Ordering.** May a notification interleave inside a batch response?

**Cost, accepted.** The capability that has no cost is the one nobody depends on
yet, which is precisely the argument for removing it now. A 0.2 client asking for
`events` against a 0.1 server gets `-32003`, which is the correct answer and the
mechanism working as designed (001, 002).

---

## 021 — The bindings follow MCP, not A2M's own taste

**Decision.** Align §8 with what MCP and A2A already do: JSON-RPC 2.0 on every
binding, no batches, no server-initiated requests, stdio framed exactly as MCP
frames it, one POST endpoint over HTTP. Add `Origin` validation, an
`A2M-Protocol-Version` header and a well-known profile.

**Why conform rather than choose.** A2M is not going to win an argument about
transport design, and winning it would not help: an agent runtime that already
speaks MCP has pipe handling, framing, a JSON-RPC client and an HTTP endpoint
already built. Every place A2M differs is a place that runtime needs a second
code path, and the differences bought nothing — the interesting part of A2M is
the memory model, not the envelope.

Checked against the specifications rather than from memory (MCP current revision
`2025-11-25`, draft `2026-07-28`; A2A `v1.0.0`). A2M's stdio binding already
matched MCP's clause for clause, which is the strongest evidence that conforming
costs nothing here.

**Batching, and how the spec was already wrong.** §8.3 said the HTTP body could
be "an array for a batch". `a2m_minimal.py` had never implemented it — it
rejects any non-object with `-32600` — so the specification had a clause one of
its four conformant implementations did not honour, and the conformance suite
never noticed because it had no way to *send* a batch through the client
abstraction. A batch has no `id` of its own for a response to bind to, and
`memory/remember` already takes many records in one call, which is where
batching actually pays. MCP removed batching in `2025-06-18` for the same
reason. Now normative in the other direction, and checked.

That gap is also why `Transport.request_raw` exists: a suite that can only send
well-formed single requests cannot verify that malformed ones are refused.

**Why `Origin` validation is a MUST and not a SHOULD.** A memory server's most
interesting deployment is local, unauthenticated, and holding everything an
agent has ever been told. §6's authentication rules explicitly do not apply
there. Without `Origin` validation, any page the user happens to be browsing can
POST to `127.0.0.1:8778` and read the lot. Binding loopback does not help — that
is where the browser already is. This is the same family as 004 and 019: the
dangerous default is the convenient one.

**Why the version header is a SHOULD and its absence is not an error.** It
exists so a gateway can route or reject without parsing a body. Making it
mandatory would break every hand-written `curl` against a local server for no
protocol benefit, since `memory/describe` is still where negotiation happens.
Present-and-wrong is refused; absent is fine.

**Where A2M deliberately still differs.** MCP's `2026-07-28` draft drops the
`initialize` handshake in favour of per-request version metadata plus an
optional `server/discover`. A2M keeps `memory/describe`-first (§2). Capability
negotiation is load-bearing here in a way it is not for MCP — a client must know
before it calls whether `keys`, `embeddings` or `tiers` exist — and one
negotiation point is cheaper than repeating capabilities in the `_meta` of every
request. `memory/describe` already *is* `server/discover`. Revisit if MCP's
draft becomes current and the ecosystem follows it.

**Well-known profile, borrowed from A2A.** A2A publishes an Agent Card at
`/.well-known/agent-card.json`; A2M publishes its `describe` result at
`/.well-known/a2m-server.json`. One handler, and a server becomes discoverable
by a directory or an operator holding no A2M client at all. Advisory and
possibly stale by construction — a client that needs the truth calls
`memory/describe`.

---

## 022 — A second language, and no build step

**Decision.** [a2m_minimal.ts](a2m_minimal.ts) ports the minimal server to
TypeScript, run directly by `node --experimental-strip-types`. No `package.json`,
no `tsconfig.json`, no dependencies, no compiled output.

**Why a second language at all.** "Transport-agnostic" and "language-independent"
were assertions. Five implementations in one language demonstrate that Python
agrees with itself; the conformance suite already speaks only the protocol, so
validating a non-Python server cost nothing but the server. It now passes the
same unmodified suite, 34/34, exactly as `a2m_minimal.py` does.

**Why no build step.** A TypeScript project with a compile stage would have made
this the first thing in the repository that has to be *built* before it can be
run, and the second that needs a package manager. Node strips the types and runs
the file, which keeps the promise the Python side makes: clone it, run it.

**What the port actually caught.** Two rules that a same-language port would have
carried across for free, and that a real implementer will meet:

- **Timestamps.** JavaScript's instinct is `Date.now()` — a millisecond epoch
  integer, which is exactly the ambiguity 003 removed. `toISOString()` happens to
  be the right shape, but nothing warns you when it is not.
- **Id types.** JSON-RPC ids may be strings or numbers, and JavaScript coerces
  between them without complaint. An id normalised on the way through is a
  response a client waits for forever. Echoing it untouched is a rule the Python
  implementation never had to think about.

Neither is exotic. Both are the kind of thing a specification is *for*, and
neither would have surfaced without leaving the reference language.

**Consequence.** The most useful demonstration in the repository is now a pair:
`a2m_client.py` against `a2m_minimal.ts` is a Python client and a TypeScript
server sharing no code, neither written against the other.

---

## 023 — Adapters, and the silo they nearly rebuilt

**Decision.** `adapters/langchain.py` and `adapters/agno.py`, each importing one
framework, neither imported by anything else. `examples/cross_framework.py` runs
both against one store.

**Why they belong here after all.** The pre-0.1 draft shipped four adapters and a
cross-framework example; the 0.1 reset dropped them, and the README went on
claiming that frameworks cannot share memory without anything demonstrating that
A2M fixes it. A protocol that only its own reference implementation speaks is not
a protocol (001), and the same argument applies to the frameworks it is supposed
to join. The stdlib-only property is preserved by isolation, not by absence:
`adapters/__init__.py` imports no framework, so a checkout without either still
runs every test, every conformance target and both demos.

**The mapping is thin because the protocol did the work.** LangChain splits chat
history from retrieval, which is exactly the `timeline`/`recall` split of 007 —
so each side maps onto one method and neither emulates the other. Agno wants
`upsert`, which is `keys` (017) doing precisely what it was added for: writing an
occupied key replaces, so a corrected document does not sit beside the stale one.

**What the adapters caught.** Two things, both real:

- **A namespace scopes reads as well as writes.** `A2MVectorDb` originally
  filtered every search by its own namespace, so an Agno knowledge base could
  only ever find documents Agno had written. That is a private store with extra
  steps — the exact failure A2M exists to remove, reintroduced one layer up.
  `namespace=None` now means "write mine, search everything", and the
  cross-framework example uses it.
- **A conversation is not knowledge yet.** The first version of the example
  asserted that Agno could find what LangChain had just written. It could not,
  correctly: those turns were in working memory, which is replayed and never
  searched (007). The example now asserts the *absence* first, closes the
  session, and finds it afterwards — which demonstrates why the tiers exist
  rather than working around them.

The second one is the better argument for adapters existing at all. A rule that
survives a conformance suite can still be a rule nobody understands until two
real frameworks meet on top of it.

---

## 024 — A second engine, and the seam it proved was not real yet

**Decision.** [a2m_postgres.py](a2m_postgres.py) serves the same tier model on
PostgreSQL and pgvector, reusing `TieredMemoryStack` unchanged. To make that
possible, `a2m_store.py` was split: `TieredMemoryStack` holds every tier decision
and touches no SQL, `SqliteMemoryStack` holds the SQLite wiring.

**Why a second engine.** `a2m_store.py`'s docstring claimed that "swapping
DurableStore for Postgres means implementing that class, not rewriting the
server". That was an assertion. One backend proves a store works; two prove the
*seam* does, and the seam is what the file exists to demonstrate.

**It was not true when it was written.** The stack reached past `TierStore` and
wrote `INSERT INTO durable_vec` directly, in two places, guarded by
`isinstance(store, DurableStore)` and a `self.vec` flag that only sqlite-vec
could set. A different engine could not have been dropped underneath it, because
the tier logic knew which index it was talking to. The fix moves indexing into
the store that owns it, and gives `TierStore.knn` a third answer:

- a dict — "I have an index, here is what it ranked",
- `{}` — "I consulted it and nothing matched",
- `None` — "I have no index; rank these another way".

That is the same abstain/empty distinction as the `Scorer` seam in 013, and for
the same reason: collapsing the last two makes a store with no vectors return
either everything or nothing.

**What the port cost.** Three `TierStore` subclasses and one `TieredMemoryStack`
subclass. No tier logic, and nothing at all in `a2m.py`, which does not know
either engine exists. 78/78 on stdio, 82/82 over HTTP.

**What differed, and mattered.** Not the dialect — the semantics underneath it:

- **`INSERT OR IGNORE` is `ON CONFLICT (id) DO NOTHING`.** This is not a
  translation detail: it is what makes a client-supplied id idempotent (011), the
  only retry-safety in the protocol. Written as a plain insert, a network retry
  becomes a duplicate.
- **pgvector's `<=>` is cosine distance, so similarity is `1 - d`;** sqlite-vec
  reports L2 over normalised vectors and needs `1 - d²/2`. Both must land in
  `0..1` or the blend with recency and salience silently changes meaning. Two
  engines, two conversions, one range — which is exactly the argument in 005 for
  `score` being ranking information and nothing more.
- **A vector column's width is fixed at DDL time,** so the index is created
  lazily on the first embedding, as on the SQLite side. A store configured
  without an embedder never creates one, which keeps `embed=None` a working
  configuration rather than a degraded one.

**Where the engines legitimately disagree.** SQLite keeps procedural memory as
files on disk, so a runbook can be reviewed and version-controlled like the code
it describes. A server reachable over a network has no such disk to share, so the
PostgreSQL store keeps the text in the row. The tier's *meaning* is identical —
written deliberately, never spilled into (006) — and that is the part the
protocol constrains. Storage is where implementations are supposed to differ.
