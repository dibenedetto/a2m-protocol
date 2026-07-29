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
