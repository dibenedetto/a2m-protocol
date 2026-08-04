# Implementing A2M: what each tier means, and where it should actually live

**Status: non-normative.** [a2m-0.1.md](a2m-0.1.md) is the specification and
deliberately says nothing about storage — §5.1 leaves ranking entirely to
implementations, and nothing mandates a database of any kind. This document is
engineering advice about the four `kind` values, for people building a server.
None of it is binding.

Two working implementations of everything below ship with this repository:
[implementations/store_sqlite.py](../implementations/store_sqlite.py) puts each tier in the store this document argues
for, and [implementations/server_federated.py](../implementations/server_federated.py) federates the same stack across one
process per tier. Both pass the conformance suite unchanged.

---

## 1. The matrix

| Kind | Means | Written by | Read as | Turnover | Volume | Natural store |
|---|---|---|---|---|---|---|
| `working` | the live transcript | every turn | **replay**, chronological, in full | minutes | 10¹–10² | a list in process, or Redis |
| `episodic` | what happened | spilling from `working` | **search**, relevance + time filter | days–months | 10³–10⁷ | relational **+** vector (pgvector) |
| `semantic` | what is true | promotion, or LLM distillation | **search**, relevance | years | 10²–10⁵ | vector store, or a fact table, or a graph |
| `procedural` | how to do things | deliberate write only | **search**, at task start | permanent | 10⁰–10² | files in version control |

The single most important row is the first, and it is the one most often got
wrong.

---

## 2. `working` — replayed, never searched

**Meaning.** What is in the context window right now. In this framework the
agent's transcript *is* this tier, which is why eviction from it is what bounds
the context window.

**Data flow.**

```
user turn ──▶ working ──▶ (over capacity) ──▶ episodic
                 │
                 └──▶ read in full, in order, before every model call
```

**Bounded per conversation.** Capacity applies *within* a session, not across the
tier — otherwise two concurrent conversations compete for the same slots and the
busier one evicts the quieter one's context purely by talking more. This is the
only tier where that is true: what a team *knows* is one pool regardless of which
chat it was said in.

**Access shape.** Written 2–6 times per turn (user message, assistant message,
any tool calls and their results). Read *entirely*, in creation order, before
every single model call. It is on the critical path of every turn, so it has the
tightest latency budget of the four and the smallest volume.

**Where it lives.** A list in the agent's process. Redis or any KV store if the
agent must be stateless between HTTP requests. That is genuinely all.

**The mistake.** Putting `working` in a vector database. You never ask working
memory *"what is relevant?"* — you ask it *"what happened, in order?"* It is
served by `memory/timeline`, not `memory/recall`. Ranking it is not merely
wasteful, it is wrong: reordering a transcript by relevance destroys it, and
splitting an assistant `tool_calls` message from its results produces a
transcript most providers reject outright.

**The corollary that saves money.** Do not embed on write. Working memory turns
over in minutes and is usually evicted having never been searched once, so
embedding every message as it arrives pays for vectors nobody reads. Embed at
*spill* time, when a record earns a place in episodic. (`EmbeddingScorer` in this
framework embeds lazily at recall, and the agent's recall skips the working tier
— so working records are never embedded at all.)

---

## 3. `episodic` — the tier that is actually RAG

**Meaning.** Specific things that happened, still carrying their context: when,
who, in what conversation.

**Data flow.**

```
working ──spill──▶ episodic ──promote (recalled 3×)──▶ semantic
                       │
                       └──▶ recall(query, where={session, owner, after})
```

**Access shape.** Written only on spill, so writes are bursty and rare relative
to turns. Read by relevance on most turns. This is the tier that grows without
limit, and the one whose size decides your infrastructure.

**Where it lives.** Relational storage *with* a vector column — Postgres with
pgvector is the canonical shape — rather than a pure vector store.

**Why not a pure vector database.** Episodic records are not free-floating text.
They have structure the queries need: `created_at`, `owner`, `group`, `session`,
`role`. Real recalls are conjunctive — *"what did **this user** say about
deployments **last week**"* — and that is a similarity search **and** a
metadata-filtered range scan. Vector stores bolt metadata filtering on; relational
engines bolt vectors on. For episodic, the second bolt is the cheaper one, because
`group` integrity, ownership and time ranges are relational problems.

**Retention is a product decision, not a technical one.** Episodic is the tier
that will dominate storage cost. `memory/consolidate` is where you make that
choice: drop, or distil into semantic and then drop.

---

## 4. `semantic` — small, precious, and mutable in a way episodic is not

**Meaning.** What is true, stripped of the occasion on which it was learnt.
`"the user is called Marco"` rather than `"[2026-07-28 10:15] user: mi chiamo
marco"`.

**Data flow.**

```
episodic ──promote (earned by repeated recall)──▶ semantic
episodic ──distil (llm_consolidator on spill)───▶ semantic
```

**Access shape.** Written rarely, read on most turns, small enough that an
exhaustive scan is often faster than an index. High value per record.

**Where it lives.** A vector store is the default. At the low end a plain fact
table with in-memory cosine is entirely adequate — at 10³ records you do not need
an index. A knowledge graph earns its keep only if you query *relations*
(`Marco → works_on → compilers`) rather than just retrieving statements.

**The problem episodic does not have: supersession.** Episodic records are
immutable history — *"on Tuesday the user said they lived in Bologna"* stays true
forever. Semantic records are claims about the present, and claims expire. When
the user moves to Milan, `"the user lives in Bologna"` does not become a false
memory of a true event; it becomes a **wrong fact that will be recalled with high
confidence**.

A2M gives you two ways to handle it and does not choose for you:

- `memory/forget` the superseded record, then `memory/remember` the new one.
- Keep both and mark the old one in `metadata` (`{"superseded_by": "<id>"}`),
  filtering it out with `where` at recall.

The second is better when you need an audit trail of what the agent believed and
when. The first is better for everything else. Doing *neither* is the most common
way a memory system starts confidently lying.

---

## 5. `procedural` — code-adjacent, so treat it like code

**Meaning.** How to do things. In this framework, loading a skill writes here.

**Data flow.** Nothing spills into procedural and nothing expires out of it. It
is written deliberately — `memory/remember` with an explicit tier, or
`memory/promote` — which is exactly why it is not the fourth link in the spill
chain. A fact does not decay into a procedure.

**Access shape.** Written very rarely, read at task start, tiny.

**Where it lives.** Files on disk, under version control, with a small index for
`recall`. It does not need a database.

**Why version control specifically.** A procedure is closer to code than to data.
You want to see what your agent learnt to do, diff it against last week, review
it before it takes effect, and revert it when it turns out to be wrong. Rows in a
table give you none of that. This is also the tier where a bad write does the most
damage, because a wrong *fact* produces one wrong answer while a wrong *procedure*
produces wrong answers indefinitely.

---

## 6. Which method touches which tier

| Method | `working` | `episodic` | `semantic` | `procedural` |
|---|---|---|---|---|
| `memory/remember` | ●●● every turn | ○ via spill | ○ via distil | ● deliberate |
| `memory/timeline` | ●●● every turn | ○ rarely | — | — |
| `memory/recall` | ○ never, by design | ●●● | ●●● | ●● at task start |
| `memory/forget` | ○ via eviction | ● retention | ● supersession | ● unlearning |
| `memory/promote` | — | ●● earned | ● → procedural | — |
| `memory/consolidate` | ●●● spills out | ●● spills/distils | ○ | — |
| `memory/reinforce` | — | ●● drives promotion | ● | ● |

`●●●` hot path · `●●` regular · `●` occasional · `○` rare or deliberately absent

---

## 7. Common architectures, as A2M

Every one of these is a conformant A2M server. The point of the core-plus-
capabilities design is that they are all reachable without forcing the small ones
to pretend to be the large ones.

| Architecture | Tiers | Capabilities | Notes |
|---|---|---|---|
| Prompt-only chatbot | `working` | `core` | nothing persisted; `recall` may return nothing |
| **Classic RAG** | `semantic` | `core` | corpus pre-loaded; `remember` returns `-32004 READ_ONLY` |
| Chat with history | `working` `episodic` | `core` `tiers` | the common case |
| Agent with skills | `+ procedural` | `core` `tiers` | skills are procedural memory |
| Learning agent | all four | `core` `tiers` `salience` | `reinforce` drives promotion |
| Multi-agent team | all four | `+ scopes` | private transcripts, pooled knowledge |

**RAG is a degenerate case of A2M**: one tier, read-only, `recall` only. That is
worth stating plainly, because it means an existing RAG stack can be exposed over
A2M in an afternoon — implement `describe`, `recall`, and return `READ_ONLY` from
`remember` and `forget` — and every A2M client works against it immediately. It
just declares less than a full stack does.

---

## 8. Cost per turn

For a stack with all four tiers, per agent turn:

| Tier | Writes | Reads | Embeddings |
|---|---|---|---|
| `working` | 2–6 | 1 (full replay) | **0** |
| `episodic` | 0, or a burst on spill | 1 relevance query | on spill only |
| `semantic` | ~0 | 1 relevance query | on write only |
| `procedural` | ~0 | 1, at task start | on write only |

Two consequences worth designing around:

**Embeddings are a spill-time cost, not a turn-time cost.** The write volume that
looks alarming — every message, every turn — never reaches the embedding model,
because it dies in working memory first.

**Recall latency is on the critical path; consolidation is not.** Consolidation
can move records, call an LLM to distil them, and rewrite tiers, all off the hot
path. In this framework that is why `MemoryStack` takes two locks: the record lock
is held in short bursts, while `consolidate_fn` — which may be an LLM call — runs
holding nothing.

---

## 9. Where a database gets injected

There are **five** seams, at different levels, and picking the wrong one is the
usual way this gets messy. They are not alternatives to each other -- a real
deployment uses several at once.

| # | Seam | Type | What you inject | Purpose |
|---|---|---|---|---|
| 1 | `Scorer` | ranking | a vector index | decide *relevance* |
| 2 | `TierStore` | storage | relational, KV, files | hold *one tier's* bytes |
| 3 | the stack | storage | a whole engine | replace *all* storage |
| 4 | an A2M backend | both | another A2M server | a different technology *per tier* |
| 5 | `consolidate_fn` | policy | an LLM | decide what is *worth* keeping |

### 9.1 `Scorer` — a vector database as a ranker

```python
MemoryStack(scorer=EmbeddingScorer(my_vector_db_search))
MemoryStack(scorer=HybridScorer([(LexicalScorer(), 0.4), (EmbeddingScorer(...), 0.6)]))
```

Injected here, a vector database is **only an index**. The records still live in
the stack; the vector store answers "which ids, in what order". This is the
lightest injection and usually the first one worth making, because it changes
recall quality without touching where anything is stored.

The seam is deliberately narrow: `relevance(query, candidates) -> {id: 0..1}`.
It knows nothing about tiers, recency or salience, because those depend on where
a record lives and the stack blends them in afterwards.

### 9.2 `TierStore` — relational and vector as storage, per tier

```python
class RedisWorkingStore(TierStore):     # working: replayed, never searched
class PostgresDurableStore(TierStore):  # episodic: filter *then* rank
class GitProceduralStore(TierStore):    # procedural: diffable, revertible
```

This is the seam [implementations/store_sqlite.py](../implementations/store_sqlite.py) exists to demonstrate, and the
one the matrix in §1 is really about. Each tier owns its own table and its own
idea of what reading means.

**Relational belongs at `episodic`.** Episodic records are not free-floating
text: real recalls are conjunctive -- *"what did **this user** say about
deployments **last week**"* -- which is a similarity search **and** a
metadata-filtered range scan. Vector stores bolt metadata filtering on;
relational engines bolt vectors on. For episodic the second bolt is cheaper,
because `group` integrity, ownership, sessions and time ranges are relational
problems. Postgres with pgvector is the canonical shape. The shipped sample uses
SQLite with `sqlite-vec` for exactly the same reason, at a smaller size.

**Filter during the search, not after it.** In `DurableStore.knn`, `owner` is a
vec0 *partition key* and `tier` an auxiliary column, so a scoped search never
over-fetches and discards. A post-hoc join would ask for `k` neighbours and then
throw most away, silently returning fewer results than asked for.

**Do not inject a vector store at `working`.** It is replayed, not searched (§2).

### 9.3 The stack — replacing the whole engine

```python
MemoryServer(stack=SqliteMemoryStack("memory.db"))
```

`MemoryServer` was written against a dict-backed stack and drives
`SqliteMemoryStack` unchanged. Anything presenting the same interface --
`remember`, `recall`, `timeline`, `forget`, `promote`, `consolidate`,
`describe`, `count` -- works. Use this when your storage does not decompose into
per-tier tables at all.

### 9.4 An A2M backend — a different technology per tier

```python
backends = {
	"working"   : connect_http("http://redis-memory/"),
	"episodic"  : connect_http("http://pgvector-memory/"),
	"semantic"  : connect_stdio([...]),
	"procedural": connect_local(...),
}
MemoryRouter(backends, merge=make_merge("rerank"))
```

The strongest form: each tier is a whole A2M server, in its own process, on its
own host, with its own database. The router owns the *policy* -- capacity,
spilling, promotion -- because only it can see more than one tier. The backends
own the *bytes*.

Two costs are real and unavoidable here. Scores from different backends are not
comparable (spec §5.3), so a merge strategy must fuse *rankings* or re-rank
everything itself. And a spill becomes a distributed write with no transaction
spanning it, so the router writes with the original id, verifies, and only then
deletes.

### 9.5 `consolidate_fn` — the LLM as a policy, not a lookup

```python
MemoryStack(consolidate_fn=llm_consolidator(model))
```

Runs when a record spills, never in the recall path. This is where a model
decides what a conversation *meant* rather than what it *said*.

---

## 10. Where RAG enters

**RAG is already here.** Retrieval-augmented generation is `memory/recall`
followed by putting the results in the prompt, which is precisely what
`Agent._recall_into` does before every model call. There is no separate RAG
component to add, and adding one would duplicate the stack.

What differs is **who writes the corpus**:

| | RAG | agent memory |
|---|---|---|
| written by | ingestion, ahead of time | the conversation, as it happens |
| enters at | `semantic` directly | `working`, then spills down |
| lifecycle | static until re-ingested | spills, promotes, is forgotten |
| chunks | a chunker splits documents | a turn is already a record |

A document has no `working` phase -- nobody said it in a conversation -- so it
enters at `semantic`, which is exactly why **classic RAG is the degenerate case
of A2M**: one tier, read-only, `recall` only. That is also why an existing RAG
stack can be exposed over A2M in an afternoon: implement `describe` and
`recall`, return `-32004 READ_ONLY` from `remember` and `forget`, and every A2M
client works against it.

**Where a RAG system's three moving parts end up.** A2M owns one of them and
deliberately declines the other two:

| | who owns it | what A2M provides |
|---|---|---|
| **chunker** | the ingester, entirely | nothing — but `group` keeps a document's chunks together through eviction (§3.4) |
| **embedder** | the caller, or the server | a vector is stored **verbatim** and never regenerated (§3.7); `describe` names the model |
| **comparer** | the server's index | `describe` names the `metric`; it is declared, not selected, and a mismatch is invisible |

The asymmetry is worth internalising before you bring your own vectors: you can
own the embedder completely and cannot own the comparer at all. Check `metric`,
because nothing will check it for you.

Three ways to bring a corpus in, in increasing order of separation:

**Load it into `semantic`.** Simplest. Documents become ordinary records and
participate in normal recall alongside what the agent learnt itself. Give each
document's chunks a shared `group`, which is what A2M already uses to keep
things that must not be separated together -- chunks of one document are exactly
that.

```python
for chunk in chunks_of(document):
	memory.remember(chunk, tier="semantic", group=document_id,
	                metadata={"source": path, "page": n})
```

**Give it its own tier.** A fifth tier of kind `semantic`, unbounded, that
nothing spills into and consolidation never touches -- so ingested knowledge and
learnt knowledge cannot evict each other, and `recall(tier="corpus")` can ask
only one of them.

```python
MemoryTier("corpus", kind="semantic", capacity=0)
```

**Federate it.** Wrap the existing RAG stack as an A2M server and mount it as a
tier in the router (§9.4). The corpus keeps its own ingestion pipeline, its own
database and its own release cycle; the agent sees one memory.

The distinction worth keeping is that **RAG retrieves what someone put there and
memory retrieves what happened**. Mixing them in one tier is fine until you need
to forget a conversation without forgetting the manual.

---

## 11. Watching a store — `events` in practice

The capability is two delivery modes over one log, and the log is the part to
get right.

**Implement the cursor first; push is a mirror.** Every mutation appends one
event to a bounded, in-memory log; `memory/events` walks it, and push — where
the transport can carry a notification at all — simply forwards each append to
the one subscribed connection. If an event can be pushed but not polled, a
client that misses the notification has no way back; the reference emits
nothing that did not go through the log first.

**The log is server state, not storage state.** It does not survive a restart,
and that is correct: a cursor is valid only on the server that issued it, and
a client holding a stale cursor gets `reset: true` plus the oldest retained
events — the signal to re-read whatever it was tracking. Persisting the log
buys almost nothing (the client must handle `reset` anyway, for retention) and
costs a schema, a growth policy and a vacuum job in every backend.

**Bound it.** The reference retains 1024 events. An abandoned cursor then
costs nothing, which is what makes `events` safe to declare unconditionally —
an unbounded log would make the capability affordable only to servers with
garbage collection.

**Coalesce ruthlessly.** One `consolidated` event per reorganisation, carrying
counts; one `forgotten` event per forget, carrying a count. Only `written` is
per-record. A watcher that needs the details re-reads through `recall` or
`timeline` — the events say *that* and *when*, not *what*.

**A consumer is a loop, not a callback.** The robust shape on any transport:

	position = events()["cursor"]              # subscribe from now
	loop:
		reply    = events(cursor=position)
		handle(reply["events"])
		if reply.get("reset"):  re-read tracked state
		position = reply["cursor"]

Push, where available, only changes how long the loop sleeps.


## 12. Anti-patterns

**One store for all four tiers.** They have opposing access shapes: working is
replay-everything with the tightest latency, episodic is filtered similarity at
the largest volume, procedural is a handful of documents you want to diff. A
single backend serves one of them well and the others badly.

**Vector search over working memory.** Covered in §2. You are asking the wrong
question of that tier.

**Embedding on write.** Pays for vectors on records that will be evicted unread.

**Spilling into `procedural`.** Procedures are authored, not aged into. If your
consolidation pushes stale facts into procedural, your agent will start following
instructions nobody wrote.

**Treating `score` as a probability.** Spec §5.3: scores are ranking information
only, never comparable across servers or calls. A `min_score` threshold tuned
against one backend is meaningless against another — and swapping the ranker,
which A2M explicitly permits, silently changes what that threshold means.

**Letting semantic memory accumulate contradictions.** §4. This is the failure
that turns a memory system from useful into actively harmful, because the wrong
fact is recalled with exactly the same confidence as the right one.

**Rebuilding state from events alone.** §11: events are notification, not
replication. A consumer that reconstructs the store from `written` events has
built a second store that silently diverges on the first `reset`. Watch with
events; read with `recall`, `timeline` and `fetch`.
