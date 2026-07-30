# Blog series — outline

Internal. [DECISIONS.md](../DECISIONS.md) is a pre-written blog series: each
entry is already an argument with evidence, which is the hard part. What is left
is turning "why we chose X" into "here is a mistake you are probably making",
which is the same content aimed outward.

**Cadence:** one post every week or two after launch. Sustained low-effort
presence beats a second big push — the goal is that someone searching "agent
memory" in three months finds A2M through a post about their actual problem, not
through the launch.

**Shape, every time:** a concrete failure the reader recognises → why the
obvious fix does not work → what A2M does → one paragraph linking the spec.
Never lead with the protocol. Nobody searches for a protocol; they search for
the bug it prevents.

**Where:** Marco's own blog if there is one, otherwise dev.to and cross-posted
to Hashnode, with links from the relevant subreddit only when the post genuinely
answers a question being asked there.

---

### 1. Your agent's memory is lying to you with total confidence

**From DECISION 017 (keys).** The strongest opener, because everyone has hit it.

The user moves city. You store the new fact. Recall now returns both, ranked by
relevance, and relevance has no opinion about which one is *true*. Append-only
memory does not become wrong slowly — it becomes wrong immediately and stays
plausible.

Covers: why delete-then-write leaves a window and `superseded_by` leaves a
filter everyone forgets; why upsert is the only honest answer; why `id`
identifies a write while `key` addresses a fact. Ends on `revision`.

Demo: [examples/rag_ingest.py](../examples/rag_ingest.py) §2 — re-ingesting a
revised document and watching the stale passage *disappear* rather than compete.

---

### 2. Never compare two vector search scores

**From DECISION 005.** The most broadly useful post and the least A2M-specific,
which makes it the best candidate to travel.

`min_score=0.7` is meaningless. Different rankers occupy entirely different
ranges; a model that rates everything 0.9 may discriminate better than one
spreading across 0..1. Swapping an embedding model silently changes what your
threshold means, and nothing fails loudly.

Covers: why A2M says scores are ranking information only, normatively; the
consequence that a federating router cannot merge by score — and what it does
instead (reciprocal rank fusion, discarding the numbers and keeping the order).

---

### 3. Don't put your chat transcript in a vector database

**From DECISION 007 and DECISION 023.** The most counterintuitive post, and the
one most likely to start an argument, which is useful.

A transcript is *replayed*, not searched. Ranking it by relevance destroys the
thing that made it a transcript. Worse, splitting an assistant `tool_calls`
message from the tool results answering it produces a message list most
providers reject outright.

Covers: `timeline` versus `recall` as separate methods neither of which may be
implemented in terms of the other; `group`; why working memory is never embedded
(it turns over before anything searches it); per-session capacity, so a busy
conversation cannot evict a quiet one's context by talking more.

Demo: the moment in [examples/cross_framework.py](../examples/cross_framework.py)
where Agno *correctly cannot* find what LangChain just wrote — until the session
closes.

---

### 4. Your memory server should never fetch that URL

**From DECISION 019.** Short, sharp, security-flavoured — the one most likely to
be shared by people who do not care about agent memory at all.

A memory server's whole job is accepting arbitrary strings from agents. A server
that dereferences those strings is issuing requests of its own choosing to
addresses its callers supplied, from whatever network position and with whatever
credentials the memory service happens to hold. That is SSRF with extra steps.

Covers: why `external` stores a `uri` and normatively forbids fetching it; why
resolution belongs to the client; the same family of reasoning behind `owner`
not being a security boundary and `Origin` validation being a MUST.

---

### 5. How to ship events without server-initiated requests

**From DECISION 026.** The most technical post, aimed at protocol designers
rather than agent builders.

Not every transport can push. If a capability only works on one binding, it is
not really in the protocol. So: a cursor that any request/response channel can
carry, with push as an optional mirror where the connection allows it — and push
and poll reading one log, so a client that misses a notification can always
catch up.

Covers: opaque cursors and why they carry a salt (a cursor from another server
must fail loudly, not be misread as a position); `reset`; coalescing, so a
consolidation moving ten thousand records is one event; why the log is bounded
and in-memory.

---

### 6. Two implementations, or you have not written a specification

**From DECISIONS 022 and 025**, and the methodological argument for the whole
project. Save it for when there is at least one external implementation to cite
— it lands very differently as "someone else did it" than as "I did it twice".

One implementation proves your document describes the program you already wrote.
Writing a second from the document alone found a real bug in the first: the
reference was rejecting unrecognised parameters, breaking the exact
forward-compatibility rule that lets a newer client talk to an older server.

Covers: a conformance suite that never imports what it tests; what changing
language caught that changing storage did not (epoch timestamps, id coercion);
why the file boundary between tier logic and SQL was the thing that exposed an
`isinstance` check silently failing for every PostgreSQL tier.

---

### 7. RAG is a special case of agent memory

**From DECISION 001 and implementing-a2m.md §10.** The best post for people who
do not think they have a memory problem.

RAG and agent memory differ in exactly one thing: who wrote the corpus. A
document has no working phase — nobody said it in a conversation — so it enters
at `semantic` and never spills. That is the whole difference, and it means
classic RAG is A2M with one tier, read-only.

Covers: exposing an existing RAG stack over A2M in an afternoon; the corpus tier
so ingested and learnt knowledge cannot evict each other; why "forget the
conversation without forgetting the manual" is the requirement that forces them
apart.

---

## Two posts to write only if asked

**"A2M versus Mem0/Zep/Letta."** Tempting and mostly a trap: it invites a
comparison A2M loses on maturity and wins on a dimension nobody asked about.
Write it only in response to a direct question, and frame it as layers rather
than alternatives — they are products, A2M is the interface they could serve.

**"Why not MCP."** Already covered by DECISION 027 and the README section. A
standalone post reads as picking a fight with a much larger project. If the
question keeps coming up, the answer is a better README section, not a post.
