# Seeded issues

Internal. Bodies for issues **Marco creates** on GitHub after launch — arriving
attention needs somewhere to land, and an empty issue tracker converts nobody.

Create them in this order. The first three carry the `good first issue` label
and are the ones that matter; the rest are for people looking for something
larger.

Labels to create first: `good first issue`, `help wanted`, `implementation`,
`adapter`, `spec`, `store`.

---

## 1. Write an A2M server in your language

**Labels:** `good first issue`, `help wanted`, `implementation`

> This is the contribution the project most needs, and it is more approachable
> than it sounds.
>
> A2M's core is five methods. A conformant server can declare `core` and nothing
> else, answering `-32003 CAPABILITY_NOT_SUPPORTED` for everything it does not
> implement — that is not a lesser server, it is what the capability model is
> for. [implementations/server_minimal.py](../implementations/server_minimal.py)
> does exactly that in 470 lines and imports nothing from this repository;
> [server_minimal.ts](../implementations/server_minimal.ts) is the same thing in
> TypeScript with no dependencies and no build step.
>
> Languages nobody has claimed yet: **Go, Rust, Ruby, Java, C#, Elixir, PHP,
> Swift, Zig.** Comment to claim one.
>
> Check your work with the suite, which speaks only the protocol and never
> imports what it tests:
>
> ```
> python -m tools.conformance --stdio <your server command>
> ```
>
> Then send a PR adding a row to the conformance table in the README. Your code
> can live in your own repository — we only need the link.
>
> The five rules that catch every implementation are listed in
> [CONTRIBUTING.md](../CONTRIBUTING.md), and Appendix B of the spec is the same
> list as a checklist.

---

## 2. Adapter: LlamaIndex

**Labels:** `good first issue`, `help wanted`, `adapter`

> LlamaIndex has both a chat store and a vector store interface, which maps onto
> A2M unusually cleanly: the chat store is `memory/timeline` (replay) and the
> vector store is `memory/recall` (search) — the same split
> [adapters/langchain.py](../implementations/adapters/langchain.py) already
> exercises.
>
> Four adapters exist to copy from, in
> [implementations/adapters/](../implementations/adapters/). The rules are:
> import exactly one framework, have nothing import you, and keep
> `adapters/__init__.py` free of framework imports so a checkout without
> LlamaIndex still runs every test.
>
> Two things to get right, both of which earlier adapters got wrong first:
>
> - **A namespace must scope writes without scoping reads.** The Agno adapter
>   originally filtered every search to its own namespace, which made it a
>   private store with extra steps — the exact failure A2M removes.
> - **Pass embeddings through verbatim.** If LlamaIndex has already produced a
>   vector, it is stored as-is and never regenerated (spec §3.7).

---

## 3. Adapter: Semantic Kernel, Haystack, Pydantic AI or Mastra

**Labels:** `good first issue`, `help wanted`, `adapter`

> Same shape as the LlamaIndex issue — pick whichever you actually use. Mastra
> is TypeScript, so that one also needs a TS client, which makes it the most
> interesting of the four.
>
> Comment to claim one so two people do not write the same adapter.

---

## 4. A store on Redis

**Labels:** `help wanted`, `store`

> The two SQL stores share [store.py](../implementations/store.py), which holds
> the tier logic and contains **no SQL at all**. A Redis store would be the
> first backend to exercise that seam from outside the relational world, which
> is the point: two implementations sharing a base class prove nothing if either
> can quietly reach around it.
>
> Interesting because the access shapes differ per tier — working memory is a
> list, semantic memory wants a vector index (RediSearch), and procedural memory
> is a handful of documents. If the tier model is honest, that should be an
> advantage rather than a problem.
>
> `TierStore.EMBEDS` is how the stack decides what gets a vector. Ask the store,
> never its class.

---

## 5. Expose an existing RAG stack over A2M

**Labels:** `help wanted`, `implementation`

> Classic RAG is the degenerate case of A2M: one tier, read-only, `recall` only.
> Implement `memory/describe` and `memory/recall`, return `-32004 READ_ONLY`
> from `memory/remember` and `memory/forget`, and every A2M client works against
> your corpus immediately.
>
> A façade over Chroma, Qdrant, Weaviate, pgvector or an in-house pipeline would
> demonstrate the claim in DECISION 001 with something running, and it should
> genuinely take an afternoon. If it takes longer than that, the specification
> has a problem and the issue is worth more as a bug report.

---

## 6. Spec: is `recall`'s filter surface complete?

**Labels:** `spec`, `help wanted`

> `memory/recall` takes `query`, `tier`, `limit`, `where`, `min_score`, `owner`,
> `embedding`, `key_prefix` and `embeddings`. `where` is a conjunction of
> equality tests with array values meaning "any of".
>
> Question for anyone who has built retrieval in anger: **what is missing that
> you would reach for on day one?** Date ranges are the obvious candidate.
> Negation is another.
>
> This is deliberately asked before 1.0. `recall` is the most-used method and
> the hardest to extend without a version bump, so the cheap moment to get it
> right is now. Answers that describe a real use case are worth more than
> answers that describe a feature.

---

## 7. Documentation: a quickstart for someone who has never seen A2M

**Labels:** `good first issue`, `help wanted`

> The README argues for the protocol and the spec defines it. Neither is a
> tutorial.
>
> What is missing: a short guide that takes someone from `pip install
> a2m-protocol` to an agent that remembers something across two runs, without
> explaining the tier model first. If you learned A2M recently, you are the
> right person — you still remember which part was confusing, and that is
> information nobody who has been here for months still has.
