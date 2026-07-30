# Launch post — drafts

Internal. Four versions of the same argument, sized for four rooms. All of them
are yours to post; none of them should go out before
[RELEASE.md](RELEASE.md) §7 is done.

The argument, in one line, is the thing to keep intact everywhere:
**agent frameworks each ship an incompatible memory model, and A2M is one
interface they can all read and write through — small enough that a plain
vector store can conform in an afternoon.**

Two things to avoid saying, because both are false and both are checkable:

- that A2M competes with MCP (it shares MCP's bindings on purpose);
- that it replaces Mem0, Zep or Letta (those are products; this is the
  interface they could all serve).

---

## A. Show HN

**Title** (80 char limit, and it is the whole of the first impression):

> Show HN: A2M – an open protocol so agents from different frameworks share memory

Alternatives, if that reads flat:

> Show HN: A memory protocol for AI agents, with six conformant implementations
> Show HN: A2M – putting agent memory behind the same boundary MCP puts tools

**Body:**

---

Every agent framework ships its own memory model. LangChain has chat history and
retrievers, Agno has a vector knowledge base, CrewAI has its own storage
backend, AutoGen has a Memory protocol, n8n has workflow context. Point two of
them at the same problem and they cannot see each other's state — even running
inside one workflow.

A2M is a small JSON-RPC 2.0 protocol that puts a memory store behind the same
kind of boundary MCP puts a tool behind. An agent talks to `memory/*` and never
learns whether the store is an object in its own process, a subprocess, or a
service across the network.

    pip install a2m-protocol

    memory = connect_local(MemoryStack())              # in-process
    memory = connect_stdio(["python", "-m", "a2m"])    # a child process
    memory = connect_http("http://127.0.0.1:8778/")    # over the network

    memory.remember("the deploy key rotates every ninety days")
    memory.recall(query="how often does the key change?")

Two decisions shaped it.

**A memory store is not a database.** Agents do not query memory, they *recall*
from it: they hand over the situation they are in and expect back whatever is
worth knowing, ranked. Ranking is the primitive, not filtering.

**Most stores are not layered.** A vector database can write, search and delete
in an afternoon and has no concept of tiers, consolidation or salience. A
protocol demanding all of it would be implementable only by its own reference
implementation. So there is a five-method mandatory core, and everything else is
a capability a server declares and a client checks. Classic RAG turns out to be
the degenerate case — one tier, read-only, `recall` only — so an existing RAG
stack can be exposed over A2M by implementing `describe` and `recall` and
returning `READ_ONLY` from `remember`.

The part I would most like feedback on is the conformance discipline. The suite
speaks only the protocol and never imports the server it tests, and six
implementations pass it unmodified: the reference, a stdlib-only minimal server,
a TypeScript server run with `node --experimental-strip-types` (no build step),
SQLite + sqlite-vec, PostgreSQL + pgvector, and a federation of four servers
behind a router. Writing the minimal server from the spec alone found a real bug
in the reference — it was rejecting unrecognised parameters, breaking the
forward-compatibility rule that lets a newer client talk to an older server.

It is not an MCP tool server, deliberately, though it shares MCP's bindings
exactly (same JSON-RPC, same stdio framing, no batches). Behind `tools/call` a
record stops being typed, capability negotiation collapses into a tool list, and
the replay-versus-search distinction the tier model rests on survives only as a
sentence in a description nothing enforces. There is a bundled bridge that
exposes any A2M server as an MCP tool server, so the tradeoff is visible rather
than argued about.

Everything — protocol, reference implementation, conformance suite — is standard
library only, and there is a CI check that walks every import to keep it that
way.

Spec: https://a2m-protocol.org
Code: https://github.com/dibenedetto/a2m-protocol

Draft 0.1 and pre-1.0, so I would rather hear now that something in the wire
format is wrong than after anyone depends on it.

---

**Timing.** Weekday, 13:00–16:00 UTC lands mid-morning US Eastern, which is when
Show HN gets read. Be at the keyboard for the following three hours — on Show
HN, replying fast matters more than the post.

**Prepared answers** for what will certainly be asked:

- *"Why not MCP?"* — Because a tool result is text for a model, and a memory
  store's consumers are mostly programs. Point at the bridge; it makes the
  tradeoff concrete.
- *"Why not Mem0 / Zep / Letta?"* — Those are products, and good ones. A2M is
  the interface they could each serve. The read-only façade in the roadmap is
  the demonstration.
- *"Isn't this just RAG with extra steps?"* — RAG is the degenerate case, and
  the spec says so. The difference is who writes the corpus and whether it has a
  lifecycle.
- *"Who else uses it?"* — Nobody yet, and say so plainly. The honest pitch is
  the conformance suite: if you implement it, there is an objective way to know
  you did it right.
- *"Python 3.10+? Windows?"* — Measured, not assumed: CI runs the floor and the
  ceiling.

---

## B. Reddit — r/LocalLLaMA, r/LangChain

Same content, lower formality, no "Show HN" framing. Lead with the concrete
problem, not the protocol.

**Title:** I got a LangChain agent and an Agno agent to share one memory store

**Body:** open with the cross-framework demo, then explain what had to exist for
it to work. r/LocalLLaMA cares most about it being local-first, standard-library
only, and not requiring a hosted service — say that in the first two lines. Link
the spec last, not first.

Post to r/LangChain **after** r/LocalLLaMA, and only if the first lands: two
simultaneous subreddit posts read as marketing.

---

## C. X / Bluesky

A thread, one idea per post, the demo GIF on the first.

1. Five agent frameworks. Five incompatible memory models. Agents in the same
   workflow that cannot read each other's state. [GIF]
2. A2M is one interface they can all speak. JSON-RPC 2.0, in-process, stdio or
   HTTP — the transport changes, the client does not.
3. Five mandatory methods. Everything else is a capability a server declares and
   a client checks, so a plain vector store conforms in an afternoon. Classic
   RAG is the degenerate case.
4. Six implementations pass one unmodified conformance suite: two storage
   engines, two languages, two topologies. Writing the second one found a real
   bug in the first.
5. It is not an MCP tool server — but it shares MCP's bindings exactly, and
   ships a bridge that turns any A2M store into one.
6. MIT, standard library only, pre-1.0 and looking for people to argue with the
   wire format while it is still free to change. → a2m-protocol.org

---

## D. W3C AI Agent Memory Interoperability CG

Cross-post the same week, but rewrite the framing entirely: this room cares
about specification quality and scope boundaries, not about installation.

See [w3c-cg-note.md](w3c-cg-note.md) — it is written as a contribution to their
work rather than an announcement of yours, which is the only version worth
sending.
