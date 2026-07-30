# Directory submissions

Internal. Each of these is a listing in **someone else's** documentation, which
outperforms anything posted in ours — a listing in the LangChain integration
docs reaches people who are already solving the problem A2M solves.

All of them are for **Marco to send**. Each entry gives the target, what it
needs, and drafted text.

Order matters: the MCP registry first (largest audience, lowest effort, and the
bridge needs no adapter to maintain), then n8n, then the framework docs.

---

## 1. MCP servers registry — do this first

**Where:** `modelcontextprotocol/servers` (the `README.md` community list) and
<https://github.com/punkpeye/awesome-mcp-servers>, under **Memory**.

**Why first:** the largest concentration of people who already want exactly this
and can use it without reading the spec. It also costs nothing to maintain — the
bridge is stdlib-only and tracks the protocol, not any framework's API.

**Entry:**

> **[A2M Memory](https://github.com/dibenedetto/a2m-protocol)** — Shared agent
> memory over the Agent-to-Memory Protocol. Exposes any A2M server as MCP tools,
> so the same memory store is readable by Claude, Cursor, LangChain, Agno,
> CrewAI, AutoGen and n8n. Tiered (working / episodic / semantic / procedural),
> with SQLite and PostgreSQL backends. Python, no dependencies.

**Config block to include in the PR** — this is what people copy, so it must be
correct:

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "implementations.bridge_mcp", "--stdio",
               "python", "-m", "implementations.store_sqlite", "memory.db"]
    }
  }
}
```

**Before submitting:** verify that exact block in a real MCP client, not just in
the scripted test. A registry entry whose config does not work is worse than no
entry.

---

## 2. n8n template gallery

**Where:** <https://n8n.io/workflows/> — submit via the creator dashboard.
Needs an n8n account.

**Asset:** [examples/n8n_workflow.json](../examples/n8n_workflow.json), already
importable and verified against a live server.

**Title:** Shared AI agent memory (A2M protocol)

**Description:**

> Give your n8n workflows a memory that other AI agents can read and write too.
>
> This workflow talks to an A2M server using nothing but stock HTTP Request
> nodes — no custom node, no community package. It describes the server, stores
> a fact, and recalls it by meaning rather than by keyword.
>
> The point is what happens outside n8n: the same memory store is readable by
> LangChain, Agno, CrewAI and AutoGen agents, and by any MCP client through the
> included bridge. A fact your workflow stores is a fact your agents can recall,
> and the other way round.
>
> **Setup:** run a memory server, then import and execute.
>
> ```
> pip install a2m-protocol
> python -m implementations.store_sqlite memory.db --http 8778
> ```
>
> A2M is an open MIT-licensed protocol: https://a2m-protocol.org

**Screenshot needed:** the four-node canvas plus the output panel showing the
recalled record.

---

## 3. LangChain integrations documentation

**Where:** `langchain-ai/langchain` — `docs/docs/integrations/memory/` or
`docs/docs/integrations/retrievers/`. Check `CONTRIBUTING.md` first; they
require the notebook format and a working example.

**What to submit:** a notebook covering `A2MChatMessageHistory` and
`A2MRetriever` from
[implementations/adapters/langchain.py](../implementations/adapters/langchain.py).

**Lead paragraph:**

> **A2M** is an open protocol for agent memory. It matters for LangChain
> specifically because LangChain's split between chat history and retrieval is
> the same split A2M draws between `memory/timeline` (replayed, chronological)
> and `memory/recall` (searched, ranked) — so each side maps onto one method
> and neither has to emulate the other.
>
> The reason to use it over an in-process memory is that the store is not
> LangChain's. An Agno knowledge base, a CrewAI crew, an AutoGen agent or an n8n
> workflow can read what your chain wrote, and the other way round.

**Note in the PR:** the adapter lives in the A2M repository and imports only
`langchain-core`, so there is nothing for LangChain to maintain. This is
documentation, not a code dependency — say so explicitly, it makes the review
much easier.

---

## 4. CrewAI, Agno, AutoGen community docs

Same shape as the LangChain submission, adjusted per project. Each of these
should be sent **only after** the adapter has been exercised against the current
release of that framework — the CrewAI interface already changed once under us
(`Storage` became the `StorageBackend` protocol), and submitting against a stale
interface is the fastest way to be dismissed.

| project | where | angle |
|---|---|---|
| **Agno** | `agno-agi/agno` docs, VectorDb integrations | A knowledge base that other frameworks can write into. Emphasise caller-owned embeddings: Agno's embedder output is stored verbatim and never regenerated. |
| **CrewAI** | `crewAIInc/crewAI` docs, custom storage | A crew memory that survives the crew. Emphasise that scopes map onto A2M keys, so two crews share a store without resetting each other. |
| **AutoGen** | `microsoft/autogen`, `autogen_ext.memory` ecosystem page | A `Memory` implementation whose store is shared. Emphasise `update_context` querying rather than replaying. |

---

## 5. Awesome lists

Low effort, non-trivial long-tail discovery. One PR each:

- `awesome-mcp-servers` (covered in §1)
- `e2b-dev/awesome-ai-agents`
- `kyrolabs/awesome-langchain`
- `Shubhamsaboo/awesome-llm-apps`
- `steven2358/awesome-generative-ai`

**One-liner:**

> **[A2M](https://github.com/dibenedetto/a2m-protocol)** — an open protocol
> letting agents from different frameworks share one memory store. JSON-RPC 2.0;
> in-process, stdio or HTTP; six conformant implementations; standard library
> only.

---

## Rules for all of these

- **Never submit to more than two places in one day.** A wave of identical
  submissions across ecosystems reads as spam and gets all of them rejected.
- **Read each project's contribution guide first.** Several require an issue
  before a PR.
- **Disclose authorship.** "I built this" is fine and expected; pretending to be
  a neutral third party is not, and it is the one mistake that cannot be walked
  back.
- **Never submit a broken example.** Every config block, notebook and workflow
  above must be run once, in a clean environment, immediately before it is sent.
