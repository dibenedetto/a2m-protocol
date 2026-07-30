# Handoff — A2M, state and plan

**Internal working document, not part of the protocol.** It exists so a fresh
assistant session can pick up mid-stream without re-deriving context. Delete it
or move it somewhere private before the repository goes public; nothing else
references it.

Written 2026-07-30. If the dates below are more than a few weeks stale, verify
against `git log` before trusting any "current state" claim here.

---

## 1. What this project is

A2M (Agent-to-Memory Protocol) is a **specification**, not an application. The
document is the product; the code exists to prove the document is implementable
and to give implementers something to copy. When code and
[spec/a2m-0.1.md](spec/a2m-0.1.md) disagree, the specification wins and the code
is the bug.

Read [CLAUDE.md](CLAUDE.md) first — it is the operating manual for working in
this repository, and its "Invariants" section lists the rules that are easy to
break by accident. Read [DECISIONS.md](DECISIONS.md) when you need to know *why*
something is the way it is; argue with the reasoning recorded there rather than
rediscovering it.

Marco owns this. It is published under his GitHub account
(`dibenedetto/a2m-protocol`) at the domain he registered,
<https://a2m-protocol.org>. **Adoption by the community is the goal**, which is
why three things are load-bearing and must not be traded away without asking
him: the specification is the product, the core stays small enough that a plain
vector store can conform, and the repository stays standard-library only.

## 2. State as of 2026-07-30

- **Spec version `a2m/0.1`, unpublished.** Amending 0.1 in place is therefore
  still free. **That stops the moment it is announced** — after which additions
  are new capabilities and breaking changes need a version bump. This is the one
  irreversible gate in the whole plan.
- **All work is committed through `c134367`**; check `git status` for anything
  newer that a session left uncommitted.
- **Nine capabilities**: `core`, `tiers`, `salience`, `scopes`, `sessions`,
  `keys`, `embeddings`, `external`, `events`. No names are reserved (spec §9.1).
- **Seven conformance targets pass** — see §4 for the commands and counts.
- **Not yet done**: PyPI publication, the website, CI, the launch. That is §6.

## 3. Environment — read this before running anything

- **Never use bare `python`.** On this machine it is Python 3.8 and cannot even
  import `a2m` (PEP 585 generics). The project interpreter is
  `.venv\Scripts\python.exe` (Python 3.14, managed by `uv`).
- **Use its absolute path**, including as the child command in stdio
  conformance targets. A relative path fails `CreateProcess` on Windows. In
  PowerShell: `$py = (Resolve-Path .venv\Scripts\python.exe).Path`.
- **PowerShell is the working shell**; a Bash tool exists but `cygpath`
  translation mangles Windows paths — prefer PowerShell for anything spawning a
  process.
- **CrewAI cannot run on Python 3.14** (its `chromadb` dependency uses pydantic
  v1, which breaks there). That is CrewAI's problem, not A2M's. To exercise
  `implementations/adapters/crewai.py`, make a scratch venv on 3.12:
  `uv venv --python 3.12 <path> && uv pip install --python <path>\Scripts\python.exe crewai`.
- **PostgreSQL** runs in docker as container `a2m-pg`
  (`pgvector/pgvector:pg16`, port 55432, user/password/db all `a2m`). Usually
  already running; `docker ps` to check. The federated Postgres target needs a
  separate `a2mfed` database:
  `docker exec a2m-pg psql -U a2m -d a2m -c "CREATE DATABASE a2mfed"`.
- **Node** (v22) is installed, for the TypeScript server.

## 4. How to verify anything

Everything runs from the repository root. **A change is not done until all
seven conformance targets still pass.** They share no storage code — one is not
even Python, two need a database — so a change that passes only against
`python -m a2m` has probably leaked an implementation assumption into the
protocol layer. If Postgres is not running, say so rather than reporting five of
seven as a pass.

| target | expected |
|---|---|
| `python -m tools.test_a2m` | 232 passed, 0 failed |
| `python -m doctest a2m/*.py` (memory, text, retrieval, jsonrpc, protocol) | silent |
| `--stdio python -m a2m` | 94/94, 2 skipped |
| `--stdio python implementations/server_minimal.py` | 36/36, 9 skipped |
| `--stdio node --experimental-strip-types implementations/server_minimal.ts` | 46/46, 8 skipped |
| `--stdio python -m implementations.store_sqlite s.db` | 94/94 |
| `--stdio python -m implementations.store_postgres postgresql://a2m:a2m@127.0.0.1:55432/a2m` | 94/94 |
| `--stdio python -m implementations.server_federated r/` | 94/94 |
| `--stdio python -m implementations.server_federated postgresql://a2m:a2m@127.0.0.1:55432/a2mfed --backend postgres` | 94/94 |
| `--http http://127.0.0.1:8778/` (start a server with `--http` first) | 94/94 |
| `python -m tools.demo_stack` and `--router` | 18/18 each |
| `python -m examples.cross_framework` | 6/6 |
| `python -m examples.embedders` | 11/11, offline |
| `python -m examples.rag_ingest` | 16/16 |
| `python -m examples.procedural` | 14/14 |

Known rough edge, not yet fixed: `SqliteMemoryStack` holds its connection for
the life of the process and exposes no `close()`, while `PgMemoryStack` does.
On Windows the open handle blocks removing a temp directory, which is why
`examples/embedders.py` and `examples/procedural.py` pass
`ignore_cleanup_errors=True`. Adding `close()` to the SQLite store for parity is
Marco's call.

Conformance is `python -m tools.conformance <target>`. Over stdio the
full-capability targets exercise push delivery end to end; over HTTP the suite
checks push is honestly refused and runs the binding checks stdio cannot reach.

## 5. What the previous session completed

Two rounds of work, both finished and verified:

**The `events` capability** (spec §4.12–§4.14, DECISION 026), which supersedes
DECISION 020's deferral. One `EventLog` per server, two views of it: a cursor
poll that works on every transport, and optional push as JSON-RPC notifications
where the transport can carry one. Landed simultaneously in the spec, the JSON
schema, `a2m/protocol.py`, both SQL stores, the federated router, both minimal
servers' refusal tables, `implementations/client.py` and the conformance suite.

**Adoption surfaces**, each verified against a live server:
`implementations/bridge_mcp.py` (any A2M server as an MCP tool server, stdlib
only, tool list derived from `describe`), `adapters/crewai.py` (written against
CrewAI's *current* `StorageBackend` protocol, not the older `Storage` class),
`adapters/autogen.py`, and `examples/n8n_workflow.json`.

**Also**: the TypeScript server gained `keys`; spec §6.1 added non-normative
auth-over-HTTP guidance; DECISION 027 wrote down the A2M-versus-MCP-tools
argument and the README gained a section on it; hatchling packaging was added
and a wheel verified in a clean venv; two latent bugs were fixed (a duplicated
`timeline` definition in `protocol.py`, and `examples/cross_framework.py`
importing a pre-restructure path).

## 6. The promotion plan

Agreed with Marco on 2026-07-30. The organizing principle: **you get one launch,
and the wedge is the two-minute experience, not the document.** People adopt
something that works immediately and discover the protocol underneath. The MCP
bridge is that wedge.

Phases 0 and 1 are time-sensitive. The niche — a runtime memory protocol beside
MCP and A2A — is currently unoccupied, but Mem0 (61K+ stars), Zep and Letta are
all products that could publish an "open memory protocol" any quarter and win on
installed base alone. Weeks, not months.

### Phase 0 — become launchable (gates everything else)

- [ ] Commit and tag `v0.1.0`.
- [ ] **Freeze review of the wire format.** Marco's call, and irreversible.
- [ ] CI: GitHub Actions running tests plus every conformance target (Postgres
      via a service container, TS via Node) on push. For a project whose
      credibility *is* "94/94 across six implementations", the badge is the
      claim, re-proven publicly.
- [ ] a2m-protocol.org: static site — rendered spec, 60-second quickstart,
      conformance table, DECISIONS. Every file references this domain; a 404
      there undoes the polish everywhere else.
- [ ] Publish to PyPI so `pip install a2m-protocol` works in the launch post's
      first code block. **Needs Marco's credentials.**

### Phase 1 — the launch (one shot)

- [ ] Demo recording (asciinema or GIF): Claude Code remembers through
      `bridge_mcp` → a LangChain script recalls it → an n8n workflow reads the
      same store. Under two minutes. `tools/demo_stack.py` and
      `examples/cross_framework.py` are the raw material.
- [ ] The post: problem (five frameworks, five silos) → small-core argument
      (RAG is the degenerate case) → "why not MCP tools" → demo.
- [ ] Venues, same week: Show HN, r/LocalLLaMA, r/LangChain, lobste.rs, X.
      Cross-post to the W3C CG list the same day.

### Phase 2 — standing

- [ ] Join the **W3C AI Agent Memory Interoperability Community Group**
      (<https://www.w3.org/community/ai-agent-memory-interop/>), chartered
      2026-07-16. Its scope — encrypted memory cells, post-quantum signatures,
      GDPR erasure — is *custody and portability*, orthogonal to A2M's *runtime
      recall*. Position A2M as complementary and offer a crosswalk. It has no
      artifacts yet; being early is how A2M becomes what the eventual report
      cites.
- [ ] List `bridge_mcp` in the MCP servers registry and awesome-mcp-servers
      under "memory".
- [ ] Optional: a short arXiv report (design, conformance methodology, the
      four-kind model). The interop-protocol surveys have an empty memory slot.

### Phase 3 — adoption loops

- [ ] Submit into each framework's own directory: LangChain integrations docs,
      n8n template gallery (the workflow JSON is submission-ready), CrewAI and
      Agno community listings, AutoGen ecosystem docs.
- [ ] Make the conformance suite the contributor funnel: a page saying "write a
      server in your language, run one command, send a PR adding your row",
      plus seeded good-first-issues (Go minimal server, Redis store, Rust core).
- [ ] Build one façade over an incumbent — a read-only A2M server over Mem0 OSS
      (`describe` + `recall`, `-32004` from `remember`). Demonstrates "A2M is the
      interface, products compete underneath" without asking permission.

### Phase 4 — cadence

DECISIONS.md is a pre-written blog series ("Why scores never compare", "Why the
server never fetches your URI", "Shipping events without server-initiated
requests"). One post every week or two, each ending at the spec.

**Metrics that matter**: external implementations passing conformance, adapter
downloads, inbound issues from strangers. Stars are noise. The first conformance
badge Marco did not write himself is when A2M stops being a solo project.

## 7. What only Marco can do

Do not attempt these; prepare them and hand them over.

- The **freeze review** and the decision to announce.
- **PyPI publication** (credentials), **DNS/hosting** for a2m-protocol.org.
- **Posting** anywhere under his name — Show HN, Reddit, X, mailing lists.
- **W3C CG membership** (requires his W3C account).
- Anything that spends money or speaks publicly as him.

## 8. How to work with Marco

From [CLAUDE.md](CLAUDE.md) and accumulated experience:

- **Ask before implementing anything design-shaped.** He asked for this
  explicitly: "tell me what you are going to do and ask me confirmation before
  proceeding." He is building this to understand it, not merely to have it —
  presenting a finished artifact removes the part he wants. Present genuine
  decision points with a recommendation; he often answers with something better
  than the options offered (the events design came back as "implement both,
  configurable at setup time").
- **Measure claims rather than asserting them.** The default embedding model was
  chosen by a 250-query benchmark that overturned two prior assumptions. That is
  the standard.
- **Tell him plainly when something is wrong**, including something you built.
  Corrections land well; hedging does not.
- **Run the probe, do not assume.** Report what is actually true.

## 9. Landmines

Beyond CLAUDE.md's invariants, which you should read in full:

- **`tools/conformance.py` must never import anything under
  `implementations/`.** A suite that imported a server could only confirm the
  server agrees with itself.
- **`server_minimal.py`, `server_minimal.ts` and `client.py` import nothing from
  this repository.** That is the claim they exist to make. Do not "helpfully"
  refactor them onto shared code.
- **`store.py` contains no SQL.** Tier logic goes there or in neither SQL store.
  A table name or a placeholder in that file is the bug the Postgres port was
  written to catch.
- **Adapters are the only place a third-party import is allowed**, and
  `adapters/__init__.py` imports none of them. A checkout without langchain-core
  or agno must still run every test, every conformance target and both demos.
- **Docstring examples are executed by doctest.** If you write one, it must be
  true.
- **Style is deliberate**: tabs, `=` aligned within a block, imports in three
  groups, Google-style docstrings that never open with a section header.

## 10. Things an assistant is likely to get wrong

- Assuming `events` is still deferred. It shipped; DECISION 020 is superseded by
  026. The conformance suite now *requires* a declaring server to honour it.
- Assuming the TypeScript server is `core`-only. It declares `core` **and**
  `keys`, and answers 46 checks.
- Writing the CrewAI adapter against `crewai.memory.storage.interface.Storage`.
  That class is gone; the current interface is the `StorageBackend` Protocol in
  `crewai/memory/storage/backend.py`, which is embedding-based and hierarchical.
- Reaching for `serve_stdio` when serving a full-capability server. Use
  `serve_a2m_stdio`, which wires the push drain; the bare one would accept a
  subscription and then deliver nothing.
- Treating `score` as comparable across servers or calls. It is not, ever.
