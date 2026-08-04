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
- **Work is committed on branch `launch-prep`**, not merged to `main`; check
  `git log main..HEAD` and `git status`.
- **Eleven capabilities**: `core`, `tiers`, `salience`, `scopes`, `sessions`,
  `keys`, `embeddings`, `external`, `events`, `summarize`, `prompt`. No names
  are reserved (spec §9.1).
- **Eight conformance targets pass** — see §4 for the commands and counts.
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
eight conformance targets still pass.** They share no storage code — one is not
even Python, two need a database — so a change that passes only against
`python -m a2m` has probably leaked an implementation assumption into the
protocol layer. If Postgres is not running, say so rather than reporting six of
eight as a pass.

| target | expected |
|---|---|
| `python -m tools.test_a2m` | 271 passed, 0 failed |
| `python -m doctest a2m/*.py` (memory, text, retrieval, jsonrpc, protocol, prompt) | silent |
| `--stdio python -m a2m` | 121/121, 2 skipped |
| `--stdio python implementations/server_minimal.py` | 37/37, 11 skipped |
| `--stdio node --experimental-strip-types implementations/server_minimal.ts` | 47/47, 10 skipped |
| `--stdio python -m implementations.store_sqlite s.db` | 121/121 |
| `--stdio python -m implementations.store_postgres postgresql://a2m:a2m@127.0.0.1:55432/a2m` | 121/121 |
| `--stdio python -m implementations.server_federated r/` | 96/96 (declares no summarizer) |
| `--stdio python -m implementations.server_federated postgresql://a2m:a2m@127.0.0.1:55432/a2mfed --backend postgres` | 96/96 |
| `--http http://127.0.0.1:8778/` (start a server with `--http` first) | 121/121 |
| `python -m tools.demo_stack` and `--router` | 37 and 33 |
| `python -m examples.cross_framework` | 6/6 |
| `--stdio python implementations/server_readonly.py` | 32/32, read-only profile |
| `python -m examples.embedders` | 11/11, offline |
| `python -m examples.rag_ingest` | 16/16 |
| `python -m examples.procedural` | 14/14 |
| `python -m examples.llm_wiki` | 21/21 |
| `python -m tools.test_interop` (needs 4 frameworks, py3.12) | 35/35, 5x5 grid |
| `python -m tools.check_n8n` (needs a server on 8778) | 3/3 |

Both SQL stores inherit `close()` from `TieredMemoryStack`; anything opening
one must call it, or on Windows the open handle blocks deleting the file.

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

## 5b. Open findings — read before the freeze review

Two things found by measurement on 2026-07-30, neither fixed, both deliberately
left for Marco because both are decisions rather than maintenance.

**The read-only conformance gap — RESOLVED 2026-07-30.** DECISION 001 claimed an
existing RAG stack could be exposed by implementing `describe` and `recall` and
refusing writes. Building that server found the suite could not run it: the
`-32004` escaped every handler and the run died reporting `the server did not
answer memory/describe`, blaming the one method that worked. Fixed in three
parts: spec §2.1 makes read-only servers normatively conformant and requires
**both** write methods to refuse consistently; the suite treats its first write
as a probe and switches to a read-only profile; and
`implementations/server_readonly.py` ships as an eighth conformance target at
**31/31**, wired into CI. The FATAL handler no longer blames `describe` for
every escaping error.

**`requires-python` was `>=3.14`, which excluded almost everyone.** Measured
across 3.9–3.14: the full suite and every conformance target pass on **3.10**
and up; 3.9 fails on PEP 604 unions evaluated at runtime. Now `>=3.10`, with CI
running both the floor and the ceiling so it cannot silently rise again. Worth
knowing because the same mistake — an interpreter requirement nobody measured —
is easy to reintroduce.

**Correction to an earlier note in this file:** it claimed `SqliteMemoryStack`
exposes no `close()`. That was wrong — `close()` lives on the shared
`TieredMemoryStack` base in `implementations/store.py`, so both SQL stores have
it. The temp-directory failure that prompted the note was two examples not
*calling* it; they now do.

## 6. The promotion plan

Agreed with Marco on 2026-07-30. The organizing principle: **you get one launch,
and the wedge is the two-minute experience, not the document.** People adopt
something that works immediately and discover the protocol underneath. The MCP
bridge is that wedge.

Phases 0 and 1 are time-sensitive. The niche — a runtime memory protocol beside
MCP and A2A — is currently unoccupied, but Mem0 (61K+ stars), Zep and Letta are
all products that could publish an "open memory protocol" any quarter and win on
installed base alone. Weeks, not months.

**Status as of 2026-07-30: everything buildable is built, on branch
`launch-prep`.** What is left is the outward-facing half, which is Marco's by
definition — see §7. Internal materials live in [promotion/](promotion/) and,
like this file, should be deleted or moved before the repository is public.

### Phase 0 — become launchable (gates everything else)

- [x] Committed on `launch-prep`. **Not** tagged — tagging waits for the freeze
      review, since a tag is the closest thing to an announcement the repository
      has.
- [ ] **Freeze review of the wire format.** Marco's call, and irreversible.
      Checklist and the open read-only question: [promotion/RELEASE.md](promotion/RELEASE.md).
- [x] CI: `.github/workflows/ci.yml` runs the library on the floor and the
      ceiling interpreter, every conformance target (Postgres via a service
      container, TypeScript via Node), the HTTP binding, the examples, both
      demos, and a wheel install from outside the repository. Counts are
      asserted, not just exit codes.
- [x] `tools/check_stdlib_only.py` walks every import and fails CI on a stray
      dependency. Verified by injecting a violation.
- [x] a2m-protocol.org: `www/`, hand-written HTML plus a standard-library
      renderer that builds the spec, the guide and the decision log out of the
      repository. `.github/workflows/pages.yml` deploys it. **Needs Marco:**
      repository Pages settings and the DNS records.
- [x] Packaging: classifiers, keywords, an `a2m` console script, and
      `requires-python` corrected from `>=3.14` to a measured `>=3.10`.
- [ ] Publish to PyPI. **Needs Marco's credentials.** Checklist in
      [promotion/RELEASE.md](promotion/RELEASE.md) §3–4; reserve the name before
      the launch post, since a `pip install` that 404s is the one unrecoverable
      launch mistake.

### Phase 1 — the launch (one shot)

- [x] Shot list written: [promotion/demo.md](promotion/demo.md) — exact
      commands, timings, what to cut if it runs long, and which two shots must
      never be cut. **Needs Marco:** the recording itself.
- [x] Post drafted in four registers (Show HN, Reddit, X thread, CG list),
      with prepared answers for the questions that will certainly be asked:
      [promotion/launch-post.md](promotion/launch-post.md).
- [ ] **Needs Marco:** posting, and being at the keyboard for the three hours
      after. On Show HN, replying fast matters more than the post.

### Phase 2 — standing

- [x] W3C CG introduction drafted: [promotion/w3c-cg-note.md](promotion/w3c-cg-note.md).
      Framed as a contribution to their charter work, not an announcement —
      their scope is custody and portability, A2M's is runtime access, and the
      note offers a crosswalk rather than asking for adoption. **Needs Marco:**
      joining the group and sending it.
- [x] MCP registry entry and every other directory submission drafted:
      [promotion/directory-submissions.md](promotion/directory-submissions.md).
      **Needs Marco:** sending them, no more than two a day.
- [ ] Optional: a short arXiv report (design, conformance methodology, the
      four-kind model). The interop-protocol surveys have an empty memory slot.

### Phase 3 — adoption loops

- [x] Framework directory submissions drafted, with the rules that stop them
      reading as spam: [promotion/directory-submissions.md](promotion/directory-submissions.md).
- [x] Contributor funnel: [CONTRIBUTING.md](CONTRIBUTING.md) leads with "write a
      server in your language", names the five rules every implementation gets
      wrong, and ends at a PR adding a row to the conformance table. Seven issue
      bodies ready to create in [promotion/seeded-issues.md](promotion/seeded-issues.md).
- [ ] **Blocked, deliberately:** the read-only façade over an incumbent. It
      cannot be built honestly until the read-only conformance gap in §5b is
      resolved — shipping a façade that crashes the conformance suite would
      demonstrate the opposite of the intended point.

### Phase 4 — cadence

- [x] Seven posts outlined from DECISIONS, each shaped as "a bug you recognise"
      rather than "a protocol you have not heard of":
      [promotion/blog-series.md](promotion/blog-series.md). Includes the two
      posts *not* to write and why.

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
