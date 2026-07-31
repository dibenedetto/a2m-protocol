# Release checklist — a2m-protocol 0.1.0

Internal. Everything here is mechanical except the two items marked **Marco
only**, which are the irreversible ones.

Run every command from the repository root, with `.venv\Scripts\python.exe`.

---

## 1. Before anything: the freeze review — **Marco only**

The specification is unpublished, so amending `0.1` in place is still free.
**That freedom ends the moment the announcement goes out.** After it, additions
are new capabilities and breaking changes need a version bump.

So this is the last cheap moment to change your mind about the wire format. The
places worth a deliberate second look, in descending order of how expensive they
are to change later:

| | why it is worth re-reading now |
|---|---|
| ~~§2 conformance for a read-only server~~ | **Resolved 2026-07-30 — see below. No longer a review item.** |
| §4.3 `recall` parameters | The most-used method, and the hardest to extend without a version bump. Is the filter surface complete — is there a filter clients will obviously want that is missing? |
| §6.1 auth guidance | Non-normative today. If it should ever be normative, saying so before publication costs nothing and after costs a version. |
| §4.12 event kinds | Adding a kind later is safe (clients must tolerate unknown kinds). Renaming one is not. |
| §7 error codes | `-32001`..`-32008` are allocated. A code that should exist and does not becomes an awkward addition later. |
| §3.1 record fields | Adding a field is safe; changing the meaning of one is not. |

### The read-only conformance gap — resolved 2026-07-30

Recorded here because the resolution changed the specification, so it is worth
re-reading during the freeze review rather than taken on trust.

DECISION 001 makes the project's best adoption claim: an existing RAG stack can
be exposed over A2M by implementing `describe` and `recall` and refusing writes,
and every A2M client then works against it. Building that server found the claim
had nothing behind it — `tools/conformance.py` writes a record as its first act,
the `-32004` escaped every handler, and the run died reporting `the server did
not answer memory/describe`, blaming the one method that had worked perfectly.

Resolved in three parts:

1. **Spec §2.1** now states normatively that a read-only server is conformant,
   that it **MUST** refuse both `remember` and `forget` with `READ_ONLY`, and
   that it must refuse **consistently** — a store accepting some writes and
   refusing others leaves a client no way to tell which is which. A client
   **MUST** tolerate the error.
2. **The suite** treats its first write as the probe: on `-32004` it switches to
   a read-only profile, verifies both write methods refuse, and checks the read
   side against whatever the corpus actually holds. The mislabelled FATAL
   handler was fixed at the same time — it named `memory/describe` for *every*
   escaping error.
3. **`implementations/server_readonly.py`** ships as an eighth conformance
   target at **31/31**, imports nothing from the repository, and isolates the
   retrieval into a single `search` function for an implementer to replace. CI
   runs it.

**What to re-read during the freeze review:** spec §2.1 itself, since it is the
only conformance rule added after the rest of the document settled.

Nothing else here is known to be wrong — the rest is a review, not a fix list.

## 2. Verify everything

```powershell
$py = (Resolve-Path .venv\Scripts\python.exe).Path

& $py -m tools.check_stdlib_only                 # no dependency crept in
& $py -m tools.test_a2m                          # 232 passed
& $py -m doctest a2m/memory.py a2m/text.py a2m/retrieval.py a2m/jsonrpc.py a2m/protocol.py
& $py -m examples.embedders                      # 11/11
& $py -m examples.rag_ingest                     # 16/16
& $py -m examples.procedural                     # 14/14
& $py -m tools.demo_stack                        # 18/18
& $py -m tools.demo_stack --router               # 18/18
```

Then all seven conformance targets — see [HANDOFF.md](../HANDOFF.md) §4 for the
table and the expected counts. Do not release on five of seven.

## 3. Build and check the artifacts

```powershell
Remove-Item dist -Recurse -Force -ErrorAction SilentlyContinue
uv build
uv pip install --python <a fresh venv> dist\a2m_protocol-0.1.0-py3-none-any.whl
```

- [ ] The wheel imports and works **from a directory that is not the repository**.
- [ ] `a2m --help` prints usage rather than starting a server.
- [ ] The floor holds: repeat the install on Python 3.10.
- [ ] `twine check dist/*` passes — this is what catches a README that renders
      badly on PyPI, which is the project's shop window.

```powershell
uv pip install --python .venv\Scripts\python.exe twine
.venv\Scripts\python.exe -m twine check dist/*
```

## 4. Publish to PyPI — **Marco only**

Test first. TestPyPI is free and catches a broken description or a name clash
before it is permanent — **a version number on PyPI can never be reused**, even
after deletion.

```powershell
.venv\Scripts\python.exe -m twine upload --repository testpypi dist/*
# then, in a throwaway venv:
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple a2m-protocol
```

Then the real one:

```powershell
.venv\Scripts\python.exe -m twine upload dist/*
```

- [ ] Reserve the name **before** the launch post. A Show HN pointing at a
      `pip install` that 404s is the one unrecoverable launch mistake.
- [ ] Use an API token (`__token__` as username), scoped to this project.
- [ ] Check the rendered page: <https://pypi.org/project/a2m-protocol/>.

## 5. GitHub

- [ ] Merge `launch-prep` into `main`.
- [ ] Repository settings → Pages → Source: **GitHub Actions**.
- [ ] Repository settings → Pages → Custom domain: `a2m-protocol.org`.
      The workflow writes the `CNAME` file; the DNS records are yours to add at
      the registrar (an `ALIAS`/`ANAME` at the apex to
      `dibenedetto.github.io`, or four `A` records to GitHub's IPs).
- [ ] Wait for the `pages` workflow to go green, then check
      <https://a2m-protocol.org> actually serves.
- [ ] Repository → About: set the description, the website, and topics
      (`agent-memory`, `protocol`, `jsonrpc`, `mcp`, `llm`, `rag`,
      `interoperability`, `specification`).
- [ ] Confirm the CI badge in the README is green on `main`.
- [ ] Tag `v0.1.0` **after** the freeze review, and only then.
- [ ] Create the GitHub Release from the tag; its body is
      [launch-post.md](launch-post.md) trimmed to the summary and the table.

## 6. Delete or hide the internal files

`HANDOFF.md` and this whole `promotion/` directory are working state, not the
product. Before or at launch, either delete them or move them to a private
location. Nothing in the repository references them, so removing them breaks
nothing.

## 7. The order that matters

Getting this sequence wrong is the only way to waste the one launch:

1. Freeze review.
2. Tag, PyPI, Pages — all live and verified.
3. **Click through the quickstart yourself, on a machine that has never seen
   this repository.** `pip install a2m-protocol`, paste the three lines from
   the landing page, watch them work.
4. Only then, post.
