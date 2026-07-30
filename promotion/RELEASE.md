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
| **§2 conformance, for a read-only server** | **The one with evidence behind it — see below.** |
| §4.3 `recall` parameters | The most-used method, and the hardest to extend without a version bump. Is the filter surface complete — is there a filter clients will obviously want that is missing? |
| §6.1 auth guidance | Non-normative today. If it should ever be normative, saying so before publication costs nothing and after costs a version. |
| §4.12 event kinds | Adding a kind later is safe (clients must tolerate unknown kinds). Renaming one is not. |
| §7 error codes | `-32001`..`-32008` are allocated. A code that should exist and does not becomes an awkward addition later. |
| §3.1 record fields | Adding a field is safe; changing the meaning of one is not. |

### The read-only conformance gap — found 2026-07-30, unresolved

DECISION 001 makes the project's best adoption claim:

> An existing RAG stack can be exposed over A2M in an afternoon — implement
> `describe` and `recall`, return `-32004 READ_ONLY` from `remember` — and every
> A2M client works against it.

**That claim has no implementation behind it, and it does not survive contact
with the conformance suite.** A minimal read-only server built exactly as
described was run against `tools/conformance.py`: `test_core` calls
`memory/remember`, the `-32004` propagates out of the check, and the run dies
with `FATAL the server did not answer memory/describe` — which blames the wrong
method entirely.

So a read-only server is, today, **unconformant and untestable**, while the
README and DECISION 001 both advertise it as the on-ramp for every existing RAG
stack. This is precisely the situation DECISION 020 deferred `events` for: a
claim in the document that nothing exercises.

Three ways out, in increasing order of cost:

1. **Spec sentence only.** Say in §2 that a server **MAY** answer `-32004` from
   `remember` and `forget` while still being core-conformant, and that a client
   must expect it. Cheapest, and it makes the existing claim true by fiat.
2. **Spec sentence plus suite support.** The above, plus `tools/conformance.py`
   detecting `READ_ONLY` on the first write and switching to a read-only
   profile — reporting something like `18/18, read-only`. This is what makes the
   claim *checkable*, which is the standard the rest of the project holds itself
   to.
3. **The above plus a shipped façade.** A seventh target: a read-only server
   over a static corpus, standard library only. Turns the strongest adoption
   argument into a runnable example.

**Recommendation: 2 before launch, 3 after.** 1 alone repeats the mistake the
repository exists to avoid. This is a spec decision, so it is Marco's — it is
recorded here rather than fixed because fixing it silently would have changed
the wire contract during a release checklist.

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
