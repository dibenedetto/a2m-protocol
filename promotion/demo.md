# The demo — shot list

Internal. The single most valuable launch asset, because it is the thing people
watch before they read anything. Target: **under two minutes**, no cuts, no
narration track.

The story is one sentence: *two different frameworks, and a workflow tool, all
reading and writing one memory.* Everything below serves that and nothing else.

Record with [asciinema](https://asciinema.org) (`asciinema rec a2m.cast`) and
convert with `agg` for the GIF the README and X thread need. A terminal
recording beats a screencast here — it is small, it is copy-pasteable, and it
cannot look staged.

---

## Setup, before recording

```powershell
# A clean, wide, legible terminal: 100x30, large font, minimal prompt.
$env:PROMPT = "$ "
Remove-Item demo.db* -ErrorAction SilentlyContinue

# Everything the demo needs, installed and warm, so nothing downloads on camera.
uv pip install --python .venv\Scripts\python.exe langchain-core agno
```

Rehearse once with the recorder off. The commands below are exact; the point of
rehearsing is the *pauses*, which are what make it readable.

---

## Take 1 — the core story (60 seconds)

**Shot 1. The problem, stated by the tool itself.** (~8s)

```
$ python -m a2m --help
```

Let the docstring sit on screen for two seconds. It establishes that this is a
server, that it is one command, and that nothing was installed.

**Shot 2. A LangChain agent writes.** (~15s)

```
$ python -m examples.cross_framework
```

This already exists and already passes 6/6. It writes conversation through
LangChain's chat-history interface, asserts Agno *cannot yet* see it — because
those turns are in working memory, which is replayed and never searched — then
closes the session and shows Agno finding it afterwards.

**Pause on the assertion that fails first.** That is the most interesting frame
in the whole demo: it shows the tier model doing something a flat store cannot,
rather than just moving bytes between two libraries.

**Shot 3. The same store, from a different process entirely.** (~12s)

```
$ python implementations/client.py --stdio python -m implementations.store_sqlite demo.db -- remember "the deploy key rotates every ninety days"
$ python implementations/client.py --stdio python -m implementations.store_sqlite demo.db -- recall "how often does the key change?"
```

A Python client, a subprocess server, one file on disk. Note out loud in the
post — not on screen — that `client.py` imports nothing from the repository.

**Shot 4. A different language, same suite.** (~15s)

```
$ python -m tools.conformance --stdio node --experimental-strip-types implementations/server_minimal.ts
```

Let the check list scroll. Land on `46/46 checks passed`. This is the shot that
converts skeptics: a Python conformance suite passing against a TypeScript
server that shares no code with it and needs no build step.

---

## Take 2 — the MCP bridge (30 seconds)

Worth its own short recording, because it is the wedge: it makes A2M useful to
someone who has not read a word of the spec.

**Shot 5.** Add the bridge to an MCP client's config, on screen:

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

**Shot 6.** In Claude Code or Claude Desktop: ask it to remember something, start
a **new session**, and ask it back. The memory survives the session boundary and
lives in a file the viewer just saw created.

**Shot 7.** Then — this is the payoff — recall the same fact from the terminal,
outside any MCP client:

```
$ python implementations/client.py --stdio python -m implementations.store_sqlite memory.db -- recall "what did I ask it to remember?"
```

The point lands without a word of explanation: *the memory is not inside the
agent.*

---

## Take 3 — n8n (optional, 20 seconds)

Only if takes 1 and 2 are clean. Import `examples/n8n_workflow.json`, run it,
show the recalled record in the output panel — then note that the store is the
same one the previous takes wrote to. It is the least technically interesting
shot and the most convincing to a non-programmer audience.

---

## What to cut if it runs long

In order: take 3, then shot 1, then shot 3. Never cut shot 2 or shot 4 — they
are the two shots that show something no competing project can show.

## Deliverables

- `a2m.cast` — the asciinema recording, linked from the launch post.
- `a2m.gif` — first 30 seconds, for the README header and the X thread.
- A single still of `46/46 checks passed`, for the Reddit post.
