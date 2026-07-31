"""Karpathy's LLM wiki, on A2M. Nothing here is a protocol extension.

	python -m examples.llm_wiki

Andrej Karpathy's proposal (gist 442a6bf5) is that an agent should not
rediscover knowledge from raw documents on every question, but **incrementally
build and maintain a wiki** — a persistent artifact where the synthesis, the
cross-references and the flagged contradictions are already there. Three layers:

	raw sources   immutable; the LLM reads them and never modifies them
	the wiki      markdown pages the LLM writes, updates and cross-references
	the schema    a document defining the conventions the maintainer follows

Every one of them is an ordinary A2M record, and the operations map onto methods
that already exist:

	his layer        A2M
	raw sources      `external` -- a uri pointing at the document, content
	                 indexed so it is findable, and the server never fetches it
	the wiki         `keys` in a semantic tier -- a page is a *fact* with an
	                 address, so rewriting it replaces rather than accumulates
	the schema       `procedural` -- how to do things, written deliberately,
	                 and nothing spills into it

	his operation    A2M
	ingest           `remember` the sources, then `summarize` them into a page
	query            `recall`, and write the answer back as a page
	lint             `events` -- walk a cursor over what changed and re-check it

The reason it fits is not a coincidence, and it is worth naming: his complaint
about RAG is that a corpus does not *accumulate* synthesis. That is the same
problem addressable keys were added to solve — in an append-only store a
corrected page merely sits beside the stale one, and ranking has no way to
prefer the newer. A wiki is impossible on a memory that can only append.

What A2M does **not** supply is the interesting part of his proposal: deciding
what a page should say, what cross-references to draw, and which claims
contradict each other. That is agent behaviour, and the protocol deliberately
has no opinion about it. Here it is stood in for by an extractive summarizer, so
the example runs offline; swap `summarize_fn` for `llm_consolidator(model)` and
the same code becomes the real thing.
"""


import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from   a2m           import MemoryClient, MemoryServer
from   a2m.jsonrpc   import Client, LocalTransport
from   a2m.memory    import MemoryStack, MemoryTier, default_tiers
from   a2m.retrieval import extractive_summarizer


PASSED = []
FAILED = []


# Layer 1: the raw sources. Immutable, and in a real deployment these are files
# on disk or pages on a wiki -- which is why each carries a `uri` and the server
# is forbidden from fetching it.
SOURCES = {
	"rfc-deploy-keys.md": (
		"Deploy keys rotate every ninety days. Rotation is automatic. "
		"The rotation is announced on the release channel one week ahead. "
		"A failed rotation leaves the previous key valid."
	),
	"incident-2026-02.md": (
		"On 3 February the deploy key rotation failed silently. "
		"The previous key remained valid, so no deploy was blocked. "
		"The alert was missing because rotation emits no event on success."
	),
	"handbook-oncall.md": (
		"On-call rotates weekly and hands over on monday morning. "
		"The on-call engineer owns the release channel during their week."
	),
}

# Layer 3: the schema. In Karpathy's proposal this is a CLAUDE.md-shaped
# document telling the maintainer how to keep the wiki. In A2M it is procedural
# memory: written deliberately, read at task start, and never spilled into.
SCHEMA = """\
# Wiki conventions

1. One page per entity or concept, addressed `wiki/<topic>`.
2. A page is rewritten in place. Never append a second page for the same topic.
3. Every page cites the sources it was built from, in `metadata.sources`.
4. Flag a contradiction on the page that asserts the older claim.
5. A page not rebuilt since its sources changed is stale, and lint must find it.
"""


def check(label: str, condition: bool, detail=None) -> None:
	"""Record one assertion.

	Args:
		label (str): What was being checked.
		condition (bool): Whether it held.
		detail (Any, optional): Shown on failure.
	"""
	if condition:
		PASSED.append(label)
		print(f"  ok    {label}")
	else:
		FAILED.append(label)
		print(f"  FAIL  {label}  {detail if detail is not None else ''}")


def open_wiki() -> MemoryClient:
	"""A store shaped for a wiki: sources and pages kept apart.

	`sources` is a second semantic tier so that ingested documents and written
	pages cannot evict one another, and so a read can ask for exactly one of
	them. Neither is bounded, and consolidation never touches either.

	Returns:
		MemoryClient: Connected to a server declaring everything the wiki needs.
	"""
	tiers = default_tiers() + [MemoryTier("sources", kind="semantic", capacity=0)]

	server = MemoryServer(
		stack            = MemoryStack(tiers=tiers),
		name             = "llm-wiki",
		# Stands in for the model. This is the only line that changes when the
		# wiki is maintained by an LLM instead of by sentence selection.
		summarize_fn     = extractive_summarizer(keep=2),
		summarizer_model = None,
	)
	return MemoryClient(Client(LocalTransport(server.dispatcher)))


def main() -> int:
	"""Build a wiki, maintain it, and lint it -- all through `memory/*`.

	Returns:
		int: 0 when every check passed.
	"""
	memory = open_wiki()

	print("\n1. the schema: how the wiki is maintained")

	memory.remember(SCHEMA, tier="procedural", role="system", metadata={"skill": "wiki-conventions"})
	conventions = memory.recall(query="how should a wiki page be written", tier="procedural", limit=1)
	check("the maintainer can read its own conventions",
	      conventions and "rewritten in place" in conventions[0]["content"], conventions)

	print("\n2. raw sources: ingested, indexed, never modified")

	for name, text in SOURCES.items():
		memory.remember(
			text,
			tier       = "sources",
			key        = f"source/{name}",
			uri        = f"file:///wiki/sources/{name}",
			media_type = "text/markdown",
			metadata   = {"source": name},
		)

	held = memory.timeline(tier="sources")
	check("every source is loaded", len(held) == 3, len(held))
	check("each points at its document without the server fetching it",
	      all(r["uri"].startswith("file:///wiki/sources/") for r in held), held)

	print("\n3. ingest: sources become a page, and the page is a fact with an address")

	# The operation Karpathy calls ingest: read the sources on a topic and
	# integrate them into a page. `summarize` reads and writes in one call, and
	# `key` is what makes it an update rather than an accumulation.
	built = memory.summarize(
		query = "deploy key rotation",
		tier  = "sources",
		into  = "semantic",
		key   = "wiki/deploy-keys",
	)
	check("a page was built from the sources", built["written"] == 1, built)
	check("and it read more than one source to do it", built["read"] >= 2, built)
	check("the sources were not consumed", len(memory.timeline(tier="sources")) == 3)

	page = memory.fetch("wiki/deploy-keys")
	check("the page is addressable", page is not None and page["key"] == "wiki/deploy-keys", page)
	check("it lives in the wiki tier, not with the sources",
	      page and page["tier"] == "semantic", page)

	print("\n4. a source changes: the page is rewritten, not duplicated")

	# The correction Karpathy's lint step is meant to catch. Re-ingesting the
	# source replaces it -- same key, same id, revision + 1 -- and rebuilding
	# the page replaces that too.
	memory.remember(
		"Deploy keys rotate every thirty days as of March. Rotation is automatic. "
		"The rotation is announced on the release channel one week ahead.",
		tier     = "sources",
		key      = "source/rfc-deploy-keys.md",
		uri      = "file:///wiki/sources/rfc-deploy-keys.md",
		metadata = {"source": "rfc-deploy-keys.md", "revised": True},
	)
	check("the source was corrected in place", len(memory.timeline(tier="sources")) == 3)

	memory.summarize(query="deploy key rotation", tier="sources",
	                 into="semantic", key="wiki/deploy-keys")

	pages = memory.timeline(key_prefix="wiki/")
	check("there is still exactly one page on the topic", len(pages) == 1, pages)
	check("and it advanced a revision rather than forking",
	      pages and pages[0]["revision"] >= 1, pages)

	rebuilt = memory.fetch("wiki/deploy-keys")
	check("the page reflects the correction",
	      rebuilt and "thirty days" in rebuilt["content"], rebuilt)
	check("and the superseded claim is gone from it",
	      rebuilt and "ninety days" not in rebuilt["content"], rebuilt)
	print("       ^ on an append-only store both claims would still be recallable,")
	print("         with identical confidence. That is the failure a wiki cannot have.")

	print("\n5. query: and the answer becomes a page")

	answer = memory.recall(query="how often do deploy keys rotate", limit=3)
	check("the wiki answers from synthesis, not from raw sources",
	      answer and any(r.get("key") == "wiki/deploy-keys" for r in answer), answer)

	# "Valuable query results become new wiki pages, so explorations compound."
	memory.summarize(query="on-call release channel", tier="sources",
	                 into="semantic", key="wiki/on-call")
	check("an exploration compounded into a second page",
	      len(memory.timeline(key_prefix="wiki/")) == 2, memory.timeline(key_prefix="wiki/"))

	print("\n6. lint: walk what changed and re-check it")

	# The maintenance loop. A cursor gives exactly the pages touched since the
	# last pass -- no scanning, no timestamps to compare, and nothing missed.
	cursor = memory.events()["cursor"]

	memory.remember(
		"On-call handover moved to tuesday.",
		tier     = "sources",
		key      = "source/handbook-oncall.md",
		uri      = "file:///wiki/sources/handbook-oncall.md",
		metadata = {"source": "handbook-oncall.md"},
	)

	changed = [event for event in memory.events(cursor=cursor)["events"]
	           if event.get("kind") == "written" and str(event.get("key", "")).startswith("source/")]
	check("lint sees exactly which sources changed", len(changed) == 1, changed)
	check("and knows it was a correction, not a new document",
	      changed and changed[0].get("revision", 0) >= 1, changed)

	# A real lint pass would now rebuild every page citing that source. Here it
	# is one page, and rebuilding it is the same call as before.
	memory.summarize(query="on-call handover release channel", tier="sources",
	                 into="semantic", key="wiki/on-call")
	relinted = memory.fetch("wiki/on-call")
	check("the affected page was rebuilt", relinted and relinted["revision"] >= 1, relinted)

	print("\n7. the layers stayed apart")

	# The property that makes the whole thing maintainable: pressure in the
	# conversation cannot reach the wiki, the sources or the conventions.
	for turn in range(40):
		memory.remember(f"unrelated conversational turn {turn}", session="chat")
	memory.consolidate()

	check("the wiki survived consolidation", len(memory.timeline(key_prefix="wiki/")) == 2)
	check("so did the sources", len(memory.timeline(tier="sources")) == 3)
	check("and the conventions were never touched",
	      len(memory.timeline(tier="procedural")) == 1)

	print(f"\n       pages: {[r['key'] for r in memory.timeline(key_prefix='wiki/')]}")
	print( "       Nothing above is an A2M extension. The wiki is what the")
	print( "       existing capabilities are for when an agent maintains them.")

	memory.close()

	print()
	print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
	for label in FAILED:
		print(f"    - {label}")

	return 1 if FAILED else 0


if __name__ == "__main__":
	sys.exit(main())
