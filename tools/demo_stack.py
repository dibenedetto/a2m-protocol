"""The whole memory stack, over A2M, on disk.

	python -m tools.demo_stack            # single server, per-tier SQLite
	python -m tools.demo_stack --router   # federated: one process per tier

Everything is checked rather than narrated:

	1. memories survive the process that wrote them
	2. records flow working -> episodic -> semantic, by pressure and by merit
	3. two agents share one file without inheriting each other's transcript
	4. what ranks recall is a choice: lexical, embeddings, or the caller's own
	5. a corpus is ingested, corrected in place, and never duplicated
	6. a skill is installed, read back, and lands on disk as a reviewable file
	7. a set of records is distilled into one durable statement
	8. a cursor sees exactly what changed, in order and exactly once

Nothing here touches storage directly. Every line goes through `memory/*`, which
is why the same script drives a single SQLite server and a four-process
federation without knowing which it is talking to — and why sections 4 to 8 are
skipped, rather than failed, against a federation that declares less.

It runs offline. Section 4 uses a hand-built stand-in embedder so the output is
deterministic and nothing has to be installed; `ollama_embedder()` is what goes
there in production.
"""


import pathlib
import shutil
import subprocess
import sys


from   typing      import Any


from   a2m         import MemoryClient, connect_stdio
from   a2m.jsonrpc import Client, LocalTransport
from   a2m.memory  import MemoryTier


ROOT = pathlib.Path("demo-memories")

TIERS = [
	MemoryTier("working"   , capacity = 4  , spill_to = "episodic", half_life = 300.0,
	                         promote_to = None),
	MemoryTier("episodic"  , capacity = 32 , spill_to = "semantic", half_life = 86400.0,
	                         promote_to = "semantic", promote_after = 2),
	MemoryTier("semantic"  , capacity = 0),
	MemoryTier("procedural", capacity = 0),
]

PASSED : list[str] = []
FAILED : list[str] = []


def check(label: str, condition: bool, detail: Any = "") -> None:
	if condition:
		PASSED.append(label)
		print(f"    ok    {label}")
	else:
		FAILED.append(label)
		print(f"    FAIL  {label}  {detail}")


def counts(memory: MemoryClient) -> dict[str, int]:
	return {t["name"]: t["count"] for t in memory.describe(refresh=True)["tiers"]}


def open_single(path: pathlib.Path) -> tuple[MemoryClient, Any]:
	from a2m                          import MemoryServer
	from a2m.retrieval                import extractive_summarizer
	from implementations.store_sqlite import open_stack

	stack  = open_stack(str(path), tiers=list(TIERS))
	# An extractive summarizer needs no model, so the demo can show `summarize`
	# without anything installed and without leaving the machine.
	server = MemoryServer(stack=stack, name="demo-sqlite", summarize_fn=extractive_summarizer())
	return MemoryClient(Client(LocalTransport(server.dispatcher))), stack


def open_router(root: pathlib.Path) -> tuple[MemoryClient, Any]:
	from implementations.server_federated import MemoryRouter, RankFusion, spawn_backends

	backends = spawn_backends(root, tiers=list(TIERS))
	router   = MemoryRouter(backends, tiers=list(TIERS), merge=RankFusion())
	return MemoryClient(Client(LocalTransport(router.dispatcher))), backends


def demo_persistence(federated: bool) -> None:
	print("\n  1. memories outlive the process that wrote them")

	# A genuinely separate interpreter, not just a reopened handle.
	writer = (
		"import sys; sys.path.insert(0,'.');"
		"from tools.demo_stack import open_single, ROOT, TIERS;"
		"m, s = open_single(ROOT / 'single.db');"
		"m.remember('marco lives in bologna and works on compilers');"
		"m.remember('the deploy key rotates every ninety days');"
		"m.remember('always run the tests before pushing', tier='procedural',"
		" metadata={'skill': 'discipline'});"
		"s.close(); print('written')"
	)
	result = subprocess.run([sys.executable, "-c", writer], capture_output=True, text=True)
	check("a separate process wrote and exited", "written" in result.stdout, result.stderr[-200:])

	memory, stack = open_single(ROOT / "single.db")
	try:
		found = memory.recall(query="where does marco live", tier="working")
		check("the memory is still there afterwards", bool(found), found)
		check("and it is the right one", found and "bologna" in found[0]["content"], found)

		procedures = memory.timeline(tier="procedural")
		check("procedural memory survived too", bool(procedures), procedures)

		files = list((ROOT / "single.procedural").glob("*.md")) if (ROOT / "single.procedural").exists() else []
		check("a procedure is a file on disk, not a row", bool(files), files)
	finally:
		stack.close()


def demo_flow(memory: MemoryClient) -> None:
	print("\n  2. records flow down the stack, by pressure and by merit")

	for i in range(4):
		memory.remember(f"turn {i}: discussing the deployment pipeline", group=f"turn-{i}")

	before = counts(memory)
	check("working filled to capacity", before["working"] == 4, before)

	memory.remember("marco prefers tabs over spaces", group="preference")
	memory.consolidate()

	after = counts(memory)
	check("spilling relieved the pressure", after["working"] <= 4, after)
	check("what spilled landed in episodic", after["episodic"] > 0, after)
	check("nothing was lost", sum(after.values()) >= sum(before.values()), (before, after))

	# Eviction takes the *weakest* group, which with equal salience means the
	# oldest -- so whatever reached episodic is whatever the stack chose, not
	# whatever this script expected. Take it as given and promote that.
	spilled = memory.timeline(tier="episodic")
	check("something reached episodic to promote", bool(spilled), spilled)
	if not spilled:
		return

	candidate = spilled[0]
	print(f"      (the stack evicted {candidate['content'][:34]!r} -- oldest, as it should)")

	# Merit: recalling a record repeatedly is what earns it a durable tier.
	# `promote_after` is 2, and each recall counts as an access.
	for _ in range(3):
		memory.recall(query=candidate["content"], tier="episodic")

	report = memory.consolidate()
	landed = [r["id"] for r in memory.timeline(tier="semantic")]

	check("a repeatedly recalled record earned semantic",
	      candidate["id"] in landed or report.get("promoted", 0) > 0,
	      {"report": report, "semantic": landed})


def demo_sharing(memory: MemoryClient) -> None:
	print("\n  3. two agents, one store, private transcripts")

	alice = memory.for_agent("alice")
	bob   = memory.for_agent("bob")

	alice.remember("alice is reviewing the payment service")
	bob.remember("bob is debugging the scheduler")

	mine   = [r["content"] for r in alice.timeline(tier="working")]
	theirs = [r["content"] for r in bob.timeline(tier="working")]

	check("alice sees her own turn" , any("payment" in c for c in mine), mine)
	check("alice does not see bob's", not any("scheduler" in c for c in mine), mine)
	check("bob does not see alice's", not any("payment" in c for c in theirs), theirs)

	# Spec §6: a scoped reader sees shared tiers, its own records, *and* records
	# nobody claimed -- so the unscoped writes earlier in this demo are legitimately
	# visible to alice. What matters is that her own writes carry her name, and
	# that nothing owned by bob leaks in.
	hers   = [r for r in alice.timeline(tier="working") if "payment" in r["content"]]
	others = [r for r in alice.timeline(tier="working") if r.get("owner") not in (None, "alice")]

	check("alice's own writes are stamped with her name",
	      hers and all(r.get("owner") == "alice" for r in hers), hers)
	check("no other agent's records leak in", not others, others)

	# Spilling carries a private turn into a shared tier: what one agent
	# experienced becomes what the team knows, provenance intact.
	for i in range(6):
		alice.remember(f"alice filler {i}")
	memory.consolidate()

	pooled = bob.recall(query="alice reviewing payment service", tier="episodic")
	check("a spilled turn becomes visible to the team", bool(pooled), pooled)
	check("provenance survives the move", pooled and pooled[0].get("owner") == "alice", pooled)


def demo_ranking(memory: MemoryClient) -> None:
	"""What ranks recall is a choice, and the store is honest about which.

	Args:
		memory (MemoryClient): The store under demonstration.
	"""
	print("\n  4. choosing what ranks recall")

	profile = memory.describe(refresh=True)
	print(f"      the store reports its ranker: {profile.get('scorer', {})}")

	# Nothing so far has called a model. The default is idf-weighted term
	# overlap: offline, deterministic, and genuinely good when words match.
	memory.remember("the deploy key rotates every ninety days", tier="semantic")

	shared = memory.recall(query="when does the deploy key rotate", tier="semantic")
	check("lexical ranking finds what shares words", bool(shared), shared)

	missed = memory.recall(query="how often are credentials cycled", tier="semantic")
	check("and misses what says the same thing in other words", not missed, missed)

	if not memory.supports("embeddings"):
		return

	# A caller's own vector is stored verbatim and never regenerated, which is
	# what lets two frameworks using different models share one store.
	near = [1.0, 0.0, 0.0, 0.0]
	memory.remember("credentials are cycled on a quarterly schedule",
	                tier="semantic", embedding=near)

	found = memory.recall(embedding=near, limit=1, embeddings=True, tier="semantic")
	check("a caller-supplied vector searches without any model", bool(found), found)
	check("and comes back exactly as it went in",
	      found and found[0].get("embedding") == near, found)


def demo_corpus(memory: MemoryClient) -> None:
	"""A corpus is ingested, corrected, and never duplicated.

	Args:
		memory (MemoryClient): The store under demonstration.
	"""
	print("\n  5. a corpus, ingested and corrected")

	if not memory.supports("keys"):
		print("      skipped: this server does not declare 'keys'")
		return

	for index, chunk in enumerate([
		"the rollback window is twenty-four hours",
		"after the window the database migration is irreversible",
	]):
		memory.remember(chunk, tier="semantic", group="doc/runbook",
		                key=f"corpus/runbook/{index:03d}",
		                uri="file:///corpus/runbook.md", media_type="text/markdown")

	check("the corpus is loaded", len(memory.timeline(key_prefix="corpus/")) == 2)

	# Re-ingesting the same document replaces its chunks. Without keys the stale
	# passage would still be recallable, with identical confidence.
	memory.remember("the rollback window is four hours", tier="semantic", group="doc/runbook",
	                key="corpus/runbook/000", uri="file:///corpus/runbook.md")

	after = memory.timeline(key_prefix="corpus/runbook/")
	check("re-ingesting replaced rather than duplicated", len(after) == 2, after)
	check("and the stale passage is gone, not outnumbered",
	      all("twenty-four" not in r["content"] for r in after), after)
	check("the server stored the reference without fetching it",
	      all(r.get("uri") == "file:///corpus/runbook.md" for r in after), after)


def demo_skill(memory: MemoryClient) -> None:
	"""A skill is procedural memory: written deliberately, never spilled into.

	Args:
		memory (MemoryClient): The store under demonstration.
	"""
	print("\n  6. installing a skill")

	memory.remember(
		"# Rotating the deploy key\n\n1. Announce.\n2. Update the vault.\n3. Revoke last.\n",
		tier     = "procedural",
		role     = "system",
		metadata = {"skill": "rotate-deploy-key"},
	)

	found = memory.recall(query="how do I rotate the deploy key", tier="procedural", limit=1)
	check("the procedure is recalled at task start", bool(found), found)
	check("and comes back whole, ready to follow",
	      found and "Revoke last" in found[0]["content"], found)

	before = len(memory.timeline(tier="procedural"))
	for turn in range(12):
		memory.remember(f"pressure {turn}", session="noise")
	memory.consolidate()
	check("capacity pressure cannot reach procedural memory",
	      len(memory.timeline(tier="procedural")) == before, before)


def demo_summarize(memory: MemoryClient) -> None:
	"""Records distilled into a durable statement, without losing the originals.

	Args:
		memory (MemoryClient): The store under demonstration.
	"""
	print("\n  7. distilling what was learnt")

	if not memory.supports("summarize"):
		print("      skipped: this server declares no summarizer")
		return

	written = memory.remember(records=[
		{"content": "the incident began when the vault became unreachable", "tier": "episodic"},
		{"content": "deploys were paused rather than forced during the outage", "tier": "episodic"},
	])

	result = memory.summarize(ids=written, into="semantic", key="wiki/vault-outage")
	check("a set of records became one durable statement", result["written"] == 1, result)
	check("it read what it was given", result["read"] == len(written), result)

	# The rule that separates this from consolidation.
	survivors = memory.timeline(tier="episodic")
	check("and the originals were not consumed",
	      all(any(r["id"] == id for r in survivors) for id in written), survivors)

	# Summarizing to the same address maintains one page instead of piling up.
	memory.summarize(ids=written, into="semantic", key="wiki/vault-outage")
	check("summarizing to the same key replaces rather than accumulates",
	      len(memory.timeline(key_prefix="wiki/vault-outage")) == 1)


def demo_events(memory: MemoryClient) -> None:
	"""A cursor sees what changed: in order, exactly once, on any transport.

	Args:
		memory (MemoryClient): The store under demonstration.
	"""
	print("\n  8. watching the stack change")

	if not memory.supports("events"):
		print("      skipped: this server does not declare 'events'")
		return

	cursor = memory.events()["cursor"]
	check("a poll with no cursor yields a position, not a backlog",
	      isinstance(cursor, str) and bool(cursor), cursor)

	ids = memory.remember("a fact worth watching for", tier="semantic")
	seen = memory.events(cursor=cursor)["events"]
	check("the write shows up on the cursor",
	      any(e.get("kind") == "written" and e.get("id") == ids[0] for e in seen), seen)

	# Draining is the whole contract: poll again from the returned cursor and
	# the same events are never offered twice.
	cursor = memory.events(cursor=cursor)["cursor"]
	check("and is not offered again", memory.events(cursor=cursor)["events"] == [])

	# A consolidation is one coalesced event, however many records it moves.
	before = memory.events()["cursor"]
	memory.consolidate()
	bulk = [e for e in memory.events(cursor=before)["events"] if e.get("kind") == "consolidated"]
	check("a consolidation is one event, not one per record", len(bulk) == 1, bulk)


def main() -> int:
	federated = "--router" in sys.argv

	if ROOT.exists():
		shutil.rmtree(ROOT, ignore_errors=True)
	ROOT.mkdir(parents=True, exist_ok=True)

	print(f"\n  A2M memory stack demo -- {'federated, one process per tier' if federated else 'single server, per-tier SQLite'}")

	demo_persistence(federated)

	if federated:
		memory, handle = open_router(ROOT / "federated")
		closer = lambda: [b.close() for b in handle.values()]
	else:
		memory, handle = open_single(ROOT / "flow.db")
		closer = handle.close

	try:
		profile = memory.describe()
		print(f"\n  server: {profile['name']}  capabilities: {profile['capabilities']}")
		print(f"  storage: {profile.get('scorer', {})}")

		demo_flow(memory)
		demo_sharing(memory)
		demo_ranking(memory)
		demo_corpus(memory)
		demo_skill(memory)
		demo_summarize(memory)
		demo_events(memory)

		print(f"\n  final tiers: {counts(memory)}")
	finally:
		closer()

	print(f"\n  {len(PASSED)} passed, {len(FAILED)} failed")
	for label in FAILED:
		print(f"    - {label}")

	return 1 if FAILED else 0


if __name__ == "__main__":
	sys.exit(main())
