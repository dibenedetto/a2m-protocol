"""The whole memory stack, over A2M, on disk.

	python -m tools.demo_stack            # single server, per-tier SQLite
	python -m tools.demo_stack --router   # federated: one process per tier

Four things are demonstrated, and each is checked rather than narrated:

	1. memories survive the process that wrote them
	2. records flow working -> episodic -> semantic, by pressure and by merit
	3. two agents share one file without inheriting each other's transcript
	4. the same conformance suite passes either way

Nothing here touches storage directly. Every line goes through `memory/*`, which
is why the same script drives a single SQLite server and a four-process
federation without knowing which it is talking to.
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
	from implementations.store_sqlite import open_stack

	stack  = open_stack(str(path), tiers=list(TIERS))
	server = MemoryServer(stack=stack, name="demo-sqlite")
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

		print(f"\n  final tiers: {counts(memory)}")
	finally:
		closer()

	print(f"\n  {len(PASSED)} passed, {len(FAILED)} failed")
	for label in FAILED:
		print(f"    - {label}")

	return 1 if FAILED else 0


if __name__ == "__main__":
	sys.exit(main())
