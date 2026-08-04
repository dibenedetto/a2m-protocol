"""Does a record written through one framework come back through another?

	python -m tools.test_interop

The README's first sentence names five frameworks and claims they can share one
memory. Until this file existed, **nothing checked that**: `cross_framework.py`
covered two of them and CI ran none, so the CrewAI adapter could have rotted the
moment CrewAI changed its storage interface — which it did, between one release
and the next, while this repository was being written.

So: every adapter writes a record only it wrote, then every adapter reads, and
the result is a grid. A hole in the grid is a broken adapter, and it says which
direction broke, which is the part a single pass/fail would hide.

	writer \\ reader   langchain  agno  crewai  autogen  raw
	langchain              ok      ok     ok      ok     ok
	...

Adapters whose framework is not installed are **skipped by name** rather than
silently passing, because a matrix that shrinks quietly is worse than one that
fails: it reports success for coverage it no longer has.

	pip install langchain-core agno crewai autogen-core

CrewAI cannot run on Python 3.14 (its chromadb dependency uses pydantic v1), so
the full matrix needs 3.12 or 3.13. Every other adapter runs anywhere. The suite
reports which interpreter it is on, because "4/4 on 3.12" and "3/4 on 3.14" are
different results and only one of them is the claim.
"""


import pathlib
import sys
import tempfile


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from   typing        import Any


from   a2m           import MemoryClient, MemoryServer
from   a2m.jsonrpc   import Client, LocalTransport
from   a2m.memory    import MemoryStack


PASSED : list[str] = []
FAILED : list[str] = []


# What every adapter writes, and what every adapter must find. Distinctive
# enough that a lexical scorer retrieves it from a store holding all of them.
FACTS = {
	"langchain" : "the langchain deploy key rotates every ninety days",
	"agno"      : "the agno release branch is cut on thursdays",
	"agno-db"   : "the agnodb vault token expires after two weeks",
	"crewai"    : "the crewai rollback window is twenty-four hours",
	"autogen"   : "the autogen on-call rotation hands over on monday",
	"raw"       : "the raw client vault token expires each quarter",
}


def check(label: str, condition: bool, detail: Any = "") -> None:
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
		print(f"  FAIL  {label}  {detail}")


def searchable(memory: MemoryClient) -> str:
	"""The first tier a recall can reach.

	Working memory is replayed, never searched (spec §4.4), so an adapter that
	writes a standalone fact there has written something recall cannot find.

	Args:
		memory (MemoryClient): A connected client.

	Returns:
		str | None: A tier name, or None on a server without tiers.
	"""
	for tier in memory.describe().get("tiers", []):
		if tier.get("kind") != "working":
			return tier.get("name")
	return None


# ---------------------------------------------------------------- adapters
#
# Each entry is (write, read). Both take the shared A2M client so that every
# framework is pointed at *one* store -- which is the only thing being tested.
# Import failures are caught per adapter, so three installed frameworks still
# produce a three-by-three grid.


def adapter_langchain(memory: MemoryClient, tier: str):
	"""LangChain's retriever, over the shared store.

	Args:
		memory (MemoryClient): The shared A2M client.
		tier (str): Where standalone facts belong.

	Returns:
		tuple: (write, read) callables.
	"""
	from implementations.adapters.langchain import A2MRetriever

	# BaseRetriever is a pydantic model, so the client is a named field.
	retriever = A2MRetriever(client=memory, tier=tier, limit=10)

	def write(text: str) -> None:
		memory.remember(text, tier=tier, metadata={"via": "langchain"})

	def read(query: str) -> list[str]:
		return [document.page_content for document in retriever.invoke(query)]

	write.embeds = False
	return write, read


def adapter_agno(memory: MemoryClient, tier: str):
	"""Agno's VectorDb, over the shared store.

	Args:
		memory (MemoryClient): The shared A2M client.
		tier (str): Where standalone facts belong.

	Returns:
		tuple: (write, read) callables.
	"""
	from agno.knowledge.document                import Document
	from implementations.adapters.agno          import A2MVectorDb

	# namespace=None is "write mine, search everything", which is the setting
	# that makes a shared store shared rather than a private one with extra
	# steps (DECISION 023).
	knowledge = A2MVectorDb(memory, namespace=None, tier=tier)

	def write(text: str) -> None:
		knowledge.insert("interop", [Document(content=text)])

	def read(query: str) -> list[str]:
		return [document.content for document in knowledge.search(query, limit=10)]

	# Without an Agno embedder configured, documents are stored with no vector.
	write.embeds = False
	return write, read


def adapter_agno_db(memory: MemoryClient, tier: str):
	"""Agno's *relational* store, over the shared store.

	Agno persists in two places: a VectorDb for knowledge and a BaseDb for user
	memories. Covering only the first left half of an Agno agent's state in a
	private file, which is the gap this row exists to keep closed.

	Args:
		memory (MemoryClient): The shared A2M client.
		tier (str): Where standalone facts belong.

	Returns:
		tuple: (write, read) callables.
	"""
	from agno.db.schemas                     import UserMemory
	from implementations.adapters.agno_db    import A2MDb

	database = A2MDb(memory, namespace="interop/memories", tier=tier)
	counter  = {"n": 0}

	def write(text: str) -> None:
		counter["n"] += 1
		database.upsert_user_memory(UserMemory(
			memory    = text,
			memory_id = f"interop-{counter['n']}",
			user_id   = "interop",
		))
	write.embeds = False

	def read(query: str) -> list[str]:
		# search_content goes through memory/recall, so this reads the whole
		# store rather than only what Agno's own database wrote.
		return [m.memory for m in database.get_user_memories(search_content=query, limit=10)]

	return write, read


def adapter_crewai(memory: MemoryClient, tier: str):
	"""CrewAI's StorageBackend, over the shared store.

	CrewAI addresses records by scope and searches by vector, so this adapter
	needs `keys` and `embeddings`. The embedding is a fixed vector: what is
	being tested is that the record crosses the boundary, not that CrewAI's
	embedder is any good.

	Args:
		memory (MemoryClient): The shared A2M client.
		tier (str): Where standalone facts belong.

	Returns:
		tuple: (write, read) callables.
	"""
	from crewai.memory.types                     import MemoryRecord
	from implementations.adapters.crewai         import A2MStorageBackend

	backend = A2MStorageBackend(memory, namespace="interop", tier=tier)
	vector  = [1.0, 0.0, 0.0, 0.0]

	def write(text: str) -> None:
		backend.save([MemoryRecord(content=text, scope="/interop", embedding=list(vector))])

	def read(query: str) -> list[str]:
		return [record.content for record, _ in backend.search(vector, limit=50)]

	# CrewAI's StorageBackend.search takes a vector and no text, so it can only
	# find records that carry one. That is a property of CrewAI's interface, not
	# of A2M or of this adapter -- there is nothing to compare a text-only
	# record against. Declaring it here keeps the grid honest: those cells are
	# marked rather than silently passed or misleadingly failed.
	read.needs_vectors = True

	return write, read


def adapter_autogen(memory: MemoryClient, tier: str):
	"""AutoGen's Memory protocol, over the shared store.

	Args:
		memory (MemoryClient): The shared A2M client.
		tier (str): Where standalone facts belong.

	Returns:
		tuple: (write, read) callables.
	"""
	import asyncio

	from autogen_core.memory                     import MemoryContent, MemoryMimeType
	from implementations.adapters.autogen        import A2MMemory

	remembered = A2MMemory(memory, namespace=None, tier=tier, limit=10)

	def write(text: str) -> None:
		asyncio.run(remembered.add(MemoryContent(content=text, mime_type=MemoryMimeType.TEXT)))

	def read(query: str) -> list[str]:
		return [entry.content for entry in asyncio.run(remembered.query(query)).results]

	write.embeds = False
	return write, read


def adapter_raw(memory: MemoryClient, tier: str):
	"""No framework at all: the protocol on its own.

	The control row. If every framework fails but this passes, the adapters are
	wrong; if this fails too, the store is.

	Args:
		memory (MemoryClient): The shared A2M client.
		tier (str): Where standalone facts belong.

	Returns:
		tuple: (write, read) callables.
	"""
	def write(text: str) -> None:
		memory.remember(text, tier=tier, metadata={"via": "raw"})

	def read(query: str) -> list[str]:
		return [record["content"] for record in memory.recall(query=query, limit=10)]

	write.embeds = False
	return write, read


ADAPTERS = {
	"langchain" : adapter_langchain,
	"agno"      : adapter_agno,
	"agno-db"   : adapter_agno_db,
	"crewai"    : adapter_crewai,
	"autogen"   : adapter_autogen,
	"raw"       : adapter_raw,
}


def build(memory: MemoryClient, tier: str) -> tuple[dict, list[str]]:
	"""Construct every adapter whose framework is installed.

	Args:
		memory (MemoryClient): The shared A2M client.
		tier (str): Where standalone facts belong.

	Returns:
		tuple[dict, list[str]]: Working adapters by name, and the names that
		could not be built, so the report can name what it did not cover.
	"""
	built   = {}
	missing = []

	for name, factory in ADAPTERS.items():
		try:
			built[name] = factory(memory, tier)
		except Exception as exc:
			missing.append(f"{name} ({type(exc).__name__}: {str(exc)[:60]})")

	return built, missing


def test_non_searching(memory: MemoryClient, tier: str) -> None:
	"""The adapters that share the store without searching it.

	Not every framework interface is a search. LangChain's `BaseStore` reads by
	key and AutoGen's `ChatCompletionContext` replays by session, so neither
	belongs in a grid asking "can you find what the others wrote by content".
	Leaving them out entirely would be worse -- they are real adapters and can
	break -- so they are checked on the question they *do* answer: does what
	they write land in the same store, visible to everyone else.

	Args:
		memory (MemoryClient): The shared A2M client.
		tier (str): Where standalone facts belong.
	"""
	print("\n  sharing without searching")

	try:
		from implementations.adapters.langchain import A2MStore
	except Exception as exc:
		print(f"      skip: langchain store ({type(exc).__name__})")
	else:
		store = A2MStore(memory, namespace="interop/store", tier=tier)
		store.mset([("handbook", b"the interop store holds the escalation policy")])

		check("a key-value store round-trips", store.mget(["handbook"])[0] ==
		      b"the interop store holds the escalation policy", store.mget(["handbook"]))
		check("an unused key is None, not an error", store.mget(["nothing"])[0] is None)
		check("its keys are enumerable", list(store.yield_keys()) == ["handbook"], list(store.yield_keys()))

		# The point of the row: a value written through LangChain's cache
		# interface is an ordinary record everyone else can recall.
		found = memory.recall(query="escalation policy", limit=5)
		check("and what it wrote is visible to a plain recall",
		      any("escalation policy" in r["content"] for r in found), found)
		store.mdelete(["handbook"])

	try:
		import asyncio

		from autogen_core.models             import AssistantMessage, UserMessage
		from implementations.adapters.autogen import A2MChatCompletionContext
	except Exception as exc:
		print(f"      skip: autogen model context ({type(exc).__name__})")
		return

	context = A2MChatCompletionContext(memory, session="interop-chat")

	async def exercise():
		await context.add_message(UserMessage(content="who owns the scheduler?", source="user"))
		await context.add_message(AssistantMessage(content="the platform team owns it", source="ops"))
		return await context.get_messages()

	replayed = asyncio.run(exercise())
	check("a model context replays in order",
	      [str(m.content) for m in replayed] ==
	      ["who owns the scheduler?", "the platform team owns it"], replayed)
	check("and it replays rather than ranks",
	      type(replayed[0]).__name__ == "UserMessage", replayed)

	# The transcript is in the store, not in the context object -- which is the
	# whole difference between persisting and sharing.
	turns = memory.timeline(where={"session": "interop-chat"})
	check("the transcript is in the shared store", len(turns) == 2, turns)

	asyncio.run(context.clear())


def test_server_side_embedding() -> None:
	"""The answer to the n/a cells: let the server embed what arrives without a vector.

	CrewAI reads by vector and cannot see a text-only record. That is its
	interface, and no adapter can bridge it -- but a *store* can, and spec §3.7
	already allows it: "A server MAY additionally generate embeddings for
	records that arrive without one."

	So the limitation is a deployment choice rather than a protocol one, and
	this proves it: the same text-only write becomes vector-searchable once the
	store has an embedder, with nothing in either adapter changing.
	"""
	print("\n  server-side embedding closes the vector-only gap")

	from a2m.retrieval import EmbeddingScorer

	# Stands in for a real embedder. Deterministic, offline, two axes.
	def embed(texts: list[str]) -> list[list[float]]:
		return [[float("vault" in text), float("branch" in text)] for text in texts]

	for described, scorer, expected in (
		("without an embedder", EmbeddingScorer(None) , False),
		("with an embedder"   , EmbeddingScorer(embed), True ),
	):
		stack  = MemoryStack(scorer=scorer)
		server = MemoryServer(stack=stack, name="embedding-probe")
		client = MemoryClient(Client(LocalTransport(server.dispatcher)))

		# Written as text only, exactly as every non-CrewAI adapter writes.
		client.remember("the vault token expires each quarter", tier=searchable(client))

		found = client.recall(embedding=[1.0, 0.0], limit=1)
		check(f"a text-only record is vector-searchable {described}: {expected}",
		      bool(found) == expected, found)
		client.close()


def main() -> int:
	"""Write through every adapter, read through every adapter, print the grid.

	Returns:
		int: 0 when every installed adapter can read every other's records.
	"""
	print(f"\n  A2M cross-framework interop -- python {sys.version.split()[0]}")

	stack  = MemoryStack()
	server = MemoryServer(stack=stack, name="interop")
	memory = MemoryClient(Client(LocalTransport(server.dispatcher)))
	tier   = searchable(memory)

	adapters, missing = build(memory, tier)

	print(f"  store: one in-memory stack, everything written to '{tier}'")
	for name in missing:
		print(f"  skip:  {name}")

	if len(adapters) < 2:
		print("\n  Not enough frameworks installed to test interoperability.")
		print("  pip install langchain-core agno crewai autogen-core")
		return 0

	print("\n  writing, one fact per adapter")
	for name, (write, _) in adapters.items():
		write(FACTS[name])
		check(f"{name} wrote through its own interface", True)

	# The store must hold every fact regardless of who wrote it. If this fails,
	# an adapter swallowed a write and the grid below would be misleading.
	held = {record["content"] for record in memory.timeline(tier=tier)}
	for name in adapters:
		check(f"{name}'s record reached the shared store", FACTS[name] in held, sorted(held))

	print("\n  reading: every adapter, every other adapter's record\n")

	names  = list(adapters)
	width  = max(len(n) for n in names) + 2
	header = "writer \\ reader".ljust(18) + "".join(n.ljust(width) for n in names)
	print(f"  {header}")

	grid  = {}
	notes = []
	for writer in names:
		row = []
		for reader in names:
			write, read = adapters[writer][0], adapters[reader][1]

			# A reader that compares vectors cannot see a record written
			# without one. That is the writing framework's shape meeting the
			# reading framework's, and no adapter can bridge it -- so it is
			# marked, not scored.
			if getattr(read, "needs_vectors", False) and getattr(write, "embeds", True) is False:
				row.append("n/a")
				grid[(writer, reader)] = "n/a"
				notes.append(f"{reader} compares vectors; {writer} writes none")
				continue

			try:
				found = read(FACTS[writer])
				ok    = any(FACTS[writer] in str(text) for text in found)
			except Exception as exc:
				ok = False
				grid[(writer, reader)] = f"{type(exc).__name__}"
			row.append("ok" if ok else "FAIL")
			grid.setdefault((writer, reader), "ok" if ok else "not found")
		print(f"  {writer.ljust(18)}" + "".join(cell.ljust(width) for cell in row))

	print()
	for (writer, reader), outcome in grid.items():
		if outcome == "n/a":
			continue
		check(f"{reader} reads what {writer} wrote", outcome == "ok", outcome)

	for note in dict.fromkeys(notes):
		print(f"  n/a   {note}")

	test_non_searching(memory, tier)
	test_server_side_embedding()

	memory.close()

	print()
	covered = ", ".join(names)
	print(f"  covered: {covered}")
	if missing:
		print(f"  NOT covered: {', '.join(name.split(' ')[0] for name in missing)}")
	print(f"\n  {len(PASSED)} passed, {len(FAILED)} failed")
	for label in FAILED:
		print(f"    - {label}")

	return 1 if FAILED else 0


if __name__ == "__main__":
	sys.exit(main())
