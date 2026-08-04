"""Putting a corpus into a memory stack: RAG as the degenerate case of A2M.

	python -m examples.rag_ingest

There is no separate RAG component to add, and adding one would duplicate the
stack. Retrieval-augmented generation *is* `memory/recall` followed by putting
the results in the prompt. What differs between a corpus and a memory is only
**who wrote it**:

	                RAG                        agent memory
	written by      ingestion, ahead of time   the conversation, as it happens
	enters at       semantic, directly         working, then spills down
	lifecycle       static until re-ingested   spills, promotes, is forgotten
	chunks          a chunker splits documents a turn is already a record

A document has no `working` phase — nobody said it in a conversation — so it
enters at `semantic`. Three ways to bring one in, in increasing order of
separation, and this file runs the first two:

	1. load it into `semantic`   simplest; corpus and conversation share a tier
	2. give it its own tier      a second tier of kind `semantic`, so neither
	                             can evict the other and recall can ask for one
	3. federate it               wrap the existing RAG stack as an A2M server
	                             and mount it as a tier (implementing-a2m.md §9.4)

The interesting part is not the loading. It is that three capabilities already
in the protocol answer the three questions every ingestion pipeline has to:

	group       chunks of one document must not be separated (spec §3.4)
	keys        re-ingesting must replace, not duplicate (spec §3.6)
	external    the record points at the source; the server never fetches it (§3.8)
"""


import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from   a2m        import connect_local
from   a2m.memory import MemoryStack, MemoryTier, default_tiers


PASSED = []
FAILED = []


# A stand-in corpus. In production these are files, PDFs, wiki pages -- whatever
# an ingestion pipeline already reads. Nothing here is A2M-specific.
CORPUS = {
	"runbook.md": (
		"# Deploy runbook\n\n"
		"The deploy key rotates every ninety days. Rotation is automatic and "
		"announced on the release channel one week ahead.\n\n"
		"## Rollback\n\n"
		"Roll back by re-running the previous release tag. The rollback window "
		"is twenty-four hours, after which the database migration is irreversible.\n"
	),
	"handbook.md": (
		"# Engineering handbook\n\n"
		"The release branch is cut on thursdays. Anything merged after the cut "
		"waits for the following week.\n\n"
		"## On-call\n\n"
		"On-call rotates weekly and hands over on monday morning.\n"
	),
}


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


def chunks_of(text: str) -> list[str]:
	"""Split a document into passages, each carrying the heading above it.

	Deliberately simple -- split on blank lines, then attach a heading to the
	passage it heads. Chunking strategy is entirely the ingester's business and
	A2M has no opinion about it: a chunk is just a record whose `content` is the
	text that gets indexed.

	The heading is folded in rather than left as its own chunk because a lone
	heading is a record that can be recalled and says nothing, while a passage
	that lost its heading is one that no longer knows what it is about. Both are
	retrieval failures a chunker owns, not the protocol.

	Args:
		text (str): The whole document.

	Returns:
		list[str]: Non-empty passages, in order.
	"""
	blocks  = [block.strip() for block in text.split("\n\n") if block.strip()]
	chunks  = []
	heading = ""

	for block in blocks:
		if block.startswith("#"):
			heading = block
			continue
		chunks.append(f"{heading}\n{block}".strip())
		heading = ""

	return chunks


def ingest(memory, name: str, text: str, tier: str = "semantic", version: str = "v1") -> list[str]:
	"""Load one document into a memory stack.

	Every field here is doing a job that an ingestion pipeline would otherwise
	have to invent:

	- **`tier`** puts the document where recall can reach it. A document never
	  passes through working memory, because nobody said it in a conversation.
	- **`group`** ties the chunks of one document together, so a store that
	  evicts under pressure moves them whole rather than leaving a document
	  half-present (spec §3.4).
	- **`key`** addresses each chunk, so re-ingesting the same document
	  **replaces** its chunks instead of appending a second copy. Without this,
	  the stale version stays recallable with exactly the same confidence as the
	  fresh one (spec §3.6).
	- **`uri`** says where the real document lives. The server stores it and
	  **never dereferences it** — resolution belongs to the client, which has
	  the credentials and the reason (spec §3.8).
	- **`metadata`** carries whatever the pipeline needs back at recall time.

	Args:
		memory (MemoryClient): The store to load into.
		name (str): The document's name, used to build its keys and group.
		text (str): The whole document.
		tier (str, optional): Destination tier.
		version (str, optional): Recorded in metadata, so a recall can tell
			which ingestion a passage came from.

	Returns:
		list[str]: The ids written, one per chunk.
	"""
	written = []

	for index, chunk in enumerate(chunks_of(text)):
		written += memory.remember(
			chunk,
			tier       = tier,
			group      = f"doc/{name}",
			key        = f"corpus/{name}/{index:03d}",
			uri        = f"file:///corpus/{name}",
			media_type = "text/markdown",
			metadata   = {"source": name, "chunk": index, "version": version},
		)

	return written


def main() -> int:
	"""Ingest a corpus two ways and show what each buys.

	Returns:
		int: 0 when every check passed.
	"""
	print("\n1. into semantic: the corpus and the conversation share a tier")

	memory = connect_local(MemoryStack())

	for name, text in CORPUS.items():
		ingest(memory, name, text)

	held = memory.timeline(key_prefix="corpus/")
	check("the corpus is loaded", len(held) == 4, len(held))

	found = memory.recall(query="how often does the deploy key rotate?", limit=1)
	check("a passage is recalled by meaning", found and "ninety days" in found[0]["content"], found)
	check("and reports where it came from",
	      found and found[0]["metadata"]["source"] == "runbook.md", found)
	check("and points at the document without the server fetching it",
	      found and found[0]["uri"] == "file:///corpus/runbook.md", found)

	# The agent's own memory lands in the same tier, and is not distinguishable
	# from the corpus by tier alone -- which is fine until you need to forget one
	# without the other. That is what the keys and the metadata are for.
	memory.remember("marco prefers the rollback window kept at twenty-four hours", tier="semantic")

	both = memory.recall(query="rollback window", limit=5)
	check("corpus and conversation are recalled together", len(both) >= 2, both)

	print("\n2. re-ingesting: keys make it a replacement, not a duplicate")

	# The runbook changed. Ingest it again, unchanged pipeline, same keys.
	revised = CORPUS["runbook.md"].replace("every ninety days", "every thirty days")
	ingest(memory, "runbook.md", revised, version="v2")

	after = memory.timeline(key_prefix="corpus/runbook.md/")
	check("re-ingesting did not duplicate the document", len(after) == 2, len(after))
	check("every chunk advanced its revision",
	      all(record["revision"] == 1 for record in after), [r["revision"] for r in after])

	fresh = memory.recall(query="how often does the deploy key rotate?", limit=1)
	check("recall returns the new text", fresh and "thirty days" in fresh[0]["content"], fresh)
	check("and the stale fact is gone, not merely outnumbered",
	      all("ninety days" not in r["content"] for r in memory.recall(query="deploy key rotates", limit=5)),
	      memory.recall(query="deploy key rotates", limit=5))
	print("       ^ this is the failure mode keys exist to remove. An append-only")
	print("         corpus recalls last month's answer with this month's confidence.")

	print("\n3. forgetting one document, and only that one")

	removed = memory.forget(key_prefix="corpus/handbook.md/")
	check("a document can be forgotten by key prefix", removed == 2, removed)
	check("the other document survives", len(memory.timeline(key_prefix="corpus/runbook.md/")) == 2)
	check("and so does the conversation",
	      bool(memory.recall(query="marco prefers the rollback window", limit=1)))

	print("\n4. a dedicated corpus tier: neither can evict the other")

	# A fifth tier of kind `semantic`, unbounded, that nothing spills into and
	# consolidation never touches. Ingested knowledge and learnt knowledge stop
	# competing for the same capacity, and recall can ask for exactly one of them.
	tiers  = default_tiers() + [MemoryTier("corpus", kind="semantic", capacity=0)]
	memory = connect_local(MemoryStack(tiers=tiers))

	for name, text in CORPUS.items():
		ingest(memory, name, text, tier="corpus")

	memory.remember("marco prefers the rollback window kept at twenty-four hours", tier="semantic")

	only = memory.recall(query="rollback window", tier="corpus", limit=5)
	check("recall can ask the corpus alone",
	      only and all(r["tier"] == "corpus" for r in only), only)
	check("and the agent's own memory is not in it",
	      all("marco prefers" not in r["content"] for r in only), only)

	learnt = memory.recall(query="rollback window", tier="semantic", limit=5)
	check("or the agent's memory alone",
	      learnt and all("marco prefers" in r["content"] for r in learnt), learnt)

	# The tier is unbounded and nothing spills into it, so consolidating the
	# stack cannot evict the corpus no matter how busy the conversation gets.
	before = len(memory.timeline(tier="corpus"))
	memory.consolidate()
	check("consolidation cannot displace an unbounded corpus tier",
	      len(memory.timeline(tier="corpus")) == before, before)

	print("\n5. a corpus is not only text")

	# A2M carries an image, a recording or a model with no field for any of
	# them: `uri` is the thing, `media_type` says what it is, and `content` is
	# whatever representation makes it findable -- a caption, a transcript, a
	# description. Producing that representation is the caller's job; the
	# protocol only requires that something ranked ends up in `content`.
	memory = connect_local(MemoryStack())

	media = [
		("diagram.png",  "image/png",        "architecture diagram: the router sits between the agent and four tier backends"),
		("standup.m4a",  "audio/mp4",        "standup recording: the vault outage is resolved and deploys resume on monday"),
		("demo.mp4",     "video/mp4",        "screencast: rotating the deploy key end to end, including the vault update step"),
		("chassis.gltf", "model/gltf+json",  "chassis model: mounting bracket revision C, four M3 holes on a 40mm pitch"),
	]

	for name, media_type, described in media:
		memory.remember(
			described,                          # what gets indexed
			tier       = "semantic",
			key        = f"media/{name}",
			uri        = f"file:///corpus/media/{name}",
			media_type = media_type,            # what is actually at the other end
			metadata   = {"asset": name},
		)

	found = memory.recall(query="how do I rotate the deploy key", limit=1)
	check("a video is recalled by what it is about",
	      found and found[0]["media_type"] == "video/mp4", found)
	check("and the record points at it rather than containing it",
	      found and found[0]["uri"] == "file:///corpus/media/demo.mp4", found)

	drawing = memory.recall(query="which component sits between the agent and the backends", limit=1)
	check("an image is found through its caption",
	      drawing and drawing[0]["media_type"] == "image/png", drawing)

	# The bytes are never in the record, and the server never goes to get them.
	check("no record carries the asset itself",
	      all(len(r["content"]) < 200 for r in memory.timeline(key_prefix="media/")),
	      [len(r["content"]) for r in memory.timeline(key_prefix="media/")])

	# A multimodal vector reaches a picture with no caption at all: the record
	# is ranked on the vector, and `content` can be empty (spec §3.8.1).
	picture = [0.0, 1.0, 0.0, 0.0]
	memory.remember("", tier="semantic", key="media/uncaptioned.png",
	                uri="file:///corpus/media/uncaptioned.png",
	                media_type="image/png", embedding=picture)
	by_vector = memory.recall(embedding=picture, limit=1)
	check("an uncaptioned image is still reachable by vector",
	      by_vector and by_vector[0]["key"] == "media/uncaptioned.png", by_vector)
	print("       ^ same three fields for every medium. A2M has no image type,")
	print("         no audio type and no blob field, and needs none.")

	print("\n   third way, not run here: federate it (implementing-a2m.md §9.4).")
	print("   Wrap the existing RAG stack as an A2M server and mount it as a tier in")
	print("   the router. The corpus keeps its own pipeline, database and release")
	print("   cycle; the agent sees one memory. That is also why classic RAG is the")
	print("   degenerate case of A2M -- one tier, read-only, recall only: implement")
	print("   describe and recall, return -32004 READ_ONLY from remember, and every")
	print("   A2M client works against it.")

	print()
	print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
	for label in FAILED:
		print(f"    - {label}")

	return 1 if FAILED else 0


if __name__ == "__main__":
	sys.exit(main())
