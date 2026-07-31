"""Choosing what ranks recall: lexical, embeddings, hybrid, or the caller's own.

	python -m examples.embedders

A2M deliberately does not specify how a server ranks (spec §5.1) — that is where
implementations are supposed to compete. What it specifies is the *seam*, and
there are two of them, at different layers:

	MemoryStack(scorer=...)      how relevance is judged, in the reference stack
	open_stack(path, embed=...)  what fills the vector index, in the SQL stores

A `Scorer` answers one question — how well does this record match what was
asked, in 0..1 — and knows nothing about recency, salience or tiers, because
those depend on where a record lives. The stack blends them in afterwards.

This example runs offline. A hand-built stub embedder stands in for a real
model so the output is deterministic and nothing has to be installed; every
place it appears is where `ollama_embedder()` goes in production. The model to
put there was chosen by measurement, not reputation — see §6 below and
`tools/bench_embeddings.py`.
"""


import pathlib
import sys
import tempfile


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from   a2m           import connect_local
from   a2m.memory    import MemoryStack
from   a2m.retrieval import EMBEDDING_MODEL, EmbeddingScorer, HybridScorer, LexicalScorer, make_scorer


PASSED = []
FAILED = []


# A stand-in for a real embedding model: five concept axes, so words that share
# a meaning share a dimension. It is not an embedding model and makes no claim
# to be one -- it exists so this file runs with nothing installed, and so the
# difference between lexical and semantic matching is visible in one screen.
# Everywhere it is passed below, `ollama_embedder()` is what goes in production.
CONCEPTS = {
	"rotate" : 0, "rotates": 0, "rotation": 0, "change": 0, "changes": 0, "cycle": 0, "cycled": 0,
	"key"    : 1, "keys"   : 1, "credential": 1, "credentials": 1, "secret": 1, "token": 1,
	"often"  : 2, "ninety" : 2, "days": 2, "schedule": 2, "when": 2, "thursdays": 2, "frequency": 2,
	"deploy" : 3, "release": 3, "branch": 3, "ship": 3, "cut": 3, "deployment": 3,
	"pasta"  : 4, "ragu"   : 4, "recipe": 4, "cook": 4, "hours": 4, "sauce": 4,
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


def stub_embedder(texts: list[str]) -> list[list[float]]:
	"""Embed a batch of texts onto five concept axes.

	The signature is the contract A2M cares about, and the only one: a callable
	taking a list of texts and returning one vector per input, **in order**.
	Batched, because recall embeds a whole backlog at once rather than one
	record at a time.

	Args:
		texts (list[str]): Texts to embed.

	Returns:
		list[list[float]]: One vector per input, in order.
	"""
	vectors = []

	for text in texts:
		vector = [0.0] * 5
		for word in str(text).lower().replace("?", " ").replace(",", " ").split():
			axis = CONCEPTS.get(word.strip("."), None)
			if axis is not None:
				vector[axis] += 1.0
		vectors.append(vector)

	return vectors


def facts(memory) -> None:
	"""Write the same three facts into any stack, in a searchable tier.

	Working memory is replayed, never searched (spec §4.4), so a fact written
	there would be invisible to recall by design. Facts go to `semantic`.

	Args:
		memory (MemoryClient): Where to write.
	"""
	memory.remember("the deploy key rotates every ninety days", tier="semantic")
	memory.remember("the release branch is cut on thursdays"  , tier="semantic")
	memory.remember("pasta al ragu needs three hours"          , tier="semantic")


def main() -> int:
	"""Run every way of choosing a ranker, in order of how much they need.

	Returns:
		int: 0 when every check passed.
	"""
	print("\n1. the default: lexical, no model, nothing to install")

	# MemoryStack() with no scorer is LexicalScorer(): idf-weighted term overlap.
	# Offline, deterministic, free, and genuinely good when the words match.
	memory = connect_local(MemoryStack())
	facts(memory)

	shared = memory.recall(query="when does the deploy key rotate?")
	check("lexical finds what shares words", shared and "ninety days" in shared[0]["content"], shared)

	missed = memory.recall(query="how often are credentials cycled?")
	check("and misses what does not", not missed, missed)
	print("       ^ the same question in other words. No shared terms, no match.")

	print("\n2. embeddings: meaning without shared words")

	# In production: EmbeddingScorer(ollama_embedder()).
	memory = connect_local(MemoryStack(scorer=EmbeddingScorer(stub_embedder)))
	facts(memory)

	found = memory.recall(query="how often are credentials cycled?")
	check("embeddings find what lexical missed", found and "ninety days" in found[0]["content"], found)
	check("and still rank the unrelated fact below",
	      all("pasta" not in r["content"] for r in found[:1]), found)

	print("\n3. hybrid: a union, so neither pass can lose a record")

	# Weighted union, not intersection -- a record only the embedding pass found
	# still surfaces. Weights are renormalised over whichever scorers did not
	# abstain, so the result stays in 0..1 when one has nothing to say.
	memory = connect_local(MemoryStack(scorer=HybridScorer([
		(LexicalScorer()                  , 0.4),
		(EmbeddingScorer(stub_embedder)   , 0.6),
	])))
	facts(memory)

	check("hybrid finds the lexical case" , bool(memory.recall(query="when does the deploy key rotate?")))
	check("hybrid finds the semantic case", bool(memory.recall(query="how often are credentials cycled?")))

	# The same three, by name, so a stack can be configured from a config file
	# rather than from code: make_scorer("hybrid", embed=...).
	check("make_scorer builds the same thing",
	      make_scorer("hybrid", embed=stub_embedder).describe()["scorer"] == "hybrid")

	print("\n4. no model at all: the caller owns the vectors")

	# EmbeddingScorer(None) never embeds anything. It ranks only what callers
	# brought with them -- which is the whole point of caller-owned embeddings
	# (spec §3.7): two frameworks using different models can share one store
	# precisely because neither has its vectors rewritten into the other's space.
	memory = connect_local(MemoryStack(scorer=EmbeddingScorer(None)))

	near = stub_embedder(["the deploy key rotates every ninety days"])[0]
	far  = stub_embedder(["pasta al ragu needs three hours"])[0]

	memory.remember("the deploy key rotates every ninety days", tier="semantic", embedding=near)
	memory.remember("pasta al ragu needs three hours"          , tier="semantic", embedding=far)

	asked = stub_embedder(["how often are credentials cycled?"])[0]
	found = memory.recall(embedding=asked, limit=1)
	check("a store with no model still answers vector search",
	      found and "ninety days" in found[0]["content"], found)

	back = memory.recall(embedding=asked, limit=1, embeddings=True)
	check("and returns the caller's vector verbatim, never regenerated",
	      back and back[0]["embedding"] == near, back)

	print("\n5. the other seam: what fills a SQL store's vector index")

	# The scorer seam is the reference stack's. A store that owns an index takes
	# an embedder instead, and decides for itself which tiers get a vector --
	# working memory never does, because it turns over before anything searches it.
	from implementations.store_sqlite import open_stack

	with tempfile.TemporaryDirectory() as directory:
		stack  = open_stack(str(pathlib.Path(directory) / "memory.db"), embed=stub_embedder)
		try:
			memory = connect_local(stack)
			facts(memory)

			found = memory.recall(query="how often are credentials cycled?")
			check("a SQL store ranks with the embedder it was given",
			      found and "ninety days" in found[0]["content"], found)
		finally:
			# Both SQL stores inherit close() from TieredMemoryStack. Skipping it
			# leaves the database handle open, which on Windows blocks removing
			# the directory it lives in.
			stack.close()

	print("\n6. which model, and why that one")

	# Chosen by measurement, not reputation: bench_embeddings.py ranks ten facts
	# against queries asked in five languages and scores precision@1.
	#
	#   model                     size      same-language   cross-lingual
	#   lexical (no model)        --             76.0%             7.0%
	#   nomic-embed-text          0.27 GB        82.0%            24.5%
	#   granite-embedding:278m    0.56 GB        88.0%            84.5%
	#   paraphrase-multilingual   0.56 GB        94.0%            89.0%
	#   bge-m3                    1.16 GB       100.0%            99.5%
	#
	# bge-m3 costs disk, not latency: 20.2s against 18.6s for the fastest
	# mid-sized model over the same 50 batched calls. An English-only workload
	# can take nomic-embed-text and its 3.7x speed instead.
	check("the measured default is what ollama_embedder() uses", EMBEDDING_MODEL == "bge-m3", EMBEDDING_MODEL)
	print(f"       ollama_embedder() -> {EMBEDDING_MODEL}   (ollama pull {EMBEDDING_MODEL})")
	print( "       one line in production:")
	print( "         from a2m.retrieval import ollama_embedder, make_scorer")
	print( "         MemoryStack(scorer=make_scorer('hybrid', embed=ollama_embedder()))")
	print( "         open_stack('memory.db', embed=ollama_embedder())")

	print()
	print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
	for label in FAILED:
		print(f"    - {label}")

	return 1 if FAILED else 0


if __name__ == "__main__":
	sys.exit(main())
