"""Two frameworks, one memory. The claim A2M exists to make, as a script.

	pip install langchain-core agno
	python examples/cross_framework.py

LangChain writes a conversation. Agno, which knows nothing about LangChain and
shares no code with it, reads that conversation back as knowledge and adds its
own. LangChain then retrieves what Agno wrote.

Neither framework was modified. Neither adapter knows the other exists. The only
thing they have in common is an A2M server, which is the entire point: the
problem is not that frameworks lack memory, it is that each one's memory is a
private format the next one cannot read.

What makes this work is not the adapters. It is that both sides agree on what a
record is (spec §3.1) and that a conversation is *replayed* while knowledge is
*searched* (spec §4.4). Those are protocol decisions; the adapters are thin
because the decisions were made in the right place.

Two things this file deliberately does **not** demonstrate, so that what it does
demonstrate can be trusted:

- **No vector crosses this boundary.** Neither framework here is given an
  embedder, so nothing is written with one and the server ranks these records
  however it ranks anything. Caller-owned embeddings are spec §3.7 and they are
  real, but they are exercised in [embedders.py](embedders.py) — "returns the
  caller's vector verbatim, never regenerated" — against the raw client, where
  the claim can be checked rather than assumed.
- **Neither agent is an agent.** Both frameworks are driven through their
  storage interfaces by hand. Whether a shared record actually reaches a running
  agent's *model call* is a different question and a harder one, and
  [agent_interop.py](agent_interop.py) is where it is asked.
"""


import pathlib
import sys
import tempfile


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from implementations import client as a2m_client


from   implementations.adapters.agno      import AgnoA2MVectorDb
from   implementations.adapters.langchain import LangChainA2MChatMessageHistory, LangChainA2MRetriever
from   agno.knowledge.document import Document


PASSED = []
FAILED = []


# Two concept axes, and the words that land on each. Enough to make a vector mean
# something without a model, a network or a download -- which is what lets the
# last section of this file check that a vector really crossed, rather than
# describing an embedder it would have needed an API key to run.
CONCEPTS = {
	"deploy"      : 0, "key"        : 0, "keys"    : 0, "rotate" : 0, "rotates" : 0,
	"rotation"    : 0, "ninety"     : 0, "days"    : 0, "day"    : 0,
	"credentials" : 0, "cycled"     : 0,

	"fallback"    : 1, "region"     : 1, "regions" : 1, "traffic": 1,
	"eu-central-1": 1, "datacentre" : 1, "outage"  : 1, "failover": 1,
}


class ConceptEmbedder:
	"""A deterministic embedder both frameworks are given, and the point of it.

	The interesting thing here is not the arithmetic, which is a word count. It
	is that *one object* is handed to Agno's knowledge base and to LangChain's
	retriever, so the vectors in the store and the vector a query is ranked
	against are in the same space by construction rather than by luck.

	A2M is what makes that possible and not merely convenient: a caller's vector
	is stored verbatim and never regenerated (spec §3.7). A store that re-embedded
	on write would silently move Agno's documents into its own model's space, and
	LangChain's query vector would then be pointing at nothing.

	`get_embedding` is Agno's embedder method name; `LangChainA2MRetriever` looks for it
	too, so the same object satisfies both without a wrapper.
	"""

	def get_embedding(self, text: str) -> list:
		"""Embed one text onto the concept axes.

		Args:
			text (str): What to embed.

		Returns:
			list[float]: One weight per axis.
		"""
		vector = [0.0] * 2

		for word in str(text).lower().replace("?", " ").replace(",", " ").split():
			axis = CONCEPTS.get(word.strip("."), None)
			if axis is not None:
				vector[axis] += 1.0

		return vector


def check(label: str, condition: bool, detail=None) -> None:
	"""Record one assertion.

	Args:
		label (str): What was being checked.
		condition (bool): Whether it held.
		detail (Any, optional): Printed when it did not.
	"""
	(PASSED if condition else FAILED).append(label)
	print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
	if not condition and detail is not None:
		print(f"        {detail}")


def main() -> int:
	"""Run the cross-framework demonstration.

	Returns:
		int: 0 when every check passed.
	"""
	database = pathlib.Path(tempfile.mkdtemp(prefix="a2m-cross-")) / "shared.db"

	print(f"\n  one A2M server at {database.name}, two frameworks\n")

	client = a2m_client.connect_stdio([sys.executable, "-m", "implementations.store_sqlite", str(database)])

	try:
		# ---------------------------------------------------- LangChain writes
		print("  langchain writes a conversation")

		history = LangChainA2MChatMessageHistory(client, session="support-42")
		history.add_user_message("our deploy key rotates every ninety days")
		history.add_ai_message("noted -- ninety day rotation for the deploy key")

		check("langchain sees its own turns", len(history.messages) == 2, history.messages)

		# ------------------------------- Agno cannot see it yet, and should not
		print("\n  agno looks, and correctly finds nothing")

		# namespace=None: this knowledge base writes its own records but searches
		# the whole store. Scoped to a namespace it would only ever find what it
		# wrote, which is the silo A2M exists to remove.
		knowledge = AgnoA2MVectorDb(client, namespace=None, tier="episodic")

		found = knowledge.search("how often does the deploy key rotate", limit=5)
		check("a live conversation is not yet knowledge",
		      not any("ninety days" in document.content for document in found),
		      [d.content for d in found])

		# This is the tier model, not a bug. Those turns are in working memory,
		# which is replayed and never searched (spec §4.4): ranking a transcript
		# destroys the thing that made it a transcript. A conversation becomes
		# searchable by moving down the chain, not by being indexed where it lies.

		# ------------------------------------------- the conversation finishes
		print("\n  the conversation ends, and percolates")

		client.close_session("support-42")

		# spec §4.11 -- closing percolates the whole stack rather than one hop. A
		# conversation that will never be replayed leaves working memory at once
		# instead of waiting for capacity pressure, and each tier below applies its
		# own rules to what arrives.

		found = knowledge.search("how often does the deploy key rotate", limit=5)
		check("now agno finds what langchain said",
		      any("ninety days" in document.content for document in found),
		      [d.content for d in found])

		# --------------------------------------------------------- Agno writes
		print("\n  agno adds knowledge of its own")

		knowledge.insert("runbook-1", [Document(
			content    = "the fallback region is eu-central-1",
			content_id = "runbook-fallback",
			name       = "fallback-region",
		)])

		# --------------------------------------------------- LangChain reads it
		print("\n  langchain retrieves what agno wrote")

		retriever = LangChainA2MRetriever(client=client, limit=5, tier="episodic")
		documents = retriever.invoke("which fallback region do we use")

		check("langchain finds what agno wrote",
		      any("eu-central-1" in document.page_content for document in documents),
		      [d.page_content for d in documents])

		# Text crossing is the easy half. The interesting question is what *else*
		# survived: Agno addresses documents by content_id and name, A2M addresses
		# facts by key (spec §3.6), and the adapter keeps the rest in metadata --
		# which LangChain hands back untouched. So the document is still
		# identifiable on the other side, not just readable.
		crossed = next((d for d in documents if "eu-central-1" in d.page_content), None)
		check("and agno's identity for it survived the crossing",
		      crossed is not None
		      and crossed.metadata.get("agno_content_id") == "runbook-fallback"
		      and crossed.metadata.get("agno_name")       == "fallback-region",
		      crossed.metadata if crossed else None)

		# ------------------------------------------- and the vector crosses too
		print("\n  and the vector crosses too")

		# The same embedder object on both sides. Agno embeds what it writes;
		# LangChain embeds what it asks. Neither knows the other exists, and the
		# only reason their vectors are comparable is that the server stored
		# Agno's verbatim instead of making one of its own (spec §3.7).
		embedder = ConceptEmbedder()

		vectored = AgnoA2MVectorDb(client, namespace=None, tier="episodic", embedder=embedder)
		vectored.upsert("runbook-2", [Document(
			content    = "traffic failover moves to the eu-central-1 region",
			content_id = "runbook-failover",
			name       = "failover-plan",
		)])

		# Not one word of this appears in the document above. Whatever finds it,
		# it will not be the text.
		question = "which datacentre takes over during an outage?"
		document = "traffic failover moves to the eu-central-1 region"

		check("the question shares no word with the document",
		      not (set(question.lower().strip("?").split()) & set(document.lower().split())),
		      set(question.lower().strip("?").split()) & set(document.lower().split()))

		# The same retriever twice, differing only in whether it was given the
		# embedder. Without the blind one the assertion below would be worth
		# nothing: a store this small could return the right document by accident,
		# and a test that cannot fail is not evidence. It comes back empty, which
		# is the sharpest version of the result -- there is no lexical path to
		# this document at all, so the vector is the only thing that found it.
		blind  = LangChainA2MRetriever(client=client, limit=3, tier="episodic")
		seeing = LangChainA2MRetriever(client=client, limit=3, tier="episodic", embedder=embedder)

		unranked = blind.invoke(question)
		ranked   = seeing.invoke(question)

		check("text alone finds nothing at all", not unranked,
		      [d.page_content for d in unranked])

		check("and langchain, searching by vector, finds it first",
		      ranked and ranked[0].page_content == document,
		      [d.page_content for d in ranked])

		# The other half of §3.7, checked rather than assumed: what came back is
		# the vector Agno computed, not one the store made on the way in. A store
		# that re-embedded would have moved this record into its own model's
		# space, and the query vector above would have been pointing at nothing.
		stored = client.recall("", where={"agno_content_id": "runbook-failover"},
		                       tier="episodic", limit=1, embeddings=True)
		check("and the store kept agno's vector verbatim, never regenerating it",
		      stored and stored[0].get("embedding") == embedder.get_embedding(document),
		      stored[0].get("embedding") if stored else None)

		# ------------------------------------------------- and the shared store
		print("\n  the store itself")

		everything = client.timeline(tier="episodic")
		check("one store holds both frameworks' records", len(everything) >= 3, len(everything))

		contents = " ".join(record.get("content", "") for record in everything)
		check("the conversation and the runbook sit in the same tier",
		      "ninety days" in contents and "eu-central-1" in contents, contents[:200])

	finally:
		client.close()

	print(f"\n  {len(PASSED)} passed, {len(FAILED)} failed\n")
	return 1 if FAILED else 0


if __name__ == "__main__":
	sys.exit(main())
