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
record is (spec §3.1), that a conversation is *replayed* while knowledge is
*searched* (spec §4.4), and that whoever produced an embedding owns it (spec
§3.7). Those are protocol decisions; the adapters are thin because the decisions
were made in the right place.
"""


import pathlib
import sys
import tempfile


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from implementations import client as a2m_client


from   implementations.adapters.agno      import A2MVectorDb
from   implementations.adapters.langchain import A2MChatMessageHistory, A2MRetriever
from   agno.knowledge.document import Document
from   langchain_core.messages import AIMessage


PASSED = []
FAILED = []


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

		history = A2MChatMessageHistory(client, session="support-42")
		history.add_user_message("our deploy key rotates every ninety days")
		history.add_ai_message("noted -- ninety day rotation for the deploy key")

		check("langchain sees its own turns", len(history.messages) == 2, history.messages)

		# ------------------------------- Agno cannot see it yet, and should not
		print("\n  agno looks, and correctly finds nothing")

		# namespace=None: this knowledge base writes its own records but searches
		# the whole store. Scoped to a namespace it would only ever find what it
		# wrote, which is the silo A2M exists to remove.
		knowledge = A2MVectorDb(client, namespace=None, tier="episodic")

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

		retriever = A2MRetriever(client=client, limit=5, tier="episodic")
		documents = retriever.invoke("which fallback region do we use")

		check("langchain finds what agno wrote",
		      any("eu-central-1" in document.page_content for document in documents),
		      [d.page_content for d in documents])

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
