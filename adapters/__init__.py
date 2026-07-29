"""Framework adapters: existing agents talking to an A2M server unmodified.

Nothing in this package is part of the protocol, and nothing in the protocol
depends on it. Each module imports one framework, and importing this package
imports none of them — so a checkout with no framework installed still runs
every test, every conformance target and both demos.

	adapters.langchain   LangChain chat history and retriever
	adapters.agno        Agno vector database

The point they make together is the one A2M exists for: two frameworks with
incompatible memory models, pointed at one store, reading each other's records.
[examples/cross_framework.py](../examples/cross_framework.py) is that claim as a
runnable script.
"""
