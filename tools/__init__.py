"""What judges the implementations, rather than being one.

	conformance.py      the conformance suite. Speaks only the protocol.
	test_a2m.py         implementation tests, offline, no test runner.
	demo_stack.py       both topologies, end to end.
	bench_embeddings.py which embedding model backs recall, measured.

`conformance.py` is the important distinction: it imports the JSON-RPC client and
nothing else from this repository, because a suite that imported a server could
only ever confirm that the server agrees with itself. Everything it knows about
A2M it learned from [spec/a2m-0.1.md](../spec/a2m-0.1.md).

Run these as modules from the repository root:

	python -m tools.test_a2m
	python -m tools.conformance --stdio python -m implementations.minimal
"""
