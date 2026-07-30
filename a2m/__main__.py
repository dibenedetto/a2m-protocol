"""The reference A2M server, on the command line.

	python -m a2m                  # stdio, everything in a dict
	python -m a2m my-agent         # stdio, named
	python -m a2m --http 8778      # http

Everything is held in memory and nothing survives the process, which is the
point: this is the stack the conformance suite runs against when the question is
what the *protocol* requires rather than what any storage engine makes
convenient. For a server that survives a restart, see
[implementations/store_sqlite.py](../implementations/store_sqlite.py).

This lives in its own module rather than under `if __name__ == "__main__"` in
protocol.py, because `a2m/__init__.py` imports that module -- running it as a
script too would execute it twice under two names.
"""


import sys


from   a2m.protocol import serve, serve_over_http


def main(argv: list[str] = None) -> int:
	"""Serve A2M over stdio, or over HTTP with --http.

	Args:
		argv (list[str], optional): Arguments after the program name. Defaults to
			sys.argv[1:].

	Returns:
		int: Process exit code.
	"""
	argv = sys.argv[1:] if argv is None else argv

	if "--http" in argv:
		index = argv.index("--http")
		port  = int(argv[index + 1]) if len(argv) > index + 1 else 8778
		serve_over_http(port=port)
	else:
		name = next((a for a in argv if not a.startswith("--")), "agent-memory")
		serve(name=name)

	return 0


if __name__ == "__main__":
	sys.exit(main())
