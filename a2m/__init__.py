"""The reference library: the pieces an A2M implementation is built out of.

Importable, in dependency order. Nothing here is normative — [spec/a2m-0.1.md](../spec/a2m-0.1.md)
is. This package exists so the specification has something that demonstrably
runs, and so the implementations in `implementations/` have something to share
instead of each other.

	a2m.text        Unicode tokenizer, CJK segmentation, per-language stopwords
	a2m.retrieval   the scorer seam: lexical, embedding, hybrid; LLM consolidator
	a2m.jsonrpc     JSON-RPC 2.0 from scratch, plus local, stdio and HTTP transports
	a2m.memory      records, tiers, consolidation — the stack, in memory
	a2m.protocol    the A2M server and client: every capability, over any transport

The names below are re-exported so that `from a2m import MemoryServer` works, as
that is the surface an implementer actually reaches for. Everything else is
addressed through its module.

Standard library only, and that is a property worth keeping: a dependency here
would become a dependency of everyone who reads this as an example.
"""


from   a2m.protocol import (
	A2M_CAPABILITIES, A2M_VERSION, CAPABILITY_NOT_SUPPORTED, EMBEDDING_MISMATCH,
	INVALID_PARAMS, PROTOCOL_NOT_SUPPORTED, QUOTA_EXCEEDED, READ_ONLY, SCOPE_DENIED,
	UNKNOWN_RECORD, UNKNOWN_TIER, EventLog, MemoryClient, MemoryServer,
	connect_http, connect_local, connect_stdio, serve_a2m_http, serve_a2m_stdio,
	serve_stdio,
)


__all__ = [
	# spec 9.2 -- the error codes A2M adds to JSON-RPC's own, plus the one from
	# JSON-RPC that A2M handlers raise most.
	"CAPABILITY_NOT_SUPPORTED", "EMBEDDING_MISMATCH", "INVALID_PARAMS",
	"PROTOCOL_NOT_SUPPORTED", "QUOTA_EXCEEDED", "READ_ONLY", "SCOPE_DENIED",
	"UNKNOWN_RECORD", "UNKNOWN_TIER",

	"A2M_CAPABILITIES", "A2M_VERSION", "EventLog", "MemoryClient", "MemoryServer",
	"connect_http", "connect_local", "connect_stdio", "serve_a2m_http",
	"serve_a2m_stdio", "serve_stdio",
]
