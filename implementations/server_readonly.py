"""A pre-existing corpus, exposed as an A2M server. Read-only, and conformant.

	python implementations/server_readonly.py                # stdio, the sample corpus
	python implementations/server_readonly.py my-docs.json   # stdio, your own

This is the smallest useful A2M server, and the answer to "I already have a
retrieval system — what would it take to join?" It takes `memory/describe`,
`memory/recall` and `memory/timeline`, and `READ_ONLY` from the two methods
that write (spec §2.1). Nothing else. Every A2M client works against it
immediately, because a client that honours `describe` never asks for more than
`core`.

Like [server_minimal.py](server_minimal.py) it **imports nothing from this
repository** — only the standard library — so it runs as a plain file from
anywhere. That is the claim it exists to make: wrapping an existing corpus is
not an adoption project.

**Where your own retrieval goes.** Exactly one function, `search`, and one list,
`CORPUS`. Replace them with a call into Chroma, Pinecone, pgvector, Elasticsearch
or whatever already holds the documents; everything above and below is protocol
plumbing that does not change. The ranking here is deliberately naive term
overlap, because §5.1 leaves ranking entirely to the implementation and a real
corpus already has a ranker better than anything this file should pretend to.

Check it like any other server:

	python -m tools.conformance --stdio python implementations/server_readonly.py

The suite detects that writes are refused and runs its read-only profile, so a
corpus is judged on what it actually promises rather than failed for declining
to be a memory it never claimed to be.
"""


import json
import re
import sys
import uuid


PROTOCOL = "a2m/0.1"

# spec §7. A read-only server needs three of these and no more.
INVALID_REQUEST  = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS   = -32602
INTERNAL_ERROR   = -32603
PARSE_ERROR      = -32700
CAPABILITY_NOT_SUPPORTED = -32003
READ_ONLY                = -32004
PROTOCOL_NOT_SUPPORTED   = -32007

# A stand-in corpus: what an ingestion pipeline would already have loaded.
# Timestamps are RFC 3339 (spec §3.3) and fixed, because a corpus record's
# creation time is when the document was ingested, not when it was served.
CORPUS = [
	{
		"id"         : "doc-runbook-001",
		"content"    : "The deploy key rotates every ninety days. Rotation is automatic "
		               "and announced on the release channel one week ahead.",
		"created_at" : "2026-01-14T09:00:00.000Z",
		"role"       : "memory",
		"metadata"   : {"source": "runbook.md", "section": "keys"},
	},
	{
		"id"         : "doc-runbook-002",
		"content"    : "Roll back by re-running the previous release tag. The rollback "
		               "window is twenty-four hours, after which the database migration "
		               "is irreversible.",
		"created_at" : "2026-01-14T09:00:01.000Z",
		"role"       : "memory",
		"metadata"   : {"source": "runbook.md", "section": "rollback"},
	},
	{
		"id"         : "doc-handbook-001",
		"content"    : "The release branch is cut on thursdays. Anything merged after "
		               "the cut waits for the following week.",
		"created_at" : "2026-02-02T11:30:00.000Z",
		"role"       : "memory",
		"metadata"   : {"source": "handbook.md", "section": "releases"},
	},
	{
		"id"         : "doc-handbook-002",
		"content"    : "On-call rotates weekly and hands over on monday morning.",
		"created_at" : "2026-02-02T11:30:01.000Z",
		"role"       : "memory",
		"metadata"   : {"source": "handbook.md", "section": "on-call"},
	},
]


class A2MError(Exception):
	"""An error carrying the code the specification assigns it."""

	def __init__(self, code: int, message: str, data=None) -> None:
		"""Build one.

		Args:
			code (int): The A2M or JSON-RPC error code.
			message (str): For humans; clients must not parse it.
			data: Structured detail, when there is any.
		"""
		super().__init__(message)
		self.code    = code
		self.message = message
		self.data    = data


def terms(text) -> set:
	"""Split text into indexable terms.

	Args:
		text: Anything stringable.

	Returns:
		set: Lowercased terms of more than one character.
	"""
	return {word for word in re.findall(r"[^\W_]+", str(text).lower(), re.UNICODE) if len(word) > 1}


def search(query: str, limit: int) -> list:
	"""Rank the corpus against a query. **Replace this with your retriever.**

	Everything else in this file is protocol plumbing. This is the seam: take a
	string, return records ranked best-first, each carrying at least `id`,
	`content` and `created_at`. Whether that is a vector search, BM25, a hosted
	API or a hybrid is entirely the implementation's business (spec §5.1).

	Args:
		query (str): What to rank against. Empty means "no opinion", in which
			case recency is a reasonable order (spec §4.3).
		limit (int): Maximum records. 0 means no limit.

	Returns:
		list: Records in wire form, each with a `score`, descending.
	"""
	wanted = terms(query) if query else set()

	if not wanted:
		# spec §4.3 -- an absent query is not an error. Newest first.
		ranked = [dict(record, score=0.0) for record in sorted(CORPUS, key=lambda r: r["created_at"], reverse=True)]
	else:
		ranked = []
		for record in CORPUS:
			held    = terms(record["content"])
			overlap = wanted & held
			if overlap:
				ranked.append(dict(record, score=len(overlap) / len(wanted | held)))
		ranked.sort(key=lambda record: -record["score"])

	return ranked[:limit] if limit and limit > 0 else ranked


def matches(record: dict, where) -> bool:
	"""Apply a `where` filter to one record (spec §5.2).

	Args:
		record (dict): The candidate.
		where: A conjunction of equality tests, or None.

	Returns:
		bool: True when the record satisfies every test.
	"""
	if not where:
		return True

	for key, expected in where.items():
		actual = record.get(key, (record.get("metadata") or {}).get(key))
		if isinstance(expected, list):
			if actual not in expected:
				return False
		elif actual != expected:
			return False

	return True


def describe(params: dict) -> dict:
	"""Handle `memory/describe`.

	`capabilities` says `core` and nothing else, which is the truth: this store
	has no tiers, no salience and no sessions. Declaring less is how a small
	store stays honest, and a client that honours `describe` will never ask for
	what is not here.

	Args:
		params (dict): May carry `protocol`.

	Returns:
		dict: The server profile.

	Raises:
		A2MError: -32007 when the client speaks a version this server does not.
	"""
	declared = params.get("protocol")
	if declared is not None and declared != PROTOCOL:
		raise A2MError(PROTOCOL_NOT_SUPPORTED, f"This server speaks {PROTOCOL}", {"supported": [PROTOCOL]})

	return {
		"protocol"     : PROTOCOL,
		"name"         : "a2m-readonly-corpus",
		"capabilities" : ["core"],
		"methods"      : ["memory/describe", "memory/remember", "memory/recall",
		                  "memory/timeline", "memory/forget"],
		# Advisory, and not part of the protocol: a hint to an operator that
		# this store will never accept a write, without having to try one.
		"read_only"    : True,
		"total"        : len(CORPUS),
	}


def recall(params: dict) -> dict:
	"""Handle `memory/recall` -- the method this server exists for.

	Args:
		params (dict): May carry `query`, `limit`, `where`, `min_score`.

	Returns:
		dict: `records`, descending by score.

	Raises:
		A2MError: -32003 for a field belonging to a capability not declared.
	"""
	for field, capability in (("tier", "tiers"), ("embedding", "embeddings"), ("key_prefix", "keys")):
		if params.get(field) is not None:
			raise A2MError(CAPABILITY_NOT_SUPPORTED, f"This server does not implement the '{capability}' capability")

	found     = search(params.get("query") or "", int(params.get("limit") or 8))
	where     = params.get("where")
	min_score = float(params.get("min_score") or 0.0)

	return {"records": [record for record in found
	                    if matches(record, where) and record["score"] >= min_score]}


def timeline(params: dict) -> dict:
	"""Handle `memory/timeline` -- ascending by creation, newest kept on a limit.

	Args:
		params (dict): May carry `limit`, `where`.

	Returns:
		dict: `records`, ascending by `created_at`.

	Raises:
		A2MError: -32003 for a field belonging to a capability not declared.
	"""
	for field, capability in (("tier", "tiers"), ("key_prefix", "keys")):
		if params.get(field) is not None:
			raise A2MError(CAPABILITY_NOT_SUPPORTED, f"This server does not implement the '{capability}' capability")

	# created_at is RFC 3339, so lexicographic order is chronological order.
	ordered = sorted((record for record in CORPUS if matches(record, params.get("where"))),
	                 key=lambda record: record["created_at"])

	limit = int(params.get("limit") or 0)
	return {"records": ordered[-limit:] if limit > 0 else ordered}


def refuse_write(params: dict) -> dict:
	"""Handle `memory/remember` and `memory/forget` -- spec §2.1.

	Refusing with the code the specification allocates for exactly this *is*
	implementing the method. Both write methods refuse, always and
	consistently: a store that accepted some writes and refused others would
	leave a client no way to tell which is which.

	Args:
		params (dict): Ignored.

	Raises:
		A2MError: -32004, always.
	"""
	raise A2MError(READ_ONLY, "This store is a read-only corpus and does not accept writes")


def unsupported(capability: str):
	"""Build a handler refusing an undeclared capability.

	Every A2M method is registered even when unimplemented: an unregistered one
	would answer -32601 METHOD_NOT_FOUND, which a client cannot distinguish from
	a typo in its own call (spec §2).

	Args:
		capability (str): The capability the method belongs to.

	Returns:
		Callable: A handler that always raises -32003.
	"""
	def handler(params: dict) -> dict:
		raise A2MError(CAPABILITY_NOT_SUPPORTED, f"This server does not implement the '{capability}' capability")
	return handler


HANDLERS = {
	"memory/describe"    : describe,
	"memory/recall"      : recall,
	"memory/timeline"    : timeline,
	"memory/remember"    : refuse_write,
	"memory/forget"      : refuse_write,
	"memory/promote"     : unsupported("tiers"),
	"memory/consolidate" : unsupported("tiers"),
	"memory/reinforce"   : unsupported("salience"),
	"memory/session/list": unsupported("sessions"),
	"memory/session/close": unsupported("sessions"),
	"memory/fetch"       : unsupported("keys"),
	"memory/events"      : unsupported("events"),
	"memory/events/subscribe"  : unsupported("events"),
	"memory/events/unsubscribe": unsupported("events"),
	"memory/summarize"   : unsupported("summarize"),
}


def error_response(id, code: int, message: str, data=None) -> dict:
	"""Build a JSON-RPC error response.

	Args:
		id: The request id, echoed exactly.
		code (int): The error code.
		message (str): For humans.
		data: Structured detail, when there is any.

	Returns:
		dict: A JSON-RPC error response object.
	"""
	body = {"code": code, "message": message}
	if data is not None:
		body["data"] = data
	return {"jsonrpc": "2.0", "id": id, "error": body}


def handle(payload) -> dict:
	"""Turn one parsed request into one response, or None for a notification.

	Args:
		payload: One parsed JSON value.

	Returns:
		dict | None: The response, or None when the request was a notification.
	"""
	# An array here is a JSON-RPC batch, which A2M does not use (spec §8).
	if not isinstance(payload, dict):
		return error_response(None, INVALID_REQUEST, "Request must be a JSON object")

	is_notification = "id" not in payload
	id              = payload.get("id")

	def fail(code, message, data=None):
		return None if is_notification else error_response(id, code, message, data)

	if payload.get("jsonrpc") != "2.0":
		return fail(INVALID_REQUEST, "Expected jsonrpc '2.0'")

	handler = HANDLERS.get(payload.get("method"))
	if handler is None:
		return fail(METHOD_NOT_FOUND, f"Unknown method '{payload.get('method')}'")

	params = payload.get("params") or {}
	if not isinstance(params, dict):
		return fail(INVALID_PARAMS, "'params' must be an object")

	try:
		# spec §2 -- unrecognised parameters are ignored, never rejected.
		# Reading only the fields it knows is how each handler does that.
		result = handler(params)
		return None if is_notification else {"jsonrpc": "2.0", "id": id, "result": result}
	except A2MError as exc:
		return fail(exc.code, exc.message, exc.data)
	except Exception as exc:
		return fail(INTERNAL_ERROR, "Handler failed", str(exc))


def load(path: str) -> None:
	"""Replace the sample corpus with one from a JSON file.

	The file holds a list of objects, each needing at least `content`. Anything
	missing an `id` or a `created_at` is given one, because those two are
	required on every record (spec §3.1) and an ingestion pipeline should not
	have to care.

	Args:
		path (str): The JSON file to read.
	"""
	global CORPUS

	with open(path, encoding="utf-8") as handle_:
		loaded = json.load(handle_)

	CORPUS = [{
		"id"         : record.get("id") or f"doc-{uuid.uuid4().hex[:12]}",
		"content"    : str(record.get("content", "")),
		"created_at" : record.get("created_at") or "2026-01-01T00:00:00.000Z",
		"role"       : record.get("role", "memory"),
		"metadata"   : record.get("metadata") or {},
	} for record in loaded]


def main() -> int:
	"""Serve A2M on stdin/stdout, one JSON value per line (spec §8.2).

	Returns:
		int: Process exit code.
	"""
	argv = sys.argv[1:]

	if "--help" in argv or "-h" in argv:
		print(__doc__)
		return 0

	corpus = next((argument for argument in argv if not argument.startswith("--")), None)
	if corpus:
		load(corpus)

	# Nothing but JSON-RPC may go to stdout; anything else corrupts the stream.
	for line in sys.stdin:
		line = line.strip()
		if not line:
			continue

		try:
			payload = json.loads(line)
		except json.JSONDecodeError as exc:
			response = error_response(None, PARSE_ERROR, "Invalid JSON", str(exc))
		else:
			response = handle(payload)

		if response is not None:
			sys.stdout.write(json.dumps(response) + "\n")
			sys.stdout.flush()

	return 0


if __name__ == "__main__":
	sys.exit(main())
