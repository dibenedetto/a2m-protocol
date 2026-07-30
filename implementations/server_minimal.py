"""A minimal A2M 0.1 server, declaring `core` and nothing else.

	python implementations/server_minimal.py            # stdio

This file deliberately imports **nothing from this project** — not jsonrpc.py,
not a2m/memory.py, not a2m/protocol.py. Only the standard library. It exists to answer a
question the reference implementation cannot: *is the specification enough on
its own?* An implementation that shares code with the reference proves only that
the reference agrees with itself.

It is also the smallest thing that can honestly call itself an A2M server. There
are no tiers, no salience, no consolidation and no scoping; it declares `core`,
so a conformant client will never call for any of that, and a call that arrives
anyway gets CAPABILITY_NOT_SUPPORTED rather than a confusing METHOD_NOT_FOUND.

Ranking is word overlap, which is bad. The specification does not care: §5.1
leaves ranking entirely to the implementation, and that is where implementations
should compete. Being bad at it is conformant.
"""


import datetime
import json
import re
import sys
import uuid


PROTOCOL     = "a2m/0.1"
CAPABILITIES = ["core"]
METHODS      = [
	"memory/describe",
	"memory/remember",
	"memory/recall",
	"memory/timeline",
	"memory/forget",
]

# spec §7
INVALID_REQUEST          = -32600
METHOD_NOT_FOUND         = -32601
INVALID_PARAMS           = -32602
INTERNAL_ERROR           = -32603
PARSE_ERROR              = -32700
UNKNOWN_TIER             = -32002
CAPABILITY_NOT_SUPPORTED = -32003
PROTOCOL_NOT_SUPPORTED   = -32007

WORD_RE = re.compile(r"[^\W_]+")

RECORDS: dict[str, dict] = {}


class A2MError(Exception):
	"""An A2M error, carrying the code the specification assigns it.
	"""
	def __init__(self, code: int, message: str, data=None):
		"""Build an error with a code the specification assigns.

		Args:
			code (int): One of the codes in spec 7.
			message (str): For humans; clients must not parse it.
			data (Any, optional): Structured detail.
		"""
		super().__init__(message)
		self.code    = code
		self.message = message
		self.data    = data


def now_rfc3339() -> str:
	"""The current time in the A2M wire format (spec 3.3).

	Returns:
		str: RFC 3339, UTC, millisecond precision, 'Z' suffix.
	"""
	moment = datetime.datetime.now(datetime.timezone.utc)
	return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def terms(text) -> set:
	"""Split text into indexable terms.

	Args:
		text (Any): Anything with a string form.

	Returns:
		set: Lowercased words of two characters or more. Deliberately naive --
		spec 5.1 leaves ranking to the implementation, so being bad at it is
		still conformant.
	"""
	return {w for w in WORD_RE.findall(str(text).lower()) if len(w) > 1}


def matches(record: dict, where) -> bool:
	"""Apply a 'where' filter to one record (spec 5.2).

	Args:
		record (dict): The stored record.
		where (dict): Equality tests against fields, falling back to metadata. A
			list value means "any of".

	Returns:
		bool: True if the record satisfies every condition.
	"""
	if not where:
		return True

	for key, expected in where.items():
		actual = record.get(key, record.get("metadata", {}).get(key))
		if isinstance(expected, list):
			if actual not in expected:
				return False
		elif actual != expected:
			return False

	return True


def public(record: dict, score=None) -> dict:
	"""Project a stored record down to the fields 'core' defines.

	A record must not advertise a field whose capability this server does not
	declare, which is why tier, salience and owner never appear here.

	Args:
		record (dict): The stored record.
		score (float, optional): Attached only for recall results.

	Returns:
		dict: The record in wire form.
	"""
	out = {
		"id"         : record["id"],
		"content"    : record["content"],
		"created_at" : record["created_at"],
		"role"       : record["role"],
		"metadata"   : record["metadata"],
	}
	if record.get("group"):
		out["group"] = record["group"]
	if score is not None:
		out["score"] = score
	return out


def describe(protocol=None, **ignored) -> dict:
	"""Handle 'memory/describe'.

	Args:
		protocol (str, optional): The version the client speaks.
		**ignored: Unrecognised parameters are ignored, not rejected (spec 2).

	Returns:
		dict: protocol, name, capabilities, methods and limits.

	Raises:
		A2MError: -32007 on an incompatible protocol version.
	"""
	if protocol is not None and protocol != PROTOCOL:
		raise A2MError(PROTOCOL_NOT_SUPPORTED, f"This server speaks {PROTOCOL}", {"supported": [PROTOCOL]})

	return {
		"protocol"     : PROTOCOL,
		"name"         : "a2m-minimal",
		"capabilities" : list(CAPABILITIES),
		"methods"      : list(METHODS),
		"limits"       : {"max_records_per_call": 100},
	}


def remember(records=None, **ignored) -> dict:
	"""Handle 'memory/remember'.

	Args:
		records (list[dict]): Partial records, each needing a string 'content'.
			A supplied 'id' makes the write idempotent (spec 3.2).
		**ignored: Unrecognised parameters are ignored.

	Returns:
		dict: {'ids': [...]} in the caller's order.

	Raises:
		A2MError: -32602 for a malformed record, -32006 past the batch limit,
			-32003 if a tier is named at all.
	"""
	if not isinstance(records, list) or not records:
		raise A2MError(INVALID_PARAMS, "'records' must be a non-empty array")

	if len(records) > 100:
		raise A2MError(-32006, "At most 100 records per call")

	ids = []
	for entry in records:
		if not isinstance(entry, dict) or not isinstance(entry.get("content"), str):
			raise A2MError(INVALID_PARAMS, "Each record needs a string 'content'")

		if entry.get("tier") is not None:
			raise A2MError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'tiers' capability")

		if entry.get("key") is not None:
			raise A2MError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'keys' capability")

		if entry.get("embedding") is not None:
			raise A2MError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'embeddings' capability")

		if entry.get("uri") is not None:
			raise A2MError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'external' capability")

		# spec §3.2 -- a client-supplied id makes the write idempotent.
		id = entry.get("id")
		if id is not None and id in RECORDS:
			ids.append(id)
			continue

		id = id or uuid.uuid4().hex
		RECORDS[id] = {
			"id"         : id,
			"content"    : entry["content"],
			"role"       : entry.get("role", "user"),
			"metadata"   : entry.get("metadata") or {},
			"group"      : entry.get("group"),
			"created_at" : now_rfc3339(),
			"sequence"   : len(RECORDS),
		}
		ids.append(id)

	return {"ids": ids}


def recall(query=None, limit=8, where=None, min_score=0.0, tier=None,
           embedding=None, key_prefix=None, **ignored) -> dict:
	"""Handle 'memory/recall'.

	Args:
		query (str, optional): What to rank against. Absent means rank by recency,
			which spec 4.3 requires rather than an error.
		limit (int, optional): Maximum records.
		where (dict, optional): Metadata filter.
		min_score (float, optional): Drop results below this.
		tier (str, optional): Rejected -- this server has no tiers.
		**ignored: Unrecognised parameters are ignored.

	Returns:
		dict: {'records': [...]} descending by score.

	Raises:
		A2MError: -32003 if a tier is named.
	"""
	if tier is not None:
		raise A2MError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'tiers' capability")
	if embedding is not None:
		raise A2MError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'embeddings' capability")
	if key_prefix is not None:
		raise A2MError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'keys' capability")

	candidates = [r for r in RECORDS.values() if matches(r, where)]
	wanted     = terms(query) if query else set()

	scored = []
	if not wanted:
		# spec §4.3 -- an absent query is not an error; rank by recency instead.
		for record in sorted(candidates, key=lambda r: -r["sequence"]):
			scored.append((record, 0.0))
	else:
		for record in candidates:
			held    = terms(record["content"])
			overlap = wanted & held
			if overlap:
				scored.append((record, len(overlap) / len(wanted | held)))
		scored.sort(key=lambda pair: -pair[1])

	scored = [(r, s) for r, s in scored if s >= (min_score or 0.0)]
	if limit and limit > 0:
		scored = scored[:limit]

	return {"records": [public(r, s) for r, s in scored]}


def timeline(limit=0, tier=None, **ignored) -> dict:
	"""Handle 'memory/timeline'.

	Args:
		limit (int, optional): Keep the most recent N, still ascending.
		tier (str, optional): Rejected -- this server has no tiers.
		**ignored: Unrecognised parameters are ignored.

	Returns:
		dict: {'records': [...]} ascending by creation.

	Raises:
		A2MError: -32003 if a tier is named.
	"""
	if tier is not None:
		raise A2MError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'tiers' capability")

	# spec §4.4 -- ascending by creation; `limit` takes the newest, still ascending.
	ordered = sorted(RECORDS.values(), key=lambda r: r["sequence"])
	if limit and limit > 0:
		ordered = ordered[-limit:]

	return {"records": [public(r) for r in ordered]}


def forget(ids=None, query=None, where=None, tier=None, **ignored) -> dict:
	"""Handle 'memory/forget'.

	Args:
		ids (list[str], optional): Delete these exactly.
		query (str, optional): Delete whatever this recalls.
		where (dict, optional): Metadata filter.
		tier (str, optional): Rejected -- this server has no tiers.
		**ignored: Unrecognised parameters are ignored.

	Returns:
		dict: {'forgotten': count}.

	Raises:
		A2MError: -32602 when no selector is given, -32003 if a tier is named.
	"""
	if tier is not None:
		raise A2MError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'tiers' capability")

	# spec §4.5 -- emptying the store must take more than an empty request.
	if not ids and not query and not where:
		raise A2MError(INVALID_PARAMS, "Pass ids, query or where; refusing to forget everything")

	doomed = set()
	if ids:
		doomed.update(id for id in ids if id in RECORDS)
	if query or where:
		for record in recall(query=query, where=where, limit=0)["records"]:
			doomed.add(record["id"])

	for id in doomed:
		RECORDS.pop(id, None)

	return {"forgotten": len(doomed)}


def unsupported(capability: str):
	"""Build a handler that refuses an undeclared capability.

	Every A2M method is registered even when unimplemented, because an
	unregistered one would answer -32601 METHOD_NOT_FOUND and a client cannot tell
	that from a typo. Spec 2 requires -32003.

	Args:
		capability (str): The capability this server does not implement.

	Returns:
		Callable: A handler that always raises -32003.
	"""
	def handler(**ignored):
		"""Always refuse, naming the capability that is missing.
		"""
		raise A2MError(CAPABILITY_NOT_SUPPORTED, f"This server does not implement the '{capability}' capability")
	return handler


HANDLERS = {
	"memory/describe"    : describe,
	"memory/remember"    : remember,
	"memory/recall"      : recall,
	"memory/timeline"    : timeline,
	"memory/forget"      : forget,
	# spec §2 -- an undeclared capability answers -32003, never -32601.
	"memory/promote"     : unsupported("tiers"),
	"memory/consolidate" : unsupported("tiers"),
	"memory/reinforce"   : unsupported("salience"),
	"memory/session/list": unsupported("sessions"),
	"memory/session/close": unsupported("sessions"),
	"memory/fetch"       : unsupported("keys"),
	"memory/events"      : unsupported("events"),
	"memory/events/subscribe"  : unsupported("events"),
	"memory/events/unsubscribe": unsupported("events"),
}


def handle(payload):
	"""Turn one parsed request into one response.

	Args:
		payload (Any): A parsed JSON-RPC request.

	Returns:
		dict | None: The response, or None for a notification.
	"""
	# A JSON array here is a JSON-RPC batch, which A2M does not use (spec 8).
	if not isinstance(payload, dict):
		return error(None, INVALID_REQUEST, "Request must be a JSON object")

	is_notification = "id" not in payload
	id              = payload.get("id")

	def fail(code, message, data=None):
		"""Build an error response, or None when the request was a notification.
		"""
		return None if is_notification else error(id, code, message, data)

	if payload.get("jsonrpc") != "2.0":
		return fail(INVALID_REQUEST, "Expected jsonrpc '2.0'")

	handler = HANDLERS.get(payload.get("method"))
	if handler is None:
		return fail(METHOD_NOT_FOUND, f"Unknown method '{payload.get('method')}'")

	params = payload.get("params") or {}
	if not isinstance(params, dict):
		return fail(INVALID_PARAMS, "'params' must be an object")

	try:
		result = handler(**params)
	except A2MError as exc:
		return fail(exc.code, exc.message, exc.data)
	except TypeError as exc:
		return fail(INVALID_PARAMS, str(exc))
	except Exception as exc:
		return fail(INTERNAL_ERROR, "Handler failed", repr(exc))

	return None if is_notification else {"jsonrpc": "2.0", "id": id, "result": result}


def error(id, code, message, data=None):
	"""Build a JSON-RPC error response.

	Args:
		id (Any): The id being answered.
		code (int): Error code.
		message (str): For humans.
		data (Any, optional): Structured detail.

	Returns:
		dict: The error response.
	"""
	body = {"code": code, "message": message}
	if data is not None:
		body["data"] = data
	return {"jsonrpc": "2.0", "id": id, "error": body}


def main() -> int:
	# spec §8.2 -- one JSON value per line; stdout carries nothing else.
	"""Serve A2M on stdin/stdout, one JSON value per line (spec 8.2).

	Nothing but JSON-RPC may go to stdout; anything else corrupts the stream.

	Returns:
		int: Process exit code.
	"""
	for line in sys.stdin:
		line = line.strip()
		if not line:
			continue

		try:
			payload = json.loads(line)
		except json.JSONDecodeError as exc:
			response = error(None, PARSE_ERROR, "Invalid JSON", str(exc))
		else:
			response = handle(payload)

		if response is not None:
			sys.stdout.write(json.dumps(response) + "\n")
			sys.stdout.flush()

	return 0


if __name__ == "__main__":
	sys.exit(main())
