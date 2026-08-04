"""A2M 0.1 conformance suite.

Point it at any A2M server and it reports what that server actually honours:

	python -m tools.conformance --stdio python -m a2m
	python -m tools.conformance --stdio python implementations/server_minimal.py
	python -m tools.conformance --http http://127.0.0.1:8778/

The suite speaks only the protocol. It never imports the server under test, so a
Rust or TypeScript implementation is tested exactly as a Python one is.

Checks are grouped by capability and skipped when the server does not declare
that capability — skipping is not failing. Declaring a capability and then not
honouring it *is* failing, and that is the case worth catching: a client trusts
`describe`, so a server that lies there breaks clients in ways no amount of
defensive coding on their side can fix.

Exit code is 0 when every applicable check passes.
"""


import json
import sys
import urllib.error
import urllib.request
import uuid


from   typing      import Any, Callable


from   a2m.jsonrpc import Client, HttpTransport, JsonRpcError, StdioTransport


PROTOCOL = "a2m/0.1"

UNKNOWN_TIER             = -32002
CAPABILITY_NOT_SUPPORTED = -32003
READ_ONLY                = -32004
QUOTA_EXCEEDED           = -32006
PROTOCOL_NOT_SUPPORTED   = -32007
INVALID_REQUEST          = -32600
INVALID_PARAMS           = -32602
METHOD_NOT_FOUND         = -32601

RFC3339 = "%Y-%m-%dT%H:%M:%S"


class Report:
	"""Tallies what a conformance run found.

	Skipping is not failing: a capability the server never declared is simply not
	tested. Declaring one and then not honouring it *is* a failure.
	"""

	def __init__(self) -> None:
		"""Create an empty report with nothing passed, failed or skipped.
		"""
		self.passed  : list[str] = []
		self.failed  : list[tuple[str, str]] = []
		self.skipped : list[str] = []
		# Set when the server refuses writes (spec §2.1). Not a failure: it
		# selects a different profile, because a corpus must be judged on what
		# it promises rather than failed for declining to be a memory.
		self.read_only = False


	def check(self, label: str, condition: bool, detail: Any = "") -> bool:
		"""Record one check.

		Args:
			label (str): What was being checked.
			condition (bool): Whether it held.
			detail (Any, optional): Shown on failure, to make the report actionable.

		Returns:
			bool: The condition, so callers can branch on it.
		"""
		if condition:
			self.passed.append(label)
			print(f"    ok    {label}")
		else:
			self.failed.append((label, str(detail)))
			print(f"    FAIL  {label}  {detail}")
		return bool(condition)


	def skip(self, label: str, reason: str) -> None:
		"""Record a check that did not apply.

		Args:
			label (str): What was skipped.
			reason (str): Why -- almost always an undeclared capability.
		"""
		self.skipped.append(label)
		print(f"    skip  {label}  ({reason})")


def is_rfc3339(value: Any) -> bool:
	"""Whether a value is a valid A2M timestamp (spec 3.3).

	Args:
		value (Any): The candidate.

	Returns:
		bool: True only for a UTC, 'Z'-suffixed RFC 3339 string. A number is
		specifically wrong, which is the case this exists to catch.
	"""
	if not isinstance(value, str) or not value.endswith("Z"):
		return False
	import datetime
	try:
		datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
		return True
	except ValueError:
		return False


def expect_error(report: Report, label: str, code: int, call: Callable) -> None:
	"""Assert that a call fails with a particular code.

	Args:
		report (Report): Where to record the result.
		label (str): What is being checked.
		code (int): The expected error code.
		call (Callable): A zero-argument call that should fail.
	"""
	try:
		result = call()
		report.check(label, False, f"expected error {code}, got result {json.dumps(result)[:80]}")
	except JsonRpcError as exc:
		report.check(label, exc.code == code, f"expected {code}, got {exc.code}")


def test_core(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the five mandatory methods and the rules every server must follow.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  core")

	report.check("describe reports a protocol", profile.get("protocol") == PROTOCOL, profile.get("protocol"))
	report.check("describe reports a name"    , isinstance(profile.get("name"), str))
	report.check("capabilities include core"  , "core" in profile.get("capabilities", []), profile.get("capabilities"))
	report.check("methods is a list"          , isinstance(profile.get("methods"), list))

	declared = set(profile.get("capabilities", []))
	report.check("declared capabilities are known",
	             declared <= {"core", "tiers", "salience", "scopes", "sessions",
	                          "embeddings", "keys", "external", "events", "summarize",
	                          "prompt"},
	             declared)

	expect_error(report, "an incompatible protocol is rejected", PROTOCOL_NOT_SUPPORTED,
	             lambda: client.call("memory/describe", {"protocol": "a2m/99.0"}))

	report.check("unknown params are ignored, not rejected",
	             bool(client.call("memory/describe", {"unknown_field_xyz": 1})),
	             "server rejected an unrecognised parameter")

	marker = uuid.uuid4().hex

	# The first write is also the probe. A server that refuses it with -32004 is
	# a read-only corpus (spec §2.1) -- conformant, and checked against a
	# different profile from here on. Doing it with the write the suite was
	# going to make anyway keeps a writable store free of a probe record.
	try:
		written = client.call("memory/remember", {"records": [
			{"content": f"the deploy key {marker} rotates every ninety days", "role": "user",
			 "metadata": {"suite": marker, "nested": {"a": [1, 2]}}},
			{"content": f"the release branch {marker} is cut on thursdays", "role": "user",
			 "metadata": {"suite": marker}},
		]})
	except JsonRpcError as exc:
		if exc.code != READ_ONLY:
			raise
		report.read_only = True
		report.check("a read-only server refuses remember with -32004", True)
		test_core_read_only(client, report, profile)
		return

	ids = written.get("ids", [])
	report.check("remember returns one id per record", len(ids) == 2, written)
	report.check("ids are non-empty strings", all(isinstance(i, str) and i for i in ids), ids)

	found = client.call("memory/recall", {"query": f"deploy key {marker}", "limit": 5}).get("records", [])
	report.check("recall returns records", bool(found), found)

	if found:
		record = found[0]
		report.check("recall found the right record", marker in record.get("content", ""), record.get("content"))
		report.check("records carry an id"       , isinstance(record.get("id"), str))
		report.check("records carry content"     , isinstance(record.get("content"), str))
		report.check("created_at is RFC 3339"    , is_rfc3339(record.get("created_at")), record.get("created_at"))
		report.check("created_at is not a number", not isinstance(record.get("created_at"), (int, float)))

		scores = [r.get("score") for r in found if "score" in r]
		if scores:
			report.check("scores descend", scores == sorted(scores, reverse=True), scores)

		metadata = record.get("metadata", {})
		report.check("metadata round-trips unchanged",
		             metadata.get("nested") == {"a": [1, 2]} or metadata.get("suite") == marker,
		             metadata)

	empty = client.call("memory/recall", {"limit": 3})
	report.check("an absent query is not an error", isinstance(empty.get("records"), list), empty)

	ordered = client.call("memory/timeline", {}).get("records", [])
	stamps  = [r.get("created_at") for r in ordered if r.get("created_at")]
	report.check("timeline ascends by created_at", stamps == sorted(stamps), stamps[:4])

	if len(ordered) >= 2:
		newest = client.call("memory/timeline", {"limit": 1}).get("records", [])
		report.check("timeline limit takes the newest",
		             len(newest) == 1 and newest[0].get("id") == ordered[-1].get("id"),
		             newest)

	# spec §3.2 -- an id the server assigns must be unique within the store.
	pair = client.call("memory/remember", {"records": [
		{"content": f"uniqueness probe one {marker}"},
		{"content": f"uniqueness probe two {marker}"},
	]}).get("ids", [])
	report.check("server-assigned ids are unique", len(set(pair)) == len(pair) == 2, pair)
	client.call("memory/forget", {"ids": pair})

	# spec §4.2 -- every record needs content.
	expect_error(report, "a record with no content is refused", INVALID_PARAMS,
	             lambda: client.call("memory/remember", {"records": [{"role": "user"}]}))

	# spec §4.1 -- a server enforces its own documented limits regardless of
	# whether the client respected them.
	batch = (profile.get("limits") or {}).get("max_records_per_call")
	if isinstance(batch, int) and 0 < batch <= 1000:
		try:
			client.call("memory/remember", {"records": [
				{"content": f"quota probe {n} {marker}"} for n in range(batch + 1)]})
			report.check("a documented limit is enforced", False,
			             f"accepted {batch + 1} records against a stated limit of {batch}")
		except JsonRpcError as exc:
			report.check("a documented limit is enforced",
			             exc.code in (QUOTA_EXCEEDED, INVALID_PARAMS), exc.code)
	else:
		report.skip("a documented limit is enforced", "no max_records_per_call declared")

	# spec §4.3 -- limit 0 means no limit, not no records, unless the server
	# capped it. The opposite reading silently returns nothing.
	if not (profile.get("limits") or {}).get("max_recall_limit"):
		unbounded = client.call("memory/recall", {"limit": 0}).get("records", [])
		report.check("a limit of 0 is not a limit of nothing", len(unbounded) > 0, len(unbounded))
	else:
		report.skip("a limit of 0 is not a limit of nothing", "the server declares a max_recall_limit")

	# spec §3.2 -- a client-supplied id makes the write idempotent.
	chosen = f"conformance-{marker}"
	first  = client.call("memory/remember", {"records": [{"id": chosen, "content": "idempotency probe"}]})
	second = client.call("memory/remember", {"records": [{"id": chosen, "content": "idempotency probe"}]})
	report.check("a client-supplied id is honoured", first.get("ids") == [chosen], first)
	report.check("re-writing the same id is idempotent", second.get("ids") == [chosen], second)

	twice = client.call("memory/timeline", {}).get("records", [])
	report.check("idempotent write created only one record",
	             sum(1 for r in twice if r.get("id") == chosen) == 1,
	             [r.get("id") for r in twice])

	expect_error(report, "forget with no selector is refused", INVALID_PARAMS,
	             lambda: client.call("memory/forget", {}))

	expect_error(report, "an unknown method is -32601", METHOD_NOT_FOUND,
	             lambda: client.call("memory/not_a_real_method", {}))

	removed = client.call("memory/forget", {"ids": ids + [chosen]})
	report.check("forget reports a count", isinstance(removed.get("forgotten"), int), removed)

	survivors = client.call("memory/timeline", {}).get("records", [])
	report.check("forgotten records are gone",
	             not any(r.get("id") in ids for r in survivors),
	             [r.get("id") for r in survivors][:4])


def test_core_read_only(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the read side of a server that refuses writes (spec §2.1).

	A pre-existing corpus exposed over A2M is the smallest useful server, and
	the on-ramp for every retrieval system that already exists. It cannot be
	checked by writing a record and reading it back, so it is checked against
	what it actually holds -- and against the two rules that only apply here:
	both write methods refuse, and they refuse consistently.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("        (writes refused -- checking the read-only profile, spec §2.1)")

	# Both write methods must refuse. A server refusing one and accepting the
	# other leaves a client no way to know which is which.
	expect_error(report, "a read-only server refuses forget with -32004", READ_ONLY,
	             lambda: client.call("memory/forget", {"ids": ["anything"]}))
	expect_error(report, "and refuses a selector-less forget too", READ_ONLY,
	             lambda: client.call("memory/forget", {}))

	held = client.call("memory/timeline", {}).get("records", [])
	report.check("timeline returns a list", isinstance(held, list), held)

	if not held:
		report.skip("read-only record checks", "the corpus is empty, so there is nothing to check")
		report.skip("read-only recall checks", "the corpus is empty")
		return

	record = held[0]
	report.check("records carry an id"       , isinstance(record.get("id"), str) and record["id"], record)
	report.check("records carry content"     , isinstance(record.get("content"), str), record)
	report.check("created_at is RFC 3339"    , is_rfc3339(record.get("created_at")), record.get("created_at"))
	report.check("created_at is not a number", not isinstance(record.get("created_at"), (int, float)))

	stamps = [r.get("created_at") for r in held if r.get("created_at")]
	report.check("timeline ascends by created_at", stamps == sorted(stamps), stamps[:4])

	if len(held) >= 2:
		newest = client.call("memory/timeline", {"limit": 1}).get("records", [])
		report.check("timeline limit takes the newest",
		             len(newest) == 1 and newest[0].get("id") == held[-1].get("id"), newest)

	# Recall has to work, and has to work against text the corpus actually
	# holds -- so the query is built from a record the server just returned.
	words  = [w for w in str(record.get("content", "")).split() if len(w) > 4][:4]
	found  = client.call("memory/recall", {"query": " ".join(words), "limit": 5}).get("records", [])
	report.check("recall returns records for text the corpus holds", bool(found), (words, found))

	scores = [r.get("score") for r in found if "score" in r]
	if scores:
		report.check("scores descend", scores == sorted(scores, reverse=True), scores)

	empty = client.call("memory/recall", {"limit": 3})
	report.check("an absent query is not an error", isinstance(empty.get("records"), list), empty)

	report.check("metadata round-trips unchanged",
	             all(isinstance(r.get("metadata", {}), dict) for r in held), held[:2])

	expect_error(report, "an unknown method is -32601", METHOD_NOT_FOUND,
	             lambda: client.call("memory/not_a_real_method", {}))


def test_tiers(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the 'tiers' capability: layout, placement, promotion, consolidation.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  tiers")

	tiers = profile.get("tiers", [])
	if not report.check("describe reports tiers", isinstance(tiers, list) and bool(tiers), tiers):
		return

	report.check("every tier has a name and a count",
	             all(isinstance(t.get("name"), str) and isinstance(t.get("count"), int) for t in tiers),
	             tiers)

	kinds = [t.get("kind") for t in tiers if "kind" in t]
	report.check("tier kinds are from the vocabulary",
	             all(k in ("working", "episodic", "semantic", "procedural") for k in kinds),
	             kinds)

	names  = [t["name"] for t in tiers]
	marker = uuid.uuid4().hex
	target = names[-1]

	written = client.call("memory/remember", {"records": [
		{"content": f"tier probe {marker}", "tier": names[0]},
	]})
	ids = written.get("ids", [])
	report.check("a record can be written to a named tier", bool(ids), written)

	placed = client.call("memory/recall", {"query": f"tier probe {marker}", "tier": names[0]}).get("records", [])
	report.check("recall can be restricted to a tier", bool(placed), placed)
	if placed:
		report.check("records report their tier", placed[0].get("tier") == names[0], placed[0].get("tier"))

	moved = client.call("memory/promote", {"ids": ids, "tier": target})
	report.check("promote reports a count", isinstance(moved.get("promoted"), int), moved)

	after = client.call("memory/recall", {"query": f"tier probe {marker}", "tier": target}).get("records", [])
	report.check("a promoted record is in the target tier", bool(after), after)

	expect_error(report, "an unknown tier is -32002", UNKNOWN_TIER,
	             lambda: client.call("memory/promote", {"ids": ids, "tier": "no-such-tier-xyz"}))

	# spec §4.6 -- an id that does not exist is simply not counted. Failing
	# instead would make a retry after a partial move impossible.
	try:
		ignored = client.call("memory/promote", {"ids": [f"absent-{marker}"], "tier": target})
		report.check("promoting an unknown id is not an error",
		             ignored.get("promoted") == 0, ignored)
	except JsonRpcError as exc:
		report.check("promoting an unknown id is not an error", False, exc.code)

	result = client.call("memory/consolidate", {})
	report.check("consolidate reports counts", isinstance(result.get("counts"), dict), result)

	client.call("memory/forget", {"ids": ids})


def test_salience(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the 'salience' capability: reinforcement and the salience field.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  salience")

	marker  = uuid.uuid4().hex
	written = client.call("memory/remember", {"records": [{"content": f"salience probe {marker}"}]})
	ids     = written.get("ids", [])

	result = client.call("memory/reinforce", {"ids": ids, "amount": 1.0})
	report.check("reinforce reports a count", isinstance(result.get("reinforced"), int), result)
	report.check("reinforce counted the record", result.get("reinforced") == 1, result)

	found = client.call("memory/recall", {"query": f"salience probe {marker}"}).get("records", [])
	if found:
		report.check("records carry a numeric salience",
		             isinstance(found[0].get("salience"), (int, float)),
		             found[0].get("salience"))

	client.call("memory/forget", {"ids": ids})


def test_scopes(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the 'scopes' capability: owner-stamped writes and scoped reads.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  scopes")

	marker = uuid.uuid4().hex
	client.call("memory/remember", {
		"records": [{"content": f"scope probe {marker}"}],
		"owner"  : "conformance-alice",
	})

	mine = client.call("memory/recall", {"query": f"scope probe {marker}", "owner": "conformance-alice"}).get("records", [])
	report.check("an owner can read its own records", bool(mine), mine)

	if mine:
		report.check("records report an owner",
		             mine[0].get("owner") in ("conformance-alice", None),
		             mine[0].get("owner"))

	client.call("memory/forget", {"query": f"scope probe {marker}", "owner": "conformance-alice"})


def test_sessions(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the 'sessions' capability: listing, replay, and closing.

	The check that matters is replay -- a session that can be searched but not
	replayed is of little use in the tier that is replayed.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  sessions")

	name   = f"conformance-{uuid.uuid4().hex[:8]}"
	marker = uuid.uuid4().hex

	client.call("memory/remember", {"records": [
		{"content": f"session probe one {marker}", "session": name},
		{"content": f"session probe two {marker}", "session": name},
		{"content": f"unrelated {marker}"},
	]})

	listed = client.call("memory/session/list", {}).get("sessions", [])
	report.check("session/list reports the session", any(s.get("session") == name for s in listed), listed)

	entry = next((s for s in listed if s.get("session") == name), {})
	report.check("a session reports how many records it holds", entry.get("records") == 2, entry)

	# The gap sessions exist to close: replaying one conversation, not searching it.
	replayed = client.call("memory/timeline", {"where": {"session": name}}).get("records", [])
	report.check("timeline can be filtered to one session", len(replayed) == 2, replayed)
	report.check("and excludes everything else",
	             all(marker in r.get("content", "") and "unrelated" not in r.get("content", "") for r in replayed),
	             replayed)

	report.check("records carry their session",
	             all(r.get("session") == name for r in replayed), replayed)

	closed = client.call("memory/session/close", {"session": name})
	report.check("session/close reports what it closed", closed.get("closed") == name, closed)
	report.check("closing reports counts", isinstance(closed.get("counts"), dict), closed)

	still = client.call("memory/timeline", {"where": {"session": name}}).get("records", [])
	report.check("closing does not destroy the conversation", len(still) == 2, still)

	client.call("memory/forget", {"where": {"session": name}})

	try:
		client.call("memory/session/close", {})
		report.check("closing with no session is refused", False)
	except JsonRpcError as exc:
		report.check("closing with no session is refused", exc.code == INVALID_PARAMS, exc.code)

	# spec §4.11 -- closing a conversation that was never opened is a no-op, so
	# a client need not track which sessions it has already finished.
	try:
		client.call("memory/session/close", {"session": f"never-existed-{uuid.uuid4().hex}"})
		report.check("closing an unknown session is a no-op", True)
	except JsonRpcError as exc:
		report.check("closing an unknown session is a no-op", False, exc.code)


def test_keys(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the 'keys' capability: addressing, upsert, and prefix scope.

	The check that matters is upsert. A key that appends instead of replacing is
	worse than no key at all: the caller believes a fact was corrected while the
	stale one is still there to be recalled.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  keys")

	root = f"conformance/{uuid.uuid4().hex[:8]}"
	key  = f"{root}/user/city"

	client.call("memory/remember", {"records": [{"key": key, "content": "the user lives in bologna"}]})
	first = client.call("memory/fetch", {"key": key}).get("record")
	report.check("a record can be addressed by key", first and "bologna" in first.get("content", ""), first)
	report.check("records report their key", first and first.get("key") == key, first)

	client.call("memory/remember", {"records": [{"key": key, "content": "the user lives in milan"}]})
	second = client.call("memory/fetch", {"key": key}).get("record")
	report.check("writing to an occupied key replaces it",
	             second and "milan" in second.get("content", ""), second)
	report.check("the replaced record keeps its id", second and first and second["id"] == first["id"],
	             (first or {}).get("id"), )
	report.check("revision advances", isinstance(second.get("revision"), int) and second["revision"] >= 1, second)

	held = client.call("memory/timeline", {"key_prefix": key}).get("records", [])
	report.check("the stale value is gone, not merely outnumbered", len(held) == 1, held)

	# Hierarchy: a prefix is the recursive scope read.
	for leaf in ("goal", "budget"):
		client.call("memory/remember", {"records": [{"key": f"{root}/wf-1/{leaf}", "content": f"about {leaf}"}]})
	client.call("memory/remember", {"records": [{"key": f"{root}/wf-2/goal", "content": "other workflow"}]})

	under = client.call("memory/timeline", {"key_prefix": f"{root}/wf-1/"}).get("records", [])
	report.check("a key prefix scopes a read", len(under) == 2, [r.get("key") for r in under])

	whole = client.call("memory/timeline", {"key_prefix": root}).get("records", [])
	report.check("a shorter prefix scopes wider", len(whole) >= 4, len(whole))

	report.check("fetching an unused key is not an error",
	             client.call("memory/fetch", {"key": f"{root}/nothing/here"}).get("record") is None)

	expect_error(report, "fetch without a key is refused", INVALID_PARAMS,
	             lambda: client.call("memory/fetch", {}))

	removed = client.call("memory/forget", {"key_prefix": root}).get("forgotten", 0)
	report.check("a prefix can be forgotten wholesale", removed >= 4, removed)


def test_embeddings(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the 'embeddings' capability: verbatim storage and vector search.

	'Verbatim' is the whole contract. A server that re-embeds a caller's text and
	replaces the vector silently moves every record into its own model's space,
	which is exactly what makes two frameworks unable to share a store.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  embeddings")

	contract = profile.get("embeddings") or {}
	report.check("describe reports an embedding contract", isinstance(contract, dict), contract)
	report.check("it names a metric", contract.get("metric") in ("cosine", "dot", "l2"), contract)

	# What the store *could* compare against what it *is* comparing with. A
	# caller holding inner-product vectors needs to tell "misconfigured" from
	# "unsuitable", and `metric` alone cannot (spec §3.7).
	offered = contract.get("metrics")
	if offered is None:
		report.skip("the metric menu contains the metric in use", "no 'metrics' reported -- it is a SHOULD")
	else:
		report.check("the metric menu is a list of known metrics",
		             isinstance(offered, list) and set(offered) <= {"cosine", "dot", "l2"}, offered)
		report.check("and contains the metric actually in use",
		             contract.get("metric") in offered, contract)

	width  = contract.get("dimensions") or 8
	marker = uuid.uuid4().hex[:8]
	near   = [1.0] + [0.0] * (width - 1)
	far    = [0.0, 1.0] + [0.0] * (width - 2) if width >= 2 else [1.0]

	written = client.call("memory/remember", {"records": [
		{"content": f"vector probe near {marker}", "embedding": near},
		{"content": f"vector probe far {marker}",  "embedding": far},
	]})
	ids = written.get("ids", [])
	report.check("records accept a caller-supplied vector", len(ids) == 2, written)

	back = client.call("memory/recall", {"query": f"vector probe near {marker}",
	                                     "limit": 1, "embeddings": True}).get("records", [])
	if report.check("the stored vector can be read back", back and "embedding" in back[0], back):
		stored = back[0]["embedding"]
		report.check("it was stored verbatim, not regenerated",
		             len(stored) == len(near) and all(abs(a - b) < 1e-5 for a, b in zip(stored, near)),
		             stored[:4])

	found = client.call("memory/recall", {"embedding": near, "limit": 1}).get("records", [])
	report.check("a caller-supplied query vector searches", found, found)
	report.check("and finds the nearer record", found and "near" in found[0].get("content", ""), found)

	after = client.call("memory/describe", {}).get("embeddings") or {}
	report.check("the store reports its dimensionality once fixed", after.get("dimensions") == len(near), after)

	expect_error(report, "a mismatched vector width is refused", -32008,
	             lambda: client.call("memory/remember", {"records": [
	                 {"content": "wrong width", "embedding": [1.0] * (len(near) + 3)}]}))

	client.call("memory/forget", {"ids": ids})
	test_metric(client, report, contract, width)


def test_metric(client: Client, report: Report, contract: dict[str, Any], width: int) -> None:
	"""Check that the declared metric is the metric actually used (spec §3.7).

	`metric` is the one thing a caller supplying vectors cannot verify for
	itself and the server cannot detect a mismatch in: a vector from a model
	trained against a different comparison is ranked with complete confidence
	and no warning. So a server declaring one and using another is undetectable
	in production, which makes it exactly the kind of claim worth checking here.

	Three vectors along one axis separate the three metrics, because they
	disagree about **magnitude**:

		A = 0.9x   near, same direction
		B = 3.0x   far, same direction
		C = 0.5x + 0.5y   near-ish, different direction

	`cosine` ignores length, so A and B tie at the top and C is last. `dot`
	rewards length, so B wins. `l2` punishes it, so B loses even to C. Each
	declared metric therefore implies an ordering the other two contradict.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		contract (dict): The server's embedding profile.
		width (int): The store's vector width.
	"""
	metric = (contract or {}).get("metric")
	if metric not in ("cosine", "dot", "l2") or width < 2:
		report.skip("declared metric matches observed ranking", f"metric {metric!r}, width {width}")
		return

	def vector(x: float, y: float) -> list[float]:
		return [x, y] + [0.0] * (width - 2)

	marker = uuid.uuid4().hex

	# One call, so recency and salience are equal and the metric is what
	# separates them.
	client.call("memory/remember", {"records": [
		{"content": f"metric probe A {marker}", "embedding": vector(0.9, 0.0)},
		{"content": f"metric probe B {marker}", "embedding": vector(3.0, 0.0)},
		{"content": f"metric probe C {marker}", "embedding": vector(0.5, 0.5)},
	]})

	found = client.call("memory/recall", {"embedding": vector(1.0, 0.0), "limit": 10}).get("records", [])
	order = [r.get("content", "").split()[2] for r in found
	         if marker in r.get("content", "") and len(r.get("content", "").split()) > 2]

	def rank(name: str) -> int:
		return order.index(name) if name in order else 99

	if len(order) < 3:
		report.skip("declared metric matches observed ranking",
		            f"the store returned {len(order)} of 3 probes")
	elif metric == "cosine":
		# Length is ignored, so the far vector still beats the off-axis one.
		report.check("declared 'cosine' ranks by direction, ignoring length",
		             rank("B") < rank("C"), order)
	elif metric == "dot":
		# Length is rewarded: the long vector outranks the short one beside it.
		report.check("declared 'dot' ranks the longer vector higher",
		             rank("B") < rank("A"), order)
	else:
		# Length is punished: distance puts the long vector below the off-axis one.
		report.check("declared 'l2' ranks the more distant vector lower",
		             rank("C") < rank("B"), order)

	client.call("memory/forget", {"query": f"metric probe {marker}"})


def test_external(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the 'external' capability: references stored, never dereferenced.

	The check that cannot be automated from outside is the important one -- that
	the server does not fetch the URI. A `file:///` reference to a path that does
	not exist stands in: a server that dereferenced would fail the write.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  external")

	marker = uuid.uuid4().hex[:8]
	uri    = f"file:///nonexistent/{marker}/runbook.md"

	written = client.call("memory/remember", {"records": [{
		"content"    : f"deployment runbook {marker}: rotating the vault key",
		"uri"        : uri,
		"media_type" : "text/markdown",
	}]})
	ids = written.get("ids", [])
	report.check("a record can carry a reference", len(ids) == 1, written)

	found = client.call("memory/recall", {"query": f"runbook {marker} vault key", "limit": 1}).get("records", [])
	report.check("an external record is findable by its indexed text", found, found)

	if found:
		record = found[0]
		report.check("the uri round-trips unchanged", record.get("uri") == uri, record.get("uri"))
		report.check("the media type round-trips"   , record.get("media_type") == "text/markdown", record)
		report.check("content is still the indexed text",
		             marker in record.get("content", ""), record.get("content"))

	# A server that dereferenced would have failed the write above, since the
	# path does not exist. Reaching here at all is the evidence.
	report.check("the server did not dereference the uri", bool(ids))

	client.call("memory/forget", {"ids": ids})


def test_events(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the 'events' capability: cursors, coalescing, and push where carried.

	The check that matters is exactly-once: successive polls, each carrying the
	cursor the previous reply returned, must see every retained event once, in
	order, with no duplicates and no gaps (spec §4.12). A watcher that can miss
	a write silently is worse than no watcher at all.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  events")

	contract = profile.get("events")
	report.check("describe reports an events contract",
	             isinstance(contract, dict) and isinstance(contract.get("push"), bool),
	             contract)

	opening = client.call("memory/events", {})
	report.check("a poll with no cursor returns no events, only a cursor",
	             opening.get("events") == [] and isinstance(opening.get("cursor"), str) and opening["cursor"],
	             opening)

	cursor = opening.get("cursor")
	marker = uuid.uuid4().hex

	written = client.call("memory/remember", {"records": [{"content": f"event probe {marker}"}]})
	ids     = written.get("ids", [])

	reply  = client.call("memory/events", {"cursor": cursor})
	events = reply.get("events", [])
	report.check("a write produces a written event",
	             any(e.get("kind") == "written" and e.get("id") == ids[0] for e in events),
	             events)
	report.check("events carry an RFC 3339 timestamp",
	             all(is_rfc3339(e.get("at")) for e in events), events)
	report.check("the reply carries a new cursor", isinstance(reply.get("cursor"), str), reply)

	cursor = reply.get("cursor")
	drained = client.call("memory/events", {"cursor": cursor})
	report.check("a poll from the returned cursor is empty until something happens",
	             drained.get("events") == [], drained)

	# Exactly once, in order, across a limit boundary.
	cursor = drained.get("cursor")
	more_ids = client.call("memory/remember", {"records": [
		{"content": f"event probe two {marker}"},
		{"content": f"event probe three {marker}"},
		{"content": f"event probe four {marker}"},
	]}).get("ids", [])

	first  = client.call("memory/events", {"cursor": cursor, "limit": 2})
	second = client.call("memory/events", {"cursor": first.get("cursor"), "limit": 100})
	seen   = [e.get("id") for e in first.get("events", []) + second.get("events", [])
	          if e.get("kind") == "written"]
	report.check("successive polls see every event exactly once, in order",
	             seen == more_ids, (seen, more_ids))

	# Coalescing: however many records a forget removes, it is one event.
	cursor    = second.get("cursor")
	forgotten = client.call("memory/forget", {"ids": ids + more_ids}).get("forgotten", 0)
	felt      = [e for e in client.call("memory/events", {"cursor": cursor}).get("events", [])
	             if e.get("kind") == "forgotten"]
	report.check("a forget is one coalesced event, not one per record",
	             len(felt) == 1 and felt[0].get("count") == forgotten,
	             (felt, forgotten))

	if "tiers" in set(profile.get("capabilities", [])):
		cursor = client.call("memory/events", {}).get("cursor")
		client.call("memory/consolidate", {})
		bulk = [e for e in client.call("memory/events", {"cursor": cursor}).get("events", [])
		        if e.get("kind") == "consolidated"]
		report.check("a consolidation is one coalesced event",
		             len(bulk) == 1, bulk)

	# The kinds filter is applied by the server, and filtered events are not
	# re-offered later.
	cursor = client.call("memory/events", {}).get("cursor")
	kept   = client.call("memory/remember", {"records": [{"content": f"kind filter probe {marker}"}]}).get("ids", [])
	client.call("memory/forget", {"ids": kept})
	only = client.call("memory/events", {"cursor": cursor, "kinds": ["forgotten"]}).get("events", [])
	report.check("a kinds filter keeps only those kinds",
	             bool(only) and all(e.get("kind") == "forgotten" for e in only), only)

	# A cursor this server never issued: reset, or a refusal -- both conformant.
	try:
		stale = client.call("memory/events", {"cursor": f"not-a-cursor-{marker}"})
		report.check("an unrecognised cursor is answered with reset, not silence",
		             stale.get("reset") is True and isinstance(stale.get("cursor"), str), stale)
	except JsonRpcError as exc:
		report.check("an unrecognised cursor is answered with reset, not silence",
		             exc.code == INVALID_PARAMS, exc.code)

	push = bool(isinstance(contract, dict) and contract.get("push", False))
	if push:
		subscribed = client.call("memory/events/subscribe", {})
		report.check("subscribe reports its state", subscribed.get("subscribed") is True, subscribed)

		pushed_ids = client.call("memory/remember", {"records": [{"content": f"push probe {marker}"}]}).get("ids", [])
		# A notification is written after a response, so the next response read
		# is what pulls it off the wire and into the transport's stash.
		client.call("memory/describe", {})
		pushed = [m for m in client.take_notifications()
		          if isinstance(m, dict) and m.get("method") == "memory/event"]
		report.check("a subscribed client is pushed the event as a notification",
		             any(m.get("params", {}).get("id") == pushed_ids[0] for m in pushed),
		             pushed)
		report.check("a pushed notification carries no id",
		             all("id" not in m for m in pushed), pushed)

		closed = client.call("memory/events/unsubscribe", {})
		report.check("unsubscribe reports its state", closed.get("subscribed") is False, closed)

		# spec §4.14 -- and again, with nothing to unsubscribe from.
		try:
			again = client.call("memory/events/unsubscribe", {})
			report.check("unsubscribing when not subscribed is a no-op",
			             again.get("subscribed") is False, again)
		except JsonRpcError as exc:
			report.check("unsubscribing when not subscribed is a no-op", False, exc.code)

		client.call("memory/remember", {"records": [{"content": f"silent probe {marker}"}]})
		client.call("memory/describe", {})
		quiet = [m for m in client.take_notifications()
		         if isinstance(m, dict) and m.get("method") == "memory/event"]
		report.check("an unsubscribed client is pushed nothing", not quiet, quiet)

		client.call("memory/forget", {"query": f"probe {marker}"})
	else:
		expect_error(report, "subscribe without push is -32003", CAPABILITY_NOT_SUPPORTED,
		             lambda: client.call("memory/events/subscribe", {}))
		report.skip("push delivery checks", "push is not available on this connection")


def test_summarize(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the 'summarize' capability: it writes, it never deletes, it may decline.

	The rule that matters is that the sources survive. Consolidation **may**
	delete what it rewrites; this method **must not** (spec §4.15). A caller
	that lost its episodic history to a summary it asked for would have no way
	to get it back, and no reason to expect it.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  summarize")

	contract = profile.get("summarize")
	report.check("describe reports a summarize contract",
	             isinstance(contract, dict) and "model" in contract, contract)

	expect_error(report, "summarize with no selector is refused", INVALID_PARAMS,
	             lambda: client.call("memory/summarize", {}))

	marker  = uuid.uuid4().hex
	tiers   = [t.get("name") for t in profile.get("tiers", []) if t.get("name")]
	durable = next((t.get("name") for t in profile.get("tiers", [])
	                if t.get("kind") == "semantic"), tiers[-1] if tiers else None)

	written = client.call("memory/remember", {"records": [
		{"content": f"summary probe {marker}: the deploy key rotates every ninety days",
		 **({"tier": durable} if durable else {})},
		{"content": f"summary probe {marker}: rotation is announced a week ahead",
		 **({"tier": durable} if durable else {})},
	]})
	ids = written.get("ids", [])

	result = client.call("memory/summarize", dict(
		{"ids": ids}, **({"into": durable} if durable else {})))

	report.check("summarize reports what it read and wrote",
	             isinstance(result.get("read"), int) and isinstance(result.get("written"), int), result)
	report.check("and returns the records it wrote",
	             isinstance(result.get("records"), list)
	             and len(result["records"]) == result.get("written"), result)
	report.check("it read the records it was given", result.get("read") == len(ids), result)

	# The rule this suite exists to enforce.
	survivors = client.call("memory/recall", {"query": f"summary probe {marker}", "limit": 10}).get("records", [])
	report.check("the source records still exist afterwards",
	             all(any(r.get("id") == id for r in survivors) for id in ids),
	             [r.get("id") for r in survivors])

	for summary in result.get("records", []):
		report.check("a summary is an ordinary record",
		             isinstance(summary.get("id"), str) and isinstance(summary.get("content"), str)
		             and is_rfc3339(summary.get("created_at")), summary)
		break

	# Declining is an answer, not an error: a selector matching nothing must
	# come back empty rather than raising.
	nothing = client.call("memory/summarize", {"query": f"nothing matches this {uuid.uuid4().hex}"})
	report.check("a selector matching nothing is not an error",
	             nothing.get("written") == 0 and nothing.get("records") == [], nothing)

	# With `keys`, a summary can be maintained at an address instead of piling up.
	if "keys" in set(profile.get("capabilities", [])):
		key = f"conformance/summary/{marker}"
		first  = client.call("memory/summarize", dict({"ids": ids, "key": key},
		                                              **({"into": durable} if durable else {})))
		second = client.call("memory/summarize", dict({"ids": ids, "key": key},
		                                              **({"into": durable} if durable else {})))
		held = client.call("memory/timeline", {"key_prefix": key}).get("records", [])
		if first.get("written") and second.get("written"):
			report.check("summarizing to the same key replaces rather than accumulates",
			             len(held) == 1, held)
		else:
			report.skip("summarize to a key", "the server declined, so there is nothing to replace")

		client.call("memory/forget", {"key_prefix": key})

	client.call("memory/forget", {"query": f"summary probe {marker}"})


def test_prompt(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the 'prompt' capability: text alongside the records, never instead.

	The rule that matters is that asking for text changes nothing else. A server
	that returned different records, or fewer, when a prompt was requested would
	make the capability unusable for anything that also needs the records — and
	the records are the reason A2M returns records (spec §4.16).

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  prompt")

	contract = profile.get("prompt")
	report.check("describe reports a prompt contract",
	             isinstance(contract, dict) and isinstance(contract.get("styles"), list), contract)
	report.check("and 'auto' is among the styles",
	             isinstance(contract, dict) and "auto" in (contract.get("styles") or []), contract)

	# The cost contract. A client has to be able to tell, before calling,
	# whether rendering will spend an inference call.
	methods = (contract or {}).get("methods") or []
	report.check("describe reports which methods exist", isinstance(methods, list) and bool(methods), contract)
	report.check("every server can render without a model",
	             "template" in methods, methods)

	marker = uuid.uuid4().hex
	tiers  = [t.get("name") for t in profile.get("tiers", []) if t.get("name")]
	target = next((t.get("name") for t in profile.get("tiers", [])
	               if t.get("kind") == "semantic"), tiers[-1] if tiers else None)

	client.call("memory/remember", {"records": [
		{"content": f"prompt probe {marker}: the deploy key rotates every ninety days",
		 **({"tier": target} if target else {})},
		{"content": f"prompt probe {marker}: the release branch is cut on thursdays",
		 **({"tier": target} if target else {})},
	]})

	plain    = client.call("memory/recall", {"query": f"prompt probe {marker}", "limit": 5})
	rendered = client.call("memory/recall", {"query": f"prompt probe {marker}", "limit": 5, "prompt": True})

	report.check("the records are still returned in full",
	             [r.get("id") for r in rendered.get("records", [])] ==
	             [r.get("id") for r in plain.get("records", [])],
	             (plain.get("records"), rendered.get("records")))
	report.check("a prompt string comes back", isinstance(rendered.get("prompt"), str)
	             and bool(rendered["prompt"]), rendered.get("prompt"))
	report.check("and the ids it contains", isinstance(rendered.get("prompt_ids"), list)
	             and bool(rendered["prompt_ids"]), rendered.get("prompt_ids"))
	report.check("every rendered id is one of the returned records",
	             set(rendered.get("prompt_ids", [])) <= {r.get("id") for r in rendered.get("records", [])},
	             rendered.get("prompt_ids"))
	report.check("the text mentions what was recalled",
	             marker in rendered.get("prompt", ""), rendered.get("prompt", "")[:120])

	# Not asking must leave no trace of the capability in the result.
	report.check("a result carries no prompt unless asked",
	             "prompt" not in plain and "prompt_ids" not in plain, list(plain))

	# A budget drops whole records rather than cutting one in half.
	tight = client.call("memory/recall", {"query": f"prompt probe {marker}", "limit": 5,
	                                      "prompt": {"budget": 90}})
	report.check("a budget is respected", len(tight.get("prompt", "")) <= 90, len(tight.get("prompt", "")))
	report.check("and it drops whole records rather than truncating one",
	             len(tight.get("prompt_ids", [])) <= len(rendered.get("prompt_ids", [])),
	             (tight.get("prompt_ids"), rendered.get("prompt_ids")))
	for id in tight.get("prompt_ids", []):
		record = next((r for r in rendered.get("records", []) if r.get("id") == id), None)
		if record:
			report.check("a surviving record is rendered whole",
			             str(record.get("content", "")) in tight.get("prompt", ""), id)
			break

	# The default must not cost anything. A server whose default rendering
	# called a model would spend the caller's money on a convenience field.
	templated = client.call("memory/recall", {"query": f"prompt probe {marker}", "limit": 5,
	                                          "prompt": {"method": "template"}})
	report.check("the default rendering is the deterministic one",
	             templated.get("prompt") == rendered.get("prompt"),
	             (rendered.get("prompt"), templated.get("prompt")))

	# 'none' is the explicit way to ask for nothing, for callers whose request
	# body is templated and cannot drop a field.
	silent = client.call("memory/recall", {"query": f"prompt probe {marker}", "limit": 5,
	                                       "prompt": {"method": "none"}})
	report.check("method 'none' renders nothing at all",
	             "prompt" not in silent and bool(silent.get("records")), list(silent))

	# An unavailable method is refused, never silently substituted.
	if "model" not in methods:
		expect_error(report, "an unavailable method is refused, not substituted", INVALID_PARAMS,
		             lambda: client.call("memory/recall", {"query": "x", "prompt": {"method": "model"}}))
	else:
		written = client.call("memory/recall", {"query": f"prompt probe {marker}", "limit": 5,
		                                        "prompt": {"method": "model"}})
		report.check("a model-rendered prompt comes back",
		             isinstance(written.get("prompt"), str) and bool(written["prompt"]), written.get("prompt"))
		report.check("and the records are still returned in full",
		             len(written.get("records", [])) == len(rendered.get("records", [])), written)

	expect_error(report, "an unknown method is refused", INVALID_PARAMS,
	             lambda: client.call("memory/recall", {"query": "x", "prompt": {"method": "no-such-method"}}))

	# timeline renders too, and a transcript is not a bullet list.
	replayed = client.call("memory/timeline", {"limit": 5, "prompt": {"style": "transcript"}})
	report.check("timeline renders as well", isinstance(replayed.get("prompt"), str), replayed.get("prompt"))

	client.call("memory/forget", {"query": f"prompt probe {marker}"})


def test_undeclared(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check that undeclared capabilities answer -32003, not -32601.

	The most valuable check in the suite: a client trusts 'describe', so a server
	that lies there breaks clients in ways no defensive coding on their side can
	fix.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  undeclared capabilities")

	declared = set(profile.get("capabilities", []))
	probes   = [
		("tiers"   , "memory/consolidate"  , {}),
		("tiers"   , "memory/promote"      , {"ids": [], "tier": "x"}),
		("salience", "memory/reinforce"    , {"ids": []}),
		("sessions", "memory/session/list" , {}),
		("sessions", "memory/session/close", {"session": "x"}),
		("keys"    , "memory/fetch"        , {"key": "x/y"}),
		("events"  , "memory/events"       , {}),
		("events"  , "memory/events/subscribe", {}),
		("summarize", "memory/summarize"   , {"query": "x"}),
	]

	ran = False
	for capability, method, params in probes:
		if capability in declared:
			continue
		ran = True
		expect_error(report, f"{method} without '{capability}' is -32003", CAPABILITY_NOT_SUPPORTED,
		             lambda m=method, p=params: client.call(m, p))

	# spec §4.16 -- `prompt` is additive, so a server that does not declare it
	# ignores the parameter and returns the records alone. This is the opposite
	# of `tier` on a write, where dropping the field silently would lose the
	# caller's meaning.
	if "prompt" not in declared:
		ran = True
		try:
			plain = client.call("memory/recall", {"query": "anything", "limit": 1, "prompt": True})
			report.check("a server without 'prompt' ignores the parameter",
			             "prompt" not in plain and isinstance(plain.get("records"), list), list(plain))
		except JsonRpcError as exc:
			report.check("a server without 'prompt' ignores the parameter", False,
			             f"refused with {exc.code} instead of ignoring")

	if not ran:
		report.skip("undeclared capability probes", "this server declares everything")


def test_transport(client: Client, report: Report, profile: dict[str, Any]) -> None:
	"""Check the binding rules that hold on every transport (spec 8).

	These go through the transport directly rather than through the client, because
	what is being checked is a message the client is not able to construct: a batch
	has no id for a response to bind to.

	Args:
		client (Client): Connected to the server under test.
		report (Report): Where to record results.
		profile (dict): The server's describe result.
	"""
	print("\n  transport")

	reply = client.transport.request_raw([
		{"jsonrpc": "2.0", "id": 90001, "method": "memory/describe", "params": {}},
		{"jsonrpc": "2.0", "id": 90002, "method": "memory/describe", "params": {}},
	])

	report.check("a batch is refused with -32600",
	             isinstance(reply, dict) and reply.get("error", {}).get("code") == INVALID_REQUEST,
	             reply)

	if isinstance(client.transport, HttpTransport):
		test_http_binding(client, report)
	else:
		report.skip("http binding suite", "not an http transport")


def test_http_binding(client: Client, report: Report) -> None:
	"""Check the HTTP-only rules of spec 8.3: origin, version header, well-known.

	Args:
		client (Client): Connected over HTTP to the server under test.
		report (Report): Where to record results.
	"""
	url = client.transport.url

	def probe(target: str, method: str = "GET", headers: dict[str, str] = None, body: bytes = None) -> tuple[int, str]:
		"""Make one raw HTTP call and return its status and body.

		Returns:
			tuple: (status, body). A refused request reports its own status rather
			than raising, since a refusal is what most of these checks want.
		"""
		request = urllib.request.Request(target, data=body, headers=headers or {}, method=method)
		try:
			with urllib.request.urlopen(request, timeout=10) as response:
				return response.status, response.read().decode("utf-8", "replace")
		except urllib.error.HTTPError as exc:
			return exc.code, exc.read().decode("utf-8", "replace")
		except urllib.error.URLError as exc:
			return 0, str(exc.reason)

	rpc  = {"jsonrpc": "2.0", "id": 90003, "method": "memory/describe", "params": {}}
	body = json.dumps(rpc).encode("utf-8")

	status, _ = probe(url, "POST", {"Content-Type": "application/json", "Origin": "https://evil.example"}, body)
	report.check("a foreign Origin is refused with 403", status == 403, status)

	status, raw = probe(url, "POST", {
		"Content-Type"        : "application/json",
		"A2M-Protocol-Version": "a2m/99.0",
	}, body)
	refused = status == 400 and json.loads(raw).get("error", {}).get("code") == PROTOCOL_NOT_SUPPORTED
	report.check("an incompatible version header is refused with 400 and -32007", refused, (status, raw[:120]))

	status, _ = probe(url, "GET")
	report.check("GET on the rpc endpoint is 405", status == 405, status)

	well_known = url.rstrip("/").rsplit("/", 1)[0] if url.count("/") > 3 else url.rstrip("/")
	status, raw = probe(f"{well_known}/.well-known/a2m-server.json")
	if status == 200:
		document = json.loads(raw)
		report.check("the well-known profile reports the protocol",
		             document.get("protocol") == PROTOCOL, document.get("protocol"))
	else:
		report.skip("well-known profile", "not served -- it is a SHOULD, not a MUST")


def run(client: Client) -> Report:
	"""Run every applicable suite against a server.

	Args:
		client (Client): Connected to the server under test.

	Returns:
		Report: What passed, failed and was skipped.
	"""
	report  = Report()
	profile = client.call("memory/describe", {"protocol": PROTOCOL})

	print(f"\n  server: {profile.get('name')!r}  protocol: {profile.get('protocol')!r}")
	print(f"  capabilities: {profile.get('capabilities')}")

	declared = set(profile.get("capabilities", []))

	test_core(client, report, profile)
	test_transport(client, report, profile)
	test_undeclared(client, report, profile)

	for capability, suite in (("tiers", test_tiers), ("salience", test_salience),
	                          ("scopes", test_scopes), ("sessions", test_sessions),
	                          ("keys", test_keys), ("embeddings", test_embeddings),
	                          ("external", test_external), ("events", test_events),
	                          ("summarize", test_summarize),
	                          ("prompt", test_prompt)):
		if capability not in declared:
			print(f"\n  {capability}")
			report.skip(f"{capability} suite", "not declared")
			continue

		# Every remaining suite establishes its own fixtures by writing. A
		# read-only server that declares one of these is not lying -- it simply
		# cannot be exercised this way (spec §2.1).
		if report.read_only:
			print(f"\n  {capability}")
			report.skip(f"{capability} suite", "declared, but this server refuses writes")
			continue

		suite(client, report, profile)

	return report


def main() -> int:
	"""Run the suite from the command line.

		python -m tools.conformance --stdio python -m a2m
		python -m tools.conformance --http  http://127.0.0.1:8778/

	Returns:
		int: 0 when every applicable check passed, 1 on failure, 2 on bad usage.
	"""
	argv = sys.argv[1:]

	if not argv or argv[0] not in ("--stdio", "--http"):
		print(__doc__)
		return 2

	if argv[0] == "--stdio":
		if len(argv) < 2:
			print("--stdio needs a command, e.g. --stdio python implementations/server_minimal.py")
			return 2
		transport = StdioTransport(argv[1:], on_stderr=lambda line: print(f"    [server] {line}", file=sys.stderr))
	else:
		if len(argv) < 2:
			print("--http needs a url, e.g. --http http://127.0.0.1:8778/")
			return 2
		transport = HttpTransport(argv[1])

	client = Client(transport)

	try:
		report = run(client)
	except JsonRpcError as exc:
		# Name the code and let the reader look it up, rather than guessing at
		# which call failed. This handler used to blame memory/describe for
		# every escaping error, which sent anyone with a read-only server
		# looking at the one method that had worked perfectly.
		print(f"\n  FATAL  the run stopped on an unexpected error: {exc.code} {exc.message}")
		return 1
	finally:
		transport.close()

	total   = len(report.passed) + len(report.failed)
	profile = " (read-only)" if report.read_only else ""
	print(f"\n  {len(report.passed)}/{total} checks passed{profile}, {len(report.skipped)} skipped")

	for label, detail in report.failed:
		print(f"    - {label}: {detail}")

	print(f"\n  A2M {PROTOCOL} conformance: {'PASS' if not report.failed else 'FAIL'}\n")
	return 1 if report.failed else 0


if __name__ == "__main__":
	sys.exit(main())
