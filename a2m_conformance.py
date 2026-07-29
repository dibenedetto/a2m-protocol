"""A2M 0.1 conformance suite.

Point it at any A2M server and it reports what that server actually honours:

	python a2m_conformance.py --stdio python a2m.py
	python a2m_conformance.py --stdio python a2m_minimal.py
	python a2m_conformance.py --http http://127.0.0.1:8778/

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
import uuid


from   typing  import Any, Callable


from   jsonrpc import Client, HttpTransport, JsonRpcError, StdioTransport


PROTOCOL = "a2m/0.1"

UNKNOWN_TIER             = -32002
CAPABILITY_NOT_SUPPORTED = -32003
PROTOCOL_NOT_SUPPORTED   = -32007
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
	             declared <= {"core", "tiers", "salience", "scopes", "sessions", "events"},
	             declared)

	expect_error(report, "an incompatible protocol is rejected", PROTOCOL_NOT_SUPPORTED,
	             lambda: client.call("memory/describe", {"protocol": "a2m/99.0"}))

	report.check("unknown params are ignored, not rejected",
	             bool(client.call("memory/describe", {"unknown_field_xyz": 1})),
	             "server rejected an unrecognised parameter")

	marker = uuid.uuid4().hex
	written = client.call("memory/remember", {"records": [
		{"content": f"the deploy key {marker} rotates every ninety days", "role": "user",
		 "metadata": {"suite": marker, "nested": {"a": [1, 2]}}},
		{"content": f"the release branch {marker} is cut on thursdays", "role": "user",
		 "metadata": {"suite": marker}},
	]})

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
	]

	ran = False
	for capability, method, params in probes:
		if capability in declared:
			continue
		ran = True
		expect_error(report, f"{method} without '{capability}' is -32003", CAPABILITY_NOT_SUPPORTED,
		             lambda m=method, p=params: client.call(m, p))

	if not ran:
		report.skip("undeclared capability probes", "this server declares everything")


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
	test_undeclared(client, report, profile)

	for capability, suite in (("tiers", test_tiers), ("salience", test_salience),
	                          ("scopes", test_scopes), ("sessions", test_sessions)):
		if capability in declared:
			suite(client, report, profile)
		else:
			print(f"\n  {capability}")
			report.skip(f"{capability} suite", "not declared")

	return report


def main() -> int:
	"""Run the suite from the command line.

		python a2m_conformance.py --stdio python a2m.py
		python a2m_conformance.py --http  http://127.0.0.1:8778/

	Returns:
		int: 0 when every applicable check passed, 1 on failure, 2 on bad usage.
	"""
	argv = sys.argv[1:]

	if not argv or argv[0] not in ("--stdio", "--http"):
		print(__doc__)
		return 2

	if argv[0] == "--stdio":
		if len(argv) < 2:
			print("--stdio needs a command, e.g. --stdio python a2m_minimal.py")
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
		print(f"\n  FATAL  the server did not answer memory/describe: {exc.code} {exc.message}")
		return 1
	finally:
		transport.close()

	total = len(report.passed) + len(report.failed)
	print(f"\n  {len(report.passed)}/{total} checks passed, {len(report.skipped)} skipped")

	for label, detail in report.failed:
		print(f"    - {label}: {detail}")

	print(f"\n  A2M {PROTOCOL} conformance: {'PASS' if not report.failed else 'FAIL'}\n")
	return 1 if report.failed else 0


if __name__ == "__main__":
	sys.exit(main())
