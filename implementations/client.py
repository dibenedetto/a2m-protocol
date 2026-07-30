"""A minimal A2M 0.1 client, written from the specification alone.

	python implementations/client.py --stdio python implementations/server_minimal.py -- describe
	python implementations/client.py --stdio python -m implementations.store_sqlite memory.db -- remember "the deploy key rotates every ninety days"
	python implementations/client.py --http http://127.0.0.1:8778/ -- recall "how often does the key change?"

This file deliberately imports **nothing from this project** — not jsonrpc.py,
not a2m/protocol.py. Only the standard library. [server_minimal.py](server_minimal.py) does the
same thing for the server side, and for the same reason: an implementation that
shares code with the reference proves only that the reference agrees with itself.

The two prove different halves, though. A server is judged by what it answers,
and the conformance suite can interrogate it. A client is judged by what it
*refrains* from doing, and nothing can interrogate that from outside. The rules
below have no error code attached because breaking them produces no error — just
a client that works against one server and misbehaves against the next:

- **Call `memory/describe` first, and honour it** (spec §2). A capability the
  server did not declare is one this client must not call. Asking anyway earns
  `-32003`, which is a fair answer to a question that should never have been put.
- **Tolerate unknown fields** (spec §2). Records are carried as dictionaries and
  handed back whole. A client that parses them into a fixed structure discards
  what a later version adds, which is the failure the rule exists to prevent.
- **Never compare `score` across calls or servers** (spec §5.3). There is no
  thresholding here, and no merging of two servers' results by score. A score
  orders one result list and means nothing outside it.
- **Resolve `uri` yourself, or not at all** (spec §3.8). The server must never
  dereference a caller's URI, so if anyone is going to fetch it, it is this side.
  `resolve()` exists, is opt-in, and says why.
- **Reuse an `id` when retrying** (spec §4.2). A write that fails at the network
  is indistinguishable from one that failed before arriving; a client-supplied id
  is the only thing that makes the retry safe.
"""


import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import uuid


PROTOCOL       = "a2m/0.1"
VERSION_HEADER = "A2M-Protocol-Version"

# spec §7. A client needs far fewer of these than a server: most are things it
# should have prevented rather than things it should recover from.
CAPABILITY_NOT_SUPPORTED = -32003
PROTOCOL_NOT_SUPPORTED   = -32007
INTERNAL_ERROR           = -32603


class A2MError(Exception):
	"""An error the server reported, carrying the code the specification assigns.
	"""

	def __init__(self, code: int, message: str, data=None) -> None:
		"""Build an error from a JSON-RPC error object.

		Args:
			code (int): The A2M or JSON-RPC error code.
			message (str): For humans. Never parse it -- spec §7 says so explicitly.
			data (Any, optional): Structured detail, when the server sent any.
		"""
		super().__init__(f"[{code}] {message}")
		self.code    = code
		self.message = message
		self.data    = data


class StdioTransport:
	"""The server as a child process, one JSON value per line (spec §8.2)."""

	def __init__(self, command: list[str]) -> None:
		"""Launch the server and start draining its stderr.

		Args:
			command (list[str]): What to run, e.g. ["python", "implementations/server_minimal.py"].

		Raises:
			OSError: If the command cannot be launched.
		"""
		self.command       = list(command)
		self.notifications = []
		self.counter       = 0
		self.process       = subprocess.Popen(
			self.command,
			stdin              = subprocess.PIPE,
			stdout             = subprocess.PIPE,
			stderr             = subprocess.PIPE,
			text               = True,
			encoding           = "utf-8",
			bufsize            = 1,
		)

		# A child that fills its stderr pipe blocks forever, whether or not anyone
		# is reading. The server is entitled to log there (spec §8.2).
		threading.Thread(target=self._drain_stderr, daemon=True).start()


	def call(self, method: str, params: dict) -> dict:
		"""Send one request and wait for the response with the matching id.

		Args:
			method (str): The A2M method name.
			params (dict): Its parameters.

		Returns:
			dict: The 'result' payload.

		Raises:
			A2MError: Carrying whatever the server reported.
		"""
		self.counter += 1
		id            = self.counter

		self._write({"jsonrpc": "2.0", "id": id, "method": method, "params": params})

		while True:
			message = self._read()

			# spec §8.2: notifications may arrive while a response is outstanding.
			# Stashing rather than discarding them costs nothing and means a client
			# that later grows an events capability does not lose the first one.
			if message.get("id", None) != id:
				self.notifications.append(message)
				continue

			return unwrap(message)


	def _write(self, payload: dict) -> None:
		"""Write one JSON value, newline-terminated.

		Args:
			payload (dict): The message to send.

		Raises:
			A2MError: -32603 if the child has already exited.
		"""
		if self.process.poll() is not None:
			raise A2MError(INTERNAL_ERROR, f"Server '{self.command[0]}' has exited")

		self.process.stdin.write(json.dumps(payload) + "\n")
		self.process.stdin.flush()


	def _read(self) -> dict:
		"""Read lines until one parses as JSON.

		Returns:
			dict: The parsed message.

		Raises:
			A2MError: -32603 if the stream closes first.
		"""
		while True:
			line = self.process.stdout.readline()
			if not line:
				raise A2MError(INTERNAL_ERROR, f"Server '{self.command[0]}' closed the stream")

			line = line.strip()
			if not line:
				continue

			try:
				return json.loads(line)
			except json.JSONDecodeError:
				continue


	def _drain_stderr(self) -> None:
		"""Consume the child's stderr so it cannot block on a full pipe.
		"""
		for line in self.process.stderr:
			print(f"    [server] {line.rstrip()}", file=sys.stderr)


	def close(self) -> None:
		"""Close stdin and wait, killing the child if it will not exit.

		Closing stdin is the shutdown signal (spec §8.2), and the only portable one.
		"""
		if self.process.poll() is None:
			try:
				self.process.stdin.close()
				self.process.wait(timeout=5)
			except Exception:
				self.process.kill()


class HttpTransport:
	"""One endpoint, POST, `application/json` (spec §8.3)."""

	def __init__(self, url: str, headers: dict = None, timeout: float = 30.0) -> None:
		"""Point the client at one endpoint.

		Args:
			url (str): The endpoint to POST to.
			headers (dict, optional): Extra headers, typically Authorization. Over a
				network these are what the server derives scope from -- not the
				'owner' this client claims (spec §6).
			timeout (float, optional): Per-request timeout in seconds.
		"""
		self.url     = url
		self.timeout = float(timeout)
		self.counter = 0
		self.headers = dict(headers) if headers else {}

		self.headers.setdefault("Content-Type" , "application/json")
		self.headers.setdefault("Accept"       , "application/json")
		self.headers.setdefault(VERSION_HEADER , PROTOCOL)


	def call(self, method: str, params: dict) -> dict:
		"""POST one request and read the response.

		Args:
			method (str): The A2M method name.
			params (dict): Its parameters.

		Returns:
			dict: The 'result' payload.

		Raises:
			A2MError: For a protocol failure, which arrives as HTTP 200 carrying a
				JSON-RPC error, or as a 4xx whose body carries one; and -32603 for a
				transport failure, which has no JSON-RPC error to report.
		"""
		self.counter += 1

		payload = {"jsonrpc": "2.0", "id": self.counter, "method": method, "params": params}
		request = urllib.request.Request(
			self.url,
			data    = json.dumps(payload).encode("utf-8"),
			headers = self.headers,
			method  = "POST",
		)

		try:
			with urllib.request.urlopen(request, timeout=self.timeout) as response:
				raw = response.read().decode("utf-8").strip()

		except urllib.error.HTTPError as exc:
			# A refused binding still explains itself in a JSON-RPC error object --
			# a mismatched version header is 400 with -32007 (spec §8.3.2).
			raw = exc.read().decode("utf-8", "replace").strip()
			try:
				return unwrap(json.loads(raw))
			except (json.JSONDecodeError, A2MError) as failure:
				if isinstance(failure, A2MError):
					raise
				raise A2MError(INTERNAL_ERROR, f"HTTP {exc.code} from {self.url}", raw[:500])

		except urllib.error.URLError as exc:
			raise A2MError(INTERNAL_ERROR, f"Cannot reach {self.url}", str(exc.reason))

		return unwrap(json.loads(raw)) if raw else None


	def close(self) -> None:
		"""Nothing to release: every call is its own request.
		"""
		pass


def unwrap(message: dict) -> dict:
	"""Turn a JSON-RPC response into a result, or raise its error.

	Args:
		message (dict): One parsed JSON-RPC response object.

	Returns:
		dict: The 'result' payload.

	Raises:
		A2MError: Carrying the server's code, message and data.
	"""
	error = message.get("error", None)
	if error:
		raise A2MError(error.get("code", INTERNAL_ERROR), error.get("message", ""), error.get("data", None))

	return message.get("result", None)


class A2MClient:
	"""An A2M client that refuses to call what the server did not declare."""

	def __init__(self, transport, owner: str = None) -> None:
		"""Negotiate with the server and remember what it can do.

		spec §2 requires a client to call memory/describe before anything else, and
		this is why: every other method here consults the profile first. Doing it in
		the constructor makes it impossible to forget.

		Args:
			transport: StdioTransport or HttpTransport.
			owner (str, optional): The scope this client claims. Over a network the
				server derives scope from the credentials instead and ignores this --
				it is not a security mechanism on either side (spec §6).

		Raises:
			A2MError: -32007 if the server does not speak this client's version.
		"""
		self.transport = transport
		self.owner     = owner
		self.profile   = transport.call("memory/describe", {"protocol": PROTOCOL})

		self.capabilities = list(self.profile.get("capabilities", []))
		self.tiers        = list(self.profile.get("tiers", []))


	def supports(self, capability: str) -> bool:
		"""Whether the server declared a capability.

		Args:
			capability (str): core, tiers, salience, scopes, sessions, embeddings,
				keys or external.

		Returns:
			bool: True if declared.
		"""
		return capability in self.capabilities


	def _require(self, capability: str) -> None:
		"""Refuse locally what the server never declared.

		The server would answer -32003 anyway. Failing here instead makes it a bug
		in the caller rather than a round trip, and names the capability that is
		missing rather than the method that needed it.

		Args:
			capability (str): The capability the caller is about to need.

		Raises:
			A2MError: -32003 without contacting the server.
		"""
		if not self.supports(capability):
			raise A2MError(
				CAPABILITY_NOT_SUPPORTED,
				f"This server declares {self.capabilities}, not '{capability}'",
			)


	def _call(self, method: str, params: dict) -> dict:
		"""Add the claimed scope, if any, and call.

		Args:
			method (str): The A2M method name.
			params (dict): Its parameters.

		Returns:
			dict: The result.
		"""
		if self.owner is not None and self.supports("scopes"):
			params = dict(params, owner=self.owner)

		return self.transport.call(method, params)


	def describe(self) -> dict:
		"""The profile negotiated at construction.

		Returns:
			dict: protocol, name, capabilities, methods, tiers and counts.
		"""
		return self.profile


	def remember(self, content: str, id: str = None, **fields) -> list[str]:
		"""Write one record.

		Args:
			content (str): The text to remember. It is what gets indexed, even for a
				record that also carries a 'uri' (spec §3.8).
			id (str, optional): Supply one to make the write idempotent: if a record
				with this id exists the server returns it rather than writing a second
				(spec §4.2). This is the only retry-safety A2M has, which is why this
				client generates one by default.
			**fields: Anything else the record should carry -- tier, key, role,
				metadata, session, embedding, uri, media_type. Unrecognised fields are
				the server's business to ignore, not this client's to filter.

		Returns:
			list[str]: The ids written.
		"""
		record = dict(fields, content=content, id=id or uuid.uuid4().hex)

		if "key" in record:
			self._require("keys")
		if "embedding" in record:
			self._require("embeddings")
		if "uri" in record:
			self._require("external")
		if "tier" in record:
			self._require("tiers")

		return self._call("memory/remember", {"records": [record]}).get("ids", [])


	def recall(self, query: str, limit: int = 5, **filters) -> list[dict]:
		"""Search by relevance.

		The records come back as the server sent them, whole. Each carries a 'score'
		that orders *this* list and nothing else: it is not comparable with a score
		from another call, another server or a fixed threshold (spec §5.3).

		Args:
			query (str): What the caller is trying to remember.
			limit (int, optional): How many records at most.
			**filters: tier, where, session, key_prefix, embeddings -- whichever the
				server's declared capabilities support.

		Returns:
			list[dict]: Records in descending score order.
		"""
		if "key_prefix" in filters:
			self._require("keys")

		return self._call("memory/recall", dict(filters, query=query, limit=limit)).get("records", [])


	def timeline(self, tier: str = None, limit: int = 0, **filters) -> list[dict]:
		"""Replay in creation order.

		This is not recall with a different sort. Working memory is *replayed*, and
		reordering a transcript by relevance destroys it -- which is why the two are
		separate methods and neither may implement the other (spec §4.4).

		Args:
			tier (str, optional): Which tier to replay. The working tier by default,
				on servers that have tiers.
			limit (int, optional): How many records; the newest when it truncates.
			**filters: where, session.

		Returns:
			list[dict]: Records ascending by created_at.
		"""
		params = dict(filters, limit=limit)
		if tier is not None:
			self._require("tiers")
			params["tier"] = tier

		return self._call("memory/timeline", params).get("records", [])


	def forget(self, **selectors) -> int:
		"""Delete by id, key or query.

		Args:
			**selectors: ids, key, query, tier -- at least one. A call with no
				selector is -32602 rather than a store-wide delete, which is the only
				sane way for that mistake to end.

		Returns:
			int: How many records were removed.
		"""
		if "key" in selectors:
			self._require("keys")

		return self._call("memory/forget", selectors).get("forgotten", 0)


	def sessions(self) -> list[dict]:
		"""List the conversations this store knows about.

		Returns:
			list[dict]: One entry per session.
		"""
		self._require("sessions")
		return self._call("memory/session/list", {}).get("sessions", [])


	def close_session(self, session: str) -> dict:
		"""Finish a conversation.

		Closing percolates the whole stack rather than one hop (spec §4.11): a
		conversation that will never be replayed leaves working memory immediately
		instead of waiting for capacity pressure, and every tier below applies its
		own rules to what arrives. It is not a delete -- the facts survive below.

		Args:
			session (str): The conversation to close.

		Returns:
			dict: What moved where.
		"""
		self._require("sessions")
		return self._call("memory/session/close", {"session": session})


	def fetch(self, key: str) -> dict:
		"""Read the one record at a key.

		Args:
			key (str): The address, e.g. "user/city".

		Returns:
			dict | None: The record, or None if the key holds nothing.
		"""
		self._require("keys")
		return self._call("memory/fetch", {"key": key}).get("record", None)


	def resolve(self, record: dict, opener=None):
		"""Dereference a record's 'uri' -- deliberately, and on this side.

		The server must never do this (spec §3.8): its job is accepting arbitrary
		strings from agents, and a component that fetches those strings is a
		request-forgery primitive sitting behind whatever credentials the memory
		service happens to hold. So resolution is the client's, which has the
		context to know whether a given URI should be fetched at all.

		That context is not here either. This method takes an opener rather than
		calling urlopen, so that deciding what is safe to fetch stays with the
		caller who knows.

		Args:
			record (dict): A record that may carry 'uri'.
			opener (Callable, optional): Called with the uri. Without one, this
				method returns None rather than guessing that fetching is safe.

		Returns:
			Any: Whatever the opener returned, or None.
		"""
		uri = record.get("uri", None)
		if uri is None or opener is None:
			return None

		return opener(uri)


	def close(self) -> None:
		"""Release the transport.
		"""
		self.transport.close()


def connect_stdio(command: list[str], owner: str = None) -> A2MClient:
	"""Start a server as a child process and negotiate with it.

	Args:
		command (list[str]): What to run.
		owner (str, optional): The scope to claim.

	Returns:
		A2MClient: Connected and negotiated.
	"""
	return A2MClient(StdioTransport(command), owner=owner)


def connect_http(url: str, headers: dict = None, owner: str = None) -> A2MClient:
	"""Negotiate with a server over HTTP.

	Args:
		url (str): The endpoint.
		headers (dict, optional): Extra headers, typically Authorization.
		owner (str, optional): The scope to claim. A correct server ignores it in
			favour of the credentials (spec §6).

	Returns:
		A2MClient: Connected and negotiated.
	"""
	return A2MClient(HttpTransport(url, headers=headers), owner=owner)


def searchable_tier(client: A2MClient) -> str:
	"""The first tier a recall can actually reach.

	Working memory is replayed, never searched (spec §4.4), so a record written
	there is invisible to recall by design -- ranking a transcript destroys it.
	A caller writing a standalone fact wants the tier below.

	Args:
		client (A2MClient): A negotiated client.

	Returns:
		str | None: A tier name, or None if the server exposes no tiers.
	"""
	for tier in client.tiers:
		if tier.get("kind", None) != "working":
			return tier.get("name", None)

	return None


def main() -> int:
	"""Run one command against a server.

		python implementations/client.py --stdio python implementations/server_minimal.py -- describe
		python implementations/client.py --http http://127.0.0.1:8778/ -- recall "which fallback region"

	Returns:
		int: 0 on success, 1 on an A2M error, 2 on bad usage.
	"""
	argv = sys.argv[1:]

	if not argv or argv[0] not in ("--stdio", "--http"):
		print(__doc__)
		return 2

	if "--" not in argv:
		print("Separate the connection from the command with '--'.")
		return 2

	split      = argv.index("--")
	connection = argv[:split]
	command    = argv[split + 1:]

	if len(connection) < 2 or not command:
		print("Both a server and a command are needed.")
		return 2

	if connection[0] == "--stdio":
		client = connect_stdio(connection[1:])
	else:
		client = connect_http(connection[1])

	try:
		verb, rest = command[0], command[1:]

		if verb == "describe":
			print(json.dumps(client.describe(), indent=2))

		elif verb == "remember":
			fields = {}

			if "--tier" in rest:
				index          = rest.index("--tier")
				fields["tier"] = rest[index + 1]
				rest           = rest[:index] + rest[index + 2:]
			elif client.supports("tiers"):
				# Otherwise this lands in working memory, which recall never searches,
				# and the next command appears to lose it. See searchable_tier.
				fields["tier"] = searchable_tier(client)

			if not rest:
				print("remember needs some text.")
				return 2

			print(json.dumps(client.remember(" ".join(rest), **fields)))

		elif verb == "recall":
			if not rest:
				print("recall needs a query.")
				return 2
			for record in client.recall(" ".join(rest)):
				print(f"  {record.get('score', 0):.3f}  {record.get('content', '')}")

		elif verb == "timeline":
			for record in client.timeline(tier=rest[0] if rest else None):
				print(f"  {record.get('created_at', '')}  {record.get('content', '')}")

		elif verb == "fetch":
			if not rest:
				print("fetch needs a key.")
				return 2
			print(json.dumps(client.fetch(rest[0]), indent=2))

		else:
			print(f"Unknown command '{verb}'. Try describe, remember, recall, timeline or fetch.")
			return 2

	except A2MError as exc:
		print(f"  {exc}", file=sys.stderr)
		if exc.data is not None:
			print(f"  data: {json.dumps(exc.data)}", file=sys.stderr)
		return 1

	finally:
		client.close()

	return 0


if __name__ == "__main__":
	sys.exit(main())
