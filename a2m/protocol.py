"""A2M — the Agent-to-Memory Protocol. Reference implementation of `a2m/0.1`.

The normative specification is in spec/a2m-0.1.md; this module is one
implementation of it, and where the two disagree the specification wins.

A small JSON-RPC 2.0 method namespace that puts a memory store behind the same
kind of boundary MCP puts a tool behind. An agent talks to `memory/*` and never
learns whether the store is a Python object in its own process, a subprocess, or
a service across the network.

	core      memory/describe    version, capabilities, tier layout
	          memory/remember    write records
	          memory/recall      relevance-ordered search
	          memory/timeline    creation-ordered read, for rebuilding a transcript
	          memory/forget      delete by id, query, tier or metadata
	tiers     memory/promote     move records to a tier they have earned
	          memory/consolidate promote what was earned, push down what overflows
	salience  memory/reinforce   raise the salience of records that proved useful
	sessions  memory/session/list   which conversations exist
	          memory/session/close  end one and let it percolate down the stack
	keys      memory/fetch          read the record at an address
	embeddings                      caller-owned vectors, stored verbatim
	external                        records that point at a file, URL or blob

This server declares every capability. A store that cannot do tiers or salience
is still conformant if it declares only `core` — see server_minimal.py, which is
written from the specification alone and implements exactly that.
"""


import inspect
import json
import os
import sys
import threading
import time


from   collections import deque
from   typing      import Any, Callable


from   a2m.jsonrpc import (
	Client, Dispatcher, INVALID_PARAMS, JsonRpcError,
	HttpTransport, LocalTransport, StdioTransport, make_error_response,
	make_notification, serve_http, serve_stdio,
)
from   a2m.memory  import MemoryStack, to_rfc3339


A2M_VERSION = "a2m/0.1"

# The HTTP binding mirrors the protocol version into a header so that gateways can
# route and reject without parsing a body, and publishes the describe profile at a
# well-known path so a server can be found before it is called. See spec §8.3.
A2M_VERSION_HEADER = "A2M-Protocol-Version"
A2M_WELL_KNOWN     = "/.well-known/a2m-server.json"

# The A2M error block, allocated from the -32000..-32099 range JSON-RPC reserves
# for implementation-defined server errors. See spec §7.
UNKNOWN_RECORD           = -32001
UNKNOWN_TIER             = -32002
CAPABILITY_NOT_SUPPORTED = -32003
READ_ONLY                = -32004
SCOPE_DENIED             = -32005
QUOTA_EXCEEDED           = -32006
PROTOCOL_NOT_SUPPORTED   = -32007
EMBEDDING_MISMATCH       = -32008

# Which capability each method belongs to. A server must answer a call into an
# undeclared capability with CAPABILITY_NOT_SUPPORTED rather than
# METHOD_NOT_FOUND, since a client cannot tell the latter from a typo.
A2M_CAPABILITIES = {
	"core"       : ["memory/describe", "memory/remember", "memory/recall", "memory/timeline", "memory/forget"],
	"tiers"      : ["memory/promote", "memory/consolidate"],
	"salience"   : ["memory/reinforce"],
	"scopes"     : [],
	"sessions"   : ["memory/session/list", "memory/session/close"],
	"embeddings" : [],
	"keys"       : ["memory/fetch"],
	"external"   : [],
	"events"     : ["memory/events", "memory/events/subscribe", "memory/events/unsubscribe"],
}

A2M_METHODS = [method for methods in A2M_CAPABILITIES.values() for method in methods]

# The one server-to-client message in the protocol: an event, pushed after
# memory/events/subscribe. A notification, never a request (spec §4.13, §8).
A2M_EVENT_NOTIFICATION = "memory/event"


class EventLog:
	"""A bounded, in-memory event log with opaque cursors (spec §4.12).

	The log is the single source both delivery modes read from: polling walks it
	with a cursor, push mirrors appends to a subscriber. Bounding it is what
	makes `events` safe to declare on any server — an unpolled log costs a fixed
	amount of memory, never an unbounded one.

	A cursor encodes a per-log salt plus a sequence number. The salt is what
	makes it *opaque in practice*, not only by decree: a cursor from another
	server, or from a previous run of this one, fails the salt check and is
	answered with `reset` rather than misread as a position.

	Example:
		>>> log = EventLog(retain=2)
		>>> start = log.head()
		>>> _ = log.append("written", id="a")
		>>> _ = log.append("written", id="b")
		>>> [e["id"] for e in log.read(cursor=start)["events"]]
		['a', 'b']
	"""

	def __init__(self, retain: int = 1024) -> None:
		"""Create an empty log.

		Args:
			retain (int, optional): How many events to keep. Older events fall
				off, and a cursor pointing before them answers with `reset`.
		"""
		self.retain  = int(retain)
		self._salt   = os.urandom(4).hex()
		self._events : deque = deque(maxlen=self.retain)
		self._seq    = 0
		self._lock   = threading.Lock()


	def append(self, kind: str, owner: str = None, **fields: Any) -> dict[str, Any]:
		"""Record one event.

		Args:
			kind (str): written, forgotten, promoted, consolidated,
				session_closed, or anything else — clients must tolerate kinds
				they do not recognise.
			owner (str, optional): The scope the event belongs to. Scoped reads
				see only their own events plus unscoped ones, mirroring recall
				visibility (spec §4.12); the owner itself is not exposed on the
				wire.
			**fields: Event payload. None values are dropped.

		Returns:
			dict: The event as it will appear on the wire, with `at` stamped.
		"""
		event = {"kind": kind, "at": to_rfc3339(time.time())}
		event.update({name: value for name, value in fields.items() if value is not None})

		with self._lock:
			self._seq += 1
			self._events.append((self._seq, owner, event))

		return event


	def head(self) -> str:
		"""The cursor for "now": a poll from here returns nothing yet.

		Returns:
			str: An opaque cursor.
		"""
		with self._lock:
			return f"{self._salt}:{self._seq}"


	def read(self, cursor: str = None, limit: int = 256, kinds: list[str] = None, agent: str = None) -> dict[str, Any]:
		"""Read events after a cursor, oldest first (spec §4.12).

		Successive reads, each passing the cursor the previous one returned, see
		every retained event exactly once, in order. A cursor this log did not
		issue — another server's, or a previous run's — comes back with `reset`
		set and the read continuing from the oldest retained event, which is
		also what happens when a valid cursor has fallen off the retention
		window.

		Args:
			cursor (str, optional): Where to read from. Absent means now: no
				events, just the current cursor.
			limit (int, optional): Maximum events to return. 0 means no limit.
			kinds (list[str], optional): Keep only these kinds.
			agent (str, optional): Scope. A scoped read sees its own events and
				unscoped ones; an unscoped read sees everything.

		Returns:
			dict: `events` and `cursor`, plus `more` and `reset` when true.
		"""
		with self._lock:
			if cursor is None:
				return {"events": [], "cursor": f"{self._salt}:{self._seq}"}

			salt, _, seq_text = str(cursor).partition(":")
			oldest            = self._events[0][0] if self._events else self._seq + 1
			reset             = False

			if salt != self._salt or not seq_text.isdigit():
				reset = True
				since = oldest - 1
			else:
				since = min(int(seq_text), self._seq)
				if since < oldest - 1:
					reset = True
					since = oldest - 1

			selected = []
			more     = False
			for seq, owner, event in self._events:
				if seq <= since:
					continue
				if agent is not None and owner is not None and owner != agent:
					continue
				if kinds and event["kind"] not in kinds:
					continue
				if limit and len(selected) >= limit:
					more = True
					break
				selected.append((seq, event))

			# When the walk completed, the cursor jumps to the head so events
			# this reader filtered out are never offered to it again.
			position = selected[-1][0] if more and selected else self._seq
			result   = {"events": [event for _, event in selected], "cursor": f"{self._salt}:{position}"}
			if more:
				result["more"] = True
			if reset:
				result["reset"] = True
			return result


class MemoryServer:
	"""Exposes a memory stack over A2M.

	Storage-agnostic by construction: it drives anything presenting the MemoryStack
	interface. store_sqlite.SqliteMemoryStack is a completely different engine and this
	class runs it unchanged, which is the payoff for having a protocol boundary.

	Example:
		server = MemoryServer(MemoryStack(), name="my-memory")
		serve_stdio(server.dispatcher)              # speak A2M on stdin/stdout

		core_only = MemoryServer(capabilities=["core"])
	"""

	def __init__(
		self,
		stack                : MemoryStack = None,
		name                 : str         = "agent-memory",
		capabilities         : list[str]   = None,
		max_records_per_call : int         = 256,
		dimensions           : int         = None,
		embedding_model      : str         = None,
		push_events          : bool        = True,
		event_retention      : int         = 1024,
	) -> None:
		"""Wrap a store as an A2M server.

		Args:
			stack (MemoryStack, optional): The store to expose. A fresh in-memory
				MemoryStack when omitted.
			name (str, optional): Human-readable server name, returned by describe.
			capabilities (list[str], optional): What this server admits to
				implementing. Narrowing it is how a store that cannot do tiers or
				salience stays conformant. Defaults to everything.
			max_records_per_call (int, optional): Batch limit enforced on remember,
				reported to clients under 'limits'.
			push_events (bool, optional): Whether this server is willing to push
				events after memory/events/subscribe. Even when True, push is only
				reported on a binding that can deliver a notification — the
				serving wiring sets `push_transport`, and HTTP never does
				(spec §4.13). Polling is unaffected either way.
			event_retention (int, optional): How many events the poll log keeps.

		Raises:
			ValueError: If 'core' is not among the capabilities. Every A2M server
				must implement core.
		"""
		self.stack                = stack or MemoryStack()
		self.name                 = name
		self.capabilities         = list(capabilities) if capabilities else list(A2M_CAPABILITIES)
		self.max_records_per_call = int(max_records_per_call)
		self.dimensions           = dimensions
		self.embedding_model      = embedding_model
		self.push_events          = bool(push_events)
		self.events_log           = EventLog(retain=event_retention)
		# A2M messages are single objects: a batch has no id to bind a response to,
		# and nothing in the protocol needs one (spec §8).
		self.dispatcher           = Dispatcher(allow_batch=False)

		# Set True by a binding that can carry a notification back to the client:
		# serve_a2m_stdio and connect_local do, serve_a2m_http never does.
		self.push_transport       = False
		self._subscription        = None
		self._notify_sink         = None
		self._outbox              : list[dict[str, Any]] = []

		if "core" not in self.capabilities:
			raise ValueError("An A2M server must implement the 'core' capability")

		self.methods = [m for c in self.capabilities for m in A2M_CAPABILITIES.get(c, [])]

		handlers = {
			"memory/describe"    : self.describe,
			"memory/remember"    : self.remember,
			"memory/recall"      : self.recall,
			"memory/timeline"    : self.timeline,
			"memory/forget"      : self.forget,
			"memory/promote"     : self.promote,
			"memory/consolidate" : self.consolidate,
			"memory/reinforce"   : self.reinforce,
			"memory/session/list": self.session_list,
			"memory/session/close": self.session_close,
			"memory/fetch"       : self.fetch,
			"memory/events"      : self.events,
			"memory/events/subscribe"  : self.events_subscribe,
			"memory/events/unsubscribe": self.events_unsubscribe,
		}

		# Every A2M method is registered, including those of undeclared
		# capabilities: they must answer CAPABILITY_NOT_SUPPORTED, and an
		# unregistered method would answer METHOD_NOT_FOUND instead.
		for method, handler in handlers.items():
			self.dispatcher.register(method, self._guard(method, handler))


	def _guard(self, method: str, handler: Callable) -> Callable:
		"""Wrap a handler so an undeclared capability answers correctly.

		Every A2M method is registered even when its capability is not declared,
		because an unregistered method would answer -32601 METHOD_NOT_FOUND and a
		client cannot tell that from a typo. Spec 2 requires -32003 instead.

		Args:
			method (str): The A2M method name.
			handler (Callable): The bound method implementing it.

		Returns:
			Callable: The guarded handler, carrying the original signature so the
			dispatcher can still bind parameters against it.
		"""
		def guarded(*args: Any, **kwargs: Any) -> Any:
			if method not in self.methods:
				capability = next((c for c, ms in A2M_CAPABILITIES.items() if method in ms), "unknown")
				raise JsonRpcError(
					CAPABILITY_NOT_SUPPORTED,
					f"This server does not implement the '{capability}' capability",
					{"capability": capability, "capabilities": list(self.capabilities)},
				)
			return handler(*args, **kwargs)

		# The dispatcher binds parameters against the signature, so it must see the
		# real one rather than (*args, **kwargs).
		guarded.__signature__ = inspect.signature(handler)
		return guarded


	def supports(self, capability: str) -> bool:
		"""Whether this server declares a capability.

		Args:
			capability (str): core, tiers, salience, scopes, sessions, embeddings,
				keys or external.

		Returns:
			bool: True if declared.
		"""
		return capability in self.capabilities


	# Every handler ends in **ignored. Spec §2 requires a server to ignore
	# parameters it does not recognise rather than reject them, which is what
	# lets an 0.2 client talk to an 0.1 server at all.

	def describe(self, protocol: str = None, owner: str = None, **ignored: Any) -> dict[str, Any]:
		# Pre-1.0, compatibility requires an exact minor match (spec §4.1).
		"""Handle 'memory/describe' -- discovery and version negotiation.

		Args:
			protocol (str, optional): The version the client speaks. A mismatch raises
				rather than pretending, since pre-1.0 compatibility requires an exact
				minor match (spec 4.1).
			owner (str, optional): Scope the counts.
			**ignored: Unrecognised parameters are ignored, not rejected -- this is
				what lets a newer client talk to an older server (spec 2).

		Returns:
			dict: protocol, name, capabilities, methods, tiers, limits and counts.

		Raises:
			JsonRpcError: -32007 PROTOCOL_NOT_SUPPORTED on an incompatible version.
		"""
		if protocol is not None and protocol != A2M_VERSION:
			raise JsonRpcError(
				PROTOCOL_NOT_SUPPORTED,
				f"This server speaks {A2M_VERSION}, not {protocol}",
				{"supported": [A2M_VERSION]},
			)

		described = self.stack.describe(agent=owner)

		return {
			"protocol"     : A2M_VERSION,
			"name"         : self.name,
			"capabilities" : list(self.capabilities),
			"methods"      : sorted(self.methods),
			"tiers"        : described.get("tiers", []),
			"limits"       : {"max_records_per_call": self.max_records_per_call},
			"embeddings"   : self._embedding_profile() if self.supports("embeddings") else None,
			"events"       : {"push": self._push_available()} if self.supports("events") else None,
			"scorer"       : described.get("scorer", None),
			"total"        : described.get("total", 0),
			"working"      : described.get("working", None),
		}


	def remember(self, records: list[dict[str, Any]] = None, owner: str = None,
	             session: str = None, **record) -> dict[str, Any]:
		"""Handle 'memory/remember' -- write records.

		Args:
			records (list[dict], optional): Partial records, each needing 'content'.
				Each may also carry id, role, metadata, group, tier, salience, owner
				and session.
			owner (str, optional): Default owner for records that do not set one.
			session (str, optional): Default session for records that do not set one.
			**record: A single record passed as loose keyword arguments, for
				convenience when 'records' is omitted.

		Returns:
			dict: 'ids' (one per input record, in order), the destination tier and the
			store's total.

		Raises:
			JsonRpcError: -32602 when a record has no content, -32006 when the batch
				exceeds max_records_per_call, -32003 when a tier is named but 'tiers'
				is not declared, -32002 for an unknown tier.
		"""
		if records is None:
			records = [record] if record else []

		if not records:
			raise JsonRpcError(INVALID_PARAMS, "Nothing to remember: pass 'records' or a single record")

		if len(records) > self.max_records_per_call:
			raise JsonRpcError(
				QUOTA_EXCEEDED,
				f"At most {self.max_records_per_call} records per call, got {len(records)}",
			)

		written = []
		for entry in records:
			if not isinstance(entry, dict) or "content" not in entry:
				raise JsonRpcError(INVALID_PARAMS, "Each record needs a 'content' field")

			# Silently dropping a caller's tier placement is a correctness bug, not
			# a cosmetic one, so this fails rather than ignoring the field.
			if entry.get("tier", None) is not None and not self.supports("tiers"):
				raise JsonRpcError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'tiers' capability")

			# Silently dropping a key or a vector is worse than refusing it: the
			# caller would believe it had addressed a record, or stored a
			# comparable embedding, and neither would be true.
			if entry.get("key", None) is not None and not self.supports("keys"):
				raise JsonRpcError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'keys' capability")

			if entry.get("uri", None) is not None and not self.supports("external"):
				raise JsonRpcError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'external' capability")

			if entry.get("embedding", None) is not None:
				if not self.supports("embeddings"):
					raise JsonRpcError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'embeddings' capability")
				self._check_dimensions(entry["embedding"])

			try:
				stored = self.stack.remember(
					content  = entry["content"],
					tier     = entry.get("tier"    , None  ),
					role     = entry.get("role"    , "user"),
					salience = entry.get("salience", 1.0   ),
					group    = entry.get("group"   , None  ),
					metadata = entry.get("metadata", None  ),
					owner    = entry.get("owner"   , owner ),
					session  = entry.get("session" , session),
					key      = entry.get("key"     , None  ),
					embedding= entry.get("embedding", None ),
					uri      = entry.get("uri"     , None  ),
					media_type = entry.get("media_type", None),
					id       = entry.get("id"      , None  ),
				)
			except KeyError as exc:
				raise JsonRpcError(UNKNOWN_TIER, str(exc))

			written.append(stored)

		for stored in written:
			self._emit(
				"written",
				owner    = stored.owner,
				id       = stored.id,
				tier     = stored.tier,
				session  = stored.session,
				key      = stored.key,
				revision = stored.revision if getattr(stored, "key", None) else None,
			)

		return {"ids": [r.id for r in written], "tier": written[-1].tier, "total": self.stack.count()}


	def recall(
		self,
		query     : str            = None,
		tier      : str            = None,
		limit     : int            = 8,
		where     : dict[str, Any] = None,
		min_score : float          = 0.0,
		owner     : str            = None,
		embedding : list[float]    = None,
		key_prefix: str            = None,
		embeddings: bool           = False,
		**ignored : Any,
	) -> dict[str, Any]:
		"""Handle 'memory/recall' -- relevance-ordered search.

		Args:
			query (str, optional): What to rank against.
			tier (str, optional): Restrict to one tier.
			limit (int, optional): Maximum records.
			where (dict, optional): Metadata filter.
			min_score (float, optional): Drop results below this score.
			owner (str, optional): Scope the search.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: 'records', each in wire form with a 'score', descending.

		Raises:
			JsonRpcError: -32002 for an unknown tier.
		"""
		if embedding is not None:
			if not self.supports("embeddings"):
				raise JsonRpcError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'embeddings' capability")
			self._check_dimensions(embedding)

		if key_prefix is not None and not self.supports("keys"):
			raise JsonRpcError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'keys' capability")

		try:
			scored = self.stack.recall(
				query     = query,
				tier      = tier,
				limit     = limit,
				where     = where,
				min_score = min_score,
				agent     = owner,
				embedding = embedding,
				key_prefix= key_prefix,
			)
		except KeyError as exc:
			raise JsonRpcError(UNKNOWN_TIER, str(exc))

		return {"records": [record.to_dict(score, embeddings) for record, score in scored]}


	def timeline(self, tier: str = None, limit: int = 0, owner: str = None,
	             where: dict[str, Any] = None, key_prefix: str = None,
	             embeddings: bool = False, **ignored: Any) -> dict[str, Any]:
		"""Handle 'memory/timeline' -- creation-ordered read.

		Args:
			tier (str, optional): Restrict to one tier.
			limit (int, optional): Keep the most recent N, still ascending.
			owner (str, optional): Scope the read.
			where (dict, optional): Metadata filter. This is what makes replaying one
				session possible.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: 'records', ascending by created_at.
		"""
		if key_prefix is not None and not self.supports("keys"):
			raise JsonRpcError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'keys' capability")

		records = self.stack.timeline(tier=tier, limit=limit, agent=owner,
		                              where=where, key_prefix=key_prefix)
		return {"records": [record.to_dict(embedding=embeddings) for record in records]}


	def reinforce(self, ids: list[str], amount: float = 0.5, owner: str = None, **ignored: Any) -> dict[str, Any]:
		"""Handle 'memory/reinforce' -- raise salience.

		Args:
			ids (list[str]): Records to reinforce.
			amount (float, optional): How much to add.
			owner (str, optional): Scope.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: {'reinforced': count}.
		"""
		return {"reinforced": self.stack.reinforce(ids, amount, agent=owner)}


	def promote(self, ids: list[str], tier: str, salience: float = 0.5, owner: str = None, **ignored: Any) -> dict[str, Any]:
		"""Handle 'memory/promote' -- move records to a tier they earned.

		Args:
			ids (list[str]): Records to move.
			tier (str): Destination tier.
			salience (float, optional): Added on arrival.
			owner (str, optional): Scope.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: {'promoted': count}.

		Raises:
			JsonRpcError: -32002 for an unknown tier.
		"""
		try:
			promoted = self.stack.promote(ids, tier, salience)
		except KeyError as exc:
			raise JsonRpcError(UNKNOWN_TIER, str(exc))

		if promoted:
			self._emit("promoted", owner=owner, ids=list(ids), tier=tier)
		return {"promoted": promoted}


	def forget(
		self,
		ids   : list[str]      = None,
		query : str            = None,
		tier  : str            = None,
		where : dict[str, Any] = None,
		owner : str            = None,
		key_prefix: str        = None,
		**ignored : Any,
	) -> dict[str, Any]:
		"""Handle 'memory/forget' -- delete records.

		Args:
			ids (list[str], optional): Delete these exactly.
			query (str, optional): Delete whatever this recalls.
			tier (str, optional): Restrict to one tier.
			where (dict, optional): Metadata filter.
			owner (str, optional): Scope.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: {'forgotten': count}.

		Raises:
			JsonRpcError: -32602 when no selector is given at all. Emptying a store
				must take more than an empty request.
		"""
		if not any([ids, query, tier, where, key_prefix]):
			raise JsonRpcError(INVALID_PARAMS, "Refusing to forget everything: pass ids, query, tier, where or key_prefix")

		if key_prefix is not None and not self.supports("keys"):
			raise JsonRpcError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'keys' capability")

		forgotten = self.stack.forget(ids=ids, query=query, tier=tier, where=where,
		                              agent=owner, key_prefix=key_prefix)
		if forgotten:
			self._emit("forgotten", owner=owner, count=forgotten, ids=list(ids) if ids else None)
		return {"forgotten": forgotten}


	def session_list(self, owner: str = None, **ignored: Any) -> dict[str, Any]:
		"""Handle 'memory/session/list' -- which conversations exist.

		Args:
			owner (str, optional): Scope the listing.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: {'sessions': [...]}, ascending by opened_at.
		"""
		return {"sessions": self.stack.sessions(agent=owner)}


	def session_close(self, session: str, owner: str = None, **ignored: Any) -> dict[str, Any]:
		"""Handle 'memory/session/close' -- end a conversation.

		Closing percolates the conversation down the whole stack rather than one tier:
		the transcript leaves working memory immediately, and every tier below then
		applies its own ordinary rules to what arrives.

		Args:
			session (str): The conversation to close.
			owner (str, optional): Scope.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: The consolidate report plus 'closed' and 'flushed'.

		Raises:
			JsonRpcError: -32602 if 'session' is missing or empty.
		"""
		if not session:
			raise JsonRpcError(INVALID_PARAMS, "'session' is required")

		closed = self.stack.close_session(session, agent=owner)

		# One coalesced event for the reorganisation, one for the closure —
		# never one per record (spec §4.12).
		self._emit(
			"consolidated",
			owner      = owner,
			moved      = closed.get("moved"     , None),
			promoted   = closed.get("promoted"  , None),
			dropped    = closed.get("dropped"   , None),
			summarized = closed.get("summarized", None),
		)
		self._emit("session_closed", owner=owner, session=session)
		return closed


	def fetch(self, key: str, owner: str = None, embeddings: bool = False, **ignored: Any) -> dict[str, Any]:
		"""Handle 'memory/fetch' -- read the record at an address.

		Args:
			key (str): The address to read.
			owner (str, optional): Whose key. Keys are unique per owner.
			embeddings (bool, optional): Include the stored vector.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: {'record': ...} or {'record': None} when nothing is at that key.
			An empty address is not an error -- it is the ordinary way to ask
			whether a fact is known yet.

		Raises:
			JsonRpcError: -32602 if 'key' is missing.
		"""
		if not key:
			raise JsonRpcError(INVALID_PARAMS, "'key' is required")

		record = self.stack.by_key(key, agent=owner)
		return {"record": record.to_dict(embedding=embeddings) if record else None}


	def _embedding_profile(self) -> dict[str, Any]:
		"""What vectors this server will accept.

		A client must know the dimensionality before sending a vector, because
		vectors of different widths -- or from different models -- cannot be
		compared, and a store that mixes them ranks nonsense with confidence.

		Returns:
			dict: 'dimensions' (None until the first vector fixes it), 'metric',
			and 'model' when the server has an opinion about which space it is in.
		"""
		return {
			"dimensions" : self.dimensions,
			"metric"     : "cosine",
			"model"      : self.embedding_model,
		}


	def _check_dimensions(self, vector: list[float]) -> None:
		"""Reject a vector that cannot be compared with what is already stored.

		The first vector a store receives fixes its width. Everything after must
		match, because cosine between vectors of different lengths is not a worse
		answer -- it is not an answer.

		Args:
			vector (list[float]): The caller's embedding.

		Raises:
			JsonRpcError: -32602 if it is not a list of numbers, -32008 if its
				width disagrees with the store's.
		"""
		if not isinstance(vector, list) or not all(isinstance(v, (int, float)) for v in vector):
			raise JsonRpcError(INVALID_PARAMS, "'embedding' must be an array of numbers")

		if self.dimensions is None:
			self.dimensions = len(vector)
			return

		if len(vector) != self.dimensions:
			raise JsonRpcError(
				EMBEDDING_MISMATCH,
				f"This store holds {self.dimensions}-dimensional vectors, got {len(vector)}",
				{"expected": self.dimensions, "got": len(vector), "model": self.embedding_model},
			)


	def consolidate(self, **ignored: Any) -> dict[str, Any]:
		"""Handle 'memory/consolidate' -- reorganise the store.

		Promotion runs before spilling, so a record that keeps proving useful is not
		displaced by sheer volume of new material arriving above it.

		Args:
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: moved, dropped, summarized, promoted and per-tier counts.
		"""
		report = self.stack.consolidate()
		self._emit(
			"consolidated",
			moved      = report.get("moved"     , None),
			promoted   = report.get("promoted"  , None),
			dropped    = report.get("dropped"   , None),
			summarized = report.get("summarized", None),
		)
		return report


	def events(self, cursor: str = None, limit: int = 256, kinds: list[str] = None,
	           owner: str = None, **ignored: Any) -> dict[str, Any]:
		"""Handle 'memory/events' -- read what changed since a cursor.

		Args:
			cursor (str, optional): Where to read from. Absent means now: no
				events, just the current cursor to poll from next.
			limit (int, optional): Maximum events to return.
			kinds (list[str], optional): Keep only these event kinds.
			owner (str, optional): Scope. A scoped caller sees its own events and
				unscoped ones, mirroring recall visibility (spec §4.12).
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: 'events' (oldest first) and 'cursor', plus 'more' and 'reset'
			when true.
		"""
		return self.events_log.read(cursor=cursor, limit=limit, kinds=kinds, agent=owner)


	def events_subscribe(self, kinds: list[str] = None, owner: str = None, **ignored: Any) -> dict[str, Any]:
		"""Handle 'memory/events/subscribe' -- start pushing events.

		Args:
			kinds (list[str], optional): Deliver only these event kinds.
			owner (str, optional): Scope the delivery, as for memory/events.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: {'subscribed': True}.

		Raises:
			JsonRpcError: -32003 when push is unavailable -- disabled at setup,
				or a binding that cannot carry a notification back, which is
				every HTTP connection (spec §4.13).
		"""
		if not self._push_available():
			raise JsonRpcError(
				CAPABILITY_NOT_SUPPORTED,
				"Push is not available on this connection; poll memory/events instead",
				{"push": False},
			)

		self._subscription = {"kinds": list(kinds) if kinds else None, "owner": owner}
		return {"subscribed": True}


	def events_unsubscribe(self, **ignored: Any) -> dict[str, Any]:
		"""Handle 'memory/events/unsubscribe' -- stop pushing events.

		Unsubscribing without a subscription is a no-op, not an error.

		Args:
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: {'subscribed': False}.
		"""
		self._subscription = None
		return {"subscribed": False}


	def take_notifications(self) -> list[dict[str, Any]]:
		"""Drain the notifications queued for the connected client.

		The stdio binding calls this after every handled message and writes each
		one as its own line -- which is what keeps a notification a whole
		message that can interleave *between* responses but never inside one.

		Returns:
			list[dict]: Pending 'memory/event' notification objects, oldest
			first.
		"""
		taken        = self._outbox
		self._outbox = []
		return taken


	def _push_available(self) -> bool:
		"""Whether memory/events/subscribe can work on this connection.

		Three things must all hold: the capability is declared, the operator did
		not disable push at setup, and the binding can physically carry a
		notification back to the client. serve_a2m_stdio and connect_local set
		`push_transport`; HTTP never does.

		Returns:
			bool: True when a subscription would deliver.
		"""
		return self.supports("events") and self.push_events and self.push_transport


	def _emit(self, kind: str, owner: str = None, **fields: Any) -> None:
		"""Record an event, and push it if anyone subscribed.

		Every event goes through the log first: push and poll are two views of
		one sequence, so a client that misses a notification can always catch up
		with a cursor.

		Args:
			kind (str): The event kind.
			owner (str, optional): The scope the event belongs to.
			**fields: Event payload. None values are dropped.
		"""
		if not self.supports("events"):
			return

		event        = self.events_log.append(kind, owner=owner, **fields)
		subscription = self._subscription

		if subscription is None or not self._push_available():
			return
		if subscription["kinds"] and kind not in subscription["kinds"]:
			return
		if owner is not None and subscription["owner"] is not None and owner != subscription["owner"]:
			return

		notification = make_notification(A2M_EVENT_NOTIFICATION, event)
		if self._notify_sink is not None:
			self._notify_sink(notification)
		else:
			self._outbox.append(notification)


class MemoryClient:
	"""The agent-facing half of A2M. Every method is one JSON-RPC call.

	A client may be *scoped* to an agent. Scoping stamps every write with that
	agent as owner and passes it along on every read, which is what lets one store
	serve a whole team: private tiers stay private, shared tiers stay pooled, and
	no agent has to be trusted to filter its own reads.

	Example:
		memory = connect_local(MemoryStack())
		memory.remember("marco lives in bologna")
		memory.recall(query="where does marco live")

		alice = memory.for_agent("alice")     # same store, scoped view
		bob   = memory.for_agent("bob")
	"""

	def __init__(self, client: Client, agent: str = None) -> None:
		"""Wrap a JSON-RPC client as an A2M client.

		Args:
			client (Client): A JSON-RPC client over any transport.
			agent (str, optional): The agent this view belongs to. Unscoped when
				omitted, which sees everything -- the single-agent case.
		"""
		self.client   = client
		self.agent    = agent
		self._profile : dict[str, Any] = None


	def for_agent(self, agent: str) -> "MemoryClient":
		"""Another view of the same memory, scoped to a different agent.

		The transport is shared, so this costs nothing and stays consistent.

		Args:
			agent (str): The agent to scope to.

		Returns:
			MemoryClient: A scoped view over the same connection.
		"""
		return MemoryClient(self.client, agent=agent)


	def _scoped(self, params: dict[str, Any]) -> dict[str, Any]:
		"""Attach this client's agent to a parameter object, if it has one.

		Args:
			params (dict): The call parameters.

		Returns:
			dict: The same object, with 'owner' set when scoped.
		"""
		if self.agent is not None:
			params["owner"] = self.agent
		return params


	def describe(self, refresh: bool = False) -> dict[str, Any]:
		"""Discover the server: version, capabilities, tier layout.

		Cached after the first call, since a server's shape does not change.

		Args:
			refresh (bool, optional): Re-fetch instead of using the cache. Needed when
				reading counts, which do change.

		Returns:
			dict: The describe result.
		"""
		if self._profile is None or refresh:
			self._profile = self.client.call("memory/describe", self._scoped({"protocol": A2M_VERSION}))
		return self._profile


	def capabilities(self) -> list[str]:
		"""What the server declares it implements.

		Returns:
			list[str]: Capability names, always including 'core'.
		"""
		return list(self.describe().get("capabilities", ["core"]))


	def supports(self, capability: str) -> bool:
		"""Whether the server declares a capability.

		Check this before calling anything outside 'core' -- spec 2 requires clients
		not to call into undeclared capabilities.

		Args:
			capability (str): tiers, salience, scopes, sessions, embeddings, keys
				or external.

		Returns:
			bool: True if declared.
		"""
		return capability in self.capabilities()


	def tiers(self) -> list[str]:
		"""The tier names, in stack order.

		Returns:
			list[str]: Tier names, or an empty list for a store without tiers.
		"""
		profile = self.describe()
		if "order" in profile:
			return list(profile["order"])
		return [tier["name"] for tier in profile.get("tiers", [])]


	def working(self) -> str:
		"""The tier holding the live transcript.

		Found by kind where the server reports it, so a client never has to hardcode
		the name 'working'.

		Returns:
			str: The transcript tier's name.
		"""
		profile = self.describe()
		if profile.get("working", None):
			return profile["working"]

		# A server that reports tier kinds lets a client find the transcript tier
		# without hardcoding a name (spec §4.7).
		for tier in profile.get("tiers", []):
			if tier.get("kind", None) == "working":
				return tier["name"]

		tiers = self.tiers()
		return tiers[0] if tiers else "working"


	def of_kind(self, kind: str) -> list[str]:
		"""Tiers serving a given purpose.

		Args:
			kind (str): working, episodic, semantic or procedural.

		Returns:
			list[str]: Matching tier names.
		"""
		profile = self.describe()
		if "kinds" in profile:
			return list(profile["kinds"].get(kind, []))
		return [t["name"] for t in profile.get("tiers", []) if t.get("kind", None) == kind]


	def fetch(self, key: str, embeddings: bool = False) -> dict[str, Any] | None:
		"""Read the record at an address.

		Args:
			key (str): The address, e.g. "user/city".
			embeddings (bool, optional): Include the stored vector.

		Returns:
			dict | None: The record, or None if nothing is at that key.
		"""
		params = self._scoped({"key": key, "embeddings": embeddings})
		return self.client.call("memory/fetch", params).get("record", None)


	def remember(self, content: str = None, records: list[dict[str, Any]] = None, **fields) -> list[str]:
		"""Write one record, or a batch.

		Args:
			content (str, optional): The text, for the single-record form.
			records (list[dict], optional): A batch of partial records, each needing
				'content'. Takes precedence over 'content'.
			**fields: Extra fields for the single-record form -- tier, role, salience,
				group, metadata, session, or an explicit id for an idempotent write.

		Returns:
			list[str]: The ids written, in order.

		Example:
			memory.remember("marco lives in bologna", role="user")
			memory.remember(records=[{"content": "a"}, {"content": "b", "tier": "semantic"}])
		"""
		if records is None:
			records = [dict(fields, content=content)]

		result = self.client.call("memory/remember", self._scoped({"records": records}))
		return result.get("ids", [])


	def recall(
		self,
		query     : str            = None,
		tier      : str            = None,
		limit     : int            = 8,
		where     : dict[str, Any] = None,
		min_score : float          = 0.0,
		embedding : list[float]    = None,
		key_prefix: str            = None,
		embeddings: bool           = False,
	) -> list[dict[str, Any]]:
		"""Search by relevance.

		Args:
			query (str, optional): What to rank against.
			tier (str, optional): Restrict to one tier.
			limit (int, optional): Maximum records.
			where (dict, optional): Metadata filter.
			min_score (float, optional): Server-relative threshold. Leave it alone
				unless calibrated against that specific server -- scores are not
				comparable across stores (spec 5.3).

		Returns:
			list[dict]: Records in wire form, descending by score.
		"""
		params = {"limit": limit, "min_score": min_score}
		if embedding is not None:
			params["embedding"] = list(embedding)
		if key_prefix is not None:
			params["key_prefix"] = key_prefix
		if embeddings:
			params["embeddings"] = True
		if query is not None:
			params["query"] = query
		if tier is not None:
			params["tier"] = tier
		if where is not None:
			params["where"] = where

		return self.client.call("memory/recall", self._scoped(params)).get("records", [])


	def timeline(self, tier: str = None, limit: int = 0, where: dict[str, Any] = None,
	             key_prefix: str = None, embeddings: bool = False) -> list[dict[str, Any]]:
		"""Read records in creation order.

		Args:
			tier (str, optional): Restrict to one tier.
			limit (int, optional): Keep the most recent N, still ascending.
			where (dict, optional): Metadata filter -- pass {'session': ...} to replay
				a single conversation.

		Returns:
			list[dict]: Records ascending by created_at.
		"""
		params = {"limit": limit}
		if tier is not None:
			params["tier"] = tier
		if where is not None:
			params["where"] = where
		if key_prefix is not None:
			params["key_prefix"] = key_prefix
		if embeddings:
			params["embeddings"] = True

		return self.client.call("memory/timeline", self._scoped(params)).get("records", [])


	def reinforce(self, ids: list[str], amount: float = 0.5) -> int:
		"""Raise the salience of records that proved useful.

		Args:
			ids (list[str]): Records to reinforce.
			amount (float, optional): How much to add.

		Returns:
			int: How many were reinforced.
		"""
		params = {"ids": list(ids), "amount": amount}
		return self.client.call("memory/reinforce", self._scoped(params)).get("reinforced", 0)


	def promote(self, ids: list[str], tier: str, salience: float = 0.5) -> int:
		"""Move records to a more durable tier.

		Args:
			ids (list[str]): Records to move.
			tier (str): Destination.
			salience (float, optional): Added on arrival.

		Returns:
			int: How many moved.
		"""
		params = {"ids": list(ids), "tier": tier, "salience": salience}
		return self.client.call("memory/promote", self._scoped(params)).get("promoted", 0)


	def forget(self, ids: list[str] = None, query: str = None, tier: str = None,
	           where: dict[str, Any] = None, key_prefix: str = None) -> int:
		"""Delete records.

		Args:
			ids (list[str], optional): Delete these exactly.
			query (str, optional): Delete whatever this recalls.
			tier (str, optional): Restrict to one tier.
			where (dict, optional): Metadata filter.

		Returns:
			int: How many were removed.
		"""
		params = {}
		if key_prefix is not None:
			params["key_prefix"] = key_prefix
		if ids is not None:
			params["ids"] = list(ids)
		if query is not None:
			params["query"] = query
		if tier is not None:
			params["tier"] = tier
		if where is not None:
			params["where"] = where

		return self.client.call("memory/forget", self._scoped(params)).get("forgotten", 0)


	def sessions(self) -> list[dict[str, Any]]:
		"""Which conversations exist.

		Returns:
			list[dict]: One entry per session with its record count and tier spread.
		"""
		return self.client.call("memory/session/list", self._scoped({})).get("sessions", [])


	def close_session(self, session: str) -> dict[str, Any]:
		"""End a conversation and let it percolate down the stack.

		Args:
			session (str): The conversation to close.

		Returns:
			dict: The consolidate report plus 'closed' and 'flushed'.
		"""
		return self.client.call("memory/session/close", self._scoped({"session": session}))


	def consolidate(self) -> dict[str, Any]:
		"""Ask the store to reorganise itself.

		Returns:
			dict: moved, dropped, summarized, promoted and per-tier counts.
		"""
		return self.client.call("memory/consolidate")


	def events(self, cursor: str = None, limit: int = 256, kinds: list[str] = None) -> dict[str, Any]:
		"""Read what changed since a cursor.

		Call once with no cursor to get a starting position, then poll with the
		cursor each reply returns. If a reply carries 'reset', events were
		missed -- re-read whatever state was being tracked.

		Args:
			cursor (str, optional): Where to read from. Opaque; never parse it.
			limit (int, optional): Maximum events per poll.
			kinds (list[str], optional): Keep only these event kinds.

		Returns:
			dict: 'events' (oldest first) and 'cursor', plus 'more' and 'reset'
			when true.

		Example:
			position = memory.events()["cursor"]          # subscribe from now
			...
			reply    = memory.events(cursor=position)
			position = reply["cursor"]
		"""
		params = {"limit": limit}
		if cursor is not None:
			params["cursor"] = cursor
		if kinds is not None:
			params["kinds"] = list(kinds)

		return self.client.call("memory/events", self._scoped(params))


	def subscribe_events(self, kinds: list[str] = None) -> bool:
		"""Ask the server to push events as they happen.

		Only works where the transport can carry a notification back -- stdio
		and in-process, never HTTP. Check describe()['events']['push'] first,
		or be ready for -32003. Delivered events arrive via take_events().

		Args:
			kinds (list[str], optional): Deliver only these event kinds.

		Returns:
			bool: True when subscribed.
		"""
		params = {"kinds": list(kinds)} if kinds else {}
		return bool(self.client.call("memory/events/subscribe", self._scoped(params)).get("subscribed", False))


	def unsubscribe_events(self) -> bool:
		"""Stop the server pushing events.

		Returns:
			bool: False, the new subscription state.
		"""
		return bool(self.client.call("memory/events/unsubscribe", self._scoped({})).get("subscribed", False))


	def take_events(self) -> list[dict[str, Any]]:
		"""The events pushed since the last call, oldest first.

		Push delivery is passive on the client side: notifications accumulate on
		the transport as responses are read, and this drains them. A client that
		wants to block for events should poll memory/events instead.

		Returns:
			list[dict]: Event objects from 'memory/event' notifications.
		"""
		events = []
		for message in self.client.take_notifications():
			if isinstance(message, dict) and message.get("method", None) == "memory/event":
				events.append(message.get("params", {}))
		return events


	def close(self) -> None:
		"""Close the underlying transport, terminating a subprocess or connection.
		"""
		self.client.close()


def connect_local(stack: MemoryStack = None, name: str = "agent-memory", agent: str = None, capabilities: list[str] = None) -> MemoryClient:
	"""A2M over an in-process transport. Same wire format, no subprocess.

	Payloads still round-trip through JSON, so a local server and a remote one
	cannot quietly disagree about what is representable.

	Args:
		stack (MemoryStack, optional): The store to serve.
		name (str, optional): Server name.
		agent (str, optional): Scope the returned client.
		capabilities (list[str], optional): Narrow what the server admits to
			implementing -- this is how a core-only store is simulated.

	Returns:
		MemoryClient: Connected client.

	Example:
		memory = connect_local(MemoryStack())
		agent  = Agent(model, memory=memory)
	"""
	server    = MemoryServer(stack=stack, name=name, capabilities=capabilities)
	transport = LocalTransport(server.dispatcher)

	# In-process, a notification is delivered by callback -- there is no byte
	# stream to carry it (spec §8.1). It still round-trips through JSON so a
	# local subscriber sees exactly what a stdio one would.
	server.push_transport = True
	server._notify_sink   = lambda notification: transport.notifications.append(
		json.loads(json.dumps(notification))
	)

	return MemoryClient(Client(transport), agent=agent)


def connect_stdio(command: list[str], env: dict[str, str] = None, cwd: str = None, on_stderr: Callable = None, agent: str = None) -> MemoryClient:
	"""A2M against a memory server running as a child process.

	Args:
		command (list[str]): The command to launch, e.g.
			[sys.executable, "-m", "implementations.store_sqlite", "memory.db"].
		env (dict, optional): Child environment.
		cwd (str, optional): Child working directory.
		on_stderr (Callable, optional): Called per stderr line. The child must
			keep stdout free of anything but JSON-RPC, so logging arrives here.
		agent (str, optional): Scope the returned client.

	Returns:
		MemoryClient: Connected client.
	"""
	return MemoryClient(Client(StdioTransport(command, env=env, cwd=cwd, on_stderr=on_stderr)), agent=agent)


def connect_http(url: str, headers: dict[str, str] = None, agent: str = None, timeout: float = 30.0) -> MemoryClient:
	"""A2M against a memory server over HTTP.

	Note spec 6: a network server must derive the scope from the authenticated
	principal. The 'agent' passed here is what this client *claims*, and a correct
	server will ignore it in favour of whatever the credentials say.

	Args:
		url (str): The endpoint.
		headers (dict, optional): Extra headers, typically Authorization.
		agent (str, optional): Claimed scope.
		timeout (float, optional): Request timeout in seconds.

	Returns:
		MemoryClient: Connected client.
	"""
	headers = dict(headers) if headers else {}
	headers.setdefault(A2M_VERSION_HEADER, A2M_VERSION)

	return MemoryClient(Client(HttpTransport(url, headers=headers, timeout=timeout)), agent=agent)


def serve_a2m_http(
	server,
	host            : str       = "127.0.0.1",
	port            : int       = 8778,
	path            : str       = "/",
	allowed_origins : list[str] = None,
):
	"""An HTTP server carrying A2M's binding rules, per spec §8.3.

	Three things separate this from a bare JSON-RPC endpoint: Origin validation,
	the protocol version header, and the well-known profile. All three exist so
	that something in front of the server -- a browser, a gateway, a directory --
	can act correctly without understanding A2M itself.

	Args:
		server: Anything exposing 'dispatcher' and 'describe'. MemoryServer and
			MemoryRouter both do.
		host (str, optional): Bind address. Loopback by default.
		port (int, optional): Bind port.
		path (str, optional): The RPC endpoint path.
		allowed_origins (list[str], optional): Browser origins permitted to call
			this server. The default admits none, which is what a local server
			wants; see jsonrpc.serve_http.

	Returns:
		http.server.HTTPServer: Not yet started -- call serve_forever().

	Example:
		server = MemoryServer()
		serve_a2m_http(server, port=8778).serve_forever()
	"""
	# HTTP has no server-to-client channel, so push must not be advertised on
	# it -- and the describe snapshot below is what the well-known profile
	# serves, so this must be settled first (spec §4.13).
	server.push_transport = False

	def check_version(headers: Any) -> tuple[int, dict[str, Any]] | None:
		"""Refuse a request that declares a version this server does not speak.

		Absent is not a mismatch: the header is a convenience for intermediaries,
		and memory/describe remains the negotiation that matters.
		"""
		declared = headers.get(A2M_VERSION_HEADER, None)
		if declared is None or declared == A2M_VERSION:
			return None

		return (400, make_error_response(
			None,
			PROTOCOL_NOT_SUPPORTED,
			f"This server speaks {A2M_VERSION}, not {declared}",
			{"supported": [A2M_VERSION]},
		))

	return serve_http(
		server.dispatcher,
		host            = host,
		port            = port,
		path            = path,
		allowed_origins = allowed_origins,
		on_headers      = check_version,
		well_known      = {A2M_WELL_KNOWN: server.describe()},
	)


def serve_a2m_stdio(server) -> None:
	"""Run an A2M server on stdin/stdout, with push delivery wired up.

	Prefer this over bare jsonrpc.serve_stdio for anything declaring `events`:
	the bare loop never drains the server's notification outbox, so a
	subscription would accept and then deliver nothing.

	Args:
		server: Anything exposing 'dispatcher' and 'take_notifications'.
			MemoryServer and MemoryRouter both do.
	"""
	server.push_transport = True
	serve_stdio(server.dispatcher, drain=server.take_notifications)


def serve(stack: MemoryStack = None, name: str = "agent-memory", capabilities: list[str] = None) -> None:
	"""Run this process as an A2M server on stdin/stdout.

	Args:
		stack (MemoryStack, optional): The store to serve.
		name (str, optional): Server name.
		capabilities (list[str], optional): What to declare.
	"""
	serve_a2m_stdio(MemoryServer(stack=stack, name=name, capabilities=capabilities))


def serve_over_http(stack: MemoryStack = None, name: str = "agent-memory", host: str = "127.0.0.1", port: int = 8778) -> None:
	"""Run this process as an A2M server over HTTP, blocking forever.

	Args:
		stack (MemoryStack, optional): The store to serve.
		name (str, optional): Server name.
		host (str, optional): Bind address.
		port (int, optional): Bind port.
	"""
	server = serve_a2m_http(MemoryServer(stack=stack, name=name), host=host, port=port)
	print(f"A2M {A2M_VERSION} on http://{host}:{port}/", file=sys.stderr)
	server.serve_forever()
