"""An MCP server whose tools are an A2M server. Standard library only.

	python -m implementations.bridge_mcp --stdio python -m a2m
	python -m implementations.bridge_mcp --stdio python -m implementations.store_sqlite memory.db
	python -m implementations.bridge_mcp --http  http://127.0.0.1:8778/

Speak MCP on stdin/stdout, and every tool call becomes an A2M call against the
server named on the command line. Point Claude Desktop, Claude Code, Cursor or
any other MCP client at this command and it can remember and recall through any
conformant A2M store — which makes every MCP client an A2M client, at the cost
of one process in between.

The bridge is also the two protocols' relationship made runnable. A2M
deliberately shares MCP's bindings — JSON-RPC 2.0, newline-delimited stdio, no
batches (DECISION 021) — so bridging is a method table, not a translation
layer: both sides of this file read and write the same framing.

What the flattening costs is the argument for A2M existing separately (DECISION
027). A tool result is text, so a record's fields come back as JSON *prose* the
model must re-parse; capability negotiation collapses into which tools are
listed; and `timeline` versus `recall` — replay versus search, the distinction
the tier model is built on — survives only as a sentence in a tool description
that nothing enforces. The bridge is the right adapter for an agent that
already speaks MCP, and the wrong place to build a memory stack.

Tools are derived from the A2M server's `memory/describe`: a capability the
store did not declare is a tool the MCP client never sees, which is A2M's
negotiation surfacing through MCP's.

An MCP *client* for A2M would be the reverse bridge and is deliberately absent:
an A2M server behind `tools/call` would be a store reachable only through the
flattening above.
"""


import json
import sys


from   typing       import Any


from   a2m          import MemoryClient, connect_http, connect_stdio
from   a2m.jsonrpc  import JsonRpcError


MCP_PROTOCOL_VERSION = "2025-06-18"
BRIDGE_VERSION       = "0.1.0"

PARSE_ERROR      = -32700
INVALID_REQUEST  = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS   = -32602
INTERNAL_ERROR   = -32603


def build_tools(memory: MemoryClient) -> list[dict[str, Any]]:
	"""The MCP tool list this A2M server earns.

	Every tool is gated on a declared capability, so the negotiation a2m does
	through `memory/describe` surfaces through MCP's `tools/list`: a store that
	cannot address records by key simply has no memory_fetch tool.

	Args:
		memory (MemoryClient): A connected, negotiated A2M client.

	Returns:
		list[dict]: MCP tool definitions, each with a JSON Schema for input.
	"""
	profile = memory.describe()
	tiers   = [t.get("name") for t in profile.get("tiers", []) if t.get("name", None)]
	tools   = []

	remember_properties = {
		"content": {"type": "string", "description": "The text to remember. This is what gets indexed and recalled."},
		"role"   : {"type": "string", "description": "Who produced it: user, assistant, tool or system."},
		"metadata": {"type": "object", "description": "Arbitrary JSON to store with the record, filterable later."},
	}
	if memory.supports("tiers"):
		remember_properties["tier"] = {
			"type": "string", "enum": tiers or None,
			"description": "Where to store it. Facts belong in a searchable tier, not the conversation transcript.",
		}
		if not tiers:
			del remember_properties["tier"]["enum"]
	if memory.supports("keys"):
		remember_properties["key"] = {
			"type": "string",
			"description": "An address like 'user/city'. Writing to an occupied key REPLACES the fact there, correcting it.",
		}
	if memory.supports("sessions"):
		remember_properties["session"] = {"type": "string", "description": "The conversation this belongs to."}

	tools.append({
		"name"        : "memory_remember",
		"description" : "Store something in the shared agent memory so it can be recalled later, "
		                "by this agent or another one.",
		"inputSchema" : {"type": "object", "properties": remember_properties, "required": ["content"]},
	})

	recall_properties = {
		"query": {"type": "string", "description": "What you are trying to remember, in natural language."},
		"limit": {"type": "integer", "description": "Maximum records to return.", "default": 5},
	}
	if memory.supports("tiers"):
		recall_properties["tier"] = {"type": "string", "description": "Restrict the search to one tier."}

	tools.append({
		"name"        : "memory_recall",
		"description" : "Search the shared agent memory by relevance. Returns records ranked best-first; "
		                "the scores order this list only and are not comparable across calls.",
		"inputSchema" : {"type": "object", "properties": recall_properties, "required": ["query"]},
	})

	timeline_properties = {
		"limit": {"type": "integer", "description": "How many of the most recent records to return.", "default": 20},
	}
	if memory.supports("sessions"):
		timeline_properties["session"] = {"type": "string", "description": "Replay one conversation only."}

	tools.append({
		"name"        : "memory_timeline",
		"description" : "Replay memory in the order it was written, oldest first. Use this to rebuild a "
		                "conversation; use memory_recall to search. They are different reads and neither "
		                "substitutes for the other.",
		"inputSchema" : {"type": "object", "properties": timeline_properties},
	})

	forget_properties = {
		"ids"  : {"type": "array", "items": {"type": "string"}, "description": "Exact record ids to delete."},
		"query": {"type": "string", "description": "Delete whatever this query recalls."},
	}
	if memory.supports("keys"):
		forget_properties["key_prefix"] = {
			"type": "string",
			"description": "Delete every record whose key sits at or under this prefix.",
		}

	tools.append({
		"name"        : "memory_forget",
		"description" : "Delete records from the shared agent memory. At least one selector is required; "
		                "there is no way to delete everything at once, deliberately.",
		"inputSchema" : {"type": "object", "properties": forget_properties},
	})

	if memory.supports("keys"):
		tools.append({
			"name"        : "memory_fetch",
			"description" : "Read the one record at an address like 'user/city'. Returns null when the "
			                "fact is not known yet, which is an ordinary answer, not an error.",
			"inputSchema" : {"type": "object",
			                 "properties": {"key": {"type": "string", "description": "The address to read."}},
			                 "required": ["key"]},
		})

	if memory.supports("events"):
		tools.append({
			"name"        : "memory_events",
			"description" : "What changed in the memory since a cursor. Call once without a cursor to get "
			                "a starting position, then pass each reply's cursor back to see what happened "
			                "in between. The cursor is opaque: store it, never interpret it.",
			"inputSchema" : {"type": "object", "properties": {
				"cursor": {"type": "string", "description": "The cursor from the previous call, if any."},
				"limit" : {"type": "integer", "description": "Maximum events to return.", "default": 50},
			}},
		})

	return tools


def call_tool(memory: MemoryClient, name: str, arguments: dict[str, Any]) -> Any:
	"""Run one MCP tool call as an A2M call.

	Args:
		memory (MemoryClient): The A2M server behind the bridge.
		name (str): The tool name, as listed by build_tools.
		arguments (dict): The tool arguments.

	Returns:
		Any: A JSON-serialisable result for the tool's text content.

	Raises:
		JsonRpcError: Whatever the A2M server raised; the caller renders it as
			a tool execution error rather than a protocol one.
		KeyError: If the tool name is unknown.
	"""
	if name == "memory_remember":
		fields = {k: v for k, v in arguments.items() if k != "content"}
		ids    = memory.remember(arguments["content"], **fields)
		return {"ids": ids}

	if name == "memory_recall":
		records = memory.recall(
			query = arguments["query"],
			limit = int(arguments.get("limit", 5)),
			tier  = arguments.get("tier", None),
		)
		return {"records": records}

	if name == "memory_timeline":
		where = {"session": arguments["session"]} if arguments.get("session", None) else None
		return {"records": memory.timeline(limit=int(arguments.get("limit", 20)), where=where)}

	if name == "memory_forget":
		selectors = {k: v for k, v in arguments.items() if k in ("ids", "query", "key_prefix") and v}
		return {"forgotten": memory.forget(**selectors)}

	if name == "memory_fetch":
		return {"record": memory.fetch(arguments["key"])}

	if name == "memory_events":
		return memory.events(
			cursor = arguments.get("cursor", None),
			limit  = int(arguments.get("limit", 50)),
		)

	raise KeyError(name)


class Bridge:
	"""One MCP session in front of one A2M server."""

	def __init__(self, memory: MemoryClient, name: str = "a2m-memory") -> None:
		"""Wire the bridge to a connected A2M client.

		Args:
			memory (MemoryClient): The store to expose.
			name (str, optional): The server name reported to the MCP client.
		"""
		self.memory = memory
		self.name   = name
		self.tools  = build_tools(memory)


	def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
		"""Answer one MCP message.

		Args:
			message (dict): A parsed JSON-RPC request or notification.

		Returns:
			dict | None: The response object, or None for a notification.
		"""
		method          = message.get("method", None)
		params          = message.get("params", None) or {}
		is_notification = "id" not in message
		id              = message.get("id", None)

		def result(payload: Any) -> dict[str, Any] | None:
			return None if is_notification else {"jsonrpc": "2.0", "id": id, "result": payload}

		def error(code: int, text: str) -> dict[str, Any] | None:
			return None if is_notification else {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": text}}

		if message.get("jsonrpc", None) != "2.0" or not isinstance(method, str):
			return error(INVALID_REQUEST, "Expected a JSON-RPC 2.0 request")

		if method == "initialize":
			return result({
				"protocolVersion" : params.get("protocolVersion", MCP_PROTOCOL_VERSION),
				"capabilities"    : {"tools": {}},
				"serverInfo"      : {"name": self.name, "version": BRIDGE_VERSION},
				"instructions"    : "A shared agent memory served over A2M. Use memory_recall before "
				                    "answering questions about prior context, and memory_remember for "
				                    "anything worth keeping beyond this conversation.",
			})

		if method in ("notifications/initialized", "notifications/cancelled"):
			return None

		if method == "ping":
			return result({})

		if method == "tools/list":
			return result({"tools": self.tools})

		if method == "tools/call":
			name      = params.get("name", None)
			arguments = params.get("arguments", None) or {}

			if not any(tool["name"] == name for tool in self.tools):
				return error(INVALID_PARAMS, f"Unknown tool '{name}'")

			# A failure inside a tool is a tool result with isError, not a
			# protocol error: the MCP client's model is entitled to read it
			# and try something else.
			try:
				answer = call_tool(self.memory, name, arguments)
				return result({"content": [{"type": "text", "text": json.dumps(answer, indent=2)}]})
			except JsonRpcError as exc:
				return result({"content": [{"type": "text", "text": f"[{exc.code}] {exc.message}"}],
				               "isError": True})
			except (KeyError, TypeError, ValueError) as exc:
				return result({"content": [{"type": "text", "text": f"Bad arguments: {exc}"}],
				               "isError": True})

		return error(METHOD_NOT_FOUND, f"Unknown method '{method}'")


	def serve(self, stdin=None, stdout=None) -> None:
		"""Speak MCP on stdin/stdout until the stream closes.

		The framing is identical to A2M's stdio binding — one JSON value per
		line, logs to stderr — which is no coincidence (DECISION 021).

		Args:
			stdin (TextIO, optional): Defaults to sys.stdin.
			stdout (TextIO, optional): Defaults to sys.stdout.
		"""
		stdin  = stdin  or sys.stdin
		stdout = stdout or sys.stdout

		for line in stdin:
			line = line.strip()
			if not line:
				continue

			try:
				message = json.loads(line)
			except json.JSONDecodeError as exc:
				response = {"jsonrpc": "2.0", "id": None,
				            "error": {"code": PARSE_ERROR, "message": f"Invalid JSON: {exc}"}}
			else:
				try:
					response = self.handle(message)
				except Exception as exc:   # a bridge must not die mid-session
					response = {"jsonrpc": "2.0", "id": message.get("id", None),
					            "error": {"code": INTERNAL_ERROR, "message": str(exc)}}

			if response is not None:
				stdout.write(json.dumps(response) + "\n")
				stdout.flush()


def main() -> int:
	"""Run the bridge from the command line.

		python -m implementations.bridge_mcp --stdio python -m a2m
		python -m implementations.bridge_mcp --http http://127.0.0.1:8778/

	Returns:
		int: Process exit code.
	"""
	argv = sys.argv[1:]

	if not argv or argv[0] not in ("--stdio", "--http") or len(argv) < 2:
		print(__doc__, file=sys.stderr)
		return 2

	if argv[0] == "--stdio":
		memory = connect_stdio(argv[1:], on_stderr=lambda line: print(f"[a2m] {line}", file=sys.stderr))
	else:
		memory = connect_http(argv[1])

	try:
		profile = memory.describe()
		print(f"MCP bridge over A2M server {profile.get('name')!r} "
		      f"({', '.join(profile.get('capabilities', []))})", file=sys.stderr)
		Bridge(memory, name=f"a2m:{profile.get('name', 'memory')}").serve()
	finally:
		memory.close()

	return 0


if __name__ == "__main__":
	sys.exit(main())
