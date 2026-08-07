"""The OpenAI Agents SDK against an A2M server: conversation history as a Session.

	pip install openai-agents

`OpenAIAgentsA2MSession` implements the SDK's `Session` protocol, so an agent's
conversation lives in an A2M store instead of the SDK's own SQLite file:

	from agents import Agent, Runner

	session = OpenAIAgentsA2MSession(client, "incident-77")
	await Runner.run(agent, "how do we roll back?", session=session)

Which is one line of difference at the call site and a different thing entirely
underneath: the transcript is now somewhere an Agno agent or an AutoGen agent can
read it. [examples/agent_interop.py](../../examples/agent_interop.py) is that
claim as a runnable script.

The mapping is unusually close, and one detail is worth naming because it is a
coincidence that pays:

- **`get_items(limit)` is `memory/timeline`, exactly.** The SDK asks for "the
  latest N items in chronological order". spec §4.4 requires a server given a
  `limit` to return "the *most recent* `limit` records, still in ascending
  order". Same sentence. Neither was written with the other in mind — they agree
  because replaying the tail of a conversation is the only thing that request can
  reasonably mean.
- **`get_items` never ranks.** It would be a one-word change to call `recall`
  and it would even look like it worked, until a reordered transcript reached a
  provider that rejects it.
- **A function call and its output share a `call_id`.** That is a group (spec
  §3.4) already named by the SDK, so this adapter does not have to invent one:
  a store evicting under pressure moves the call and its result together or not
  at all.
- **Items round-trip verbatim.** The SDK feeds these dicts straight back to the
  model, so the whole item is kept in metadata and rebuilt untouched. `content`
  gets a readable rendering of it, which costs nothing and leaves the turn
  indexable by everything else sharing the store.

Async by delegation, like the Agno and AutoGen adapters: A2M's transports are
blocking, and wrapping them in a thread pool would advertise concurrency this
adapter cannot deliver.

**This is the one adapter here that imports nothing.** `Session` is a
`runtime_checkable` `Protocol` rather than a base class, so conformance is
structural: having the four methods *is* being a session, and there is no import
path to break when the SDK moves one. That is a property of how the SDK drew its
memory interface, not of A2M — but it is the shape every adapter in this
directory wishes it had, and it is why this file has no `pip install` line in its
imports despite having one at the top. `examples/agent_interop.py` asserts the
`isinstance` rather than this module assuming it.
"""


from   typing import Any, Dict, List, Optional


# Roles the Responses API accepts on an input item. A2M's `role` is advisory and
# open (spec §3.1), so anything else a foreign writer used becomes "user" rather
# than being handed to a provider that will reject it.
_INPUT_ROLES = {"user", "assistant", "system", "developer"}


def _text_of(item: Dict[str, Any]) -> str:
	"""The indexable text of one Responses item, however it is carrying it.

	Args:
		item (dict): A `TResponseInputItem` — a message, a function call, or a
			function call's output.

	Returns:
		str: Its text. A tool call renders as `name(arguments)` so that the turn
		says what it did rather than sitting in the store as an empty string.
	"""
	kind = item.get("type")

	if kind == "function_call":
		return f"{item.get('name', 'tool')}({item.get('arguments', '')})"
	if kind == "function_call_output":
		return str(item.get("output", ""))

	content = item.get("content")
	if isinstance(content, str):
		return content

	parts = []
	for block in content or []:
		if isinstance(block, str):
			parts.append(block)
		elif isinstance(block, dict):
			parts.append(block.get("text") or block.get("transcript") or "")

	return "".join(parts)


def _role_of(item: Dict[str, Any]) -> str:
	"""A2M's advisory role for one Responses item.

	Args:
		item (dict): The item.

	Returns:
		str: user, assistant, system or tool.
	"""
	if item.get("type") in ("function_call", "function_call_output"):
		return "tool"

	return item.get("role") or "user"


class OpenAIAgentsA2MSession:
	"""An OpenAI Agents SDK session backed by an A2M server.

	Example:
		from implementations import client as a2m_client

		client  = a2m_client.connect_stdio(["python", "-m", "implementations.store_sqlite", "memory.db"])
		session = OpenAIAgentsA2MSession(client, "incident-77")

		await Runner.run(agent, "how do we roll back?", session=session)
	"""

	# Part of the Session protocol. This adapter has no settings of its own: what
	# would be configured here — capacity, eviction, what happens when the
	# conversation ends — belongs to the server and is described by it.
	session_settings = None

	def __init__(self, client, session_id: str, tier: str = None) -> None:
		"""Bind a session to one conversation in an A2M store.

		Args:
			client: A negotiated A2M client — implementations.client.A2MClient or
				a2m.MemoryClient, over any transport.
			session_id (str): The conversation id, as the SDK names it. Servers
				declaring `sessions` scope both the timeline and the eviction
				policy to it, which is what stops a busy conversation evicting a
				quiet one.
			tier (str, optional): Where turns live. The working tier by default,
				found by *kind* rather than by name so a store that calls it
				something else still works. That tier is replayed and never
				embedded, which is what a transcript wants (spec §4.4).
		"""
		self.client     = client
		self.session_id = session_id
		self.tier       = tier if tier is not None else self._working_tier()


	def _working_tier(self) -> Optional[str]:
		"""The tier holding the live transcript.

		Returns:
			str | None: Its name, or None on a server without tiers.
		"""
		if not self.client.supports("tiers"):
			return None

		for tier in self.client.describe().get("tiers", []):
			if tier.get("kind") == "working":
				return tier.get("name")

		return None


	def _filters(self) -> dict:
		"""How this conversation is selected from the store.

		Returns:
			dict: A `where` filter, empty on a server without sessions.
		"""
		return {"session": self.session_id} if self.client.supports("sessions") else {}


	def _replay(self, limit: int = 0) -> List[dict]:
		"""This conversation's records, oldest first.

		Args:
			limit (int, optional): At most this many, the most recent ones.

		Returns:
			list[dict]: Records ascending by created_at.
		"""
		params: Dict[str, Any] = {"tier": self.tier, "limit": limit}

		filters = self._filters()
		if filters:
			params["where"] = filters

		return self.client.timeline(**params)


	def _to_item(self, record: dict) -> Dict[str, Any]:
		"""Rebuild the Responses item a record was written from.

		Args:
			record (dict): A record as the server returned it.

		Returns:
			dict: The original item when this adapter wrote it, verbatim — the
			SDK hands these straight back to the model, so anything reconstructed
			approximately is a bug that surfaces as a provider error. A record
			another framework wrote comes back as a plain user message rather
			than an error: a shared store is one somebody else has been writing
			to, and refusing to read it would defeat the point.
		"""
		carried = (record.get("metadata") or {}).get("openai_item")
		if isinstance(carried, dict):
			return carried

		role = record.get("role")
		return {
			"role"    : role if role in _INPUT_ROLES else "user",
			"content" : record.get("content", ""),
		}


	def _to_record(self, item: Dict[str, Any]) -> dict:
		"""Turn a Responses item into an A2M record.

		Args:
			item (dict): What the SDK wants stored.

		Returns:
			dict: Fields for memory/remember.
		"""
		fields: Dict[str, Any] = {
			"role"     : _role_of(item),
			"metadata" : {"openai_item": item},
		}

		if self.tier is not None:
			fields["tier"] = self.tier
		if self.client.supports("sessions"):
			fields["session"] = self.session_id

		# spec §3.4 -- the SDK already correlates a call with its output by
		# `call_id`, so the group is named rather than invented. A store evicting
		# under pressure moves them together; an assistant turn separated from
		# the result answering it is a transcript most providers refuse.
		if item.get("call_id"):
			fields["group"] = str(item["call_id"])

		fields["content"] = _text_of(item)
		return fields


	async def get_items(self, limit: int = None) -> List[Dict[str, Any]]:
		"""The conversation, oldest first.

		Args:
			limit (int, optional): At most this many items, the most recent ones.
				Passed straight through: spec §4.4 requires exactly the SDK's
				semantics, so there is nothing to translate and nothing to
				re-sort locally.

		Returns:
			list[dict]: Responses items in the order they happened.
		"""
		return [self._to_item(record) for record in self._replay(limit or 0)]


	async def add_items(self, items: List[Dict[str, Any]]) -> None:
		"""Append items to the conversation.

		Args:
			items (list[dict]): Responses items, in order.
		"""
		for item in items:
			record = self._to_record(item)
			self.client.remember(record.pop("content"), **record)


	async def pop_item(self) -> Optional[Dict[str, Any]]:
		"""Remove and return the most recent item.

		The SDK uses this to undo a turn — typically to retry it. It deletes
		rather than percolating, because an item being taken back was never part
		of the conversation.

		Returns:
			dict | None: The item, or None when the conversation is empty.
		"""
		records = self._replay(1)
		if not records:
			return None

		last = records[-1]
		if last.get("id"):
			self.client.forget(ids=[last["id"]])

		return self._to_item(last)


	async def clear_session(self) -> None:
		"""Forget this conversation, and only this one.

		A delete rather than a `close_session`: the SDK's contract is "clear all
		items", and a caller who asked for that would be surprised to find the
		turns still recallable one tier down. Closing the session is the other
		thing — see `LangChainA2MChatMessageHistory.clear` in the LangChain adapter, and
		spec §4.11 for what percolation does with a conversation that ended
		rather than one that was retracted.
		"""
		filters = self._filters()

		if filters:
			self.client.forget(where=filters)
		elif self.tier is not None:
			# spec §4.5: a forget with no selector is an error, not an emptied
			# store. The tier is the only selector left on a server with no
			# sessions, and it is a real one.
			self.client.forget(tier=self.tier)
