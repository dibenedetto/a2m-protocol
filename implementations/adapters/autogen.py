"""AutoGen against an A2M server: agent memory over the Memory protocol.

	pip install autogen-core

`A2MMemory` implements `autogen_core.memory.Memory`, so an AutoGen agent's
memory — the thing `update_context` injects before each model call — is an A2M
store, shared with whatever else is connected to it:

	from autogen_agentchat.agents import AssistantAgent

	agent = AssistantAgent(name="ops", model_client=model,
	                       memory=[A2MMemory(client, namespace="ops")])

The mapping is close because the shapes agree: AutoGen's `add` is
`memory/remember`, its `query` is `memory/recall`, and `update_context` is a
query driven by the conversation's tail. The edges:

- **`update_context` queries; it does not replay.** The memories worth
  injecting are the ones relevant to what the conversation just said, so the
  last message is the query and `memory/recall` ranks. The conversation itself
  already lives in the model context — replaying working memory into it would
  double the transcript.
- **`clear` is scoped, like every other adapter here.** The store is shared;
  this namespace is not (spec §4.5).
- **Async by delegation.** A2M's transports are blocking, and wrapping them in
  a thread pool would advertise concurrency this adapter cannot deliver. The
  async methods exist because the protocol requires them, and they call the
  blocking client directly — same choice as the Agno adapter, for the same
  reason.
"""


import json
import uuid


from   typing import Any, List, Optional


from   autogen_core.memory import (
	Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult,
)
from   autogen_core.model_context import ChatCompletionContext
from   autogen_core.model_context import ChatCompletionContext
from   autogen_core.models import SystemMessage


class A2MMemory(Memory):
	"""AutoGen memory backed by an A2M server.

	Example:
		from implementations import client as a2m_client

		client = a2m_client.connect_stdio(["python", "-m", "implementations.store_sqlite", "memory.db"])
		memory = A2MMemory(client, namespace="ops")

		await memory.add(MemoryContent(content="the deploy key rotates every ninety days",
		                               mime_type=MemoryMimeType.TEXT))
		await memory.query("how often does the key change?")
	"""

	component_type = "memory"

	def __init__(self, client, namespace: str = "autogen", tier: str = None, limit: int = 5) -> None:
		"""Wire AutoGen memory to a connected A2M client.

		Args:
			client: A negotiated A2M client — implementations.client.A2MClient
				or a2m.MemoryClient, over any transport.
			namespace (str, optional): Stamped into every record's metadata;
				scopes `clear` so two agents on one store cannot wipe each
				other.
			tier (str, optional): Where writes land. Defaults to the first
				searchable tier the server describes, because these memories
				are recalled and the working tier is replayed (spec §4.4).
			limit (int, optional): How many memories update_context injects.
		"""
		self.client    = client
		self.namespace = namespace
		self.limit     = int(limit)
		self.tier      = tier or next(
			(t.get("name") for t in client.describe().get("tiers", [])
			 if t.get("kind", None) != "working"),
			None,
		)


	def _remember(self, content: MemoryContent) -> None:
		"""Write one memory content object.

		Args:
			content (MemoryContent): AutoGen's memory unit. Its text is what
				gets indexed; its metadata rides along unchanged.
		"""
		fields: dict[str, Any] = {
			"metadata": dict(content.metadata or {}, namespace=self.namespace,
			                 mime_type=str(getattr(content.mime_type, "value", content.mime_type))),
		}
		if self.tier is not None:
			fields["tier"] = self.tier

		self.client.remember(str(content.content), **fields)


	def _recall(self, query: str, limit: int) -> list[MemoryContent]:
		"""Ranked memories for a query, as AutoGen content objects.

		Args:
			query (str): What to rank against.
			limit (int): Maximum results.

		Returns:
			list[MemoryContent]: Best first. The record's score rides in the
			metadata for the caller that wants it — it orders this list and
			means nothing beyond it (spec §5.3).
		"""
		found = self.client.recall(query, limit=limit, where={"namespace": self.namespace})

		return [MemoryContent(
			content   = record.get("content", ""),
			mime_type = MemoryMimeType.TEXT,
			metadata  = dict(record.get("metadata") or {}, score=record.get("score", 0.0)),
		) for record in found]


	async def add(self, content: MemoryContent, cancellation_token=None) -> None:
		"""Store one memory (the Memory protocol's write).

		Args:
			content (MemoryContent): What to remember.
			cancellation_token: Accepted for interface compatibility; the
				underlying call is one blocking round trip.
		"""
		self._remember(content)


	async def query(self, query: str | MemoryContent, cancellation_token=None, **kwargs: Any) -> MemoryQueryResult:
		"""Search this namespace by relevance.

		Args:
			query (str | MemoryContent): The query, either as text or as a
				content object whose text is used.
			cancellation_token: Accepted for interface compatibility.
			**kwargs: Unrecognised arguments are ignored, in the same spirit as
				the protocol's own rule (spec §2).

		Returns:
			MemoryQueryResult: Ranked memories, best first.
		"""
		text = query.content if isinstance(query, MemoryContent) else query
		return MemoryQueryResult(results=self._recall(str(text), self.limit))


	async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
		"""Inject relevant memories before a model call.

		The tail of the conversation is the query: what was just said decides
		what is worth remembering into the context. Nothing is replayed — the
		transcript is already there.

		Args:
			model_context (ChatCompletionContext): The context about to be sent
				to the model. Mutated by appending one SystemMessage when
				anything relevant was found.

		Returns:
			UpdateContextResult: The memories that were injected.
		"""
		messages = await model_context.get_messages()
		if not messages:
			return UpdateContextResult(memories=MemoryQueryResult(results=[]))

		tail  = str(getattr(messages[-1], "content", ""))
		found = self._recall(tail, self.limit) if tail else []

		if found:
			listed = "\n".join(f"- {m.content}" for m in found)
			await model_context.add_message(
				SystemMessage(content=f"Relevant memories:\n{listed}")
			)

		return UpdateContextResult(memories=MemoryQueryResult(results=found))


	async def clear(self) -> None:
		"""Forget this namespace — and only this namespace (spec §4.5).
		"""
		self.client.forget(where={"namespace": self.namespace})


	async def close(self) -> None:
		"""Release the underlying transport.
		"""
		self.client.close()


class A2MChatCompletionContext(ChatCompletionContext):
	"""AutoGen's model context — the transcript itself — kept in A2M.

	`A2MMemory` above is what an agent *recalls*; this is what it *replays*. The
	distinction is the one spec §4.4 draws between `memory/recall` and
	`memory/timeline`, and it is why this class reads with `timeline` and never
	ranks: a conversation reordered by relevance is no longer a conversation,
	and most providers reject a message list whose turns have been shuffled.

	So the messages land in the **working** tier, which exists to be replayed,
	is bounded per conversation, and is never embedded.

	Two protocol rules do real work here:

	- **A tool call and its results share a `group`** (spec §3.4), so a store
	  evicting under pressure moves them together. An assistant message carrying
	  function calls, separated from the results answering it, is exactly the
	  transcript a provider refuses.
	- **The session is the conversation** (spec §3.5). Two contexts on one store
	  cannot evict each other, because working capacity is enforced per session
	  rather than across the tier.

	Structured content — function calls, execution results — is kept as JSON in
	metadata, with readable text in `content` so the turn is still indexable by
	anything else sharing the store.

	Example:
		from implementations import client as a2m_client

		client  = a2m_client.connect_stdio(["python", "-m", "implementations.store_sqlite", "memory.db"])
		context = A2MChatCompletionContext(client, session="chat-1")
	"""

	def __init__(self, client, session: str, tier: str = None, initial_messages=None) -> None:
		"""Bind a model context to one conversation in an A2M store.

		Args:
			client: A negotiated A2M client.
			session (str): The conversation id. Required, because a transcript
				with no conversation is indistinguishable from every other
				transcript in the store.
			tier (str, optional): Where turns live. The working tier by default,
				found by *kind* rather than by name so a store that calls it
				something else still works.
			initial_messages: Seed messages, as AutoGen's own contexts take.
		"""
		super().__init__(initial_messages=initial_messages)

		self.client  = client
		self.session = session
		self.tier    = tier if tier is not None else self._working_tier()
		self._group  = None


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
		return {"session": self.session} if self.client.supports("sessions") else {}


	async def add_message(self, message: Any) -> None:
		"""Append one turn.

		Args:
			message: Any AutoGen model message — system, user, assistant, or a
				function execution result.
		"""
		kind    = type(message).__name__
		content = getattr(message, "content", "")

		# Function calls and their results are objects, not text. A readable
		# rendering goes in `content` so the turn stays indexable; the exact
		# structure goes in metadata so it can be rebuilt.
		if isinstance(content, str):
			text      = content
			structure = None
		else:
			structure = json.loads(json.dumps(content, default=lambda o: getattr(o, "__dict__", str(o))))
			text      = " ".join(str(part.get("name") or part.get("content") or "")
			                     for part in structure if isinstance(part, dict)) or kind

		# An assistant turn carrying calls opens a group; the results answering
		# it join the same one, so eviction cannot separate them (spec §3.4).
		group = None
		if kind == "AssistantMessage" and structure is not None:
			self._group = f"call-{uuid.uuid4().hex[:8]}"
			group       = self._group
		elif kind == "FunctionExecutionResultMessage":
			group       = self._group
			self._group = None

		fields: dict = {
			"role"     : {"SystemMessage": "system", "UserMessage": "user",
			              "AssistantMessage": "assistant"}.get(kind, "tool"),
			"metadata" : {"autogen": {"type": kind,
			                          "source": getattr(message, "source", None),
			                          "structure": structure}},
		}
		if self.tier is not None:
			fields["tier"] = self.tier
		if self.client.supports("sessions"):
			fields["session"] = self.session
		if group is not None:
			fields["group"] = group

		self.client.remember(text, **fields)


	async def get_messages(self) -> List[Any]:
		"""The conversation, in the order it happened.

		Returns:
			list: AutoGen messages. A record another framework wrote comes back
			as a `UserMessage` rather than an error — a shared store is one
			somebody else has been writing to, and refusing to read it would
			defeat the point.
		"""
		from autogen_core.models import AssistantMessage, SystemMessage, UserMessage

		messages = []
		for record in self.client.timeline(tier=self.tier, where=self._filters() or None):
			carried   = (record.get("metadata") or {}).get("autogen") or {}
			kind      = carried.get("type")
			source    = carried.get("source") or record.get("role") or "user"
			structure = carried.get("structure")

			if kind == "SystemMessage":
				messages.append(SystemMessage(content=record.get("content", "")))
			elif kind == "AssistantMessage":
				content = structure if structure is not None else record.get("content", "")
				messages.append(AssistantMessage(content=content, source=source))
			else:
				# Function results and anything another framework wrote replay
				# as text rather than being dropped: rebuilding AutoGen's own
				# result objects needs types a foreign writer never had.
				messages.append(UserMessage(content=record.get("content", ""), source=source))

		return messages


	async def clear(self) -> None:
		"""Forget this conversation, and only this one."""
		filters = self._filters()
		if filters:
			self.client.forget(where=filters)
		elif self.tier is not None:
			self.client.forget(tier=self.tier)


	async def save_state(self) -> dict:
		"""Where the transcript lives, rather than the transcript itself.

		AutoGen's own contexts serialise their messages here. This one does not
		need to: they are already durable in the store, so the state is the
		pointer. That is the difference between a context that happens to
		persist and one whose storage is shared.

		Returns:
			dict: Enough to reattach.
		"""
		return {"session": self.session, "tier": self.tier}


	async def load_state(self, state) -> None:
		"""Reattach to a conversation.

		Args:
			state (Mapping): What `save_state` returned.
		"""
		self.session = state.get("session", self.session)
		self.tier    = state.get("tier", self.tier)
