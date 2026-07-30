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


from   typing import Any


from   autogen_core.memory import (
	Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult,
)
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
