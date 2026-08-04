"""LangChain against an A2M server: chat history and retrieval.

	pip install langchain-core

Two adapters, because LangChain splits into two what A2M keeps in one store:

	A2MChatMessageHistory   BaseChatMessageHistory -> the working tier, replayed
	A2MRetriever            BaseRetriever          -> recall, ranked

That split is the whole reason this file is short. LangChain's chat history is
*replayed* in order and its retriever is *searched* by relevance, which is
exactly the distinction spec §4.4 draws between `memory/timeline` and
`memory/recall` — so each side maps onto one method and neither has to emulate
the other.

Three things this adapter is careful about, all of them protocol rules rather
than LangChain ones:

- **Tool calls stay whole.** An `AIMessage` carrying `tool_calls` and the
  `ToolMessage`s answering it are written with one `group`, so a store that
  evicts under pressure cannot split them (spec §3.4). A transcript missing half
  a tool exchange is one most providers reject outright.
- **History is replayed, never ranked.** `messages` calls `memory/timeline`. It
  would be trivial to call `recall` with an empty query and it would even look
  like it worked, right up until the ordering mattered.
- **Whatever the store does not recognise is preserved.** LangChain's message
  fields that A2M has no column for go into `metadata` and come back out, rather
  than being dropped on the way through.
"""


import base64
import json
import uuid


from   typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple


from   langchain_core.chat_history import BaseChatMessageHistory
from   langchain_core.documents    import Document
from   langchain_core.messages     import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from   langchain_core.retrievers   import BaseRetriever
from   langchain_core.stores       import BaseStore
from   pydantic                    import ConfigDict


# LangChain's message types and A2M's advisory 'role' (spec §3.1). The mapping is
# lossy in one direction only: several LangChain types share the 'tool' role, so
# the original type is kept in metadata rather than inferred back.
ROLE_OF_TYPE = {
	"human"  : "user",
	"ai"     : "assistant",
	"system" : "system",
	"tool"   : "tool",
}


def text_of(message: BaseMessage) -> str:
	"""The indexable text of a message, however LangChain is carrying it.

	`content` is a string on a plain message and a list of content blocks on a
	multimodal one. A2M indexes text, so the blocks that are not text are dropped
	here rather than stringified into the index -- a base64 image rendered as
	terms is noise that will match nothing and dilute everything.

	Args:
		message (BaseMessage): The message to read.

	Returns:
		str: Its text, concatenated in order.
	"""
	content = message.content

	if isinstance(content, str):
		return content

	parts = []
	for block in content or []:
		if isinstance(block, str):
			parts.append(block)
		elif isinstance(block, dict) and block.get("type") == "text":
			parts.append(block.get("text", ""))

	return "".join(parts)


def to_record(message: BaseMessage, session: str = None, group: str = None) -> dict:
	"""Turn a LangChain message into an A2M record.

	Args:
		message (BaseMessage): The message to store.
		session (str, optional): The conversation this belongs to.
		group (str, optional): Ties a tool call to the results answering it.

	Returns:
		dict: A partial record for memory/remember.
	"""
	metadata = {"lc_type": message.type}

	# Everything LangChain carries that A2M has no field for. Round-tripping it
	# through metadata is what keeps this adapter from being lossy.
	if getattr(message, "name", None):
		metadata["lc_name"] = message.name
	if getattr(message, "tool_calls", None):
		metadata["lc_tool_calls"] = json.loads(json.dumps(message.tool_calls, default=str))
	if getattr(message, "tool_call_id", None):
		metadata["lc_tool_call_id"] = message.tool_call_id
	if message.additional_kwargs:
		metadata["lc_additional_kwargs"] = json.loads(json.dumps(message.additional_kwargs, default=str))

	record = {
		"content"  : text_of(message),
		"role"     : ROLE_OF_TYPE.get(message.type, "memory"),
		"metadata" : metadata,
	}

	if session is not None:
		record["session"] = session
	if group is not None:
		record["group"] = group

	return record


def from_record(record: dict) -> BaseMessage:
	"""Turn an A2M record back into a LangChain message.

	Args:
		record (dict): A record as the server returned it.

	Returns:
		BaseMessage: The closest LangChain message. A record this adapter did not
		write still becomes a message rather than an error -- a shared store is
		one another framework has been writing to.
	"""
	metadata = record.get("metadata") or {}
	content  = record.get("content", "")
	kind     = metadata.get("lc_type") or record.get("role")

	if kind in ("ai", "assistant"):
		return AIMessage(
			content          = content,
			tool_calls       = metadata.get("lc_tool_calls") or [],
			additional_kwargs= metadata.get("lc_additional_kwargs") or {},
		)

	if kind == "system":
		return SystemMessage(content=content)

	if kind == "tool":
		return ToolMessage(
			content      = content,
			tool_call_id = metadata.get("lc_tool_call_id", ""),
		)

	return HumanMessage(content=content)


class A2MChatMessageHistory(BaseChatMessageHistory):
	"""LangChain chat history stored in A2M's working tier.

	Example:
		from implementations import client as a2m_client

		client  = a2m_client.connect_stdio(["python", "-m", "implementations.store_sqlite", "memory.db"])
		history = A2MChatMessageHistory(client, session="chat-1")

		history.add_user_message("where does marco live?")
		history.messages
	"""

	def __init__(self, client, session: str = None, tier: str = None) -> None:
		"""Bind a history to one conversation.

		Args:
			client: Any A2M client -- a2m_client.A2MClient or a2m.MemoryClient. This
				adapter calls describe-derived capability checks through it rather
				than assuming what the server supports.
			session (str, optional): The conversation id. Servers declaring
				'sessions' scope both the timeline and the eviction policy to it,
				which is what stops a busy conversation evicting a quiet one.
			tier (str, optional): Where turns live. The server's working tier by
				default, which is the one that is replayed rather than searched.
		"""
		self.client  = client
		self.session = session
		self.tier    = tier


	def _records(self) -> List[dict]:
		"""The stored records for this conversation, as the server returned them.

		Returns:
			list[dict]: Ascending by creation.
		"""
		filters = {}
		if self.session is not None and self.client.supports("sessions"):
			filters["session"] = self.session

		return self.client.timeline(tier=self.tier, **filters)


	@property
	def messages(self) -> List[BaseMessage]:
		"""The conversation, in the order it happened.

		spec §4.4: this is memory/timeline, not memory/recall. Relevance order is
		what a search wants; creation order is what rebuilding a conversation
		requires, and ranking a transcript destroys it.

		Returns:
			list[BaseMessage]: Ascending by creation.
		"""
		return [from_record(record) for record in self._records()]


	def add_messages(self, messages: Sequence[BaseMessage]) -> None:
		"""Append messages, keeping a tool exchange together.

		An assistant message carrying tool_calls and the tool results answering it
		are written with one 'group' so that a store evicting under pressure moves
		them together or not at all (spec §3.4).

		Args:
			messages (Sequence[BaseMessage]): What to append, in order.
		"""
		group   = None
		pending = None

		for message in messages:
			if getattr(message, "tool_calls", None):
				# This message opens a tool exchange; everything answering it shares
				# the group.
				pending = uuid.uuid4().hex
				group   = pending
			elif message.type == "tool":
				group = pending
			else:
				group   = None
				pending = None

			record = to_record(message, session=self.session, group=group)

			if self.tier is not None:
				record["tier"] = self.tier

			self.client.remember(record.pop("content"), **record)


	def add_message(self, message: BaseMessage) -> None:
		"""Append one message.

		Args:
			message (BaseMessage): What to append.
		"""
		self.add_messages([message])


	def clear(self) -> None:
		"""Forget this conversation.

		A server declaring 'sessions' can close the session instead, which
		percolates the whole stack rather than deleting outright -- a finished
		conversation leaves working memory but its facts survive below it.
		"""
		if self.session is not None and self.client.supports("sessions"):
			self.client.close_session(self.session)
			return

		# There is no "forget everything" in A2M, deliberately: a call with no
		# selector is -32602 rather than an emptied store (spec §4.5). So this
		# names what it is deleting.
		ids = [record["id"] for record in self._records() if record.get("id")]
		if ids:
			self.client.forget(ids=ids)


class A2MRetriever(BaseRetriever):
	"""LangChain retrieval backed by memory/recall.

	Example:
		retriever = A2MRetriever(client=client, limit=5)
		retriever.invoke("how often does the deploy key rotate?")
	"""

	# BaseRetriever is a pydantic model, so these are fields rather than plain
	# attributes. The client is an arbitrary object, which pydantic has to be told
	# to allow.
	model_config = ConfigDict(arbitrary_types_allowed=True)

	client : Any  = None
	limit  : int  = 5
	tier   : str  = None
	where  : dict = None


	def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
		"""Recall, and hand back what the store ranked.

		The 'score' each record carries orders *this* list and nothing else. It is
		put in Document.metadata unchanged and never thresholded here: spec §5.3
		makes it incomparable across calls, servers, or against a constant.

		Args:
			query (str): What the chain is trying to find.

		Returns:
			list[Document]: In the order the server ranked them.
		"""
		filters = {}
		if self.tier is not None:
			filters["tier"] = self.tier
		if self.where:
			filters["where"] = self.where

		records = self.client.recall(query, limit=self.limit, **filters)

		return [
			Document(
				page_content = record.get("content", ""),
				metadata     = dict(record.get("metadata") or {}, a2m_id=record.get("id"), a2m_score=record.get("score")),
			)
			for record in records
		]


class A2MStore(BaseStore[str, bytes]):
	"""LangChain's key-value store, addressed by A2M keys.

	`BaseStore` is the interface behind `CacheBackedEmbeddings`, document
	caches and anything else LangChain wants to put somewhere by name. Its four
	methods are almost exactly the `keys` capability (spec §3.6):

		mset        remember, with a key -- an occupied one is replaced
		mget        fetch, which returns None for an unused key rather than raising
		mdelete     forget
		yield_keys  timeline over a key prefix

	Which is why this file is short. The one design choice is what to do with a
	value that is not text.

	**Values are stored as text when they are text.** LangChain's `ByteStore`
	deals in bytes, and bytes that decode as UTF-8 are written as `content`
	unchanged -- so a cached document stays readable and, incidentally,
	recallable by anything else sharing the store. Bytes that do not decode are
	base64-encoded and marked in metadata, because a memory protocol should not
	silently corrupt a value it was handed.

	A base64 blob is addressable and not meaningfully searchable, and that is
	the honest outcome: an embedding cache is not a memory, it merely lives in
	the same store.
	"""

	def __init__(self, client, namespace: str = "langchain/store", tier: str = None) -> None:
		"""Point a LangChain key-value store at an A2M server.

		Args:
			client: A negotiated A2M client.
			namespace (str, optional): Prefixed onto every key, so a cache and a
				corpus can share a store without colliding.
			tier (str, optional): Where values live. The first searchable tier
				by default.

		Raises:
			ValueError: If the server does not declare `keys`, which this whole
				interface is.
		"""
		if not client.supports("keys"):
			raise ValueError(
				"A2MStore needs the 'keys' capability: it is a key-value store, and "
				"without addressable records mset would append rather than replace"
			)

		self.client    = client
		self.namespace = namespace.strip("/")
		self.tier      = tier if tier is not None else self._searchable_tier()


	def _searchable_tier(self) -> Optional[str]:
		"""The first tier a read can reach.

		Returns:
			str | None: A tier name, or None on a server without tiers.
		"""
		if not self.client.supports("tiers"):
			return None

		for tier in self.client.describe().get("tiers", []):
			if tier.get("kind") != "working":
				return tier.get("name")
		return None


	def _key(self, key: str) -> str:
		"""The A2M address of one stored value.

		Args:
			key (str): LangChain's key.

		Returns:
			str: `<namespace>/<key>`.
		"""
		return f"{self.namespace}/{key}"


	def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
		"""Store values by key, replacing whatever was there.

		Args:
			key_value_pairs (Sequence[tuple]): Keys and their values.
		"""
		for key, value in key_value_pairs:
			raw     = value if isinstance(value, (bytes, bytearray)) else str(value).encode("utf-8")
			encoded = False

			try:
				content = bytes(raw).decode("utf-8")
			except UnicodeDecodeError:
				content = base64.b64encode(bytes(raw)).decode("ascii")
				encoded = True

			fields = {"key": self._key(key), "role": "system",
			          "metadata": {"base64": encoded, "store": self.namespace}}
			if self.tier is not None:
				fields["tier"] = self.tier

			self.client.remember(content, **fields)


	def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
		"""Read values by key.

		Args:
			keys (Sequence[str]): Keys to read.

		Returns:
			list[bytes | None]: One entry per key, in order. None where the key
			holds nothing -- which spec §4.9 makes an ordinary answer rather
			than an error, and is exactly what BaseStore expects.
		"""
		values: List[Optional[bytes]] = []

		for key in keys:
			record = self.client.fetch(self._key(key))
			if record is None:
				values.append(None)
				continue

			content = record.get("content", "")
			if (record.get("metadata") or {}).get("base64"):
				values.append(base64.b64decode(content))
			else:
				values.append(content.encode("utf-8"))

		return values


	def mdelete(self, keys: Sequence[str]) -> None:
		"""Delete values by key.

		Deletes by id rather than by key prefix, because a prefix would also
		match every longer key beneath it -- `mdelete(["a"])` must not remove
		`a/b`.

		Args:
			keys (Sequence[str]): Keys to delete.
		"""
		doomed = []
		for key in keys:
			record = self.client.fetch(self._key(key))
			if record is not None:
				doomed.append(record["id"])

		if doomed:
			self.client.forget(ids=doomed)


	def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
		"""Every key this store holds, optionally under a prefix.

		Args:
			prefix (str, optional): A LangChain-level key prefix.

		Yields:
			str: Keys, with the namespace stripped back off.
		"""
		under = f"{self.namespace}/{prefix}" if prefix else f"{self.namespace}/"
		start = len(self.namespace) + 1

		for record in self.client.timeline(limit=0, key_prefix=under):
			key = record.get("key") or ""
			if key:
				yield key[start:]
