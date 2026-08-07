"""Agno's *other* store: user memories on A2M, not just the knowledge base.

	pip install agno

[agno.py](agno.py) maps Agno's `VectorDb` — its knowledge base — onto A2M. That
is only half of what an Agno agent persists. The other half goes through
`agno.db.base.BaseDb`, a relational store holding user memories, sessions,
evaluation runs, traces and metrics. An agent whose knowledge is shared but
whose *memories* are in a private SQLite file is not really sharing its memory,
which is the claim this repository makes on its first line.

`AgnoA2MDb` closes that half for the part of `BaseDb` that is genuinely memory:

	user memories   -> A2M records, addressed by key, in a searchable tier
	everything else -> inherited from Agno's own InMemoryDb

**Why not all 47 methods.** `BaseDb` also stores evaluation runs, spans, traces,
metrics and schema versions. Those are observability and bookkeeping, not
memory, and backing them with a memory protocol would be a category error — the
kind of scope creep that makes a small protocol stop being implementable. They
are inherited rather than raised, so an agent using them still works; they just
live for the life of the process, and an agent that needs them durable should
give Agno a real database for them and use this only for memories.

The mapping itself is close, because a "user memory" is exactly what A2M calls
a fact with an address:

- **`memory_id` becomes a key**, so re-upserting one **replaces** it (spec §3.6)
  rather than appending a second copy that competes with the first.
- **`topics` become metadata**, filterable through `where`, which is how
  `get_user_memories(topics=...)` is answered without a second index.
- **`user_id` scopes by key prefix**, so one store serves many users and
  `clear_memories()` cannot reach past the one it was called for.
- **`search_content` becomes `recall`**, which is the one operation Agno's own
  relational backends do worst and A2M does best.

Reads are deliberately **not** confined to this adapter's namespace, for the
reason DECISION 023 records: a store you can only read your own writes from is a
private store with extra steps.
"""


from   datetime import datetime, timezone
from   typing   import Any, Dict, List, Optional


from   agno.db.in_memory import InMemoryDb
from   agno.db.schemas   import UserMemory


class AgnoA2MDb(InMemoryDb):
	"""Agno's database interface, with user memories kept in an A2M store.

	Example:
		from implementations import client as a2m_client
		from agno.agent import Agent

		client = a2m_client.connect_stdio(["python", "-m", "implementations.store_sqlite", "memory.db"])
		agent  = Agent(db=AgnoA2MDb(client), enable_user_memories=True)
	"""

	def __init__(self, client, namespace: str = "agno/memories", tier: str = None) -> None:
		"""Point Agno's memory storage at an A2M server.

		Args:
			client: A negotiated A2M client — implementations.client.A2MClient or
				a2m.MemoryClient, over any transport.
			namespace (str, optional): Key prefix owning this adapter's writes.
				Scopes writes and deletion, never reads.
			tier (str, optional): Where memories land. The first searchable tier
				by default: working memory is replayed rather than ranked, so a
				memory written there would be invisible to `search_content`.

		Raises:
			ValueError: If the server does not declare `keys`, which the whole
				mapping rests on — without addressable records an upserted
				memory would append rather than replace.
		"""
		super().__init__()

		if not client.supports("keys"):
			raise ValueError(
				"AgnoA2MDb needs the 'keys' capability: a user memory is addressed by its "
				"memory_id, and without keys an upsert would append a second copy"
			)

		self.client    = client
		self.namespace = namespace.strip("/")
		self.tier      = tier if tier is not None else self._searchable_tier()


	# ---------------------------------------------------------------- plumbing

	def _searchable_tier(self) -> Optional[str]:
		"""The first tier a search can reach.

		Returns:
			str | None: A tier name, or None on a server without tiers.
		"""
		if not self.client.supports("tiers"):
			return None

		for tier in self.client.describe().get("tiers", []):
			if tier.get("kind") != "working":
				return tier.get("name")
		return None


	def _prefix(self, user_id: str = None) -> str:
		"""The key prefix owning one user's memories, or all of them.

		Args:
			user_id (str, optional): Whose memories.

		Returns:
			str: A key prefix ending in a slash.
		"""
		return f"{self.namespace}/{user_id}/" if user_id else f"{self.namespace}/"


	def _key(self, memory: UserMemory) -> str:
		"""The address of one memory.

		Args:
			memory (UserMemory): Agno's record.

		Returns:
			str: `<namespace>/<user>/<memory_id>`.
		"""
		return f"{self._prefix(memory.user_id)}{memory.memory_id}"


	def _to_wire(self, memory: UserMemory) -> Dict[str, Any]:
		"""A UserMemory as A2M write fields.

		Args:
			memory (UserMemory): Agno's record.

		Returns:
			dict: Keyword fields for remember().
		"""
		fields: Dict[str, Any] = {
			"key"      : self._key(memory),
			"role"     : "memory",
			"metadata" : {
				"topics"     : list(memory.topics or []),
				"user_id"    : memory.user_id,
				"agent_id"   : memory.agent_id,
				"team_id"    : memory.team_id,
				"memory_id"  : memory.memory_id,
				"input"      : memory.input,
				"feedback"   : memory.feedback,
				"created_at" : memory.created_at,
				"updated_at" : memory.updated_at,
			},
		}
		if self.tier is not None:
			fields["tier"] = self.tier
		return fields


	def _to_memory(self, wire: Dict[str, Any]) -> UserMemory:
		"""An A2M wire record as a UserMemory.

		Args:
			wire (dict): The record as the server returned it.

		Returns:
			UserMemory: Rebuilt. Fields A2M does not carry come from metadata,
			and anything unrecognised is ignored rather than carried, which is
			the tolerance spec §2 asks of every client.
		"""
		metadata = wire.get("metadata") or {}
		key      = wire.get("key") or ""

		return UserMemory(
			memory     = wire.get("content", ""),
			memory_id  = metadata.get("memory_id") or key.rsplit("/", 1)[-1],
			topics     = list(metadata.get("topics") or []),
			user_id    = metadata.get("user_id"),
			agent_id   = metadata.get("agent_id"),
			team_id    = metadata.get("team_id"),
			input      = metadata.get("input"),
			feedback   = metadata.get("feedback"),
			created_at = metadata.get("created_at"),
			updated_at = metadata.get("updated_at"),
		)


	def _held(self, user_id: str = None, where: Dict[str, Any] = None) -> List[Dict[str, Any]]:
		"""Every memory record under a prefix, oldest first.

		Args:
			user_id (str, optional): Whose memories.
			where (dict, optional): A metadata filter applied server-side.

		Returns:
			list[dict]: Records in wire form.
		"""
		return self.client.timeline(limit=0, key_prefix=self._prefix(user_id), where=where)


	# ------------------------------------------------------- user memories

	def upsert_user_memory(self, memory: UserMemory, deserialize: Optional[bool] = True):
		"""Write one memory, replacing any memory already at its id.

		Args:
			memory (UserMemory): What to store.
			deserialize (bool, optional): Return the object rather than a dict,
				matching Agno's own backends.

		Returns:
			UserMemory | dict: What was stored.
		"""
		memory.updated_at = memory.updated_at or int(datetime.now(timezone.utc).timestamp())
		self.client.remember(memory.memory, **self._to_wire(memory))
		return memory if deserialize else memory.to_dict()


	def upsert_memories(self, memories: List[UserMemory], deserialize: Optional[bool] = True,
	                    preserve_updated_at: bool = False) -> List[Any]:
		"""Write several memories.

		Args:
			memories (list[UserMemory]): What to store.
			deserialize (bool, optional): Return objects rather than dicts.
			preserve_updated_at (bool, optional): Leave `updated_at` alone.

		Returns:
			list: What was stored, in order.
		"""
		written = []
		for memory in memories:
			if not preserve_updated_at:
				memory.updated_at = int(datetime.now(timezone.utc).timestamp())
			written.append(self.upsert_user_memory(memory, deserialize=deserialize))
		return written


	def get_user_memory(self, memory_id: str, deserialize: Optional[bool] = True,
	                    user_id: Optional[str] = None):
		"""Read one memory by id.

		Args:
			memory_id (str): Which memory.
			deserialize (bool, optional): Return the object rather than a dict.
			user_id (str, optional): Whose. Without it every user is searched,
				since a memory id is unique on its own.

		Returns:
			UserMemory | dict | None: The memory, or None.
		"""
		if user_id is not None:
			record = self.client.fetch(f"{self._prefix(user_id)}{memory_id}")
			if record is None:
				return None
			memory = self._to_memory(record)
			return memory if deserialize else memory.to_dict()

		for wire in self._held():
			if (wire.get("key") or "").endswith(f"/{memory_id}"):
				memory = self._to_memory(wire)
				return memory if deserialize else memory.to_dict()
		return None


	def get_user_memories(
		self,
		user_id        : Optional[str]       = None,
		agent_id       : Optional[str]       = None,
		team_id        : Optional[str]       = None,
		topics         : Optional[List[str]] = None,
		search_content : Optional[str]       = None,
		limit          : Optional[int]       = None,
		page           : Optional[int]       = None,
		sort_by        : Optional[str]       = None,
		sort_order     : Optional[str]       = None,
		deserialize    : Optional[bool]      = True,
		**ignored      : Any,
	) -> List[Any]:
		"""Read memories, filtered and optionally ranked.

		`search_content` goes to `memory/recall`, which is the operation Agno's
		relational backends do worst and an A2M store does best -- ranked rather
		than pattern-matched.

		Args:
			user_id (str, optional): Whose memories.
			agent_id (str, optional): Filter by the agent that wrote it.
			team_id (str, optional): Filter by team.
			topics (list[str], optional): Keep memories carrying any of these.
			search_content (str, optional): Rank by relevance to this.
			limit (int, optional): Maximum results.
			page (int, optional): 1-based page, with `limit` as the page size.
			sort_by (str, optional): Accepted; ordering is the store's.
			sort_order (str, optional): Accepted; ordering is the store's.
			deserialize (bool, optional): Return objects rather than dicts.
			**ignored: Unrecognised arguments are ignored.

		Returns:
			list: Matching memories.
		"""
		where = {name: value for name, value in
		         (("agent_id", agent_id), ("team_id", team_id)) if value is not None}

		if search_content:
			# Searching is the operation a shared store exists for, so it is
			# **not** confined to this adapter's namespace unless a user was
			# named. A memory store you can only find your own writes in is a
			# private store with extra steps (DECISION 023) -- and this adapter
			# reintroduced exactly that bug, caught by the interop matrix on the
			# first run that included it.
			found = self.client.recall(
				search_content,
				limit      = limit or 20,
				where      = where or None,
				key_prefix = self._prefix(user_id) if user_id else None,
			)
		else:
			# Enumeration is a different question -- "what memories do I have"
			# rather than "what is relevant" -- so it stays inside the namespace.
			found = self._held(user_id, where or None)

		memories = [self._to_memory(wire) for wire in found]

		# A2M's `where` is a conjunction of equality tests (spec §5.2), and
		# "carries any of these topics" is not one, so it is applied here.
		if topics:
			wanted   = set(topics)
			memories = [m for m in memories if wanted & set(m.topics or [])]

		if limit:
			start    = ((page or 1) - 1) * limit
			memories = memories[start:start + limit]

		return memories if deserialize else [m.to_dict() for m in memories]


	def delete_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> None:
		"""Delete one memory.

		Args:
			memory_id (str): Which memory.
			user_id (str, optional): Whose.
		"""
		self.delete_user_memories([memory_id], user_id=user_id)


	def delete_user_memories(self, memory_ids: List[str], user_id: Optional[str] = None) -> None:
		"""Delete several memories.

		Args:
			memory_ids (list[str]): Which memories.
			user_id (str, optional): Whose.
		"""
		wanted = set(memory_ids or [])
		doomed = [wire["id"] for wire in self._held(user_id)
		          if (wire.get("metadata") or {}).get("memory_id") in wanted
		          or (wire.get("key") or "").rsplit("/", 1)[-1] in wanted]

		if doomed:
			self.client.forget(ids=doomed)


	def clear_memories(self) -> None:
		"""Delete every memory this adapter owns — and nothing else.

		Bounded by the namespace prefix, so a store shared with another
		framework, or with a second Agno agent, survives intact.
		"""
		self.client.forget(key_prefix=self._prefix())


	def get_all_memory_topics(self, user_id: Optional[str] = None) -> List[str]:
		"""Every topic in use, deduplicated.

		Args:
			user_id (str, optional): Whose memories to look at.

		Returns:
			list[str]: Topic names, in first-seen order.
		"""
		topics: List[str] = []
		for wire in self._held(user_id):
			for topic in (wire.get("metadata") or {}).get("topics") or []:
				if topic not in topics:
					topics.append(topic)
		return topics


	def get_user_memory_stats(self, limit: Optional[int] = None, page: Optional[int] = None,
	                          **ignored: Any) -> tuple:
		"""How many memories each user has.

		Args:
			limit (int, optional): Page size.
			page (int, optional): 1-based page.
			**ignored: Unrecognised arguments are ignored.

		Returns:
			tuple: (rows, total), each row carrying `user_id` and
			`total_memories`, matching Agno's own backends.
		"""
		counts: Dict[str, int] = {}
		for wire in self._held():
			user = (wire.get("metadata") or {}).get("user_id")
			counts[user] = counts.get(user, 0) + 1

		rows = [{"user_id": user, "total_memories": total} for user, total in counts.items()]
		if limit:
			start = ((page or 1) - 1) * limit
			return rows[start:start + limit], len(rows)
		return rows, len(rows)
