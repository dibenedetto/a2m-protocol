import datetime
import threading
import time
import uuid


from   typing        import Any, Callable


from   a2m.retrieval import EmbeddingScorer, HybridScorer, LexicalScorer, Scorer


WEIGHTS = {"lexical": 0.60, "recency": 0.25, "salience": 0.15}

# What a tier is for. `working` holds the live transcript, `episodic` what
# happened, `semantic` what is true, `procedural` how to do things.
KINDS   = ("working", "episodic", "semantic", "procedural")

# Only the live transcript is private by default. What happened, what is true
# and how to do things are the things a team has any reason to pool.
PRIVATE = ("working",)


def to_rfc3339(timestamp: float) -> str:
	"""Encode an epoch timestamp as the A2M wire format.

	A2M puts timestamps on the wire as RFC 3339 UTC strings (spec 3.3). Epoch
	floats are ambiguous about unit and timezone, and lose precision in any
	language whose only number is a double -- which is most of them.

	Args:
		timestamp (float): Seconds since the Unix epoch.

	Returns:
		str: RFC 3339, UTC, millisecond precision, 'Z' suffix.

	Example:
		>>> to_rfc3339(1800000000.5)
		'2027-01-15T08:00:00.500Z'
	"""
	moment = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
	return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def from_rfc3339(value: str | float) -> float:
	"""Decode a wire timestamp back to epoch seconds.

	Accepts a number unchanged, so a caller can pass either form without having
	to check which one it holds.

	Args:
		value (str | float): RFC 3339 string, or epoch seconds.

	Returns:
		float: Seconds since the Unix epoch.

	Example:
		>>> from_rfc3339("2027-01-15T08:00:00.500Z")
		1800000000.5
	"""
	if isinstance(value, (int, float)):
		return float(value)
	return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def recency(age: float, half_life: float) -> float:
	"""Exponential decay, expressed as a half-life.

	Args:
		age (float): Seconds since the record was last touched.
		half_life (float): Seconds after which the score halves. 0 disables decay
			and always returns 1.0, which is what an unbounded tier such as
			'semantic' wants.

	Returns:
		float: A multiplier in (0, 1].

	Example:
		>>> recency(300, 300)
		0.5
		>>> recency(999, 0)
		1.0
	"""
	if half_life <= 0.0:
		return 1.0
	return 0.5 ** (max(age, 0.0) / half_life)


class MemoryRecord:
	"""One remembered thing. `group` ties records that must live or die together,
	such as an assistant tool call and the tool results that answer it."""

	def __init__(
		self,
		content    : str,
		tier       : str,
		role       : str            = "user",
		salience   : float          = 1.0,
		group      : str            = None,
		metadata   : dict[str, Any] = None,
		id         : str            = None,
		created_at : float          = None,
		owner      : str            = None,
		session    : str            = None,
		key        : str            = None,
		embedding  : list[float]    = None,
		revision   : int            = 0,
		uri        : str            = None,
		media_type : str            = None,
	) -> None:
		"""Create one remembered thing.

		Args:
			content (str): The text remembered, and the text that gets indexed.
			tier (str): Which tier it lives in.
			role (str, optional): Who produced it -- user, assistant, tool, system,
				memory. Advisory.
			salience (float, optional): How much it is worth keeping.
			group (str, optional): Records sharing a group live and die together.
				Defaults to the record's own id, so an ungrouped record is a group of
				one rather than a special case.
			metadata (dict, optional): Arbitrary JSON, round-tripped unchanged.
			id (str, optional): Generated when omitted.
			created_at (float, optional): Epoch seconds. Defaults to now.
			owner (str, optional): Which agent wrote it.
			session (str, optional): Which conversation it belongs to.
		"""
		self.id           = id or uuid.uuid4().hex
		self.content      = content
		self.tier         = tier
		self.role         = role
		self.salience     = float(salience)
		self.group        = group or self.id
		self.metadata     = dict(metadata) if metadata else {}
		self.created_at   = float(created_at) if created_at is not None else time.time()
		self.accessed_at  = self.created_at
		self.access_count = 0

		# Who wrote it. In a private tier this decides who may read it; in a shared
		# tier it survives as provenance, so a team can still tell who learnt what.
		self.owner        = owner

		# Which conversation it belongs to. The level between `group` (one turn)
		# and `owner` (one agent) that was missing: without it two conversations
		# share one working tier and evict each other's context.
		self.session      = session

		# A caller-chosen address, slash-delimited by convention:
		# "myapp/wf-42/user/city". Writing to a key that already holds a record
		# *replaces* it rather than adding another, which is the difference between
		# a memory that can be corrected and one that can only be appended to.
		self.key          = key
		self.revision     = int(revision)

		# A caller-supplied vector, stored verbatim. The store never generates it
		# and never replaces it. That is what keeps A2M model-agnostic: two
		# frameworks embedding with different models can share one store without
		# silently comparing numbers from incomparable spaces.
		self.embedding    = list(embedding) if embedding else None

		# Where the real thing lives, when the record is a reference rather than
		# the thing itself. `content` still carries whatever text should be
		# indexed -- a title, a summary, an extracted passage -- because a record
		# that is only a URI cannot be recalled by anything except its address.
		#
		# The store never dereferences this. Fetching a caller's URI would make
		# every write a request the server chose to make on the client's behalf.
		self.uri          = uri
		self.media_type   = media_type


	def touch(self, now: float = None) -> None:
		"""Mark the record as just recalled.

		Recall is what earns a record its place: 'accessed_at' feeds the recency term
		in ranking, and 'access_count' is what 'promote_after' compares against, so
		touching is how a record climbs the stack.

		Args:
			now (float, optional): Epoch seconds. Defaults to the current time.

		Example:
			>>> record = MemoryRecord("the deploy key rotates", tier="episodic")
			>>> record.touch()
			>>> record.access_count
			1
		"""
		self.accessed_at   = now if now is not None else time.time()
		self.access_count += 1


	def to_dict(self, score: float = None, embedding: bool = False) -> dict[str, Any]:
		"""The A2M wire form of this record.

		Timestamps become RFC 3339 strings here rather than the floats used
		internally, because this is what crosses a process boundary.

		Args:
			score (float, optional): Relevance, attached only by 'recall'. Omitted
				entirely when None, since a record outside a search result has no
				meaningful score.

		Returns:
			dict: The record as specified in spec 3.1.

		Example:
			>>> MemoryRecord("hello", tier="working").to_dict()["tier"]
			'working'
		"""
		record = {
			"id"           : self.id,
			"content"      : self.content,
			"tier"         : self.tier,
			"role"         : self.role,
			"salience"     : self.salience,
			"group"        : self.group,
			"metadata"     : self.metadata,
			"created_at"   : to_rfc3339(self.created_at),
			"accessed_at"  : to_rfc3339(self.accessed_at),
			"access_count" : self.access_count,
			"owner"        : self.owner,
			"session"      : self.session,
			"key"          : self.key,
			"revision"     : self.revision,
			"uri"          : self.uri,
			"media_type"   : self.media_type,
		}
		if score is not None:
			record["score"] = score

		# Vectors are large and almost never wanted, so they are opt-in. A client
		# checking that its embedding was stored verbatim asks for them.
		if embedding and self.embedding:
			record["embedding"] = list(self.embedding)

		return record


class MemoryTier:
	"""One level of the stack.

	A tier is moved through by two different forces, and keeping them apart is the
	whole point of the four-kind model:

		spill_to    where records go when the tier is over `capacity`. Pressure.
		            Nothing is earned; the weakest records are simply displaced.
		promote_to  where a record goes once it has been recalled `promote_after`
		            times. Reinforcement. An episode retrieved again and again is
		            no longer an episode, it is a fact.

	`kind` says what the tier is *for*, independently of what it is called, so a
	stack can be renamed or reshaped without the agent losing track of which tier
	holds the live transcript."""

	def __init__(
		self,
		name          : str   = None,
		kind          : str   = None,
		capacity      : int   = 0,
		spill_to      : str   = None,
		half_life     : float = 0.0,
		promote_to    : str   = None,
		promote_after : int   = 0,
		shared        : bool  = None,
		per_session   : bool  = None,
	) -> None:
		"""Define one layer of the stack, and how records move through it.

		Args:
			name (str): What this tier is called. Anything you like.
			kind (str, optional): What it is *for* -- working, episodic, semantic or
				procedural. Defaults to the name, which is why conventionally named
				tiers need not say it twice.
			capacity (int, optional): Maximum records before spilling. 0 is unbounded.
			spill_to (str, optional): Where overflow goes. Records are dropped when
				there is nowhere below.
			half_life (float, optional): Seconds after which a record's recency score
				halves. 0 disables decay.
			promote_to (str, optional): Where a record goes once it has earned it.
			promote_after (int, optional): Recalls needed to earn promotion. 0
				disables promotion.
			shared (bool, optional): Readable by every agent. Defaults by kind --
				only the transcript is private, because it is the one thing agents
				must not inherit from each other.
			per_session (bool, optional): Enforce capacity *within* each conversation
				rather than across the tier. Defaults to True for the working kind,
				so two chats cannot evict each other's context.

		Raises:
			ValueError: If 'kind' is not one of the four in KINDS.
		"""
		kind = kind or name
		if kind not in KINDS:
			raise ValueError(f"Tier '{name}' has unknown kind '{kind}'; expected one of {', '.join(KINDS)}")

		self.name          = name
		self.kind          = kind
		self.capacity      = int(capacity)
		self.spill_to      = spill_to
		self.half_life     = float(half_life)
		self.promote_to    = promote_to
		self.promote_after = int(promote_after)

		# A private tier is readable only by the agent that wrote each record; a
		# shared one is readable by the whole team. Defaults by kind, because the
		# live transcript is the one thing agents must not inherit from each other.
		self.shared        = (kind not in PRIVATE) if shared is None else bool(shared)

		# Capacity applies within a conversation, not across the tier. Only the
		# transcript needs this: two chats each deserve their own context window,
		# whereas what a team *knows* is one pool regardless of where it was said.
		self.per_session   = (kind == "working") if per_session is None else bool(per_session)


	def to_dict(self) -> dict[str, Any]:
		"""The tier's configuration, as reported by 'memory/describe'.

		Returns:
			dict: name, kind, capacity, spill_to, half_life, promote_to,
			promote_after, shared and per_session.

		Example:
			>>> MemoryTier("working", capacity=24, spill_to="episodic").to_dict()["kind"]
			'working'
		"""
		return {
			"name"          : self.name,
			"kind"          : self.kind,
			"capacity"      : self.capacity,
			"spill_to"      : self.spill_to,
			"half_life"     : self.half_life,
			"promote_to"    : self.promote_to,
			"promote_after" : self.promote_after,
			"shared"        : self.shared,
			"per_session"   : self.per_session,
		}


def default_tiers() -> list[MemoryTier]:
	"""The four canonical layers.

	Only the first three form the spill chain. Procedural memory is not something
	a fact decays into -- it is how to do things, written deliberately by loading
	a skill or by promoting a record that keeps proving useful, so nothing spills
	into it and it never expires.

	Returns:
		list[MemoryTier]: working (24), episodic (256), semantic and procedural,
		both unbounded.

	Example:
		>>> [t.kind for t in default_tiers()]
		['working', 'episodic', 'semantic', 'procedural']
	"""
	return [
		MemoryTier("working"   , capacity = 24 , spill_to = "episodic", half_life = 300.0  ),
		MemoryTier("episodic"  , capacity = 256, spill_to = "semantic", half_life = 86400.0,
		                         promote_to = "semantic", promote_after = 3),
		MemoryTier("semantic"  , capacity = 0  , spill_to = None      , half_life = 0.0    ),
		MemoryTier("procedural", capacity = 0  , spill_to = None      , half_life = 0.0    ),
	]


class MemoryStack:
	"""A layered store, safe to share between agents.

	Writes land on the working tier; consolidation pushes what no longer fits down
	the stack; recall searches any tier by relevance.

	One stack can back a whole team. Each agent reads and writes through its own
	scope (`agent=`), which keeps private tiers private while everything below
	stays pooled. Records carry an `owner` so that survives spilling.

	Two locks, deliberately. `_lock` guards the record table and is held only for
	short critical sections. `_consolidating` serialises consolidation runs so the
	cascade down the tiers is never interleaved, while leaving `consolidate_fn`
	free to call an LLM without blocking every other agent's reads and writes."""

	def __init__(
		self,
		tiers          : list[MemoryTier] = None,
		consolidate_fn : Callable         = None,
		weights        : dict[str, float] = None,
		scorer         : Scorer           = None,
	) -> None:
		"""Assemble a layered store.

		Args:
			tiers (list[MemoryTier], optional): The layers. Defaults to the four
				canonical ones.
			consolidate_fn (Callable, optional): Decides what spilling *means*.
				Called as fn(records, target_tier) and may return replacement
				contents, or None to move the records intact. This is where an LLM
				summariser belongs -- see retrieval.llm_consolidator.
			weights (dict, optional): How relevance, recency and salience combine.
				Merged over the defaults, so a partial dict is fine.
			scorer (Scorer, optional): How relevance is judged. Defaults to
				LexicalScorer, which needs no model and never leaves the process.

		Raises:
			ValueError: If a tier spills or promotes into a tier that does not exist.

		Example:
			stack = MemoryStack()

			stack = MemoryStack(
				tiers          = [MemoryTier("working", capacity=8, spill_to="episodic"),
				                  MemoryTier("episodic", capacity=0)],
				scorer         = make_scorer("hybrid"),
				consolidate_fn = llm_consolidator(model),
			)
		"""
		tiers = tiers or default_tiers()

		self.tiers          = {tier.name: tier for tier in tiers}
		self.order          = [tier.name for tier in tiers]
		self.consolidate_fn = consolidate_fn
		# Lexical for text, and vectors for callers who bring their own. The
		# embedding half carries **no model** -- it ranks only records that
		# already have a vector and abstains otherwise, so this costs nothing
		# and reaches no network.
		#
		# It has to be here rather than opt-in, because a store declaring the
		# `embeddings` capability and then ranking by recency is a store that
		# lies: a caller supplying a query vector got an answer that ignored it,
		# with a `metric` in `describe` insisting otherwise. Purely lexical
		# stacks are still available by passing LexicalScorer() explicitly.
		self.scorer         = scorer or HybridScorer([
			(LexicalScorer()        , 0.5),
			(EmbeddingScorer(None)  , 0.5),
		])
		self.weights        = dict(WEIGHTS)
		self.records        : dict[str, MemoryRecord] = {}

		self._lock          = threading.RLock()
		self._consolidating = threading.Lock()

		if weights:
			self.weights.update(weights)

		for tier in tiers:
			if tier.spill_to and tier.spill_to not in self.tiers:
				raise ValueError(f"Tier '{tier.name}' spills into unknown tier '{tier.spill_to}'")
			if tier.promote_to and tier.promote_to not in self.tiers:
				raise ValueError(f"Tier '{tier.name}' promotes into unknown tier '{tier.promote_to}'")


	@property
	def top(self) -> str:
		"""The first tier in stack order.

		Prefer 'working', which finds the transcript tier by *kind*. This one is
		positional, and only correct for a conventionally ordered stack.

		Returns:
			str: The name of the first configured tier.
		"""
		return self.order[0]


	@property
	def working(self) -> str:
		"""The tier holding the live transcript.

		Found by kind, not by position, so reordering or renaming a stack cannot
		silently repoint it at the wrong tier.

		Returns:
			str: The name of the working-kind tier, falling back to 'top'.

		Example:
			>>> stack = MemoryStack(tiers=[
			...     MemoryTier("context", kind="working", capacity=2, spill_to="history"),
			...     MemoryTier("history", kind="episodic")])
			>>> stack.working
			'context'
		"""
		return self.of_kind("working")[0] if self.of_kind("working") else self.top


	def of_kind(self, kind: str) -> list[str]:
		"""Every tier serving a given purpose.

		Args:
			kind (str): One of working, episodic, semantic, procedural.

		Returns:
			list[str]: Tier names in stack order. Empty if none serve that kind.

		Example:
			>>> MemoryStack().of_kind("procedural")
			['procedural']
		"""
		return [name for name in self.order if self.tiers[name].kind == kind]


	def tier(self, name: str) -> MemoryTier:
		"""Look up a tier by name.

		Args:
			name (str): The tier name.

		Returns:
			MemoryTier: The configured tier.

		Raises:
			KeyError: If no such tier exists. Callers at the protocol boundary turn
				this into -32002 UNKNOWN_TIER.
		"""
		tier = self.tiers.get(name, None)
		if tier is None:
			raise KeyError(f"Unknown tier '{name}'")
		return tier


	def visible(self, record: MemoryRecord, agent: str = None) -> bool:
		"""Whether 'agent' may see this record (spec 6).

		An unscoped caller sees everything -- that is the single-agent case, and the
		administrative one. A scoped caller sees shared tiers in full, plus whatever
		it wrote itself, plus records nobody claimed.

		This is data partitioning, **not** access control: the caller asserts its own
		identity and nothing here verifies it. Over a network the scope must come
		from the authenticated transport instead.

		Args:
			record (MemoryRecord): The record being considered.
			agent (str, optional): The requesting agent, or None for unscoped.

		Returns:
			bool: True if the record should be visible.
		"""
		if agent is None:
			return True

		tier = self.tiers.get(record.tier, None)
		return bool(tier and tier.shared) or record.owner is None or record.owner == agent


	def records_in(self, tier: str, agent: str = None) -> list[MemoryRecord]:
		"""Every visible record in one tier.

		Args:
			tier (str): Tier name.
			agent (str, optional): Scope the read to this agent.

		Returns:
			list[MemoryRecord]: Unordered. Use 'timeline' when order matters.

		Example:
			>>> stack = MemoryStack()
			>>> _ = stack.remember("hello")
			>>> len(stack.records_in("working"))
			1
		"""
		with self._lock:
			return [r for r in self.records.values() if r.tier == tier and self.visible(r, agent)]


	def remember(
		self,
		content  : str,
		tier     : str            = None,
		role     : str            = "user",
		salience : float          = 1.0,
		group    : str            = None,
		metadata : dict[str, Any] = None,
		owner    : str            = None,
		id       : str            = None,
		session  : str            = None,
		key      : str            = None,
		embedding: list[float]    = None,
		uri      : str            = None,
		media_type: str           = None,
	) -> MemoryRecord:
		"""Write one record.

		Args:
			content (str): The text to remember and to index.
			tier (str, optional): Where to put it. Defaults to the working tier.
			role (str, optional): Who produced it -- user, assistant, tool, system,
				memory. Advisory only.
			salience (float, optional): How much it is worth keeping. Higher survives
				eviction longer.
			group (str, optional): Records sharing a group live and die together, so
				an assistant tool call is never separated from its results. Defaults
				to the record's own id.
			metadata (dict, optional): Arbitrary JSON, round-tripped unchanged and
				filterable through 'where'.
			owner (str, optional): Which agent wrote it.
			session (str, optional): Which conversation it belongs to.
			id (str, optional): Supplying it makes the write **idempotent** -- if a
				record with that id already exists it is returned unchanged rather
				than duplicated (spec 3.2). This is the only retry-safety mechanism
				in A2M.

		Returns:
			MemoryRecord: The stored record, or the pre-existing one when 'id' was
			supplied and already present.

		Raises:
			KeyError: If 'tier' names a tier that does not exist.

		Example:
			>>> stack  = MemoryStack()
			>>> first  = stack.remember("marco lives in bologna", role="user")
			>>> second = stack.remember("anything at all", id=first.id)
			>>> first.id == second.id and stack.count() == 1
			True
		"""
		# A client-supplied id makes the write idempotent, which is the only thing
		# standing between a retried request and a duplicated memory.
		if id is not None:
			with self._lock:
				existing = self.records.get(id, None)
			if existing is not None:
				return existing

		# A key *addresses* a record rather than identifying a write, so writing to
		# an occupied key replaces what is there. That is the difference between a
		# memory that can be corrected and one that can only be appended to: when
		# the user moves city, the old fact must stop being recalled, not merely be
		# outnumbered.
		if key is not None:
			held = self.by_key(key, owner)
			if held is not None:
				with self._lock:
					held.content   = content
					held.revision += 1
					held.role      = role
					if metadata is not None:
						held.metadata = dict(metadata)
					if embedding is not None:
						held.embedding = list(embedding)
					if uri is not None:
						held.uri = uri
					if media_type is not None:
						held.media_type = media_type
					if tier is not None:
						held.tier = self.tier(tier).name
					held.salience   = float(salience)
					held.accessed_at = time.time()
					self.scorer.drop(held.id)
					self.scorer.index(held)
				return held

		tier   = tier or self.working
		record = MemoryRecord(
			content  = content,
			tier     = self.tier(tier).name,
			role     = role,
			salience = salience,
			group    = group,
			metadata  = metadata,
			owner     = owner,
			session   = session,
			key        = key,
			embedding  = embedding,
			uri        = uri,
			media_type = media_type,
			id         = id,
		)

		with self._lock:
			self.records[record.id] = record
			self.scorer.index(record)

		return record


	def get(self, id: str, agent: str = None) -> MemoryRecord | None:
		"""Fetch one record by id.

		Args:
			id (str): The record id.
			agent (str, optional): Scope the read; a record the agent may not see
				reads as absent rather than raising.

		Returns:
			MemoryRecord | None: The record, or None if missing or not visible.
		"""
		with self._lock:
			record = self.records.get(id, None)
		return record if record and self.visible(record, agent) else None


	def timeline(self, tier: str = None, limit: int = 0, agent: str = None,
	             where: dict[str, Any] = None, key_prefix: str = None) -> list[MemoryRecord]:
		"""Read records in creation order, oldest first.

		Separate from 'recall' on purpose: relevance order is what a search wants,
		creation order is what rebuilding a conversation requires, and implementing
		one in terms of the other silently corrupts transcripts.

		Args:
			tier (str, optional): Restrict to one tier.
			limit (int, optional): Keep only the most recent N -- still returned in
				ascending order. 0 means unlimited.
			agent (str, optional): Scope the read.
			where (dict, optional): Same filter as 'recall'. This is what makes
				replaying a single **session** possible; without it a caller can
				search within a conversation but never replay one.

		Returns:
			list[MemoryRecord]: Ascending by created_at.

		Example:
			>>> stack = MemoryStack()
			>>> _ = stack.remember("first",  session="chat-1")
			>>> _ = stack.remember("second", session="chat-2")
			>>> [r.content for r in stack.timeline(where={"session": "chat-1"})]
			['first']
		"""
		with self._lock:
			records = [
				r for r in self.records.values()
				if (tier is None or r.tier == tier) and self.visible(r, agent)
				and self._matches(r, where) and self._under(r, key_prefix)
			]

		records.sort(key=lambda r: r.created_at)
		return records[-limit:] if limit and limit > 0 else records


	def recall(
		self,
		query     : str            = None,
		tier      : str            = None,
		limit     : int            = 8,
		where     : dict[str, Any] = None,
		min_score : float          = 0.0,
		touch     : bool           = True,
		now       : float          = None,
		agent     : str            = None,
		embedding : list[float]    = None,
		key_prefix: str            = None,
	) -> list[tuple[MemoryRecord, float]]:
		"""Search by relevance. **This is the method that matters.**

		Ranking is delegated to the configured Scorer; this method blends what the
		scorer says with recency and tier half-life, because those depend on where a
		record lives and a scorer should not know that.

		A scorer may *abstain* (return None) when it cannot judge a query -- an empty
		one, or one that reduces to nothing after tokenizing. Abstaining falls back to
		recency and salience, which is not the same as judging and finding nothing.

		Args:
			query (str, optional): What to rank against. Absent means "rank by
				recency", not "raise an error".
			tier (str, optional): Restrict to one tier. All tiers when omitted.
			limit (int, optional): Maximum records. 0 means unlimited.
			where (dict, optional): Equality filter over record fields, falling back
				to metadata keys. A list value means "any of".
			min_score (float, optional): Drop results below this blended score.
			touch (bool, optional): Whether recall counts as an access. Pass False for
				internal reads that should not influence promotion.
			now (float, optional): Epoch seconds, for deterministic tests.
			agent (str, optional): Scope the search.

		Returns:
			list[tuple[MemoryRecord, float]]: Ordered by descending score. The score
			is **ranking information only** and is never comparable across stores or
			calls (spec 5.3).

		Example:
			>>> stack = MemoryStack()
			>>> _ = stack.remember("the deploy key rotates every ninety days")
			>>> _ = stack.remember("pasta al ragu needs three hours")
			>>> [r.content for r, _ in stack.recall("how often does the key rotate?")]
			['the deploy key rotates every ninety days']
			>>> stack.recall("something entirely unrelated")
			[]
		"""
		now = now if now is not None else time.time()

		with self._lock:
			candidates = [
				r for r in self.records.values()
				if (tier is None or r.tier == tier)
				and self.visible(r, agent)
				and self._matches(r, where)
				and self._under(r, key_prefix)
			]

		if not candidates:
			return []

		# Deliberately outside the lock: an EmbeddingScorer calls out to a model
		# here, and no other agent should have to wait on that to write a message.
		relevance = self.scorer.relevance(query, candidates, embedding)

		scored = []
		if relevance is None:
			# The scorer abstained, so ranking is recency and salience alone.
			for record in candidates:
				scored.append((record, self._blend(0.0, record, now)))
		else:
			for record in candidates:
				score = (relevance or {}).get(record.id, 0.0)
				if score > 0.0:
					scored.append((record, self._blend(score, record, now)))

		scored = [(record, score) for record, score in scored if score >= min_score]
		scored.sort(key=lambda pair: (-pair[1], -pair[0].created_at))

		if limit and limit > 0:
			scored = scored[:limit]

		if touch:
			for record, _ in scored:
				record.touch(now)

		return scored


	def forget(
		self,
		ids   : list[str]      = None,
		query : str            = None,
		tier  : str            = None,
		where : dict[str, Any] = None,
		agent : str            = None,
		key_prefix: str        = None,
	) -> int:
		"""Delete records.

		Args:
			ids (list[str], optional): Delete these exactly.
			query (str, optional): Delete whatever this recalls.
			tier (str, optional): Restrict the deletion to one tier.
			where (dict, optional): Metadata filter.
			agent (str, optional): Scope; an agent cannot forget another's private
				records.

		Returns:
			int: How many records were removed.

		Example:
			>>> stack = MemoryStack()
			>>> _ = stack.remember("pasta al ragu needs three hours")
			>>> stack.forget(query="pasta ragu")
			1
		"""
		targets: set[str] = set()

		if ids:
			with self._lock:
				targets.update(
					id for id in ids
					if id in self.records and self.visible(self.records[id], agent)
				)

		if query or where or key_prefix or (tier and not ids):
			for record, _ in self.recall(query=query, tier=tier, where=where, limit=0,
			                             touch=False, agent=agent, key_prefix=key_prefix):
				targets.add(record.id)

		with self._lock:
			for id in targets:
				self.records.pop(id, None)
				self.scorer.drop(id)

		return len(targets)


	def reinforce(self, ids: list[str], amount: float = 0.5, agent: str = None) -> int:
		"""Raise the salience of records that proved useful.

		Salience resists eviction and feeds ranking, so reinforcing is how a caller
		says "this mattered" without moving the record anywhere.

		Args:
			ids (list[str]): Records to reinforce. Unknown ids are ignored.
			amount (float, optional): How much to add.
			agent (str, optional): Scope.

		Returns:
			int: How many records were actually reinforced.
		"""
		count = 0
		with self._lock:
			for id in ids or []:
				record = self.records.get(id, None)
				if record and self.visible(record, agent):
					record.salience += amount
					record.touch()
					count += 1
		return count


	def promote(self, ids: list[str], tier: str, salience: float = 0.5) -> int:
		"""Move records to a more durable tier because they earned it.

		This is the *merit* half of the stack, as against 'consolidate''s *pressure*.
		It is also the only way anything reaches procedural memory, since nothing
		ever spills into it.

		Args:
			ids (list[str]): Records to move. Unknown ids, and records already in the
				target tier, are skipped rather than failing.
			tier (str): Destination tier.
			salience (float, optional): Added on arrival, so a promoted record is
				harder to displace than a merely spilled one.

		Returns:
			int: How many records moved.

		Raises:
			KeyError: If 'tier' does not exist.

		Example:
			>>> stack  = MemoryStack()
			>>> record = stack.remember("prefer tabs over spaces in this repo")
			>>> stack.promote([record.id], "procedural")
			1
		"""
		target = self.tier(tier).name
		count  = 0

		with self._lock:
			for id in ids or []:
				record = self.records.get(id, None)
				if record and record.tier != target:
					record.tier      = target
					record.salience += salience
					count           += 1

		return count


	def consolidate(self, now: float = None) -> dict[str, Any]:
		"""Two passes, in this order.

		First promotion: records that have been recalled often enough have earned a
		more durable tier, and moving them up before the overflow pass is what stops
		a repeatedly useful record from being displaced by sheer volume.

		Then spilling: whatever still does not fit is pushed down. Groups move
		whole, so a tool call is never separated from its results.

		Consolidation is serialised against itself, because the cascade down the
		tiers has to run uninterrupted — a spill into episodic may be what pushes
		episodic over its own capacity. It is *not* serialised against readers and
		writers: the record lock is taken in short bursts and released around
		`consolidate_fn`, which may well be an LLM call."""
		now        = now if now is not None else time.time()
		moved      = 0
		dropped    = 0
		summarized = 0
		promoted   = 0

		with self._consolidating:
			for name in self.order:
				tier = self.tiers[name]
				if not tier.promote_to or tier.promote_after <= 0:
					continue

				with self._lock:
					earned = [
						r.id for r in self.records.values()
						if r.tier == name and r.access_count >= tier.promote_after
					]

				promoted += self.promote(earned, tier.promote_to)

			for name in self.order:
				tier = self.tiers[name]

				with self._lock:
					records = [r for r in self.records.values() if r.tier == name]
					if tier.capacity <= 0:
						continue

					# On a per-session tier, capacity is enforced inside each
					# conversation. Otherwise one busy chat evicts another's context
					# purely by talking more.
					if tier.per_session:
						buckets : dict[str, list[MemoryRecord]] = {}
						for record in records:
							buckets.setdefault(record.session, []).append(record)
					else:
						buckets = {None: records}

					selected = []
					for bucket in buckets.values():
						if len(bucket) <= tier.capacity:
							continue

						excess = len(bucket) - tier.capacity
						groups : dict[str, list[MemoryRecord]] = {}
						for record in bucket:
							groups.setdefault(record.group, []).append(record)

						ranked = sorted(
							groups.values(),
							key = lambda group: min(self._retention(r, tier, now) for r in group),
						)

						released = 0
						for group in ranked:
							if released >= excess:
								break
							released += len(group)
							selected.append(group)

					if not selected:
						continue

				for group in selected:
					if not tier.spill_to:
						with self._lock:
							for record in group:
								self.records.pop(record.id, None)
								self.scorer.drop(record.id)
						dropped += len(group)
						continue

					# Outside the lock on purpose. This is where an LLM would run.
					contents = self.consolidate_fn(group, tier.spill_to) if self.consolidate_fn else None

					with self._lock:
						if contents is None:
							for record in group:
								record.tier = tier.spill_to
							moved += len(group)
						else:
							for record in group:
								self.records.pop(record.id, None)
								self.scorer.drop(record.id)
							for content in contents:
								self.remember(
									content  = content,
									tier     = tier.spill_to,
									role     = "memory",
									salience = max(r.salience for r in group),
									group    = group[0].group,
									owner    = group[0].owner,
									metadata = {"consolidated_from": [r.id for r in group]},
								)
							summarized += 1
							moved      += len(contents)

		with self._lock:
			counts = {name: sum(1 for r in self.records.values() if r.tier == name) for name in self.order}

		return {
			"moved"      : moved,
			"dropped"    : dropped,
			"summarized" : summarized,
			"promoted"   : promoted,
			"counts"     : counts,
		}


	def sessions(self, agent: str = None) -> list[dict[str, Any]]:
		"""Which conversations exist, and where their records currently sit.

		Args:
			agent (str, optional): Scope the listing.

		Returns:
			list[dict]: One entry per session, ascending by opened_at, each carrying
			session, records, tiers (tier name to count), opened_at and touched_at as
			RFC 3339 strings.

		Example:
			>>> stack = MemoryStack()
			>>> _ = stack.remember("hello", session="chat-1")
			>>> [(s["session"], s["records"]) for s in stack.sessions()]
			[('chat-1', 1)]
		"""
		found: dict[str, dict[str, Any]] = {}

		with self._lock:
			for record in self.records.values():
				if record.session is None or not self.visible(record, agent):
					continue

				entry = found.setdefault(record.session, {
					"session": record.session, "records": 0, "tiers": {},
					"opened_at": record.created_at, "touched_at": record.accessed_at,
				})
				entry["records"]        += 1
				entry["tiers"][record.tier] = entry["tiers"].get(record.tier, 0) + 1
				entry["opened_at"]       = min(entry["opened_at"], record.created_at)
				entry["touched_at"]      = max(entry["touched_at"], record.accessed_at)

		for entry in found.values():
			entry["opened_at"]  = to_rfc3339(entry["opened_at"])
			entry["touched_at"] = to_rfc3339(entry["touched_at"])

		return sorted(found.values(), key=lambda e: e["opened_at"])


	def close_session(self, session: str, agent: str = None) -> dict[str, Any]:
		"""End a conversation and let it percolate down the stack.

		Closing is not one hop. The transcript is flushed out of working memory
		unconditionally — the conversation is over, so those records will never be
		replayed and there is no reason to make them wait for capacity pressure —
		and then every tier below applies its *own* rule to what arrives: records
		that earned promotion are promoted, a tier over capacity spills, and a
		consolidator rewrites what it is given.

		Procedural is untouched, because nothing spills into procedural. A finished
		conversation does not become a procedure.
		"""
		flushed = 0

		for name in self.of_kind("working"):
			tier = self.tiers[name]
			if not tier.spill_to:
				continue

			with self._lock:
				leaving = [r for r in self.records.values()
				           if r.tier == name and r.session == session and self.visible(r, agent)]

			# Grouped, so a tool call is never separated from its results even here.
			groups: dict[str, list[MemoryRecord]] = {}
			for record in leaving:
				groups.setdefault(record.group, []).append(record)

			for group in groups.values():
				contents = self.consolidate_fn(group, tier.spill_to) if self.consolidate_fn else None

				with self._lock:
					if contents is None:
						for record in group:
							record.tier = tier.spill_to
						flushed += len(group)
					else:
						for record in group:
							self.records.pop(record.id, None)
							self.scorer.drop(record.id)
						for content in contents:
							self.remember(
								content  = content,
								tier     = tier.spill_to,
								role     = "memory",
								salience = max(r.salience for r in group),
								group    = group[0].group,
								owner    = group[0].owner,
								session  = session,
								metadata = {"consolidated_from": [r.id for r in group]},
							)
						flushed += len(contents)

		# Everything below now applies its own policy to what just arrived.
		report = self.consolidate()
		return dict(report, closed=session, flushed=flushed)


	def describe(self, agent: str = None) -> dict[str, Any]:
		"""Everything a client needs to know about this store.

		Args:
			agent (str, optional): Scope the record counts.

		Returns:
			dict: tiers (each with its configuration and count), order, kinds,
			working, private, total, weights and scorer.

		Example:
			>>> [t["name"] for t in MemoryStack().describe()["tiers"]]
			['working', 'episodic', 'semantic', 'procedural']
		"""
		return {
			"tiers"  : [
				dict(self.tiers[name].to_dict(), count=len(self.records_in(name, agent)))
				for name in self.order
			],
			"order"  : list(self.order),
			"kinds"  : {kind: self.of_kind(kind) for kind in KINDS if self.of_kind(kind)},
			"working": self.working,
			"private": [name for name in self.order if not self.tiers[name].shared],
			"scorer" : self.scorer.describe(),
			"total"  : len(self.records_in_all(agent)),
			"weights": dict(self.weights),
		}


	def count(self, agent: str = None) -> int:
		"""How many records are visible.

		A method rather than len(stack.records) so that a server can sit on a store
		with no dict to measure -- which is exactly what
		store_sqlite.SqliteMemoryStack does.

		Args:
			agent (str, optional): Scope the count.

		Returns:
			int: Visible record count across every tier.
		"""
		with self._lock:
			return sum(1 for r in self.records.values() if self.visible(r, agent))


	def records_in_all(self, agent: str = None) -> list[MemoryRecord]:
		"""Every visible record, across every tier.

		Args:
			agent (str, optional): Scope the read.

		Returns:
			list[MemoryRecord]: Unordered.
		"""
		with self._lock:
			return [r for r in self.records.values() if self.visible(r, agent)]


	def by_key(self, key: str, agent: str = None) -> MemoryRecord | None:
		"""The record addressed by a key.

		Keys are unique per owner, so two agents may each hold their own
		"user/city" without colliding.

		Args:
			key (str): The address, slash-delimited by convention.
			agent (str, optional): Whose key.

		Returns:
			MemoryRecord | None: The record, or None if the key is empty.

		Example:
			>>> stack = MemoryStack()
			>>> _ = stack.remember("bologna", key="user/city")
			>>> stack.by_key("user/city").content
			'bologna'
		"""
		with self._lock:
			for record in self.records.values():
				if record.key == key and record.owner == agent:
					return record
		return None


	def _under(self, record: MemoryRecord, prefix: str = None) -> bool:
		"""Whether a record's key sits under a prefix.

		This is what makes keys hierarchical: "myapp/wf-42/" matches everything
		beneath it, which is the recursive scope read a separate namespace field
		would have provided -- without a second addressing dimension to keep in
		sync with the first.

		Args:
			record (MemoryRecord): The candidate.
			prefix (str, optional): A key prefix. Absent matches everything.

		Returns:
			bool: True if the record is at or under the prefix.
		"""
		if not prefix:
			return True
		return bool(record.key) and record.key.startswith(prefix)


	def _matches(self, record: MemoryRecord, where: dict[str, Any] = None) -> bool:
		"""Apply a 'where' filter to one record (spec 5.2).

		Record attributes are checked first, then metadata keys, so {"role": "user"}
		and {"skill": "weather"} both work. A list value means "any of".

		Args:
			record (MemoryRecord): The candidate.
			where (dict, optional): The filter. Absent or empty matches everything.

		Returns:
			bool: True if the record satisfies every condition.
		"""
		if not where:
			return True

		for key, expected in where.items():
			actual = record.metadata.get(key, getattr(record, key, None))
			if isinstance(expected, list):
				if actual not in expected:
					return False
			elif actual != expected:
				return False

		return True


	def _blend(self, relevance: float, record: MemoryRecord, now: float) -> float:
		"""Combine relevance with recency and salience into the final ranking score.

		The scorer said how well the record matches; the stack decides how much that
		is worth against how recent and how salient it is. Weights come from
		self.weights, and the recency half-life comes from the record's own tier --
		which is precisely why this cannot live inside the scorer.

		Args:
			relevance (float): The scorer's verdict, 0..1.
			record (MemoryRecord): The candidate.
			now (float): Epoch seconds.

		Returns:
			float: The blended score, 0..1.
		"""
		tier = self.tiers[record.tier]
		return (
			self.weights["lexical" ] * relevance
			+ self.weights["recency" ] * recency(now - record.accessed_at, tier.half_life)
			+ self.weights["salience"] * (record.salience / (1.0 + record.salience))
		)


	def _retention(self, record: MemoryRecord, tier: MemoryTier, now: float) -> float:
		"""How strongly a record has earned its place. Lowest goes over the edge first.

		Args:
			record (MemoryRecord): The candidate.
			tier (MemoryTier): Its tier, for the half-life.
			now (float): Epoch seconds.

		Returns:
			float: Higher means more likely to survive eviction.
		"""
		return record.salience * recency(now - record.accessed_at, tier.half_life) * (1.0 + record.access_count)
