"""CrewAI against an A2M server: unified memory on a shared store.

	pip install crewai

`A2MStorageBackend` implements CrewAI's `StorageBackend` protocol, so a crew's
unified memory lives in an A2M store — the same one a LangChain agent replays,
an Agno knowledge base searches and an AutoGen agent recalls from. Wire it in
wherever CrewAI accepts a storage backend.

The fit is unusually close, because the two data models arrived at the same
shapes independently:

- **CrewAI's `scope` is a hierarchy; so are A2M keys.** A record at scope
  `/company/team` is stored under the key `<namespace>/company/team/<id>`, and
  every scope-prefixed operation — search, list, count, reset — becomes a
  `key_prefix` read (spec §3.6). No second addressing scheme is invented; the
  one the protocol already has is the one CrewAI wanted.
- **CrewAI embeds on its side; A2M stores vectors verbatim.** `search` hands
  this backend a `query_embedding` and `save` hands it records that may carry
  one. Both pass through untouched (spec §3.7): the server never re-embeds,
  which is exactly what lets CrewAI's embedder coexist with anyone else's on
  the same store.
- **`importance` is salience.** CrewAI's 0..1 importance rides A2M's
  `salience` field, so a store that consolidates keeps what CrewAI said
  mattered.
- **`update` replaces; so does a key write.** Writing an occupied key keeps
  the id, takes the new content and advances `revision` — which is precisely
  the contract `update` asks for, and why `save` writes each record under a
  key derived from its CrewAI id.

Scores are the store's, not cosine. A2M deliberately does not specify how a
server ranks (spec §5.1), and the blended score that comes back orders one
result list only (spec §5.3). `min_score` is forwarded to the server, where it
is applied against that server's own scale — a caller who thresholds should
have calibrated there, which is equally true of CrewAI's native backends.

This backend needs the `keys` and `embeddings` capabilities and refuses at
construction a server that lacks them, per spec §2: a client must not call
into a capability that was not declared.
"""


from   datetime import datetime, timezone
from   typing   import Any


from   crewai.memory.types import MemoryRecord, ScopeInfo


class A2MStorageBackend:
	"""CrewAI unified-memory storage on an A2M server.

	Satisfies `crewai.memory.storage.backend.StorageBackend`, which is a
	runtime-checkable Protocol — structural, so nothing here inherits anything.

	Example:
		from implementations import client as a2m_client

		client  = a2m_client.connect_stdio(["python", "-m", "implementations.store_sqlite", "memory.db"])
		backend = A2MStorageBackend(client, namespace="crew-1")

		backend.save([MemoryRecord(content="the deploy key rotates every ninety days",
		                           scope="/ops", categories=["infra"])])
		backend.search(query_embedding, scope_prefix="/ops")
	"""

	def __init__(self, client, namespace: str = "crewai", tier: str = None) -> None:
		"""Wire CrewAI's storage protocol to a connected A2M client.

		Args:
			client: A negotiated A2M client — implementations.client.A2MClient
				or a2m.MemoryClient, over any transport.
			namespace (str, optional): Prefixed onto every key. This is what
				lets two crews share one store: each one's scope tree, and each
				one's `reset`, is confined to its own prefix.
			tier (str, optional): Where writes land. Defaults to the first
				searchable tier the server describes — never the working tier,
				which is replayed rather than searched (spec §4.4).

		Raises:
			ValueError: If the server does not declare `keys` and `embeddings`,
				which this backend is built on.
		"""
		self.client    = client
		self.namespace = namespace.strip("/")

		for capability in ("keys", "embeddings"):
			if not client.supports(capability):
				raise ValueError(
					f"CrewAI's storage backend needs the '{capability}' capability, "
					f"which this A2M server does not declare"
				)

		self.tier = tier or next(
			(t.get("name") for t in client.describe().get("tiers", [])
			 if t.get("kind", None) != "working"),
			None,
		)


	# ---------------------------------------------------------------- plumbing

	def _prefix(self, scope_prefix: str = None) -> str:
		"""The key prefix selecting a scope subtree — or the whole namespace.

		Args:
			scope_prefix (str, optional): A CrewAI scope path like "/company".

		Returns:
			str: The A2M key prefix.
		"""
		scope = (scope_prefix or "/").strip()
		if not scope.startswith("/"):
			scope = "/" + scope
		return f"{self.namespace}{scope}".rstrip("/") + "/"


	def _read_prefix(self, scope_prefix: str = None) -> str | None:
		"""The key prefix a *read* should use — which is not the one a write uses.

		A namespace scopes writes and deletions, so two crews on one store
		cannot overwrite or reset each other. It must **not** scope reads: a
		backend that only finds what it wrote is a private store with extra
		steps, which is the exact failure A2M exists to remove.

		So an unscoped read reads everything the store is willing to show,
		including records written by another framework entirely — which is the
		point of sharing a store. A read naming a scope is confined to that
		scope beneath this namespace, because then the caller asked.

		This is the same mistake, and the same fix, as the Agno adapter's
		`namespace=None` (DECISION 023). It was reintroduced here and caught by
		`tools/test_interop.py` on its first run.

		Args:
			scope_prefix (str, optional): A CrewAI scope path.

		Returns:
			str | None: A key prefix, or None to search the whole store.
		"""
		if scope_prefix is None:
			return None
		return self._prefix(scope_prefix)


	def _key(self, scope: str, id: str) -> str:
		"""The key addressing one record.

		Args:
			scope (str): The record's scope path.
			id (str): The record's CrewAI id.

		Returns:
			str: `<namespace>/<scope>/<id>`.
		"""
		return f"{self._prefix(scope)}{id}"


	def _fields(self, record: MemoryRecord) -> dict[str, Any]:
		"""A MemoryRecord as A2M write fields.

		The caller's metadata stays at the top level of A2M `metadata`, where
		`metadata_filter` can reach it through `where`; everything CrewAI-shaped
		that A2M has no field for nests under one reserved key.

		Args:
			record (MemoryRecord): CrewAI's record.

		Returns:
			dict: Keyword fields for remember().
		"""
		system = {
			"scope"         : record.scope,
			"categories"    : list(record.categories),
			"source"        : record.source,
			"private"       : record.private,
			"created_at"    : record.created_at.isoformat(),
			"last_accessed" : record.last_accessed.isoformat(),
		}

		fields: dict[str, Any] = {
			"key"      : self._key(record.scope, record.id),
			"salience" : float(record.importance),
			"metadata" : dict(record.metadata, _crewai=system),
		}
		if record.embedding is not None:
			fields["embedding"] = list(record.embedding)
		if self.tier is not None:
			fields["tier"] = self.tier

		return fields


	def _to_record(self, wire: dict[str, Any], scope_prefix_len: int = None) -> MemoryRecord:
		"""An A2M wire record as a MemoryRecord.

		Args:
			wire (dict): The record as the server returned it.
			scope_prefix_len (int, optional): Unused; kept for clarity of intent.

		Returns:
			MemoryRecord: Rebuilt, with unknown wire fields ignored rather than
			carried — the tolerance spec §2 asks of every client.
		"""
		metadata = dict(wire.get("metadata") or {})
		system   = metadata.pop("_crewai", {}) or {}

		def stamp(name: str, fallback: str) -> datetime:
			value = system.get(name, None) or wire.get(fallback, None)
			try:
				return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
			except (TypeError, ValueError):
				return datetime.now(timezone.utc)

		key = wire.get("key") or ""
		id  = key.rsplit("/", 1)[-1] if key else wire.get("id", "")

		return MemoryRecord(
			id            = id,
			content       = wire.get("content", ""),
			scope         = system.get("scope", "/"),
			categories    = list(system.get("categories", []) or []),
			metadata      = metadata,
			importance    = min(1.0, max(0.0, float(wire.get("salience", 0.5)))),
			created_at    = stamp("created_at", "created_at"),
			last_accessed = stamp("last_accessed", "accessed_at"),
			source        = system.get("source", None),
			private       = bool(system.get("private", False)),
		)


	def _under(self, scope_prefix: str = None, where: dict[str, Any] = None,
	           mine_only: bool = False) -> list[dict[str, Any]]:
		"""Every wire record in a scope subtree, oldest first.

		Args:
			scope_prefix (str, optional): The subtree.
			where (dict, optional): A metadata filter to apply server-side.
			mine_only (bool, optional): Confine to this namespace regardless of
				scope. Deletion and reset pass True; reads never do, so that a
				crew can see what other frameworks wrote.

		Returns:
			list[dict]: Records in wire form.
		"""
		prefix = self._prefix(scope_prefix) if mine_only else self._read_prefix(scope_prefix)
		return self.client.timeline(limit=0, key_prefix=prefix, where=where)


	# ------------------------------------------------------- StorageBackend

	def save(self, records: list[MemoryRecord]) -> None:
		"""Persist records, one key each.

		Args:
			records (list[MemoryRecord]): What CrewAI wants stored. A record
				saved twice replaces itself — its key is derived from its id.
		"""
		for record in records:
			self.client.remember(record.content, **self._fields(record))


	def search(
		self,
		query_embedding : list[float],
		scope_prefix    : str = None,
		categories      : list[str] = None,
		metadata_filter : dict[str, Any] = None,
		limit           : int = 10,
		min_score       : float = 0.0,
	) -> list[tuple[MemoryRecord, float]]:
		"""Rank a scope subtree against a query vector.

		The vector goes to the server verbatim and the ranking comes back on
		the server's own scale (spec §5.1, §5.3). Category filtering happens
		here, because a category list is not an equality test A2M's `where`
		makes: the server is over-asked and the surplus trimmed.

		Args:
			query_embedding (list[float]): CrewAI's embedder's output.
			scope_prefix (str, optional): Restrict to a subtree.
			categories (list[str], optional): Keep records sharing at least one.
			metadata_filter (dict, optional): Equality tests on caller metadata.
			limit (int, optional): Maximum results.
			min_score (float, optional): Forwarded to the server, applied
				against its own score scale.

		Returns:
			list[tuple[MemoryRecord, float]]: Best first.
		"""
		want  = max(limit * 3, 15) if categories else limit
		found = self.client.recall(
			None,
			limit      = want,
			embedding  = list(query_embedding),
			key_prefix = self._read_prefix(scope_prefix),
			where      = metadata_filter,
			min_score  = min_score,
		)

		matches = []
		for wire in found:
			record = self._to_record(wire)
			if categories and not set(categories) & set(record.categories):
				continue
			matches.append((record, float(wire.get("score", 0.0))))
			if len(matches) >= limit:
				break

		return matches


	def delete(
		self,
		scope_prefix    : str = None,
		categories      : list[str] = None,
		record_ids      : list[str] = None,
		older_than      : datetime = None,
		metadata_filter : dict[str, Any] = None,
	) -> int:
		"""Delete what matches every given criterion.

		Args:
			scope_prefix (str, optional): Restrict to a subtree.
			categories (list[str], optional): Restrict to these categories.
			record_ids (list[str], optional): Restrict to these ids.
			older_than (datetime, optional): Restrict to records created before.
			metadata_filter (dict, optional): Restrict by caller metadata.

		Returns:
			int: How many records were removed.
		"""
		wanted = set(record_ids or [])
		doomed = []

		for wire in self._under(scope_prefix, where=metadata_filter, mine_only=True):
			record = self._to_record(wire)
			if wanted and record.id not in wanted:
				continue
			if categories and not set(categories) & set(record.categories):
				continue
			if older_than is not None:
				created = record.created_at
				anchor  = older_than if older_than.tzinfo else older_than.replace(tzinfo=timezone.utc)
				created = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
				if created >= anchor:
					continue
			doomed.append(wire.get("id"))

		return self.client.forget(ids=doomed) if doomed else 0


	def update(self, record: MemoryRecord) -> None:
		"""Replace the record with the same id.

		Writing to its occupied key is exactly this contract: the record there
		keeps its id, takes the new content, and advances `revision`
		(spec §3.6). No id is supplied here, deliberately — a supplied id makes
		a write idempotent, which is the opposite of an update.

		Args:
			record (MemoryRecord): The new state.
		"""
		self.client.remember(record.content, **self._fields(record))


	def get_record(self, record_id: str) -> MemoryRecord | None:
		"""One record by CrewAI id, wherever in the namespace it lives.

		Args:
			record_id (str): The id to find.

		Returns:
			MemoryRecord | None: The record, or None when nothing has that id.
		"""
		for wire in self._under(None):
			if (wire.get("key") or "").endswith(f"/{record_id}"):
				return self._to_record(wire)
		return None


	def list_records(self, scope_prefix: str = None, limit: int = 200, offset: int = 0) -> list[MemoryRecord]:
		"""Records in a scope, newest first.

		Args:
			scope_prefix (str, optional): The subtree.
			limit (int, optional): Page size.
			offset (int, optional): How many newest records to skip.

		Returns:
			list[MemoryRecord]: Newest first.
		"""
		newest_first = list(reversed(self._under(scope_prefix)))
		page         = newest_first[offset:offset + limit] if limit else newest_first[offset:]
		return [self._to_record(wire) for wire in page]


	def get_scope_info(self, scope: str) -> ScopeInfo:
		"""What a scope holds: counts, categories, date range, children.

		Args:
			scope (str): The scope path.

		Returns:
			ScopeInfo: Computed from the records under the scope.
		"""
		records  = [self._to_record(w) for w in self._under(scope)]
		children = self.list_scopes(scope)

		seen: list[str] = []
		for record in records:
			for category in record.categories:
				if category not in seen:
					seen.append(category)

		return ScopeInfo(
			path          = scope,
			record_count  = len(records),
			categories    = seen,
			oldest_record = min((r.created_at for r in records), default=None),
			newest_record = max((r.created_at for r in records), default=None),
			child_scopes  = children,
		)


	def list_scopes(self, parent: str = "/") -> list[str]:
		"""Immediate child scopes under a parent, derived from the keys.

		Args:
			parent (str): The parent scope path.

		Returns:
			list[str]: Child scope paths, each one segment deeper.
		"""
		prefix   = self._prefix(parent)
		base     = parent.rstrip("/")
		children = []

		for wire in self._under(parent):
			rest = (wire.get("key") or "")[len(prefix):]
			if "/" in rest:
				child = f"{base}/{rest.split('/', 1)[0]}"
				if child not in children:
					children.append(child)

		return children


	def list_categories(self, scope_prefix: str = None) -> dict[str, int]:
		"""Category names and their record counts within a scope.

		Args:
			scope_prefix (str, optional): The subtree. All of the namespace
				when omitted.

		Returns:
			dict[str, int]: Category to count.
		"""
		counts: dict[str, int] = {}
		for wire in self._under(scope_prefix):
			for category in self._to_record(wire).categories:
				counts[category] = counts.get(category, 0) + 1
		return counts


	def count(self, scope_prefix: str = None) -> int:
		"""How many records a scope subtree holds.

		Args:
			scope_prefix (str, optional): The subtree. The namespace when omitted.

		Returns:
			int: The count.
		"""
		return len(self._under(scope_prefix))


	def reset(self, scope_prefix: str = None) -> None:
		"""Delete a scope subtree — never more than this namespace.

		The store is shared; the namespace is not. Even a reset with no scope
		is bounded by the namespace prefix, which also satisfies A2M's refusal
		to forget without any selector at all (spec §4.5).

		Args:
			scope_prefix (str, optional): The subtree. The whole namespace when
				omitted.
		"""
		self.client.forget(key_prefix=self._prefix(scope_prefix))


	# Async delegates. A2M's transports are blocking, and a thread pool here
	# would advertise concurrency this backend cannot deliver — the same
	# choice as the Agno and AutoGen adapters, for the same reason.

	async def asave(self, records: list[MemoryRecord]) -> None:
		"""save, awaited."""
		self.save(records)


	async def asearch(
		self,
		query_embedding : list[float],
		scope_prefix    : str = None,
		categories      : list[str] = None,
		metadata_filter : dict[str, Any] = None,
		limit           : int = 10,
		min_score       : float = 0.0,
	) -> list[tuple[MemoryRecord, float]]:
		"""search, awaited."""
		return self.search(query_embedding, scope_prefix, categories, metadata_filter, limit, min_score)


	async def adelete(
		self,
		scope_prefix    : str = None,
		categories      : list[str] = None,
		record_ids      : list[str] = None,
		older_than      : datetime = None,
		metadata_filter : dict[str, Any] = None,
	) -> int:
		"""delete, awaited."""
		return self.delete(scope_prefix, categories, record_ids, older_than, metadata_filter)
