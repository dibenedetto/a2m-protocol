"""A federated A2M server: one tier per backend, one router in front.

	python a2m_router.py memories/            # stdio, spawns four backends
	python a2m_router.py memories/ --http 8778

The router is an A2M **server** to its caller and an A2M **client** of four
backends, each of which is an ordinary single-tier A2M server holding its own
database file in its own process:

	         client
	           │  A2M
	     ┌─────┴──────┐
	     │   router   │  ← speaks A2M in both directions
	     └─────┬──────┘
	  ┌────┬───┴───┬────────┐  A2M
	working episodic semantic procedural
	  .db     .db     .db      .db + files

Nothing here reaches past the protocol. The router cannot see inside a backend
and does not try; every operation is `memory/*` calls, which is the point. If it
works, the protocol composes.

**The router owns the policy; backends own the bytes.** Capacity, spilling and
promotion live here, because only the router can see more than one tier. A
backend is a store, not a stack — which is why they are started with `--tier`.

Two problems federation creates that a single process never has:

**Scores do not survive a merge.** Spec §5.3 says `score` is never comparable
between servers, and a router that fans `recall` out to four backends and sorts
the union by score is making exactly the mistake the specification forbids
clients from making. A router is a client. Both strategies below avoid it, by
different routes — see `MergeStrategy`.

**A spill is a distributed write.** Moving a record from working to episodic
means writing to one server and deleting from another, with no transaction
spanning them. `_move` writes first, verifies, and only then deletes, so a crash
in between leaves a duplicate rather than a hole. Duplicates are recoverable;
losing a memory is not.
"""


import json
import pathlib
import subprocess
import sys
import time


from   typing    import Any, Callable


from   a2m       import (
	A2M_CAPABILITIES, A2M_VERSION, CAPABILITY_NOT_SUPPORTED, INVALID_PARAMS,
	UNKNOWN_TIER, MemoryClient, connect_stdio, serve_a2m_http,
)
from   jsonrpc   import Dispatcher, JsonRpcError, serve_stdio
from   memory    import KINDS, MemoryTier, default_tiers, recency


class MergeStrategy:
	"""Turns per-backend results into one ranked list without ever comparing two
	backends' scores to each other.

	`wants_candidates` decides what the router asks its backends for, and it is
	not a tuning knob — it follows from who is doing the ranking.

	A strategy that fuses *rankings* needs the backends to rank, so the router
	forwards the query and each backend returns its own best results.

	A strategy that ranks the records itself needs the backends to stay out of the
	way. If the router forwarded the query, each backend would apply its own
	relevance filter first, and the router could only ever re-rank what survived —
	a record the router would have scored highest is invisible if the backend's
	scorer dropped it. So the query is withheld and the backends return a broad,
	recency-ordered pool for the router to judge.

	The useful consequence: with `wants_candidates`, only the router needs a
	model. The backends are storage."""

	name             = "merge"
	wants_candidates = False

	def merge(self, query: str, results: dict[str, list[dict]], limit: int) -> list[dict]:
		"""Combine per-backend results into one ranked list.

		Args:
			query (str): What was asked.
			results (dict[str, list[dict]]): Tier name to that backend's records.
			limit (int): Maximum records to return.

		Returns:
			list[dict]: Merged records, descending, each carrying a score that is the
			*router's* rather than any backend's.
		"""
		raise NotImplementedError


class RankFusion(MergeStrategy):
	"""Reciprocal rank fusion: use each backend's *order* and discard its numbers.

	Order is the only thing the specification guarantees — §4.3 requires results
	to descend within one response and says nothing across responses. So this
	throws the scores away, which is the whole trick, and emits a score of its own
	that is honestly the router's rather than any backend's.

	Costs nothing and needs no model. It cannot tell a backend's excellent match
	from its mediocre one: only that one outranked the other."""

	name             = "rank-fusion"
	wants_candidates = False

	def __init__(self, k: float = 60.0) -> None:
		"""Configure reciprocal-rank fusion.

		Args:
			k (float, optional): The reciprocal-rank constant. Larger flattens the
				difference between ranks, so a backend's top hit outranks another's
				by less.
		"""
		self.k = float(k)


	def merge(self, query: str, results: dict[str, list[dict]], limit: int) -> list[dict]:
		"""Fuse by reciprocal rank, ignoring every backend's numbers.

		Args:
			query (str): What was asked. Unused -- the backends already ranked.
			results (dict[str, list[dict]]): Tier name to that backend's records.
			limit (int): Maximum records.

		Returns:
			list[dict]: Merged, with scores normalised so the best is 1.0.
		"""
		fused : dict[str, float] = {}
		kept  : dict[str, dict]  = {}

		for records in results.values():
			for rank, record in enumerate(records):
				id        = record["id"]
				fused[id] = fused.get(id, 0.0) + 1.0 / (self.k + rank + 1)
				kept[id]  = record

		ordered = sorted(fused.items(), key=lambda pair: -pair[1])
		if limit and limit > 0:
			ordered = ordered[:limit]

		top = ordered[0][1] if ordered else 1.0

		merged = []
		for id, score in ordered:
			record          = dict(kept[id])
			record["score"] = score / top if top else 0.0
			merged.append(record)

		return merged


class Rerank(MergeStrategy):
	"""Score every candidate with one model, so the numbers are comparable because
	they came from the same ranker.

	Backends become recall engines whose ordering is ignored; the router decides.
	Better quality than fusion, and the only strategy that can say *how much*
	better one result is than another. Costs one embedding call per recall, over
	the query plus every candidate."""

	name             = "rerank"
	wants_candidates = True

	def __init__(self, embed: Callable, fallback: MergeStrategy = None) -> None:
		"""Configure router-side re-ranking.

		Args:
			embed (Callable): Takes a list of texts and returns vectors. Only the
				router needs this -- with 'wants_candidates' the backends are pure
				storage.
			fallback (MergeStrategy, optional): Used when there is no query, or when
				the embedder fails. Defaults to RankFusion, so a ranker that is down
				degrades recall rather than breaking it.
		"""
		self.embed    = embed
		self.fallback = fallback or RankFusion()


	def merge(self, query: str, results: dict[str, list[dict]], limit: int) -> list[dict]:
		"""Score every candidate with one model, so the numbers are comparable.

		Args:
			query (str): What was asked.
			results (dict[str, list[dict]]): Tier name to that backend's candidates.
			limit (int): Maximum records.

		Returns:
			list[dict]: Merged, descending by cosine. Falls back to rank fusion when
			there is no query or the ranker is unavailable -- a ranker that is down
			must not take recall down with it.
		"""
		candidates : dict[str, dict] = {}
		for records in results.values():
			for record in records:
				candidates.setdefault(record["id"], record)

		texts = [str(r.get("content") or "") for r in candidates.values()]
		if not query or not texts:
			return self.fallback.merge(query, results, limit)

		try:
			vectors = self.embed([query] + texts)
		except Exception:
			# A ranker that is down must not take recall down with it.
			return self.fallback.merge(query, results, limit)

		from retrieval import cosine

		asked  = vectors[0]
		scored = []
		for (id, record), vector in zip(candidates.items(), vectors[1:]):
			scored.append((record, max(0.0, cosine(asked, vector))))

		scored.sort(key=lambda pair: -pair[1])
		if limit and limit > 0:
			scored = scored[:limit]

		merged = []
		for record, score in scored:
			copy          = dict(record)
			copy["score"] = score
			merged.append(copy)

		return merged


def make_merge(kind: str = "rank-fusion", embed: Callable = None) -> MergeStrategy:
	"""Build a merge strategy by name.

	Args:
		kind (str, optional): 'rank-fusion' (or 'fusion'/'rrf'), or 'rerank'.
		embed (Callable, optional): Embedder for 'rerank'. Defaults to the local
			ollama embedder.

	Returns:
		MergeStrategy: The strategy.

	Raises:
		ValueError: For an unknown name.

	Example:
		MemoryRouter(backends, merge=make_merge("rerank"))
	"""
	if kind in ("rank-fusion", "fusion", "rrf"):
		return RankFusion()
	if kind in ("rerank", "re-rank"):
		if embed is None:
			from retrieval import ollama_embedder
			embed = ollama_embedder()
		return Rerank(embed)
	raise ValueError(f"Unknown merge strategy '{kind}'; expected rank-fusion or rerank")


class MemoryRouter:
	"""An A2M server whose storage is other A2M servers."""

	def __init__(
		self,
		backends : dict[str, MemoryClient],
		tiers    : list[MemoryTier] = None,
		merge    : MergeStrategy    = None,
		name     : str              = "a2m-router",
	) -> None:
		"""Assemble a router over per-tier A2M backends.

		Args:
			backends (dict[str, MemoryClient]): Tier name to the A2M server serving
				it. Tiers with no backend are simply not served.
			tiers (list[MemoryTier], optional): The layout and its policy -- capacity,
				spilling, promotion. This lives here rather than in the backends,
				because only the router can see more than one tier.
			merge (MergeStrategy, optional): How to combine results from backends
				whose scores are not comparable. Defaults to RankFusion.
			name (str, optional): Server name, returned by describe.

		Raises:
			ValueError: If no backend matches any configured tier.

		Example:
			backends = spawn_backends("memories")
			router   = MemoryRouter(backends, merge=make_merge("rerank"))
			serve_stdio(router.dispatcher)
		"""
		tiers = tiers or default_tiers()

		self.backends = dict(backends)
		self.tiers    = {t.name: t for t in tiers if t.name in self.backends}
		self.order    = [t.name for t in tiers if t.name in self.backends]
		self.merge    = merge or RankFusion()
		self.name     = name

		if not self.order:
			raise ValueError("No backend matches any configured tier")

		self.capabilities = ["core", "tiers", "salience", "scopes", "sessions",
		                     "keys", "embeddings", "external"]
		self.methods      = [m for c in self.capabilities for m in A2M_CAPABILITIES.get(c, [])]

		self.dispatcher = Dispatcher(allow_batch=False)   # spec §8: no batches
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
		}
		for method, handler in handlers.items():
			self.dispatcher.register(method, handler)


	# ------------------------------------------------------------------ plumbing

	def _client(self, tier: str) -> MemoryClient:
		"""The backend serving a tier.

		Args:
			tier (str): Tier name.

		Returns:
			MemoryClient: Its backend.

		Raises:
			JsonRpcError: -32002 if no backend serves that tier.
		"""
		backend = self.backends.get(tier, None)
		if backend is None:
			raise JsonRpcError(UNKNOWN_TIER, f"No backend serves tier '{tier}'")
		return backend


	def _call(self, tier: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
		"""Every interaction with a backend goes through here, and through the
		protocol. The router has no other way to reach its storage."""
		return self._client(tier).client.call(method, params) or {}


	def _scoped(self, params: dict[str, Any], owner: str = None) -> dict[str, Any]:
		"""Attach an owner to parameters, if there is one.

		Args:
			params (dict): Call parameters.
			owner (str, optional): The scope to attach.

		Returns:
			dict: Parameters, with 'owner' set when scoped.
		"""
		if owner is not None:
			params = dict(params, owner=owner)
		return params


	def _records(self, tier: str, owner: str = None, where: dict[str, Any] = None,
	             key_prefix: str = None) -> list[dict]:
		"""Every record in one backend, chronologically.

		Args:
			tier (str): Tier name.
			owner (str, optional): Scope.
			where (dict, optional): Metadata filter.

		Returns:
			list[dict]: Records in wire form.
		"""
		params = {"tier": tier, "limit": 0}
		if where is not None:
			params["where"] = where
		if key_prefix is not None:
			params["key_prefix"] = key_prefix
		return self._call(tier, "memory/timeline", self._scoped(params, owner)).get("records", [])


	# ------------------------------------------------------------------- methods

	def describe(self, protocol: str = None, owner: str = None, **ignored: Any) -> dict[str, Any]:
		"""Merge every backend's description into one.

		Args:
			protocol (str, optional): The version the client speaks.
			owner (str, optional): Scope the counts.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: The router's own capabilities and tier layout, with counts gathered
			from the backends and each tier naming the backend serving it.

		Raises:
			JsonRpcError: -32007 on an incompatible protocol version.
		"""
		if protocol is not None and protocol != A2M_VERSION:
			raise JsonRpcError(-32007, f"This router speaks {A2M_VERSION}, not {protocol}")

		tiers = []
		for name in self.order:
			described = self._call(name, "memory/describe", self._scoped({}, owner))
			counts    = sum(t.get("count", 0) for t in described.get("tiers", []))
			tiers.append(dict(self.tiers[name].to_dict(), count=counts, backend=described.get("name")))

		return {
			"protocol"     : A2M_VERSION,
			"name"         : self.name,
			"capabilities" : list(self.capabilities),
			"methods"      : sorted(self.methods),
			"tiers"        : tiers,
			"order"        : list(self.order),
			"kinds"        : {k: [n for n in self.order if self.tiers[n].kind == k] for k in KINDS
			                  if any(self.tiers[n].kind == k for n in self.order)},
			"working"      : self.working(),
			"limits"       : {"max_records_per_call": 256},
			"embeddings"   : self._embedding_profile(),
			"total"        : sum(t["count"] for t in tiers),
			"scorer"       : {"scorer": "router", "merge": self.merge.name,
			                  "backends": {n: type(self.backends[n]).__name__ for n in self.order}},
		}


	def _embedding_profile(self) -> dict[str, Any]:
		"""The vector contract, gathered from the backends.

		A federation is only coherent if its backends agree on what a vector is:
		the same record can spill from one to another, and a width that changed on
		the way across would make it silently unrankable. So a disagreement is
		reported rather than averaged away -- there is no sensible middle value
		between 768 and 1024 dimensions.

		Returns:
			dict: 'dimensions', 'metric' and 'model', plus 'disagreement' listing
			every distinct width the backends reported when they do not match.
		"""
		seen = []
		for name in self.order:
			contract = self._call(name, "memory/describe", {}).get("embeddings") or {}
			if contract.get("dimensions") is not None:
				seen.append(contract["dimensions"])

		profile = {"dimensions": seen[0] if seen else None, "metric": "cosine", "model": None}
		if len(set(seen)) > 1:
			profile["disagreement"] = sorted(set(seen))

		return profile


	def working(self) -> str:
		"""The tier holding the live transcript.

		Returns:
			str: Its name.
		"""
		for name in self.order:
			if self.tiers[name].kind == "working":
				return name
		return self.order[0]


	def remember(self, records: list[dict] = None, owner: str = None,
	             session: str = None, **ignored: Any) -> dict[str, Any]:
		"""Route records to the backend owning each destination tier.

		One call per destination rather than one per record, so a mixed batch costs as
		many round trips as it has distinct tiers.

		Args:
			records (list[dict]): Partial records, each needing 'content'.
			owner (str, optional): Default owner.
			session (str, optional): Default session.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: 'ids' in the caller's original order, and a total.

		Raises:
			JsonRpcError: -32602 for a record with no content, -32002 for an unknown
				tier.
		"""
		if not records:
			raise JsonRpcError(INVALID_PARAMS, "Nothing to remember: pass 'records'")

		# One call per destination tier rather than one per record.
		batches : dict[str, list[dict]] = {}
		order   : list[tuple[str, int]] = []

		for entry in records:
			if not isinstance(entry, dict) or "content" not in entry:
				raise JsonRpcError(INVALID_PARAMS, "Each record needs a 'content' field")

			tier = entry.get("tier") or self.working()
			if tier not in self.backends:
				raise JsonRpcError(UNKNOWN_TIER, f"Unknown tier '{tier}'")

			entry = dict(entry, tier=tier)
			if session is not None and entry.get("session") is None:
				entry["session"] = session
			batches.setdefault(tier, []).append(entry)
			order.append((tier, len(batches[tier]) - 1))

		written : dict[str, list[str]] = {}
		for tier, batch in batches.items():
			written[tier] = self._call(tier, "memory/remember", self._scoped({"records": batch}, owner)).get("ids", [])

		return {"ids": [written[tier][index] for tier, index in order], "total": self._total()}


	def _total(self) -> int:
		"""How many records exist across every backend.

		Returns:
			int: Total record count.
		"""
		return sum(len(self._records(name)) for name in self.order)


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
		"""Fan the search out and merge what comes back.

		The merge strategy decides what the backends are asked for: a fusing strategy
		forwards the query so each backend ranks, while a re-ranking strategy withholds
		it and asks for a broad pool instead, because it can only rank what the
		backends chose to return.

		Args:
			query (str, optional): What to rank against.
			tier (str, optional): Restrict to one backend, in which case no merge
				happens and the backend's own scores survive.
			limit (int, optional): Maximum records.
			where (dict, optional): Metadata filter.
			min_score (float, optional): Applied after merging, against the router's
				own scale.
			owner (str, optional): Scope.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: 'records', descending.

		Raises:
			JsonRpcError: -32002 for an unknown tier.
		"""
		names = [tier] if tier else list(self.order)
		for name in names:
			if name not in self.backends:
				raise JsonRpcError(UNKNOWN_TIER, f"Unknown tier '{name}'")

		# Over-fetch: merging discards, so asking each backend for exactly `limit`
		# would starve the merge of anything to choose between. A re-ranking
		# strategy needs a far wider pool, because it is doing all the judging.
		candidates = self.merge.wants_candidates
		want       = max((limit or 8) * 8, 64) if candidates else max((limit or 8) * 2, 8)
		results    : dict[str, list[dict]] = {}

		for name in names:
			params = {"tier": name, "limit": want}
			# Withheld on purpose when the router ranks: see MergeStrategy.
			if query is not None and not candidates:
				params["query"] = query
			if where is not None:
				params["where"] = where
			if embedding is not None:
				params["embedding"] = list(embedding)
			if key_prefix is not None:
				params["key_prefix"] = key_prefix
			if embeddings:
				params["embeddings"] = True

			results[name] = self._call(name, "memory/recall", self._scoped(params, owner)).get("records", [])

		# A single backend needs no merge, and merging would replace its scores
		# with the router's for no benefit -- unless the router is the only thing
		# that ranked at all, in which case it must.
		if len(names) == 1 and not candidates:
			records = results[names[0]][:limit] if limit and limit > 0 else results[names[0]]
		else:
			records = self.merge.merge(query, results, limit)

		if min_score:
			records = [r for r in records if r.get("score", 0.0) >= min_score]

		return {"records": records}


	def timeline(self, tier: str = None, limit: int = 0, owner: str = None,
	             where: dict[str, Any] = None, key_prefix: str = None,
	             embeddings: bool = False, **ignored: Any) -> dict[str, Any]:
		"""Gather records from every backend in creation order.

		Args:
			tier (str, optional): Restrict to one backend.
			limit (int, optional): Keep the most recent N.
			owner (str, optional): Scope.
			where (dict, optional): Metadata filter.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: 'records', ascending. Sorting is lexicographic on created_at, which
			is chronological because the timestamps are RFC 3339.

		Raises:
			JsonRpcError: -32002 for an unknown tier.
		"""
		names = [tier] if tier else list(self.order)
		for name in names:
			if name not in self.backends:
				raise JsonRpcError(UNKNOWN_TIER, f"Unknown tier '{name}'")

		records = []
		for name in names:
			records.extend(self._records(name, owner, where, key_prefix))

		# created_at is RFC 3339, so lexicographic order is chronological order.
		records.sort(key=lambda r: r.get("created_at", ""))
		if limit and limit > 0:
			records = records[-limit:]

		return {"records": records}


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
		"""Delete from every backend in scope.

		Args:
			ids (list[str], optional): Delete these exactly.
			query (str, optional): Delete whatever this recalls.
			tier (str, optional): Restrict to one backend.
			where (dict, optional): Metadata filter.
			owner (str, optional): Scope.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: {'forgotten': total across backends}.

		Raises:
			JsonRpcError: -32602 when no selector is given.
		"""
		if not any([ids, query, tier, where, key_prefix]):
			raise JsonRpcError(INVALID_PARAMS, "Refusing to forget everything: pass ids, query, tier, where or key_prefix")

		names     = [tier] if tier else list(self.order)
		forgotten = 0

		for name in names:
			if name not in self.backends:
				raise JsonRpcError(UNKNOWN_TIER, f"Unknown tier '{name}'")

			params = {"tier": name}
			if ids is not None:
				params["ids"] = list(ids)
			if query is not None:
				params["query"] = query
			if where is not None:
				params["where"] = where
			if key_prefix is not None:
				params["key_prefix"] = key_prefix

			forgotten += self._call(name, "memory/forget", self._scoped(params, owner)).get("forgotten", 0)

		return {"forgotten": forgotten}


	def reinforce(self, ids: list[str], amount: float = 0.5, owner: str = None, **ignored: Any) -> dict[str, Any]:
		"""Reinforce across every backend, since the ids could be anywhere.

		Args:
			ids (list[str]): Records to reinforce.
			amount (float, optional): How much to add.
			owner (str, optional): Scope.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: {'reinforced': total}.
		"""
		count = 0
		for name in self.order:
			count += self._call(name, "memory/reinforce",
			                    self._scoped({"ids": list(ids or []), "amount": amount}, owner)).get("reinforced", 0)
		return {"reinforced": count}


	def promote(self, ids: list[str], tier: str, salience: float = 0.5, owner: str = None, **ignored: Any) -> dict[str, Any]:
		"""Move records into another backend.

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
		if tier not in self.backends:
			raise JsonRpcError(UNKNOWN_TIER, f"Unknown tier '{tier}'")

		wanted = set(ids or [])
		moved  = 0

		for name in self.order:
			if name == tier:
				continue
			group = [r for r in self._records(name, owner) if r["id"] in wanted]
			if group:
				moved += self._move(group, name, tier, salience)

		return {"promoted": moved}


	def _move(self, records: list[dict], source: str, target: str, salience: float = 0.0) -> int:
		"""Move records between two servers, with no transaction spanning them.

		Write, verify, *then* delete. If the delete fails the record exists in both
		tiers, which a later consolidation resolves; if the order were reversed a
		crash in between would lose it outright. Writing with the original id makes
		the write idempotent (spec §3.2), so a retry cannot duplicate it either."""
		if not records:
			return 0

		payload = [{
			"id"       : r["id"],
			"content"  : r.get("content", ""),
			"role"     : r.get("role", "user"),
			"metadata" : r.get("metadata") or {},
			"group"    : r.get("group") or r["id"],
			"salience" : float(r.get("salience", 1.0)) + salience,
			"owner"    : r.get("owner"),
			"session"  : r.get("session"),
			"key"      : r.get("key"),
			"uri"      : r.get("uri"),
			"media_type": r.get("media_type"),
			"tier"     : target,
		} for r in records]

		written  = self._call(target, "memory/remember", {"records": payload}).get("ids", [])
		expected = [r["id"] for r in records]

		if sorted(written) != sorted(expected):
			# The destination did not take them. Leave the source alone: a record
			# stuck in the wrong tier is a far better outcome than one that is gone.
			return 0

		self._call(source, "memory/forget", {"ids": expected})
		return len(expected)


	def session_list(self, owner: str = None, **ignored: Any) -> dict[str, Any]:
		"""Which conversations exist. Sessions span backends, so this is a fan-out.

		Args:
			owner (str, optional): Scope.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: {'sessions': [...]}, merged across backends.
		"""
		found: dict[str, dict[str, Any]] = {}

		for name in self.order:
			for record in self._records(name, owner):
				session = record.get("session")
				if session is None:
					continue
				entry = found.setdefault(session, {
					"session": session, "records": 0, "tiers": {},
					"opened_at": record.get("created_at"), "touched_at": record.get("accessed_at") or "",
				})
				entry["records"]     += 1
				entry["tiers"][name]  = entry["tiers"].get(name, 0) + 1
				entry["opened_at"]    = min(entry["opened_at"], record.get("created_at"))
				entry["touched_at"]   = max(entry["touched_at"], record.get("accessed_at") or "")

		return {"sessions": sorted(found.values(), key=lambda e: e["opened_at"])}


	def session_close(self, session: str, owner: str = None, **ignored: Any) -> dict[str, Any]:
		"""End a conversation across servers.

		The same verified move as spilling, applied to a whole conversation at once
		rather than to whatever overflowed.

		Args:
			session (str): The conversation to close.
			owner (str, optional): Scope.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: The consolidate report plus 'closed' and 'flushed'.

		Raises:
			JsonRpcError: -32602 if 'session' is missing.
		"""
		if not session:
			raise JsonRpcError(INVALID_PARAMS, "'session' is required")

		flushed = 0
		for name in self.order:
			tier = self.tiers[name]
			if tier.kind != "working" or not tier.spill_to or tier.spill_to not in self.backends:
				continue

			leaving = [r for r in self._records(name, owner) if r.get("session") == session]
			flushed += self._move(leaving, name, tier.spill_to)

		report = self.consolidate()
		return dict(report, closed=session, flushed=flushed)


	def fetch(self, key: str, owner: str = None, embeddings: bool = False, **ignored: Any) -> dict[str, Any]:
		"""Handle 'memory/fetch' -- find the record at an address, in any backend.

		A key is unique per owner across the whole stack, not per tier, so this
		asks each backend in turn and stops at the first hit.

		Args:
			key (str): The address.
			owner (str, optional): Whose key.
			embeddings (bool, optional): Include the stored vector.
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: {'record': ...} or {'record': None}.

		Raises:
			JsonRpcError: -32602 if 'key' is missing.
		"""
		if not key:
			raise JsonRpcError(INVALID_PARAMS, "'key' is required")

		for name in self.order:
			found = self._call(name, "memory/fetch",
			                   self._scoped({"key": key, "embeddings": embeddings}, owner)).get("record")
			if found:
				return {"record": found}

		return {"record": None}


	def consolidate(self, **ignored: Any) -> dict[str, Any]:
		"""Promote what was earned, then spill what overflows -- across servers.

		Args:
			**ignored: Unrecognised parameters are ignored.

		Returns:
			dict: moved, dropped, promoted and per-tier counts.
		"""
		now      = time.time()
		moved    = dropped = promoted = 0

		for name in self.order:
			tier = self.tiers[name]
			if not tier.promote_to or tier.promote_after <= 0 or tier.promote_to not in self.backends:
				continue

			earned = [r for r in self._records(name) if r.get("access_count", 0) >= tier.promote_after]
			promoted += self._move(earned, name, tier.promote_to)

		for name in self.order:
			tier    = self.tiers[name]
			records = self._records(name)

			if tier.capacity <= 0:
				continue

			if tier.per_session:
				buckets : dict[str, list[dict]] = {}
				for record in records:
					buckets.setdefault(record.get("session"), []).append(record)
			else:
				buckets = {None: records}

			for bucket in buckets.values():
				if len(bucket) <= tier.capacity:
					continue

				excess = len(bucket) - tier.capacity
				groups : dict[str, list[dict]] = {}
				for record in bucket:
					groups.setdefault(record.get("group") or record["id"], []).append(record)

				ranked = sorted(groups.values(), key=lambda g: min(self._retention(r, tier, now) for r in g))

				released = 0
				for group in ranked:
					if released >= excess:
						break
					released += len(group)

					if tier.spill_to and tier.spill_to in self.backends:
						moved += self._move(group, name, tier.spill_to)
					else:
						dropped += self._call(name, "memory/forget",
						                      {"ids": [r["id"] for r in group]}).get("forgotten", 0)

		return {
			"moved"      : moved,
			"dropped"    : dropped,
			"promoted"   : promoted,
			"summarized" : 0,
			"counts"     : {name: len(self._records(name)) for name in self.order},
		}


	def _retention(self, record: dict, tier: MemoryTier, now: float) -> float:
		"""How strongly a record has earned its place, computed from wire fields.

		Args:
			record (dict): A record in wire form.
			tier (MemoryTier): Its tier, for the half-life.
			now (float): Epoch seconds.

		Returns:
			float: Higher survives longer.
		"""
		from memory import from_rfc3339
		try:
			accessed = from_rfc3339(record.get("accessed_at") or record.get("created_at"))
		except Exception:
			accessed = now

		salience = float(record.get("salience", 1.0))
		return salience * recency(now - accessed, tier.half_life) * (1.0 + record.get("access_count", 0))


def spawn_backends(root: str | pathlib.Path, tiers: list[MemoryTier] = None, embed: bool = False) -> dict[str, MemoryClient]:
	"""Start one a2m_store.py process per tier, each with its own database.

	Args:
		root (str | Path): Directory for the database files. Created if absent.
		tiers (list[MemoryTier], optional): Which tiers to serve.
		embed (bool, optional): Give each backend an embedder.

	Returns:
		dict[str, MemoryClient]: Tier name to its backend client. Close them when
		finished, which terminates the child processes.

	Example:
		backends = spawn_backends("memories")
		router   = MemoryRouter(backends, merge=RankFusion())
		try:
			serve_stdio(router.dispatcher)
		finally:
			for backend in backends.values():
				backend.close()
	"""
	root  = pathlib.Path(root)
	root.mkdir(parents=True, exist_ok=True)
	tiers = tiers or default_tiers()

	backends = {}
	for tier in tiers:
		command = [sys.executable, "a2m_store.py", str(root / f"{tier.name}.db"), "--tier", tier.kind]
		if embed:
			command.append("--embed")

		backends[tier.name] = connect_stdio(
			command,
			on_stderr = lambda line, t=tier.name: print(f"  [{t}] {line}", file=sys.stderr),
		)

	return backends


def main() -> int:
	"""Run this file as a federated A2M server, spawning one backend per tier.

		python a2m_router.py memories/
		python a2m_router.py memories/ --http 8778
		python a2m_router.py memories/ --rerank

	Returns:
		int: Process exit code.
	"""
	argv = sys.argv[1:]
	root = next((a for a in argv if not a.startswith("--") and not a.isdigit()), "memories")

	merge = "rerank" if "--rerank" in argv else "rank-fusion"
	embed = merge == "rerank"

	backends = spawn_backends(root, embed=embed)
	router   = MemoryRouter(backends, merge=make_merge(merge))

	try:
		if "--http" in argv:
			index = argv.index("--http")
			port  = int(argv[index + 1]) if len(argv) > index + 1 else 8778
			print(f"A2M router ({merge}) on http://127.0.0.1:{port}/ over {len(backends)} backends", file=sys.stderr)
			serve_a2m_http(router, port=port).serve_forever()
		else:
			serve_stdio(router.dispatcher)
	finally:
		for backend in backends.values():
			backend.close()

	return 0


if __name__ == "__main__":
	sys.exit(main())
