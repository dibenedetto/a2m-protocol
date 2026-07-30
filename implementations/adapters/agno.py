"""Agno against an A2M server: knowledge as a vector database.

	pip install agno

`A2MVectorDb` implements Agno's `VectorDb` on top of `memory/recall`, so an Agno
agent's knowledge base *is* the A2M store — the same one a LangChain agent is
writing conversation into. That is the claim A2M exists to make, and this file is
half of the evidence; [langchain.py](langchain.py) is the other half.

Where the LangChain adapter maps onto the tier that is replayed, this one maps
onto the tiers that are searched. Agno hands over documents and asks for ranked
results, which is `memory/recall` almost exactly, and the impedance is all in the
edges:

- **Agno addresses documents; A2M addresses facts.** `content_id`, `content_hash`
  and `name` are three identities Agno expects to look up by. A2M has one that
  fits — `key` (spec §3.6) — so the adapter builds a key from the content id and
  keeps the rest in `metadata`. Writing an occupied key **replaces**, which is
  what makes `upsert` honest rather than an append that leaves the stale copy
  recallable.
- **Embeddings belong to whoever made them.** When Agno's embedder has already
  produced a vector, it is passed through and stored verbatim (spec §3.7). The
  server never regenerates it. A store that re-embeds moves every record into its
  own model's space, which is the interoperability failure A2M exists to remove.
- **`search` cannot promise a search type.** Agno asks which of vector, keyword
  or hybrid are supported. The honest answer is whatever the server's ranker
  happens to be, which the protocol deliberately does not specify (spec §5.1) and
  which `describe` reports rather than guarantees.

Async is implemented by delegating to the synchronous path. A2M's transports are
blocking, and pretending otherwise by wrapping them in a thread pool would buy
concurrency this adapter cannot actually deliver.
"""


import hashlib


from   typing import Any, Dict, List, Optional


from   agno.knowledge.document import Document
from   agno.vectordb.base      import VectorDb
from   agno.vectordb.search    import SearchType


class A2MVectorDb(VectorDb):
	"""An Agno knowledge base backed by an A2M server.

	Example:
		from implementations import client as a2m_client

		client    = a2m_client.connect_stdio(["python", "-m", "implementations.store_sqlite", "memory.db"])
		knowledge = A2MVectorDb(client, namespace="handbook")

		knowledge.insert("hash-1", [Document(content="the deploy key rotates every ninety days")])
		knowledge.search("how often does the key change?")
	"""

	def __init__(self, client, namespace: str = "agno", tier: str = None, embedder=None) -> None:
		"""Point an Agno knowledge base at an A2M store.

		Args:
			client: Any A2M client -- a2m_client.A2MClient or a2m.MemoryClient.
			namespace (str, optional): Key prefix for everything this knowledge base
				owns, so one store can carry several without collision. Keys are
				hierarchical by convention rather than by a second addressing
				dimension (spec §3.6, DECISION 017).

				**Pass None to read the whole store.** A namespace scopes reads as
				well as writes, which is what you want for isolation and exactly what
				you do not want for the case A2M exists for: a knowledge base that
				can only find what it wrote is a private store again, whatever it is
				sitting on top of. With None this knowledge base still writes its own
				records but searches everything, including records another framework
				put there.
			tier (str, optional): Where documents live. A searchable tier by
				default -- never the working tier, which is replayed rather than
				ranked and would make every document invisible to search.
			embedder: An Agno embedder. Optional: without one, documents are stored
				with no vector and the server ranks them however it ranks anything.
		"""
		self.client    = client
		self.namespace = namespace.strip("/") if namespace else None
		self.embedder  = embedder
		self.tier      = tier if tier is not None else self._default_tier()


	# ---------------------------------------------------------------- plumbing

	def _default_tier(self) -> Optional[str]:
		"""The first tier a search can actually reach.

		Working memory is replayed, never searched (spec §4.4), so a document put
		there is invisible to recall by design.

		Returns:
			str | None: A tier name, or None on a server without tiers.
		"""
		if not self.client.supports("tiers"):
			return None

		for tier in self.client.describe().get("tiers", []):
			if tier.get("kind") == "semantic":
				return tier.get("name")

		for tier in self.client.describe().get("tiers", []):
			if tier.get("kind") != "working":
				return tier.get("name")

		return None


	def _key(self, *parts: str) -> str:
		"""Build the addressable key for a document.

		Args:
			*parts: Key segments below this knowledge base's namespace.

		Returns:
			str: A slash-delimited key.
		"""
		segments = [str(p) for p in parts if p is not None]
		if self.namespace:
			segments.insert(0, self.namespace)

		return "/".join(segments)


	def _document_key(self, document: Document, content_hash: str = None) -> str:
		"""The key a document is addressed by.

		Agno offers three identities and does not guarantee any of them, so this
		falls back until something is stable enough to address by. Content hash is
		last because it changes when the document does, and a key that changes is a
		key that leaves the old copy behind.

		Args:
			document (Document): The document being written.
			content_hash (str, optional): Agno's hash of the source content.

		Returns:
			str: The key.
		"""
		for candidate in (document.content_id, document.id, document.name):
			if candidate:
				return self._key(candidate)

		digest = hashlib.sha256(document.content.encode("utf-8")).hexdigest()[:16]
		return self._key(content_hash or digest)


	def _to_record(self, document: Document, content_hash: str = None, filters: Dict[str, Any] = None) -> dict:
		"""Turn an Agno document into an A2M record.

		Args:
			document (Document): What Agno wants stored.
			content_hash (str, optional): Agno's content hash, kept for lookup.
			filters (dict, optional): Extra metadata Agno attaches at write time.

		Returns:
			dict: Fields for memory/remember.
		"""
		metadata = dict(document.meta_data or {})
		metadata.update(filters or {})
		metadata.update({
			"agno_namespace"    : self.namespace,
			"agno_name"         : document.name,
			"agno_content_id"   : document.content_id,
			"agno_content_hash" : content_hash,
		})

		record = {"content": document.content, "role": "memory", "metadata": metadata}

		if self.tier is not None:
			record["tier"] = self.tier

		if self.client.supports("keys"):
			record["key"] = self._document_key(document, content_hash)

		# spec §3.7 -- a caller's vector is stored verbatim and never regenerated.
		vector = document.embedding
		if vector is None and self.embedder is not None:
			vector = self.embedder.get_embedding(document.content)

		if vector is not None and self.client.supports("embeddings"):
			record["embedding"] = list(vector)

		return record


	def _to_document(self, record: dict) -> Document:
		"""Turn an A2M record back into an Agno document.

		Args:
			record (dict): A record as the server returned it.

		Returns:
			Document: With the A2M id and score preserved in meta_data. The score
			orders this result list and means nothing outside it (spec §5.3), so it
			is carried rather than compared.
		"""
		metadata = dict(record.get("metadata") or {})
		metadata["a2m_id"]    = record.get("id")
		metadata["a2m_score"] = record.get("score")
		metadata["a2m_key"]   = record.get("key")

		return Document(
			content    = record.get("content", ""),
			id         = record.get("id"),
			name       = metadata.get("agno_name"),
			meta_data  = metadata,
			embedding  = record.get("embedding"),
			content_id = metadata.get("agno_content_id"),
		)


	def _matching(self, where: Dict[str, Any], limit: int = 1000) -> List[dict]:
		"""Every record in this knowledge base satisfying a filter.

		Args:
			where (dict): Metadata equality tests (spec §5.2).
			limit (int, optional): Cap on how many to pull back.

		Returns:
			list[dict]: The raw records.
		"""
		conditions = dict(where or {})
		if self.namespace:
			conditions["agno_namespace"] = self.namespace

		params = {"limit": limit, "where": conditions}
		if self.tier is not None:
			params["tier"] = self.tier

		return self.client.recall("", **params)


	# ------------------------------------------------------------- lifecycle

	def create(self) -> None:
		"""Nothing to create: the server owns its own schema.
		"""
		pass


	async def async_create(self) -> None:
		"""Nothing to create.
		"""
		pass


	def exists(self) -> bool:
		"""Whether the store is reachable and speaking A2M.

		Returns:
			bool: True if describe answered.
		"""
		return bool(self.client.describe())


	async def async_exists(self) -> bool:
		"""Whether the store is reachable.

		Returns:
			bool: True if describe answered.
		"""
		return self.exists()


	def drop(self) -> None:
		"""Forget everything in this knowledge base's namespace.
		"""
		self.delete()


	async def async_drop(self) -> None:
		"""Forget everything in this knowledge base's namespace.
		"""
		self.drop()


	def optimize(self) -> None:
		"""Nothing to optimise from this side.
		"""
		pass


	# ---------------------------------------------------------------- lookup

	def name_exists(self, name: str) -> bool:
		"""Whether a document with this name is stored.

		Args:
			name (str): The Agno document name.

		Returns:
			bool: True if at least one matches.
		"""
		return bool(self._matching({"agno_name": name}, limit=1))


	def async_name_exists(self, name: str) -> bool:
		"""Whether a document with this name is stored.

		Args:
			name (str): The Agno document name.

		Returns:
			bool: True if at least one matches.
		"""
		return self.name_exists(name)


	def id_exists(self, id: str) -> bool:
		"""Whether a document with this id is stored.

		Args:
			id (str): The Agno document id.

		Returns:
			bool: True if it exists.
		"""
		if self.client.supports("keys") and self.client.fetch(self._key(id)):
			return True

		return bool(self._matching({"agno_content_id": id}, limit=1))


	def content_hash_exists(self, content_hash: str) -> bool:
		"""Whether content with this hash is stored.

		Args:
			content_hash (str): Agno's hash of the source content.

		Returns:
			bool: True if at least one record carries it.
		"""
		return bool(self._matching({"agno_content_hash": content_hash}, limit=1))


	# ----------------------------------------------------------------- write

	def insert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
		"""Write documents.

		Args:
			content_hash (str): Agno's hash of the source content.
			documents (list[Document]): What to store.
			filters (dict, optional): Metadata to attach to every one.
		"""
		for document in documents:
			record = self._to_record(document, content_hash, filters)
			self.client.remember(record.pop("content"), **record)


	async def async_insert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
		"""Write documents.

		Args:
			content_hash (str): Agno's hash of the source content.
			documents (list[Document]): What to store.
			filters (dict, optional): Metadata to attach to every one.
		"""
		self.insert(content_hash, documents, filters)


	def upsert_available(self) -> bool:
		"""Whether this store can address a record and replace it.

		Returns:
			bool: True when the server declares 'keys'. Without it there is no
			handle on a *fact* -- only on a write -- and a correction could only be
			appended beside the thing it corrects.
		"""
		return bool(self.client.supports("keys"))


	def upsert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
		"""Write documents, replacing whatever occupies their keys.

		spec §3.6: writing to an occupied key replaces, keeping the id and advancing
		'revision'. That is one operation with no window in which the fact is
		missing or doubled.

		Args:
			content_hash (str): Agno's hash of the source content.
			documents (list[Document]): What to store.
			filters (dict, optional): Metadata to attach to every one.
		"""
		if not self.upsert_available():
			self.insert(content_hash, documents, filters)
			return

		self.insert(content_hash, documents, filters)


	async def async_upsert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
		"""Write documents, replacing whatever occupies their keys.

		Args:
			content_hash (str): Agno's hash of the source content.
			documents (list[Document]): What to store.
			filters (dict, optional): Metadata to attach to every one.
		"""
		self.upsert(content_hash, documents, filters)


	def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> None:
		"""Replace the metadata of every record carrying a content id.

		Args:
			content_id (str): Agno's content id.
			metadata (dict): What to merge in.
		"""
		for record in self._matching({"agno_content_id": content_id}):
			merged = dict(record.get("metadata") or {}, **metadata)
			fields = {"metadata": merged, "id": record.get("id")}

			if self.tier is not None:
				fields["tier"] = self.tier
			if record.get("key") and self.client.supports("keys"):
				fields["key"] = record["key"]

			self.client.remember(record.get("content", ""), **fields)


	# ---------------------------------------------------------------- search

	def search(self, query: str, limit: int = 5, filters: Optional[Any] = None) -> List[Document]:
		"""Rank this knowledge base against a query.

		Args:
			query (str): What the agent is looking for.
			limit (int, optional): How many documents at most.
			filters (dict, optional): Metadata conditions (spec §5.2).

		Returns:
			list[Document]: In the order the server ranked them.
		"""
		conditions = dict(filters or {}) if isinstance(filters, dict) else {}
		if self.namespace:
			conditions["agno_namespace"] = self.namespace

		params = {"limit": limit, "where": conditions}
		if self.tier is not None:
			params["tier"] = self.tier

		# spec §3.7 -- when this knowledge base has an embedder, the query vector is
		# the caller's too, and travels with the query.
		if self.embedder is not None and self.client.supports("embeddings"):
			params["embedding"] = list(self.embedder.get_embedding(query))

		return [self._to_document(record) for record in self.client.recall(query, **params)]


	async def async_search(self, query: str, limit: int = 5, filters: Optional[Any] = None) -> List[Document]:
		"""Rank this knowledge base against a query.

		Args:
			query (str): What the agent is looking for.
			limit (int, optional): How many documents at most.
			filters (dict, optional): Metadata conditions.

		Returns:
			list[Document]: In ranked order.
		"""
		return self.search(query, limit, filters)


	def get_supported_search_types(self) -> List[SearchType]:
		"""Which search types this store can honestly claim.

		Ranking is not specified (spec §5.1) and a server may swap its ranker, so
		this reports what `describe` says the server declares rather than promising
		a strategy the protocol never fixed.

		Returns:
			list[SearchType]: vector when the server takes embeddings, keyword
			always, hybrid when it does both.
		"""
		if self.client.supports("embeddings"):
			return [SearchType.vector, SearchType.keyword, SearchType.hybrid]

		return [SearchType.keyword]


	# ---------------------------------------------------------------- delete

	def delete(self) -> bool:
		"""Forget everything in this knowledge base's namespace.

		Returns:
			bool: True if anything was removed.
		"""
		ids = [record["id"] for record in self._matching({}) if record.get("id")]
		if not ids:
			return False

		return bool(self.client.forget(ids=ids))


	def delete_by_id(self, id: str) -> bool:
		"""Forget one document by its Agno id.

		Args:
			id (str): The document id.

		Returns:
			bool: True if anything was removed.
		"""
		return bool(self.client.forget(ids=[id]))


	def delete_by_name(self, name: str) -> bool:
		"""Forget every document with a name.

		Args:
			name (str): The Agno document name.

		Returns:
			bool: True if anything was removed.
		"""
		return self._delete_matching({"agno_name": name})


	def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
		"""Forget every document matching a metadata filter.

		Args:
			metadata (dict): Equality conditions.

		Returns:
			bool: True if anything was removed.
		"""
		return self._delete_matching(metadata)


	def delete_by_content_id(self, content_id: str) -> bool:
		"""Forget every document carrying a content id.

		Args:
			content_id (str): Agno's content id.

		Returns:
			bool: True if anything was removed.
		"""
		return self._delete_matching({"agno_content_id": content_id})


	def _delete_matching(self, where: Dict[str, Any]) -> bool:
		"""Forget everything in this namespace satisfying a filter.

		Args:
			where (dict): Equality conditions.

		Returns:
			bool: True if anything was removed.
		"""
		ids = [record["id"] for record in self._matching(where) if record.get("id")]
		if not ids:
			return False

		return bool(self.client.forget(ids=ids))
