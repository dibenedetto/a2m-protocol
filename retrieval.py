"""Pluggable policies for making memory findable.

A scorer answers exactly one question — *how well does this record match what was
asked* — as a number in 0..1. It deliberately knows nothing about recency,
salience or tiers; the stack blends those in afterwards, because they depend on
where a record lives and a scorer should not.

Three are provided, and which one a stack uses is decided at construction:

	LexicalScorer     idf-weighted term overlap. Offline, deterministic, free.
	EmbeddingScorer   cosine over embeddings. Finds meaning that shares no words.
	HybridScorer      both, so embeddings rescue what the lexical pass misses.

`relevance()` returns None to abstain — "I cannot judge this query" — which is
not the same as returning {} for "I judged it and nothing matched". The stack
needs to tell those apart: abstaining falls back to recency and salience, while
an empty verdict correctly recalls nothing.
"""


import math


from   typing import Any, Callable


from   text   import tokenize


K1 = 1.2


def cosine(a: list[float], b: list[float]) -> float:
	"""Cosine similarity between two vectors.

	Args:
		a (list[float]): First vector.
		b (list[float]): Second vector.

	Returns:
		float: Similarity in [-1, 1], or 0.0 for empty, mismatched or zero
		vectors -- degenerate input returns "no opinion" rather than raising.

	Example:
		>>> cosine([1.0, 0.0], [1.0, 0.0])
		1.0
		>>> cosine([1.0, 0.0], [0.0, 1.0])
		0.0
	"""
	if not a or not b or len(a) != len(b):
		return 0.0

	dot   = sum(x * y for x, y in zip(a, b))
	norm  = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
	return dot / norm if norm else 0.0


def idf(document_frequency: int, total: int) -> float:
	"""Inverse document frequency, smoothed.

	This is why a hand-written stopword list is a cold-start aid rather than a
	law: across four documents 'the' scores 0.105 against 'marco' at 1.204, so a
	corpus of any size suppresses common words on its own and by a wider margin
	than any list.

	Args:
		document_frequency (int): How many candidates contain the term.
		total (int): How many candidates there are.

	Returns:
		float: Higher for rarer terms.

	Example:
		>>> round(idf(4, 4), 3), round(idf(1, 4), 3)
		(0.105, 1.204)
	"""
	return math.log(1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5))


class Scorer:
	"""Answers one question: how well does this record match what was asked.

	A scorer deliberately knows nothing about recency, salience or tiers. The
	stack blends those in afterwards, because they depend on where a record lives
	and a scorer should not know that.

	Subclass it to plug in any ranking you like -- spec 5.1 leaves ranking
	entirely to the implementation, which is where implementations should compete.

	Example:
		class AlwaysFirst(Scorer):
			def relevance(self, query, candidates):
				return {candidates[0].id: 1.0} if candidates else {}

		MemoryStack(scorer=AlwaysFirst())
	"""

	def index(self, record: Any) -> None:
		"""Called once per record written, so the scorer can build whatever it needs.

		Args:
			record (MemoryRecord): The record just stored.
		"""
		pass


	def drop(self, id: str) -> None:
		"""Called when a record is deleted, so the scorer can release what it held.

		Args:
			id (str): The record id that no longer exists.
		"""
		pass


	def relevance(self, query: str, candidates: list[Any], vector: list[float] = None) -> dict[str, float] | None:
		"""Rank candidates against a query.

		Args:
			query (str): What was asked.
			candidates (list[MemoryRecord]): The records in scope, already filtered by
				tier, owner and metadata.
			vector (list[float], optional): A caller-supplied query embedding. A
				scorer that cannot use one ignores it.

		Returns:
			dict[str, float] | None: Record id to relevance in 0..1, containing only
			records that matched. Return **None to abstain** -- "I cannot judge this
			query" -- which is not the same as returning {} for "I judged it and
			nothing matched". The stack needs to tell those apart: abstaining falls
			back to recency and salience, an empty verdict correctly recalls nothing.
		"""
		raise NotImplementedError


	def describe(self) -> dict[str, Any]:
		"""What this scorer is, for 'memory/describe'.

		Returns:
			dict: At minimum a 'scorer' key naming the strategy.
		"""
		return {"scorer": type(self).__name__}


class LexicalScorer(Scorer):
	"""Term overlap weighted by idf, with saturating term frequency.

	Leaving `stopwords` as None detects the language of each text and filters it
	with that language's function words — because a fixed English list is not
	merely incomplete elsewhere, it deletes Italian content words. Passing a set
	pins the filter; passing an empty set disables it. `language` pins the
	detection instead, for a stack known to be monolingual.

	`unindexed` counts records that hold text but produced no terms at all. That
	is the signature of a script the tokenizer cannot handle, and it is worth
	surfacing: such records are invisible to lexical recall while looking
	perfectly healthy in every other view."""

	def __init__(
		self,
		stopwords  : frozenset[str] = None,
		k1         : float          = K1,
		min_length : int            = 2,
		language   : str            = None,
	) -> None:
		"""Configure term-overlap ranking.

		Args:
			stopwords (frozenset, optional): Wins if given -- **including an empty
				set**, which means "filter nothing". When None the language is
				detected per text, so a store holding several languages filters each
				record by its own function words.
			k1 (float, optional): Term-frequency saturation. Higher lets repeated
				terms keep counting for longer.
			min_length (int, optional): Shortest token to index. Waived for CJK.
			language (str, optional): Pin the language instead of detecting it, for a
				store known to be monolingual.
		"""
		self.stopwords  = None if stopwords is None else frozenset(stopwords)
		self.k1         = float(k1)
		self.min_length = int(min_length)
		self.language   = language
		self.tokens     : dict[str, list[str]] = {}
		self.unindexed  : set[str] = set()


	def _tokenize(self, text: str) -> list[str]:
		"""Tokenize with this scorer's stopword and language settings.

		Args:
			text (str): Any text.

		Returns:
			list[str]: Terms.
		"""
		return tokenize(text, self.stopwords, self.min_length, self.language)


	def index(self, record: Any) -> None:
		"""Tokenize and remember a record's terms.

		Also tracks records that hold text but produce no terms at all. That is the
		signature of a script the tokenizer cannot handle, and it is worth surfacing:
		such records are invisible to lexical recall while looking perfectly healthy
		in every other view.

		Args:
			record (MemoryRecord): The record just stored.
		"""
		tokens = self.tokens[record.id] = self._tokenize(record.content)

		if str(record.content).strip() and not tokens:
			self.unindexed.add(record.id)
		else:
			self.unindexed.discard(record.id)


	def drop(self, id: str) -> None:
		"""Forget a record's terms.

		Args:
			id (str): The record id.
		"""
		self.tokens.pop(id, None)
		self.unindexed.discard(id)


	def terms_of(self, record: Any) -> list[str]:
		"""A record's terms, tokenizing on demand if it was never indexed.

		Args:
			record (MemoryRecord): The record.

		Returns:
			list[str]: Its terms.
		"""
		tokens = self.tokens.get(record.id, None)
		if tokens is None:
			self.index(record)
			tokens = self.tokens[record.id]
		return tokens


	def relevance(self, query: str, candidates: list[Any], vector: list[float] = None) -> dict[str, float] | None:
		"""Idf-weighted term overlap with saturating term frequency.

		Args:
			query (str): What was asked.
			candidates (list[MemoryRecord]): Records in scope.
			vector (list[float], optional): Ignored -- this scorer reads words.

		Returns:
			dict[str, float] | None: Ids to relevance, containing only records sharing
			at least one term. None when the query has no usable terms at all, which
			is an abstention rather than a verdict.

		Example:
			>>> from memory import MemoryRecord
			>>> scorer = LexicalScorer()
			>>> one = MemoryRecord("the deploy key rotates", tier="episodic")
			>>> scorer.relevance("", [one]) is None
			True
			>>> list(scorer.relevance("deploy key", [one]))  == [one.id]
			True
		"""
		terms = set(self._tokenize(query)) if query else set()
		if not terms:
			return None

		total              = len(candidates)
		document_frequency : dict[str, int] = {}

		for record in candidates:
			unique = set(self.terms_of(record))
			for term in terms:
				if term in unique:
					document_frequency[term] = document_frequency.get(term, 0) + 1

		ideal = sum(idf(document_frequency.get(term, 0), total) for term in terms)
		if not ideal:
			return {}

		scores = {}
		for record in candidates:
			counts : dict[str, int] = {}
			for token in self.terms_of(record):
				counts[token] = counts.get(token, 0) + 1

			score = 0.0
			for term in terms:
				frequency = counts.get(term, 0)
				if frequency:
					score += idf(document_frequency.get(term, 0), total) * (frequency / (frequency + self.k1))

			if score > 0.0:
				scores[record.id] = min(score / ideal, 1.0)

		return scores


	def describe(self) -> dict[str, Any]:
		"""What this scorer is, and how healthy its index is.

		Returns:
			dict: The strategy, its stopword setting ('auto' when detecting per text),
			the language, k1, min_length, and how many records are indexed versus
			**unindexed** -- a non-zero 'unindexed' means those records cannot be
			found lexically at all.
		"""
		return {
			"scorer"     : "lexical",
			"stopwords"  : "auto" if self.stopwords is None else len(self.stopwords),
			"language"   : self.language or "auto",
			"k1"         : self.k1,
			"min_length" : self.min_length,
			"indexed"    : len(self.tokens),
			"unindexed"  : len(self.unindexed),
		}


class EmbeddingScorer(Scorer):
	"""Cosine similarity over embeddings.

	Records are embedded lazily and in one batch at recall time rather than on
	write. Writes happen on every single message; recalls are rarer and can
	amortise the whole backlog into a single call."""

	def __init__(self, embed: Callable, threshold: float = 0.0) -> None:
		"""Configure similarity ranking over embeddings.

		Args:
			embed (Callable): Takes a list of texts and returns a list of vectors, in
				order. Batched, because recall embeds the whole backlog at once.
			threshold (float, optional): Similarities at or below this are dropped
				rather than returned with a low score.
		"""
		self.embed     = embed
		self.threshold = float(threshold)
		self.vectors   : dict[str, list[float]] = {}


	def drop(self, id: str) -> None:
		"""Forget a record's vector.

		Args:
			id (str): The record id.
		"""
		self.vectors.pop(id, None)


	def relevance(self, query: str, candidates: list[Any], vector: list[float] = None) -> dict[str, float] | None:
		"""Cosine similarity over embeddings.

		A record carrying its own `embedding` is ranked with that vector, verbatim.
		This scorer never regenerates or replaces a caller-supplied embedding, which
		is what lets two frameworks using different models share one store: whoever
		wrote a record owns the space it is compared in.

		With `vector` supplied, no model is called for the query either -- so a
		store with no embedder at all still answers vector searches. That is the
		fully caller-owned case.

		Records are embedded lazily and in one batch here rather than on write. Writes
		happen on every message; recalls are rarer and can amortise the whole backlog
		into a single call. The practical effect is that working memory is never
		embedded at all, because it is evicted before anything searches it.

		Args:
			query (str): What was asked.
			candidates (list[MemoryRecord]): Records in scope.

		Returns:
			dict[str, float] | None: Ids to similarity clamped to 0..1, or None when
			there is no usable query.

		Example:
			from retrieval import EmbeddingScorer, ollama_embedder
			MemoryStack(scorer=EmbeddingScorer(ollama_embedder()))
		"""
		asked = list(vector) if vector else None

		if asked is None and (not query or not str(query).strip() or self.embed is None):
			return None

		# Caller-supplied vectors win, always, and are never overwritten.
		for record in candidates:
			own = getattr(record, "embedding", None)
			if own:
				self.vectors[record.id] = list(own)

		if self.embed is not None:
			missing = [r for r in candidates if r.id not in self.vectors and str(r.content).strip()]
			if missing:
				for record, made in zip(missing, self.embed([str(r.content) for r in missing])):
					self.vectors[record.id] = list(made)

		vector = asked if asked is not None else self.embed([str(query)])[0]

		scores = {}
		for record in candidates:
			known = self.vectors.get(record.id, None)
			if not known:
				continue

			similarity = cosine(vector, known)
			if similarity > self.threshold:
				scores[record.id] = min(max(similarity, 0.0), 1.0)

		return scores


	def describe(self) -> dict[str, Any]:
		"""What this scorer is, and how many vectors it holds.

		Returns:
			dict: The strategy, its similarity threshold, and how many vectors are
			currently cached.
		"""
		return {"scorer": "embedding", "threshold": self.threshold, "vectors": len(self.vectors)}


class HybridScorer(Scorer):
	"""Weighted union of other scorers.

	A union, not an intersection: a record that only the embedding pass found
	still surfaces, which is the entire reason for combining them. Weights are
	renormalised over whichever scorers did not abstain, so the result stays in
	0..1 even when one of them had nothing to say."""

	def __init__(self, scorers: list[Any]) -> None:
		"""Combine several scorers into one.

		Args:
			scorers (list): Either Scorer instances, or (scorer, weight) pairs.
				A bare scorer is given weight 1.0. Weights are renormalised at query
				time over whichever scorers did not abstain.

		Raises:
			ValueError: If the list is empty.

		Example:
			HybridScorer([(LexicalScorer(), 0.4), (EmbeddingScorer(embed), 0.6)])
		"""
		self.scorers = []
		for entry in scorers:
			scorer, weight = entry if isinstance(entry, (tuple, list)) else (entry, 1.0)
			self.scorers.append((scorer, float(weight)))

		if not self.scorers:
			raise ValueError("HybridScorer needs at least one scorer")


	def index(self, record: Any) -> None:
		"""Index into every component scorer.

		Args:
			record (MemoryRecord): The record just stored.
		"""
		for scorer, _ in self.scorers:
			scorer.index(record)


	def drop(self, id: str) -> None:
		"""Drop from every component scorer.

		Args:
			id (str): The record id.
		"""
		for scorer, _ in self.scorers:
			scorer.drop(id)


	def relevance(self, query: str, candidates: list[Any], vector: list[float] = None) -> dict[str, float] | None:
		"""Weighted union of the component scorers.

		A union, not an intersection: a record that only the embedding pass found
		still surfaces, which is the entire reason for combining them.

		Args:
			query (str): What was asked.
			candidates (list[MemoryRecord]): Records in scope.

		Returns:
			dict[str, float] | None: Combined relevance, renormalised over whichever
			scorers did not abstain so the result stays in 0..1 even when one of them
			had nothing to say. None only when every component abstained.
		"""
		usable = []
		for scorer, weight in self.scorers:
			scores = scorer.relevance(query, candidates, vector)
			if scores is not None:
				usable.append((scores, weight))

		if not usable:
			return None

		total = sum(weight for _, weight in usable) or 1.0

		combined : dict[str, float] = {}
		for scores, weight in usable:
			for id, score in scores.items():
				combined[id] = combined.get(id, 0.0) + (weight / total) * score

		return combined


	def describe(self) -> dict[str, Any]:
		"""What this scorer is, and what it is made of.

		Returns:
			dict: The strategy and each component's own description with its weight.
		"""
		return {"scorer": "hybrid", "parts": [dict(s.describe(), weight=w) for s, w in self.scorers]}


# Chosen by measurement, not reputation. See bench_embeddings.py, which ranks
# ten facts against queries asked in five languages and scores precision@1:
#
#   model                     size      same-language   cross-lingual
#   lexical (no model)        --             76.0%             7.0%
#   nomic-embed-text          0.27 GB        82.0%            24.5%
#   granite-embedding:278m    0.56 GB        88.0%            84.5%
#   paraphrase-multilingual   0.56 GB        94.0%            89.0%
#   bge-m3                    1.16 GB       100.0%            99.5%
#
# bge-m3 wins on both axes and is no slower than the two mid-sized models
# (20.2s against 20.4s and 18.6s for the same 50 batched calls). It costs disk,
# not latency. nomic-embed-text is 3.7x faster and fine for English alone.
EMBEDDING_MODEL = "bge-m3"


def ollama_embedder(model: str = EMBEDDING_MODEL, **kwargs) -> Callable:
	"""Batched embeddings from a local ollama model.

	Embeddings are what make recall work in a script the tokenizer would struggle
	with, and they need no word lists at all — but only a genuinely multilingual
	model retrieves across a language change. The English-centric default this
	started with scored 24.5% cross-lingual, barely above coincidence: a
	conversation that switched language could not retrieve its own history.

	Pass `nomic-embed-text` back if the work is English-only and the 3.7x speed
	matters more than the 75-point cross-lingual gap."""
	from ollama import embed

	def embed_fn(texts: list[str]) -> list[list[float]]:
		"""Embed a batch of texts.

		Args:
			texts (list[str]): Texts to embed.

		Returns:
			list[list[float]]: One vector per input, in order.

		Raises:
			RuntimeError: If the model is not installed, with the 'ollama pull'
				command needed to fix it.
		"""
		try:
			response = embed(model=model, input=list(texts), **kwargs)
		except Exception as exc:
			if "not found" in str(exc).lower():
				raise RuntimeError(f"Embedding model '{model}' is not installed. Run: ollama pull {model}") from exc
			raise

		return [list(vector) for vector in response.embeddings]

	return embed_fn


def make_scorer(kind: str = "lexical", embed: Callable = None, weights: tuple = (0.4, 0.6), **kwargs) -> Scorer:
	"""Pick a scorer by name, so a stack can be configured from data."""
	if kind == "lexical":
		return LexicalScorer(**kwargs)

	if kind == "embedding":
		return EmbeddingScorer(embed or ollama_embedder(), **kwargs)

	if kind == "hybrid":
		lexical, embedding = weights
		return HybridScorer([
			(LexicalScorer()                              , lexical  ),
			(EmbeddingScorer(embed or ollama_embedder()) , embedding),
		])

	raise ValueError(f"Unknown scorer '{kind}'; expected lexical, embedding or hybrid")


def llm_consolidator(model: Any, tiers: set[str] = None, system: str = None, limit: int = 12) -> Callable:
	"""A `consolidate_fn` that rewrites records into durable statements as they
	spill, so that cheap retrieval works better because the stored text got
	better — rather than because the scorer got cleverer.

	It runs only when a group lands in one of `tiers` (semantic by default), so
	the frequent working-to-episodic spill stays free. Anything it cannot parse
	returns None, which makes the stack fall back to moving the records intact:
	a bad summary must never be able to destroy the originals."""
	tiers  = tiers or {"semantic"}
	system = system or (
		"You compress an agent's memory. Rewrite the numbered turns below as standalone "
		"factual statements that will still make sense with no surrounding conversation. "
		"Resolve every pronoun into the name or thing it refers to. Write one statement "
		"per line, each starting with '- '. Keep names, numbers and dates exactly. Add "
		"nothing that is not stated. If there is no durable fact, reply with NOTHING. "
		# Without this the summary comes back in English and the record stops
		# matching the language the rest of the conversation is being held in.
		"Write each statement in the same language as the turn it came from."
	)

	def consolidate(records: list[Any], target: str) -> list[str] | None:
		"""Rewrite a group of records into durable statements, or decline.

		Args:
			records (list[MemoryRecord]): The group being spilled.
			target (str): The tier they are heading into.

		Returns:
			list[str] | None: Replacement contents, or **None to decline** -- which
			makes the stack fall back to moving the originals intact. Declining is
			the safe path and is taken whenever the target tier is not one this
			consolidator handles, the model fails, or the reply cannot be parsed.
			A bad summary must never be able to destroy what it was summarising.
		"""
		if target not in tiers:
			return None

		lines = [f"{i}. [{r.role}] {r.content}" for i, r in enumerate(records[:limit], 1) if str(r.content).strip()]
		if not lines:
			return None

		try:
			response = model.query(system=system, messages=[{"role": "user", "content": "\n".join(lines)}])
		except Exception:
			return None

		content = (response or {}).get("content", None)
		if not content or "NOTHING" in content:
			return None

		facts = [line.strip()[2:].strip() for line in content.splitlines() if line.strip().startswith("- ")]
		return facts or None

	return consolidate
