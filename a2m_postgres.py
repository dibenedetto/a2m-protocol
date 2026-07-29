"""A persistent A2M server on PostgreSQL, with pgvector for search.

	python a2m_postgres.py postgresql://a2m:a2m@127.0.0.1:55432/a2m
	python a2m_postgres.py postgresql://... --http 8778
	python a2m_postgres.py postgresql://... --tier working   # one tier of a federation

	pip install "psycopg[binary]"        # and a server with CREATE EXTENSION vector

This exists to test a claim [a2m_store.py](a2m_store.py) makes in its own
docstring: that swapping a tier's storage means implementing `TierStore`, not
rewriting the server. It reuses `TieredMemoryStack` **unchanged** — every rule
about spilling, promotion, sessions and blending relevance with recency and
salience is the same object — and replaces only what touches rows.

What the port cost, precisely:

	three TierStore subclasses      SQL dialect and the vector index
	one TieredMemoryStack subclass  opening a connection, building the stores

and nothing else. Not one line of tier logic, and nothing at all in a2m.py, which
does not know either engine exists.

The differences worth naming, because they are where a port like this usually
goes wrong rather than where it is tedious:

- **`INSERT OR IGNORE` becomes `ON CONFLICT (id) DO NOTHING`.** That is not a
  dialect detail: it is what makes a client-supplied id idempotent (spec §4.2),
  the only retry-safety in the protocol. Getting it wrong turns a network retry
  into a duplicate.
- **pgvector's `<=>` is cosine *distance*.** Similarity is `1 - distance`.
  sqlite-vec reports L2 over normalised vectors instead, which needs
  `1 - d²/2`. Two engines, two conversions, and both must land in `0..1` or the
  blend with recency and salience quietly changes meaning.
- **The vector column has a fixed width, set at DDL time.** So the index is
  created lazily on the first embedding, exactly as the sqlite side does, and a
  store configured without an embedder never creates one at all.
- **A caller's vector is stored verbatim in every tier** (spec §3.7), including
  working memory, which has no index. Storing is not indexing, and conflating
  them is how the first version of the SQLite store silently dropped vectors in
  three tiers out of four (DECISION 018).
"""


import json
import pathlib
import sys
import threading
import time
import uuid


from   typing    import Any, Callable


import psycopg


from   psycopg.rows import dict_row


from   a2m       import A2M_VERSION, MemoryServer, serve_a2m_http, serve_stdio
from   a2m_store import COLUMNS, TieredMemoryStack, TierStore, pack, to_record, unpack
from   memory    import KINDS, MemoryRecord, MemoryTier, default_tiers


# The same columns as the SQLite store, in PostgreSQL's types. `id` is UNIQUE
# rather than the primary key for the same reason it is there: working orders by
# an autoincrement sequence and the durable tiers are content-addressed by id, so
# ON CONFLICT keys off the unique constraint either way.
PG_COLUMNS = """
	id           TEXT NOT NULL UNIQUE,
	content      TEXT NOT NULL,
	role         TEXT,
	metadata     TEXT,
	grp          TEXT,
	owner        TEXT,
	session      TEXT,
	akey         TEXT,
	revision     INTEGER NOT NULL DEFAULT 0,
	embedding    BYTEA,
	uri          TEXT,
	media_type   TEXT,
	salience     DOUBLE PRECISION NOT NULL DEFAULT 1.0,
	created_at   DOUBLE PRECISION NOT NULL,
	accessed_at  DOUBLE PRECISION NOT NULL,
	access_count INTEGER NOT NULL DEFAULT 0
"""


def to_pg_record(row: dict, tier: str) -> MemoryRecord:
	"""Turn a PostgreSQL row into a MemoryRecord.

	The SQLite version of this takes an sqlite3.Row; with psycopg's dict_row the
	subscripting is identical, so the only real difference is that bytea arrives
	as memoryview and struct wants bytes.

	Args:
		row (dict): A row from any of the tier tables.
		tier (str): Which tier it came from.

	Returns:
		MemoryRecord: With accessed_at and access_count restored, so ranking and
		promotion survive a restart.
	"""
	blob   = row["embedding"]
	record = MemoryRecord(
		content    = row["content"],
		tier       = tier,
		role       = row["role"],
		salience   = row["salience"],
		group      = row["grp"],
		metadata   = json.loads(row["metadata"]) if row["metadata"] else {},
		id         = row["id"],
		created_at = row["created_at"],
		owner      = row["owner"],
		session    = row["session"],
		key        = row["akey"],
		revision   = row["revision"],
		embedding  = unpack(bytes(blob)) if blob else None,
		uri        = row["uri"],
		media_type = row["media_type"],
	)
	record.accessed_at  = row["accessed_at"]
	record.access_count = row["access_count"]
	return record


class PgTierStore(TierStore):
	"""Shared PostgreSQL plumbing for the three tier stores."""

	def execute(self, sql: str, params=None):
		"""Run one statement and return its cursor.

		Args:
			sql (str): The statement.
			params: Sequence or mapping of parameters.

		Returns:
			psycopg.Cursor: For rowcount or fetching.
		"""
		cursor = self.db.cursor(row_factory=dict_row)
		cursor.execute(sql, params)
		return cursor


	def visible(self, owner: str = None) -> str:
		"""The SQL fragment implementing spec §6 scoping, in psycopg's placeholder style.

		Args:
			owner (str, optional): The requesting agent.

		Returns:
			str: A fragment to append to a WHERE clause -- empty for a shared tier or
			an unscoped read, since neither restricts anything.
		"""
		if owner is None or self.tier.shared:
			return ""
		return " AND (owner IS NULL OR owner = %(owner)s)"


	def delete(self, ids: list[str]) -> int:
		"""Remove records by id.

		Args:
			ids (list[str]): Record ids to remove.

		Returns:
			int: How many rows were actually deleted.
		"""
		if not ids:
			return 0

		return self.execute(f"DELETE FROM {self.TABLE} WHERE id = ANY(%s)", (list(ids),)).rowcount


	def count(self, owner: str = None) -> int:
		"""How many records are visible in this tier.

		Args:
			owner (str, optional): Scope the count.

		Returns:
			int: Visible record count in this tier.
		"""
		sql = f"SELECT COUNT(*) AS n FROM {self.TABLE} WHERE {self.SCOPE}" + self.visible(owner)
		return self.execute(sql, {"tier": self.name, "owner": owner}).fetchone()["n"]


	def touch(self, ids: list[str], now: float) -> None:
		"""Record that these were just recalled.

		Args:
			ids (list[str]): Records returned by a search.
			now (float): Epoch seconds.
		"""
		if not ids:
			return

		self.execute(
			f"UPDATE {self.TABLE} SET accessed_at = %s, access_count = access_count + 1 WHERE id = ANY(%s)",
			(now, list(ids)),
		)


	def bump(self, ids: list[str], amount: float) -> int:
		"""Add to the salience of these records.

		Args:
			ids (list[str]): Records to reinforce.
			amount (float): How much to add.

		Returns:
			int: How many rows were updated.
		"""
		if not ids:
			return 0

		return self.execute(
			f"UPDATE {self.TABLE} SET salience = salience + %s WHERE id = ANY(%s)",
			(amount, list(ids)),
		).rowcount


class PgWorkingStore(PgTierStore):
	"""The live transcript, in PostgreSQL.

	No vector index, deliberately: working memory is replayed rather than
	searched. A caller's own vector is still *stored*, because it is the caller's
	and the store does not get to drop it (spec §3.7)."""

	TABLE = "working"
	SCOPE = "TRUE"

	def create(self) -> None:
		"""Create the transcript table.
		"""
		self.execute(f"""
			CREATE TABLE IF NOT EXISTS working (
				seq BIGSERIAL PRIMARY KEY,
				{PG_COLUMNS}
			)""")
		self.execute("CREATE INDEX IF NOT EXISTS working_owner ON working(owner, seq)")


	def add(self, record: MemoryRecord, embedding: list[float] = None) -> None:
		"""Store one transcript record. No embedding is generated -- see the class docstring.

		Args:
			record (MemoryRecord): The record to write.
			embedding (list[float], optional): Accepted and ignored.
		"""
		self.execute(
			"""INSERT INTO working
			   (id, content, role, metadata, grp, owner, session, akey, revision,
			    embedding, uri, media_type, salience, created_at, accessed_at, access_count)
			   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
			   ON CONFLICT (id) DO NOTHING""",
			(record.id, record.content, record.role, json.dumps(record.metadata), record.group,
			 record.owner, record.session, record.key, record.revision,
			 pack(record.embedding) if record.embedding else None, record.uri, record.media_type,
			 record.salience, record.created_at, record.accessed_at, record.access_count),
		)


	def rows(self, owner: str = None) -> list[MemoryRecord]:
		"""Every visible record in this tier, oldest first.

		Ordered by 'seq', so replay order is exact even within one millisecond.

		Args:
			owner (str, optional): Scope the read (spec §6).

		Returns:
			list[MemoryRecord]: Oldest first.
		"""
		sql = "SELECT * FROM working WHERE TRUE" + self.visible(owner) + " ORDER BY seq"
		return [to_pg_record(r, self.name) for r in self.execute(sql, {"owner": owner}).fetchall()]


class PgDurableStore(PgTierStore):
	"""Episodic and semantic: relational rows, ranked by a pgvector index.

	One table serves both tiers, discriminated by a column, for the same reason the
	SQLite store does it: they differ in what writes them and how long they live,
	not in how they are read."""

	TABLE = "durable"
	SCOPE = "tier = %(tier)s"

	def __init__(self, db, tier: MemoryTier) -> None:
		"""Bind this store to a connection and its tier.

		Args:
			db (psycopg.Connection): The shared connection.
			tier (MemoryTier): The tier this store serves.
		"""
		super().__init__(db, tier)
		self.indexed = False


	def create(self) -> None:
		"""Create the shared episodic/semantic table and its indexes.
		"""
		self.execute(f"""
			CREATE TABLE IF NOT EXISTS durable (
				{PG_COLUMNS},
				tier TEXT NOT NULL,
				seq  BIGSERIAL
			)""")
		self.execute("CREATE INDEX IF NOT EXISTS durable_tier ON durable(tier, owner)")
		self.execute("CREATE INDEX IF NOT EXISTS durable_time ON durable(created_at)")

		# The vector table may already exist from an earlier run, in which case the
		# width is settled and nothing has to be created lazily.
		exists = self.execute(
			"SELECT to_regclass('durable_vec') IS NOT NULL AS present"
		).fetchone()["present"]
		self.indexed = bool(exists)


	def create_index(self, dimensions: int) -> None:
		"""Create the pgvector table and its ANN index.

		Called lazily, because a vector column's width is fixed at DDL time and is
		not known until the first embedding arrives -- and nothing forces one to.

		Args:
			dimensions (int): The embedding width.
		"""
		self.execute("CREATE EXTENSION IF NOT EXISTS vector")
		self.execute(f"""
			CREATE TABLE IF NOT EXISTS durable_vec (
				id        TEXT PRIMARY KEY,
				owner     TEXT,
				tier      TEXT NOT NULL,
				embedding vector({dimensions}) NOT NULL
			)""")
		self.execute("CREATE INDEX IF NOT EXISTS durable_vec_tier ON durable_vec(tier, owner)")
		self.execute("""
			CREATE INDEX IF NOT EXISTS durable_vec_ann
			ON durable_vec USING hnsw (embedding vector_cosine_ops)""")


	def add(self, record: MemoryRecord, embedding: list[float] = None) -> None:
		"""Store one durable record, with its vector when there is one.

		Args:
			record (MemoryRecord): The record to write.
			embedding (list[float], optional): Its vector.
		"""
		vector = embedding or record.embedding

		self.execute(
			"""INSERT INTO durable
			   (id, content, role, metadata, grp, owner, session, akey, revision, embedding,
			    uri, media_type, salience, created_at, accessed_at, access_count, tier)
			   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
			   ON CONFLICT (id) DO NOTHING""",
			(record.id, record.content, record.role, json.dumps(record.metadata), record.group,
			 record.owner, record.session, record.key, record.revision,
			 pack(vector) if vector else None, record.uri, record.media_type,
			 record.salience, record.created_at, record.accessed_at, record.access_count,
			 record.tier),
		)

		self.index(record, vector)


	def index(self, record: MemoryRecord, embedding: list[float] = None) -> None:
		"""Add a vector to the pgvector table, creating it on first use.

		Args:
			record (MemoryRecord): The record just written.
			embedding (list[float], optional): Its vector.
		"""
		if not embedding:
			return

		if not self.indexed:
			self.create_index(len(embedding))
			self.indexed = True

		# A width that disagrees with the column is refused by PostgreSQL. That is
		# the same answer spec §3.7 requires of the protocol -- a cosine between a
		# 768- and a 1024-dimensional vector is not a worse answer, it is not an
		# answer -- so letting the database enforce it is correct, not lazy.
		try:
			self.execute(
				"""INSERT INTO durable_vec (id, owner, tier, embedding)
				   VALUES (%s,%s,%s,%s)
				   ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding""",
				(record.id, record.owner or "", self.name, str(list(embedding))),
			)
		except psycopg.errors.DataException:
			self.db.rollback()


	def rows(self, owner: str = None) -> list[MemoryRecord]:
		"""Every visible record in this tier, oldest first.

		Args:
			owner (str, optional): Scope the read (spec §6).

		Returns:
			list[MemoryRecord]: Oldest first.
		"""
		sql = "SELECT * FROM durable WHERE tier = %(tier)s" + self.visible(owner) + " ORDER BY created_at, seq"
		rows = self.execute(sql, {"tier": self.name, "owner": owner}).fetchall()
		return [to_pg_record(r, self.name) for r in rows]


	def delete(self, ids: list[str]) -> int:
		"""Remove records and their vectors.

		Args:
			ids (list[str]): Record ids to remove.

		Returns:
			int: How many rows were actually deleted.
		"""
		if not ids:
			return 0

		if self.indexed:
			self.execute("DELETE FROM durable_vec WHERE id = ANY(%s)", (list(ids),))

		return self.execute("DELETE FROM durable WHERE id = ANY(%s)", (list(ids),)).rowcount


	def knn(self, embedding: list[float], owner: str, limit: int) -> dict[str, float] | None:
		"""Nearest neighbours from the pgvector index, filtered during the search.

		Args:
			embedding (list[float]): The query vector.
			owner (str, optional): Scope.
			limit (int): How many neighbours to ask for.

		Returns:
			dict | None: Record id to cosine similarity, or None when no index
			exists yet -- which means "rank these another way", not "nothing
			matched".
		"""
		if not self.indexed:
			return None

		clauses = ["tier = %(tier)s"]
		params  = {"tier": self.name, "owner": owner, "vector": str(list(embedding)), "k": max(limit, 1)}

		if owner is not None and not self.tier.shared:
			clauses.append("(owner IS NULL OR owner = %(owner)s OR owner = '')")

		sql = f"""
			SELECT id, embedding <=> %(vector)s AS distance
			FROM   durable_vec
			WHERE  {' AND '.join(clauses)}
			ORDER  BY distance
			LIMIT  %(k)s"""

		try:
			rows = self.execute(sql, params).fetchall()
		except psycopg.Error:
			self.db.rollback()
			return None

		# pgvector's <=> is cosine *distance*, so similarity is 1 - it. sqlite-vec
		# reports L2 over normalised vectors instead and needs 1 - d²/2. Both have
		# to land in 0..1 or the blend with recency and salience changes meaning.
		return {r["id"]: max(0.0, 1.0 - float(r["distance"])) for r in rows}


	def vectors(self, owner: str = None) -> dict[str, list[float]]:
		"""Every stored vector in this tier, for ranking without an index.

		Args:
			owner (str, optional): Scope the read.

		Returns:
			dict: id -> vector.
		"""
		sql  = "SELECT id, embedding FROM durable WHERE tier = %(tier)s AND embedding IS NOT NULL" + self.visible(owner)
		rows = self.execute(sql, {"tier": self.name, "owner": owner}).fetchall()
		return {r["id"]: unpack(bytes(r["embedding"])) for r in rows}


class PgProceduralStore(PgTierStore):
	"""Procedural memory: documents in the database, not files on disk.

	The SQLite store keeps these as files beside the database, so they can be
	reviewed and version-controlled like the code they describe. A server
	reachable over a network has no such disk to share, so here the text lives in
	the row. The tier's *meaning* is unchanged -- deliberately written, never
	spilled into (spec §3.1) -- which is the part the protocol cares about."""

	TABLE = "procedural"
	SCOPE = "TRUE"

	def create(self) -> None:
		"""Create the procedural table.
		"""
		self.execute(f"""
			CREATE TABLE IF NOT EXISTS procedural (
				seq BIGSERIAL PRIMARY KEY,
				{PG_COLUMNS}
			)""")
		self.execute("CREATE INDEX IF NOT EXISTS procedural_owner ON procedural(owner, seq)")


	def add(self, record: MemoryRecord, embedding: list[float] = None) -> None:
		"""Store one procedure.

		Args:
			record (MemoryRecord): The record to write.
			embedding (list[float], optional): Stored verbatim if present.
		"""
		vector = embedding or record.embedding

		self.execute(
			"""INSERT INTO procedural
			   (id, content, role, metadata, grp, owner, session, akey, revision,
			    embedding, uri, media_type, salience, created_at, accessed_at, access_count)
			   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
			   ON CONFLICT (id) DO NOTHING""",
			(record.id, record.content, record.role, json.dumps(record.metadata), record.group,
			 record.owner, record.session, record.key, record.revision,
			 pack(vector) if vector else None, record.uri, record.media_type,
			 record.salience, record.created_at, record.accessed_at, record.access_count),
		)


	def rows(self, owner: str = None) -> list[MemoryRecord]:
		"""Every visible procedure, oldest first.

		Args:
			owner (str, optional): Scope the read.

		Returns:
			list[MemoryRecord]: Oldest first.
		"""
		sql = "SELECT * FROM procedural WHERE TRUE" + self.visible(owner) + " ORDER BY seq"
		return [to_pg_record(r, self.name) for r in self.execute(sql, {"owner": owner}).fetchall()]


class PostgresMemoryStack(TieredMemoryStack):
	"""The tier logic of a2m_store.py, stored in PostgreSQL.

	Every method that decides anything -- remember, recall, consolidate, promote,
	sessions -- is inherited unchanged. Only the storage differs, which is the
	whole claim being tested."""

	SCORER = "pgvector"

	def __init__(
		self,
		dsn            : str               = "postgresql:///a2m",
		tiers          : list[MemoryTier]  = None,
		embed          : Callable          = None,
		consolidate_fn : Callable          = None,
		weights        : dict[str, float]  = None,
	) -> None:
		"""Connect, and create anything missing.

		Args:
			dsn (str, optional): A libpq connection string.
			tiers (list[MemoryTier], optional): The layers. Defaults to the four
				canonical ones. Pass a single tier to serve one tier of a federation.
			embed (Callable, optional): A batched embedder. Without one, ranking is
				lexical only.
			consolidate_fn (Callable, optional): Decides what spilling means.
			weights (dict, optional): How relevance, recency and salience combine.

		Raises:
			ValueError: If a tier spills into a tier that does not exist.
			psycopg.Error: If the server cannot be reached.

		Example:
			stack = PostgresMemoryStack("postgresql://a2m:a2m@127.0.0.1:5432/a2m")
			try:
				MemoryServer(stack)
			finally:
				stack.close()
		"""
		super().__init__(tiers=tiers, embed=embed, consolidate_fn=consolidate_fn, weights=weights)

		self.dsn = dsn
		self.db  = psycopg.connect(dsn, autocommit=False, row_factory=dict_row)

		for tier in self.tiers.values():
			if tier.kind == "working":
				store = PgWorkingStore(self.db, tier)
			elif tier.kind == "procedural":
				store = PgProceduralStore(self.db, tier)
			else:
				store = PgDurableStore(self.db, tier)

			store.create()
			self.stores[tier.name] = store

		self.db.commit()
		self._check_spills()


	@property
	def BACKEND(self) -> str:
		"""Where the rows actually are, for describe.

		Returns:
			str: The database name, without the credentials in the DSN.
		"""
		return self.dsn.rsplit("/", 1)[-1].split("?")[0] or "postgres"


	def close(self) -> None:
		"""Commit and close the connection.
		"""
		with self._lock:
			self.db.commit()
			self.db.close()


def open_stack(dsn: str, embed: Callable = None, **kwargs) -> PostgresMemoryStack:
	"""Open a durable memory stack backed by PostgreSQL.

	Args:
		dsn (str): A libpq connection string.
		embed (Callable, optional): A batched embedder.
		**kwargs: Passed through to PostgresMemoryStack.

	Returns:
		PostgresMemoryStack: Ready to serve.
	"""
	return PostgresMemoryStack(dsn, embed=embed, **kwargs)


def single_tier(kind: str) -> list[MemoryTier]:
	"""One tier, for a process serving one layer of a federation.

	Args:
		kind (str): working, episodic, semantic or procedural.

	Returns:
		list[MemoryTier]: Exactly one tier, spilling nowhere.

	Raises:
		ValueError: If the kind is not one of the four.
	"""
	if kind not in KINDS:
		raise ValueError(f"Unknown kind '{kind}'; expected one of {KINDS}")

	for tier in default_tiers():
		if tier.kind == kind:
			tier.spill_to   = None
			tier.promote_to = None
			return [tier]

	raise ValueError(f"No default tier of kind '{kind}'")


def main() -> int:
	"""Run this file as an A2M server on PostgreSQL.

		python a2m_postgres.py postgresql://a2m:a2m@127.0.0.1:55432/a2m
		python a2m_postgres.py postgresql://... --http 8778
		python a2m_postgres.py postgresql://... --tier working
		python a2m_postgres.py postgresql://... --embed

	Returns:
		int: Process exit code.
	"""
	argv = sys.argv[1:]
	dsn  = next((a for a in argv if not a.startswith("--") and not a.isdigit()), None)

	if not dsn:
		print(__doc__)
		return 2

	embed = None
	if "--embed" in argv:
		from retrieval import ollama_embedder
		embed = ollama_embedder()

	tiers = None
	name  = "a2m-postgres"
	if "--tier" in argv:
		kind  = argv[argv.index("--tier") + 1]
		tiers = single_tier(kind)
		name  = f"a2m-postgres:{kind}"

	stack  = open_stack(dsn, embed=embed, tiers=tiers)
	server = MemoryServer(stack=stack, name=name)

	try:
		if "--http" in argv:
			index = argv.index("--http")
			port  = int(argv[index + 1]) if len(argv) > index + 1 else 8778
			print(f"A2M {A2M_VERSION} on http://127.0.0.1:{port}/ backed by postgres", file=sys.stderr)
			serve_a2m_http(server, port=port).serve_forever()
		else:
			serve_stdio(server.dispatcher)
	finally:
		stack.close()

	return 0


if __name__ == "__main__":
	sys.exit(main())
