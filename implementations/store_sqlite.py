"""A persistent A2M server: one SQLite file, one store per tier.

	python -m implementations.store_sqlite memory.db              # stdio server
	python -m implementations.store_sqlite memory.db --http 8778  # http server

Every decision this server makes lives in [store.py](store.py) and is shared
verbatim with [store_postgres.py](store_postgres.py). What is left here is
storage: the tables, the dialect, the vector index. That is the claim the pair
is making, and the reason the tier logic is not in this file.

Where a2m/ keeps everything in a dict and server_minimal.py proves the
specification is implementable from the document alone, this is the shape you
would actually deploy: it survives a restart, and each tier is stored the way
its access pattern asks for rather than the way the tier above it happened to be.

	working      a plain table, replayed in insertion order, never embedded
	episodic     relational rows + a vec0 index, filtered *then* ranked
	semantic     the same rows, small enough that the index is a convenience
	procedural   files on disk, with the table holding only a pointer

The seam is `TierStore`. Each tier owns its own table and its own idea of what
reading means, which is what makes the mapping in spec/implementing-a2m.md
concrete instead of advisory. Swapping WorkingStore for Redis means implementing
that class, not rewriting the server.

Vector search uses sqlite-vec when the extension loads, and falls back to cosine
in Python when it does not. The fallback is not a lesser mode of the protocol --
spec 5.1 leaves ranking entirely to the implementation -- it just gets slower
sooner.
"""


import json
import os
import pathlib
import sqlite3
import sys


from   typing                import Callable


from   a2m                   import A2M_VERSION, MemoryServer, serve_a2m_http, serve_a2m_stdio
from   a2m.memory            import MemoryRecord, MemoryTier, recency
from   a2m.retrieval         import cosine, extractive_summarizer
from   implementations.store import TierStore, TieredMemoryStack, pack, single_tier, unpack



try:
	import sqlite_vec
	HAVE_VEC = True
except ImportError:
	HAVE_VEC = False


def to_record(row: sqlite3.Row, tier: str) -> MemoryRecord:
	"""Turn a database row into a MemoryRecord.

	Args:
		row (sqlite3.Row): A row from any of the tier tables.
		tier (str): Which tier it came from -- the row does not always carry it,
			since a table may serve only one tier.

	Returns:
		MemoryRecord: With accessed_at and access_count restored, so ranking and
		promotion survive a restart.
	"""
	record              = MemoryRecord(
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
		embedding  = unpack(row["embedding"]) if row["embedding"] else None,
		uri        = row["uri"],
		media_type = row["media_type"],
	)
	record.accessed_at  = row["accessed_at"]
	record.access_count = row["access_count"]
	return record


# `id` is a unique constraint rather than the primary key, because the tiers
# disagree about what their key should be: working is ordered by an autoincrement
# sequence, the others are content-addressed by id. INSERT OR IGNORE keys off the
# unique constraint either way, which is what makes writes idempotent.
COLUMNS = """
	id           TEXT NOT NULL UNIQUE,
	content      TEXT NOT NULL,
	role         TEXT,
	metadata     TEXT,
	grp          TEXT,
	owner        TEXT,
	session      TEXT,
	akey         TEXT,
	revision     INTEGER NOT NULL DEFAULT 0,
	embedding    BLOB,
	uri          TEXT,
	media_type   TEXT,
	salience     REAL    NOT NULL DEFAULT 1.0,
	created_at   REAL    NOT NULL,
	accessed_at  REAL    NOT NULL,
	access_count INTEGER NOT NULL DEFAULT 0
"""


class SqliteTierStore(TierStore):
	"""What every tier of this backend shares: SQLite's placeholder style.

	The tier logic in [store.py](store.py) cannot spell this condition, because
	the two SQL backends spell it differently -- `:owner` here, `%(owner)s` in
	PostgreSQL -- and a module with no dialect in it cannot pick one."""

	def visible(self, owner: str = None) -> str:
		"""The SQL fragment implementing spec 6 scoping for this tier.

		Args:
			owner (str, optional): The requesting agent.

		Returns:
			str: A fragment to append to a WHERE clause -- empty for a shared tier or
			an unscoped read, since neither restricts anything.
		"""
		if owner is None or self.tier.shared:
			return ""
		return " AND (owner IS NULL OR owner = :owner)"


class WorkingStore(SqliteTierStore):
	"""The live transcript.

	No embedding column and no vector index, deliberately. Working memory is
	replayed, not searched (spec/implementing-a2m.md §2), and it turns over fast
	enough that embedding on write pays for vectors nobody ever reads. `seq` is a
	monotonic counter because replay order must survive identical timestamps."""

	def create(self) -> None:
		"""Create the transcript table.

		No embedding column and no vector index, deliberately: working memory is
		replayed, not searched. 'seq' is a monotonic counter because replay order must
		survive identical timestamps.
		"""
		self.db.execute(f"""
			CREATE TABLE IF NOT EXISTS working (
				seq INTEGER PRIMARY KEY AUTOINCREMENT,
				{COLUMNS}
			)""")
		self.db.execute("CREATE INDEX IF NOT EXISTS working_owner ON working(owner, seq)")


	def add(self, record: MemoryRecord, embedding: list[float] = None) -> None:
		"""Store one transcript record. No embedding is taken -- see the class docstring.

		Args:
			record (MemoryRecord): The record to write.
			embedding (list[float], optional): Accepted and ignored.
		"""
		self.db.execute(
			"""INSERT OR IGNORE INTO working
			   (id, content, role, metadata, grp, owner, session, akey, revision,
			    embedding, uri, media_type, salience, created_at, accessed_at, access_count)
			   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
			(record.id, record.content, record.role, json.dumps(record.metadata), record.group,
			 record.owner, record.session, record.key, record.revision,
			 pack(record.embedding) if record.embedding else None, record.uri, record.media_type,
			 record.salience, record.created_at, record.accessed_at, record.access_count),
		)


	def rows(self, owner: str = None) -> list[MemoryRecord]:
		"""Every visible record in this tier, oldest first.

		Args:
			owner (str, optional): Scope the read (spec 6).

		Returns:
			list[MemoryRecord]: Every visible record in this tier, oldest first.

		Ordered by 'seq', so replay order is exact even within one millisecond.
		"""
		sql = "SELECT * FROM working WHERE 1=1" + self.visible(owner) + " ORDER BY seq"
		return [to_record(r, self.name) for r in self.db.execute(sql, {"owner": owner})]


	def delete(self, ids: list[str]) -> int:
		"""Remove records by id.

		Args:
			ids (list[str]): Record ids to remove.

		Returns:
			int: How many rows were actually deleted.
		"""
		if not ids:
			return 0
		marks  = ",".join("?" * len(ids))
		cursor = self.db.execute(f"DELETE FROM working WHERE id IN ({marks})", list(ids))
		return cursor.rowcount


	def count(self, owner: str = None) -> int:
		"""How many records are visible in this tier.

		Args:
			owner (str, optional): Scope the count.

		Returns:
			int: Visible record count in this tier.
		"""
		sql = "SELECT COUNT(*) FROM working WHERE 1=1" + self.visible(owner)
		return self.db.execute(sql, {"owner": owner}).fetchone()[0]


	def touch(self, ids: list[str], now: float) -> None:
		"""Record that these were just recalled, updating accessed_at and access_count.

		Args:
			ids (list[str]): Records that were returned by a search.
			now (float): Epoch seconds.
		"""
		if ids:
			marks = ",".join("?" * len(ids))
			self.db.execute(
				f"UPDATE working SET accessed_at=?, access_count=access_count+1 WHERE id IN ({marks})",
				[now, *ids],
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
		marks  = ",".join("?" * len(ids))
		cursor = self.db.execute(f"UPDATE working SET salience=salience+? WHERE id IN ({marks})", [amount, *ids])
		return cursor.rowcount


class DurableStore(SqliteTierStore):
	"""Episodic and semantic: relational rows carrying the structure recall needs
	to filter on, with a vector index carrying the meaning it ranks by.

	One table serves both tiers, discriminated by a column. That is not a
	shortcut: episodic and semantic differ in what *writes* them and how long they
	live, not in how they are read, and both reads are "filter by owner and
	metadata, then rank by similarity".

	Filtering happens inside the vec0 query rather than after it. `owner` is a
	partition key and `tier` an auxiliary column, so a scoped search never has to
	over-fetch and discard."""

	EMBEDS = True
	TABLE  = "durable"

	def __init__(self, db: sqlite3.Connection, tier: MemoryTier, vec: bool = False) -> None:
		"""Bind this store to a connection, its tier, and whether sqlite-vec loaded.

		Args:
			db (sqlite3.Connection): The shared connection.
			tier (MemoryTier): The tier this store serves.
			vec (bool, optional): Whether the sqlite-vec extension is available. When
				it is not, this store still stores vectors -- it just ranks them in
				Python instead of in an index.
		"""
		super().__init__(db, tier)
		self.vec     = vec
		self.indexed = False


	def create(self) -> None:
		"""Create the shared episodic/semantic table and its indexes.
		"""
		self.db.execute(f"""
			CREATE TABLE IF NOT EXISTS {self.TABLE} (
				{COLUMNS},
				tier      TEXT NOT NULL,
				seq       INTEGER
			)""")
		self.db.execute(f"CREATE INDEX IF NOT EXISTS durable_tier ON {self.TABLE}(tier, owner)")
		self.db.execute(f"CREATE INDEX IF NOT EXISTS durable_time ON {self.TABLE}(created_at)")


	def create_index(self, dimensions: int) -> None:
		"""Create the vec0 virtual table.

		Called lazily, because the index's dimensionality is not known until the first
		embedding arrives.

		Args:
			dimensions (int): The embedding width.
		"""
		self.db.execute(f"""
			CREATE VIRTUAL TABLE IF NOT EXISTS durable_vec USING vec0(
				owner TEXT partition key,
				tier  TEXT,
				id    TEXT,
				embedding float[{dimensions}]
			)""")


	def add(self, record: MemoryRecord, embedding: list[float] = None) -> None:
		"""Store one durable record, with its vector when there is one.

		Args:
			record (MemoryRecord): The record to write.
			embedding (list[float], optional): Its vector.
		"""
		self.db.execute(
			f"""INSERT OR IGNORE INTO {self.TABLE}
			    (id, content, role, metadata, grp, owner, session, akey, revision, embedding,
			     uri, media_type, salience, created_at, accessed_at, access_count, tier, seq)
			    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
			(record.id, record.content, record.role, json.dumps(record.metadata), record.group,
			 record.owner, record.session, record.key, record.revision,
			 pack(embedding or record.embedding) if (embedding or record.embedding) else None,
			 record.uri, record.media_type,
			 record.salience, record.created_at, record.accessed_at,
			 record.access_count, record.tier,
			 int(record.created_at * 1_000_000)),
		)

		self.index(record, embedding or record.embedding)


	def index(self, record: MemoryRecord, embedding: list[float] = None) -> None:
		"""Add a vector to the vec0 index, creating the index on first use.

		Lazily, because the index's width is not known until a vector arrives -- and
		nothing forces one to. A store that never sees an embedding never creates an
		index, which is what makes `embed=None` a working configuration rather than
		a degraded one.

		Args:
			record (MemoryRecord): The record just written.
			embedding (list[float], optional): Its vector.
		"""
		if not self.vec or not embedding:
			return

		if not self.indexed:
			self.create_index(len(embedding))
			self.indexed = True

		self.db.execute(
			"INSERT INTO durable_vec(owner, tier, id, embedding) VALUES (?,?,?,?)",
			(record.owner or "", self.name, record.id, pack(embedding)),
		)


	def rows(self, owner: str = None) -> list[MemoryRecord]:
		"""Every visible record in this tier, oldest first.

		Args:
			owner (str, optional): Scope the read (spec 6).

		Returns:
			list[MemoryRecord]: Every visible record in this tier, oldest first.
		"""
		sql = f"SELECT * FROM {self.TABLE} WHERE tier=:tier" + self.visible(owner) + " ORDER BY created_at, seq"
		return [to_record(r, self.name) for r in self.db.execute(sql, {"tier": self.name, "owner": owner})]


	def delete(self, ids: list[str]) -> int:
		"""Remove records by id.

		Args:
			ids (list[str]): Record ids to remove.

		Returns:
			int: How many rows were actually deleted.

		Also removes the vector, so the index cannot outlive the row.
		"""
		if not ids:
			return 0
		marks  = ",".join("?" * len(ids))
		cursor = self.db.execute(f"DELETE FROM {self.TABLE} WHERE id IN ({marks})", list(ids))
		try:
			self.db.execute(f"DELETE FROM durable_vec WHERE id IN ({marks})", list(ids))
		except sqlite3.OperationalError:
			pass
		return cursor.rowcount


	def count(self, owner: str = None) -> int:
		"""How many records are visible in this tier.

		Args:
			owner (str, optional): Scope the count.

		Returns:
			int: Visible record count in this tier.
		"""
		sql = f"SELECT COUNT(*) FROM {self.TABLE} WHERE tier=:tier" + self.visible(owner)
		return self.db.execute(sql, {"tier": self.name, "owner": owner}).fetchone()[0]


	def touch(self, ids: list[str], now: float) -> None:
		"""Record that these were just recalled, updating accessed_at and access_count.

		Args:
			ids (list[str]): Records that were returned by a search.
			now (float): Epoch seconds.
		"""
		if ids:
			marks = ",".join("?" * len(ids))
			self.db.execute(
				f"UPDATE {self.TABLE} SET accessed_at=?, access_count=access_count+1 WHERE id IN ({marks})",
				[now, *ids],
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
		marks  = ",".join("?" * len(ids))
		cursor = self.db.execute(f"UPDATE {self.TABLE} SET salience=salience+? WHERE id IN ({marks})", [amount, *ids])
		return cursor.rowcount


	def knn(self, embedding: list[float], owner: str, limit: int) -> dict[str, float] | None:
		"""Nearest neighbours, filtered *during* the search rather than after it.

		'owner' is a vec0 partition key and 'tier' an auxiliary column, so a scoped
		search never has to over-fetch and discard -- which is what a post-hoc join
		would force.

		Args:
			embedding (list[float]): The query vector.
			owner (str, optional): Scope.
			limit (int): How many neighbours to ask for.

		Returns:
			dict | None: Record id to cosine similarity, or None when there is no
			index to ask -- which tells the caller to rank another way rather than
			that nothing matched. An empty dict means the index was consulted and had
			nothing, and the two must not be confused (spec §5.1).
		"""
		if not self.vec:
			return None

		clauses = ["embedding MATCH ?", "k = ?", "tier = ?"]
		params  = [pack(embedding), max(limit, 1), self.name]

		if owner is not None and not self.tier.shared:
			clauses.append("owner = ?")
			params.append(owner)

		sql = f"SELECT id, distance FROM durable_vec WHERE {' AND '.join(clauses)} ORDER BY distance"

		try:
			rows = self.db.execute(sql, params).fetchall()
		except sqlite3.OperationalError:
			return {}

		# vec0 reports L2 distance over normalised vectors; 1 - d²/2 recovers cosine.
		return {r["id"]: max(0.0, 1.0 - (r["distance"] ** 2) / 2.0) for r in rows}


	def vectors(self, owner: str = None) -> dict[str, list[float]]:
		"""Every stored vector in this tier, for the no-extension fallback path.

		Args:
			owner (str, optional): Scope.

		Returns:
			dict[str, list[float]]: Record id to embedding.
		"""
		sql  = f"SELECT id, embedding FROM {self.TABLE} WHERE tier=:tier AND embedding IS NOT NULL"
		sql += self.visible(owner)
		return {r["id"]: unpack(r["embedding"]) for r in self.db.execute(sql, {"tier": self.name, "owner": owner})}


class ProceduralStore(SqliteTierStore):
	"""How to do things — kept as files, with the table holding only a pointer.

	A procedure is closer to code than to data: you want to read it, diff it and
	revert it, none of which a row in a table gives you. The directory sits beside
	the database file and can be put under version control on its own."""

	def __init__(self, db: sqlite3.Connection, tier: MemoryTier, root: pathlib.Path) -> None:
		"""Bind this store to a connection and the directory it writes to.

		Args:
			db (sqlite3.Connection): The shared connection, holding only pointers.
			tier (MemoryTier): The procedural tier.
			root (Path): Directory the procedure files live in. Created on demand,
				and safe to put under version control on its own.
		"""
		super().__init__(db, tier)
		self.root = root


	def create(self) -> None:
		"""Create the pointer table and the directory the procedures live in.
		"""
		self.db.execute(f"""
			CREATE TABLE IF NOT EXISTS procedural (
				{COLUMNS},
				path TEXT NOT NULL
			)""")
		self.root.mkdir(parents=True, exist_ok=True)


	def _path(self, record: MemoryRecord) -> pathlib.Path:
		"""Where a procedure's file belongs.

		Args:
			record (MemoryRecord): The record being written.

		Returns:
			Path: A readable filename derived from the skill name where there is one,
			suffixed with part of the id to stay unique.
		"""
		stem = "".join(c if c.isalnum() or c in "-_" else "-" for c in (record.metadata.get("skill") or record.id))
		return self.root / f"{stem}-{record.id[:8]}.md"


	def add(self, record: MemoryRecord, embedding: list[float] = None) -> None:
		"""Write the procedure to disk and record a pointer to it.

		Args:
			record (MemoryRecord): The record to write.
			embedding (list[float], optional): Accepted and ignored.
		"""
		path = self._path(record)
		path.write_text(record.content, encoding="utf-8")

		self.db.execute(
			"""INSERT OR IGNORE INTO procedural
			   (id, content, role, metadata, grp, owner, session, akey, revision, embedding,
			    uri, media_type, salience, created_at, accessed_at, access_count, path)
			   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
			(record.id, "", record.role, json.dumps(record.metadata), record.group, record.owner,
			 record.session, record.key, record.revision,
			 pack(record.embedding) if record.embedding else None,
			 record.uri, record.media_type, record.salience,
			 record.created_at, record.accessed_at, record.access_count, str(path)),
		)


	def rows(self, owner: str = None) -> list[MemoryRecord]:
		"""Every visible record in this tier, oldest first.

		Args:
			owner (str, optional): Scope the read (spec 6).

		Returns:
			list[MemoryRecord]: Every visible record in this tier, oldest first.

		Content is read back from disk, so editing a procedure in an editor -- or
		reverting it in git -- takes effect without touching the database.
		"""
		sql     = "SELECT * FROM procedural WHERE 1=1" + self.visible(owner) + " ORDER BY created_at"
		records = []

		for row in self.db.execute(sql, {"owner": owner}):
			record = to_record(row, self.name)
			try:
				record.content = pathlib.Path(row["path"]).read_text(encoding="utf-8")
			except OSError:
				record.content = ""
			records.append(record)

		return records


	def delete(self, ids: list[str]) -> int:
		"""Remove records by id.

		Args:
			ids (list[str]): Record ids to remove.

		Returns:
			int: How many rows were actually deleted.

		Removes the file as well as the row.
		"""
		if not ids:
			return 0

		marks = ",".join("?" * len(ids))
		for row in self.db.execute(f"SELECT path FROM procedural WHERE id IN ({marks})", list(ids)):
			try:
				os.remove(row["path"])
			except OSError:
				pass

		return self.db.execute(f"DELETE FROM procedural WHERE id IN ({marks})", list(ids)).rowcount


	def count(self, owner: str = None) -> int:
		"""How many records are visible in this tier.

		Args:
			owner (str, optional): Scope the count.

		Returns:
			int: Visible record count in this tier.
		"""
		sql = "SELECT COUNT(*) FROM procedural WHERE 1=1" + self.visible(owner)
		return self.db.execute(sql, {"owner": owner}).fetchone()[0]


	def touch(self, ids: list[str], now: float) -> None:
		"""Record that these were just recalled, updating accessed_at and access_count.

		Args:
			ids (list[str]): Records that were returned by a search.
			now (float): Epoch seconds.
		"""
		if ids:
			marks = ",".join("?" * len(ids))
			self.db.execute(
				f"UPDATE procedural SET accessed_at=?, access_count=access_count+1 WHERE id IN ({marks})",
				[now, *ids],
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
		marks = ",".join("?" * len(ids))
		return self.db.execute(f"UPDATE procedural SET salience=salience+? WHERE id IN ({marks})",
		                       [amount, *ids]).rowcount


class SqliteMemoryStack(TieredMemoryStack):
	"""The tier logic above, stored in one SQLite file."""

	def __init__(
		self,
		path           : str               = "memory.db",
		tiers          : list[MemoryTier]  = None,
		embed          : Callable          = None,
		consolidate_fn : Callable          = None,
		weights        : dict[str, float]  = None,
	) -> None:
		"""Open, and create anything missing.

		Args:
			path (str, optional): The database file. Created if absent, along with a
				sibling directory for procedural files.
			tiers (list[MemoryTier], optional): The layers. Defaults to the four
				canonical ones. Pass a single tier to serve one tier of a federation.
			embed (Callable, optional): A batched embedder. Without one, ranking is
				lexical only -- which is a degraded ranking, not a broken store.
			consolidate_fn (Callable, optional): Decides what spilling means.
			weights (dict, optional): How relevance, recency and salience combine.

		Raises:
			ValueError: If a tier spills into a tier that does not exist.

		Example:
			stack = SqliteMemoryStack("memory.db", embed=ollama_embedder())
			try:
				MemoryServer(stack)
			finally:
				stack.close()
		"""
		super().__init__(tiers=tiers, embed=embed, consolidate_fn=consolidate_fn, weights=weights)

		self.path = pathlib.Path(path)

		self.db = sqlite3.connect(self.path, check_same_thread=False)
		self.db.row_factory = sqlite3.Row
		self.db.execute("PRAGMA journal_mode=WAL")
		self.db.execute("PRAGMA synchronous=NORMAL")

		self.vec = False
		if HAVE_VEC:
			try:
				self.db.enable_load_extension(True)
				sqlite_vec.load(self.db)
				self.db.enable_load_extension(False)
				self.vec = True
			except Exception:
				self.vec = False

		for tier in self.tiers.values():
			if tier.kind == "working":
				store = WorkingStore(self.db, tier)
			elif tier.kind == "procedural":
				store = ProceduralStore(self.db, tier, self.root)
			else:
				store = DurableStore(self.db, tier, vec=self.vec)

			store.create()
			self.stores[tier.name] = store

		self.db.commit()
		self._check_spills()


	@property
	def SCORER(self) -> str:
		"""What ranked this store, for describe.

		Returns:
			str: The index actually in use, which is not the same as the one that
			was configured -- the extension may not have loaded.
		"""
		return "sqlite-vec" if self.vec else "cosine+lexical"


	@property
	def BACKEND(self) -> str:
		"""Where the rows actually are, for describe.

		Returns:
			str: The database path.
		"""
		return str(self.path)


	@property
	def root(self) -> pathlib.Path:
		"""The directory holding procedural files, beside the database.

		Returns:
			Path: '<name>.procedural' next to the database file, so it can be put
			under version control on its own.
		"""
		return self.path.parent / f"{self.path.stem}.procedural"


def open_stack(path: str = "memory.db", embed: Callable = None, **kwargs) -> SqliteMemoryStack:
	"""Open a durable memory stack backed by a SQLite file.

	Args:
		path (str, optional): The database file. Created if absent, along with the
			sibling directory for procedural files.
		embed (Callable, optional): A batched embedder. Without one, ranking is
			lexical only.
		**kwargs: Passed to SqliteMemoryStack -- tiers, consolidate_fn, weights.

	Returns:
		SqliteMemoryStack: Ready to serve.

	Example:
		from a2m.retrieval import ollama_embedder

		stack  = open_stack("memory.db", embed=ollama_embedder())
		server = MemoryServer(stack)
		serve_a2m_stdio(server)
	"""
	return SqliteMemoryStack(path=path, embed=embed, **kwargs)


def main() -> int:
	"""Run this file as an A2M server.

		python -m implementations.store_sqlite memory.db                  stdio, all four tiers
		python -m implementations.store_sqlite memory.db --http 8778      http
		python -m implementations.store_sqlite working.db --tier working  one tier, for a federation
		python -m implementations.store_sqlite memory.db --embed          with vector search

	Returns:
		int: Process exit code.
	"""
	argv = [a for a in sys.argv[1:]]
	path = next((a for a in argv if not a.startswith("--") and not a.isdigit()), "memory.db")

	embed = None
	if "--embed" in argv:
		from a2m.retrieval import ollama_embedder
		embed = ollama_embedder()

	tiers = None
	name  = f"a2m-sqlite({pathlib.Path(path).name})"
	if "--tier" in argv:
		kind  = argv[argv.index("--tier") + 1]
		tiers = single_tier(kind)
		name  = f"a2m-sqlite:{kind}"

	stack  = open_stack(path, embed=embed, tiers=tiers)
	# An extractive summarizer needs no model, so `summarize` is declared and
	# exercised here exactly as it is by the reference server (spec §4.15).
	server = MemoryServer(stack=stack, name=name, summarize_fn=extractive_summarizer())

	if "--http" in argv:
		index = argv.index("--http")
		port  = int(argv[index + 1]) if len(argv) > index + 1 else 8778
		print(f"A2M {A2M_VERSION} on http://127.0.0.1:{port}/ backed by {path}", file=sys.stderr)
		serve_a2m_http(server, port=port).serve_forever()
	else:
		serve_a2m_stdio(server)

	return 0


if __name__ == "__main__":
	sys.exit(main())
