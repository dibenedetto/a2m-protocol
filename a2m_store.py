"""A persistent A2M server: one SQLite file, one store per tier.

	python a2m_store.py memory.db              # stdio server
	python a2m_store.py memory.db --http 8778  # http server

This is the sample implementation of the whole memory stack. Where a2m.py keeps
everything in a dict and a2m_minimal.py proves the specification is implementable
from the document alone, this one is the shape you would actually deploy: it
survives a restart, and each tier is stored the way its access pattern asks for
rather than the way the tier above it happened to be.

	working      a plain table, replayed in insertion order, never embedded
	episodic     relational rows + a vec0 index, filtered *then* ranked
	semantic     the same rows, small enough that the index is a convenience
	procedural   files on disk, with the table holding only a pointer

The seam is `TierStore`. Each tier owns its own table and its own idea of what
reading means, which is what makes the mapping in spec/implementing-a2m.md
concrete instead of advisory. Swapping WorkingStore for Redis or DurableStore for
Postgres means implementing that class, not rewriting the server.

Vector search uses sqlite-vec when the extension loads, and falls back to cosine
in Python when it does not. The fallback is not a lesser mode of the protocol —
spec §5.1 leaves ranking entirely to the implementation — it just gets slower
sooner.
"""


import json
import os
import pathlib
import sqlite3
import struct
import sys
import threading
import time
import uuid


from   typing    import Any, Callable


from   a2m       import A2M_VERSION, MemoryServer, serve_a2m_http, serve_stdio
from   memory    import KINDS, MemoryRecord, MemoryTier, default_tiers, recency
from   retrieval import LexicalScorer, cosine


try:
	import sqlite_vec
	HAVE_VEC = True
except ImportError:
	HAVE_VEC = False


WEIGHTS = {"lexical": 0.60, "recency": 0.25, "salience": 0.15}


def pack(vector: list[float]) -> bytes:
	"""Pack a vector into the float32 blob sqlite-vec expects.

	Args:
		vector (list[float]): The embedding.

	Returns:
		bytes: Little-endian float32.
	"""
	return struct.pack(f"{len(vector)}f", *vector)


def unpack(blob: bytes) -> list[float]:
	"""Unpack a float32 blob back into a vector.

	Args:
		blob (bytes): As written by 'pack'.

	Returns:
		list[float]: The embedding.
	"""
	return list(struct.unpack(f"{len(blob) // 4}f", blob))


class TierStore:
	"""One tier's storage. The seam the whole file exists to demonstrate.

	Subclasses differ in what they index and what "read" means, because the tiers
	differ in how they are read: working is replayed whole, the durable tiers are
	searched, procedural is a handful of documents."""

	def __init__(self, db: sqlite3.Connection, tier: MemoryTier) -> None:
		"""Bind this store to a connection and the tier it serves.

		Args:
			db (sqlite3.Connection): The shared connection. Every tier lives in one
				file, so a spill between tiers is a single transaction.
			tier (MemoryTier): The tier this store serves, carrying its capacity,
				sharing and per-session policy.
		"""
		self.db   = db
		self.tier = tier
		self.name = tier.name


	def create(self) -> None:
		"""Create this tier's tables and indexes. Idempotent.
		"""
		raise NotImplementedError


	def add(self, record: MemoryRecord, embedding: list[float] = None) -> None:
		"""Store one record.

		Args:
			record (MemoryRecord): The record to write.
			embedding (list[float], optional): Its vector, for tiers that are searched.
				Ignored by tiers that are replayed rather than searched.
		"""
		raise NotImplementedError


	def rows(self, owner: str = None) -> list[MemoryRecord]:
		"""Every visible record in this tier, oldest first.

		Args:
			owner (str, optional): Scope the read (spec 6).

		Returns:
			list[MemoryRecord]: Every visible record in this tier, oldest first.
		"""
		raise NotImplementedError


	def delete(self, ids: list[str]) -> int:
		"""Remove records by id.

		Args:
			ids (list[str]): Record ids to remove.

		Returns:
			int: How many rows were actually deleted.
		"""
		raise NotImplementedError


	def count(self, owner: str = None) -> int:
		"""How many records are visible in this tier.

		Args:
			owner (str, optional): Scope the count.

		Returns:
			int: Visible record count in this tier.
		"""
		raise NotImplementedError


	def index(self, record: MemoryRecord, embedding: list[float] = None) -> None:
		"""Put a record's vector into whatever ANN index this tier keeps.

		The default does nothing, which is the honest answer for a tier that is
		replayed rather than searched. Keeping this on the store rather than in the
		stack is what lets a different engine bring a different index without the
		tier logic above it knowing.

		Args:
			record (MemoryRecord): The record just written.
			embedding (list[float], optional): Its vector, if it has one.
		"""
		pass


	def knn(self, embedding: list[float], owner: str, limit: int) -> dict[str, float] | None:
		"""Rank this tier against a vector, using an index if there is one.

		Returns:
			dict | None: id -> similarity, or None when this tier has no vector
			index. None means "rank these some other way", not "nothing matched" --
			the same abstain/empty distinction the Scorer seam draws (spec §5.1).
		"""
		return None


	def vectors(self, owner: str = None) -> dict[str, list[float]]:
		"""Every stored vector in this tier, for ranking without an index.

		Returns:
			dict: id -> vector. Empty when this tier stores none.
		"""
		return {}


	def touch(self, ids: list[str], now: float) -> None:
		"""Record that these were just recalled, updating accessed_at and access_count.

		Args:
			ids (list[str]): Records that were returned by a search.
			now (float): Epoch seconds.
		"""
		pass


	def bump(self, ids: list[str], amount: float) -> int:
		"""Add to the salience of these records.

		Args:
			ids (list[str]): Records to reinforce.
			amount (float): How much to add.

		Returns:
			int: How many rows were updated.
		"""
		return 0


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


class WorkingStore(TierStore):
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


class DurableStore(TierStore):
	"""Episodic and semantic: relational rows carrying the structure recall needs
	to filter on, with a vector index carrying the meaning it ranks by.

	One table serves both tiers, discriminated by a column. That is not a
	shortcut: episodic and semantic differ in what *writes* them and how long they
	live, not in how they are read, and both reads are "filter by owner and
	metadata, then rank by similarity".

	Filtering happens inside the vec0 query rather than after it. `owner` is a
	partition key and `tier` an auxiliary column, so a scoped search never has to
	over-fetch and discard."""

	TABLE = "durable"

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


class ProceduralStore(TierStore):
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


class TieredMemoryStack:
	"""The tier logic, with no opinion about what stores the rows.

	Everything A2M's tier model actually *is* lives here: spilling under pressure,
	promotion by reinforcement, session lifecycle, the blend of relevance with
	recency and salience. None of it touches SQL. It drives `TierStore` instances
	and asks them to store, read, delete and rank.

	That is the seam this file exists to demonstrate, and the proof it is real is
	[a2m_postgres.py](a2m_postgres.py): a completely different engine reuses this
	class unchanged and implements only the stores beneath it.

	Nothing in a2m.py knows this class exists either. The server was written
	against a dict-backed stack and drives this one unmodified — which is the
	argument for having had a protocol boundary in the first place."""

	# What describe reports about this engine. Subclasses say what actually ranked
	# and what actually stored, which is not always what was configured -- an
	# extension may not have loaded.
	SCORER  = "cosine+lexical"
	BACKEND = "memory"

	def __init__(
		self,
		tiers          : list[MemoryTier]  = None,
		embed          : Callable          = None,
		consolidate_fn : Callable          = None,
		weights        : dict[str, float]  = None,
	) -> None:
		"""Set up the tier policy. Subclasses open the storage and build the stores.

		Args:
			tiers (list[MemoryTier], optional): The layers. Defaults to the four
				canonical ones. Pass a single tier to serve one tier of a federation.
			embed (Callable, optional): A batched embedder. Without one, ranking is
				lexical only -- which is a degraded ranking, not a broken store.
			consolidate_fn (Callable, optional): Decides what spilling means.
			weights (dict, optional): How relevance, recency and salience combine.
		"""
		tiers = tiers or default_tiers()

		self.tiers          = {t.name: t for t in tiers}
		self.order          = [t.name for t in tiers]
		self.embed          = embed
		self.consolidate_fn = consolidate_fn
		self.lexical        = LexicalScorer()
		self.weights        = dict(WEIGHTS)
		self.dimensions     = None

		if weights:
			self.weights.update(weights)

		self._lock          = threading.RLock()
		self._consolidating = threading.Lock()

		self.stores: dict[str, TierStore] = {}


	def _check_spills(self) -> None:
		"""Refuse a tier configuration that spills into nowhere.

		Raises:
			ValueError: If a tier spills into a tier that does not exist.
		"""
		for name in self.order:
			if tier_spill := self.tiers[name].spill_to:
				if tier_spill not in self.tiers:
					raise ValueError(f"Tier '{name}' spills into unknown tier '{tier_spill}'")



	@property
	def working(self) -> str:
		"""The tier holding the live transcript, found by kind.

		Returns:
			str: The transcript tier's name.
		"""
		found = self.of_kind("working")
		return found[0] if found else self.order[0]


	def of_kind(self, kind: str) -> list[str]:
		"""Every tier serving a given purpose.

		Args:
			kind (str): working, episodic, semantic or procedural.

		Returns:
			list[str]: Tier names in stack order.
		"""
		return [n for n in self.order if self.tiers[n].kind == kind]


	def tier(self, name: str) -> MemoryTier:
		"""Look up a tier by name.

		Args:
			name (str): The tier name.

		Returns:
			MemoryTier: The configured tier.

		Raises:
			KeyError: If no such tier exists.
		"""
		found = self.tiers.get(name, None)
		if found is None:
			raise KeyError(f"Unknown tier '{name}'")
		return found


	def _embed(self, texts: list[str]) -> list[list[float]] | None:
		"""Embed texts, creating the vector index on first use.

		Args:
			texts (list[str]): Texts to embed.

		Returns:
			list[list[float]] | None: Vectors, or None when no embedder is configured
			or the embedder failed -- ranking then falls back to lexical rather than
			the write failing.
		"""
		if not self.embed or not texts:
			return None
		try:
			vectors = self.embed(texts)
		except Exception:
			return None

		if vectors and self.dimensions is None:
			self.dimensions = len(vectors[0])

		return vectors


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
		"""Write one record to the store that owns its tier.

		Working memory is never embedded: it is replayed, not searched, and most of it
		is evicted having never been read by a query.

		Args:
			content (str): The text to remember.
			tier (str, optional): Destination. Defaults to the working tier.
			role (str, optional): Who produced it.
			salience (float, optional): How much it is worth keeping.
			group (str, optional): Records that live and die together.
			metadata (dict, optional): Arbitrary JSON.
			owner (str, optional): Which agent wrote it.
			id (str, optional): Makes the write idempotent (spec 3.2).
			session (str, optional): Which conversation it belongs to.

		Returns:
			MemoryRecord: The stored record, or the existing one for a repeated id.

		Raises:
			KeyError: For an unknown tier.
		"""
		with self._lock:
			if id is not None:
				existing = self.get(id)
				if existing is not None:
					return existing

			# Writing to an occupied key replaces what is there, so a corrected fact
			# stops being recalled instead of merely being outnumbered by its
			# successor.
			if key is not None:
				held = self.by_key(key, agent=owner)
				if held is not None:
					self.stores[held.tier].delete([held.id])
					held.content   = content
					held.revision += 1
					held.role      = role
					held.salience  = float(salience)
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
					self._place(held)
					self.db.commit()
					return held

			name  = tier or self.working
			store = self.stores.get(name, None)
			if store is None:
				raise KeyError(f"Unknown tier '{name}'")

			record = MemoryRecord(
				content  = content,
				tier     = name,
				role     = role,
				salience = salience,
				group    = group,
				metadata = metadata,
				owner     = owner,
				session   = session,
				key        = key,
				embedding  = embedding,
				uri        = uri,
				media_type = media_type,
				id         = id,
			)

			# Working memory is never embedded: it is replayed, not searched, and
			# most of it is evicted having never been read by a query.
			# A caller-supplied vector is stored verbatim and never regenerated.
			if embedding is None and isinstance(store, DurableStore) and str(content).strip():
				vectors   = self._embed([content])
				embedding = vectors[0] if vectors else None

			# The store indexes its own vector, if it keeps an index at all.
			store.add(record, embedding)

			self.db.commit()
			return record


	def get(self, id: str, agent: str = None) -> MemoryRecord | None:
		"""Fetch one record by id, across every tier.

		Args:
			id (str): The record id.
			agent (str, optional): Scope.

		Returns:
			MemoryRecord | None: The record, or None.
		"""
		for name, store in self.stores.items():
			for record in store.rows(agent):
				if record.id == id:
					return record
		return None


	def count(self, agent: str = None) -> int:
		"""How many records are visible across every tier.

		Args:
			agent (str, optional): Scope.

		Returns:
			int: Visible record count.
		"""
		with self._lock:
			return sum(store.count(agent) for store in self.stores.values())


	def records_in(self, tier: str, agent: str = None) -> list[MemoryRecord]:
		"""Every visible record in one tier.

		Args:
			tier (str): Tier name.
			agent (str, optional): Scope.

		Returns:
			list[MemoryRecord]: Empty for an unknown tier.
		"""
		store = self.stores.get(tier, None)
		return store.rows(agent) if store else []


	def timeline(self, tier: str = None, limit: int = 0, agent: str = None,
	             where: dict[str, Any] = None, key_prefix: str = None) -> list[MemoryRecord]:
		"""Read records in creation order.

		Args:
			tier (str, optional): Restrict to one tier.
			limit (int, optional): Keep the most recent N, still ascending.
			agent (str, optional): Scope.
			where (dict, optional): Metadata filter -- pass {'session': ...} to replay
				one conversation.

		Returns:
			list[MemoryRecord]: Ascending by created_at.

		Raises:
			KeyError: For an unknown tier.
		"""
		with self._lock:
			names   = [tier] if tier else self.order
			records = []
			for name in names:
				if name not in self.stores:
					raise KeyError(f"Unknown tier '{name}'")
				records.extend(r for r in self.stores[name].rows(agent)
				               if self._matches(r, where) and self._under(r, key_prefix))

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
		"""Search by relevance, using the vector index when there is one.

		Records with no vector -- everything in working memory, anything written
		before an embedder was configured -- are still ranked, by the lexical scorer,
		so a partially embedded store degrades rather than going blind.

		Args:
			query (str, optional): What to rank against.
			tier (str, optional): Restrict to one tier.
			limit (int, optional): Maximum records.
			where (dict, optional): Metadata filter.
			min_score (float, optional): Drop results below this blended score.
			touch (bool, optional): Whether this counts as an access.
			now (float, optional): Epoch seconds.
			agent (str, optional): Scope.

		Returns:
			list[tuple[MemoryRecord, float]]: Descending by score.

		Raises:
			KeyError: For an unknown tier.
		"""
		now   = now if now is not None else time.time()
		names = [tier] if tier else self.order

		with self._lock:
			for name in names:
				if name not in self.stores:
					raise KeyError(f"Unknown tier '{name}'")

			candidates : list[MemoryRecord] = []
			relevance  : dict[str, float]   = {}

			if embedding is not None:
				vector = list(embedding)
			else:
				vectors = self._embed([query]) if query else None
				vector  = vectors[0] if vectors else None

			for name in names:
				store = self.stores[name]
				rows  = [r for r in store.rows(agent)
				         if self._matches(r, where) and self._under(r, key_prefix)]
				candidates.extend(rows)

				if vector is None:
					continue

				# The tier answers with its index if it has one, and abstains with
				# None if it does not. Abstaining is not "nothing matched": it means
				# rank these another way, so the fallback is cosine in Python over the
				# survivors this tier already filtered.
				ranked = store.knn(vector, agent, max(limit or 8, 8) * 4)

				if ranked is not None:
					relevance.update(ranked)
					continue

				known = store.vectors(agent)
				for record in rows:
					if record.id in known:
						relevance[record.id] = max(0.0, cosine(vector, known[record.id]))

			if not candidates:
				return []

			# A record carrying its own vector is comparable wherever it lives --
			# including working memory, which has no vector index. The store never
			# generated that embedding, so it is not the store's business which
			# tier it happens to sit in.
			if vector is not None:
				for record in candidates:
					if record.embedding and record.id not in relevance:
						relevance[record.id] = max(0.0, cosine(vector, record.embedding))

			# Records with no vector -- most of working, anything written before an
			# embedder was configured -- still deserve a ranking, so the lexical
			# scorer covers them.
			lexical = self.lexical.relevance(query, candidates) if query else None
			if embedding is not None:
				lexical = None
			if lexical:
				for id, score in lexical.items():
					relevance[id] = max(relevance.get(id, 0.0), score)

			scored = []
			if (query or embedding is not None) and (relevance or lexical is not None):
				for record in candidates:
					score = relevance.get(record.id, 0.0)
					if score > 0.0:
						scored.append((record, self._blend(score, record, now)))
			else:
				for record in candidates:
					scored.append((record, self._blend(0.0, record, now)))

			scored = [(r, s) for r, s in scored if s >= min_score]
			scored.sort(key=lambda pair: (-pair[1], -pair[0].created_at))

			if limit and limit > 0:
				scored = scored[:limit]

			if touch and scored:
				by_tier: dict[str, list[str]] = {}
				for record, _ in scored:
					by_tier.setdefault(record.tier, []).append(record.id)
				for name, ids in by_tier.items():
					self.stores[name].touch(ids, now)
				self.db.commit()

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
			tier (str, optional): Restrict to one tier.
			where (dict, optional): Metadata filter.
			agent (str, optional): Scope.

		Returns:
			int: How many were removed.
		"""
		targets: set[str] = set()

		if ids:
			targets.update(ids)

		if query or where or key_prefix or (tier and not ids):
			for record, _ in self.recall(query=query, tier=tier, where=where, limit=0,
			                             touch=False, agent=agent, key_prefix=key_prefix):
				targets.add(record.id)

		with self._lock:
			removed = 0
			for store in self.stores.values():
				removed += store.delete(list(targets))
			self.db.commit()
			return removed


	def reinforce(self, ids: list[str], amount: float = 0.5, agent: str = None) -> int:
		"""Raise the salience of records that proved useful.

		Args:
			ids (list[str]): Records to reinforce.
			amount (float, optional): How much to add.
			agent (str, optional): Scope.

		Returns:
			int: How many were reinforced.
		"""
		with self._lock:
			bumped = sum(store.bump(list(ids or []), amount) for store in self.stores.values())
			self.db.commit()
			return bumped


	def promote(self, ids: list[str], tier: str, salience: float = 0.5) -> int:
		"""Move records to a tier they earned, re-embedding on arrival if needed.

		Args:
			ids (list[str]): Records to move.
			tier (str): Destination.
			salience (float, optional): Added on arrival.

		Returns:
			int: How many moved.

		Raises:
			KeyError: For an unknown tier.
		"""
		target = self.tier(tier).name

		with self._lock:
			moved = 0
			for id in ids or []:
				record = self.get(id)
				if record is None or record.tier == target:
					continue

				self.stores[record.tier].delete([id])
				record.tier      = target
				record.salience += salience
				self._place(record)
				moved += 1

			self.db.commit()
			return moved


	def _place(self, record: MemoryRecord) -> None:
		"""Insert a record into whichever store now owns it.

		Embeds it when the destination is a tier that gets searched, which is why a
		record spilling out of working memory acquires a vector it never had.

		Args:
			record (MemoryRecord): The record, with 'tier' already set to its
				destination.
		"""
		store     = self.stores[record.tier]
		embedding = None

		if isinstance(store, DurableStore) and str(record.content).strip():
			vectors   = self._embed([record.content])
			embedding = vectors[0] if vectors else None

		store.add(record, embedding)


	def consolidate(self, now: float = None) -> dict[str, Any]:
		"""Promote what was earned, then push down what overflows.

		Capacity is enforced per session on tiers marked per_session, so two
		conversations cannot evict each other.

		Args:
			now (float, optional): Epoch seconds.

		Returns:
			dict: moved, dropped, summarized, promoted and per-tier counts.
		"""
		now        = now if now is not None else time.time()
		moved      = dropped = summarized = promoted = 0

		with self._consolidating:
			for name in self.order:
				tier = self.tiers[name]
				if not tier.promote_to or tier.promote_after <= 0:
					continue

				with self._lock:
					earned = [r.id for r in self.stores[name].rows() if r.access_count >= tier.promote_after]

				promoted += self.promote(earned, tier.promote_to)

			for name in self.order:
				tier  = self.tiers[name]
				store = self.stores[name]

				with self._lock:
					records = store.rows()
					if tier.capacity <= 0:
						continue

					# Capacity inside a conversation, not across the tier.
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

						ranked = sorted(groups.values(),
						                key=lambda g: min(self._retention(r, tier, now) for r in g))

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
							store.delete([r.id for r in group])
							self.db.commit()
						dropped += len(group)
						continue

					# Outside the record lock: this may be an LLM call.
					contents = self.consolidate_fn(group, tier.spill_to) if self.consolidate_fn else None

					with self._lock:
						store.delete([r.id for r in group])

						if contents is None:
							for record in group:
								record.tier = tier.spill_to
								self._place(record)
							moved += len(group)
						else:
							for content in contents:
								self._place(MemoryRecord(
									content  = content,
									tier     = tier.spill_to,
									role     = "memory",
									salience = max(r.salience for r in group),
									group    = group[0].group,
									owner    = group[0].owner,
									metadata = {"consolidated_from": [r.id for r in group]},
								))
							summarized += 1
							moved      += len(contents)

						self.db.commit()

		with self._lock:
			counts = {name: store.count() for name, store in self.stores.items()}

		return {"moved": moved, "dropped": dropped, "summarized": summarized,
		        "promoted": promoted, "counts": counts}


	def sessions(self, agent: str = None) -> list[dict[str, Any]]:
		"""Which conversations exist, and where their records sit.

		Args:
			agent (str, optional): Scope.

		Returns:
			list[dict]: One entry per session, ascending by opened_at.
		"""
		from memory import to_rfc3339

		found: dict[str, dict[str, Any]] = {}
		with self._lock:
			for name, store in self.stores.items():
				for record in store.rows(agent):
					if record.session is None:
						continue
					entry = found.setdefault(record.session, {
						"session": record.session, "records": 0, "tiers": {},
						"opened_at": record.created_at, "touched_at": record.accessed_at,
					})
					entry["records"]     += 1
					entry["tiers"][name]  = entry["tiers"].get(name, 0) + 1
					entry["opened_at"]    = min(entry["opened_at"], record.created_at)
					entry["touched_at"]   = max(entry["touched_at"], record.accessed_at)

		for entry in found.values():
			entry["opened_at"]  = to_rfc3339(entry["opened_at"])
			entry["touched_at"] = to_rfc3339(entry["touched_at"])

		return sorted(found.values(), key=lambda e: e["opened_at"])


	def close_session(self, session: str, agent: str = None) -> dict[str, Any]:
		"""Flush a conversation out of working memory, then let it percolate.

		Args:
			session (str): The conversation to close.
			agent (str, optional): Scope.

		Returns:
			dict: The consolidate report plus 'closed' and 'flushed'.
		"""
		flushed = 0

		for name in self.of_kind("working"):
			tier = self.tiers[name]
			if not tier.spill_to:
				continue

			with self._lock:
				leaving = [r for r in self.stores[name].rows(agent) if r.session == session]

			groups: dict[str, list[MemoryRecord]] = {}
			for record in leaving:
				groups.setdefault(record.group, []).append(record)

			for group in groups.values():
				contents = self.consolidate_fn(group, tier.spill_to) if self.consolidate_fn else None

				with self._lock:
					self.stores[name].delete([r.id for r in group])

					if contents is None:
						for record in group:
							record.tier = tier.spill_to
							self._place(record)
						flushed += len(group)
					else:
						for content in contents:
							self._place(MemoryRecord(
								content  = content,
								tier     = tier.spill_to,
								role     = "memory",
								salience = max(r.salience for r in group),
								group    = group[0].group,
								owner    = group[0].owner,
								session  = session,
								metadata = {"consolidated_from": [r.id for r in group]},
							))
						flushed += len(contents)

					self.db.commit()

		report = self.consolidate()
		return dict(report, closed=session, flushed=flushed)


	def describe(self, agent: str = None) -> dict[str, Any]:
		"""Everything a client needs to know about this store.

		Args:
			agent (str, optional): Scope the counts.

		Returns:
			dict: Tiers with counts, kinds, the working tier, and a 'scorer' block
			naming the backing file and which TierStore serves each tier.
		"""
		with self._lock:
			return {
				"tiers"  : [dict(self.tiers[n].to_dict(), count=self.stores[n].count(agent)) for n in self.order],
				"order"  : list(self.order),
				"kinds"  : {k: self.of_kind(k) for k in KINDS if self.of_kind(k)},
				"working": self.working,
				"private": [n for n in self.order if not self.tiers[n].shared],
				"total"  : sum(s.count(agent) for s in self.stores.values()),
				"weights": dict(self.weights),
				"scorer" : {
					"scorer"  : self.SCORER,
					"backend" : self.BACKEND,
					"vectors" : bool(self.embed),
					"stores"  : {n: type(s).__name__ for n, s in self.stores.items()},
				},
			}


	def by_key(self, key: str, agent: str = None) -> MemoryRecord | None:
		"""The record addressed by a key.

		Args:
			key (str): The address.
			agent (str, optional): Whose key. Keys are unique per owner.

		Returns:
			MemoryRecord | None: The record, or None.
		"""
		with self._lock:
			for store in self.stores.values():
				for record in store.rows(agent):
					if record.key == key and record.owner == agent:
						return record
		return None


	def _under(self, record: MemoryRecord, prefix: str = None) -> bool:
		"""Whether a record's key sits under a prefix.

		Args:
			record (MemoryRecord): The candidate.
			prefix (str, optional): A key prefix.

		Returns:
			bool: True if at or under the prefix.
		"""
		if not prefix:
			return True
		return bool(record.key) and record.key.startswith(prefix)


	def _matches(self, record: MemoryRecord, where: dict[str, Any] = None) -> bool:
		"""Apply a 'where' filter to one record.

		Args:
			record (MemoryRecord): The candidate.
			where (dict, optional): The filter.

		Returns:
			bool: True if it matches.
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
		"""Combine relevance with recency and salience.

		Args:
			relevance (float): The ranker's verdict, 0..1.
			record (MemoryRecord): The candidate.
			now (float): Epoch seconds.

		Returns:
			float: The blended score.
		"""
		tier = self.tiers[record.tier]
		return (
			self.weights["lexical" ] * relevance
			+ self.weights["recency" ] * recency(now - record.accessed_at, tier.half_life)
			+ self.weights["salience"] * (record.salience / (1.0 + record.salience))
		)


	def _retention(self, record: MemoryRecord, tier: MemoryTier, now: float) -> float:
		"""How strongly a record has earned its place. Lowest is evicted first.

		Args:
			record (MemoryRecord): The candidate.
			tier (MemoryTier): Its tier, for the half-life.
			now (float): Epoch seconds.

		Returns:
			float: Higher survives longer.
		"""
		return record.salience * recency(now - record.accessed_at, tier.half_life) * (1.0 + record.access_count)


	def close(self) -> None:
		"""Commit and close the database connection.
		"""
		with self._lock:
			self.db.commit()
			self.db.close()


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
		from retrieval import ollama_embedder

		stack  = open_stack("memory.db", embed=ollama_embedder())
		server = MemoryServer(stack)
		serve_stdio(server.dispatcher)
	"""
	return SqliteMemoryStack(path=path, embed=embed, **kwargs)


def single_tier(kind: str) -> list[MemoryTier]:
	"""One tier, unbounded, named for its role.

	This is what a federated backend looks like. Capacity, spilling and promotion
	are deliberately absent: in a federation those are the *router's* policy,
	because only the router can see more than one tier. A backend is a store, not
	a stack.

	Args:
		kind (str): working, episodic, semantic or procedural.

	Returns:
		list[MemoryTier]: A single unbounded tier.
	"""
	return [MemoryTier(kind, kind=kind, capacity=0)]


def main() -> int:
	"""Run this file as an A2M server.

		python a2m_store.py memory.db                  stdio, all four tiers
		python a2m_store.py memory.db --http 8778      http
		python a2m_store.py working.db --tier working  one tier, for a federation
		python a2m_store.py memory.db --embed          with vector search

	Returns:
		int: Process exit code.
	"""
	argv = [a for a in sys.argv[1:]]
	path = next((a for a in argv if not a.startswith("--") and not a.isdigit()), "memory.db")

	embed = None
	if "--embed" in argv:
		from retrieval import ollama_embedder
		embed = ollama_embedder()

	tiers = None
	name  = f"a2m-sqlite({pathlib.Path(path).name})"
	if "--tier" in argv:
		kind  = argv[argv.index("--tier") + 1]
		tiers = single_tier(kind)
		name  = f"a2m-sqlite:{kind}"

	stack  = open_stack(path, embed=embed, tiers=tiers)
	server = MemoryServer(stack=stack, name=name)

	if "--http" in argv:
		index = argv.index("--http")
		port  = int(argv[index + 1]) if len(argv) > index + 1 else 8778
		print(f"A2M {A2M_VERSION} on http://127.0.0.1:{port}/ backed by {path}", file=sys.stderr)
		serve_a2m_http(server, port=port).serve_forever()
	else:
		serve_stdio(server.dispatcher)

	return 0


if __name__ == "__main__":
	sys.exit(main())
