"""Sample implementations: the specification, implemented more than once.

One implementation proves nothing — it is always possible to write a document
that describes exactly the program you already wrote. These are here so that the
same protocol has to survive storage engines that share no code, a language that
is not Python, and a topology where every tier is a different process.

	minimal.py         a dict. Imports nothing from this repository, on purpose.
	minimal.ts         the same server in TypeScript. No dependencies, no build.
	client.py          the other side, under the same rule: standard library only.
	store.py           the tier logic every persistent backend shares. No SQL.
	store_sqlite.py    that logic on SQLite, with sqlite-vec when it loads.
	store_postgres.py  the same logic on PostgreSQL and pgvector.
	router.py          the same logic federated: one A2M server per tier.
	adapters/          existing frameworks talking to any of the above, unmodified.

Modules that import `a2m` are run as modules from the repository root:

	python -m implementations.store_sqlite memory.db

`minimal.py` and `client.py` import nothing at all, so they run as plain files
from anywhere — which is the entire claim they are making.

Importing this package imports none of them.
"""
