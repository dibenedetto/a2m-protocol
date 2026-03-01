"""A2M server backends — pluggable relational and vector storage."""
from .base import AbstractRelationalBackend, AbstractVectorBackend
from .sqlite_relational import SQLiteRelationalBackend
from .numpy_vector import NumpyVectorBackend

__all__ = [
    # Abstract interfaces
    "AbstractRelationalBackend",
    "AbstractVectorBackend",
    # Default backends (zero extra deps)
    "SQLiteRelationalBackend",
    "NumpyVectorBackend",
    # Optional backends — imported lazily to avoid hard dependency errors:
    #   from server.backends.postgres_relational import PostgreSQLRelationalBackend
    #   from server.backends.pgvector_vector     import PgVectorBackend
    #   from server.backends.lancedb_vector      import LanceVectorBackend
    #   from server.backends.chroma_vector       import ChromaVectorBackend
]
