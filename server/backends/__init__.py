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
    #   from server.backends.numpy_backend    import NumpyVectorBackend
    #   from server.backends.postgres         import PostgreSQLRelationalBackend
    #   from server.backends.pgvector_backend import PgVectorBackend
    #   from server.backends.lancedb_backend  import LanceVectorBackend
    #   from server.backends.chroma_backend   import ChromaVectorBackend
]
