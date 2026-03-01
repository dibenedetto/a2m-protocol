"""
A2M — AgentToMemory Protocol
Numpy vector backend: NumpyVectorBackend.

In-memory ANN search using numpy cosine similarity.  When a path is provided,
the index is persisted to a compressed .npz file and loaded on the next startup,
avoiding a full rebuild from the relational backend.

Persistence details:
  - Saved atomically: numpy writes to a .tmp file then os.replace() renames it,
    so a crash mid-save never leaves a corrupt index file.
  - On startup:
      1. __init__ attempts to load the .npz file.  If it exists the index is
         ready immediately and rebuild() is a no-op.
      2. If the file is absent, A2MStore calls rebuild() with all embeddings
         from the relational backend, then saves the result so subsequent
         startups are fast.
  - Saved after every index(), remove(), and remove_namespace() call so the
    file stays in sync with the relational store.

Query performance:
  - Candidates are filtered by namespace under the lock (O(n) scan).
  - Cosine scores are computed with a single numpy matrix-vector multiply
    (matrix @ q) rather than a Python loop, which is much faster for large
    in-memory indexes.
  - Embeddings are stored internally as np.float32 ndarrays to avoid
    repeated list→array conversions at query time.

Usage:
    from server.backends.numpy_backend import NumpyVectorBackend
    from server.main import create_app

    # Ephemeral (in-memory only, current default behaviour):
    app = create_app(vector=NumpyVectorBackend())

    # Persistent (saved to disk between restarts):
    app = create_app(vector=NumpyVectorBackend(path="./numpy_index.npz"))

    # Or via CLI:
    #   python -m server.main --numpy-path ./numpy_index.npz
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .base import AbstractVectorBackend, Entry


class NumpyVectorBackend(AbstractVectorBackend):
    """
    In-memory cosine similarity vector index backed by numpy.

    Thread-safe: a single threading.Lock guards all mutations and the
    candidate-list snapshot taken at query time.

    Args:
        path: Optional filesystem path for the .npz persistence file.
              - None  → pure in-memory; rebuilt from the relational store
                        on every server startup (original behaviour).
              - str   → load on init if the file exists; save after every
                        mutation; rebuild() is a no-op when the file is found.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._lock = threading.Lock()

        # {entry_id: (namespace, entry_key, np.ndarray[float32])}
        self._index: dict[str, tuple[str, str, np.ndarray]] = {}

        if path is not None:
            p = Path(path)
            self._path: Optional[Path] = p if p.suffix == ".npz" else p.with_suffix(".npz")
        else:
            self._path = None

        # True when the index was loaded from a pre-existing file; in that case
        # rebuild() is skipped (the file is the authoritative source).
        self._loaded_from_file = False
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the persisted index.  Called once from __init__."""
        if self._path is None or not self._path.exists():
            return
        try:
            data = np.load(self._path, allow_pickle=False)
            entry_ids  = data["entry_ids"].tolist()
            namespaces = data["namespaces"].tolist()
            keys       = data["keys"].tolist()
            embeddings = data["embeddings"]          # shape (N, dim), float32
            for eid, ns, key, emb in zip(entry_ids, namespaces, keys, embeddings):
                self._index[eid] = (ns, key, emb)   # emb is a np.ndarray view
            self._loaded_from_file = True
        except Exception:
            # Corrupt or incompatible file — start fresh; rebuild() will repopulate.
            self._index.clear()
            self._loaded_from_file = False

    def _save(self) -> None:
        """
        Persist the current index to disk.  Called while self._lock is held.
        Writes to a temp file then atomically replaces the target path.
        """
        if self._path is None:
            return

        if not self._index:
            # Nothing to save; remove stale file if present.
            if self._path.exists():
                self._path.unlink()
            return

        entry_ids  = list(self._index.keys())
        namespaces = [v[0] for v in self._index.values()]
        keys       = [v[1] for v in self._index.values()]
        embeddings = np.stack([v[2] for v in self._index.values()])  # (N, dim)

        # np.savez_compressed appends .npz when the path doesn't have it.
        # We pass a path without extension and expect exactly <stem>.tmp.npz.
        base     = str(self._path.with_suffix(""))     # strip existing .npz
        tmp_stem = base + ".tmp"                       # e.g. ./numpy_index.tmp
        np.savez_compressed(
            tmp_stem,
            entry_ids  = np.array(entry_ids,  dtype=str),   # numpy str_ (unicode, no pickle)
            namespaces = np.array(namespaces, dtype=str),
            keys       = np.array(keys,       dtype=str),
            embeddings = embeddings,
        )
        # np.savez_compressed wrote to <tmp_stem>.npz
        tmp_file = Path(tmp_stem + ".npz")
        os.replace(str(tmp_file), str(self._path))

    # ── AbstractVectorBackend ─────────────────────────────────────────────────

    def rebuild(self, entries: list[Entry]) -> None:
        """
        Seed the index from relational-backend entries.

        Skipped when the index was already loaded from a persisted file.
        When the file is absent, populates from entries and saves so that
        subsequent startups skip this step.
        """
        if self._loaded_from_file:
            return

        with self._lock:
            self._index.clear()
            for e in entries:
                if e.embedding:
                    self._index[e.id] = (
                        e.namespace, e.key,
                        np.array(e.embedding, dtype=np.float32),
                    )
            self._save()

    def index(self, entry_id: str, namespace: str, key: str, embedding: list[float]) -> None:
        """Upsert one embedding.  Persists immediately if a path was configured."""
        with self._lock:
            self._index[entry_id] = (namespace, key, np.array(embedding, dtype=np.float32))
            self._save()

    def remove(self, entry_id: str, namespace: str, key: str) -> None:
        """Remove one embedding by entry_id.  Persists immediately."""
        with self._lock:
            self._index.pop(entry_id, None)
            self._save()

    def remove_namespace(self, namespace: str, recursive: bool = False) -> int:
        """Remove all embeddings for a namespace (and children if recursive)."""
        with self._lock:
            to_del = [
                eid for eid, (ns, _k, _e) in self._index.items()
                if ns == namespace or (recursive and ns.startswith(namespace + "/"))
            ]
            for eid in to_del:
                del self._index[eid]
            self._save()
        return len(to_del)

    def query(
        self,
        query_vec: list[float],
        namespace: str,
        top_k: int,
        min_score: Optional[float],
        recursive: bool,
    ) -> list[tuple[str, str, float]]:
        """
        Cosine ANN search via numpy matrix-vector multiply.

        Takes a snapshot of matching candidates under the lock, then releases
        it before the numpy computation so concurrent writes are not blocked
        during scoring.
        """
        if not query_vec:
            return []

        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return []
        q /= q_norm

        # --- snapshot under lock --------------------------------------------
        with self._lock:
            if not self._index:
                return []
            candidates = [
                (ns, key, emb)
                for _eid, (ns, key, emb) in self._index.items()
                if ns == namespace or (recursive and ns.startswith(namespace + "/"))
            ]
        # --------------------------------------------------------------------

        if not candidates:
            return []

        # Stack into a matrix and compute cosine scores in one multiply.
        matrix = np.stack([c[2] for c in candidates])   # (N, dim)
        norms  = np.linalg.norm(matrix, axis=1)          # (N,)
        valid  = norms >= 1e-10
        if not valid.any():
            return []

        matrix = matrix[valid] / norms[valid, np.newaxis]
        scores = (matrix @ q).tolist()

        results: list[tuple[str, str, float]] = []
        vi = 0
        for i, (ns, key, _) in enumerate(candidates):
            if not valid[i]:
                continue
            s = scores[vi]
            vi += 1
            if min_score is None or s >= min_score:
                results.append((ns, key, float(s)))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
