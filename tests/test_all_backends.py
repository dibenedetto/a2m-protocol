"""
A2M — Backend integration test.

Tests every viable backend combination with both Agno and LangChain adapters.
Each combination gets its own in-process server on a unique port.

Combinations tested:
    1. SQLite + Numpy  (default, zero extra deps)
    2. SQLite + Numpy  (persistent .npz)
    3. SQLite + LanceDB

Skipped (logged):
    - SQLite + Chroma      (chromadb incompatible with Python >= 3.13)
    - PostgreSQL + Numpy   (psycopg2 not installed)
    - PostgreSQL + PgVector(psycopg2 not installed)

Usage:
    python tests/test_all_backends.py
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

from client.client import A2MClient
from server.main import create_app
from server.backends.base import AbstractRelationalBackend, AbstractVectorBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_next_port = 19000

def _alloc_port() -> int:
    global _next_port
    p = _next_port
    _next_port += 1
    return p


def _start_server(app, port: int) -> uvicorn.Server:
    srv = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error",
    ))
    threading.Thread(target=srv.run, daemon=True).start()
    time.sleep(0.8)
    return srv


def embed(text: str) -> list[float]:
    """Toy 16-dim bigram-frequency embedding."""
    dim = 16
    v = [0.0] * dim
    for i in range(len(text) - 1):
        v[(ord(text[i]) * 256 + ord(text[i + 1])) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


# ---------------------------------------------------------------------------
# Reusable test suite
# ---------------------------------------------------------------------------

def run_core_tests(label: str, url: str) -> list[str]:
    """
    Run the full A2M test suite against a running server.
    Returns a list of failure messages (empty = all passed).
    """
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = ""):
        if not condition:
            msg = f"  FAIL  {name}" + (f": {detail}" if detail else "")
            failures.append(msg)
            print(msg)
        else:
            print(f"  ok    {name}")

    c = A2MClient(url, namespace="test/core")

    # -- health
    h = c.health()
    check("health.status", h.get("status") in ("ok", "degraded"), f"got {h}")
    check("health.relational.ok", h.get("relational", {}).get("ok") is True)

    # -- write + read
    entry = c.write("k1", type="semantic", value={"text": "hello"}, embedding=embed("hello"))
    check("write.key", entry.get("key") == "k1")

    got = c.read("k1")
    check("read.found", got is not None and got.get("key") == "k1")

    miss = c.read("no-such-key")
    check("read.miss", miss is None)

    # -- write second entry
    c.write("k2", type="episodic", value="world", meta={"tags": ["demo"]})

    # -- list
    resp = c.list()
    check("list.total", resp.get("total", 0) >= 2, f"total={resp.get('total')}")

    # -- list with type filter
    resp = c.list(type="semantic")
    check("list.type_filter", resp.get("total") == 1, f"total={resp.get('total')}")

    # -- list with tag filter
    resp = c.list(tags=["demo"])
    check("list.tag_filter", resp.get("total") == 1, f"total={resp.get('total')}")

    # -- semantic query
    results = c.query(embedding=embed("hello world"), type="semantic", top_k=5)
    check("query.returns_results", len(results) > 0, f"got {len(results)}")
    if results:
        check("query.has_score", "score" in results[0], str(results[0].keys()))
        check("query.has_entry", "entry" in results[0])

    # -- update (upsert)
    c.write("k1", type="semantic", value={"text": "updated"}, embedding=embed("updated"))
    got2 = c.read("k1")
    check("upsert.value_updated",
          got2 is not None and got2.get("value", {}).get("text") == "updated")

    # -- delete single
    c.delete("k2")
    check("delete.single", c.read("k2") is None)

    # -- bulk delete
    c.write("bulk1", type="working", value="a")
    c.write("bulk2", type="working", value="b")
    c.write("bulk3", type="working", value="c")
    count_before = c.list(type="working").get("total", 0)
    c.delete_bulk(type="working")
    count_after = c.list(type="working").get("total", 0)
    check("delete_bulk", count_before >= 3 and count_after == 0,
          f"before={count_before}, after={count_after}")

    # -- scoped client
    child = c.scoped("sub")
    child.write("nested", type="semantic", value="scoped-val", embedding=embed("scoped"))
    check("scoped.write_read", child.read("nested") is not None)

    # Cleanup
    c.delete_bulk()
    child.delete_bulk()

    return failures


def run_agno_tests(label: str, url: str) -> list[str]:
    """Test the Agno adapters against a running server."""
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = ""):
        if not condition:
            msg = f"  FAIL  {name}" + (f": {detail}" if detail else "")
            failures.append(msg)
            print(msg)
        else:
            print(f"  ok    {name}")

    from agno.memory import UserMemory
    from agno.knowledge.document import Document as AgnoDoc
    from adapters.agno_basedb import A2MAgnoBaseDb
    from adapters.agno_vectordb import A2MAgnoVectorDb

    # -- BaseDb adapter
    mem_client = A2MClient(url, namespace="test/agno/mem")
    db = A2MAgnoBaseDb(client=mem_client, embed_fn=embed)

    mem = UserMemory(memory="likes coffee", memory_id="pref1", user_id="u1", topics=["food"])
    db.upsert_user_memory(mem)
    retrieved = db.get_user_memories(user_id="u1")
    check("agno.basedb.upsert+get", len(retrieved) >= 1)

    topics = db.get_all_memory_topics(user_id="u1")
    check("agno.basedb.topics", "food" in topics, f"got {topics}")

    search_res = db.search_user_memories(embedding=embed("coffee"), user_id="u1", limit=5)
    check("agno.basedb.search", len(search_res) >= 1)

    check("agno.basedb.table_exists", db.table_exists(db.memory_table_name))

    # -- VectorDb adapter
    vec_client = A2MClient(url, namespace="test/agno/vec")
    vdb = A2MAgnoVectorDb(client=vec_client, embed_fn=embed)

    docs = [
        AgnoDoc(content="A2M is a memory protocol", id="d1", name="overview"),
        AgnoDoc(content="Embeddings are caller-provided", id="d2", name="embeddings"),
    ]
    vdb.insert("hash1", docs)
    check("agno.vectordb.insert", vdb.exists())
    check("agno.vectordb.content_hash", vdb.content_hash_exists("hash1"))
    check("agno.vectordb.name_exists", vdb.name_exists("overview"))
    check("agno.vectordb.id_exists", vdb.id_exists("d1"))

    hits = vdb.search("memory protocol", limit=2)
    check("agno.vectordb.search", len(hits) >= 1)

    vdb.delete_by_id("d1")
    check("agno.vectordb.delete_by_id", not vdb.id_exists("d1"))

    vdb.drop()
    check("agno.vectordb.drop", not vdb.exists())

    # Cleanup
    mem_client.delete_bulk()
    vec_client.delete_bulk()

    return failures


def run_langchain_tests(label: str, url: str) -> list[str]:
    """Test the LangChain adapters against a running server."""
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = ""):
        if not condition:
            msg = f"  FAIL  {name}" + (f": {detail}" if detail else "")
            failures.append(msg)
            print(msg)
        else:
            print(f"  ok    {name}")

    from langchain_core.embeddings import Embeddings
    from langchain_core.documents import Document as LCDoc
    from langchain_core.messages import HumanMessage, AIMessage
    from adapters.langchain_basechatmessagehistory import A2MLangChainBaseChatMessageHistory
    from adapters.langchain_vectorstore import A2MLangChainVectorStore

    class TestEmbed(Embeddings):
        def embed_documents(self, texts):
            return [embed(t) for t in texts]
        def embed_query(self, text):
            return embed(text)

    emb = TestEmbed()

    # -- VectorStore
    vs_client = A2MClient(url, namespace="test/lc/vs")
    vs = A2MLangChainVectorStore(client=vs_client, embeddings=emb, collection_tag="test")

    docs = [
        LCDoc(page_content="A2M is a shared memory protocol", metadata={"src": "spec"}),
        LCDoc(page_content="Embeddings are caller-provided", metadata={"src": "spec"}),
        LCDoc(page_content="Namespace hierarchy is app/wf/sess/agent", metadata={"src": "spec"}),
        LCDoc(page_content="Agents share state via REST API", metadata={"src": "blog"}),
    ]
    ids = vs.add_documents(docs)
    check("lc.vs.add_documents", len(ids) == 4)

    hits = vs.similarity_search("memory protocol", k=2)
    check("lc.vs.similarity_search", len(hits) >= 1)

    scored = vs.similarity_search_with_score("embeddings", k=2)
    check("lc.vs.similarity_search_with_score",
          len(scored) >= 1 and isinstance(scored[0], tuple))

    by_vec = vs.similarity_search_by_vector(embed("memory"), k=2)
    check("lc.vs.similarity_search_by_vector", len(by_vec) >= 1)

    fetched = vs.get_by_ids(ids[:2])
    check("lc.vs.get_by_ids", len(fetched) == 2)

    mmr = vs.max_marginal_relevance_search("memory protocol", k=2, fetch_k=4)
    check("lc.vs.mmr_search", len(mmr) >= 1)

    mmr_vec = vs.max_marginal_relevance_search_by_vector(embed("memory"), k=2, fetch_k=4)
    check("lc.vs.mmr_search_by_vector", len(mmr_vec) >= 1)

    vs.delete(ids=[ids[0]])
    after = vs.get_by_ids([ids[0]])
    check("lc.vs.delete_by_id", len(after) == 0)

    vs.delete()
    remaining = vs.similarity_search("anything", k=10)
    check("lc.vs.delete_all", len(remaining) == 0)

    # -- from_texts
    vs2 = A2MLangChainVectorStore.from_texts(
        ["text one", "text two"], embedding=emb,
        client=A2MClient(url, namespace="test/lc/vs2"),
    )
    check("lc.vs.from_texts", len(vs2.similarity_search("one", k=1)) >= 1)

    # -- BaseChatMessageHistory
    hist_client = A2MClient(url, namespace="test/lc/hist")
    hist = A2MLangChainBaseChatMessageHistory(client=hist_client)

    hist.add_messages([HumanMessage(content="Hello"), AIMessage(content="Hi there")])
    msgs = hist.messages
    check("lc.hist.add+get", len(msgs) == 2, f"got {len(msgs)}")

    hist.clear()
    check("lc.hist.clear", len(hist.messages) == 0)

    # Cleanup
    vs_client.delete_bulk()
    hist_client.delete_bulk()

    return failures


def run_cross_framework_tests(label: str, url: str) -> list[str]:
    """
    Test that Agno and LangChain adapters can share knowledge natively.
    Both adapters use the same a2m:knowledge tag, so search() / similarity_search()
    return results from both frameworks.
    """
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = ""):
        if not condition:
            msg = f"  FAIL  {name}" + (f": {detail}" if detail else "")
            failures.append(msg)
            print(msg)
        else:
            print(f"  ok    {name}")

    from langchain_core.embeddings import Embeddings
    from langchain_core.documents import Document as LCDoc
    from agno.knowledge.document import Document as AgnoDoc
    from adapters.agno_vectordb import A2MAgnoVectorDb
    from adapters.langchain_vectorstore import A2MLangChainVectorStore

    class XEmbed(Embeddings):
        def embed_documents(self, texts):
            return [embed(t) for t in texts]
        def embed_query(self, text):
            return embed(text)

    xemb = XEmbed()

    # Both adapters share the SAME namespace
    ns = "test/cross/shared"
    agno_client = A2MClient(url, namespace=ns)
    lc_client   = A2MClient(url, namespace=ns)

    agno_vdb = A2MAgnoVectorDb(client=agno_client, embed_fn=embed)
    lc_store = A2MLangChainVectorStore(client=lc_client, embeddings=xemb)

    # -- Agno writes
    agno_docs = [
        AgnoDoc(content="A2M is a shared memory protocol for AI agents.",
                id="agno-x1", name="overview"),
        AgnoDoc(content="Embeddings are always caller-provided.",
                id="agno-x2", name="embed-info"),
    ]
    agno_vdb.insert("xhash1", agno_docs)

    # -- LangChain writes
    lc_docs = [
        LCDoc(page_content="LangChain agents share state via the A2M REST API.",
              metadata={"src": "blog"}),
        LCDoc(page_content="Vector search uses cosine similarity over embeddings.",
              metadata={"src": "spec"}),
    ]
    lc_store.add_documents(lc_docs)

    # -- LangChain reads Agno docs
    lc_results = lc_store.similarity_search("shared memory protocol", k=5)
    lc_texts = [d.page_content for d in lc_results]
    agno_found_by_lc = any("shared memory protocol" in t for t in lc_texts)
    check("cross.lc_finds_agno", agno_found_by_lc,
          f"texts={[t[:40] for t in lc_texts]}")

    # -- Agno reads LangChain docs
    agno_results = agno_vdb.search("cosine similarity", limit=5)
    agno_texts = [d.content for d in agno_results]
    lc_found_by_agno = any("cosine similarity" in t for t in agno_texts)
    check("cross.agno_finds_lc", lc_found_by_agno,
          f"texts={[t[:40] for t in agno_texts]}")

    # -- Both get mixed results
    lc_mixed = lc_store.similarity_search("agents and memory", k=4)
    check("cross.lc_mixed_results", len(lc_mixed) == 4,
          f"got {len(lc_mixed)}")

    agno_mixed = agno_vdb.search("agents and memory", limit=4)
    check("cross.agno_mixed_results", len(agno_mixed) == 4,
          f"got {len(agno_mixed)}")

    # -- LangChain similarity_search_with_score works with Agno docs
    scored = lc_store.similarity_search_with_score("shared memory", k=3)
    check("cross.lc_scored_has_agno",
          any("shared memory protocol" in d.page_content for d, _ in scored),
          f"scored={[(d.page_content[:30], s) for d, s in scored]}")

    # -- Agno _entry_to_doc reconstructs LangChain doc correctly
    for doc in agno_results:
        check("cross.agno_doc_has_content", doc.content != "",
              f"content={doc.content!r}")

    # -- LangChain _entry_to_doc reconstructs Agno doc correctly
    for doc in lc_results:
        check("cross.lc_doc_has_content", doc.page_content != "",
              f"page_content={doc.page_content!r}")

    # Cleanup
    agno_client.delete_bulk()

    return failures


# ---------------------------------------------------------------------------
# Backend builders
# ---------------------------------------------------------------------------

def _build_sqlite(tmpdir: str) -> AbstractRelationalBackend:
    from server.backends.sqlite_relational import SQLiteRelationalBackend
    return SQLiteRelationalBackend(os.path.join(tmpdir, "test.db"))


def _build_numpy() -> AbstractVectorBackend:
    from server.backends.numpy_vector import NumpyVectorBackend
    return NumpyVectorBackend()


def _build_numpy_persistent(tmpdir: str) -> AbstractVectorBackend:
    from server.backends.numpy_vector import NumpyVectorBackend
    return NumpyVectorBackend(path=os.path.join(tmpdir, "numpy_index.npz"))


def _build_lancedb(tmpdir: str) -> AbstractVectorBackend:
    from server.backends.lancedb_vector import LanceVectorBackend
    return LanceVectorBackend(uri=os.path.join(tmpdir, "lancedb_data"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total_failures: list[str] = []
    tmpdir = tempfile.mkdtemp(prefix="a2m_test_")
    tmpdir_np = tempfile.mkdtemp(prefix="a2m_test_np_")

    combos: list[tuple[str, AbstractRelationalBackend, AbstractVectorBackend]] = []

    # 1. SQLite + Numpy (ephemeral)
    combos.append(("SQLite + Numpy (ephemeral)", _build_sqlite(tmpdir), _build_numpy()))

    # 2. SQLite + Numpy (persistent)
    combos.append(("SQLite + Numpy (persistent)", _build_sqlite(tmpdir_np),
                    _build_numpy_persistent(tmpdir_np)))

    # 3. SQLite + LanceDB
    try:
        lance = _build_lancedb(tmpdir)
        from server.backends.sqlite_relational import SQLiteRelationalBackend
        combos.append(("SQLite + LanceDB",
                        SQLiteRelationalBackend(os.path.join(tmpdir, "lance_test.db")),
                        lance))
    except ImportError as e:
        print(f"[skip] SQLite + LanceDB: {e}\n")

    # 4. SQLite + Chroma
    try:
        from server.backends.chroma_vector import ChromaVectorBackend
        chroma = ChromaVectorBackend()
        from server.backends.sqlite_relational import SQLiteRelationalBackend
        combos.append(("SQLite + Chroma",
                        SQLiteRelationalBackend(os.path.join(tmpdir, "chroma_test.db")),
                        chroma))
    except Exception as e:
        print(f"[skip] SQLite + Chroma: {e}\n")

    # 5. PostgreSQL + Numpy
    try:
        from server.backends.postgres_relational import PostgreSQLRelationalBackend
        pg = PostgreSQLRelationalBackend(dsn=os.environ.get("A2M_PG_DSN", "postgresql://localhost/a2m_test"))
        combos.append(("PostgreSQL + Numpy", pg, _build_numpy()))
    except Exception as e:
        print(f"[skip] PostgreSQL + Numpy: {e}\n")

    # 6. PostgreSQL + PgVector
    try:
        from server.backends.postgres_relational import PostgreSQLRelationalBackend
        from server.backends.pgvector_vector import PgVectorBackend
        dsn = os.environ.get("A2M_PG_DSN", "postgresql://localhost/a2m_test")
        combos.append(("PostgreSQL + PgVector",
                        PostgreSQLRelationalBackend(dsn),
                        PgVectorBackend(dsn=dsn, table_name="a2m_test_vectors")))
    except Exception as e:
        print(f"[skip] PostgreSQL + PgVector: {e}\n")

    print(f"Running {len(combos)} backend combination(s)...\n")

    for label, rel, vec in combos:
        port = _alloc_port()
        url = f"http://127.0.0.1:{port}"

        print("=" * 64)
        print(f"  {label}  ({url})")
        print("=" * 64)

        app = create_app(relational=rel, vector=vec)
        srv = _start_server(app, port)

        try:
            # Core A2M protocol tests
            print(f"\n--- Core tests ---")
            f1 = run_core_tests(label, url)
            total_failures.extend(f1)

            # Agno adapter tests
            print(f"\n--- Agno adapter tests ---")
            f2 = run_agno_tests(label, url)
            total_failures.extend(f2)

            # LangChain adapter tests
            print(f"\n--- LangChain adapter tests ---")
            f3 = run_langchain_tests(label, url)
            total_failures.extend(f3)

            # Cross-framework sharing tests
            print(f"\n--- Cross-framework sharing tests ---")
            f4 = run_cross_framework_tests(label, url)
            total_failures.extend(f4)

            combo_total = len(f1) + len(f2) + len(f3) + len(f4)
            status = "PASS" if combo_total == 0 else f"FAIL ({combo_total} failures)"
            print(f"\n  [{status}] {label}\n")

        finally:
            srv.should_exit = True
            time.sleep(0.3)

    # Cleanup temp dir
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(tmpdir_np, ignore_errors=True)
    except Exception:
        pass

    # Summary
    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  Backend combinations tested: {len(combos)}")
    if total_failures:
        print(f"  Total failures: {len(total_failures)}")
        for f in total_failures:
            print(f"    {f}")
        sys.exit(1)
    else:
        print("  All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
