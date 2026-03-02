"""
A2M -- Cross-Framework Memory Sharing: Agno + LangChain (Adapter-Level)

Demonstrates A2M's core value proposition: agents from different frameworks
(Agno, LangChain) share memory through the A2M REST protocol using their
NATIVE adapter APIs.  No raw A2MClient queries needed -- the adapters
themselves are cross-framework aware.

Both adapters use the same standardised `a2m:knowledge` tag and both
`_entry_to_doc` functions handle both value formats (content/page_content,
meta_data/metadata), so Agno's search() returns LangChain-written docs
and LangChain's similarity_search() returns Agno-written docs.

Setup:
    pip install agno langchain-core   # plus the a2m stack
    python examples/cross_framework_a2m.py

Scenarios demonstrated:
    A. Agno writes knowledge docs  -> LangChain similarity_search() finds them
    B. LangChain writes knowledge  -> Agno search() finds them
    C. Mixed: both write, both search, results interleaved
    D. Agno user memories + LangChain chat history coexist in namespace tree
"""

import math
import sys
import threading
import time
from pathlib import Path

# -- 1. Start the A2M server in-process ----------------------------------------

import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))   # repo root
from server.main import create_app
from client.client import A2MClient

SERVER_PORT = 18802
app = create_app(db_path=":memory:")
_server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=SERVER_PORT, log_level="error"))
threading.Thread(target=_server.run, daemon=True).start()
time.sleep(0.8)

A2M_URL = f"http://127.0.0.1:{SERVER_PORT}"
print(f"[A2M] server running at {A2M_URL}\n")


# -- 2. Shared embedding function ----------------------------------------------
#
# CRITICAL: both frameworks must use the SAME embedding function for
# cross-framework vector search to work.  In production, this means
# both sides use the same model (e.g. all-MiniLM-L6-v2).

def embed(text: str) -> list[float]:
    """Toy 16-dim bigram-frequency embedding.  Same for both frameworks."""
    dim = 16
    v = [0.0] * dim
    for i in range(len(text) - 1):
        v[(ord(text[i]) * 256 + ord(text[i + 1])) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


# LangChain needs an Embeddings object wrapping the same function
from langchain_core.embeddings import Embeddings


class DemoEmbeddings(Embeddings):
    """Wraps embed() for LangChain's Embeddings interface."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embed(t) for t in texts]
    def embed_query(self, text: str) -> list[float]:
        return embed(text)


lc_embeddings = DemoEmbeddings()


# -- 3. Framework imports -------------------------------------------------------

from agno.knowledge.document import Document as AgnoDoc
from agno.memory import UserMemory
from langchain_core.documents import Document as LCDoc
from langchain_core.messages import HumanMessage, AIMessage

from adapters.agno_basedb import A2MAgnoBaseDb
from adapters.agno_vectordb import A2MAgnoVectorDb
from adapters.langchain_basechatmessagehistory import A2MLangChainBaseChatMessageHistory
from adapters.langchain_vectorstore import A2MLangChainVectorStore


# -- 4. SCENARIO A: Agno writes knowledge, LangChain reads via adapter ---------

print("=" * 68)
print("  SCENARIO A: Agno writes -> LangChain similarity_search() reads")
print("=" * 68)

# Both frameworks share the SAME namespace
SHARED_KB_NS = "demo/shared/kb"

# Agno agent ingests documents
agno_client = A2MClient(A2M_URL, namespace=SHARED_KB_NS)
agno_vdb = A2MAgnoVectorDb(client=agno_client, embed_fn=embed)

agno_docs = [
    AgnoDoc(content="A2M is a shared memory protocol for AI agents across frameworks.",
            id="agno-doc-1", name="overview", meta_data={"source": "spec"}),
    AgnoDoc(content="Embeddings are always caller-provided; the server never generates them.",
            id="agno-doc-2", name="embeddings", meta_data={"source": "spec"}),
    AgnoDoc(content="The namespace hierarchy is: app / workflow / session / agent.",
            id="agno-doc-3", name="namespaces", meta_data={"source": "spec"}),
]

print(f"\n[agno] Ingesting {len(agno_docs)} docs into namespace '{SHARED_KB_NS}' ...")
agno_vdb.insert("agno-batch-1", agno_docs)
print("  [ok] stored via A2MAgnoVectorDb")

# LangChain reads them via its NATIVE similarity_search()
lc_client = A2MClient(A2M_URL, namespace=SHARED_KB_NS)
lc_store = A2MLangChainVectorStore(client=lc_client, embeddings=lc_embeddings)

print("\n[langchain] similarity_search('shared memory protocol') ...")
lc_results = lc_store.similarity_search_with_score("shared memory protocol", k=3)
for doc, score in lc_results:
    print(f"  [{score:.3f}] {doc.page_content}")

print(f"\n  -> LangChain's similarity_search() found {len(lc_results)} docs written by Agno!")


# -- 5. SCENARIO B: LangChain writes knowledge, Agno reads via adapter ---------

print("\n" + "=" * 68)
print("  SCENARIO B: LangChain writes -> Agno search() reads")
print("=" * 68)

# LangChain agent adds documents to the SAME namespace
lc_docs = [
    LCDoc(page_content="Agno and LangChain agents share state via the A2M REST API.",
          metadata={"source": "blog", "author": "langchain-team"}),
    LCDoc(page_content="Vector search uses cosine similarity over caller-provided float embeddings.",
          metadata={"source": "spec", "author": "langchain-team"}),
]

print(f"\n[langchain] Adding {len(lc_docs)} docs to namespace '{SHARED_KB_NS}' ...")
lc_store.add_documents(lc_docs)
print("  [ok] stored via A2MLangChainVectorStore")

# Agno reads them via its NATIVE search()
print("\n[agno] search('cosine similarity') ...")
agno_results = agno_vdb.search("cosine similarity", limit=3)
for doc in agno_results:
    src = (doc.meta_data or {}).get("source", "?")
    print(f"  [{src:6s}] {doc.content}")

print(f"\n  -> Agno's search() found {len(agno_results)} docs (including LangChain-written ones)!")


# -- 6. SCENARIO C: Mixed search -- single query, both frameworks --------------

print("\n" + "=" * 68)
print("  SCENARIO C: Both write, both search -- results interleaved")
print("=" * 68)

print(f"\n[langchain] similarity_search('agents sharing memory', k=5) ...")
lc_mixed = lc_store.similarity_search_with_score("agents sharing memory", k=5)
print(f"\n  {'Rank':>4s}  {'Score':>6s}  Content")
print(f"  {'----':>4s}  {'------':>6s}  -------")
for i, (doc, score) in enumerate(lc_mixed, 1):
    text = doc.page_content[:60]
    suffix = "..." if len(doc.page_content) > 60 else ""
    print(f"  {i:4d}  {score:.4f}  {text}{suffix}")

print(f"\n[agno] search('agents sharing memory', limit=5) ...")
agno_mixed = agno_vdb.search("agents sharing memory", limit=5)
print(f"\n  {'Rank':>4s}  Content")
print(f"  {'----':>4s}  -------")
for i, doc in enumerate(agno_mixed, 1):
    text = doc.content[:60]
    suffix = "..." if len(doc.content) > 60 else ""
    print(f"  {i:4d}  {text}{suffix}")

print(f"\n  -> Both adapters return interleaved results from both frameworks!")


# -- 7. SCENARIO D: Agno user memories + LangChain history coexist -------------

print("\n" + "=" * 68)
print("  SCENARIO D: Agno memories + LangChain history (recursive list)")
print("=" * 68)

SHARED_USER_NS = "demo/shared/user/alice"

# Agno stores user memories
mem_client = A2MClient(A2M_URL, namespace=SHARED_USER_NS)
mem_db = A2MAgnoBaseDb(client=mem_client, embed_fn=embed)

print(f"\n[agno] Storing user memories in '{SHARED_USER_NS}' ...")
mem_db.upsert_user_memory(
    UserMemory(memory="User prefers concise answers",
               memory_id="style", user_id="alice", topics=["preferences"]))
mem_db.upsert_user_memory(
    UserMemory(memory="User speaks Italian and English",
               memory_id="lang", user_id="alice", topics=["language"]))
print("  [ok] 2 memories stored")

# LangChain stores chat history in a child namespace
hist_client = A2MClient(A2M_URL, namespace=f"{SHARED_USER_NS}/chat")
history = A2MLangChainBaseChatMessageHistory(client=hist_client)

print(f"\n[langchain] Storing chat history in '{SHARED_USER_NS}/chat' ...")
history.add_messages([
    HumanMessage(content="What is A2M?"),
    AIMessage(content="A2M is a shared memory protocol for AI agents."),
])
print("  [ok] 2 messages stored")

# Universal read: list ALL entries under the user namespace recursively
print(f"\n[universal] Listing all entries under '{SHARED_USER_NS}' (recursive=True) ...")
user_client = A2MClient(A2M_URL, namespace=SHARED_USER_NS)
resp = user_client.list(recursive=True, limit=50)

def _entry_text(e: dict) -> str:
    v = e.get("value", {})
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for key in ("memory", "content", "page_content"):
            if key in v:
                return str(v[key])
        # LangChain message dict
        data = v.get("data", {})
        if "content" in data:
            return str(data["content"])
    return str(v)[:60]

def _entry_source(e: dict) -> str:
    meta = e.get("meta", {})
    return meta.get("source_framework", "unknown")

print(f"\n  {'Source':12s}  {'Type':10s}  {'Key':30s}  Content")
print(f"  {'------':12s}  {'----':10s}  {'---':30s}  -------")
for e in resp["entries"]:
    fw = _entry_source(e)
    text = _entry_text(e)
    key = e.get("key", "?")
    content = text[:40]
    suffix = "..." if len(text) > 40 else ""
    print(f"  {fw:12s}  {e['type']:10s}  {key:30s}  {content}{suffix}")

print(f"\n  -> {resp['total']} total entries from both frameworks under one namespace tree!")


# -- 8. Summary ----------------------------------------------------------------

print("\n" + "=" * 68)
print("  HOW IT WORKS")
print("=" * 68)
print("""
  Both adapters use standardised a2m: category tags:
    Agno VectorDb  -> tags: [a2m:knowledge, hash:..., docid:...]
    LangChain VS   -> tags: [a2m:knowledge, {collection}, docid:...]
    Agno BaseDb    -> tags: [a2m:memory, user:...] / [a2m:session, ...]
    LangChain Hist -> tags: [a2m:history, role:...]

  Cross-framework sharing works NATIVELY at the adapter level:
    agno_vdb.search("query")              -> finds LangChain docs too
    lc_store.similarity_search("query")   -> finds Agno docs too

  Both _entry_to_doc() functions handle both value formats:
    Agno:      {"content": ..., "meta_data": {...}}
    LangChain: {"page_content": ..., "metadata": {...}}

  Requirements for cross-framework sharing:
    1. Same A2M namespace
    2. Same embedding model / function
    3. That's it -- the adapters handle the rest
""")

print("[demo] Done.")
_server.should_exit = True
