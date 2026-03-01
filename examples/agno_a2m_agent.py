"""
A2M + Agno — sample agent using both Relational and Vector backends.

This script is self-contained and requires no LLM API key for the A2M
demonstration. It shows the full wiring and then runs direct A2M operations
so you can observe both backends in action.

To run with a real LLM, uncomment the `model=` line in the Agent section
and set your OPENAI_API_KEY (or point OpenAILike at a local Ollama server).

Setup:
    pip install agno sentence-transformers   # plus the a2m stack
    python examples/agno_a2m_agent.py

What this demonstrates:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Relational backend  (A2MAgnoBaseDb)                               │
    │  - Stores structured user facts (UserMemory objects)                 │
    │  - Exact key/tag lookup — no embeddings needed                       │
    │  - Used by Agno's MemoryManager for persistent user memories         │
    │                                                                      │
    │  Vector backend  (A2MAgnoVectorDb + Knowledge)                       │
    │  - Stores document chunks with float embeddings                      │
    │  - Cosine-ranked semantic search over the knowledge base             │
    │  - Used by Agno's Knowledge for retrieval-augmented generation       │
    └──────────────────────────────────────────────────────────────────────┘
"""

import sys
import threading
import time
from pathlib import Path

# ── 1. Start the A2M server in-process ───────────────────────────────────────

import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))   # repo root
from server.main import create_app
from client.client import A2MClient

SERVER_PORT = 18800
app = create_app(db_path=":memory:")    # ephemeral store for this demo
_server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=SERVER_PORT, log_level="error"))
threading.Thread(target=_server.run, daemon=True).start()
time.sleep(0.8)

A2M_URL = f"http://127.0.0.1:{SERVER_PORT}"
print(f"[A2M] server running at {A2M_URL}\n")


# ── 2. Embedding function ─────────────────────────────────────────────────────
#
# Replace with a real embedder for production, e.g.:
#   from sentence_transformers import SentenceTransformer
#   _model = SentenceTransformer("all-MiniLM-L6-v2")
#   embed = lambda text: _model.encode(text).tolist()

import math

def embed(text: str) -> list[float]:
    """
    Toy 16-dim embedding: character n-gram frequency histogram, L2-normalised.
    Provides meaningful cosine similarity for demonstration without any ML deps.
    """
    dim = 16
    v = [0.0] * dim
    for i in range(len(text) - 1):
        bigram = ord(text[i]) * 256 + ord(text[i + 1])
        v[bigram % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


# ── 3. Agno imports ───────────────────────────────────────────────────────────

from agno.memory import UserMemory
from agno.memory.manager import MemoryManager
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.document import Document as AgnoDoc

from adapters.agno_basedb import A2MAgnoBaseDb
from adapters.agno_vectordb import A2MAgnoVectorDb


# ── 4. Relational backend — MemoryManager with A2MAgnoBaseDb ────────────────

print("=" * 60)
print("RELATIONAL BACKEND  — User Memories (A2MAgnoBaseDb)")
print("=" * 60)

mem_client = A2MClient(A2M_URL, namespace="demo/agno/memories")
mem_db     = A2MAgnoBaseDb(client=mem_client, embed_fn=embed)

# In a real setup you'd pass mem_db to MemoryManager and let the agent
# call it automatically.  Here we exercise the interface directly to show
# what happens under the hood.
#
#   manager = MemoryManager(db=mem_db)
#   agent   = Agent(model=..., memory_manager=manager, enable_user_memories=True)

# Store a few user memories (Relational write)
facts = [
    UserMemory(memory="User speaks Italian and English",
               memory_id="lang-pref", user_id="alice", topics=["language"]),
    UserMemory(memory="User is a machine-learning researcher",
               memory_id="profession", user_id="alice", topics=["work", "ML"]),
    UserMemory(memory="User dislikes verbose answers",
               memory_id="style-pref", user_id="alice", topics=["style"]),
]
print("\n[relational] Storing user memories …")
for f in facts:
    mem_db.upsert_user_memory(f)
    print(f"  [ok] {f.memory_id}: {f.memory!r}")

# Retrieve by user (Relational list + tag filter)
print("\n[relational] Retrieving memories for alice …")
retrieved = mem_db.get_user_memories(user_id="alice")
for m in retrieved:
    print(f"  - [{m.memory_id}] {m.memory}")

# Topic aggregation (Relational aggregate)
print("\n[relational] All topics:", mem_db.get_all_memory_topics(user_id="alice"))

# Stats
stats, count = mem_db.get_user_memory_stats(user_id="alice")
print(f"[relational] Memory stats: {count} entries")

# Semantic search over memories (Vector backend, requires embed_fn)
print("\n[vector] Semantic search over memories: 'preferred language' …")
query_embedding = embed("preferred language")
results = mem_db.search_user_memories(embedding=query_embedding, user_id="alice", limit=2)
for m in results:
    print(f"  -> {m.memory!r}")


# ── 5. Vector backend — Knowledge base with A2MAgnoVectorDb ──────────────────

print("\n" + "=" * 60)
print("VECTOR BACKEND  — Knowledge Base (A2MAgnoVectorDb)")
print("=" * 60)

vec_client = A2MClient(A2M_URL, namespace="demo/agno/knowledge")
vec_db     = A2MAgnoVectorDb(client=vec_client, embed_fn=embed)

# Build a small knowledge base
knowledge = Knowledge(vector_db=vec_db)

docs = [
    AgnoDoc(content="A2M is a shared memory protocol for AI agents across frameworks.",
            id="doc-1", name="a2m-overview", meta_data={"source": "spec"}),
    AgnoDoc(content="Embeddings are always caller-provided; the A2M server never generates them.",
            id="doc-2", name="embedding-policy", meta_data={"source": "spec"}),
    AgnoDoc(content="The A2M namespace hierarchy is: app / workflow / session / agent.",
            id="doc-3", name="namespaces", meta_data={"source": "spec"}),
    AgnoDoc(content="Agno agents can share state with LangChain agents via the A2M REST API.",
            id="doc-4", name="cross-framework", meta_data={"source": "blog"}),
    AgnoDoc(content="Vector search uses cosine similarity over caller-provided float embeddings.",
            id="doc-5", name="vector-search", meta_data={"source": "spec"}),
]

print("\n[vector] Loading documents into knowledge base …")
content_hash = "demo-kb-v1"
vec_db.insert(content_hash, docs)
print(f"  [ok] {len(docs)} documents indexed")

# Existence checks (Relational tag lookup)
print(f"\n[relational] content_hash_exists({content_hash!r}): {vec_db.content_hash_exists(content_hash)}")
print(f"[relational] name_exists('a2m-overview'):           {vec_db.name_exists('a2m-overview')}")
print(f"[relational] id_exists('doc-1'):                    {vec_db.id_exists('doc-1')}")

# Semantic search (Relational pre-filter -> Vector ranking)
queries = [
    "how are embeddings handled?",
    "how do agents share state?",
    "namespace structure",
]
print()
for q in queries:
    hits = vec_db.search(q, limit=2)
    print(f"[vector] search: {q!r}")
    for h in hits:
        print(f"  -> {h.content!r}")

# Delete one document (Relational)
vec_db.delete_by_id("doc-1")
print(f"\n[relational] After delete_by_id('doc-1'), id_exists: {vec_db.id_exists('doc-1')}")


# ── 6. Agent wiring (real LLM) ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("AGENT WIRING  (uncomment model= to run with a real LLM)")
print("=" * 60)

print(
    "To create a fully wired Agno agent using both A2M backends:\n"
    "\n"
    "    from agno.agent import Agent\n"
    "    from agno.memory.manager import MemoryManager\n"
    "    from agno.models.openai import OpenAIChat  # or Ollama, Anthropic, ...\n"
    "\n"
    "    agent = Agent(\n"
    "        # LLM\n"
    "        model=OpenAIChat(id='gpt-4o'),\n"
    "\n"
    "        # Relational backend: persistent user memories\n"
    "        memory_manager=MemoryManager(db=mem_db),\n"
    "        enable_user_memories=True,\n"
    "        add_memories_to_context=True,\n"
    "\n"
    "        # Vector backend: knowledge retrieval\n"
    "        knowledge=knowledge,\n"
    "        search_knowledge=True,\n"
    "\n"
    "        instructions=[\n"
    "            'Use your memory of the user to personalise answers.',\n"
    "            'Search the knowledge base before answering A2M questions.',\n"
    "        ],\n"
    "    )\n"
    "\n"
    "    agent.print_response('What is the A2M embedding policy?', user_id='alice')\n"
)

print("[demo] Done.")
_server.should_exit = True
