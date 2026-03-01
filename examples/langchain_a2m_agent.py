"""
A2M + LangChain — sample RAG agent using both Relational and Vector backends.

This script is fully self-contained: it uses FakeListChatModel so no API key
is needed. Swap it for any real chat model (ChatOpenAI, ChatAnthropic, ...) and
the rest of the code is identical.

Setup:
    pip install langchain-core   # plus the a2m stack
    python examples/langchain_a2m_agent.py

What this demonstrates:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Relational backend  (A2MLangChainHistory)                           │
    │  - Stores every conversation turn as an A2M episodic entry           │
    │  - Exact retrieval by namespace — no embeddings needed               │
    │  - Used by RunnableWithMessageHistory for multi-turn memory          │
    │                                                                      │
    │  Vector backend  (A2MLangChainVectorStore)                           │
    │  - Stores document chunks with float embeddings                      │
    │  - Cosine-ranked similarity search for retrieval-augmented answers   │
    │  - Used as a retriever inside the RAG chain                          │
    └──────────────────────────────────────────────────────────────────────┘

Full chain data-flow:
    User question
        │
        ├─► embed_query ──► VectorStore.similarity_search ──► context docs   [Vector]
        │
        ├─► history.messages ────────────────────────────────► past turns     [Relational]
        │
        ▼
    ChatPromptTemplate  (system + history + context + question)
        │
        ▼
    ChatModel  (FakeListChatModel / ChatOpenAI / ...)
        │
        ▼
    AIMessage  ──► history.add_messages  ────────────────────► stored turn    [Relational]
"""

import math
import sys
import threading
import time
from pathlib import Path

# ── 1. Start the A2M server in-process ───────────────────────────────────────

import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))   # repo root
from server.main import create_app
from client.client import A2MClient

SERVER_PORT = 18801
app = create_app(db_path=":memory:")
_server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=SERVER_PORT, log_level="error"))
threading.Thread(target=_server.run, daemon=True).start()
time.sleep(0.8)

A2M_URL = f"http://127.0.0.1:{SERVER_PORT}"
print(f"[A2M] server running at {A2M_URL}\n")


# ── 2. Embedding implementation ───────────────────────────────────────────────
#
# Replace with a real provider for production, e.g.:
#   from langchain_openai import OpenAIEmbeddings
#   embeddings = OpenAIEmbeddings()
#
# Or with sentence-transformers:
#   from langchain_huggingface import HuggingFaceEmbeddings
#   embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_core.embeddings import Embeddings


class DemoEmbeddings(Embeddings):
    """
    Toy 16-dim bigram-frequency embeddings.  Sufficient to demonstrate cosine
    ranking without any ML dependencies.  Replace with a real Embeddings
    implementation for meaningful semantic similarity.
    """

    _DIM = 16

    def _encode(self, text: str) -> list[float]:
        v = [0.0] * self._DIM
        for i in range(len(text) - 1):
            v[(ord(text[i]) * 256 + ord(text[i + 1])) % self._DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


embeddings = DemoEmbeddings()


# ── 3. LangChain + A2M adapter imports ───────────────────────────────────────

from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from adapters.langchain_history import A2MLangChainHistory
from adapters.langchain_vectorstore import A2MLangChainVectorStore


# ── 4. Vector backend — knowledge base ───────────────────────────────────────

print("=" * 60)
print("VECTOR BACKEND  — Knowledge Base (A2MLangChainVectorStore)")
print("=" * 60)

kb_client = A2MClient(A2M_URL, namespace="demo/langchain/knowledge")
vector_store = A2MLangChainVectorStore(
    client=kb_client,
    embeddings=embeddings,
    collection_tag="a2m-docs",
)

corpus = [
    Document(page_content="A2M is a shared memory protocol for AI agents across frameworks.",
             metadata={"source": "spec", "section": "overview"}),
    Document(page_content="Embeddings are always caller-provided; the A2M server never generates them.",
             metadata={"source": "spec", "section": "embeddings"}),
    Document(page_content="The A2M namespace hierarchy is: app / workflow / session / agent.",
             metadata={"source": "spec", "section": "namespaces"}),
    Document(page_content="Agno and LangChain agents can share state via the A2M REST API.",
             metadata={"source": "blog", "section": "integration"}),
    Document(page_content="Vector search uses cosine similarity over caller-provided float embeddings.",
             metadata={"source": "spec", "section": "vector-search"}),
    Document(page_content="The /query endpoint returns entries ranked by cosine similarity score.",
             metadata={"source": "spec", "section": "api"}),
]

print(f"\n[vector] Indexing {len(corpus)} documents ...")
doc_ids = vector_store.add_documents(corpus)
print(f"  [ok] stored with ids: {[i[:8] + '...' for i in doc_ids]}")

# Quick search to verify (Relational pre-filter -> Vector ranking)
print("\n[vector] Similarity search: 'how does embedding work?' ...")
for hit in vector_store.similarity_search("how does embedding work?", k=2):
    print(f"  -> {hit.page_content!r}")

print("\n[vector] Similarity search with scores: 'namespace' ...")
for doc, score in vector_store.similarity_search_with_score("namespace", k=2):
    print(f"  [{score:.3f}] {doc.page_content!r}")

# Use the store as a retriever (standard LangChain interface)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# ── 5. Relational backend — conversation history ──────────────────────────────

print("\n" + "=" * 60)
print("RELATIONAL BACKEND  — Conversation History (A2MLangChainHistory)")
print("=" * 60)

def get_session_history(session_id: str) -> A2MLangChainHistory:
    """
    Factory used by RunnableWithMessageHistory.
    Each session_id gets its own A2M namespace, isolating histories.
    """
    session_client = A2MClient(A2M_URL, namespace=f"demo/langchain/history/{session_id}")
    return A2MLangChainHistory(client=session_client)


# ── 6. RAG chain ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RAG CHAIN  — full pipeline")
print("=" * 60)

# LLM — swap for ChatOpenAI(model="gpt-4o") or any other chat model
llm = FakeListChatModel(
    responses=[
        "A2M stands for AgentToMemory. It is a shared memory protocol that "
        "lets AI agents from different frameworks (LangChain, Agno, n8n...) "
        "read and write to a common store via a REST API.",

        "Embeddings are always provided by the caller. The A2M server stores "
        "and indexes them verbatim but never generates or replaces them. "
        "This keeps A2M model-agnostic.",

        "The namespace hierarchy is: app / workflow / session / agent. "
        "Entries are scoped to a namespace, and reads with recursive=true "
        "traverse child namespaces.",
    ]
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant with access to the A2M documentation.\n"
     "Use the retrieved context below to answer the user's question.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))


# Build the RAG chain:
#   question -> retrieve context (Vector) + inject history (Relational) -> LLM -> answer
rag_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"])))
    )
    | prompt
    | llm
    | StrOutputParser()
)

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


# ── 7. Multi-turn conversation ────────────────────────────────────────────────

questions = [
    "What is A2M?",
    "How does A2M handle embeddings?",
    "What is the namespace structure?",
]

session_cfg = {"configurable": {"session_id": "user-alice-session-1"}}

print()
for q in questions:
    print(f"  Human: {q}")
    answer = chain_with_history.invoke({"question": q}, config=session_cfg)
    print(f"  AI:    {answer}\n")

# Inspect the relational history directly
history = get_session_history("user-alice-session-1")
msgs = history.messages
print(f"[relational] {len(msgs)} messages stored in A2M for session-1:")
for m in msgs:
    role = "Human" if isinstance(m, HumanMessage) else "AI   "
    print(f"  [{role}] {m.content[:70]}{'...' if len(m.content) > 70 else ''}")


# ── 8. Cross-session retrieval demo ──────────────────────────────────────────

print("\n" + "=" * 60)
print("CROSS-SESSION  — second user, same knowledge base")
print("=" * 60)

# A different user (different session -> different history namespace)
# but same vector_store namespace -> shares the knowledge base
session2_cfg = {"configurable": {"session_id": "user-bob-session-1"}}

llm2 = FakeListChatModel(responses=[
    "Based on the context, agents from LangChain, Agno, and n8n can all "
    "share state by reading and writing to the same A2M namespace."
])
chain2 = (
    RunnablePassthrough.assign(
        context=RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"])))
    )
    | prompt
    | llm2
    | StrOutputParser()
)
chain2_with_history = RunnableWithMessageHistory(
    chain2,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

q2 = "Can agents from different frameworks share state?"
print(f"\n  Human (Bob): {q2}")
ans2 = chain2_with_history.invoke({"question": q2}, config=session2_cfg)
print(f"  AI:          {ans2}")

h2 = get_session_history("user-bob-session-1")
print(f"\n[relational] Bob's session has {len(h2.messages)} message(s) — isolated from Alice's.")
h1 = get_session_history("user-alice-session-1")
print(f"[relational] Alice's session still has {len(h1.messages)} message(s).")
print(f"[vector]     Both sessions share the same {len(corpus)}-document knowledge base.")

print("\n[demo] Done.")
_server.should_exit = True
