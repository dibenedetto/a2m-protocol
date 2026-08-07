"""Framework adapters: existing agents talking to an A2M server unmodified.

Nothing in this package is part of the protocol, and nothing in the protocol
depends on it. Each module imports one framework, and importing this package
imports none of them — so a checkout with no framework installed still runs
every test, every conformance target and both demos.

	adapters.langchain      LangChain chat history, retriever and key-value store
	adapters.agno           Agno vector database (the knowledge base)
	adapters.agno_db        Agno BaseDb (user memories) -- the other half Agno persists
	adapters.crewai         CrewAI memory storage
	adapters.autogen        AutoGen memory protocol and model context
	adapters.openai_agents  The OpenAI Agents SDK's Session -- the transcript

`openai_agents` is the one that imports nothing: the SDK declares `Session` as a
`runtime_checkable` `Protocol`, so having the four methods *is* being a session
and there is no import path to break. The suites assert the `isinstance` so the
claim is checked rather than assumed.

**Naming: `<Framework>A2M<WhatItImplements>`.** `AgnoA2MVectorDb`,
`LangChainA2MRetriever`, `AutoGenA2MMemory`, `OpenAIAgentsA2MSession`. The
framework comes first because that is the question a reader has — *whose
interface is this?* — and because the second half is not distinctive on its own:
five of these adapters implement something called a store, a memory or a session,
and `A2MMemory` in a memory protocol reads like the core class rather than
AutoGen's `Memory` on top of it. Long, and worth it: these names appear in an
import line and a constructor call, and nowhere else.

**Everything below the class is `_private`** — converters, role tables, text
extractors — however useful it looks. Two adapters both want a "the text of this
message" function and mean different types by it, so a public `text_of` would be
two incompatible functions competing for one name; `to_record` would be a third,
since [store_sqlite.py](../store_sqlite.py) already has one that means something
else again. The class carries the prefix, the module carries the underscore, and
between them there is nothing left to collide.

The point they make together is the one A2M exists for: frameworks with
incompatible memory models, pointed at one store, reading each other's records.
[examples/cross_framework.py](../../examples/cross_framework.py) is that claim at
the storage interfaces, and
[examples/agent_interop.py](../../examples/agent_interop.py) is the same claim one
level up, where three real agents share memory and the assertion is that another
framework's fact reached this one's *model call*.
"""
