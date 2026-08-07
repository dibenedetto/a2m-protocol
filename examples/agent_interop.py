"""Three agent frameworks, three agents, one memory. Interoperability at the top.

	pip install openai-agents agno autogen-agentchat autogen-ext
	python examples/agent_interop.py

[cross_framework.py](cross_framework.py) makes the same argument one level down:
it drives each framework's *storage* interface by hand and shows the records
crossing. This file never touches the store. It builds three real agents — an
OpenAI Agents SDK `Agent`, an Agno `Agent`, an AutoGen `AssistantAgent` — hands
each one its own framework's memory object, and then asks the only question that
matters at this level:

	**did the other framework's fact reach this agent's model call?**

That is the check every assertion here makes. An agent's behaviour changes only
through what its model is given, so a shared store that never reaches the prompt
is a shared store that changed nothing. Each agent below is driven by a scripted
model that records exactly what it was handed, and the assertions read that
record. No API key, no network, no cost, and the same result every run.

The flow, and who does what to whom:

	1. an OpenAI SDK agent handles an incident        writes: its Session
	2. the incident closes, and percolates            spec §4.11 -- the protocol
	3. an Agno agent answers from that transcript     reads:  its Knowledge
	4. it files what it concluded                     writes: its VectorDb
	5. an AutoGen agent sees both of the above        reads:  its Memory
	6. and its own transcript lands in the store      writes: its ModelContext
	7. the OpenAI agent comes back and is caught up   reads:  the protocol

Step 7 is the interesting one. The OpenAI SDK's `Session` is a *transcript*
interface — `get_items`/`add_items`, no retrieval hook anywhere — so there is no
framework-shaped place to put "what everyone else learned". A2M has one anyway:
`recall` with a `prompt` option renders the ranked records into text (spec
§4.16), and text goes into instructions. The framework did not need a feature;
the protocol already had the method.

What makes any of this work is not the adapters. It is that all three sides agree
on what a record is (spec §3.1), that a conversation is *replayed* while
knowledge is *searched* (spec §4.4), and that a finished conversation percolates
rather than evaporating (spec §4.11). The adapters are thin because those
decisions were made in the right place.
"""


import asyncio
import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from   a2m                      import MemoryClient, MemoryServer
from   a2m.jsonrpc              import Client, LocalTransport
from   a2m.memory               import MemoryStack


from   agents                   import Agent as OpenAIAgent, Runner, set_tracing_disabled
from   agents.items             import ModelResponse
from   agents.memory            import Session
from   agents.models.interface  import Model as OpenAIModel
from   agents.usage             import Usage
from   agno.agent               import Agent as AgnoAgent
from   agno.knowledge.document  import Document
from   agno.knowledge.knowledge import Knowledge
from   agno.models.base         import Model as AgnoModel
from   agno.models.response     import ModelResponse as AgnoModelResponse
from   autogen_agentchat.agents import AssistantAgent
from   autogen_ext.models.replay import ReplayChatCompletionClient
from   openai.types.responses   import ResponseOutputMessage, ResponseOutputText

from   implementations.adapters.agno           import AgnoA2MVectorDb
from   implementations.adapters.autogen        import AutoGenA2MChatCompletionContext, AutoGenA2MMemory
from   implementations.adapters.openai_agents  import OpenAIAgentsA2MSession


PASSED = []
FAILED = []


# The three facts, one per agent, each distinctive enough that a lexical ranker
# retrieves it from a store holding all of them.
ROLLBACK = "roll back a bad deploy with `deploy --revert`, never by force-pushing main"
FALLBACK = "when eu-west-1 is the failing region, traffic moves to eu-central-1"
HANDOVER = "the on-call handover happens monday at 09:00 UTC in the incident channel"


def check(label: str, condition: bool, detail=None) -> None:
	"""Record one assertion.

	Args:
		label (str): What was being checked.
		condition (bool): Whether it held.
		detail (Any, optional): Printed when it did not.
	"""
	(PASSED if condition else FAILED).append(label)
	print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
	if not condition and detail is not None:
		print(f"        {detail}")


# --------------------------------------------------------------- scripted models
#
# One per framework, each about twenty lines, each doing two things: return a
# fixed reply, and remember what it was asked. The second is the point. These are
# not mocks standing in for a model that would otherwise be called -- they are the
# instrument the assertions read, because "the fact reached the model" is not
# observable from outside the model.


class ScriptedOpenAIModel(OpenAIModel):
	"""An OpenAI Agents SDK model that replies from a script and records its input.
	"""

	def __init__(self, replies) -> None:
		"""Load the script.

		Args:
			replies (Sequence[str]): Returned in order, one per call.
		"""
		self.replies = list(replies)
		self.seen    = []


	async def get_response(self, system_instructions, input, model_settings, tools,
	                       output_schema, handoffs, tracing, **kwargs) -> ModelResponse:
		"""Answer, and keep what was asked.

		Args:
			system_instructions: The agent's instructions.
			input: The conversation, in Responses format.
			model_settings: Ignored.
			tools: Ignored.
			output_schema: Ignored.
			handoffs: Ignored.
			tracing: Ignored.
			**kwargs: previous_response_id, conversation_id, prompt -- ignored.

		Returns:
			ModelResponse: The next scripted reply.
		"""
		self.seen.append((system_instructions or "", input))

		text = self.replies.pop(0) if self.replies else "ok"
		return ModelResponse(
			output = [ResponseOutputMessage(
				id      = f"msg_{len(self.seen)}",
				role    = "assistant",
				status  = "completed",
				type    = "message",
				content = [ResponseOutputText(text=text, type="output_text", annotations=[])],
			)],
			usage       = Usage(),
			response_id = None,
		)


	def stream_response(self, *args, **kwargs):
		"""Not used here.

		Raises:
			NotImplementedError: Always. Nothing in this example streams, and a
				fake that pretended to would be a fake of the wrong thing.
		"""
		raise NotImplementedError


class ScriptedAgnoModel(AgnoModel):
	"""An Agno model that replies from a script and records its input.
	"""

	def __init__(self, replies) -> None:
		"""Load the script.

		Args:
			replies (Sequence[str]): Returned in order, one per call.
		"""
		super().__init__(id="scripted")

		self.replies = list(replies)
		self.seen    = []


	def invoke(self, **kwargs) -> AgnoModelResponse:
		"""Answer, and keep the messages that were built for this call.

		Args:
			**kwargs: Agno passes messages, tools, response_format and more.

		Returns:
			ModelResponse: The next scripted reply.
		"""
		self.seen.append(list(kwargs.get("messages") or []))

		return AgnoModelResponse(role="assistant",
		                         content=self.replies.pop(0) if self.replies else "ok")


	async def ainvoke(self, **kwargs) -> AgnoModelResponse:
		"""Answer asynchronously.

		Args:
			**kwargs: As invoke.

		Returns:
			ModelResponse: The next scripted reply.
		"""
		return self.invoke(**kwargs)


	def invoke_stream(self, **kwargs):
		"""Answer as a one-chunk stream.

		Args:
			**kwargs: As invoke.

		Yields:
			ModelResponse: The whole reply, once.
		"""
		yield self.invoke(**kwargs)


	async def ainvoke_stream(self, **kwargs):
		"""Answer as a one-chunk stream, asynchronously.

		Args:
			**kwargs: As invoke.

		Yields:
			ModelResponse: The whole reply, once.
		"""
		yield self.invoke(**kwargs)


	def _parse_provider_response(self, response, **kwargs) -> AgnoModelResponse:
		"""Already Agno's own shape.

		Args:
			response: What invoke returned.
			**kwargs: Ignored.

		Returns:
			ModelResponse: Unchanged.
		"""
		return response


	def _parse_provider_response_delta(self, response) -> AgnoModelResponse:
		"""Already Agno's own shape.

		Args:
			response: What invoke_stream yielded.

		Returns:
			ModelResponse: Unchanged.
		"""
		return response


class ScriptedAutoGenClient(ReplayChatCompletionClient):
	"""AutoGen's own replay client, plus a record of what it was asked.
	"""

	def __init__(self, replies) -> None:
		"""Load the script.

		Args:
			replies (Sequence[str]): Returned in order, one per call.
		"""
		super().__init__(list(replies))

		self.seen = []


	async def create(self, messages, **kwargs):
		"""Answer, and keep the messages.

		Args:
			messages: The model context as AutoGen assembled it.
			**kwargs: Passed through to the replay client.

		Returns:
			CreateResult: The next scripted reply.
		"""
		self.seen.append(list(messages))
		return await super().create(messages, **kwargs)


def said_to(model, needle: str) -> bool:
	"""Whether a string reached a scripted model on its most recent call.

	This is the assertion the whole file is built around: not "the store holds
	it" but "the model was given it".

	Args:
		model: Any of the three scripted models above.
		needle (str): What to look for.

	Returns:
		bool: True if the last call's input contains it anywhere.
	"""
	return needle in str(model.seen[-1]) if model.seen else False


# ------------------------------------------------------------------- the example


def searchable(memory: MemoryClient) -> str:
	"""The first tier a recall can reach.

	Working memory is replayed, never searched (spec §4.4), so knowledge written
	there is knowledge no agent will ever find.

	Args:
		memory (MemoryClient): A connected client.

	Returns:
		str: A tier name.
	"""
	return next(tier["name"] for tier in memory.describe()["tiers"]
	            if tier.get("kind") != "working")


async def run(memory: MemoryClient) -> None:
	"""Drive all three agents against one store.

	Args:
		memory (MemoryClient): The shared A2M client. Every framework below gets
			this same object and nothing else in common.
	"""
	tier = searchable(memory)

	# ------------------------------------------- 1. the OpenAI SDK agent works
	print("  an openai-agents agent handles an incident")

	ops_model   = ScriptedOpenAIModel([f"understood -- {ROLLBACK}"])
	ops_session = OpenAIAgentsA2MSession(memory, "incident-77")
	ops         = OpenAIAgent(name="ops", instructions="You are an incident responder.",
	                          model=ops_model)

	check("the sdk accepts an a2m store as a session", isinstance(ops_session, Session))

	await Runner.run(ops, f"remember this: {ROLLBACK}", session=ops_session)

	turns = memory.timeline(where={"session": "incident-77"})
	check("the agent's turns are in the shared store", len(turns) == 2, turns)

	items = await ops_session.get_items()
	check("and they replay as responses items, verbatim",
	      any(item.get("type") == "message" and item.get("status") == "completed"
	          for item in items), items)

	# Working memory is replayed, not searched. A live conversation is not yet
	# knowledge, and an agent looking for one here would correctly find nothing.
	check("a live conversation is not yet knowledge",
	      not any(ROLLBACK in record["content"] for record in memory.recall("roll back deploy", tier=tier)))

	# ---------------------------------------- 2. the incident ends and percolates
	print("\n  the incident closes")

	memory.close_session("incident-77")

	# spec §4.11 -- closing percolates the whole stack rather than one hop. This
	# is the step no framework has, and it is the one that turns a transcript
	# into something another agent can find.
	check("a closed conversation becomes searchable knowledge",
	      any(ROLLBACK in record["content"] for record in memory.recall("roll back deploy", tier=tier)))

	# --------------------------------------------------- 3. an Agno agent reads it
	print("\n  an agno agent answers from it")

	# namespace=None: this knowledge base writes its own records and searches the
	# whole store. Scoped, it could only find what it wrote -- which is the silo
	# A2M exists to remove (DECISION 023).
	vectors   = AgnoA2MVectorDb(memory, namespace=None, tier=tier)
	knowledge = Knowledge(vector_db=vectors)

	sre_model = ScriptedAgnoModel(["revert the deploy, then page the release owner"])
	sre       = AgnoAgent(name="sre", model=sre_model, knowledge=knowledge,
	                      search_knowledge=False, add_knowledge_to_context=True)

	sre.run("how do we roll back a bad deploy?")

	check("the openai agent's fact reached agno's model call",
	      said_to(sre_model, ROLLBACK), sre_model.seen[-1] if sre_model.seen else None)

	# Nothing about Agno was modified to make that true. `Knowledge` is stock, the
	# agent is stock; only the vector database underneath is an A2M server, and it
	# is holding a record the OpenAI SDK wrote through a completely different
	# interface.

	# --------------------------------------------- 4. and files what it concluded
	print("\n  and files what it concluded")

	vectors.upsert("runbook-fallback", [Document(
		content    = FALLBACK,
		content_id = "runbook-fallback",
		name       = "fallback-region",
	)])

	# spec §3.6 -- upsert writes to a key, so revising this runbook later replaces
	# it instead of leaving the stale version recallable beside the correction.
	check("agno's document is addressable", bool(memory.fetch("runbook-fallback")))

	# ------------------------------------------------ 5. an AutoGen agent sees both
	print("\n  an autogen agent sees both")

	remembered  = AutoGenA2MMemory(memory, namespace=None, tier=tier, limit=5)
	transcript  = AutoGenA2MChatCompletionContext(memory, session="review-3")
	analyst_llm = ScriptedAutoGenClient([f"noted. also: {HANDOVER}"])
	analyst     = AssistantAgent(name="analyst", model_client=analyst_llm,
	                             memory=[remembered], model_context=transcript)

	await analyst.run(task="how do we roll back, and where does traffic go?")

	check("the openai agent's fact reached autogen's model call",
	      said_to(analyst_llm, ROLLBACK), analyst_llm.seen[-1] if analyst_llm.seen else None)
	check("and so did the agno agent's",
	      said_to(analyst_llm, FALLBACK), analyst_llm.seen[-1] if analyst_llm.seen else None)

	# namespace=None again, and for the same reason: a namespace here scopes what
	# this memory writes and what `clear` may delete, and it would scope what it
	# can find. An agent that can only recall its own writes has a private store
	# with extra steps (DECISION 023), which is the failure this file is testing
	# for -- so the escape hatch is the setting, not the exception.

	# ----------------------------------------------- 6. and writes its own turns
	print("\n  and its own conversation lands in the store")

	review = memory.timeline(where={"session": "review-3"})
	check("autogen's transcript is in the shared store", len(review) >= 2, review)

	# Nothing is written by hand here. The analyst's reply carries the handover
	# fact, it is in working memory because that is where a transcript lives, and
	# closing the session is what makes it findable -- the same percolation the
	# OpenAI agent's incident went through in step 2, for an agent that shares no
	# code with it.
	memory.close_session("review-3")

	# ------------------------------------- 7. the openai agent, caught up by a2m
	print("\n  the openai agent returns, and is caught up")

	# The SDK's Session is get_items/add_items -- a transcript interface with no
	# retrieval hook anywhere in it. So the catching-up does not go through the
	# framework at all: the protocol renders the ranked records into text (spec
	# §4.16) and text is something every framework already takes.
	briefing, cited = memory.recall_prompt("roll back deploy, fallback region, on-call handover",
	                                       budget=800, limit=5, tier=tier)

	check("the protocol renders a briefing the framework has no hook for",
	      bool(briefing.strip()) and bool(cited), briefing)
	check("the briefing carries what the other two agents learned",
	      "deploy --revert" in briefing and "eu-central-1" in briefing, briefing)

	back      = ScriptedOpenAIModel(["rolling back now"])
	ops_again = OpenAIAgent(name="ops", model=back,
	                        instructions=f"You are an incident responder.\n\n{briefing}")

	await Runner.run(ops_again, "eu-west-1 is failing again. what do we do?",
	                 session=OpenAIAgentsA2MSession(memory, "incident-78"))

	check("the agno agent's fact reached the openai agent's model call",
	      said_to(back, "eu-central-1"), back.seen[-1] if back.seen else None)
	check("and so did the autogen agent's",
	      said_to(back, "09:00 UTC"), back.seen[-1] if back.seen else None)

	# ------------------------------------------------------------ 8. the store
	print("\n  one store, three frameworks")

	everything = memory.timeline(tier=tier, limit=0)
	contents   = " ".join(record.get("content", "") for record in everything)

	check("every framework's fact is in one tier",
	      all(fact in contents for fact in ("deploy --revert", "eu-central-1", "09:00 UTC")),
	      contents[:300])

	# The scores that ordered every recall above are ranking only (spec §5.3):
	# incomparable between calls, servers and thresholds. Nothing here compares
	# one, which is why nothing here breaks when the server's ranker changes.


def main() -> int:
	"""Run the demonstration.

	Returns:
		int: 0 when every check passed.
	"""
	set_tracing_disabled(True)

	print("\n  one A2M server, three agent frameworks\n")

	server = MemoryServer(MemoryStack())
	memory = MemoryClient(Client(LocalTransport(server.dispatcher)))

	asyncio.run(run(memory))

	print(f"\n  {len(PASSED)} passed, {len(FAILED)} failed\n")
	return 1 if FAILED else 0


if __name__ == "__main__":
	sys.exit(main())
