"""Self-contained checks for A2M and its reference stack.

	python test_a2m.py

No test runner and no network: the embedding scorer is exercised through a stub
lookup table, so the suite is deterministic and runs offline.

This checks the *implementation*. To check that a server honours the protocol --
including one written in another language -- use tools/conformance.py, which
speaks only A2M and never imports what it is testing.
"""


import json
import sys
import threading
import urllib.error
import urllib.request


from   typing        import Any, Callable


from   a2m           import MemoryServer, connect_http, connect_local, connect_stdio, serve_a2m_http
from   a2m.jsonrpc   import Client, Dispatcher, JsonRpcError, LocalTransport
from   a2m.memory    import MemoryStack, MemoryTier, from_rfc3339, to_rfc3339
from   a2m.retrieval import EmbeddingScorer, HybridScorer, LexicalScorer, cosine, llm_consolidator, make_scorer
from   a2m.text      import detect, tokenize


PASSED = []
FAILED = []


def check(label: str, condition: bool, detail: Any = "") -> None:
	if condition:
		PASSED.append(label)
		print(f"  ok    {label}")
	else:
		FAILED.append(label)
		print(f"  FAIL  {label}  {detail}")


def test_jsonrpc() -> None:
	print("== jsonrpc ==")

	dispatcher = Dispatcher()
	dispatcher.register("add" , lambda a, b: a + b)
	dispatcher.register("boom", lambda: 1 / 0)

	call = lambda payload: dispatcher.handle(payload)

	check("named params"      , call({"jsonrpc": "2.0", "id": 1, "method": "add", "params": {"a": 2, "b": 3}})["result"] == 5)
	check("positional params" , call({"jsonrpc": "2.0", "id": 1, "method": "add", "params": [2, 3]})["result"] == 5)
	check("method not found"  , call({"jsonrpc": "2.0", "id": 1, "method": "nope"})["error"]["code"] == -32601)
	check("invalid params"    , call({"jsonrpc": "2.0", "id": 1, "method": "add", "params": {"a": 1}})["error"]["code"] == -32602)
	check("handler failure"   , call({"jsonrpc": "2.0", "id": 1, "method": "boom"})["error"]["code"] == -32603)
	check("wrong version"     , call({"jsonrpc": "1.0", "id": 1, "method": "add"})["error"]["code"] == -32600)
	check("notifications are silent", call({"jsonrpc": "2.0", "method": "add", "params": [1, 2]}) is None)
	check("batch", len(call([
		{"jsonrpc": "2.0", "id": 1, "method": "add", "params": [1, 1]},
		{"jsonrpc": "2.0", "id": 2, "method": "add", "params": [2, 2]},
	])) == 2)

	client = Client(LocalTransport(dispatcher))
	check("client unwraps results", client.call("add", {"a": 4, "b": 5}) == 9)

	try:
		client.call("nope")
		check("client raises on error", False)
	except JsonRpcError as exc:
		check("client raises on error", exc.code == -32601)


def test_recall() -> None:
	print("== recall ==")

	stack = MemoryStack()
	stack.remember("marco lives in bologna and works on compilers")
	stack.remember("the deploy key rotates every 90 days")
	stack.remember("pasta al ragu needs three hours")

	hits = stack.recall("where does marco live?")
	check("ranks the relevant record first", hits and "bologna" in hits[0][0].content, hits)
	check("drops records with no overlap"  , all("pasta" not in r.content for r, _ in hits))
	check("recall counts as access"        , hits[0][0].access_count == 1)
	check("empty query returns everything" , len(stack.recall(limit=0)) == 3)

	check("timeline is chronological", [r.content[:5] for r in stack.timeline()] == ["marco", "the d", "pasta"])
	check("forget by query"          , stack.forget(query="pasta ragu") == 1)
	check("record is gone"           , len(stack.records) == 2)


def test_consolidation() -> None:
	print("== consolidation ==")

	stack = MemoryStack(tiers=[
		MemoryTier("working" , capacity = 4, spill_to = "episodic"),
		MemoryTier("episodic", capacity = 3, spill_to = "semantic"),
		MemoryTier("semantic", capacity = 0),
	])
	for i in range(10):
		stack.remember(f"note number {i}", group=f"group{i // 2}")

	report = stack.consolidate()
	check("top tier respects capacity", len(stack.records_in("working")) <= 4, report)
	check("nothing is lost"           , len(stack.records) == 10, report)

	split = [
		group for group in {r.group for r in stack.records.values()}
		if len({r.tier for r in stack.records.values() if r.group == group}) > 1
	]
	check("groups are never split across tiers", not split, split)

	bottom = MemoryStack(tiers=[MemoryTier("working", capacity=2)])
	for i in range(5):
		bottom.remember(f"x{i}")
	report = bottom.consolidate()
	check("the bottom tier drops what overflows", len(bottom.records) == 2 and report["dropped"] == 3, report)

	summarizing = MemoryStack(
		tiers          = [MemoryTier("working", capacity=1, spill_to="episodic"), MemoryTier("episodic")],
		consolidate_fn = lambda records, target: [f"summary of {len(records)} records"],
	)
	for i in range(4):
		summarizing.remember(f"y{i}", group="one")
	summarizing.consolidate()
	check("consolidate_fn rewrites what it spills", any("summary of" in r.content for r in summarizing.records.values()))


def test_kinds() -> None:
	print("== tier kinds ==")

	stack = MemoryStack()
	check("four canonical layers", [stack.tiers[n].kind for n in stack.order]
	      == ["working", "episodic", "semantic", "procedural"], stack.order)
	check("working is found by kind", stack.working == "working")
	check("nothing spills into procedural",
	      all(stack.tiers[n].spill_to != "procedural" for n in stack.order),
	      {n: stack.tiers[n].spill_to for n in stack.order})
	check("procedural is terminal", stack.tiers["procedural"].spill_to is None)

	# A tier can be called anything as long as it declares what it is for.
	renamed = MemoryStack(tiers=[
		MemoryTier("context", kind = "working"   , capacity = 2, spill_to = "history"),
		MemoryTier("history", kind = "episodic"  , capacity = 0),
		MemoryTier("habits" , kind = "procedural", capacity = 0),
	])
	check("a renamed working tier is still found", renamed.working == "context")
	check("kind lookup ignores names", renamed.of_kind("procedural") == ["habits"])
	check("writes default to the working tier", renamed.remember("hello").tier == "context")

	try:
		MemoryTier("scratch")
		check("an unknown kind is rejected", False)
	except ValueError:
		check("an unknown kind is rejected", True)

	try:
		MemoryStack(tiers=[MemoryTier("working", promote_to="nowhere")])
		check("promoting into an unknown tier is rejected", False)
	except ValueError:
		check("promoting into an unknown tier is rejected", True)


def test_promotion() -> None:
	print("== promotion ==")

	stack = MemoryStack(tiers=[
		MemoryTier("working" , capacity = 0),
		MemoryTier("episodic", capacity = 0, spill_to = "semantic", promote_to = "semantic", promote_after = 3),
		MemoryTier("semantic", capacity = 0),
	])

	kept    = stack.remember("the release branch is cut on thursdays", tier="episodic")
	ignored = stack.remember("someone mentioned the weather", tier="episodic")

	for _ in range(3):
		stack.recall("when is the release branch cut")

	report = stack.consolidate()
	check("a repeatedly recalled record is promoted", stack.get(kept.id).tier == "semantic", report)
	check("promotion is counted"                    , report["promoted"] == 1, report)
	check("an unused record stays put"              , stack.get(ignored.id).tier == "episodic")

	# Promotion runs before spilling, so volume cannot displace what was earned.
	pressured = MemoryStack(tiers=[
		MemoryTier("working" , capacity = 2, spill_to = "episodic", promote_to = "semantic", promote_after = 1),
		MemoryTier("episodic", capacity = 0),
		MemoryTier("semantic", capacity = 0),
	])
	earned = pressured.remember("the api key is in vault/prod")
	pressured.recall("where is the api key")
	for i in range(6):
		pressured.remember(f"filler {i}")

	pressured.consolidate()
	check("what was earned is not displaced by volume", pressured.get(earned.id).tier == "semantic",
	      pressured.get(earned.id).tier)

	stack = MemoryStack()
	record = stack.remember("prefer tabs over spaces in this repo")
	check("promote moves the record" , stack.promote([record.id], "procedural") == 1)
	check("the record is procedural" , stack.get(record.id).tier == "procedural")
	check("promotion raises salience", stack.get(record.id).salience > 1.0)
	check("promoting twice is a no-op", stack.promote([record.id], "procedural") == 0)


def stub_embedder(table: dict[str, list[float]], default: list[float] = None) -> Callable:
	"""A lookup-table embedder, so retrieval can be tested without a model."""
	default = default or [0.0, 0.0, 1.0]

	def embed(texts: list[str]) -> list[list[float]]:
		return [list(table.get(str(t), default)) for t in texts]

	return embed


def test_scorers() -> None:
	print("== scorers ==")

	check("cosine of identical vectors" , abs(cosine([1.0, 2.0], [1.0, 2.0]) - 1.0) < 1e-9)
	check("cosine of orthogonal vectors", abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9)
	check("cosine tolerates empty input", cosine([], [1.0]) == 0.0)
	check("cosine tolerates a zero vector", cosine([0.0, 0.0], [1.0, 1.0]) == 0.0)

	stack = MemoryStack()
	one   = stack.remember("the deploy key rotates every 90 days")
	two   = stack.remember("pasta al ragu needs three hours")

	lexical = LexicalScorer()
	for record in (one, two):
		lexical.index(record)

	check("a scorer abstains on an empty query", lexical.relevance("", [one, two]) is None)
	check("a scorer abstains on stopwords only", lexical.relevance("the and of", [one, two]) is None)

	scores = lexical.relevance("deploy key", [one, two])
	check("a judged query returns only matches", list(scores) == [one.id], scores)
	check("relevance is bounded to 0..1", 0.0 < scores[one.id] <= 1.0, scores)

	# Stopwords are a parameter, not a law.
	permissive = LexicalScorer(stopwords=frozenset(), min_length=1)
	for record in (one, two):
		permissive.index(record)
	check("a caller can drop the stopword list",
	      permissive.relevance("the", [one, two]) is not None,
	      permissive.relevance("the", [one, two]))

	# Embeddings find what shares no words at all.
	table = {
		"the user is called marco" : [1.0, 0.0, 0.0],
		"the office has a balcony" : [0.0, 1.0, 0.0],
		"what is my name?"         : [1.0, 0.0, 0.0],
	}
	semantic = MemoryStack(scorer=EmbeddingScorer(stub_embedder(table)))
	named    = semantic.remember("the user is called marco")
	semantic.remember("the office has a balcony")

	lexical_only = MemoryStack()
	lexical_only.remember("the user is called marco")
	lexical_only.remember("the office has a balcony")

	check("lexical finds nothing without shared words",
	      lexical_only.recall("what is my name?") == [],
	      lexical_only.recall("what is my name?"))
	hits = semantic.recall("what is my name?")
	check("embeddings find it anyway", hits and hits[0][0].id == named.id, hits)

	scorer = EmbeddingScorer(stub_embedder(table))
	scorer.relevance("what is my name?", [named])
	check("vectors are cached after the first look", named.id in scorer.vectors)
	scorer.drop(named.id)
	check("dropping a record drops its vector", named.id not in scorer.vectors)
	check("the embedding scorer abstains on an empty query", scorer.relevance("", [named]) is None)

	# Hybrid is a union: what either scorer found survives.
	hybrid = MemoryStack(scorer=HybridScorer([
		(LexicalScorer()                        , 0.5),
		(EmbeddingScorer(stub_embedder(table))  , 0.5),
	]))
	balcony = hybrid.remember("the office has a balcony")
	marco   = hybrid.remember("the user is called marco")
	found   = {r.id for r, _ in hybrid.recall("what is my name?")}
	check("hybrid keeps what only embeddings found", marco.id in found, found)
	check("hybrid stays bounded to 0..1", all(0.0 <= s <= 1.0 for _, s in hybrid.recall("what is my name?")))

	check("hybrid abstains only when every part does",
	      HybridScorer([LexicalScorer()]).relevance("", [balcony]) is None)

	check("scorers are selectable by name", type(make_scorer("lexical")).__name__ == "LexicalScorer")
	check("the stack reports which scorer it uses", MemoryStack().describe()["scorer"]["scorer"] == "lexical")

	try:
		make_scorer("telepathy")
		check("an unknown scorer is refused", False)
	except ValueError:
		check("an unknown scorer is refused", True)


def test_language() -> None:
	print("== languages ==")

	check("accents survive"          , tokenize("la città")        == ["città"], tokenize("la città"))
	check("umlauts survive"          , "köln" in tokenize("die Stadt Köln"), tokenize("die Stadt Köln"))
	check("eszett survives"          , "straßen" in tokenize("schöne Straßen"))
	check("tildes survive"           , tokenize("el niño") == ["niño"], tokenize("el niño"))
	check("cyrillic is indexed"      , tokenize("меня зовут марко") == ["меня", "зовут", "марко"])
	check("greek is indexed"         , "λένε" in tokenize("με λένε Μάρκο"))

	# The accent bug that made two spellings of one word not match.
	check("accented and bare spellings no longer collide",
	      tokenize("città") != tokenize("citta") and tokenize("città") == ["città"])

	check("cjk becomes bigrams"      , tokenize("我叫马可") == ["我叫", "叫马", "马可"], tokenize("我叫马可"))
	check("a lone ideograph survives", tokenize("猫") == ["猫"], tokenize("猫"))
	check("mixed scripts split"      , tokenize("gpu技术") == ["gpu", "技术"], tokenize("gpu技术"))
	check("japanese is segmented"    , len(tokenize("私の名前はマルコです")) > 3)

	check("italian is detected", detect("mi chiamo marco e abito a bologna") == "it")
	check("english is detected", detect("my name is marco and i live in bologna") == "en")
	check("an unregistered language falls back", detect("меня зовут марко") is None)

	# The English list used to delete Italian verbs and nouns.
	for word, gloss in [("so", "I know"), ("do", "I give"), ("can", "dog"), ("me", "me")]:
		kept = word in tokenize(f"io {word} la risposta con questo")
		check(f"italian '{word}' ({gloss}) survives", kept, tokenize(f"io {word} la risposta con questo"))

	noise = ["il", "lo", "la", "di", "che", "per", "con", "non", "sono", "una", "del", "ho", "ha"]
	check("italian function words are filtered",
	      tokenize("il lo la di che per con non sono una del ho ha marco") == ["marco"],
	      tokenize(" ".join(noise) + " marco"))

	check("an explicit list wins"  , tokenize("the cat", stopwords=frozenset({"cat"})) == ["the"])
	check("an empty set filters nothing", tokenize("the cat", stopwords=frozenset()) == ["the", "cat"])
	# Pinning the language changes which words count as noise, in both directions.
	check("pinned english deletes 'do'", tokenize("do it", language="en") == [], tokenize("do it", language="en"))
	check("pinned italian keeps it"    , tokenize("do it", language="it") == ["do", "it"])

	# End to end, in Italian.
	stack = MemoryStack()
	stack.remember("mi chiamo marco")
	stack.remember("abito a bologna")
	stack.remember("la chiave di deploy ruota ogni 90 giorni")

	hits = stack.recall("come mi chiamo?")
	check("italian recall finds the right record", hits and hits[0][0].content == "mi chiamo marco", hits)
	hits = stack.recall("dove abito?")
	check("italian recall follows the question", hits and hits[0][0].content == "abito a bologna", hits)

	# This used to return the deploy-key record, matched on the article "la".
	check("an article no longer produces a false match", stack.recall("qual e la citta?") == [],
	      stack.recall("qual e la citta?"))

	# End to end, in a script that used to index to nothing at all.
	cyrillic = MemoryStack()
	cyrillic.remember("меня зовут марко")
	cyrillic.remember("я живу в болонье")
	hits = cyrillic.recall("как меня зовут?")
	check("cyrillic recall works", len(hits) == 1 and "зовут" in hits[0][0].content, hits)

	# The diagnostic that would have made the original failure loud.
	scorer  = LexicalScorer()
	healthy = MemoryStack(scorer=scorer)
	healthy.remember("меня зовут марко")
	check("nothing is silently unindexed", scorer.describe()["unindexed"] == 0, scorer.describe())

	blind  = LexicalScorer(stopwords=frozenset({"marco"}))
	silent = MemoryStack(scorer=blind)
	silent.remember("marco")
	check("a record that indexes to nothing is counted", blind.describe()["unindexed"] == 1, blind.describe())
	check("the count is reported by the stack", silent.describe()["scorer"]["unindexed"] == 1)


def test_llm_consolidation() -> None:
	print("== llm consolidation ==")

	class Summarizer:
		def __init__(self, reply: str) -> None:
			self.reply = reply
			self.calls = 0

		def query(self, system: str = None, messages: list = None, **kwargs) -> dict[str, Any]:
			self.calls += 1
			return {"role": "assistant", "content": self.reply, "tool_calls": None}

	model = Summarizer("- The user is Marco.\n- Marco lives in Bologna.")
	stack = MemoryStack(
		tiers          = [MemoryTier("working", capacity=1, spill_to="episodic"),
		                  MemoryTier("episodic", capacity=1, spill_to="semantic"),
		                  MemoryTier("semantic", capacity=0)],
		consolidate_fn = llm_consolidator(model),
	)
	for text in ["i am marco", "i live in bologna", "unrelated chatter", "more chatter"]:
		stack.remember(text, group="turn")

	report   = stack.consolidate()
	contents = [r.content for r in stack.records_in("semantic")]
	check("records are distilled on the way into semantic", "The user is Marco." in contents, contents)
	check("distillation is reported", report["summarized"] >= 1, report)
	check("the working spill stayed free", model.calls >= 1)

	# A summariser that says nothing must never destroy the originals.
	silent = MemoryStack(
		tiers          = [MemoryTier("working", capacity=1, spill_to="semantic"), MemoryTier("semantic")],
		consolidate_fn = llm_consolidator(Summarizer("NOTHING")),
	)
	silent.remember("keep me")
	silent.remember("and me")
	silent.consolidate()
	check("an empty summary falls back to moving the records intact",
	      sorted(r.content for r in silent.records.values()) == ["and me", "keep me"],
	      [r.content for r in silent.records.values()])

	class Broken:
		def query(self, **kwargs):
			raise RuntimeError("model is down")

	broken = MemoryStack(
		tiers          = [MemoryTier("working", capacity=1, spill_to="semantic"), MemoryTier("semantic")],
		consolidate_fn = llm_consolidator(Broken()),
	)
	broken.remember("survive one")
	broken.remember("survive two")
	broken.consolidate()
	check("a failing summariser loses nothing", len(broken.records) == 2, len(broken.records))


def test_concurrency() -> None:
	print("== concurrency ==")

	stack  = MemoryStack(tiers=[MemoryTier("working", capacity=50, spill_to="episodic"),
	                            MemoryTier("episodic", capacity=0)])
	errors : list[str] = []

	def hammer(n: int) -> None:
		try:
			for i in range(200):
				stack.remember(f"agent {n} record {i}")
				stack.consolidate()
				stack.recall("record")
		except Exception as exc:
			errors.append(repr(exc))

	threads = [threading.Thread(target=hammer, args=(n,)) for n in range(4)]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	check("concurrent agents do not crash the stack", not errors, errors[:2])
	check("no writes are lost", len(stack.records) == 800, len(stack.records))
	check("capacity still holds under contention", len(stack.records_in("working")) <= 50)


def test_sharing() -> None:
	print("== shared memory ==")

	stack = MemoryStack()
	check("the transcript tier is private", stack.tiers["working"].shared is False)
	check("everything below is shared",
	      all(stack.tiers[n].shared for n in ("episodic", "semantic", "procedural")))

	shared = connect_local(stack)
	alice  = shared.for_agent("alice")
	bob    = shared.for_agent("bob")

	check("scoping shares the transport", alice.client is bob.client)
	check("a scoped client knows its agent", (alice.agent, bob.agent) == ("alice", "bob"))

	alice.remember("the vault password rotates on fridays")
	bob.remember("i am debugging the payment service")

	check("alice sees only her own transcript",
	      [r["content"] for r in alice.timeline(tier="working")] == ["the vault password rotates on fridays"],
	      alice.timeline(tier="working"))
	check("bob sees only his",
	      [r["content"] for r in bob.timeline(tier="working")] == ["i am debugging the payment service"])
	check("writes are stamped with an owner",
	      sorted({r.owner for r in stack.records_in("working")}) == ["alice", "bob"])
	check("an unscoped client sees everything", len(shared.timeline(tier="working")) == 2)

	# Spilling carries a private record into a shared tier: what one agent
	# experienced becomes what the team knows, and the owner survives as provenance.
	stack.promote([r.id for r in stack.records_in("working") if r.owner == "alice"], "episodic")
	recalled = bob.recall(query="vault password rotates", tier="episodic")
	check("a spilled record becomes visible to the team", recalled, recalled)
	check("provenance survives the move", recalled and recalled[0]["owner"] == "alice", recalled)

	check("counts are scoped too", alice.describe(refresh=True)["total"] < shared.describe(refresh=True)["total"])

	private = alice.remember("alice's private scratch note")
	check("an agent cannot forget another's private records", bob.forget(ids=private) == 0)
	check("but its owner can"                               , alice.forget(ids=private) == 1)


def test_sessions() -> None:
	print("== sessions ==")

	stack = MemoryStack(tiers=[
		MemoryTier("working" , capacity = 4, spill_to = "episodic"),
		MemoryTier("episodic", capacity = 0, spill_to = "semantic",
		                       promote_to = "semantic", promote_after = 2),
		MemoryTier("semantic", capacity = 0),
		MemoryTier("procedural", capacity = 0),
	])

	check("the transcript tier is per-session", stack.tiers["working"].per_session is True)
	check("the durable tiers are not"         , not stack.tiers["episodic"].per_session)

	for i in range(4):
		stack.remember(f"chat-1 turn {i}", session="chat-1")
	for i in range(4):
		stack.remember(f"chat-2 turn {i}", session="chat-2")
	stack.consolidate()

	working = [r.content for r in stack.records_in("working")]
	check("capacity is enforced inside each conversation", len(working) == 8, working)
	check("neither conversation evicted the other",
	      sum("chat-1" in c for c in working) == 4 and sum("chat-2" in c for c in working) == 4, working)

	# The gap sessions exist to close: replaying one conversation.
	replayed = [r.content for r in stack.timeline(where={"session": "chat-1"})]
	check("timeline can replay one conversation", len(replayed) == 4, replayed)
	check("and excludes the other", all("chat-2" not in c for c in replayed), replayed)

	listed = {s["session"]: s for s in stack.sessions()}
	check("sessions are listed"          , sorted(listed) == ["chat-1", "chat-2"], list(listed))
	check("a session counts its records" , listed["chat-1"]["records"] == 4, listed["chat-1"])

	# Closing percolates: flushed out of working, then every tier applies its rule.
	report = stack.close_session("chat-1")
	check("closing reports what it closed", report["closed"] == "chat-1", report)
	check("the transcript left working memory", report["flushed"] == 4, report)
	check("nothing was destroyed", len(stack.timeline(where={"session": "chat-1"})) == 4)
	check("it landed in episodic",
	      all(r.tier == "episodic" for r in stack.timeline(where={"session": "chat-1"})),
	      [(r.content, r.tier) for r in stack.timeline(where={"session": "chat-1"})])
	check("the other conversation is untouched",
	      all("chat-2" in r.content for r in stack.records_in("working")),
	      [r.content for r in stack.records_in("working")])
	check("procedural stayed empty", stack.records_in("procedural") == [])

	check("closing an unknown session is a no-op", stack.close_session("nope")["flushed"] == 0)

	# Over the wire.
	memory = connect_local(MemoryStack())
	check("sessions is declared", "sessions" in memory.capabilities(), memory.capabilities())
	memory.remember("we agreed to ship on friday", session="chat-a")
	memory.remember("unrelated chatter")
	check("records carry their session",
	      memory.timeline(where={"session": "chat-a"})[0]["session"] == "chat-a")
	check("session/list over the wire", [s["session"] for s in memory.sessions()] == ["chat-a"])
	check("session/close over the wire", memory.close_session("chat-a")["closed"] == "chat-a")

	# A core-only server must refuse, with the capability error not the method one.
	limited = connect_local(MemoryStack(), capabilities=["core"])
	for method, params in [("memory/session/list", {}), ("memory/session/close", {"session": "x"})]:
		try:
			limited.client.call(method, params)
			check(f"{method} without 'sessions' is refused", False)
		except JsonRpcError as exc:
			check(f"{method} without 'sessions' is refused", exc.code == -32003, exc.code)


def test_keys() -> None:
	print("== keys ==")

	stack = MemoryStack()
	first = stack.remember("the user lives in bologna", key="user/city")
	again = stack.remember("the user lives in milan",  key="user/city")

	check("a key addresses one record"      , first.id == again.id)
	check("writing replaces, not appends"   , stack.count() == 1, stack.count())
	check("content is the new one"          , stack.by_key("user/city").content.endswith("milan"))
	check("revision advances"               , again.revision == 1, again.revision)
	check("the stale fact is gone"          , stack.recall("bologna") == [], stack.recall("bologna"))

	# Hierarchy without a second addressing dimension.
	for k in ["myapp/wf-42/user/city", "myapp/wf-42/user/goal", "myapp/wf-7/user/city", "other/thing"]:
		stack.remember("x for " + k, key=k)

	check("a prefix scopes a read",
	      sorted(r.key for r in stack.timeline(key_prefix="myapp/wf-42/"))
	      == ["myapp/wf-42/user/city", "myapp/wf-42/user/goal"])
	check("a shorter prefix scopes wider" , len(stack.timeline(key_prefix="myapp/")) == 3)
	check("a prefix can be forgotten"     , stack.forget(key_prefix="myapp/wf-7/") == 1)
	check("unused keys read as absent"    , stack.by_key("nothing/here") is None)

	# Keys are per owner, so two agents do not collide.
	shared = MemoryStack()
	shared.remember("bologna", key="user/city", owner="alice")
	shared.remember("milan",   key="user/city", owner="bob")
	check("keys are scoped per owner",
	      (shared.by_key("user/city", "alice").content,
	       shared.by_key("user/city", "bob").content) == ("bologna", "milan"))

	# Over the wire.
	memory = connect_local(MemoryStack())
	check("keys is declared", "keys" in memory.capabilities(), memory.capabilities())
	memory.remember("first value", key="a/b")
	memory.remember("second value", key="a/b")
	fetched = memory.fetch("a/b")
	check("fetch over the wire", fetched and fetched["content"] == "second value", fetched)
	check("revision on the wire", fetched["revision"] == 1, fetched)
	check("fetch of an unused key", memory.fetch("no/such/key") is None)

	limited = connect_local(MemoryStack(), capabilities=["core"])
	for label, call in [("remember with a key", lambda: limited.remember("x", key="a/b")),
	                    ("fetch"              , lambda: limited.fetch("a/b")),
	                    ("key_prefix"         , lambda: limited.timeline(key_prefix="a/"))]:
		try:
			call()
			check(f"{label} without 'keys' is refused", False)
		except JsonRpcError as exc:
			check(f"{label} without 'keys' is refused", exc.code == -32003, exc.code)


def test_caller_embeddings() -> None:
	print("== caller-owned embeddings ==")

	# A store with no embedder at all still answers vector searches.
	stack = MemoryStack(scorer=EmbeddingScorer(None))
	near  = stack.remember("the user is called marco", key="user/name", embedding=[1.0, 0.0, 0.0])
	stack.remember("the office is in milan", key="org/office", embedding=[0.0, 1.0, 0.0])

	check("no model is configured"     , stack.scorer.embed is None)
	check("the vector is stored verbatim", stack.by_key("user/name").embedding == [1.0, 0.0, 0.0])

	hits = stack.recall(embedding=[0.95, 0.05, 0.0], limit=1)
	check("a query vector searches"    , hits and hits[0][0].id == near.id, hits)
	check("without any text query"     , hits and hits[0][0].content.endswith("marco"))

	# A supplied vector is never regenerated, even when an embedder exists.
	calls = []
	def counting(texts):
		calls.append(list(texts))
		return [[0.0, 0.0, 1.0] for _ in texts]

	both = MemoryStack(scorer=EmbeddingScorer(counting))
	both.remember("carries its own vector", embedding=[1.0, 0.0, 0.0])
	both.recall(embedding=[1.0, 0.0, 0.0])
	check("a caller's vector is never re-embedded",
	      all("carries its own vector" not in batch for batch in calls), calls)

	# Over the wire, including the contract and the mismatch rule.
	memory = connect_local(MemoryStack(scorer=EmbeddingScorer(None)))
	check("embeddings is declared", "embeddings" in memory.capabilities())
	check("a contract is reported" , memory.describe()["embeddings"]["metric"] == "cosine")

	memory.remember("vector probe", embedding=[1.0, 0.0, 0.0])
	back = memory.recall(embedding=[1.0, 0.0, 0.0], limit=1, embeddings=True)
	check("the vector round-trips", back and back[0]["embedding"] == [1.0, 0.0, 0.0], back)
	check("dimensionality is reported", memory.describe(refresh=True)["embeddings"]["dimensions"] == 3)
	check("vectors are omitted by default",
	      "embedding" not in memory.recall(embedding=[1.0, 0.0, 0.0], limit=1)[0])

	try:
		memory.remember("wrong width", embedding=[1.0, 2.0])
		check("a mismatched width is refused", False)
	except JsonRpcError as exc:
		check("a mismatched width is refused", exc.code == -32008, exc.code)

	limited = connect_local(MemoryStack(), capabilities=["core"])
	try:
		limited.remember("x", embedding=[1.0])
		check("embeddings without the capability are refused", False)
	except JsonRpcError as exc:
		check("embeddings without the capability are refused", exc.code == -32003, exc.code)


def test_external() -> None:
	print("== external records ==")

	stack = MemoryStack()
	record = stack.remember("Deployment runbook: rotating the vault key",
	                        tier="procedural", key="docs/runbook",
	                        uri="file:///docs/runbook.md", media_type="text/markdown")

	check("a record can carry a reference", record.uri == "file:///docs/runbook.md")
	check("and a media type"              , record.media_type == "text/markdown")
	check("content is still what is indexed",
	      [r.content for r, _ in stack.recall("how do I rotate the vault key")] == [record.content])

	# external is a property of content, not a tier kind: the same reference is
	# legal in any tier, because where it belongs is the caller's judgement.
	stack.remember("the onboarding guide", tier="semantic",
	               uri="https://example.com/onboarding", media_type="text/html")
	check("a reference is legal in any tier",
	      sorted(r.tier for r in stack.records.values() if r.uri) == ["procedural", "semantic"])

	check("uri is filterable like any field",
	      len(stack.timeline(where={"media_type": "text/html"})) == 1)

	# A reference with no text is legal, and honestly unfindable by search.
	bare = stack.remember("", key="docs/blob", uri="s3://bucket/blob.bin")
	check("a reference with no text is accepted", bare.uri.startswith("s3://"))
	check("but is not findable by content"      ,
	      all(r.id != bare.id for r, _ in stack.recall("blob")))
	check("it is still reachable by key"        , stack.by_key("docs/blob").id == bare.id)

	# Over the wire.
	memory = connect_local(MemoryStack())
	check("external is declared", "external" in memory.capabilities(), memory.capabilities())
	memory.remember("a referenced document", uri="https://example.com/doc", media_type="text/html")
	got = memory.recall(query="referenced document")
	check("the uri round-trips", got and got[0]["uri"] == "https://example.com/doc", got)
	check("the media type too" , got and got[0]["media_type"] == "text/html")

	limited = connect_local(MemoryStack(), capabilities=["core"])
	try:
		limited.remember("x", uri="file:///x")
		check("a uri without the capability is refused", False)
	except JsonRpcError as exc:
		check("a uri without the capability is refused", exc.code == -32003, exc.code)


def test_summarize() -> None:
	print("== summarize ==")

	from a2m           import MemoryClient
	from a2m.jsonrpc   import Client, LocalTransport
	from a2m.retrieval import extractive_summarizer

	summarize = extractive_summarizer(keep=1)
	records   = [
		MemoryStack().remember("the deploy key rotates every ninety days. it is fine."),
		MemoryStack().remember("it is fine. it is fine."),
	]
	picked = summarize(records, "semantic")
	check("the extractive summarizer keeps the informative sentence",
	      picked == ["the deploy key rotates every ninety days."], picked)
	check("and declines when there is nothing to keep",
	      extractive_summarizer()([MemoryStack().remember("a b")], "semantic") is None)

	# Spec §2: a server must not advertise a capability it cannot implement.
	plain = connect_local(MemoryStack())
	check("a stack with no summarizer does not declare summarize",
	      "summarize" not in plain.capabilities(), plain.capabilities())

	server = MemoryServer(stack=MemoryStack(), summarize_fn=summarize, summarizer_model="extractive")
	memory = MemoryClient(Client(LocalTransport(server.dispatcher)))
	check("a stack with one does declare it", "summarize" in memory.capabilities(), memory.capabilities())
	check("and reports what is behind it",
	      memory.describe()["summarize"] == {"model": "extractive"}, memory.describe().get("summarize"))

	ids = memory.remember(
		"the deploy key rotates every ninety days. rotation is announced early.",
		tier = "semantic",
	)
	result = memory.summarize(ids=ids, into="semantic")
	check("summarize reports what it read and wrote",
	      result["read"] == 1 and result["written"] == 1, result)
	check("the summary is an ordinary record",
	      result["records"] and result["records"][0]["tier"] == "semantic", result)
	check("it records what it came from",
	      result["records"][0]["metadata"]["summarized_from"] == ids, result["records"][0])

	# Spec §4.15: consolidation may delete what it rewrites; this must not.
	check("the sources survive", memory.recall(query="deploy key rotates") != [], "sources were deleted")

	# A key turns an accumulating summary into a maintained one.
	memory.summarize(ids=ids, into="semantic", key="wiki/deploys")
	memory.summarize(ids=ids, into="semantic", key="wiki/deploys")
	held = memory.timeline(key_prefix="wiki/deploys")
	check("summarizing to a key replaces rather than accumulates", len(held) == 1, held)
	check("and advances the revision", held and held[0]["revision"] >= 1, held)

	empty = memory.summarize(query="nothing here matches at all zzzz")
	check("a selector matching nothing is not an error",
	      empty["written"] == 0 and empty["records"] == [], empty)

	try:
		memory.summarize()
		check("summarize with no selector is refused", False)
	except JsonRpcError as exc:
		check("summarize with no selector is refused", exc.code == -32602, exc.code)

	memory.close()


def test_prompt() -> None:
	print("== prompt ==")

	from a2m           import MemoryClient
	from a2m.jsonrpc   import Client, LocalTransport
	from a2m.prompt    import render

	# The renderer on its own: form follows the tier kind.
	block, used = render([{"id": "a", "content": "the deploy key rotates every ninety days"}])
	check("facts render as a list", block.startswith("Relevant memory:\n- "), block)
	check("and report which ids they contain", used == ["a"], used)

	block, _ = render([{"id": "a", "role": "user", "content": "where is the runbook?"}])
	check("a conversation renders as a transcript", "user: where is the runbook?" in block, block)

	block, used = render([{"id": "a", "content": "x" * 60}, {"id": "b", "content": "y" * 60}], budget=80)
	check("a budget drops whole records", used == ["a"], used)
	check("and never truncates one", "x" * 60 in block and "y" * 60 not in block, block)

	block, _ = render([{"id": "a", "content": "a fact", "uri": "file:///r.md"}], cite=True)
	check("citing marks the source", "(file:///r.md)" in block, block)

	# Over the protocol.
	server = MemoryServer(stack=MemoryStack())
	memory = MemoryClient(Client(LocalTransport(server.dispatcher)))
	contract = memory.describe()["prompt"]

	check("describe reports styles and methods",
	      "auto" in contract["styles"] and "template" in contract["methods"], contract)
	check("a server with no renderer does not offer 'model'",
	      "model" not in contract["methods"], contract)

	ids = memory.remember("the deploy key rotates every ninety days", tier="semantic")

	plain    = memory.client.call("memory/recall", {"query": "deploy key"})
	rendered = memory.client.call("memory/recall", {"query": "deploy key", "prompt": True})
	check("no prompt unless asked", "prompt" not in plain, list(plain))
	check("records are unchanged by asking",
	      [r["id"] for r in plain["records"]] == [r["id"] for r in rendered["records"]], rendered)
	check("the prompt carries the ids it used", rendered["prompt_ids"] == ids, rendered)

	quiet = memory.client.call("memory/recall", {"query": "deploy key", "prompt": {"method": "none"}})
	check("method 'none' renders nothing", "prompt" not in quiet, list(quiet))

	# Spec §4.16: refusing beats substituting, and the default never costs money.
	try:
		memory.client.call("memory/recall", {"query": "x", "prompt": {"method": "model"}})
		check("an unavailable method is refused", False)
	except JsonRpcError as exc:
		check("an unavailable method is refused", exc.code == -32602, exc.code)

	# With a renderer configured, 'model' appears and is used.
	written = MemoryServer(
		stack        = MemoryStack(),
		prompt_fn    = lambda records, style, budget, cite: "In short: " + records[0]["content"],
		prompt_model = "demo-llm",
	)
	prose = MemoryClient(Client(LocalTransport(written.dispatcher)))
	check("a configured renderer is advertised",
	      "model" in prose.describe()["prompt"]["methods"], prose.describe()["prompt"])
	check("and names its model", prose.describe()["prompt"]["model"] == "demo-llm")

	prose.remember("the deploy key rotates every ninety days", tier="semantic")
	prose.recall(query="deploy key", prompt={"method": "model"})
	check("method 'model' uses the renderer", prose.last_prompt.startswith("In short:"), prose.last_prompt)

	prose.recall(query="deploy key", prompt=True)
	check("but the default is still the deterministic one",
	      prose.last_prompt.startswith("Relevant memory:"), prose.last_prompt)

	try:
		prose.client.call("memory/recall", {"query": "x", "prompt": {"method": "model", "model": "other"}})
		check("asking for a different model is refused", False)
	except JsonRpcError as exc:
		check("asking for a different model is refused", exc.code == -32602, exc.code)

	memory.close()
	prose.close()


def test_events() -> None:
	print("== events ==")

	from a2m import EventLog

	log   = EventLog(retain=4)
	start = log.head()
	for n in range(3):
		log.append("written", id=f"r{n}")

	reply = log.read(cursor=start)
	check("a cursor replays what happened", [e["id"] for e in reply["events"]] == ["r0", "r1", "r2"], reply)
	check("the cursor advances past what was read", log.read(cursor=reply["cursor"])["events"] == [])

	first  = log.read(cursor=start, limit=2)
	second = log.read(cursor=first["cursor"])
	check("a limit never drops or repeats events",
	      [e["id"] for e in first["events"] + second["events"]] == ["r0", "r1", "r2"],
	      (first, second))
	check("a limit reports there is more", first.get("more") is True, first)

	for n in range(3, 9):
		log.append("written", id=f"r{n}")
	lost = log.read(cursor=start)
	check("a cursor behind the retention window resets", lost.get("reset") is True, lost)
	check("and continues from the oldest retained",
	      [e["id"] for e in lost["events"]] == ["r5", "r6", "r7", "r8"], lost)

	check("a foreign cursor resets rather than misreads",
	      log.read(cursor="deadbeef:2").get("reset") is True)

	scoped = EventLog()
	head   = scoped.head()
	scoped.append("written", owner="alice", id="a")
	scoped.append("written", id="shared")
	check("a scoped read sees its own and unowned events",
	      [e["id"] for e in scoped.read(cursor=head, agent="alice")["events"]] == ["a", "shared"])
	check("a scoped read never sees another owner's events",
	      [e["id"] for e in scoped.read(cursor=head, agent="bob")["events"]] == ["shared"])
	check("filtered kinds are never re-offered",
	      scoped.read(cursor=scoped.read(cursor=head, kinds=["forgotten"])["cursor"])["events"] == [])

	# The server layer: push and poll are two views of one log.
	memory   = connect_local(MemoryStack())
	position = memory.events()["cursor"]
	memory.subscribe_events()
	ids    = memory.remember("the deploy key rotates every ninety days")
	pushed = memory.take_events()
	check("a subscribed local client is pushed the write", [e.get("id") for e in pushed] == ids, pushed)
	polled = memory.events(cursor=position)["events"]
	check("the same event is still readable by poll", [e.get("id") for e in polled] == ids, polled)
	memory.unsubscribe_events()
	memory.remember("nothing should arrive for this")
	check("an unsubscribed client is pushed nothing", memory.take_events() == [])
	memory.close()


def test_a2m_local() -> None:
	print("== a2m, in-process ==")

	memory = connect_local()
	check("describe", memory.describe()["protocol"] == "a2m/0.1", memory.describe())
	check("tiers"   , memory.tiers() == ["working", "episodic", "semantic", "procedural"], memory.tiers())
	check("working tier is reported by kind", memory.working() == "working")
	check("kinds are reported", memory.of_kind("procedural") == ["procedural"], memory.describe().get("kinds"))

	ids = memory.remember(records=[{"content": "the api key is in vault/prod"}, {"content": "ci runs on windows"}])
	check("batch remember", len(ids) == 2)

	recalled = memory.recall(query="where is the api key")
	check("recall over the wire", recalled and "vault" in recalled[0]["content"], recalled)
	check("records carry a score", "score" in recalled[0])
	check("reinforce", memory.reinforce([ids[0]]) == 1)
	check("timeline" , len(memory.timeline(tier="working")) == 2)

	try:
		memory.forget()
		check("forget-everything is refused", False)
	except JsonRpcError as exc:
		check("forget-everything is refused", exc.code == -32602)

	check("forget by id", memory.forget(ids=[ids[0]]) == 1)
	check("consolidate reports counts", "counts" in memory.consolidate())

	written = memory.remember("always run the tests before pushing", tier="procedural")
	check("writing straight to procedural", memory.timeline(tier="procedural")[0]["id"] == written[0])
	check("promote over the wire", memory.promote(written, "semantic") == 1)
	check("promotion moved it", memory.timeline(tier="procedural") == [])

	try:
		memory.promote(written, "nowhere")
		check("promoting into an unknown tier is refused", False)
	except JsonRpcError as exc:
		check("promoting into an unknown tier is refused", exc.code == -32002, exc.code)


def test_a2m_spec() -> None:
	print("== a2m/0.1 conformance ==")

	memory  = connect_local()
	profile = memory.describe()

	check("protocol is declared"    , profile["protocol"] == "a2m/0.1", profile.get("protocol"))
	check("capabilities include core", "core" in profile["capabilities"], profile.get("capabilities"))
	check("methods are listed"      , "memory/recall" in profile["methods"])
	check("limits are advertised"   , "max_records_per_call" in profile.get("limits", {}))

	# spec §3.3 -- RFC 3339 strings, never epoch numbers.
	ids     = memory.remember("a record with a timestamp")
	record  = memory.timeline(tier=memory.working())[-1]
	stamp   = record["created_at"]
	check("created_at is a string"   , isinstance(stamp, str), stamp)
	check("created_at is RFC 3339"   , stamp.endswith("Z") and "T" in stamp, stamp)
	check("timestamps round-trip"    , abs(from_rfc3339(to_rfc3339(1_800_000_000.5)) - 1_800_000_000.5) < 0.01)

	# spec §3.2 -- a client-supplied id makes the write idempotent.
	chosen = "fixed-id-001"
	first  = memory.remember("idempotent record", id=chosen)
	second = memory.remember("idempotent record", id=chosen)
	check("a client-supplied id is honoured"  , first == [chosen], first)
	check("re-writing the same id is a no-op" , second == [chosen], second)
	check("only one record was created",
	      sum(1 for r in memory.timeline() if r["id"] == chosen) == 1)

	# spec §2 -- unknown params are ignored, never rejected.
	survived = memory.client.call("memory/describe", {"protocol": "a2m/0.1", "future_field": 42})
	check("unknown params are ignored", survived["protocol"] == "a2m/0.1")

	# spec §4.1 -- version negotiation.
	try:
		memory.client.call("memory/describe", {"protocol": "a2m/99.0"})
		check("an incompatible protocol is refused", False)
	except JsonRpcError as exc:
		check("an incompatible protocol is refused", exc.code == -32007, exc.code)

	# spec §2 -- an undeclared capability answers -32003, never -32601.
	limited = connect_local(MemoryStack(), capabilities=["core"])
	check("a core-only server declares only core", limited.capabilities() == ["core"], limited.capabilities())

	for method, params in [("memory/consolidate", {}), ("memory/reinforce", {"ids": []})]:
		try:
			limited.client.call(method, params)
			check(f"{method} without its capability is refused", False)
		except JsonRpcError as exc:
			check(f"{method} without its capability is refused", exc.code == -32003, exc.code)

	try:
		limited.client.call("memory/remember", {"records": [{"content": "x", "tier": "working"}]})
		check("writing to a tier without 'tiers' is refused", False)
	except JsonRpcError as exc:
		check("writing to a tier without 'tiers' is refused", exc.code == -32003, exc.code)

	try:
		MemoryServer(capabilities=["tiers"])
		check("a server without core is rejected", False)
	except ValueError:
		check("a server without core is rejected", True)


def test_a2m_http() -> None:
	print("== a2m over http ==")

	server = serve_a2m_http(MemoryServer(name="http-memory"), port=8791)
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()

	try:
		memory = connect_http("http://127.0.0.1:8791/")
		check("describe over http", memory.describe()["name"] == "http-memory", memory.describe().get("name"))

		memory.remember("the fallback region is eu-central-1")
		recalled = memory.recall(query="which fallback region")
		check("recall over http", recalled and "eu-central-1" in recalled[0]["content"], recalled)

		# spec §8.3 -- protocol errors are 200 with a JSON-RPC error, not an HTTP status.
		try:
			memory.client.call("memory/forget", {})
			check("a protocol error is not an http error", False)
		except JsonRpcError as exc:
			check("a protocol error is not an http error", exc.code == -32602, exc.code)

		# A notification gets 202 and no body.
		memory.client.notify("memory/consolidate", {})
		check("notifications are accepted", True)

		# spec §8.3.1 -- an origin the server does not permit never reaches a handler.
		blocked = urllib.request.Request(
			"http://127.0.0.1:8791/",
			data    = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "memory/describe"}).encode("utf-8"),
			headers = {"Content-Type": "application/json", "Origin": "https://evil.example"},
			method  = "POST",
		)
		try:
			urllib.request.urlopen(blocked, timeout=5)
			check("a foreign origin is refused", False)
		except urllib.error.HTTPError as exc:
			check("a foreign origin is refused", exc.code == 403, exc.code)

		# spec §8.3.3 -- the profile is served without an A2M client.
		with urllib.request.urlopen("http://127.0.0.1:8791/.well-known/a2m-server.json", timeout=5) as response:
			profile = json.loads(response.read().decode("utf-8"))
		check("the well-known profile is served", profile.get("protocol") == "a2m/0.1", profile.get("protocol"))
	finally:
		server.shutdown()
		server.server_close()


def test_a2m_stdio() -> None:
	print("== a2m, over stdio ==")

	memory = connect_stdio([sys.executable, "-m", "a2m", "remote-memory"], on_stderr=lambda line: print("    [server]", line))
	try:
		check("remote describe", memory.describe()["name"] == "remote-memory", memory.describe())
		memory.remember("the fallback region is eu-central-1")
		recalled = memory.recall(query="which fallback region")
		check("remote recall"  , recalled and "eu-central-1" in recalled[0]["content"], recalled)
		check("remote timeline", len(memory.timeline()) == 1)
	finally:
		memory.close()


def test_a2m_client() -> None:
	"""The independent client, against two servers that share no code with it.

	client.py imports nothing from this project, so this is the only place
	that proves it still works -- the conformance suite tests servers, and nothing
	tests a client from outside.
	"""
	print("== implementations/client.py, independent ==")

	from implementations import client

	# Against the reference server: everything declared.
	memory = client.connect_stdio([sys.executable, "-m", "a2m", "client-target"])
	try:
		check("independent describe", memory.describe()["name"] == "client-target", memory.describe().get("name"))

		memory.remember("the fallback region is eu-central-1", tier="episodic")
		recalled = memory.recall("which fallback region")
		check("independent recall", recalled and "eu-central-1" in recalled[0]["content"], recalled)

		# spec §4.2 -- the same id twice is one record, which is what makes a retry safe.
		id = "fixed-id-for-idempotency"
		memory.remember("written once", id=id, tier="episodic")
		memory.remember("written once", id=id, tier="episodic")
		written = [r for r in memory.timeline(tier="episodic") if r.get("id") == id]
		check("a repeated id writes once", len(written) == 1, written)
	finally:
		memory.close()

	# Against the core-only server: the guard must refuse before the wire.
	minimal = client.connect_stdio([sys.executable, "implementations/server_minimal.py"])
	try:
		check("core-only server declares core only", minimal.capabilities == ["core"], minimal.capabilities)

		try:
			minimal.fetch("user/city")
			check("an undeclared capability is refused locally", False)
		except client.A2MError as exc:
			check("an undeclared capability is refused locally", exc.code == -32003, exc.code)
	finally:
		minimal.close()


def main() -> int:
	test_jsonrpc()
	test_recall()
	test_consolidation()
	test_kinds()
	test_promotion()
	test_scorers()
	test_language()
	test_llm_consolidation()
	test_concurrency()
	test_sharing()
	test_sessions()
	test_keys()
	test_caller_embeddings()
	test_external()
	test_summarize()
	test_prompt()
	test_events()
	test_a2m_local()
	test_a2m_spec()
	test_a2m_http()
	test_a2m_stdio()
	test_a2m_client()

	print()
	print(f"{len(PASSED)} passed, {len(FAILED)} failed")
	for label in FAILED:
		print(f"  - {label}")

	return 1 if FAILED else 0


if __name__ == "__main__":
	sys.exit(main())
