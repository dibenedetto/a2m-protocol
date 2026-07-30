"""Which embedding model should back recall, measured rather than assumed.

	python bench_embeddings.py                        # the default plus the baseline
	python bench_embeddings.py bge-m3 nomic-embed-text granite-embedding:278m

Absolute cosine is *not* comparable between models — each has its own similarity
scale, and a model that rates everything 0.9 looks impressive while
discriminating nothing. So the metric is precision@1: given ten facts, does the
query rank the right one first?

Ten topics form four deliberately confusable pairs, so a model that merely
clusters by subject is caught:

	key     / cert     both "every N days"
	release / standup  both a weekday
	pet     / dog      both a named animal
	city    / office   both a place

Each fact is rendered in five languages. The corpus is loaded in one language at
a time and queried from all five, so the diagonal of that matrix is same-language
retrieval and everything off it is cross-lingual.
"""


import sys
import time


from   typing        import Callable


from   a2m.memory    import MemoryStack
from   a2m.retrieval import EMBEDDING_MODEL, cosine, ollama_embedder


LANGUAGES = ["en", "it", "ru", "zh", "de"]

FACTS = {
	"name"    : {"en": "the user is called Marco",
	             "it": "l'utente si chiama Marco",
	             "ru": "пользователя зовут Марко",
	             "zh": "用户名叫马可",
	             "de": "der Benutzer heißt Marco"},
	"city"    : {"en": "the user lives in Bologna",
	             "it": "l'utente abita a Bologna",
	             "ru": "пользователь живёт в Болонье",
	             "zh": "用户住在博洛尼亚",
	             "de": "der Benutzer wohnt in Bologna"},
	"office"  : {"en": "the office is in Milan",
	             "it": "l'ufficio è a Milano",
	             "ru": "офис находится в Милане",
	             "zh": "办公室在米兰",
	             "de": "das Büro ist in Mailand"},
	"job"     : {"en": "the user works on compilers",
	             "it": "l'utente lavora sui compilatori",
	             "ru": "пользователь работает над компиляторами",
	             "zh": "用户从事编译器工作",
	             "de": "der Benutzer arbeitet an Compilern"},
	"key"     : {"en": "the deploy key rotates every ninety days",
	             "it": "la chiave di deploy ruota ogni novanta giorni",
	             "ru": "ключ развёртывания меняется каждые девяносто дней",
	             "zh": "部署密钥每九十天轮换一次",
	             "de": "der Deploy-Schlüssel wird alle neunzig Tage gewechselt"},
	"cert"    : {"en": "the TLS certificate expires every thirty days",
	             "it": "il certificato TLS scade ogni trenta giorni",
	             "ru": "сертификат TLS истекает каждые тридцать дней",
	             "zh": "TLS 证书每三十天过期一次",
	             "de": "das TLS-Zertifikat läuft alle dreißig Tage ab"},
	"release" : {"en": "the release branch is cut on Thursdays",
	             "it": "il ramo di rilascio viene creato il giovedì",
	             "ru": "релизная ветка создаётся по четвергам",
	             "zh": "发布分支在每周四创建",
	             "de": "der Release-Branch wird donnerstags erstellt"},
	"standup" : {"en": "the standup happens on Mondays",
	             "it": "lo standup si tiene il lunedì",
	             "ru": "стендап проходит по понедельникам",
	             "zh": "站会在每周一举行",
	             "de": "das Standup findet montags statt"},
	"pet"     : {"en": "the user has a cat named Ziggy",
	             "it": "l'utente ha un gatto di nome Ziggy",
	             "ru": "у пользователя есть кот по имени Зигги",
	             "zh": "用户有一只名叫齐吉的猫",
	             "de": "der Benutzer hat eine Katze namens Ziggy"},
	"dog"     : {"en": "the neighbour has a dog named Otto",
	             "it": "il vicino ha un cane di nome Otto",
	             "ru": "у соседа есть собака по имени Отто",
	             "zh": "邻居有一只名叫奥托的狗",
	             "de": "der Nachbar hat einen Hund namens Otto"},
}

QUERIES = {
	"name"    : {"en": "what is my name?", "it": "come mi chiamo?", "ru": "как меня зовут?",
	             "zh": "我叫什么名字？", "de": "wie heiße ich?"},
	"city"    : {"en": "where do I live?", "it": "dove abito?", "ru": "где я живу?",
	             "zh": "我住在哪里？", "de": "wo wohne ich?"},
	"office"  : {"en": "where is the office?", "it": "dov'è l'ufficio?", "ru": "где находится офис?",
	             "zh": "办公室在哪里？", "de": "wo ist das Büro?"},
	"job"     : {"en": "what do I work on?", "it": "di cosa mi occupo?", "ru": "над чем я работаю?",
	             "zh": "我从事什么工作？", "de": "woran arbeite ich?"},
	"key"     : {"en": "how often does the deploy key change?",
	             "it": "ogni quanto cambia la chiave di deploy?",
	             "ru": "как часто меняется ключ развёртывания?",
	             "zh": "部署密钥多久更换一次？",
	             "de": "wie oft wechselt der Deploy-Schlüssel?"},
	"cert"    : {"en": "how long until the certificate expires?",
	             "it": "ogni quanto scade il certificato?",
	             "ru": "как часто истекает сертификат?",
	             "zh": "证书多久过期一次？",
	             "de": "wie oft läuft das Zertifikat ab?"},
	"release" : {"en": "when is the release branch created?",
	             "it": "quando viene creato il ramo di rilascio?",
	             "ru": "когда создаётся релизная ветка?",
	             "zh": "发布分支什么时候创建？",
	             "de": "wann wird der Release-Branch erstellt?"},
	"standup" : {"en": "when is the standup?", "it": "quando è lo standup?", "ru": "когда стендап?",
	             "zh": "站会什么时候？", "de": "wann ist das Standup?"},
	"pet"     : {"en": "what is my cat called?", "it": "come si chiama il mio gatto?",
	             "ru": "как зовут моего кота?", "zh": "我的猫叫什么名字？", "de": "wie heißt meine Katze?"},
	"dog"     : {"en": "what is the neighbour's dog called?",
	             "it": "come si chiama il cane del vicino?",
	             "ru": "как зовут собаку соседа?", "zh": "邻居的狗叫什么名字？",
	             "de": "wie heißt der Hund des Nachbarn?"},
}

TOPICS = list(FACTS)


def rank_with_embeddings(embed: Callable, corpus: str, query: str) -> tuple[int, int]:
	documents = embed([FACTS[topic][corpus] for topic in TOPICS])
	asked     = embed([QUERIES[topic][query] for topic in TOPICS])

	hits = 0
	for i, vector in enumerate(asked):
		best = max(range(len(documents)), key=lambda j: cosine(vector, documents[j]))
		if best == i:
			hits += 1

	return hits, len(TOPICS)


def rank_with_lexical(corpus: str, query: str) -> tuple[int, int]:
	hits = 0
	for i, topic in enumerate(TOPICS):
		stack = MemoryStack()
		ids   = [stack.remember(FACTS[t][corpus]).id for t in TOPICS]
		found = stack.recall(QUERIES[topic][query], limit=1)
		if found and found[0][0].id == ids[i]:
			hits += 1

	return hits, len(TOPICS)


def measure(label: str, rank: Callable) -> tuple[float, float]:
	same = cross = same_total = cross_total = 0

	for corpus in LANGUAGES:
		for query in LANGUAGES:
			hits, total = rank(corpus, query)
			if corpus == query:
				same       += hits
				same_total += total
			else:
				cross       += hits
				cross_total += total

	print(f"  {label:<26} same {same:>3}/{same_total} = {same / same_total:>6.1%}   "
	      f"cross {cross:>3}/{cross_total} = {cross / cross_total:>6.1%}")

	return same / same_total, cross / cross_total


def main() -> int:
	models = sys.argv[1:] or [EMBEDDING_MODEL]

	print(f"{len(TOPICS)} topics in {len(LANGUAGES)} languages, "
	      f"{len(LANGUAGES) ** 2 * len(TOPICS)} queries per model\n")

	measure("lexical (baseline)", rank_with_lexical)

	for model in models:
		try:
			embed = ollama_embedder(model)
			embed(["warmup"])
		except Exception as exc:
			print(f"  {model:<26} unavailable: {str(exc)[:70]}")
			continue

		started = time.time()
		measure(model, lambda corpus, query, e=embed: rank_with_embeddings(e, corpus, query))
		print(f"  {'':<26} {time.time() - started:.1f}s")

	return 0


if __name__ == "__main__":
	sys.exit(main())
