"""Tokenization for lexical recall, across languages.

Three things go wrong the moment the conversation stops being English, and all
three are handled here.

**Scripts.** `[a-z0-9]+` is ASCII, so `Köln` indexed as `['ln']`, `città` as
`['citt']` — which then failed to match `citta` — and Cyrillic, Greek, Chinese
and Japanese produced nothing at all. `WORD_RE` is Unicode instead.

**Word boundaries.** Chinese and Japanese do not put spaces between words, so
Unicode word matching swallows a whole clause as one useless token. CJK runs are
cut into character bigrams, which is the cheap standard answer for retrieval and
needs no dictionary.

**Stopwords.** One global English list is not merely incomplete elsewhere, it is
actively wrong: it deleted the Italian verbs `so` (*I know*), `do` (*I give*) and
the noun `can` (*dog*), while letting all thirteen of `il lo la di che per con
non sono una del ho ha` through as if they carried meaning. So the list is keyed
by language and chosen per text, rather than assumed.
"""


import re


# Unicode word characters, minus the underscore. Python's \w is Unicode-aware
# for str, which is the whole point.
WORD_RE = re.compile(r"[^\W_]+")

# Scripts written without spaces between words.
CJK_RE  = re.compile(
	"["
	"぀-ゟ"   # hiragana
	"゠-ヿ"   # katakana
	"㐀-䶿"   # CJK unified ideographs extension A
	"一-鿿"   # CJK unified ideographs
	"豈-﫿"   # CJK compatibility ideographs
	"가-힯"   # hangul syllables
	"]+"
)

DEFAULT_LANGUAGE = "en"

STOPWORDS = {
	"en": frozenset({
		"a", "about", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "did",
		"do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "how", "i", "if",
		"in", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "she", "so", "than",
		"that", "the", "their", "them", "then", "there", "they", "this", "to", "was", "we",
		"were", "what", "when", "which", "who", "why", "will", "with", "would", "you", "your",
	}),
	"it": frozenset({
		# The one-letter words matter here even though `min_length` would drop them
		# anyway: detection runs before that filter, and "e"/"a"/"i" are among the
		# strongest signals that a text is Italian rather than English.
		"a", "e", "i", "o", "ad", "od",
		"il", "lo", "la", "gli", "le", "un", "uno", "una", "di", "del", "dello", "della", "dei",
		"degli", "delle", "al", "allo", "alla", "ai", "agli", "alle", "da", "dal", "dalla",
		"in", "nel", "nella", "nei", "con", "su", "sul", "sulla", "per", "tra", "fra", "ed",
		"ma", "se", "che", "chi", "cui", "non", "come", "dove", "quando", "perche", "perché",
		"piu", "più", "anche", "sono", "sei", "siamo", "siete", "era", "essere", "ho", "hai",
		"ha", "abbiamo", "avete", "hanno", "mi", "ti", "si", "ci", "vi", "ne", "questo",
		"questa", "questi", "queste", "quello", "quella", "qual", "quale", "quali", "molto",
		"tutto", "tutti", "gia", "già", "ancora", "solo",
	}),
	"es": frozenset({
		"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "en", "con",
		"por", "para", "sin", "sobre", "que", "quien", "cual", "cuando", "donde", "como", "no",
		"si", "pero", "mas", "más", "muy", "ya", "es", "son", "era", "ser", "estar", "esta",
		"está", "he", "has", "ha", "hemos", "han", "me", "te", "se", "nos", "os", "le", "les",
		"lo", "mi", "tu", "su", "este", "esta", "esto", "ese", "esa", "aquel", "todo", "todos",
	}),
	"fr": frozenset({
		"le", "la", "les", "un", "une", "des", "du", "de", "au", "aux", "et", "ou", "mais",
		"donc", "car", "ni", "que", "qui", "quoi", "dont", "ou", "où", "quand", "comme",
		"comment", "ne", "pas", "plus", "tres", "très", "est", "sont", "etait", "était", "etre",
		"être", "avoir", "ai", "as", "avons", "avez", "ont", "je", "tu", "il", "elle", "nous",
		"vous", "ils", "elles", "me", "te", "se", "mon", "ton", "son", "ce", "cette", "ces",
		"pour", "dans", "sur", "avec", "sans", "par", "tout", "tous",
	}),
	"de": frozenset({
		"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer",
		"eines", "und", "oder", "aber", "denn", "sondern", "dass", "wenn", "weil", "als", "wie",
		"wo", "wann", "warum", "nicht", "kein", "keine", "sehr", "mehr", "schon", "noch", "ist",
		"sind", "war", "waren", "sein", "haben", "hat", "habe", "hatte", "ich", "du", "er",
		"sie", "es", "wir", "ihr", "mich", "dich", "sich", "mein", "dein", "in", "auf", "mit",
		"von", "zu", "aus", "bei", "nach", "uber", "über", "fur", "für", "alle", "alles",
	}),
	"pt": frozenset({
		"o", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "no", "na",
		"nos", "nas", "ao", "aos", "em", "com", "por", "para", "sem", "sobre", "que", "quem",
		"qual", "quando", "onde", "como", "nao", "não", "se", "mas", "mais", "muito", "ja",
		"já", "eh", "sao", "são", "era", "ser", "estar", "esta", "está", "tenho", "tem",
		"temos", "me", "te", "se", "lhe", "meu", "teu", "seu", "este", "esse", "aquele", "todo",
	}),
}


def is_cjk(token: str) -> bool:
	"""Whether a token is entirely in a space-less script.

	Used to waive 'min_length': in Chinese and Japanese a single character is a
	morpheme, so dropping one-character tokens would discard meaning.

	Args:
		token (str): A candidate token.

	Returns:
		bool: True for hiragana, katakana, CJK ideographs or hangul.

	Example:
		>>> is_cjk("猫"), is_cjk("cat")
		(True, False)
	"""
	return bool(CJK_RE.fullmatch(token))


def bigrams(run: str) -> list[str]:
	"""Character bigrams for a space-less script.

	Chinese and Japanese put no spaces between words, so Unicode word matching
	swallows a whole clause as one useless token. Bigrams are the cheap standard
	answer for retrieval and need no dictionary -- the query is segmented the same
	way, so the two meet in the middle.

	Args:
		run (str): A run of CJK characters.

	Returns:
		list[str]: Overlapping character pairs. A single character stays whole,
		so a one-character word is never lost.

	Example:
		>>> bigrams("马可")
		['马可']
		>>> bigrams("我叫马可")
		['我叫', '叫马', '马可']
	"""
	if len(run) < 2:
		return [run]
	return [run[i:i + 2] for i in range(len(run) - 1)]


def segment(token: str) -> list[str]:
	"""Split one Unicode word into indexable pieces.

	CJK runs become bigrams while anything alphabetic stays intact, so a mixed
	token yields both halves usable.

	Args:
		token (str): One match from WORD_RE.

	Returns:
		list[str]: The indexable pieces.

	Example:
		>>> segment("gpu技术")
		['gpu', '技术']
	"""
	if not CJK_RE.search(token):
		return [token]

	parts    = []
	position = 0

	for match in CJK_RE.finditer(token):
		if match.start() > position:
			parts.append(token[position:match.start()])
		parts.extend(bigrams(match.group()))
		position = match.end()

	if position < len(token):
		parts.append(token[position:])

	return parts


def words(text: str) -> list[str]:
	"""Every indexable piece of a text, before any filtering.

	Args:
		text (str): Any text.

	Returns:
		list[str]: Lowercased pieces, CJK already segmented. No stopword or
		length filtering has happened yet, which is why 'detect' uses this
		rather than 'tokenize' -- one-letter function words are among the
		strongest language signals.

	Example:
		>>> words("Köln und Bologna")
		['köln', 'und', 'bologna']
	"""
	pieces = []
	for match in WORD_RE.findall(str(text).lower()):
		pieces.extend(segment(match))
	return pieces


def detect(text: str | list[str], languages: dict[str, frozenset] = None) -> str | None:
	"""Guess the language by how much of a text a registered stopword list explains.

	Crude, dependency-free, and good enough for its only job: choosing which
	function words to ignore. Getting it wrong costs a slightly worse ranking, not
	a wrong answer.

	Args:
		text (str | list[str]): Text, or pre-computed words.
		languages (dict, optional): Registry to score against. Defaults to
			STOPWORDS.

	Returns:
		str | None: The best-matching language code, or None when nothing matched
		-- which is what happens for an unregistered language, and correctly
		leads to no filtering rather than the wrong filtering.

	Example:
		>>> detect("mi chiamo marco e abito a bologna")
		'it'
		>>> detect("my name is marco and i live in bologna")
		'en'
		>>> detect("меня зовут марко") is None
		True
	"""
	languages = languages or STOPWORDS
	pieces    = words(text) if isinstance(text, str) else list(text)

	if not pieces:
		return None

	best  = None
	score = 0.0

	for language, stopwords in languages.items():
		hits = sum(1 for piece in pieces if piece in stopwords) / len(pieces)
		if hits > score:
			best, score = language, hits

	return best


def tokenize(
	text       : str,
	stopwords  : frozenset[str] = None,
	min_length : int            = 2,
	language   : str            = None,
) -> list[str]:
	"""Indexable terms, filtered for the language the text is in.

	Args:
		text (str): Any text.
		stopwords (frozenset, optional): Wins if given -- **including an empty
			set**, which means "filter nothing". When None, the language is
			detected per text, so a stack holding several languages filters each
			record by its own function words.
		min_length (int, optional): Shortest token to keep. Does not apply to CJK,
			where one character is a morpheme.
		language (str, optional): Pin the language instead of detecting it, for a
			stack known to be monolingual.

	Returns:
		list[str]: The terms to index or query with.

	Example:
		>>> tokenize("the deploy key rotates every ninety days")
		['deploy', 'key', 'rotates', 'every', 'ninety', 'days']
		>>> tokenize("do it", language="en")
		[]
		>>> tokenize("do it", language="it")
		['do', 'it']
	"""
	pieces = words(text)

	if stopwords is None:
		stopwords = STOPWORDS.get(language or detect(pieces) or DEFAULT_LANGUAGE, frozenset())

	return [
		piece for piece in pieces
		if piece not in stopwords and (is_cjk(piece) or len(piece) >= min_length)
	]
