"""Turning recalled records into prompt text, without losing what they were.

Every agent doing retrieval eventually writes this function, and writing it
badly is the common failure: dumping raw JSON into a prompt burns tokens on
punctuation, and flattening a transcript into a bullet list destroys the thing
that made it a transcript.

So it lives here once, and it is deliberately **dumb**: templates and a budget,
no model. Rendering is formatting. Where the goal is genuinely to *compress* --
to say the same thing in fewer words -- that is `memory/summarize` (spec §4.15),
which is where a model belongs.

The one non-obvious rule is that **how a record renders follows the kind of tier
it came from**:

	working      a transcript, chronological, roles preserved
	everything   a fact list, best first

which is the `timeline`/`recall` distinction (spec §4.4) finally visible where a
model can see it. A conversation rendered as bullet points reads as a set of
disconnected assertions, and a set of facts rendered as dialogue invites a model
to treat them as things somebody merely *said*.

Usable from either side of the protocol, which is the point of putting it in the
library rather than in a server: a server declaring the `prompt` capability
renders with this, and a client that would rather render locally calls the same
function on the records it already has.
"""


from   typing import Any


# What a rendered block is introduced with. Short on purpose: every token here
# is a token not spent on the memory itself.
HEADINGS = {
	"facts"      : "Relevant memory:",
	"transcript" : "Earlier in this conversation:",
}


def looks_like_transcript(records: list[dict[str, Any]], kinds: dict[str, str] = None) -> bool:
	"""Whether these records should be replayed rather than listed.

	Decided by the tier they came from where that is knowable, and by their
	shape otherwise -- records carrying conversational roles and no scores were
	almost certainly read with `timeline`.

	Args:
		records (list[dict]): Records in wire form.
		kinds (dict, optional): Tier name to kind, from `memory/describe`. The
			reliable signal; the fallback below is a guess.

	Returns:
		bool: True when the records are a conversation.

	Example:
		>>> looks_like_transcript([{"role": "user", "content": "hi"}])
		True
		>>> looks_like_transcript([{"content": "a fact", "score": 0.7}])
		False
	"""
	if not records:
		return False

	if kinds:
		tiers = {record.get("tier") for record in records if record.get("tier")}
		if tiers and all(kinds.get(tier) == "working" for tier in tiers):
			return True
		if tiers:
			return False

	# No tier information: a conversation is records with speakers and no
	# ranking, because nothing ranked them -- they were replayed.
	roles = [record.get("role") for record in records]
	return any(role in ("user", "assistant", "tool", "system") for role in roles) \
	   and not any("score" in record for record in records)


def render(
	records : list[dict[str, Any]],
	budget  : int  = 0,
	cite    : bool = False,
	style   : str  = "auto",
	heading : str  = None,
	kinds   : dict[str, str] = None,
) -> tuple[str, list[str]]:
	"""Render records into a block suitable for a system prompt.

	Trimming drops **whole records**, lowest-ranked first, and never truncates
	one mid-sentence: half a fact is worse than no fact, because a model cannot
	tell it was cut.

	Args:
		records (list[dict]): Records in wire form, as `memory/recall` or
			`memory/timeline` returned them.
		budget (int, optional): Maximum characters. 0 means no limit. Characters
			rather than tokens because tokenisation is model-specific and this
			module refuses to depend on a tokenizer; divide by roughly four for
			a token estimate.
		cite (bool, optional): Append a source marker to each entry -- the
			record's `uri` when it has one, its `key` otherwise, and nothing
			when it has neither.
		style (str, optional): 'facts', 'transcript', or 'auto' to decide from
			the records themselves.
		heading (str, optional): Override the introductory line. An empty string
			removes it.
		kinds (dict, optional): Tier name to kind, from `memory/describe`, which
			is what makes 'auto' reliable rather than a guess.

	Returns:
		tuple[str, list[str]]: The rendered block, and the ids of the records
		that survived the budget. The ids are the useful half: they are what a
		caller passes to `memory/reinforce`, so what the model actually saw is
		what gets reinforced -- rather than everything that was recalled,
		including whatever was trimmed away unread.

	Example:
		>>> block, used = render([
		...     {"id": "a", "content": "the deploy key rotates every ninety days"},
		...     {"id": "b", "content": "the release branch is cut on thursdays"}])
		>>> print(block)
		Relevant memory:
		- the deploy key rotates every ninety days
		- the release branch is cut on thursdays
		>>> used
		['a', 'b']
	"""
	if not records:
		return "", []

	if style == "auto":
		style = "transcript" if looks_like_transcript(records, kinds) else "facts"

	title = HEADINGS.get(style, HEADINGS["facts"]) if heading is None else heading
	lines = []

	for record in records:
		content = str(record.get("content", "")).strip()
		if not content:
			continue

		if style == "transcript":
			line = f"{record.get('role', 'user')}: {content}"
		else:
			line = f"- {content}"

		if cite:
			source = record.get("uri") or record.get("key")
			if source:
				line += f"  ({source})"

		lines.append((record.get("id"), line))

	if not lines:
		return "", []

	# Trim to the budget by dropping from the end, which is lowest-ranked for a
	# recall and oldest for a transcript -- in both cases the least worth
	# keeping. A transcript keeps its tail instead: the most recent turns are
	# the ones the next reply depends on.
	if budget and budget > 0:
		kept  = []
		spent = len(title) + 1 if title else 0
		order = reversed(lines) if style == "transcript" else lines

		for id, line in order:
			cost = len(line) + 1
			if spent + cost > budget:
				break
			kept.append((id, line))
			spent += cost

		lines = list(reversed(kept)) if style == "transcript" else kept

	if not lines:
		return "", []

	body = "\n".join(line for _, line in lines)
	used = [id for id, _ in lines if id]

	return (f"{title}\n{body}" if title else body), used


def render_result(result: dict[str, Any], **options: Any) -> tuple[str, list[str]]:
	"""Render whatever a `recall` or `timeline` result carried.

	Convenience for the common shape, so a caller does not have to reach into
	the envelope before rendering.

	Args:
		result (dict): A result object carrying `records`.
		**options: Passed to `render`.

	Returns:
		tuple[str, list[str]]: The rendered block and the ids used.
	"""
	return render(result.get("records") or [], **options)
