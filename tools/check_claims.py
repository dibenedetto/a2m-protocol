"""Every normative sentence in the specification, and what would catch it lying.

	python -m tools.check_claims          # fail on anything undispositioned
	python -m tools.check_claims --list   # print the register

Four times while this repository was being written, the same failure appeared:
a **MUST in the document that nothing exercised**. `events` was deferred for
it, §2.1 had to be written because a read-only server crashed the suite, the
framework-interop claim on the first line of the README was tested by nothing,
and `metric` was a field a server could not get wrong because no one checked it.

Each was found by accident, by building the thing that happened to test it.
This makes it deliberate. Every sentence in `spec/a2m-0.1.md` carrying an RFC
2119 keyword must be **dispositioned** in `claims.json` as one of:

	suite          a conformance check would fail; names it
	tests          tools/test_a2m.py would fail; names it
	client         it binds clients, and the suite tests servers
	unverifiable   not observable from outside, with the reason
	prose          RFC 2119 boilerplate or a definition, not a requirement
	gap            checkable, and not yet checked

`gap` fails this tool. That is the point: a gap has to be closed or argued down
to `unverifiable`, and it cannot be left implicit.

The register is keyed by a hash of the sentence, so **rewording a requirement
invalidates its disposition** and forces someone to re-confirm the check still
covers it. That is deliberate friction: a MUST that changed meaning while its
test stayed the same is exactly what this exists to catch.
"""


import hashlib
import json
import pathlib
import re
import sys


ROOT     = pathlib.Path(__file__).resolve().parent.parent
SPEC     = ROOT / "spec" / "a2m-0.1.md"
REGISTER = ROOT / "tools" / "claims.json"

# The keywords that make a sentence a requirement. MAY and OPTIONAL grant
# permission rather than impose an obligation, so they are not tracked.
KEYWORDS = re.compile(r"\b(MUST NOT|MUST|SHALL NOT|SHALL|SHOULD NOT|SHOULD)\b")

# Dispositions that satisfy the tool, and what each asserts.
ACCEPTED = {
	"suite"        : "a conformance check would fail",
	"tests"        : "the offline suite would fail",
	"client"       : "binds clients; the conformance suite tests servers",
	"unverifiable" : "not observable from outside",
	"prose"        : "boilerplate or definition, not a requirement",
}


def statements() -> list[tuple[str, str, str]]:
	"""Every normative sentence in the specification.

	Fenced code is stripped first: an example showing a `MUST` in prose inside
	a JSON comment is not itself a requirement.

	Returns:
		list[tuple[str, str, str]]: (section, key, sentence). The key is a hash
		of the section and the sentence, so a reworded requirement gets a new
		one.
	"""
	body = re.sub(r"```.*?```", "", SPEC.read_text(encoding="utf-8"), flags=re.S)

	found   = []
	section = "0"
	buffer  = []

	def flush() -> None:
		text = re.sub(r"\s+", " ", re.sub(r"[*`]", "", " ".join(buffer))).strip()
		if not text:
			return
		for sentence in re.split(r"(?<=[.:])\s+(?=[A-Z(])", text):
			if KEYWORDS.search(sentence):
				sentence = sentence.strip()
				key      = hashlib.sha1(f"{section}|{sentence}".encode("utf-8")).hexdigest()[:8]
				found.append((section, key, sentence))

	for line in body.splitlines():
		heading = re.match(r"^#{2,4}\s+([\d.]+|Appendix \w+)?\s*(.*)$", line)
		if heading and heading.group(1):
			flush()
			buffer  = []
			section = heading.group(1).rstrip(".")
			continue
		if not line.strip():
			flush()
			buffer = []
			continue
		buffer.append(line.strip())

	flush()
	return found


def labels(path: pathlib.Path) -> str:
	"""The text of a suite, for confirming a named check still exists.

	Args:
		path (Path): The suite to read.

	Returns:
		str: Its source, or an empty string when it is missing.
	"""
	return path.read_text(encoding="utf-8") if path.exists() else ""


def main() -> int:
	"""Check that every normative sentence has a disposition, and that it holds.

	Returns:
		int: 0 when the register covers the specification exactly.
	"""
	register = json.loads(REGISTER.read_text(encoding="utf-8"))
	found    = statements()
	keys     = {key for _, key, _ in found}

	conformance = labels(ROOT / "tools" / "conformance.py")
	offline     = labels(ROOT / "tools" / "test_a2m.py")

	if "--list" in sys.argv[1:]:
		for section, key, sentence in found:
			how, note = (register.get(key) or ["MISSING", ""])[:2]
			print(f"  §{section:8} {key}  {how:12} {sentence[:96]}")
		return 0

	missing  = [(s, k, t) for s, k, t in found if k not in register]
	gaps     = [(s, k, t) for s, k, t in found if (register.get(k) or [""])[0] == "gap"]
	# Keys beginning with an underscore are notes to the reader, not claims.
	stale    = [k for k in register if k not in keys and not k.startswith("_")]
	unproven = []

	for section, key, sentence in found:
		how, note = (register.get(key) or ["", ""])[:2]
		if how == "suite" and note and note not in conformance:
			unproven.append((key, "conformance.py", note))
		if how == "tests" and note and note not in offline:
			unproven.append((key, "test_a2m.py", note))

	counts = {}
	for _, key, _ in found:
		how = (register.get(key) or ["MISSING"])[0]
		counts[how] = counts.get(how, 0) + 1

	print(f"\n  {len(found)} normative statements in {SPEC.relative_to(ROOT)}")
	for how, n in sorted(counts.items(), key=lambda pair: -pair[1]):
		print(f"    {how:14} {n:3}   {ACCEPTED.get(how, 'NOT DISPOSITIONED')}")

	if missing:
		print(f"\n  {len(missing)} statement(s) with no disposition. Add to {REGISTER.name}:\n")
		for section, key, sentence in missing:
			print(f'    "{key}": ["gap", ""],   // §{section}  {sentence[:88]}')

	if stale:
		print(f"\n  {len(stale)} register entr(ies) no longer match any statement — reworded or removed:")
		for key in stale:
			print(f"    {key}  {register[key]}")

	if unproven:
		print(f"\n  {len(unproven)} disposition(s) naming a check that no longer exists:")
		for key, where, note in unproven:
			print(f"    {key}  {where} has no {note!r}")

	if gaps:
		print(f"\n  {len(gaps)} statement(s) marked as a gap — checkable, and unchecked:")
		for section, key, sentence in gaps:
			print(f"    §{section}  {sentence[:100]}")

	if missing or stale or unproven or gaps:
		print("\n  A requirement nothing exercises is a requirement that quietly stops")
		print("  being true. Close it, or argue it down to 'unverifiable' with a reason.")
		return 1

	print("\n  Every normative statement is dispositioned, and every named check exists.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
