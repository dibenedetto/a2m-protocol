"""The procedural tier: how to do things, written deliberately and never decayed into.

	python -m examples.procedural

Procedural memory holds **instructions**, not facts. "The deploy key rotates
every ninety days" is semantic; "to rotate the deploy key, do this, then this"
is procedural. In an agent framework, loading a skill writes here — which is
the honest answer to "does this mean installing skills?", with one important
qualification:

**A2M stores the text of a procedure. It never executes anything.** There is no
loader, no sandbox, no plugin mechanism, and there will not be: A2M is a memory
protocol, and a store that ran what it was handed would be a remote code
execution service that also does recall. The agent reads a procedure back at
task start and decides what to do with it. Whether that means following prose
instructions, or looking up a tool it already has, is entirely the agent's
business. `implementations/store_sqlite.py` writes each procedure as a Markdown
file precisely so a human can read, diff and revert it.

Two properties make this tier different from the other three, and both are
configuration rather than special-casing (see `default_tiers()`):

	nothing spills in     no tier declares spill_to="procedural"
	nothing expires out   capacity 0, half_life 0

So the only ways in are deliberate: `remember(tier="procedural")`, or
`promote(ids, "procedural")` for something that earned it. That is the whole
mechanism, and §4 below demonstrates that pressure elsewhere cannot reach it.
"""


import pathlib
import sys
import tempfile


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from   a2m        import connect_local
from   a2m.memory import MemoryStack


PASSED = []
FAILED = []


# A skill, as it would arrive from a skills directory: a name and a procedure.
# Markdown because a procedure is read by both an agent and a person, and the
# SQLite store writes it to disk under that name.
ROTATE_KEY = """\
# Rotating the deploy key

1. Announce on the release channel, one week ahead.
2. Generate the new key with `deploy-tool keygen --rotate`.
3. Update the secret in the vault before revoking the old key, never after.
4. Verify a deploy succeeds, then revoke the previous key.

Never revoke first: a failed rotation with the old key already gone leaves
nothing able to deploy.
"""

HANDLE_INCIDENT = """\
# Handling an incident

1. Declare the incident in the incident channel before investigating.
2. Roll back first, diagnose second. The rollback window is twenty-four hours.
3. Write the timeline as you go, not afterwards.
"""


def check(label: str, condition: bool, detail=None) -> None:
	"""Record one assertion.

	Args:
		label (str): What was being checked.
		condition (bool): Whether it held.
		detail (Any, optional): Shown on failure.
	"""
	if condition:
		PASSED.append(label)
		print(f"  ok    {label}")
	else:
		FAILED.append(label)
		print(f"  FAIL  {label}  {detail if detail is not None else ''}")


def install_skill(memory, name: str, procedure: str) -> list[str]:
	"""Write one procedure into procedural memory.

	This is what "installing a skill" means in A2M terms, and it is one
	ordinary `memory/remember` with an explicit tier. The `skill` metadata key
	is a convention the SQLite store reads to name the file it writes, so the
	procedure lands on disk as `<skill>-<id>.md` and can be version-controlled.

	Args:
		memory (MemoryClient): The store to write to.
		name (str): The skill's name.
		procedure (str): The instructions, as text.

	Returns:
		list[str]: The id written.
	"""
	return memory.remember(
		procedure,
		tier     = "procedural",
		role     = "system",
		metadata = {"skill": name},
	)


def main() -> int:
	"""Install skills, read them back, promote into the tier, and prove nothing spills in.

	Returns:
		int: 0 when every check passed.
	"""
	print("\n1. installing a skill: one deliberate write")

	memory = connect_local(MemoryStack())
	install_skill(memory, "rotate-deploy-key", ROTATE_KEY)
	install_skill(memory, "handle-incident"  , HANDLE_INCIDENT)

	held = memory.timeline(tier="procedural")
	check("both skills are in procedural memory", len(held) == 2, len(held))
	check("and are addressable by name",
	      {r["metadata"]["skill"] for r in held} == {"rotate-deploy-key", "handle-incident"}, held)

	print("\n2. reading procedures at task start")

	# The read that matters. An agent about to do something asks procedural
	# memory what it knows about doing it, and puts the answer in the prompt --
	# this is `recall` restricted to one tier, and it is the reason the tier is
	# small: everything in it is a candidate for every task.
	found = memory.recall(query="I need to rotate the deploy key", tier="procedural", limit=1)
	check("the right procedure is recalled for the task",
	      found and "rotate-deploy-key" == found[0]["metadata"]["skill"], found)
	check("and it comes back whole, ready to follow",
	      found and "Never revoke first" in found[0]["content"], found)

	print("       what the agent does with the text is the agent's business:")
	print("       A2M stores procedures, it never executes them.")

	print("\n3. promotion: the other way in")

	# A fact that keeps proving useful can be promoted deliberately. Note this
	# is `promote`, not spilling -- someone decided. The tier's own promote_after
	# machinery never targets procedural, because no tier promotes into it by
	# default; an operator who wants that configures MemoryTier(promote_to=...).
	learnt = memory.remember(
		"when the vault is unreachable, deploys must be paused rather than forced",
		tier = "semantic",
	)
	moved = memory.promote(learnt, "procedural")
	check("a record can be promoted into procedural", moved == 1, moved)
	check("and is then read with the other procedures",
	      len(memory.timeline(tier="procedural")) == 3, memory.timeline(tier="procedural"))

	print("\n4. nothing spills in, and nothing expires out")

	# Fill working memory well past its capacity and consolidate. Records
	# cascade working -> episodic -> semantic under pressure, and every one of
	# them stops there: no tier declares spill_to="procedural", so the pressure
	# has nowhere to push into it. A fact does not decay into a procedure.
	before = len(memory.timeline(tier="procedural"))

	for turn in range(60):
		memory.remember(f"conversational turn number {turn}", session="chat-1")

	report = memory.consolidate()
	check("consolidation moved records under pressure", report["moved"] > 0, report)
	check("but procedural is untouched by the cascade",
	      len(memory.timeline(tier="procedural")) == before, report["counts"])
	print(f"       tiers after consolidating: {report['counts']}")
	print("       ^ working spilled into episodic, episodic can spill into semantic,")
	print("         and semantic spills nowhere. Procedural is not in that chain.")

	print("\n5. unlearning: the only way out is also deliberate")

	# A wrong procedure is the most damaging record in the whole stack: a wrong
	# fact produces one wrong answer, a wrong procedure produces wrong answers
	# indefinitely. So removal is explicit, and there is no expiry to rely on.
	skill   = next(r for r in memory.timeline(tier="procedural")
	               if r["metadata"].get("skill") == "handle-incident")
	removed = memory.forget(ids=[skill["id"]])
	check("a procedure can be unlearnt", removed == 1, removed)
	check("and the others are unaffected", len(memory.timeline(tier="procedural")) == before - 1)

	print("\n6. on disk, so it can be reviewed like the code it resembles")

	# The SQLite store keeps procedural memory as files beside the database and
	# the table holds only a pointer. Editing a procedure in an editor -- or
	# reverting it in git -- takes effect without touching the database, which
	# is the point: a procedure is closer to code than to data.
	from implementations.store_sqlite import open_stack

	with tempfile.TemporaryDirectory() as directory:
		path  = pathlib.Path(directory) / "memory.db"
		stack = open_stack(str(path))
		try:
			disk = connect_local(stack)
			install_skill(disk, "rotate-deploy-key", ROTATE_KEY)

			files = sorted((path.parent / f"{path.stem}.procedural").glob("*.md"))
			check("the procedure is a file on disk", len(files) == 1, files)
			check("named after the skill", files and files[0].name.startswith("rotate-deploy-key-"), files)
			check("holding exactly what was written",
			      files and files[0].read_text(encoding="utf-8") == ROTATE_KEY, files)

			# Editing the file is editing the memory -- no write-back, no sync step.
			files[0].write_text(ROTATE_KEY.replace("one week ahead", "two weeks ahead"), encoding="utf-8")
			reread = disk.recall(query="rotate the deploy key", tier="procedural", limit=1)
			check("editing the file changes what is recalled",
			      reread and "two weeks ahead" in reread[0]["content"], reread)
			print(f"       {files[0].name}  <- put this directory under version control")
		finally:
			stack.close()

	print()
	print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
	for label in FAILED:
		print(f"    - {label}")

	return 1 if FAILED else 0


if __name__ == "__main__":
	sys.exit(main())
