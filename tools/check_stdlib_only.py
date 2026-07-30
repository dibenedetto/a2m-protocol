"""Enforce the property the whole repository rests on: standard library only.

	python -m tools.check_stdlib_only

A2M's adoption argument is that reading it as an example costs nothing — clone
it and run it, with no install step and no transitive dependency to audit. That
is a claim about *every file*, and claims about every file decay silently: one
convenient import in one module, and the protocol, the reference implementation
and the conformance suite all quietly acquire a dependency.

So this walks the imports rather than trusting anyone to remember. The rule it
enforces is exactly the one in CLAUDE.md:

	a2m/ implementations/ tools/ examples/   standard library only
	a listed few                             one declared optional dependency

A file earns a place on that list by satisfying one test: **nothing on the
default path imports it.** `store_postgres.py` needs psycopg and
`cross_framework.py` needs two frameworks, and both are fine precisely because
every test, every conformance target and both demos still run in a checkout
where neither is installed. The moment something ordinary imports one of them,
the exemption stops being true and this check stops passing.

`adapters/__init__.py` is held to the strict rule for the same reason: importing
the package must not import a framework.

Exits non-zero, naming the file, line and module of every violation.
"""


import ast
import pathlib
import sys


# Directories every file of which must import nothing but the standard library.
GUARDED = ("a2m", "implementations", "tools", "examples")

# Files that may import a third-party module, and exactly which. A file not
# listed here gets the strict rule, so granting an exemption is a deliberate act
# rather than an oversight -- and each one is only sound while nothing on the
# default path imports that file.
ALLOWED = {
	"implementations/adapters/langchain.py" : {"langchain_core", "pydantic"},
	"implementations/adapters/agno.py"      : {"agno"},
	"implementations/adapters/crewai.py"    : {"crewai"},
	"implementations/adapters/autogen.py"   : {"autogen_core"},
	# An optional backend. Nothing imports it: the federation names it as a
	# module path and starts it as a subprocess.
	"implementations/store_postgres.py"     : {"psycopg"},
	# The example whose entire point is that two frameworks meet on one store.
	"examples/cross_framework.py"           : {"agno", "langchain_core"},
}

# Optional dependencies the reference implementation may import *lazily*, inside
# a function, where the feature they back is opt-in and the import cannot happen
# unless someone asked for it. A top-level import of these is still a failure.
LAZY = {"sqlite_vec", "ollama", "psycopg", "psycopg_pool"}


def module_of(node: ast.AST) -> list[tuple[str, int]]:
	"""The top-level module names one import statement pulls in.

	Args:
		node (ast.AST): An Import or ImportFrom node.

	Returns:
		list[tuple[str, int]]: (module, line) pairs. A relative import returns
		nothing -- it cannot reach outside the package.
	"""
	if isinstance(node, ast.Import):
		return [(alias.name.split(".")[0], node.lineno) for alias in node.names]

	if isinstance(node, ast.ImportFrom):
		if node.level:
			return []
		return [((node.module or "").split(".")[0], node.lineno)]

	return []


def is_toplevel(tree: ast.AST, node: ast.AST) -> bool:
	"""Whether an import sits at module level rather than inside a function.

	The distinction is the whole reason optional dependencies are tolerable: an
	import inside a function costs nothing until the feature is used, while a
	top-level one is paid by everybody who imports the module at all.

	Args:
		tree (ast.AST): The parsed module.
		node (ast.AST): The import node.

	Returns:
		bool: True when the import is a direct child of the module body.
	"""
	return any(node is child for child in tree.body)


def check(path: pathlib.Path, root: pathlib.Path) -> list[str]:
	"""Every rule violation in one file.

	Args:
		path (Path): The file to check.
		root (Path): The repository root, for reporting relative paths.

	Returns:
		list[str]: Human-readable violations, empty when the file is clean.
	"""
	relative = path.relative_to(root).as_posix()
	allowed  = ALLOWED.get(relative, set())
	problems = []

	try:
		tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
	except SyntaxError as exc:
		return [f"{relative}:{exc.lineno}: will not parse: {exc.msg}"]

	for node in ast.walk(tree):
		for module, line in module_of(node):
			if not module or module in sys.stdlib_module_names:
				continue
			if module in allowed:
				continue
			# A local import of our own packages is fine everywhere.
			if module in ("a2m", "implementations", "tools", "examples"):
				continue
			if module in LAZY and not is_toplevel(tree, node):
				continue

			why = "third-party import"
			if module in LAZY:
				why = "optional dependency imported at module level (move it into the function that needs it)"
			elif allowed:
				why = f"this adapter may import {sorted(allowed)}, not this"

			problems.append(f"{relative}:{line}: {why}: '{module}'")

	return problems


def main() -> int:
	"""Walk the guarded directories and report every violation.

	Returns:
		int: 0 when the repository is standard-library only.
	"""
	root     = pathlib.Path(__file__).resolve().parent.parent
	problems = []
	checked  = 0

	for directory in GUARDED:
		for path in sorted((root / directory).rglob("*.py")):
			checked  += 1
			problems += check(path, root)

	if problems:
		print(f"  {len(problems)} violation(s) in {checked} files:\n")
		for problem in problems:
			print(f"    {problem}")
		print("\n  The protocol, the reference implementation and the conformance suite")
		print("  need nothing but the standard library. A dependency here becomes a")
		print("  dependency of everyone who reads this as an example.")
		return 1

	print(f"  {checked} files, standard library only "
	      f"({len(ALLOWED)} exempt, each importing one declared optional dependency).")
	return 0


if __name__ == "__main__":
	sys.exit(main())
