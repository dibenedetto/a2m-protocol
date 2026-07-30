"""Render the site: Markdown from the repository, HTML into `www/public/`.

	python www/render.py            # build into www/public
	python www/render.py --serve    # build, then serve it on 8000

The directory is `www/` and not `site/` because `site` is a standard-library
module name, and a package shadowing it breaks `python -m` for everything.

The specification, the implementation guide and the decision log are rendered
from the files in the repository rather than copied into the site, so the two
cannot disagree. A page here is never the source of anything.

Standard library only, like everything else. That is not decoration: a docs
generator would have made the site the first thing in this project that has to
be *built* before it can be read, in a repository whose argument is that you can
clone it and run it. The Markdown subset below is small on purpose -- it handles
what these documents actually use, and anything it does not understand passes
through as text rather than silently disappearing.
"""


import html
import pathlib
import re
import sys


ROOT   = pathlib.Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "www" / "public"

# What gets rendered, and what it is called in the navigation. Order is the
# reading order the README recommends: the specification first, because
# everything else is downstream of it.
PAGES = [
	("spec/a2m-0.1.md"        , "spec.html"       , "Specification"   , "A2M 0.1 — the normative specification"),
	("spec/implementing-a2m.md", "implementing.html", "Implementing"   , "Implementing A2M — where each tier lives"),
	("DECISIONS.md"           , "decisions.html"  , "Decisions"        , "Decisions — why the non-obvious choices are what they are"),
]

NAV = [
	("index.html"       , "Home"),
	("spec.html"        , "Specification"),
	("implementing.html", "Implementing"),
	("decisions.html"   , "Decisions"),
	("https://github.com/dibenedetto/a2m-protocol", "GitHub"),
]


def inline(text: str) -> str:
	"""Render the inline Markdown these documents use.

	Escaping happens first and exactly once, so a literal `<` in prose survives
	and no rendered tag can be smuggled in from the source.

	Args:
		text (str): One line of Markdown, unescaped.

	Returns:
		str: HTML.
	"""
	out = html.escape(text, quote=False)

	# Code first: nothing inside a span of code is markup.
	codes = []
	def stash(match: re.Match) -> str:
		codes.append(match.group(1))
		return f"\x00{len(codes) - 1}\x00"

	out = re.sub(r"`([^`]+)`", stash, out)

	out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
	out = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<em>\1</em>", out)

	# [text](target) -- a link to a repository file becomes a link to GitHub,
	# because the reader is on a website and the file is not here.
	def link(match: re.Match) -> str:
		label, target = match.group(1), match.group(2)
		if target.startswith(("http://", "https://", "#")):
			pass
		elif target.endswith(".md"):
			target = {
				"spec/a2m-0.1.md"         : "spec.html",
				"a2m-0.1.md"              : "spec.html",
				"spec/implementing-a2m.md": "implementing.html",
				"implementing-a2m.md"     : "implementing.html",
				"DECISIONS.md"            : "decisions.html",
				"../DECISIONS.md"         : "decisions.html",
			}.get(target.lstrip("./"), f"https://github.com/dibenedetto/a2m-protocol/blob/main/{target.lstrip('./')}")
		else:
			target = f"https://github.com/dibenedetto/a2m-protocol/blob/main/{target.lstrip('./')}"
		return f'<a href="{html.escape(target, quote=True)}">{label}</a>'

	out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, out)

	# An autolink is written <https://...> in the source, so by this point the
	# angle brackets have already been escaped. Matching the escaped form is the
	# whole trick -- matching the raw one silently does nothing.
	out = re.sub(r"&lt;(https?://[^&\s]+)&gt;", r'<a href="\1">\1</a>', out)

	for index, code in enumerate(codes):
		out = out.replace(f"\x00{index}\x00", f"<code>{code}</code>")

	return out


def slug(text: str) -> str:
	"""A stable anchor for a heading, so sections can be linked to directly.

	Args:
		text (str): The heading text, already stripped of its hashes.

	Returns:
		str: A url fragment.
	"""
	return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render(markdown: str) -> str:
	"""Render the Markdown subset these documents use.

	Handles headings, fenced code, tables, blockquotes, ordered and unordered
	lists, horizontal rules and paragraphs. Reference-style link definitions are
	resolved; anything else unrecognised is emitted as a paragraph rather than
	dropped, because a renderer that silently swallows content is worse than one
	that renders it plainly.

	Args:
		markdown (str): The document.

	Returns:
		str: HTML.
	"""
	# Resolve [label]: target definitions, then remove them.
	definitions = dict(re.findall(r"^\[([^\]]+)\]:\s*(\S+)\s*$", markdown, re.MULTILINE))
	markdown    = re.sub(r"^\[[^\]]+\]:\s*\S+\s*$", "", markdown, flags=re.MULTILINE)
	for label, target in definitions.items():
		markdown = markdown.replace(f"[{label}][{label}]", f"[{label}]({target})")
		markdown = re.sub(r"\[([^\]]+)\]\[" + re.escape(label) + r"\]", rf"[\1]({target})", markdown)

	lines = markdown.splitlines()
	out   = []
	index = 0

	while index < len(lines):
		line = lines[index]

		if line.startswith("```"):
			language = line[3:].strip()
			index   += 1
			block    = []
			while index < len(lines) and not lines[index].startswith("```"):
				block.append(lines[index])
				index += 1
			index += 1
			klass  = f' class="language-{html.escape(language, quote=True)}"' if language else ""
			out.append(f"<pre><code{klass}>{html.escape(chr(10).join(block), quote=False)}</code></pre>")
			continue

		if re.match(r"^#{1,6}\s", line):
			level   = len(line) - len(line.lstrip("#"))
			text    = line[level:].strip()
			anchor  = slug(text)
			out.append(f'<h{level} id="{anchor}"><a class="anchor" href="#{anchor}">#</a>{inline(text)}</h{level}>')
			index += 1
			continue

		if re.match(r"^\s*([-*_])\s*\1\s*\1[\s\-*_]*$", line):
			out.append("<hr>")
			index += 1
			continue

		# A table: a header row, a separator of dashes, then body rows.
		if line.strip().startswith("|") and index + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[index + 1]):
			def cells(row: str) -> list[str]:
				return [cell.strip() for cell in row.strip().strip("|").split("|")]

			header = cells(line)
			index += 2
			body   = []
			while index < len(lines) and lines[index].strip().startswith("|"):
				body.append(cells(lines[index]))
				index += 1

			head = "".join(f"<th>{inline(cell)}</th>" for cell in header)
			rows = "".join("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>" for row in body)
			out.append(f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')
			continue

		if line.startswith(">"):
			block = []
			while index < len(lines) and lines[index].startswith(">"):
				block.append(lines[index].lstrip(">").strip())
				index += 1
			out.append(f"<blockquote>{inline(' '.join(block))}</blockquote>")
			continue

		bullet  = re.match(r"^\s*[-*]\s+(.*)$", line)
		ordered = re.match(r"^\s*\d+\.\s+(.*)$", line)
		if bullet or ordered:
			pattern = r"^\s*\d+\.\s+(.*)$" if ordered else r"^\s*[-*]\s+(.*)$"
			tag     = "ol" if ordered else "ul"
			items   = []
			while index < len(lines):
				match = re.match(pattern, lines[index])
				if match:
					items.append(match.group(1))
					index += 1
				# A wrapped continuation line belongs to the item above it.
				elif items and lines[index].startswith(("  ", "\t")) and lines[index].strip():
					items[-1] += " " + lines[index].strip()
					index += 1
				else:
					break
			body = "".join(f"<li>{inline(item)}</li>" for item in items)
			out.append(f"<{tag}>{body}</{tag}>")
			continue

		if not line.strip():
			index += 1
			continue

		paragraph = []
		while index < len(lines) and lines[index].strip() and not re.match(r"^(#{1,6}\s|```|>|\s*[-*]\s|\s*\d+\.\s|\|)", lines[index]):
			paragraph.append(lines[index].strip())
			index += 1
		if paragraph:
			out.append(f"<p>{inline(' '.join(paragraph))}</p>")
		else:
			index += 1

	return "\n".join(out)


def page(title: str, body: str, current: str, description: str = "") -> str:
	"""Wrap rendered content in the site shell.

	Args:
		title (str): The document title.
		body (str): Rendered HTML.
		current (str): The current page's filename, for marking the nav.
		description (str, optional): Meta description, for search and previews.

	Returns:
		str: A complete HTML document.
	"""
	nav = "".join(
		f'<a href="{href}"{" class=\"here\"" if href == current else ""}>{label}</a>'
		for href, label in NAV
	)

	return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title, quote=False)}</title>
<meta name="description" content="{html.escape(description or title, quote=True)}">
<meta property="og:title" content="{html.escape(title, quote=True)}">
<meta property="og:description" content="{html.escape(description or title, quote=True)}">
<meta property="og:type" content="website">
<link rel="stylesheet" href="style.css">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#129504;</text></svg>">
</head>
<body>
<header>
<a class="wordmark" href="index.html">A2M</a>
<nav>{nav}</nav>
</header>
<main>
{body}
</main>
<footer>
<p>A2M — the Agent-to-Memory Protocol. MIT licensed.
<a href="https://github.com/dibenedetto/a2m-protocol">Source</a> ·
<a href="https://github.com/dibenedetto/a2m-protocol/blob/main/spec/a2m-0.1.md">Specification</a></p>
<p class="muted">A protocol that is expensive to implement does not get implemented.</p>
</footer>
</body>
</html>
"""


def build() -> int:
	"""Render every page into `www/public/`.

	Returns:
		int: How many files were written.
	"""
	PUBLIC.mkdir(parents=True, exist_ok=True)
	written = 0

	(PUBLIC / "style.css").write_text((ROOT / "www" / "style.css").read_text(encoding="utf-8"), encoding="utf-8")
	written += 1

	home = (ROOT / "www" / "index.html").read_text(encoding="utf-8")
	(PUBLIC / "index.html").write_text(home, encoding="utf-8")
	written += 1

	for source, target, label, description in PAGES:
		markdown = (ROOT / source).read_text(encoding="utf-8")
		(PUBLIC / target).write_text(
			page(f"{label} — A2M", render(markdown), target, description),
			encoding="utf-8",
		)
		written += 1
		print(f"    {source}  ->  www/public/{target}")

	# GitHub Pages would otherwise run the output through Jekyll.
	(PUBLIC / ".nojekyll").write_text("", encoding="utf-8")
	written += 1

	return written


def main() -> int:
	"""Build the site, and optionally serve it.

	Returns:
		int: Process exit code.
	"""
	print(f"  rendering into {PUBLIC.relative_to(ROOT)}/")
	written = build()
	print(f"  {written} files written.")

	if "--serve" in sys.argv[1:]:
		import functools
		import http.server

		handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
		print("  serving on http://127.0.0.1:8000/  (ctrl-c to stop)")
		http.server.HTTPServer(("127.0.0.1", 8000), handler).serve_forever()

	return 0


if __name__ == "__main__":
	sys.exit(main())
