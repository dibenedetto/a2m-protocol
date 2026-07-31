"""Are the n8n workflow's requests actually valid A2M?

	python -m a2m --http 8778 &
	python -m tools.check_n8n

`examples/n8n_workflow.json` is the n8n integration: no custom node, no
community package, just HTTP Request nodes carrying JSON-RPC. That makes it a
genuine A2M client — and one nothing could check, because it lives in a JSON
file that only n8n knows how to execute.

So this pulls the request bodies straight out of the workflow and sends them at
a running server. It cannot test n8n, which would need n8n; it tests the half
that can break silently while looking fine in an editor: whether the workflow
is asking a real A2M server real questions.

A wrong URL, a stale method name, a parameter renamed by a spec change — all of
them show up here as an error object rather than as a support issue from
somebody's first attempt.
"""


import json
import pathlib
import sys
import urllib.error
import urllib.request


ROOT     = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / "examples" / "n8n_workflow.json"


def bodies(workflow: dict) -> list[tuple[str, dict]]:
	"""Every JSON-RPC body the workflow's HTTP nodes send, in node order.

	Args:
		workflow (dict): The parsed n8n workflow.

	Returns:
		list[tuple[str, dict]]: (node name, parsed body) pairs.

	Raises:
		ValueError: If a node's body is not parseable JSON, which is a broken
			workflow regardless of what any server would say about it.
	"""
	found = []

	for node in workflow.get("nodes", []):
		if node.get("type") != "n8n-nodes-base.httpRequest":
			continue

		raw = (node.get("parameters") or {}).get("jsonBody")
		if not raw:
			continue

		try:
			found.append((node.get("name", "?"), json.loads(raw)))
		except json.JSONDecodeError as exc:
			raise ValueError(f"node {node.get('name')!r} carries invalid JSON: {exc}")

	return found


def send(url: str, body: dict, timeout: float = 15.0) -> dict:
	"""POST one JSON-RPC body, exactly as the workflow's node would.

	Args:
		url (str): The endpoint.
		body (dict): The JSON-RPC request.
		timeout (float, optional): Seconds.

	Returns:
		dict: The parsed response.
	"""
	request = urllib.request.Request(
		url,
		data    = json.dumps(body).encode("utf-8"),
		headers = {"Content-Type": "application/json", "A2M-Protocol-Version": "a2m/0.1"},
		method  = "POST",
	)

	with urllib.request.urlopen(request, timeout=timeout) as response:
		return json.loads(response.read().decode("utf-8"))


def main() -> int:
	"""Replay every request body in the workflow against a running server.

	Returns:
		int: 0 when every node's request is answered without a protocol error.
	"""
	url      = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8778/"
	workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
	requests = bodies(workflow)

	print(f"\n  {WORKFLOW.relative_to(ROOT)} -> {url}")

	if not requests:
		print("  FAIL  the workflow contains no HTTP request bodies at all")
		return 1

	failed = 0
	for name, body in requests:
		try:
			answer = send(url, body)
		except (urllib.error.URLError, OSError) as exc:
			print(f"    FAIL  {name}: cannot reach the server ({exc})")
			failed += 1
			continue

		error = answer.get("error")
		if error:
			print(f"    FAIL  {name}: [{error.get('code')}] {error.get('message')}")
			failed += 1
		else:
			method = body.get("method", "?")
			print(f"    ok    {name}  ({method})")

	print(f"\n  {len(requests) - failed}/{len(requests)} of the workflow's requests are valid A2M")
	return 1 if failed else 0


if __name__ == "__main__":
	sys.exit(main())
