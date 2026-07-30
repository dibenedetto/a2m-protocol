import http.server
import inspect
import itertools
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request


from   typing import Any, Callable, TextIO


JSONRPC_VERSION  = "2.0"

PARSE_ERROR      = -32700
INVALID_REQUEST  = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS   = -32602
INTERNAL_ERROR   = -32603


class JsonRpcError(Exception):
	"""A JSON-RPC 2.0 error, both as a raised exception and as a wire object."""

	def __init__(self, code: int, message: str, data: Any = None) -> None:
		"""Build an error with a code the specification assigns.

		Args:
			code (int): A JSON-RPC or A2M error code (spec 7).
			message (str): For humans; clients must not parse it.
			data (Any, optional): Structured detail, carried through to the wire.
		"""
		super().__init__(message)
		self.code    = code
		self.message = message
		self.data    = data


	def to_dict(self) -> dict[str, Any]:
		"""The error as a JSON-RPC error object.

		Returns:
			dict: 'code' and 'message', plus 'data' when structured detail exists.
		"""
		error = {"code": self.code, "message": self.message}
		if self.data is not None:
			error["data"] = self.data
		return error


def make_request(method: str, params: Any = None, id: Any = None) -> dict[str, Any]:
	"""Build a JSON-RPC request object.

	Args:
		method (str): Method name.
		params (Any, optional): Object or array. Omitted entirely when None.
		id (Any, optional): Correlation id. Present even when None, which is what
			distinguishes a request from a notification.

	Returns:
		dict: The request.

	Example:
		>>> make_request("memory/recall", {"query": "x"}, 1)["method"]
		'memory/recall'
	"""
	request = {"jsonrpc": JSONRPC_VERSION, "id": id, "method": method}
	if params is not None:
		request["params"] = params
	return request


def make_notification(method: str, params: Any = None) -> dict[str, Any]:
	"""Build a JSON-RPC notification -- a request that must not be answered.

	Args:
		method (str): Method name.
		params (Any, optional): Object or array.

	Returns:
		dict: The notification, carrying no 'id'.
	"""
	notification = {"jsonrpc": JSONRPC_VERSION, "method": method}
	if params is not None:
		notification["params"] = params
	return notification


def make_response(id: Any, result: Any) -> dict[str, Any]:
	"""Build a successful JSON-RPC response.

	Args:
		id (Any): The id being answered.
		result (Any): The payload.

	Returns:
		dict: The response.
	"""
	return {"jsonrpc": JSONRPC_VERSION, "id": id, "result": result}


def make_error_response(id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
	"""Build a JSON-RPC error response.

	Args:
		id (Any): The id being answered, or None when it could not be parsed.
		code (int): Error code.
		message (str): For humans; clients must not parse it.
		data (Any, optional): Structured detail.

	Returns:
		dict: The error response.
	"""
	error = JsonRpcError(code, message, data)
	return {"jsonrpc": JSONRPC_VERSION, "id": id, "error": error.to_dict()}


class Dispatcher:
	"""Maps method names to handlers and turns wire payloads into wire responses."""

	def __init__(self, allow_batch: bool = True) -> None:
		"""Create an empty dispatcher. Register handlers with 'register' or 'method'.

		Args:
			allow_batch (bool, optional): Whether to answer JSON-RPC batch arrays.
				JSON-RPC 2.0 permits them; a protocol layered on top may not, and
				A2M does not (spec 8). A refused batch answers -32600.
		"""
		self.handlers    : dict[str, Callable] = {}
		self.allow_batch = bool(allow_batch)


	def register(self, method: str, handler: Callable) -> None:
		"""Bind a handler to a method name.

		Args:
			method (str): The method name.
			handler (Callable): Called with the request's params. Parameters are bound
				against its signature, so a mismatch becomes -32602 rather than a
				crash.
		"""
		self.handlers[method] = handler


	def method(self, name: str = None) -> Callable:
		"""Decorator form of 'register'.

		Args:
			name (str, optional): Method name. Defaults to the function's own name.

		Returns:
			Callable: The decorator.

		Example:
			dispatcher = Dispatcher()

			@dispatcher.method("math/add")
			def add(a, b):
				return a + b
		"""
		def decorator(handler: Callable) -> Callable:
			"""Register the decorated function and return it unchanged.
			"""
			self.register(name or handler.__name__, handler)
			return handler
		return decorator


	def handle(self, payload: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
		"""Turn a wire payload into a wire response.

		Args:
			payload (Any): A parsed request object, or an array for a batch.

		Returns:
			dict | list | None: The response, a list of responses for a batch, or None
			when every request was a notification and nothing needs answering.
		"""
		if isinstance(payload, list):
			if not self.allow_batch:
				return make_error_response(None, INVALID_REQUEST, "Batches are not accepted")

			if not payload:
				return make_error_response(None, INVALID_REQUEST, "Batch must not be empty")

			responses = [r for r in (self._handle_one(p) for p in payload) if r is not None]
			return responses or None

		return self._handle_one(payload)


	def _handle_one(self, payload: Any) -> dict[str, Any] | None:
		"""Handle a single request object.

		Args:
			payload (Any): One parsed request.

		Returns:
			dict | None: The response, or None for a notification -- notifications are
			silent even when they fail.
		"""
		if not isinstance(payload, dict):
			return make_error_response(None, INVALID_REQUEST, "Request must be a JSON object")

		is_notification = "id" not in payload
		id              = payload.get("id", None)

		def fail(code: int, message: str, data: Any = None) -> dict[str, Any] | None:
			"""Build an error response, or None when the request was a notification.
			"""
			return None if is_notification else make_error_response(id, code, message, data)

		if payload.get("jsonrpc", None) != JSONRPC_VERSION:
			return fail(INVALID_REQUEST, f"Expected jsonrpc '{JSONRPC_VERSION}'")

		method = payload.get("method", None)
		if not isinstance(method, str):
			return fail(INVALID_REQUEST, "Missing or invalid 'method'")

		handler = self.handlers.get(method, None)
		if handler is None:
			return fail(METHOD_NOT_FOUND, f"Unknown method '{method}'")

		params = payload.get("params", None)
		if params is None:
			args, kwargs = [], {}
		elif isinstance(params, dict):
			args, kwargs = [], params
		elif isinstance(params, list):
			args, kwargs = params, {}
		else:
			return fail(INVALID_PARAMS, "'params' must be an object or an array")

		signature = None
		try:
			signature = inspect.signature(handler)
		except (TypeError, ValueError):
			signature = None

		if signature is not None:
			try:
				signature.bind(*args, **kwargs)
			except TypeError as exc:
				return fail(INVALID_PARAMS, str(exc))

		try:
			result = handler(*args, **kwargs)
		except JsonRpcError as exc:
			return fail(exc.code, exc.message, exc.data)
		except Exception as exc:
			return fail(INTERNAL_ERROR, f"Handler for '{method}' failed", repr(exc))

		return None if is_notification else make_response(id, result)


class Transport:
	"""Carries a JSON-RPC payload to a peer and brings back the response, if any."""

	def send(self, payload: dict[str, Any]) -> dict[str, Any] | None:
		"""Carry one payload to the peer and bring back the response.

		Args:
			payload (dict): A JSON-RPC request or notification.

		Returns:
			dict | None: The response, or None for a notification.
		"""
		raise NotImplementedError


	def request_raw(self, payload: Any) -> Any:
		"""Send exactly this payload and read exactly one reply.

		'send' pairs a response to a request by id, so it cannot express a message
		that has no id to pair on -- a batch, or a deliberately malformed request.
		A conformance suite has to send those anyway, since refusing them correctly
		is part of what it is checking.

		Args:
			payload (Any): Anything JSON-serialisable, valid or not.

		Returns:
			Any: The parsed reply, or None if the peer sent nothing.
		"""
		raise NotImplementedError


	def close(self) -> None:
		"""Release whatever the transport holds. Safe to call more than once.
		"""
		pass


class LocalTransport(Transport):
	"""In-process peer. Payloads still round-trip through JSON so that a local
	server and a remote one cannot disagree on what is representable."""

	def __init__(self, dispatcher: Dispatcher) -> None:
		"""Wrap an in-process dispatcher as a transport.

		Args:
			dispatcher (Dispatcher): The in-process server to dispatch into.
		"""
		self.dispatcher = dispatcher


	def send(self, payload: dict[str, Any]) -> dict[str, Any] | None:
		"""Dispatch in-process, still round-tripping through JSON.

		Args:
			payload (dict): A JSON-RPC request or notification.

		Returns:
			dict | None: The response, or None for a notification.
		"""
		request  = json.loads(json.dumps(payload))
		response = self.dispatcher.handle(request)
		return json.loads(json.dumps(response)) if response is not None else None


	def request_raw(self, payload: Any) -> Any:
		"""Dispatch anything at all and return whatever comes back.

		Args:
			payload (Any): Anything JSON-serialisable, valid or not.

		Returns:
			Any: The parsed reply, or None if the dispatcher answered nothing.
		"""
		return self.send(payload)


class StdioTransport(Transport):
	"""Newline-delimited JSON over a child process' stdin/stdout, as used by MCP."""

	def __init__(self, command: list[str], env: dict[str, str] = None, cwd: str = None, on_stderr: Callable = None) -> None:
		"""Launch the peer and start draining its stderr.

		Args:
			command (list[str]): The command to run, e.g.
				[sys.executable, "-m", "implementations.store_sqlite", "memory.db"].
			env (dict, optional): Child environment. Inherits when omitted.
			cwd (str, optional): Child working directory.
			on_stderr (Callable, optional): Called with each stderr line, stripped.
				A child that fills its stderr pipe blocks forever, so the drain thread
				runs whether or not this is supplied.

		Raises:
			OSError: If the command cannot be launched.
		"""
		self.command       = list(command)
		self.notifications : list[dict[str, Any]] = []
		self.process       = subprocess.Popen(
			self.command,
			stdin    = subprocess.PIPE,
			stdout   = subprocess.PIPE,
			stderr   = subprocess.PIPE,
			env      = env,
			cwd      = cwd,
			text     = True,
			encoding = "utf-8",
			bufsize  = 1,
		)

		self._lock   = threading.Lock()
		self._stderr = threading.Thread(target=self._drain_stderr, args=(on_stderr,), daemon=True)
		self._stderr.start()


	def send(self, payload: dict[str, Any]) -> dict[str, Any] | None:
		"""Write one line to the child and read until its answer arrives.

		Args:
			payload (dict): A JSON-RPC request or notification.

		Returns:
			dict | None: The response, or None for a notification.

		Raises:
			JsonRpcError: -32603 if the child has exited or closed its stream.
		"""
		with self._lock:
			if self.process.poll() is not None:
				raise JsonRpcError(INTERNAL_ERROR, f"Peer '{self.command[0]}' has exited")

			self.process.stdin.write(json.dumps(payload) + "\n")
			self.process.stdin.flush()

			if "id" not in payload:
				return None

			return self._read_until(payload["id"])


	def request_raw(self, payload: Any) -> Any:
		"""Write anything at all and read back the next message the peer sends.

		Args:
			payload (Any): Anything JSON-serialisable, valid or not.

		Returns:
			Any: The next parsed message.

		Raises:
			JsonRpcError: -32603 if the child has exited or closes the stream.
		"""
		with self._lock:
			if self.process.poll() is not None:
				raise JsonRpcError(INTERNAL_ERROR, f"Peer '{self.command[0]}' has exited")

			self.process.stdin.write(json.dumps(payload) + "\n")
			self.process.stdin.flush()

			return self._read_message()


	def _read_message(self) -> Any:
		"""Read lines until one parses as JSON, and return it.

		Returns:
			Any: The parsed message.

		Raises:
			JsonRpcError: -32603 if the stream closes first.
		"""
		while True:
			line = self.process.stdout.readline()
			if not line:
				raise JsonRpcError(INTERNAL_ERROR, f"Peer '{self.command[0]}' closed the stream")

			line = line.strip()
			if not line:
				continue

			try:
				return json.loads(line)
			except json.JSONDecodeError:
				continue


	def _read_until(self, id: Any) -> dict[str, Any]:
		"""Read lines until the response with this id appears.

		Anything else -- server notifications, messages for another in-flight request
		-- is stashed rather than discarded, since a transport must not lose messages
		it did not expect.

		Args:
			id (Any): The id being awaited.

		Returns:
			dict: The matching response.

		Raises:
			JsonRpcError: -32603 if the stream closes first.
		"""
		while True:
			line = self.process.stdout.readline()
			if not line:
				raise JsonRpcError(INTERNAL_ERROR, f"Peer '{self.command[0]}' closed the stream")

			line = line.strip()
			if not line:
				continue

			try:
				message = json.loads(line)
			except json.JSONDecodeError:
				continue

			if isinstance(message, dict) and message.get("id", None) == id:
				return message

			self.notifications.append(message)


	def _drain_stderr(self, on_stderr: Callable = None) -> None:
		"""Consume the child's stderr on a daemon thread.

		A child that fills its stderr pipe blocks forever, so this must run whether or
		not anyone is listening.

		Args:
			on_stderr (Callable, optional): Called per line, already stripped.
		"""
		for line in self.process.stderr:
			if on_stderr:
				on_stderr(line.rstrip())


	def close(self) -> None:
		"""Close the child's stdin and wait for it, killing it if it will not exit.
		"""
		if self.process.poll() is None:
			try:
				self.process.stdin.close()
				self.process.wait(timeout=5)
			except Exception:
				self.process.kill()


class HttpTransport(Transport):
	"""One endpoint, POST, `application/json`.

	Protocol-level failures come back as 200 carrying a JSON-RPC error object;
	only transport-level failures use HTTP status codes. Confusing the two is the
	usual way a JSON-RPC-over-HTTP binding goes wrong."""

	def __init__(self, url: str, headers: dict[str, str] = None, timeout: float = 30.0) -> None:
		"""Point a transport at one HTTP endpoint.

		Args:
			url (str): The endpoint to POST to.
			headers (dict, optional): Extra headers, typically Authorization.
				Content-Type is set automatically.
			timeout (float, optional): Per-request timeout in seconds.
		"""
		self.url     = url
		self.headers = dict(headers) if headers else {}
		self.timeout = float(timeout)


	def send(self, payload: dict[str, Any]) -> dict[str, Any] | None:
		"""POST one payload and read the response.

		Args:
			payload (dict): A JSON-RPC request or notification.

		Returns:
			dict | None: The response, or None for a notification (HTTP 202).

		Raises:
			JsonRpcError: -32603 for transport-level failures. Protocol-level failures
				arrive as HTTP 200 carrying a JSON-RPC error, and are returned rather
				than raised here.
		"""
		body    = json.dumps(payload).encode("utf-8")
		headers = dict(self.headers)
		headers.setdefault("Content-Type", "application/json")

		request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")

		try:
			with urllib.request.urlopen(request, timeout=self.timeout) as response:
				if response.status == 202 or "id" not in payload:
					return None

				raw = response.read().decode("utf-8").strip()
				return json.loads(raw) if raw else None

		except urllib.error.HTTPError as exc:
			raise JsonRpcError(INTERNAL_ERROR, f"HTTP {exc.code} from {self.url}", exc.read().decode("utf-8", "replace")[:500])
		except urllib.error.URLError as exc:
			raise JsonRpcError(INTERNAL_ERROR, f"Cannot reach {self.url}", str(exc.reason))


	def request_raw(self, payload: Any) -> Any:
		"""POST anything at all and parse whatever body comes back.

		Unlike 'send', an HTTP error status is not itself the failure: a binding
		that refuses a request carries its reason in a JSON-RPC error object in the
		body, and that object is the answer being asked for here.

		Args:
			payload (Any): Anything JSON-serialisable, valid or not.

		Returns:
			Any: The parsed body, or None if the server sent none.

		Raises:
			JsonRpcError: -32603 if the endpoint cannot be reached, or answered a
				failure with no JSON body to explain it.
		"""
		body    = json.dumps(payload).encode("utf-8")
		headers = dict(self.headers)
		headers.setdefault("Content-Type", "application/json")

		request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")

		try:
			with urllib.request.urlopen(request, timeout=self.timeout) as response:
				raw = response.read().decode("utf-8").strip()
				return json.loads(raw) if raw else None

		except urllib.error.HTTPError as exc:
			raw = exc.read().decode("utf-8", "replace").strip()
			try:
				return json.loads(raw)
			except json.JSONDecodeError:
				raise JsonRpcError(INTERNAL_ERROR, f"HTTP {exc.code} from {self.url}", raw[:500])
		except urllib.error.URLError as exc:
			raise JsonRpcError(INTERNAL_ERROR, f"Cannot reach {self.url}", str(exc.reason))


def serve_http(
	dispatcher      : Dispatcher,
	host            : str            = "127.0.0.1",
	port            : int            = 8778,
	path            : str            = "/",
	allowed_origins : list[str]      = None,
	on_headers      : Callable       = None,
	well_known      : dict[str, Any] = None,
) -> http.server.HTTPServer:
	"""Server side of HttpTransport. Returns the server without starting it, so a
	caller can decide between `serve_forever()` and a background thread.

	Args:
		dispatcher (Dispatcher): What answers the requests.
		host (str, optional): Bind address. Loopback by default, because a server
			reachable from the network is a decision worth taking deliberately.
		port (int, optional): Bind port.
		path (str, optional): The single endpoint path, POST only.
		allowed_origins (list[str], optional): Origins permitted to call this
			endpoint. The default refuses any request carrying an 'Origin' header
			at all, which is what stops a page the user happens to be visiting from
			driving a server bound to their own loopback interface. Pass ["*"] to
			allow every origin, or a list to allow those exactly.
		on_headers (Callable, optional): Called with the request headers before the
			body is read. Return None to accept, or (status, error object) to
			refuse. This is the seam where a protocol layered on JSON-RPC adds its
			own header rules without this module having to know them.
		well_known (dict, optional): Documents to serve on GET, keyed by exact
			path, e.g. {"/.well-known/a2m-server.json": profile}.
	"""
	origins   = list(allowed_origins) if allowed_origins else []
	documents = dict(well_known) if well_known else {}

	class Handler(http.server.BaseHTTPRequestHandler):

		"""Minimal JSON-RPC-over-HTTP handler for one endpoint.
		"""
		def _origin_allowed(self) -> bool:
			"""Whether this request's Origin, if it has one, may call this endpoint.

			A request with no Origin is not from a browser and is allowed; that is
			the ordinary case for an agent, a CLI or an SDK.

			Returns:
				bool: True when the request may proceed.
			"""
			origin = self.headers.get("Origin", None)
			if origin is None:
				return True

			return "*" in origins or origin in origins


		def do_GET(self) -> None:
			"""Serve a well-known document. The RPC endpoint itself is POST-only.
			"""
			if not self._origin_allowed():
				self.send_error(403, "Origin not allowed")
				return

			if self.path in documents:
				self._reply(200, documents[self.path])
				return

			if self.path.rstrip("/") == path.rstrip("/"):
				self.send_error(405, "This endpoint accepts POST")
				return

			self.send_error(404, "No such endpoint")


		def do_POST(self) -> None:
			"""Handle one request: 200 with a response, or 202 with nothing for a notification.
			"""
			if not self._origin_allowed():
				self.send_error(403, "Origin not allowed")
				return

			if self.path.rstrip("/") != path.rstrip("/"):
				self.send_error(404, "No such endpoint")
				return

			if on_headers is not None:
				refusal = on_headers(self.headers)
				if refusal is not None:
					status, error = refusal
					self._reply(status, error)
					return

			length = int(self.headers.get("Content-Length", 0) or 0)
			raw    = self.rfile.read(length).decode("utf-8") if length else ""

			try:
				payload = json.loads(raw)
			except json.JSONDecodeError as exc:
				self._reply(200, make_error_response(None, PARSE_ERROR, "Invalid JSON", str(exc)))
				return

			response = dispatcher.handle(payload)

			if response is None:
				self.send_response(202)
				self.send_header("Content-Length", "0")
				self.end_headers()
				return

			self._reply(200, response)


		def _reply(self, status: int, payload: Any) -> None:
			"""Write a JSON body with the right headers.

			Args:
				status (int): HTTP status.
				payload (Any): JSON-serialisable body.
			"""
			body = json.dumps(payload).encode("utf-8")
			self.send_response(status)
			self.send_header("Content-Type", "application/json")
			self.send_header("Content-Length", str(len(body)))
			self.end_headers()
			self.wfile.write(body)


		def log_message(self, *args: Any) -> None:
			"""Silence the default stderr access log.
			"""
			pass

	return http.server.HTTPServer((host, port), Handler)


class Client:
	"""Issues requests over a transport and unwraps results or raises errors."""

	def __init__(self, transport: Transport) -> None:
		"""Wrap a transport so calls return results and raise errors.

		Args:
			transport (Transport): Any transport -- local, stdio or HTTP. The client
				is identical across all three, which is the point of the boundary.
		"""
		self.transport = transport

		self._counter  = itertools.count(1)
		self._lock     = threading.Lock()


	def call(self, method: str, params: Any = None) -> Any:
		"""Issue a request and unwrap the result.

		Args:
			method (str): Method name.
			params (Any, optional): Object or array.

		Returns:
			Any: The 'result' payload.

		Raises:
			JsonRpcError: Carrying the server's code, message and data.

		Example:
			client = Client(LocalTransport(dispatcher))
			client.call("memory/recall", {"query": "where does marco live"})
		"""
		with self._lock:
			id = next(self._counter)

		response = self.transport.send(make_request(method, params, id))
		if response is None:
			raise JsonRpcError(INTERNAL_ERROR, f"No response for '{method}'")

		error = response.get("error", None)
		if error:
			raise JsonRpcError(error.get("code", INTERNAL_ERROR), error.get("message", ""), error.get("data", None))

		return response.get("result", None)


	def notify(self, method: str, params: Any = None) -> None:
		"""Send a notification, which by definition returns nothing.

		Args:
			method (str): Method name.
			params (Any, optional): Object or array.
		"""
		self.transport.send(make_notification(method, params))


	def close(self) -> None:
		"""Close the underlying transport.
		"""
		self.transport.close()


def serve_stdio(dispatcher: Dispatcher, stdin: TextIO = None, stdout: TextIO = None) -> None:
	"""Server side of StdioTransport: read requests until the stream closes."""
	stdin  = stdin  or sys.stdin
	stdout = stdout or sys.stdout

	for line in stdin:
		line = line.strip()
		if not line:
			continue

		try:
			payload = json.loads(line)
		except json.JSONDecodeError as exc:
			response = make_error_response(None, PARSE_ERROR, "Invalid JSON", str(exc))
		else:
			response = dispatcher.handle(payload)

		if response is not None:
			stdout.write(json.dumps(response) + "\n")
			stdout.flush()
