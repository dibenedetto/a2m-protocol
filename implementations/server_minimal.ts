/**
 * A minimal A2M 0.1 server in TypeScript, declaring `core` and `keys`.
 *
 *     node --experimental-strip-types implementations/server_minimal.ts        # stdio
 *
 * No dependencies, no package.json, no build step. Node strips the types and
 * runs the file; everything else is the standard library, which is the same
 * promise the Python side makes.
 *
 * This is a port of server_minimal.py, and the port is the point. That file answers
 * "is the specification enough on its own?"; this one answers "is it enough in a
 * language that is not the reference language?" — the question a transport-
 * agnostic, JSON-shaped protocol has to survive to deserve the description.
 *
 * It is checked by the same suite, which never imports the server it tests:
 *
 *     python -m tools.conformance --stdio node --experimental-strip-types implementations/server_minimal.ts
 *
 * Two things the port had to get right that a same-language port would hide:
 *
 * - **Timestamps are RFC 3339 strings, never numbers** (spec §3.3). JavaScript's
 *   instinct is `Date.now()`, a millisecond epoch integer, which is exactly the
 *   ambiguity the rule exists to remove. `toISOString()` is already the right
 *   shape.
 * - **`id` is opaque and its type must survive** (spec §3.2). JSON-RPC ids may be
 *   strings or numbers, and JavaScript is happy to coerce between them. An id is
 *   echoed back exactly as it arrived, never normalised.
 *
 * `keys` is declared as well as `core`, so the port also demonstrates an
 * *optional* capability surviving the language change: a write to an occupied
 * key replaces what is there — same id, new content, `revision` + 1 — which is
 * what makes a memory correctable rather than merely appendable (spec §3.6).
 */

const PROTOCOL = "a2m/0.1";
const CAPABILITIES = ["core", "keys"];
const METHODS = [
	"memory/describe",
	"memory/remember",
	"memory/recall",
	"memory/timeline",
	"memory/forget",
	"memory/fetch",
];

// spec §7
const INVALID_REQUEST = -32600;
const METHOD_NOT_FOUND = -32601;
const INVALID_PARAMS = -32602;
const INTERNAL_ERROR = -32603;
const PARSE_ERROR = -32700;
const QUOTA_EXCEEDED = -32006;
const CAPABILITY_NOT_SUPPORTED = -32003;
const PROTOCOL_NOT_SUPPORTED = -32007;

const MAX_RECORDS_PER_CALL = 100;

type Json = any;

interface Stored {
	id: string;
	content: string;
	role: string;
	metadata: Record<string, Json>;
	group?: string;
	key?: string;
	revision: number;
	created_at: string;
	sequence: number;
}

const RECORDS = new Map<string, Stored>();

/** An A2M error, carrying the code the specification assigns it. */
class A2MError extends Error {
	code: number;
	data?: Json;

	constructor(code: number, message: string, data?: Json) {
		super(message);
		this.code = code;
		this.data = data;
	}
}

/**
 * The current time in the A2M wire format (spec §3.3).
 *
 * Never an epoch number: a float loses precision in any language whose only
 * number is a double, which is most of them, and JavaScript is the extreme case.
 */
function nowRfc3339(): string {
	return new Date().toISOString().replace(/(\.\d{3})\d*Z$/, "$1Z");
}

/**
 * Split text into indexable terms.
 *
 * Deliberately naive. Spec §5.1 leaves ranking entirely to the implementation,
 * so being bad at it is still conformant — and the Unicode property escape keeps
 * it from being *wrong* about accents and non-Latin scripts, which would be a
 * different failure.
 */
function terms(text: Json): Set<string> {
	const found = String(text).toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
	return new Set(found.filter((word) => word.length > 1));
}

/** Apply a `where` filter to one record (spec §5.2). */
function matches(record: Stored, where: Json): boolean {
	if (!where) return true;

	for (const [key, expected] of Object.entries(where)) {
		const actual = (record as Json)[key] ?? record.metadata?.[key];
		if (Array.isArray(expected)) {
			if (!expected.includes(actual)) return false;
		} else if (actual !== expected) {
			return false;
		}
	}

	return true;
}

/**
 * Project a stored record down to the fields `core` defines.
 *
 * A record must not advertise a field whose capability this server does not
 * declare, which is why tier, salience and owner never appear here.
 */
function pub(record: Stored, score?: number): Json {
	const out: Json = {
		id: record.id,
		content: record.content,
		created_at: record.created_at,
		role: record.role,
		metadata: record.metadata,
	};
	if (record.group) out.group = record.group;
	if (record.key !== undefined) {
		out.key = record.key;
		out.revision = record.revision;
	}
	if (score !== undefined) out.score = score;
	return out;
}


/** The record at a key, if any. Keys are unique within this store's single scope. */
function byKey(key: string): Stored | undefined {
	for (const record of RECORDS.values()) {
		if (record.key === key) return record;
	}
	return undefined;
}


/** Whether a record's key sits at or under a prefix (spec §3.6). */
function underPrefix(record: Stored, prefix: Json): boolean {
	if (prefix === undefined || prefix === null || prefix === "") return true;
	return record.key !== undefined && record.key.startsWith(String(prefix));
}

/** Handle `memory/describe`. Throws -32007 on an incompatible version. */
function describe(params: Json): Json {
	const protocol = params.protocol;
	if (protocol !== undefined && protocol !== null && protocol !== PROTOCOL) {
		throw new A2MError(PROTOCOL_NOT_SUPPORTED, `This server speaks ${PROTOCOL}`, {
			supported: [PROTOCOL],
		});
	}

	return {
		protocol: PROTOCOL,
		name: "a2m-minimal-ts",
		capabilities: [...CAPABILITIES],
		methods: [...METHODS],
		limits: { max_records_per_call: MAX_RECORDS_PER_CALL },
	};
}

/** Handle `memory/remember`. A supplied `id` makes the write idempotent (spec §3.2). */
function remember(params: Json): Json {
	const records = params.records;

	if (!Array.isArray(records) || records.length === 0) {
		throw new A2MError(INVALID_PARAMS, "'records' must be a non-empty array");
	}

	if (records.length > MAX_RECORDS_PER_CALL) {
		throw new A2MError(QUOTA_EXCEEDED, `At most ${MAX_RECORDS_PER_CALL} records per call`);
	}

	const ids: string[] = [];

	for (const entry of records) {
		if (typeof entry !== "object" || entry === null || typeof entry.content !== "string") {
			throw new A2MError(INVALID_PARAMS, "Each record needs a string 'content'");
		}

		for (const [field, capability] of [
			["tier", "tiers"],
			["embedding", "embeddings"],
			["uri", "external"],
		]) {
			if (entry[field] !== undefined && entry[field] !== null) {
				throw new A2MError(
					CAPABILITY_NOT_SUPPORTED,
					`This server does not implement the '${capability}' capability`,
				);
			}
		}

		// spec §3.2 -- writing an id that already exists returns it, and does not
		// create a second record. This is the only retry-safety A2M has.
		const supplied = entry.id;
		if (supplied !== undefined && supplied !== null && RECORDS.has(String(supplied))) {
			ids.push(String(supplied));
			continue;
		}

		// spec §3.6 -- a key addresses a fact, so writing to an occupied key
		// replaces what is there: same id, new content, revision + 1. The stale
		// fact must stop being recallable, not merely be outnumbered.
		if (entry.key !== undefined && entry.key !== null) {
			const held = byKey(String(entry.key));
			if (held !== undefined) {
				held.content = entry.content;
				held.role = entry.role ?? held.role;
				if (entry.metadata !== undefined) held.metadata = entry.metadata;
				held.revision += 1;
				ids.push(held.id);
				continue;
			}
		}

		const id = supplied !== undefined && supplied !== null ? String(supplied) : crypto.randomUUID().replace(/-/g, "");

		RECORDS.set(id, {
			id,
			content: entry.content,
			role: entry.role ?? "user",
			metadata: entry.metadata ?? {},
			group: entry.group ?? undefined,
			key: entry.key !== undefined && entry.key !== null ? String(entry.key) : undefined,
			revision: 0,
			created_at: nowRfc3339(),
			sequence: RECORDS.size,
		});
		ids.push(id);
	}

	return { ids };
}

/** Handle `memory/recall`. An absent query ranks by recency rather than failing (spec §4.3). */
function recall(params: Json): Json {
	for (const [field, capability] of [
		["tier", "tiers"],
		["embedding", "embeddings"],
	]) {
		if (params[field] !== undefined && params[field] !== null) {
			throw new A2MError(
				CAPABILITY_NOT_SUPPORTED,
				`This server does not implement the '${capability}' capability`,
			);
		}
	}

	const limit = params.limit ?? 8;
	const minScore = params.min_score ?? 0.0;

	const candidates = [...RECORDS.values()]
		.filter((record) => matches(record, params.where))
		.filter((record) => underPrefix(record, params.key_prefix));
	const wanted = params.query ? terms(params.query) : new Set<string>();

	let scored: Array<[Stored, number]> = [];

	if (wanted.size === 0) {
		// spec §4.3 -- an absent query is not an error.
		scored = candidates.sort((a, b) => b.sequence - a.sequence).map((record) => [record, 0.0]);
	} else {
		for (const record of candidates) {
			const held = terms(record.content);
			const overlap = [...wanted].filter((word) => held.has(word));
			if (overlap.length > 0) {
				const union = new Set([...wanted, ...held]);
				scored.push([record, overlap.length / union.size]);
			}
		}
		scored.sort((a, b) => b[1] - a[1]);
	}

	scored = scored.filter(([, score]) => score >= minScore);
	if (limit > 0) scored = scored.slice(0, limit);

	return { records: scored.map(([record, score]) => pub(record, score)) };
}

/** Handle `memory/timeline`. Ascending by creation; `limit` takes the newest (spec §4.4). */
function timeline(params: Json): Json {
	if (params.tier !== undefined && params.tier !== null) {
		throw new A2MError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'tiers' capability");
	}

	let ordered = [...RECORDS.values()]
		.filter((record) => matches(record, params.where))
		.filter((record) => underPrefix(record, params.key_prefix))
		.sort((a, b) => a.sequence - b.sequence);

	const limit = params.limit ?? 0;
	if (limit > 0) ordered = ordered.slice(-limit);

	return { records: ordered.map((record) => pub(record)) };
}

/** Handle `memory/forget`. A call with no selector is -32602, never a store-wide delete (spec §4.5). */
function forget(params: Json): Json {
	if (params.tier !== undefined && params.tier !== null) {
		throw new A2MError(CAPABILITY_NOT_SUPPORTED, "This server does not implement the 'tiers' capability");
	}

	const { ids, query, where, key_prefix } = params;

	if (!ids && !query && !where && !key_prefix) {
		throw new A2MError(INVALID_PARAMS, "Pass ids, query, where or key_prefix; refusing to forget everything");
	}

	const doomed = new Set<string>();

	if (Array.isArray(ids)) {
		for (const id of ids) {
			if (RECORDS.has(String(id))) doomed.add(String(id));
		}
	}

	if (query || where) {
		for (const record of recall({ query, where, key_prefix, limit: 0 }).records) {
			doomed.add(record.id);
		}
	}

	if (key_prefix && !query && !where) {
		for (const record of RECORDS.values()) {
			if (underPrefix(record, key_prefix)) doomed.add(record.id);
		}
	}

	for (const id of doomed) RECORDS.delete(id);

	return { forgotten: doomed.size };
}


/** Handle `memory/fetch`. An unused key is `{record: null}`, never an error (spec §4.9). */
function fetch(params: Json): Json {
	const key = params.key;
	if (key === undefined || key === null || key === "") {
		throw new A2MError(INVALID_PARAMS, "'key' is required");
	}

	const held = byKey(String(key));
	return { record: held !== undefined ? pub(held) : null };
}

/**
 * Build a handler that refuses an undeclared capability.
 *
 * Every A2M method is registered even when unimplemented: an unregistered one
 * would answer -32601 METHOD_NOT_FOUND, and a client cannot tell that from a typo
 * in its own call. Spec §2 requires -32003.
 */
function unsupported(capability: string): (params: Json) => Json {
	return () => {
		throw new A2MError(
			CAPABILITY_NOT_SUPPORTED,
			`This server does not implement the '${capability}' capability`,
		);
	};
}

const HANDLERS: Record<string, (params: Json) => Json> = {
	"memory/describe": describe,
	"memory/remember": remember,
	"memory/recall": recall,
	"memory/timeline": timeline,
	"memory/forget": forget,
	// spec §2 -- an undeclared capability answers -32003, never -32601.
	"memory/promote": unsupported("tiers"),
	"memory/consolidate": unsupported("tiers"),
	"memory/reinforce": unsupported("salience"),
	"memory/session/list": unsupported("sessions"),
	"memory/session/close": unsupported("sessions"),
	"memory/fetch": fetch,
	"memory/events": unsupported("events"),
	"memory/events/subscribe": unsupported("events"),
	"memory/events/unsubscribe": unsupported("events"),
};

/** Build a JSON-RPC error response. */
function errorResponse(id: Json, code: number, message: string, data?: Json): Json {
	const body: Json = { code, message };
	if (data !== undefined) body.data = data;
	return { jsonrpc: "2.0", id: id ?? null, error: body };
}

/** Turn one parsed request into one response, or null for a notification. */
function handle(payload: Json): Json | null {
	// A JSON array here is a JSON-RPC batch, which A2M does not use (spec §8).
	if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
		return errorResponse(null, INVALID_REQUEST, "Request must be a JSON object");
	}

	// spec §3.2 -- the id is echoed exactly as it arrived. JavaScript would
	// happily turn 1 into "1"; a client matching responses by identity would then
	// wait forever.
	const isNotification = !("id" in payload);
	const id = payload.id ?? null;

	const fail = (code: number, message: string, data?: Json): Json | null =>
		isNotification ? null : errorResponse(id, code, message, data);

	if (payload.jsonrpc !== "2.0") {
		return fail(INVALID_REQUEST, "Expected jsonrpc '2.0'");
	}

	const handler = HANDLERS[payload.method];
	if (handler === undefined) {
		return fail(METHOD_NOT_FOUND, `Unknown method '${payload.method}'`);
	}

	const params = payload.params ?? {};
	if (typeof params !== "object" || Array.isArray(params)) {
		return fail(INVALID_PARAMS, "'params' must be an object");
	}

	try {
		// spec §2 -- unrecognised parameters are ignored, never rejected. Reading
		// only the fields it knows is how this handler does that.
		const result = handler(params);
		return isNotification ? null : { jsonrpc: "2.0", id, result };
	} catch (exc) {
		if (exc instanceof A2MError) return fail(exc.code, exc.message, exc.data);
		return fail(INTERNAL_ERROR, "Handler failed", String(exc));
	}
}

/**
 * Serve A2M on stdin/stdout, one JSON value per line (spec §8.2).
 *
 * Nothing but JSON-RPC may go to stdout; anything else corrupts the stream. Note
 * that console.log writes there, so it must never be used for logging here —
 * this is the one footgun the Python port does not have.
 */
function main(): void {
	let buffer = "";

	process.stdin.setEncoding("utf8");

	process.stdin.on("data", (chunk: string) => {
		buffer += chunk;

		let newline: number;
		while ((newline = buffer.indexOf("\n")) >= 0) {
			const line = buffer.slice(0, newline).trim();
			buffer = buffer.slice(newline + 1);

			if (!line) continue;

			let response: Json | null;
			try {
				response = handle(JSON.parse(line));
			} catch (exc) {
				response = errorResponse(null, PARSE_ERROR, "Invalid JSON", String(exc));
			}

			if (response !== null) {
				process.stdout.write(JSON.stringify(response) + "\n");
			}
		}
	});

	// spec §8.2 -- closing stdin terminates the server cleanly.
	process.stdin.on("end", () => process.exit(0));
}

main();
