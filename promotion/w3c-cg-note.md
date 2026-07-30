# W3C AI Agent Memory Interoperability CG — introduction

Internal draft. To be sent by Marco to `public-ai-agent-memory-interop@w3.org`
after joining the group at
<https://www.w3.org/community/ai-agent-memory-interop/>.

**Posture, and it matters more than the wording:** this is a contribution to
their work, not an announcement of ours. The group chartered on 2026-07-16 and
has no artifacts yet. Arriving with a finished, implemented, conformance-tested
protocol reads as either a gift or a land grab depending entirely on how it is
framed. Frame it as a gift: their scope is *custody and portability* of memory,
ours is *runtime access* to it, and those are complementary layers rather than
competing specifications.

Do not ask them to adopt A2M. Offer the artifact and the crosswalk, and let the
work speak.

---

**Subject:** A2M — a runtime memory protocol, and a possible crosswalk to the CG's scope

---

Hello,

I have just joined the group, and I would like to offer something that may be
useful to the charter work, along with the reasoning behind it so you can judge
whether it is relevant.

For the past months I have been developing **A2M (Agent-to-Memory Protocol)**, a
JSON-RPC 2.0 protocol that gives an agent access to a memory store it does not
own. It is at version 0.1, MIT licensed, and has six independent implementations
passing one unmodified conformance suite.

**Where I think it sits relative to the charter.** Reading the group's scope —
memory cell shape, identity binding, encryption envelopes, audit anchors,
cryptographic erasure, and crosswalks to NIST AI RMF, ISO/IEC 42001 and the EU
AI Act — the concerns are the *custody* of memory: what a memory cell is, who
can prove it is theirs, how it moves between vendors, and how it is provably
destroyed.

A2M addresses a different and I think adjacent question: **how an agent reads
and writes memory at runtime**, while it is working. It says nothing about
encryption, signatures or erasure guarantees, and it deliberately does not
address portability between vendors. The two layers meet at the record: a
portable, signed memory cell has to be *readable* by some interface, and a
runtime interface has to be able to carry whatever a portable cell contains.

Concretely, three places where a crosswalk seems worth having:

1. **Record shape.** A2M's record is deliberately minimal: an opaque id, indexed
   text, an RFC 3339 timestamp, arbitrary round-tripped metadata, and optional
   fields gated behind declared capabilities. If the group's memory cell carries
   a canonical metadata block, A2M's `metadata` is exactly the place it survives
   unchanged — servers must round-trip it byte-for-byte in structure or fail the
   write.

2. **Erasure.** A2M has `memory/forget` with a hard rule that a call with no
   selector is refused rather than treated as "delete everything". It makes no
   cryptographic claim. If the group specifies erasure as DEK destruction plus a
   tombstone, that is a stronger guarantee layered *underneath* a `forget`, and
   the two do not conflict.

3. **Identity and scoping.** A2M states normatively that its `owner` field is
   **data partitioning and not a security boundary** — a client asserts its own
   scope, and a server on a network transport must derive the scope from the
   authenticated principal and ignore what the client claimed. I would rather
   over-communicate this than have anyone mistake it for access control. If the
   group binds identity cryptographically, that is precisely the mechanism A2M's
   §6 says must exist and declines to specify.

**What I am offering, in order of usefulness:**

- The specification and its decision log, as prior art to argue with. The
  decision log records why each non-obvious choice was made, which I have found
  more useful than the specification itself when someone disagrees.
- A conformance methodology that may transfer. The suite speaks only the wire
  format and never imports the implementation under test, which is what let a
  Python suite validate a TypeScript server. If the group produces a testable
  specification, that separation is worth stealing.
- A concrete finding: writing a second implementation from the document alone
  found a real bug in the first. I would recommend the group require two
  independent implementations before calling anything stable, for that reason
  rather than on principle.

**What I am not asking for.** I am not proposing A2M as the group's
specification, and I am not asking for endorsement. If the runtime layer is out
of scope for the charter, that is a completely reasonable answer and the
crosswalk is still worth writing down.

Specification: <https://a2m-protocol.org>
Source and decision log: <https://github.com/dibenedetto/a2m-protocol>

I would find it genuinely useful to know whether the group considers runtime
access in scope, or whether it is deliberately left to MCP-adjacent work. That
answer would change what I do next.

Best regards,
Marco Di Benedetto

---

## Notes before sending

- **Join first, post second.** Posting to a CG list before joining reads badly.
- **Check the actual list address** on the group page; the one above is the
  conventional pattern, not a verified address.
- **Read the charter in full first**, and adjust the three crosswalk points to
  what it actually says. The summary above is from the group's public scope
  description, and a mischaracterised charter is the fastest way to lose the
  room.
- **Do not send the launch post to this list.** Different audience, different
  register. If A2M is announced elsewhere the same week, one sentence here
  linking it is enough.
- Expect a slow reply, or none. The value is being in the room and on the
  record early, not the response.
