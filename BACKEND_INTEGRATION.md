# Phone agent → Cardinal Intelligence: integration status

Reply to *Phone Voice Agent — Integration Reference*. Covers the four open
questions, plus one endpoint we need from you.

---

## Answers to your four questions

### Vendor / telephony stack

LiveKit Cloud SIP end to end. The number is a LiveKit-provided one
(`PN_PPN_WSvbp62vVPjo`) rather than a classic external SIP trunk, routed by
inbound dispatch rule `SDR_XLtpGxzhkxcS`. The agent itself is a Python worker on
`livekit-agents` 1.6.10 running Gemini Live native audio, so speech recognition
and generation both happen inside the realtime model — there is no separate
transcription vendor to reconcile against.

### How caller ID is populated

LiveKit sets attributes on the inbound SIP participant when the call is
answered. We read `sip.phoneNumber`, which carries the caller's number as
delivered by the PSTN carrier, and fall back to the participant identity
(`sip_<number>`) if the attribute is missing.

Two things worth knowing:

- **Blocked or withheld caller ID arrives empty.** We cannot distinguish that
  from a call with no caller ID at all. Both are treated as unidentified: the
  conversation happens, nothing is submitted.
- **We have not yet confirmed the exact E.164 formatting** the carrier delivers
  through this number. The first live call logs the full attribute dict so we can
  check it against a real participant record. If your stored numbers are
  normalized differently (no `+`, dashes, `1` prefix), tell us and we will
  normalize on our side before sending.

### Is webhook-on-call-end native?

No — we build it, and it is built. It fires from the agent worker's shutdown
callback, in-process, when the room closes at hangup. That means it runs on the
same machine that handled the call and has the full session history in memory.

Consequence worth flagging: if the worker process dies mid-call, no POST
happens. Failed and undeliverable POSTs (5xx, timeouts, connection failures) are
retried three times and then written to a local `undelivered_calls.jsonl` for
manual replay, so a backend outage delays a transcript rather than losing it.

### How partial / dropped calls get reported → `callStatus`

LiveKit models this as an enum, `SIPCallStatus`:

| Value | Meaning |
|---|---|
| `SCS_CALL_INCOMING` | ringing, not yet answered |
| `SCS_PARTICIPANT_JOINED` | answered, participant in the room |
| `SCS_ACTIVE` | call in progress |
| `SCS_DISCONNECTED` | call ended |
| `SCS_ERROR` | call failed |

There is a separate `DisconnectReason` alongside it that distinguishes a clean
hangup from a failure.

**The important thing:** `SCS_DISCONNECTED` is the *normal* end state for a
completed call. Every successful conversation ends that way. If you reject on it
you will reject every good call. We suggest rejecting only on `SCS_ERROR`, or on
a `DisconnectReason` that indicates failure.

**Until that is agreed, we are not sending `callStatus` at all.** Your spec says
an unrecognized value flagged as dropped returns 400, and we would rather omit
the field than have a real conversation thrown away by a guess. Confirm the
accepted set and we will start sending it.

Note that a dropped call is largely a non-issue on our side anyway: we only
submit when the conversation actually contains substance (see below), so a call
that drops during the opening never generates a POST.

---

## What we need from you: a resolve endpoint

### Verified from our side first

Before asking, we checked what already works against
`https://towera.tail24ff87.ts.net`:

- `POST /api/webhooks/phone-agent` with no key → `401`, as documented.
- Same endpoint **with our key** and an unmatchable number → `404` with
  `"No matching participant found for phone '+10000000000' or token 'N/A'.
  Reconcile record manually."` So the key we were given is valid, the endpoint
  is live, and the unmatched path behaves exactly as your doc says.
- `POST /api/webhooks/phone-agent/resolve` → the Next.js HTML 404 page, i.e. the
  route does not exist. That is the one gap.

One thing we noticed in that error string: matching is on **phone or token**.
Your spec calls the fallback field `spokenParticipantId`, and your app has an
`/interview/<token>` route. If students already receive a token through the web
app, that is a far better fallback than a spoken name and may be worth wiring up
later — we have left it out of scope for now, but tell us if the token is
something a student would reasonably have in front of them on a call.

### The ask

Your spec documents one endpoint, the transcript webhook. But the agent has to
know **at the start of the call** whether the caller matches a student, because
that decides how it opens the conversation — whether it greets them by name and
asks for consent to record, or tells them it could not identify them and that
nothing will be saved.

We cannot get that from the transcript webhook without creating a record, so we
need a lookup. Proposed contract, mirroring your existing conventions:

```
POST /api/webhooks/phone-agent/resolve
x-api-key: cardinal-voice-agent-secret-key-2026

{ "phoneNumber": "+15025551234" }
```

| Status | Body | Meaning |
|---|---|---|
| `200` | `{ "participantId": "...", "displayName": "Sarah", "consentOnFile": false }` | matched |
| `404` | — | no match, nothing created (same semantics as your transcript webhook) |
| `401` | — | missing/invalid key |

`consentOnFile` is optional. When true the agent gives a returning student a
one-line reminder instead of the full consent explanation. **Ship it without that
field if it is inconvenient** — we treat a missing value as `false`, which just
means everyone hears the full script.

`pronunciation` is also optional, and worth more than it looks. Our voice model
mispronounces short and uncommon names — on our first live call it read "Gabe"
as "Get". The agent's opening line is *"am I speaking with &lt;name&gt;?"*, so a
mangled name is the first thing a student hears. If you can carry a phonetic
respelling (`"Gabe"` → `"Gayb"`) on the participant record, even sparsely
populated for the names that need it, we will use it. Absent means we say the
name as written.

The path is configurable on our side (`CARDINAL_RESOLVE_PATH`), so name it
whatever fits your routing.

**Nothing is blocked on this.** Until the endpoint exists, every call runs the
unidentified path: the conversation works normally and no transcript is
submitted. Flip it on by pointing us at the URL.

---

## What we send you, and when

### The rule for attaching a record

A call is submitted **only** when all three of these hold:

1. The phone number resolved to a participant via the endpoint above.
2. The caller confirmed on the phone that they are that person.
3. The caller consented to being transcribed.

If any one fails, the conversation still happens and is still useful to the
student — it is simply never sent anywhere.

Point (2) is deliberate and worth being explicit about: **a name spoken aloud
never attaches a record.** If the caller ID resolves to Sarah and the person on
the line says "no, this is her classmate," we discard the match entirely rather
than holding it. There is no code path from a spoken name to a `participantId`.
If you want spoken-name matching, that is `spokenParticipantId` and we have left
it out of scope for now.

We also skip submission when a consented call produced fewer than four
conversational turns after setup finished — a call that hung up right after
consent is noise, not data.

### Payload

Exactly your documented shape:

```json
{
  "phoneNumber": "+15025551234",
  "transcript": "[00:00] agent: Hi, am I speaking with Sarah?\n[00:04] student: Yeah, that's me.\n...",
  "callDurationSec": 847
}
```

- `transcript` is a single string, one line per turn, prefixed with `[mm:ss]`
  offsets from the moment the call was answered. Speakers are labelled `agent`
  and `student`. Interrupted turns are marked `[interrupted]`.
- The setup exchange (identity, consent, time) **is** included — it is part of
  the call and it is the record of consent.
- `callDurationSec` is wall clock from answer to hangup, rounded to a whole
  second.
- `callStatus` omitted, per above.

We expect `202`. We do not retry `4xx` — a `401` or `400` means something is
wrong that a retry will not fix, and it goes straight to the local recovery file
with the response body logged.
