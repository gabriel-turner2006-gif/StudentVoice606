# MBA 606 Phone Agent

An inbound phone line students call to think out loud. The agent is a thinking
partner for MBA 606 Leadership Lab — it asks about their situation, pushes back
when the reasoning has a gap, and lands the conversation on something they can
carry. Transcripts go to UofL Cardinal Intelligence, but only with consent.

Runs as a LiveKit Cloud agent on a LiveKit-provided phone number.

---

## Quick start

```bash
uv sync                                    # install
lk agent console agent.py                  # talk to it locally, mic + speakers
lk agent dev agent.py                      # local worker on the real phone number
lk agent deploy                            # ship to LiveKit Cloud
lk agent logs                              # tail production
```

`agent.py` is the entrypoint. `lk` discovers the module-level `server =
AgentServer()` inside it.

> **Do not run `lk agent dev` while the deployed agent is up.** Dispatch is
> automatic — the worker registers with no `agent_name` and joins every room in
> the project — so both would answer the same call and talk over each other.

Local runs read `.env.local`. The deployed worker reads LiveKit Cloud secrets
(`lk agent secrets`); `.env.*` is excluded from the image, so the two are
configured separately and must be kept in sync by hand.

---

## How a call works

```
  inbound SIP call
        │
        ▼
  caller ID off the participant  ──────────────  never checked mid-call
        │                                        (see "Identity", below)
        ▼
  ┌─ intake ─────────────────────────────────┐
  │  1. consent to transcription?            │  ConsentTask
  │       └─ declined → offer to end         │  DeclineExitTask
  │  2. how much time do you have?           │  TimeBudgetTask
  └──────────────────────────────────────────┘
        │
        ▼
  mentor conversation, clock running          TimeKeeper re-asserts the budget
        │                                     as the call runs
        ▼
  hangup → one POST to Cardinal               consent decides what it contains
```

### Intake

The first thirty seconds are scripted, and deliberately know nothing about MBA
606. Each question is an `AgentTask`: awaiting one swaps the running agent's
instructions for a short single-purpose prompt, collects a typed answer through
one tool call, then restores the mentor agent.

**Consent is asked first**, before any other setup, so a caller who declines has
not first been made to answer questions. If they decline, the agent offers to
end the call — *offers*. An unclear answer keeps the call going; hanging up on
ambiguity would make ending the call the penalty for declining.

Every intake prompt carries the same two hard-won rules (`_HOW` in `intake.py`):
ask then **wait** — never call the tool in the same turn as the question — and
stay on the question if the caller asks about something else. Both cost a live
call to discover.

Each step has a 75-second ceiling. A caller who talks past a question gets a
default and the call moves on rather than looping.

### Identity — resolved after the call, not during

There is no caller lookup. The Cardinal backend exposes no side-effect-free way
to resolve a phone number: the only route to its matching logic is posting a
complete call, which stores a record and queues extraction, so using it as a
lookup would manufacture junk records.

So the agent never learns who it is talking to, never greets by name, and never
confirms an identity. It sends the number the carrier gave it, and they match it
on their side. **Misattribution is therefore impossible from here** — no spoken
name ever enters the payload.

Caller ID is read defensively from the SIP participant (`sip.phoneNumber`,
falling back to the `sip_` identity prefix). A withheld number is
indistinguishable from no number, and both mean the call cannot be posted.

### The clock

The student's stated budget is worthless as a one-time statement — the model
forgets it and talks past the end. `TimeKeeper` appends a short line to the chat
context as the call runs, on a granularity ladder that stays quiet while the call
is young and coarsens with distance, so an ordinary stretch of conversation does
not generate a push per turn. Roughly 14 updates across a 20-minute call.

Phases: `midpoint` (past halfway) → `closing` (under 20%) → `over`. Wrap-up is
soft throughout. **The agent never hangs up** on a running conversation; the
student ends the call.

`revise_time_budget` lets the model move the whole schedule mid-call when a
student says their time changed.

---

## Modules

| File | Responsibility |
|---|---|
| `agent.py` | Entrypoint. `MentorAgent`, session wiring, the end-of-call POST. |
| `pipeline.py` | The voice stack — STT, endpointing, LLM, TTS — and every tuned constant. |
| `intake.py` | `CallState` and the consent / decline-exit / time `AgentTask`s. |
| `timekeeper.py` | Keeps the time budget alive in the model's context. |
| `backend.py` | `CardinalClient` — the single end-of-call POST, retries, dead-letter queue. |
| `transcript.py` | Renders session history into a timestamped transcript string. |
| `system_prompt.txt` | The mentor persona. Edited far more often than the code. |
| `tools/dev_stub_backend.py` | Local stand-in for Cardinal. Enforces the real contract. |
| `docs/syllabus.md` | Course material. **Not wired in yet** — intended as future agent context. |

---

## The voice pipeline

A cascade on the LiveKit Inference gateway, billed through `LIVEKIT_API_KEY` —
no separate vendor credentials.

```
deepgram/nova-3  →  turn detector (EOT v1)  →  openai/gpt-5.4-mini  →  cartesia/sonic-3.5
```

**Why not Gemini Live native audio?** It was retired 2026-08-18 for one reason:
its only endpointing control is a silence timer. Inference on that stack was
never the problem — measured 1.3s and flat across a call. The problem is that a
fixed `silence_duration_ms` either cuts a student off mid-thought or adds dead
air to every turn, and there is no value that is good for someone reasoning out
loud, which this agent's whole stance invites. Taking turn detection away from
Gemini's VAD isn't supported in `livekit-agents` 1.6.10
(`can_disable_turn_detection=False`), so the voice layer had to move.

A cascade endpoints *semantically*: a model judges whether the student is
actually finished, so an unfinished-sounding pause waits and a finished one
commits in ~0.36s.

**Every constant in `pipeline.py` cost a run to find. Read its comment before
changing it.** Two that matter most:

- **Deepgram's `endpointing` defaults to 25ms**, tuned for transcription, not
  conversation. It shreds disfluent speech into fragments, and the turn detector
  judges whatever was last finalized — so it correctly says "not finished" and
  waits out `max_delay`. Set to 400ms. Lowering the EOT threshold cannot fix
  this; the detector is being handed fragments, not misjudging.
- **`min_delay` must stay above the STT endpointing window**, or the turn commits
  before the final transcript lands and one spoken thought splits across several
  user messages.

Turn detection is forced to the **cloud** model (`version="v1"`,
`local_fallback=False`). The bundled local EOT model in
`livekit-local-inference` 0.2.6 segfaults on the dev machine — a null jump during
thread-local-storage init, reproducible in isolation. `local_fallback=False`
means a gateway hiccup degrades to endpointing-on-delay rather than crashing the
process.

---

## The Cardinal integration

One POST, at the end of the call. Not a lifecycle API. Full detail in
`../integration.md`.

```
POST https://towera.tail24ff87.ts.net/api/webhooks/phone-agent
x-api-key: <CARDINAL_API_KEY>
```

### The consent contract

Their schema is strict and **fails closed**, and `backend.submit_call()` mirrors
every rule so a mistake surfaces in our logs instead of as an opaque `400`:

| `recordingConsent` | `transcript` | Result |
|---|---|---|
| `true` | required, non-empty | `202 accepted`, extraction queued |
| `false` | **must be absent** | `202 recorded_declined`, nothing extracted |
| missing / not a boolean | — | `400` — silence is never consent |

A declined call is **still posted** — phone number, duration, and
`consentCapturedAt`, with no transcript field at all. A placeholder string is
rejected by design, so a client bug cannot leak content past a decline. Declines
are stored even when the number matches no participant, so the audit trail
survives an unknown caller.

A call that produces no webhook is indistinguishable from a crash, which is why
short calls are posted too.

### Responses

| Code | Meaning | What we do |
|---|---|---|
| `202` | Accepted | Log and close |
| `404` | No participant matches the number | Do not retry — a retry returns the same. Dead-letter it. |
| `400` / `401` | Malformed payload or bad key | Terminal. Dead-letter and log the body. |
| `5xx` / timeout | Transport | Retry 3× with backoff, then dead-letter |

`callStatus` is deliberately **not sent**. Their accepted values are undefined,
and LiveKit's normal end state for a *completed* call is `SCS_DISCONNECTED` —
sending a guess risks a good call being rejected as dropped.

Transcripts carry a `[call <sip id>]` header line. The endpoint has no
idempotency key, so that is the only thing that would let a human spot a
duplicate if a retry ever double-posted.

---

## Configuration

**Required** (LiveKit Cloud secrets in production, `.env.local` locally):

| Variable | Notes |
|---|---|
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | Injected automatically in deployment. Also bills the Inference gateway. |
| `CARDINAL_API_BASE` | **Leave unset to disable posting entirely** — calls run normally and the transcript is logged instead. |
| `CARDINAL_API_KEY` | |

**Pipeline tuning** — all optional, all defaulted in `pipeline.py`:

`MBA606_STT` · `MBA606_LLM` · `MBA606_TTS` · `MBA606_VOICE` · `MBA606_TTS_SPEED`
· `MBA606_STT_ENDPOINTING_MS` · `MBA606_MIN_DELAY` · `MBA606_MAX_DELAY` ·
`MBA606_EOT_THRESHOLD` · `MBA606_TURN_DETECTION`

**Testing:**

| Variable | Effect |
|---|---|
| `MBA606_FAKE_CALLER` | Pretend the caller ID is this number (console has no SIP participant) |
| `MBA606_SNAPPY=1` | Swap the mentor prompt for a one-liner and skip intake. Isolates pipeline latency from prompt length. |

Every call logs its full pipeline config on startup (`pipeline.describe()`), so a
call's configuration is recoverable from its logs.

---

## Testing locally

```bash
# terminal 1 — stand-in for Cardinal, enforces the real consent contract
uv run tools/dev_stub_backend.py

# terminal 2
CARDINAL_API_BASE=http://127.0.0.1:8899 \
MBA606_FAKE_CALLER=+15025551234 \
  lk agent console agent.py
```

The stub prints whatever the agent posted, so you can read the exact payload a
call produced. Numbers not on its roster return `404`, which exercises the
unmatched-caller path.

`lk agent console` is the supported console. The legacy `python agent.py console`
produces no audio.

---

## Known gaps

- **The dead-letter queue is ephemeral in production.** `logs/undelivered_calls.jsonl`
  writes to the container filesystem, which vanishes when the agent scales to
  zero. Locally it's a real safety net; deployed, failed posts survive only in
  `lk agent logs`.
- **No idempotency.** If a retry ever double-posts, the backend cannot dedupe.
  The header line makes it human-detectable, nothing more.
- **`spokenParticipantId` is unimplemented here** and untested on their side.
  It's the only fallback when caller ID is withheld.
- **Governing regime unconfirmed.** Currently defaulting to Kentucky
  one-party-consent as a floor. If a research protocol turns out to govern, the
  likely delta is retention and whether a declined call may be logged at all —
  not whether we ask.
