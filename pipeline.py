"""The voice pipeline: STT, endpointing, LLM, TTS -- all of them ours.

Replaces Gemini Live native audio, which was retired on 2026-08-18 for one
reason: **its only endpointing control is a silence timer.** Inference on that
stack was never the problem (measured 1.3s and flat across a call); the problem
was that a fixed `silence_duration_ms` either cuts off a student mid-thought or
adds dead air to every turn, and there is no setting that is good for someone
reasoning out loud -- which this agent's whole stance invites. Taking turn
detection away from Gemini's VAD is not supported in livekit-agents 1.6.10
(`RealtimeModel.session()` keeps `can_disable_turn_detection=False`), so the
voice layer had to move.

A cascade endpoints semantically instead: a model judges whether the student is
actually finished, so an unfinished-sounding pause waits and a finished one
commits in ~0.36s rather than waiting out a timer.

Everything here rides the LiveKit Inference gateway, billed to this LiveKit Cloud
project via LIVEKIT_API_KEY. No new credentials, and GOOGLE_API_KEY is no longer
on the voice path at all.

The constants below are not guesses -- each one cost a run to find. Read the
comment before changing one.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import inference

# Loaded here, not just in agent.py. Every constant below is read at import
# time, and agent.py calls load_dotenv() *after* its import block -- so
# without this, an MBA606_* knob set in .env.local would be silently ignored
# locally while working fine in deployment (where secrets are real env vars).
# That is the kind of difference that costs an afternoon.
load_dotenv(Path(__file__).parent / ".env.local")

# Gateway model strings, resolved by AgentSession into inference.STT/LLM/TTS.
STT_MODEL = os.getenv("MBA606_STT", "deepgram/nova-3")

# TTFT benchmarked on this prompt shape, median over two runs of three calls:
#   openai/gpt-5.4-mini            613ms   <- and the tightest spread, 544-662
#   xai/grok-4-1-fast              750ms
#   google/gemini-3.1-flash-lite   780ms   (high variance, 496-977)
#   google/gemini-3.5-flash        980ms
# Quality did not regress at the top: gpt-5.4-mini mirrors the student back
# before asking, which is what the persona demands.
LLM_MODEL = os.getenv("MBA606_LLM", "openai/gpt-5.4-mini")

# sonic-3.5 leads Artificial Analysis's Controlled Voice Arena (same 8 cloned
# voices across every model, so it scores the model rather than whose voice you
# happen to like). ElevenLabs v3 sounds better in isolation and is the wrong
# trade here: it takes ~50% longer to speak the identical line (8.97s vs 6.00s
# measured on our own copy), which costs more per turn than anything left in the
# endpointing budget. Cartesia is built for streaming agents, not narration.
#
# Note the callers hear 8kHz G.711 anyway, which discards most of what separates
# the top models -- judge any candidate on a phone call, not on studio audio.
TTS_MODEL = os.getenv("MBA606_TTS", "cartesia/sonic-3.5")
TTS_VOICE = os.getenv("MBA606_VOICE", "")

# Cartesia speaking rate: "slow" | "normal" | "fast", or a float. Measured on
# sonic-3.5, same line, median of 3 runs each: slow 1.09x, fast 0.91x. So a real
# and roughly symmetric +/-10% -- but per-utterance variance is nearly as large
# (fast ranged 5.61-6.25s), so do not expect it to be audible on any single
# reply, and do not expect it to rescue a reply that is simply too long. Reply
# length is the lever for that, not rate.
#
# `emotion` is NOT available: Sonic 3 dropped it and the gateway 400s on it.
TTS_SPEED = os.getenv("MBA606_TTS_SPEED", "normal")

# Deepgram finalizes a segment after this much silence. The default is 25ms,
# which is tuned for transcription, not conversation -- it shreds ordinary
# disfluent speech into fragments at every stutter and breath ("Uh, I got about",
# "Or or", "an an ideation kinda").
#
# That matters far beyond tidiness, because the turn detector judges whatever the
# STT last finalized. Asked whether "Uh, I got about" is a finished turn it
# correctly says no (p=0.24, p=0.32 on two of three turns in one run), falls
# below threshold, and waits out the full max_delay. Lowering the threshold
# cannot fix this -- the detector is being handed fragments, not misjudging.
#
# 400ms lets a phrase finish before it is finalized. The added STT delay largely
# overlaps the framework's own endpointing window (which counts from last speech,
# not from the final transcript), so the trade is ~400ms of overlap against
# ~1.15s of avoided waiting.
STT_ENDPOINTING_MS = int(os.getenv("MBA606_STT_ENDPOINTING_MS", "400"))

# Endpointing is a cliff, not a slope: above this probability the turn commits in
# ~0.36s, below it it waits out max_delay. The historic 0.45 came from tuning
# against an earlier detector -- in that run six of seven turns cleared 0.56 and
# committed fast, while one landed at 0.50, fell off the cliff, and cost 2.0s
# (the session's worst e2e, 3708ms vs a 1297ms best) despite being a complete
# utterance that merely trailed off.
#
# Left unset now, because the gateway objects on every call it is overridden:
# "the server provides calibrated defaults and overriding them may be
# suboptimal". The server's calibration moves with the model, so a number tuned
# against an older one is a liability rather than an asset. Set
# MBA606_EOT_THRESHOLD=0.45 to restore it and A/B against the eot column.
EOT_THRESHOLD = os.getenv("MBA606_EOT_THRESHOLD")

# min_delay has to clear the STT's own endpointing window. With Deepgram
# finalizing after STT_ENDPOINTING_MS (400ms), a 0.3s min_delay commits the turn
# before the final transcript can arrive -- the framework says so out loud:
# "transcript arrives after turn has been committed. consider raising `min_delay`
# in the endpointing options to accommodate a slow stt." That race is also what
# splits one spoken thought into several separate user messages, because the tail
# of the sentence lands after the turn is already gone.
#
# So this is not an independent knob: keep it above STT_ENDPOINTING_MS.
MIN_ENDPOINTING_DELAY = float(
    os.getenv("MBA606_MIN_DELAY", str(round(STT_ENDPOINTING_MS / 1000 + 0.1, 2)))
)

# max_delay applies only when the detector believes the student is NOT done.
# 2.0s was the worst single latency in the reference run; 1.5s bounds it without
# cutting off someone mid-thought.
MAX_ENDPOINTING_DELAY = float(os.getenv("MBA606_MAX_DELAY", "1.5"))


def build_stt():
    """Deepgram with conversational endpointing instead of the transcription default.

    MBA606_STT=deepgram/flux-general-en switches to Deepgram's conversational
    model, which does end-of-turn detection inside the STT itself (no separate
    detector, no gateway round trip). Its knobs are different -- eot_threshold,
    not endpointing -- so extra_kwargs is skipped for it. Pair it with
    MBA606_TURN_DETECTION=stt.
    """
    if "flux" in STT_MODEL:
        return inference.STT(model=STT_MODEL)
    return inference.STT(model=STT_MODEL, extra_kwargs={"endpointing": STT_ENDPOINTING_MS})


def build_tts() -> inference.TTS:
    """Cartesia sonic-3.5. Gateway ttfb is ~160-200ms and is not a latency factor.

    Always returns a real TTS object rather than a model string. Passing a string
    lets AgentSession resolve it internally, and then there is no handle to call
    update_options() on -- which is what makes the speaking rate steerable
    mid-call. See MentorAgent.set_speaking_rate.
    """
    kwargs = {"extra_kwargs": {"speed": _coerce_speed(TTS_SPEED)}}
    if TTS_VOICE:
        kwargs["voice"] = TTS_VOICE
    return inference.TTS(model=TTS_MODEL, **kwargs)


def _coerce_speed(value):
    """Accept 'slow'/'normal'/'fast' or a numeric string; reject anything else.

    Cartesia rejects out-of-range floats with a 400, which surfaces mid-call as a
    dead turn rather than a config error, so a bad env value must not reach it.
    """
    if value in ("slow", "normal", "fast"):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return "normal"


def build_turn_detector():
    """Cloud end-of-turn detection. The local model segfaults on the dev machine.

    livekit-local-inference 0.2.6's bundled EOT model crashes the process on that
    hardware (Skylake x86_64 / macOS 12.7.6): a null jump in `_native.so` during
    thread-local-storage init. Reproduced in isolation -- `EOT().predict()` exits
    SIGSEGV while `VAD().predict()` is fine over hundreds of calls. It killed the
    first cascade run the instant the first turn ended, which is when the model
    lazily initializes. The VAD half of that library is fine and stays default.

    The framework picks the local `v1-mini` by default outside hosted mode, so it
    has to be overridden explicitly:

      version="v1"          -> gateway transport; the mini weights never load.
      local_fallback=False  -> and they must never load. Left at the default True,
                               one gateway hiccup degrades to the local model and
                               segfaults the process. False instead lets turns
                               commit on the endpointing delay: degraded
                               turn-taking, but the session survives.

    MBA606_TURN_DETECTION=vad drops semantic detection entirely and endpoints on
    silence alone -- that is the Gemini Live behaviour we just left, so it exists
    only as a comparison baseline.
    """
    mode = os.getenv("MBA606_TURN_DETECTION")
    if mode in ("vad", "stt"):
        # "stt" defers to the STT's own end-of-turn signal -- only meaningful with
        # a conversational model like flux, which emits one.
        return mode
    kwargs = {}
    if EOT_THRESHOLD:
        kwargs["unlikely_threshold"] = float(EOT_THRESHOLD)
    return inference.TurnDetector(version="v1", local_fallback=False, **kwargs)


def turn_handling() -> dict:
    """The turn_handling block for AgentSession."""
    return {
        "turn_detection": build_turn_detector(),
        "endpointing": {
            "mode": "dynamic",
            "min_delay": MIN_ENDPOINTING_DELAY,
            "max_delay": MAX_ENDPOINTING_DELAY,
        },
    }


def describe() -> str:
    """One line for the logs, so a call's config is recoverable from its logs."""
    td = os.getenv("MBA606_TURN_DETECTION") or (
        f"eot v1 (threshold {EOT_THRESHOLD})" if EOT_THRESHOLD else "eot v1 (server-calibrated)"
    )
    return (
        f"stt={STT_MODEL} (endpointing {STT_ENDPOINTING_MS}ms) llm={LLM_MODEL} "
        f"tts={TTS_MODEL}{'/' + TTS_VOICE if TTS_VOICE else ''} (speed {TTS_SPEED}) "
        f"turn_detection={td} "
        f"delay={MIN_ENDPOINTING_DELAY}-{MAX_ENDPOINTING_DELAY}s"
    )
