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

from livekit.agents import inference

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

TTS_MODEL = os.getenv("MBA606_TTS", "cartesia/sonic-3")
TTS_VOICE = os.getenv("MBA606_VOICE", "")

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
# ~0.36s, below it the turn waits out max_delay. In the reference run six of seven
# turns cleared 0.56 and committed fast; one landed at 0.50, fell off the cliff,
# and cost 2.0s -- the session's worst end-to-end (3708ms vs a 1297ms best). That
# utterance was in fact complete, it just trailed off ("...like, pretty
# consistent"). Genuinely mid-utterance predictions in the same run clustered at
# 0.0002-0.23, far below either threshold, so 0.45 rescues the borderline case
# without becoming impatient with someone who really is still thinking.
EOT_THRESHOLD = float(os.getenv("MBA606_EOT_THRESHOLD", "0.45"))

# max_delay applies only when the detector believes the student is NOT done.
# 2.0s was the worst single latency in the reference run; 1.5s bounds it without
# cutting off someone mid-thought.
MIN_ENDPOINTING_DELAY = float(os.getenv("MBA606_MIN_DELAY", "0.3"))
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


def build_tts():
    """Cartesia sonic-3. Gateway ttfb is ~160-200ms and is not a latency factor."""
    if TTS_VOICE:
        return inference.TTS(model=TTS_MODEL, voice=TTS_VOICE)
    return TTS_MODEL


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
    return inference.TurnDetector(
        version="v1",
        local_fallback=False,
        unlikely_threshold=EOT_THRESHOLD,
    )


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
    td = os.getenv("MBA606_TURN_DETECTION") or f"eot v1 (threshold {EOT_THRESHOLD})"
    return (
        f"stt={STT_MODEL} (endpointing {STT_ENDPOINTING_MS}ms) llm={LLM_MODEL} "
        f"tts={TTS_MODEL}{'/' + TTS_VOICE if TTS_VOICE else ''} turn_detection={td} "
        f"delay={MIN_ENDPOINTING_DELAY}-{MAX_ENDPOINTING_DELAY}s"
    )
