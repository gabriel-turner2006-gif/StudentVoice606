import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types
from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext
from livekit.agents.voice.room_io import AudioInputOptions, RoomOptions
from livekit.plugins import google, noise_cancellation

BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env.local")

INSTRUCTIONS = (BASE_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()

# Diagnostic only. The real prompt asks the agent to leave silence and go slowly, and
# a native-audio model obeys that -- deliberate pacing sounds exactly like lag. Run a
# call with MBA606_SNAPPY=1 to hear the same stack with the pacing removed. If the
# pauses vanish, they were pedagogy, not latency.
SNAPPY_INSTRUCTIONS = (
    "You are a test harness. Reply the instant the user stops speaking, in one short "
    "sentence. Never pause for effect. Never leave silence."
)

if os.getenv("MBA606_SNAPPY") == "1":
    INSTRUCTIONS = SNAPPY_INSTRUCTIONS

# How long the caller has to go quiet before Gemini treats the turn as over.
# Lower = snappier replies; higher = more room for a student to pause mid-thought,
# which the system prompt explicitly asks for. Tune this by ear.
END_OF_TURN_SILENCE_MS = 200

# Compress the audio context once it passes the trigger, down to the target. Starting
# point only -- raise the target if the agent starts losing the thread of a long
# session, lower it if ttft is still climbing turn over turn.
COMPRESSION_TRIGGER_TOKENS = 10000
COMPRESSION_TARGET_TOKENS = 5000

logger = logging.getLogger("mba606.latency")

# `lk agent console` / `lk agent dev` discover this module-level variable.
server = AgentServer()


def select_noise_cancellation(params):
    """Phone callers get the telephony-tuned model, everyone else the standard one."""
    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return noise_cancellation.BVCTelephony()
    return noise_cancellation.BVC()


# No agent_name: automatic dispatch, so the worker joins every room in the
# project, including the call_* rooms the SIP rule creates. Setting a name would
# switch to explicit dispatch, which requires the dispatch rule to name this agent.
@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            voice="Charon",
            # Gemini's server-side VAD owns turn-taking for realtime models, so this
            # is where response latency actually lives -- there is no TTS to stream.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    silence_duration_ms=END_OF_TURN_SILENCE_MS,
                    prefix_padding_ms=100,
                ),
            ),
            # Native-audio thinking runs before the model emits a single audio frame,
            # and the plugin drops the thought parts, so it shows up as pure silence.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            # Audio history is re-prefilled every turn, so ttft climbs as the session
            # runs. Compressing the window keeps per-turn latency roughly flat instead
            # of growing without bound over a long conversation.
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=COMPRESSION_TRIGGER_TOKENS,
                sliding_window=types.SlidingWindow(
                    target_tokens=COMPRESSION_TARGET_TOKENS,
                ),
            ),
        ),
    )

    # Per-turn latency breakdown. `end_of_turn_delay` is the endpointing cost (the
    # silence_duration_ms knob); `llm_node_ttft` is the model's own think-and-speak
    # time. Watching them separately shows which one is actually growing.
    @session.on("conversation_item_added")
    def _on_item(ev):
        m = getattr(ev.item, "metrics", None)
        if not m:
            return
        # *_at entries are absolute wall-clock stamps, not durations -- skip them.
        parts = [
            f"{k}={v * 1000:.0f}ms"
            for k, v in m.items()
            if isinstance(v, (int, float)) and not k.endswith("_at") and v >= 0
        ]
        if parts:
            logger.info("%s %s", getattr(ev.item, "role", "?"), " ".join(parts))

    await session.start(
        room=ctx.room,
        agent=Agent(instructions=INSTRUCTIONS),
        room_options=RoomOptions(
            audio_input=AudioInputOptions(
                noise_cancellation=select_noise_cancellation,
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the caller briefly and ask how much time they have."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
