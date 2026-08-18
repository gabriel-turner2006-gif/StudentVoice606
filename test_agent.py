import asyncio
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types
from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext
from livekit.agents.voice.room_io import AudioInputOptions, RoomOptions
from livekit.plugins import google, noise_cancellation

import transcript
from backend import CardinalClient
from intake import CallState, extract_caller_id, resolve_caller, run_intake
from timekeeper import TimeKeeper

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

SNAPPY = os.getenv("MBA606_SNAPPY") == "1"

if SNAPPY:
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

# AgentSession loads inference.VAD(model="silero") whenever no vad= is passed, and
# that VAD declares end-of-speech only after this much trailing silence. It does not
# do turn-taking here -- Gemini owns that -- but it is the one clock in this process
# that is independent of Google, which makes it the only usable zero point. Back it
# out of the event stamp to recover roughly when the caller actually stopped talking.
VAD_MIN_SILENCE_S = 0.25

# The resolve request is fired the moment the caller joins, so by the time the
# session is up it is normally already done. This is the ceiling on how long the
# caller waits in silence for it before we give up and treat the call as anonymous.
RESOLVE_WAIT_S = 1.5

# A consented call still needs to contain a conversation. Below this many turns
# after intake finished, there is nothing worth putting on a student's record.
MIN_SUBSTANTIVE_TURNS = 4

logger = logging.getLogger("mba606.latency")

# `lk agent console` / `lk agent dev` discover this module-level variable.
server = AgentServer()


def select_noise_cancellation(params):
    """Phone callers get the telephony-tuned model, everyone else the standard one."""
    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return noise_cancellation.BVCTelephony()
    return noise_cancellation.BVC()


class MentorAgent(Agent):
    """The MBA 606 thinking partner, fronted by the setup phase.

    on_enter does the whole of the first thirty seconds before the mentor prompt
    ever gets to speak in its own voice: waits out the caller lookup, runs the
    three intake questions as AgentTasks, then starts the clock. AgentTasks have
    to be awaited from here rather than from a function tool -- Gemini's realtime
    model reports manual_function_calls=False, which the SDK warns is undefined
    behavior for a task awaited mid-tool-call.
    """

    def __init__(self, state: CallState, resolve_task: asyncio.Task) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self._state = state
        self._resolve_task = resolve_task
        self.timekeeper: TimeKeeper | None = None

    async def on_enter(self) -> None:
        if SNAPPY:
            # Pure latency harness. Consent scripts would drown the signal.
            self._resolve_task.cancel()
            self.session.generate_reply()
            return

        state = self._state

        try:
            state.student = await asyncio.wait_for(self._resolve_task, timeout=RESOLVE_WAIT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "caller lookup still pending after %.1fs -- proceeding anonymously",
                RESOLVE_WAIT_S,
            )
        except Exception:
            logger.exception("caller lookup failed -- proceeding anonymously")

        state.bypassed = state.student is not None and state.student.participant_id.startswith(
            "ADMIN_BYPASS_"
        )

        await run_intake(state)

        keeper = TimeKeeper(self.session, self, INSTRUCTIONS)
        self.timekeeper = keeper
        if state.time_budget_s:
            keeper.start(state.time_budget_s)

        # Hand the mentor prompt back with the budget folded in. Until the first
        # time check fires this is the only thing telling the model how long it has.
        await self.update_instructions(INSTRUCTIONS + keeper.opening_note())

        self.session.generate_reply(
            instructions=(
                "Setup is done. In one short sentence, invite the student to say "
                "what is on their mind. Do not recap the setup questions and do not "
                "restate how much time they have."
            )
        )


# No agent_name: automatic dispatch, so the worker joins every room in the
# project, including the call_* rooms the SIP rule creates. Setting a name would
# switch to explicit dispatch, which requires the dispatch rule to name this agent.
@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    client = CardinalClient()

    # Caller ID lives on the SIP participant, so the lookup cannot start until it
    # joins. Firing it as a task here lets the round trip overlap session startup
    # instead of adding itself to the silence before the greeting.
    try:
        participant = await ctx.wait_for_participant()
    except Exception:
        logger.exception("no participant joined -- treating the call as anonymous")
        participant = None

    phone_number, sip_call_id = extract_caller_id(participant)
    state = CallState(phone_number=phone_number, sip_call_id=sip_call_id)
    resolve_task = asyncio.create_task(resolve_caller(client, phone_number), name="resolve_caller")

    session = AgentSession(
        userdata=state,
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

    # Per-turn latency timeline, assembled from session events.
    #
    # It has to be assembled by hand because the realtime path publishes none of the
    # pipeline latency metrics. `end_of_turn_delay` and `llm_node_ttft` are written
    # only by the STT -> LLM -> TTS cascade; the realtime generation task attaches
    # just started_speaking_at / stopped_speaking_at / playback_latency, and the user
    # message gets no metrics at all because the Gemini plugin never sets
    # turn_started_at. So the obvious `conversation_item_added` breakdown that used to
    # live here printed nothing but playback_latency, which is why the delay has been
    # hard to pin down.
    #
    # Everything below is stamped relative to end of speech. The point is to see
    # whether the transcript landing and the audio starting are two separate waits or
    # one: Gemini emits both only after it commits the turn, so if they arrive close
    # together the transcript is a symptom of the wait, not its cause.
    turn: list[tuple[str, float]] = []
    speech_end: float | None = None

    def mark(label: str) -> None:
        if speech_end is not None:
            turn.append((label, (time.time() - speech_end) * 1000))

    @session.on("user_state_changed")
    def _on_user_state(ev):
        nonlocal speech_end
        if ev.new_state == "listening":
            turn.clear()
            speech_end = ev.created_at - VAD_MIN_SILENCE_S

    @session.on("user_input_transcribed")
    def _on_transcribed(ev):
        # Input transcription streams while the caller is still talking, so the one
        # that matters is whichever chunk lands last before the reply starts -- that
        # is the moment the words finish rendering in the console.
        mark("transcript" if not ev.is_final else "transcript FINAL")

    @session.on("agent_state_changed")
    def _on_agent_state(ev):
        nonlocal speech_end
        if ev.new_state != "speaking" or speech_end is None:
            return
        mark("AGENT AUDIO")
        logger.info(
            "turn from end of speech: %s",
            " | ".join(f"{label} +{ms:.0f}ms" for label, ms in turn),
        )
        speech_end = None

    @session.on("metrics_collected")
    def _on_metrics(ev):
        # ttft here is measured from the first server message of the turn (normally
        # the input transcription) to the first audio byte -- so it is Gemini's
        # generate time only. The commit-and-transcribe window that precedes it is
        # invisible to this number; that gap is what the timeline above exposes.
        ttft = getattr(ev.metrics, "ttft", None)
        if isinstance(ttft, (int, float)) and ttft >= 0:
            logger.info("gemini ttft (first server msg -> first audio) %.0fms", ttft * 1000)

    agent = MentorAgent(state=state, resolve_task=resolve_task)

    async def submit_call() -> None:
        """Hand the finished call to the backend, if the call earned a record."""
        if agent.timekeeper is not None:
            agent.timekeeper.stop()

        try:
            duration = round(time.time() - state.answered_at)

            # phone_number is implied by recordable -- a student is only ever set
            # from a resolved number -- but state it so the invariant is visible.
            if not state.recordable or not state.phone_number:
                logger.info(
                    "call over (%ss) -- nothing submitted: %s",
                    duration,
                    state.why_not_recordable(),
                )
                return

            text, _ = transcript.render(session.history, state.answered_at)
            substantive = transcript.count_after(session.history, state.intake_finished_at)

            if substantive < MIN_SUBSTANTIVE_TURNS:
                logger.info(
                    "call over (%ss) -- only %d turns after setup, not submitting",
                    duration,
                    substantive,
                )
                return

            if state.bypassed:
                logger.warning(
                    "submitting a call that used ADMIN BYPASS -- the participant id is "
                    "synthetic, so this can only match on phone number, if at all"
                )

            await client.submit_transcript(
                phone_number=state.phone_number,
                transcript=text,
                call_duration_sec=duration,
                # callStatus is deliberately omitted. The backend rejects values it
                # reads as dropped/cancelled with a 400, and the accepted set is not
                # finalized -- sending a guess risks throwing away a good call. See
                # BACKEND_INTEGRATION.md for the values we are proposing.
                call_status=None,
            )
        finally:
            await client.aclose()

    ctx.add_shutdown_callback(submit_call)

    await session.start(
        room=ctx.room,
        agent=agent,
        room_options=RoomOptions(
            audio_input=AudioInputOptions(
                noise_cancellation=select_noise_cancellation,
            ),
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
