import asyncio
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types
from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, function_tool, inference
from livekit.agents.voice.room_io import AudioInputOptions, RoomOptions
from livekit.plugins import google, noise_cancellation

import transcript
from backend import CardinalClient
from intake import CallState, clamp_minutes, extract_caller_id, resolve_caller, run_intake
from timekeeper import TimeKeeper

BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env.local")

INSTRUCTIONS = (BASE_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()

# Diagnostic only. Run a call with MBA606_SNAPPY=1 to hear the same stack with the
# mentor prompt swapped out entirely -- it isolates transport and model latency from
# anything the prompt is doing. The old prompt asked the agent to leave silence and
# not fill it, which a native-audio model renders as literal dead air; that wording
# is gone, and this harness is how you confirm it stays gone.
SNAPPY_INSTRUCTIONS = (
    "You are a test harness. Reply the instant the user stops speaking, in one short "
    "sentence. Never pause for effect. Never leave silence."
)

SNAPPY = os.getenv("MBA606_SNAPPY") == "1"

if SNAPPY:
    INSTRUCTIONS = SNAPPY_INSTRUCTIONS

# How long the caller has to go quiet before Gemini treats the turn as over.
#
# The obvious tuning direction is wrong here. Set this too LOW and perceived gaps get
# WORSE: a student thinking mid-sentence gets committed early, Gemini starts
# generating, the student resumes, that interrupts and forces a regenerate -- so the
# wait after they actually finish is long. 200ms was doing exactly that. Tune against
# the `commit` number in the turn log below, not by feel.
END_OF_TURN_SILENCE_MS = int(os.getenv("MBA606_EOT_MS", "700"))

# 100ms clipped the start of speech, which produces bad transcripts and therefore
# bad replies. Cheap to be generous here.
PREFIX_PADDING_MS = int(os.getenv("MBA606_PREFIX_MS", "300"))

# Compress the audio context once it passes the trigger, down to the target.
#
# This was 10k/5k, which on a native-audio model (~25 tokens/sec each way) fires
# within about five minutes and then repeatedly, sliding the actual conversation out
# of the window while the agent is still trying to converge -- the agent forgets the
# first half of the call. The window is 128k, so these numbers still bound an
# hour-long call. Raising them raises steady-state prefill; watch `in_tok` below to
# see what that trade actually costs.
COMPRESSION_TRIGGER_TOKENS = int(os.getenv("MBA606_COMPRESS_TRIGGER", "32000"))
COMPRESSION_TARGET_TOKENS = int(os.getenv("MBA606_COMPRESS_TARGET", "16000"))

# Charon is the deep, informational voice and reads as flat for a mentor. Pick by ear
# on a real call -- Aoede, Leda, Sulafat and Zephyr are the warmer ones worth trying.
VOICE = os.getenv("MBA606_VOICE", "Sulafat")

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
            # One short line, not update_instructions(INSTRUCTIONS + note). That call
            # would re-send the entire mentor prompt into the conversation -- see the
            # timekeeper module docstring for why that is the thing to avoid.
            await keeper.inject(keeper.opening_note())

        self.session.generate_reply(
            instructions=(
                "Setup is done. In one short sentence, invite the student to say "
                "what is on their mind. Do not recap the setup questions and do not "
                "restate how much time they have."
            )
        )

    @function_tool
    async def revise_time_budget(self, minutes: float) -> str:
        """Update how long the student has, when they say their available time changed.

        Call this whenever the student signals a new amount of time -- "I have to jump
        in five", "my meeting got cancelled, I have longer now", "let's make this
        quick". It moves the whole schedule, including when to start wrapping up.

        Args:
            minutes: How many minutes the student now has from this moment on.
        """
        budget_s = clamp_minutes(minutes) * 60.0
        self._state.time_budget_s = budget_s

        if self.timekeeper is None:
            return "Noted."

        # start() is what arms the ticker; revise() only moves an already-running
        # clock. An intake that produced no budget leaves the keeper unstarted.
        if self.timekeeper.remaining_s is None:
            self.timekeeper.start(budget_s)
        else:
            self.timekeeper.revise(budget_s)

        return f"Time budget is now {budget_s / 60.0:.0f} minutes from now."


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

    # An explicit VAD, purely to get an honest clock.
    #
    # AgentSession loads a default VAD when none is passed, but agent_activity.py
    # UNWIRES it whenever a realtime model owns turn detection -- so user_state
    # transitions came from Gemini's own server event, and the zero point for every
    # measurement was Gemini's commit decision rather than the caller's speech. That
    # made the commit wait structurally invisible. Passing a VAD explicitly sets
    # using_default_vad False, keeps it wired, and hands us a clock Google does not
    # own. Gemini still owns turn-taking; this only observes.
    #
    # Note this must stay a VAD. The bundled end-of-turn model segfaults this
    # machine (see the hardware note in the cascade work) -- never let a local turn
    # detector load here.
    vad = inference.VAD(model="silero")

    session = AgentSession(
        userdata=state,
        vad=vad,
        llm=google.realtime.RealtimeModel(
            voice=VOICE,
            # Google's own control for flat, robotic delivery. The plugin switches to
            # the v1alpha endpoint automatically when this is set.
            enable_affective_dialog=True,
            # Gemini's server-side VAD owns turn-taking for realtime models, so this
            # is where response latency actually lives -- there is no TTS to stream.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    silence_duration_ms=END_OF_TURN_SILENCE_MS,
                    prefix_padding_ms=PREFIX_PADDING_MS,
                ),
            ),
            # Native-audio thinking runs before the model emits a single audio frame,
            # and the plugin drops the thought parts, so it shows up as pure silence.
            # Worth revisiting: zero reasoning budget is consistent with an agent that
            # asks questions but never reaches a conclusion. Price a small budget
            # against the turn log before deciding.
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
    # pipeline latency metrics: end_of_turn_delay and llm_node_ttft are written only
    # by the STT -> LLM -> TTS cascade, and the Gemini plugin never sets
    # turn_started_at, so the user message carries no metrics at all.
    #
    # The decomposition that matters is two numbers, both measured from when the
    # caller actually stopped talking:
    #
    #   commit -- how long Gemini took to decide the turn was over. Endpointing.
    #             Tune END_OF_TURN_SILENCE_MS against this.
    #   audio  -- when the caller first hears something. Everything after commit is
    #             Gemini generating.
    #
    # Watch in_tok across a long call: if it climbs steadily rather than sitting flat
    # between compressions, something is still injecting into the context, and that
    # is the first thing to fix before touching any latency knob.
    #
    # The zero point is the wired VAD above. UserStateChangedEvent.created_at is
    # already back-dated to true end of speech by the SDK (it subtracts the VAD's
    # silence and inference time), so it is used raw -- an earlier version subtracted
    # a hardcoded 250ms on top of that, and against a Gemini timestamp that was never
    # a VAD delay in the first place.
    speech_end: float | None = None
    commit_ms: float | None = None
    turn_no = 0

    @session.on("user_state_changed")
    def _on_user_state(ev):
        nonlocal speech_end, commit_ms
        if ev.new_state == "listening":
            speech_end = ev.created_at
            commit_ms = None

    @session.on("user_input_transcribed")
    def _on_transcribed(ev):
        # Gemini emits the final transcript only once it has committed the turn, so
        # this lands at the moment the wait for endpointing ends.
        nonlocal commit_ms
        if ev.is_final and speech_end is not None and commit_ms is None:
            commit_ms = (time.time() - speech_end) * 1000

    @session.on("agent_state_changed")
    def _on_agent_state(ev):
        nonlocal speech_end, turn_no
        if ev.new_state != "speaking" or speech_end is None:
            return
        audio_ms = (time.time() - speech_end) * 1000
        turn_no += 1
        commit = f"{commit_ms:.0f}ms" if commit_ms is not None else "n/a"
        logger.info(
            "turn %d: commit +%s | audio +%.0fms (generate %s)",
            turn_no,
            commit,
            audio_ms,
            f"{audio_ms - commit_ms:.0f}ms" if commit_ms is not None else "n/a",
        )
        speech_end = None

    @session.on("metrics_collected")
    def _on_metrics(ev):
        # ttft here is measured from the first server message of the turn to the first
        # audio byte, so it is Gemini's generate time only -- the commit window that
        # precedes it is invisible to this number. input_tokens is the drift read-out.
        ttft = getattr(ev.metrics, "ttft", None)
        in_tok = getattr(ev.metrics, "input_tokens", None)
        if isinstance(ttft, (int, float)) and ttft >= 0:
            logger.info(
                "turn %d: gemini ttft %.0fms | in_tok %s",
                turn_no,
                ttft * 1000,
                in_tok if in_tok is not None else "?",
            )

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
