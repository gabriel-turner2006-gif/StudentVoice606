import asyncio
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, function_tool
from livekit.agents.voice.room_io import AudioInputOptions, RoomOptions
from livekit.plugins import noise_cancellation

import pipeline
import transcript
from backend import CardinalClient
from intake import CallState, clamp_minutes, extract_caller_id, resolve_caller, run_intake
from timekeeper import TimeKeeper

BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env.local")

INSTRUCTIONS = (BASE_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()

# Diagnostic only. MBA606_SNAPPY=1 swaps the mentor prompt out entirely, which
# isolates pipeline latency from anything the prompt is doing to reply length.
SNAPPY_INSTRUCTIONS = (
    "You are a test harness. Reply in one short sentence. Never pause for effect."
)

SNAPPY = os.getenv("MBA606_SNAPPY") == "1"

if SNAPPY:
    INSTRUCTIONS = SNAPPY_INSTRUCTIONS

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
    three intake questions as AgentTasks, then starts the clock.
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

    logger.info("pipeline: %s", pipeline.describe())

    session = AgentSession(
        userdata=state,
        stt=pipeline.build_stt(),
        llm=pipeline.LLM_MODEL,
        tts=pipeline.build_tts(),
        # VAD stays the framework default -- it is the half of the local inference
        # library that works everywhere. Turn detection is forced to the cloud
        # model; see pipeline.build_turn_detector(). Preemptive generation stays
        # default-enabled: worth ~1.2s when it fires.
        turn_handling=pipeline.turn_handling(),
    )

    # Per-turn latency, straight from the framework.
    #
    # This used to be a hand-assembled timeline, because the Gemini realtime path
    # published none of these metrics -- and it measured the wrong things twice
    # over: its `ttft` was really the duration of the student's utterance, and its
    # zero point drifted to the student's first micro-pause on exactly the long
    # turns that mattered. On a cascade the framework publishes the real
    # breakdown, so none of that reconstruction is needed or wanted.
    #
    #   eot    end_of_turn_delay  -- the endpointing decision. THE number to tune;
    #                               it is what the whole move to a cascade bought.
    #   stt    transcription_delay
    #   llm    llm_node_ttft
    #   tts    tts_node_ttfb      -- ~160-200ms on the gateway; not a factor.
    #   e2e    e2e_latency        -- what the caller actually experiences.
    turn_no = 0

    @session.on("conversation_item_added")
    def _on_item(ev):
        nonlocal turn_no
        if getattr(ev.item, "role", None) != "assistant":
            return
        m = getattr(ev.item, "metrics", None)
        if m is None:
            return
        turn_no += 1

        def ms(value):
            return f"{value * 1000:.0f}ms" if isinstance(value, (int, float)) else "n/a"

        logger.info(
            "turn %d: e2e %s | eot %s | stt %s | llm %s | tts %s",
            turn_no,
            ms(getattr(m, "e2e_latency", None)),
            ms(getattr(m, "end_of_turn_delay", None)),
            ms(getattr(m, "transcription_delay", None)),
            ms(getattr(m, "llm_node_ttft", None)),
            ms(getattr(m, "tts_node_ttfb", None)),
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
