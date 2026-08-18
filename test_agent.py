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
from intake import CallState, clamp_minutes, extract_caller_id, run_intake
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
# A consented call still needs to contain a conversation. Below this many turns
# after intake finished, there is nothing worth putting on a student's record.
MIN_SUBSTANTIVE_TURNS = 4

# What goes in the transcript field when the student says no. It must not be
# empty: their validator rejects a payload with no transcript text before it ever
# reaches participant matching, so an empty string risks a 400 that looks like a
# malformed request rather than a handled decline.
DECLINED_TRANSCRIPT = "Student did not consent to recording. No transcript retained."

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
    ever gets to speak in its own voice: runs consent and the time question as
    AgentTasks, then starts the clock. There is no caller lookup to wait on --
    the backend resolves the phone number after the call, not before it.
    """

    def __init__(self, state: CallState) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self._state = state
        self.timekeeper: TimeKeeper | None = None

    async def on_enter(self) -> None:
        if SNAPPY:
            # Pure latency harness. Consent scripts would drown the signal.
            self.session.generate_reply()
            return

        state = self._state
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

    @function_tool
    async def set_speaking_rate(self, rate: str) -> str:
        """Change how fast you speak, when the student asks you to.

        Call this if they say you are talking too fast or too slow, ask you to
        slow down, or ask you to hurry. Do not call it for anything else -- it
        changes your voice, not the pace of the conversation.

        Args:
            rate: One of "slow", "normal", or "fast".
        """
        rate = (rate or "").strip().lower()
        if rate not in ("slow", "normal", "fast"):
            return "Rate must be slow, normal, or fast."

        tts = self.session.tts
        if tts is None or not hasattr(tts, "update_options"):
            return "Speaking rate cannot be changed on this pipeline."

        # Merges into extra_kwargs and is sent with each synthesis request, so it
        # takes effect from the next reply rather than mid-sentence.
        tts.update_options(extra_kwargs={"speed": rate})
        logger.info("speaking rate set to %s", rate)
        return f"Speaking rate is now {rate}."


# No agent_name: automatic dispatch, so the worker joins every room in the
# project, including the call_* rooms the SIP rule creates. Setting a name would
# switch to explicit dispatch, which requires the dispatch rule to name this agent.
@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    client = CardinalClient()

    # Caller ID comes off the SIP participant. It is not checked against anything
    # now -- it travels with the transcript at the end and the backend matches it
    # there, because there is no side-effect-free lookup to call mid-call.
    try:
        participant = await ctx.wait_for_participant()
    except Exception:
        logger.exception("no participant joined -- treating the call as anonymous")
        participant = None

    phone_number, sip_call_id = extract_caller_id(participant)
    state = CallState(phone_number=phone_number, sip_call_id=sip_call_id)

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
        # MetricsReport is a TypedDict -- a plain dict at runtime. The framework
        # builds it as `assistant_metrics: llm.MetricsReport = {}` and assigns by
        # key, so it must be read with .get(). Reading it with getattr() returns
        # None for every field and logs a full row of "n/a" without erroring,
        # which is exactly how the first cascade call came back blind.
        m = getattr(ev.item, "metrics", None) or {}
        if not m:
            return
        turn_no += 1

        def ms(key):
            value = m.get(key)
            return f"{value * 1000:.0f}ms" if isinstance(value, (int, float)) else "n/a"

        logger.info(
            "turn %d: e2e %s | eot %s | stt %s | llm %s | tts %s",
            turn_no,
            ms("e2e_latency"),
            ms("end_of_turn_delay"),
            ms("transcription_delay"),
            ms("llm_node_ttft"),
            ms("tts_node_ttfb"),
        )

    agent = MentorAgent(state=state)

    async def submit_call() -> None:
        """Post the finished call. One request, at the end -- the only one there is.

        The backend is an end-of-call ingestion webhook, not a lifecycle API, so
        this is where the caller finally gets matched to a participant record. A
        404 here means the number was not on file; that is a real outcome, not a
        failure, and the transcript stays with us for reconciliation.
        """
        if agent.timekeeper is not None:
            agent.timekeeper.stop()

        try:
            duration = round(time.time() - state.answered_at)

            if not state.submittable:
                logger.info(
                    "call over (%ss) -- no caller ID, nothing to post against",
                    duration,
                )
                return

            if not state.consent:
                # Still reported. The student declined to have their words kept,
                # not to have the call exist -- duration is what the pilot needs
                # and it carries none of what they said.
                logger.info(
                    "call over (%ss) -- student declined recording, posting duration only",
                    duration,
                )
                await client.submit_transcript(
                    phone_number=state.phone_number,
                    transcript=DECLINED_TRANSCRIPT,
                    call_duration_sec=duration,
                    call_status=None,
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

            # A call id in the header line. Their endpoint has no idempotency key,
            # so if a retry ever double-posts this is the only thing that lets a
            # human spot the duplicate.
            header = f"[call {state.sip_call_id or 'unknown'}]"

            await client.submit_transcript(
                phone_number=state.phone_number,
                transcript=f"{header}\n{text}",
                call_duration_sec=duration,
                # callStatus is deliberately omitted. The backend rejects values it
                # reads as dropped/cancelled with a 400, and the accepted set is not
                # finalized -- sending a guess risks throwing away a good call.
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
