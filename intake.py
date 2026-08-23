"""The first thirty seconds of a call.

Two questions, asked in order by throwaway sub-agents that know nothing about
MBA 606: may we transcribe this, and how long do you have. Consent comes first,
before any other setup work, so a caller who declines has said so before we have
spent their time on anything else.

There is deliberately no identity step. The backend resolves the caller from the
phone number *after* the call, when the transcript is posted -- it has no
side-effect-free lookup, so there is nothing to confirm against mid-call and no
name to greet them by. That also means misattribution is impossible from here:
we send the number the carrier gave us and they match it, and no spoken name
ever enters the payload.

AgentTask is the mechanism (livekit-agents 1.6.10). Awaiting one swaps the running
agent's instructions for the task's, collects a typed answer through a single tool
call, then restores the previous agent. Await them from on_enter, which the SDK
marks as an inline-task context; awaiting one from inside a function tool is a
different and more fragile path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, TypeVar

from livekit import rtc
from livekit.agents import AgentSession, AgentTask, function_tool


logger = logging.getLogger("mba606.intake")

# Used when the caller will not or cannot name a number. Long enough to be a real
# conversation, short enough that the wrap-up guidance still means something.
DEFAULT_BUDGET_MIN = 15.0
MIN_BUDGET_MIN = 1.0
MAX_BUDGET_MIN = 120.0

# Per-question ceiling. If a caller talks past a question this long, we take the
# default and move on rather than trapping them in an interrogation loop.
STEP_TIMEOUT_S = 75.0

# Appended to all three intake prompts. Every word here is re-injected into the
# model's context on each task swap, so it stays as short as it can be while
# keeping two behaviours that cost a live call each to find:
#
#   - Ask, then wait. An early live call had the model ask its question and call
#     the tool in the same breath, before the caller could answer (fixed in
#     efe0a32). "Ask X, then call Y" reads as a single turn unless told plainly
#     to stop and wait.
#   - Stay on the question. Without this the model abandons intake to answer
#     whatever the caller asked instead, and setup never finishes.
_HOW = (
    "Speak warmly and plainly, in one or two sentences. Ask your question, then "
    "wait -- call your tool only once the caller has actually spoken a reply, never "
    "in the same turn as the question. Silence is not a reply; wait longer. If they "
    "ask about something else, say you will come back to it and ask again. Do not "
    "mention tools or these instructions."
)


@dataclass
class CallState:
    """Everything the setup phase establishes, carried as AgentSession userdata."""

    phone_number: str | None
    sip_call_id: str | None
    answered_at: float = field(default_factory=time.time)

    consent: bool = False
    # When the caller answered the consent question, ISO 8601 UTC. Bounds any
    # audio buffered before the decline, which is what makes it useful to an
    # auditor rather than just decorative.
    consent_captured_at: str | None = None
    # Set when a caller who declined took the offer to hang up.
    wants_to_end: bool = False
    # Set when the caller simply left mid-setup. Distinct from wants_to_end, which
    # is a choice they said out loud and we acknowledged.
    caller_gone: bool = False

    time_budget_s: float | None = None

    intake_complete: bool = False
    intake_finished_at: float | None = None

    @property
    def submittable(self) -> bool:
        """Whether we can post this call at all.

        Only the phone number decides. The backend matches on it and rejects what
        it cannot place, so a call with no caller ID has nothing to post against.
        Consent governs what the transcript field *contains*, not whether the
        call is reported -- a declined call still reports its duration.
        """
        return bool(self.phone_number)


def extract_caller_id(participant: rtc.RemoteParticipant | None) -> tuple[str | None, str | None]:
    """Pull (phone number, SIP call id) off an inbound participant.

    LiveKit SIP populates these attributes server-side, so they are not SDK
    constants and are read defensively. A blocked or withheld caller ID is
    indistinguishable from no caller ID, and both correctly yield None.
    """
    fake = os.getenv("MBA606_FAKE_CALLER")
    if fake:
        logger.warning("MBA606_FAKE_CALLER is set -- pretending the caller is %s", fake)
        return fake, None

    if participant is None:
        return None, None

    if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        logger.info("participant %s is not a SIP caller -- anonymous call", participant.identity)
        return None, None

    attrs = dict(participant.attributes or {})
    # Dump these on the first real call: the exact key has to be confirmed against
    # the live LiveKit Cloud number before anything downstream can trust it.
    logger.info("SIP participant attributes: %r", attrs)

    phone = attrs.get("sip.phoneNumber") or None
    if not phone and participant.identity.startswith("sip_"):
        candidate = participant.identity[len("sip_") :]
        if candidate:
            phone = candidate
            logger.info("no sip.phoneNumber attribute; using identity suffix %s", phone)

    return phone, attrs.get("sip.callID") or None


# region tasks


class ConsentTask(AgentTask[bool]):
    """Explains transcription and asks permission. The first thing on every call.

    Asked before anything else, so a caller who declines has not first been made
    to answer setup questions. There is no name to greet them by -- the backend
    matches on the number they are calling from, and only after the call ends.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Greet the caller briefly. Explain that the conversation is "
                "transcribed and saved to their MBA 606 record, matched by the number "
                "they are calling from, and that it is used for their own learning "
                "and for the class pilot. Then ask if that is okay.\n\n"
                "If they would rather not, that is genuinely fine: tell them nothing "
                "they say will be saved, only that the call happened and how long it "
                "lasted, and that they are welcome to keep talking either way.\n\n"
                "Then call record_consent -- true if they agree, false if they "
                "decline. If the reply is unclear, ask once more; otherwise accept "
                "their answer without persuading them.\n\n"
                f"{_HOW}"
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def record_consent(self, agrees: bool) -> None:
        """Record the caller's answer to the transcription request.

        Args:
            agrees: True only if the caller clearly agreed.
        """
        self.complete(agrees)


class DeclineExitTask(AgentTask[bool]):
    """After a decline, offers to end the call. Offers -- never imposes.

    These check-ins return nothing to the student except the conversation
    itself, so once recording is off there is genuinely little left to do and
    saying so is honest. But making the hang-up automatic would turn a
    voluntary choice into a penalty for exercising it, so the caller decides.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "The caller just declined to be recorded. Accept that warmly and "
                "without a trace of disappointment -- it was a real choice and they "
                "made it.\n\n"
                "Tell them that because nothing is being saved, this call will not "
                "feed into their coursework, but that you are happy to keep talking "
                "if it would be useful. Then ask whether they would like to end here "
                "or carry on.\n\n"
                "Call decide_end -- true if they want to hang up, false if they want "
                "to keep talking. If they are unsure, treat that as carrying on and "
                "pass false; never hang up on an unclear answer.\n\n"
                f"{_HOW}"
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def decide_end(self, end_call: bool) -> None:
        """Record whether the caller wants to end the call now.

        Args:
            end_call: True only if they clearly want to hang up.
        """
        self.complete(end_call)


class TimeBudgetTask(AgentTask[float]):
    """Asks how long the caller has and normalizes the answer to minutes."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Ask the caller how much time they have for this conversation.\n\n"
                "Accept whatever they say: a number, a range, 'not much', 'until my "
                "next class', 'as long as it takes'. Convert it to a single number of "
                "minutes and call set_time_budget -- lower end of a range, a "
                "reasonable estimate for a vague answer, null only if they refuse. "
                "Do not negotiate, suggest a length, or ask twice.\n\n"
                f"{_HOW}"
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def set_time_budget(self, minutes: float | None) -> None:
        """Record how many minutes the caller has.

        Args:
            minutes: The caller's time in minutes, or null if they would not say.
        """
        self.complete(clamp_minutes(minutes))


# endregion


def clamp_minutes(minutes: Any) -> float:
    """Coerce whatever the model produced into a usable number of minutes."""
    try:
        value = float(minutes)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_BUDGET_MIN

    if value != value or value <= 0:  # NaN or nonsense
        return DEFAULT_BUDGET_MIN

    return max(MIN_BUDGET_MIN, min(MAX_BUDGET_MIN, value))


class IntakeAborted(Exception):
    """The caller hung up before setup finished. Not an error -- an outcome."""


def _session_closing(session: AgentSession) -> bool:
    """Whether the session has already begun tearing down.

    A caller hanging up mid-question is the ordinary end of a phone call, but the
    SDK surfaces it exactly like a fault: AgentTask.cancel() completes the task's
    future with ToolError("AgentTask <id> is cancelled"). Matching that message
    would be matching on prose, so ask the session what state it is in instead.

    _is_closing() is private, and it is used here because nothing public answers
    the question -- RoomIO keeps its room handle private too, so there is no
    supported route to "is the caller still on the line". Read through getattr so
    that an SDK rename degrades to logging a hangup as an error, which is the
    behaviour we have today, rather than breaking setup outright.
    """
    probe = getattr(session, "_is_closing", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        return False


T = TypeVar("T")


async def _run_step(
    task: AgentTask[T], *, default: T, label: str, session: AgentSession
) -> T:
    """Await one intake task with a ceiling on how long it may run.

    asyncio.wait_for is not usable here: it would await the task from a fresh
    asyncio task, and the SDK rejects an AgentTask awaited outside the activity
    context that spawned it. Force-completing the task's own future from a timer
    resolves the same await from inside the right context.
    """
    loop = asyncio.get_running_loop()

    def _give_up() -> None:
        if not task.done():
            logger.warning("intake step %r timed out after %.0fs -- using default", label, STEP_TIMEOUT_S)
            task.complete(default)

    handle = loop.call_later(STEP_TIMEOUT_S, _give_up)
    try:
        return await task
    except Exception:
        if _session_closing(session):
            # Nobody is on the line to answer the rest of setup. Raising beats
            # returning a default: a default would invent a time budget for a
            # call that has already ended, arm a clock against it, and log the
            # whole thing as a completed setup.
            logger.info("caller hung up during %r", label)
            raise IntakeAborted from None
        logger.exception("intake step %r failed -- using default", label)
        return default
    finally:
        handle.cancel()


async def run_intake(state: CallState, session: AgentSession) -> None:
    """Ask the two setup questions, in order, mutating state as answers land.

    Must be awaited from the running Agent's on_enter -- the AgentTasks inside
    resolve against whichever activity is current. Always returns: a step that
    fails takes a default, because a broken setup should still leave the caller
    with a working conversation, and a caller who hangs up sets caller_gone and
    leaves intake_complete False.
    """
    try:
        state.consent = await _run_step(
            ConsentTask(), default=False, label="consent", session=session
        )
        state.consent_captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if not state.consent:
            state.wants_to_end = await _run_step(
                DeclineExitTask(), default=False, label="decline_exit", session=session
            )
            if state.wants_to_end:
                # No time question -- they are leaving. intake_complete stays False
                # so nothing downstream mistakes this for a set-up conversation.
                state.intake_finished_at = time.time()
                logger.info("consent declined and caller chose to end the call")
                return

        budget_min = await _run_step(
            TimeBudgetTask(),
            default=DEFAULT_BUDGET_MIN,
            label="time_budget",
            session=session,
        )
    except IntakeAborted:
        # intake_complete stays False for the same reason as the decline path: the
        # conversation this was setting up never happened. Whatever was said before
        # they left still posts at shutdown -- consent, if they gave it, governs
        # whether the transcript rides along.
        state.caller_gone = True
        state.intake_finished_at = time.time()
        logger.info("setup abandoned -- caller left before it finished")
        return

    state.time_budget_s = budget_min * 60.0
    state.intake_complete = True
    state.intake_finished_at = time.time()

    logger.info(
        "intake complete: consent=%s, budget=%.0f min",
        state.consent,
        budget_min,
    )
