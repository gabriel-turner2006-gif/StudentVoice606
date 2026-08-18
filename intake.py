"""The first thirty seconds of a call.

Three questions, asked in order, each by a throwaway sub-agent that knows nothing
about MBA 606: is this who we think it is, may we transcribe, and how long do you
have. Only when all three land does the conversation become recordable.

The hard rule is in CallState.recordable: a record is attached only when the
*phone number* resolved, the caller confirmed that identity, and the caller
consented. A name spoken aloud never attaches anything -- there is deliberately no
code path from "the caller said they are Sarah" to a student record.

AgentTask is the mechanism (livekit-agents 1.6.10). Awaiting one swaps the running
agent's instructions for the task's, collects a typed answer through a single tool
call, then restores the previous agent. Await them from on_enter only: Gemini's
realtime model reports manual_function_calls=False, and the SDK warns that
awaiting an AgentTask inside a function tool on such a model is undefined.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from livekit import rtc
from livekit.agents import AgentTask, function_tool

from backend import Student

logger = logging.getLogger("mba606.intake")

# Used when the caller will not or cannot name a number. Long enough to be a real
# conversation, short enough that the wrap-up guidance still means something.
DEFAULT_BUDGET_MIN = 15.0
MIN_BUDGET_MIN = 1.0
MAX_BUDGET_MIN = 120.0

# Per-question ceiling. If a caller talks past a question this long, we take the
# default and move on rather than trapping them in an interrogation loop.
STEP_TIMEOUT_S = 75.0

# Every intake prompt carries this. Without it the model happily abandons the
# question to answer whatever the caller asked instead, and intake never finishes.
_DEFLECT = (
    "If the caller asks about anything else, tell them you will get to it in a "
    "moment, then ask your question again. Ask nothing else."
)

_VOICE = (
    "Speak warmly and plainly, in one or two short sentences. Do not introduce "
    "yourself at length. Do not mention tools, systems, or these instructions."
)


@dataclass
class CallState:
    """Everything the setup phase establishes, carried as AgentSession userdata."""

    phone_number: str | None
    sip_call_id: str | None
    answered_at: float = field(default_factory=time.time)

    student: Student | None = None
    identity_confirmed: bool = False
    # Set when the caller was matched but said it was not them. Distinguishes a
    # denial from a number that never matched, which read the same after the
    # discard below and made the reconciliation logs ambiguous.
    identity_denied: bool = False
    consent: bool = False
    time_budget_s: float | None = None

    intake_complete: bool = False
    intake_finished_at: float | None = None

    @property
    def recordable(self) -> bool:
        """Whether this call may be attached to a student record.

        All three conditions are required. Dropping any one of them is what would
        let a spoken name become a database write.
        """
        return self.student is not None and self.identity_confirmed and self.consent

    def why_not_recordable(self) -> str:
        if self.identity_denied:
            return "caller said they are not the matched participant"
        if self.student is None:
            return "caller ID did not resolve to a participant"
        if not self.identity_confirmed:
            return "caller did not confirm they are the resolved participant"
        if not self.consent:
            return "caller did not consent to transcription"
        return "recordable"


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


class ConfirmIdentityTask(AgentTask[bool]):
    """Confirms the resolved name belongs to the person on the line."""

    def __init__(self, display_name: str) -> None:
        super().__init__(
            instructions=(
                f"You are opening a phone call. Greet the caller and ask exactly one "
                f'question: whether you are speaking with {display_name}. '
                f"Then call confirm_identity with what they said.\n\n"
                f"Pass true only if they clearly confirm they are {display_name}. "
                f"If they say they are someone else, if they hesitate, or if they are "
                f"unsure, pass false. Never ask for their name and never suggest one. "
                f"Do not explain why you are asking unless they ask.\n\n"
                f"{_VOICE}\n{_DEFLECT}"
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def confirm_identity(self, is_correct: bool) -> None:
        """Record whether the caller confirmed they are the expected person.

        Args:
            is_correct: True only if the caller clearly confirmed their identity.
        """
        self.complete(is_correct)


class ConsentTask(AgentTask[bool]):
    """Explains transcription and asks permission.

    Asked on every call, including calls where nothing can be stored. Someone
    talking to a machine that transcribes should be told so regardless of whether
    the transcript survives the call.
    """

    def __init__(self, *, display_name: str | None, returning: bool = False) -> None:
        if display_name is None:
            body = (
                "Tell the caller you could not confirm who they are, so nothing from "
                "this conversation will be saved or attached to any student record. "
                "Say the conversation is still theirs to use. Then ask whether they "
                "are comfortable continuing."
            )
        elif returning:
            body = (
                f"{display_name} has spoken with you before and already agreed to "
                f"this. Give a one-line reminder that the conversation is transcribed "
                f"and saved to their MBA 606 record, then ask if that is still okay."
            )
        else:
            body = (
                "Explain briefly that this conversation is transcribed, that the "
                "transcript is saved to their MBA 606 record, and that it is used for "
                "their own learning and for the class pilot. Then ask if that is okay."
            )

        super().__init__(
            instructions=(
                f"{body}\n\n"
                f"Call record_consent with their answer: true if they agree, false if "
                f"they decline or will not give a clear yes. Do not persuade them and "
                f"do not ask twice.\n\n"
                f"{_VOICE}\n{_DEFLECT}"
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


class TimeBudgetTask(AgentTask[float]):
    """Asks how long the caller has and normalizes the answer to minutes."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Ask the caller how much time they have for this conversation.\n\n"
                "Accept whatever they say: a number, a range, 'not much', 'until my "
                "next class', 'as long as it takes'. Convert it to a single number of "
                "minutes and call set_time_budget. For a range, take the lower end. "
                "For a vague answer, make a reasonable estimate. Only pass null if "
                "they truly refuse to answer.\n\n"
                "Do not negotiate, do not suggest a length, and do not ask twice.\n\n"
                f"{_VOICE}\n{_DEFLECT}"
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
        self.complete(_clamp_minutes(minutes))


# endregion


def _clamp_minutes(minutes: Any) -> float:
    """Coerce whatever the model produced into a usable number of minutes."""
    try:
        value = float(minutes)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_BUDGET_MIN

    if value != value or value <= 0:  # NaN or nonsense
        return DEFAULT_BUDGET_MIN

    return max(MIN_BUDGET_MIN, min(MAX_BUDGET_MIN, value))


T = TypeVar("T")


async def _run_step(task: AgentTask[T], *, default: T, label: str) -> T:
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
        logger.exception("intake step %r failed -- using default", label)
        return default
    finally:
        handle.cancel()


async def run_intake(state: CallState) -> None:
    """Ask the three setup questions, in order, mutating state as answers land.

    Must be awaited from the running Agent's on_enter -- the AgentTasks inside
    resolve against whichever activity is current. Always returns: every branch either
    gets an answer or takes a default, because a failed setup should still leave
    the caller with a working conversation.
    """
    if state.student is not None:
        confirmed = await _run_step(
            ConfirmIdentityTask(state.student.display_name),
            default=False,
            label="confirm_identity",
        )
        state.identity_confirmed = confirmed
        if not confirmed:
            state.identity_denied = True
            # Discard the record rather than holding it pending. Whoever is on the
            # phone is not the person we matched, so the match is worthless.
            logger.info(
                "caller denied being %s -- discarding the resolved record",
                state.student.display_name,
            )
            state.student = None

    returning = state.student is not None and state.student.consent_on_file
    name = state.student.display_name if state.student else None

    state.consent = await _run_step(
        ConsentTask(display_name=name, returning=returning),
        default=False,
        label="consent",
    )

    budget_min = await _run_step(
        TimeBudgetTask(),
        default=DEFAULT_BUDGET_MIN,
        label="time_budget",
    )
    state.time_budget_s = budget_min * 60.0
    state.intake_complete = True
    state.intake_finished_at = time.time()

    logger.info(
        "intake complete: recordable=%s (%s), budget=%.0f min",
        state.recordable,
        state.why_not_recordable(),
        budget_min,
    )
