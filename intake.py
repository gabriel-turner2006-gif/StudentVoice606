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
#   - Ask, then wait. The first call had the model ask "am I speaking with Gabe?"
#     and call the tool in the same breath, before the caller could answer (fixed
#     in efe0a32). A realtime model treats "ask X, then call Y" as one turn unless
#     told plainly to stop.
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
    """Ask the two setup questions, in order, mutating state as answers land.

    Must be awaited from the running Agent's on_enter -- the AgentTasks inside
    resolve against whichever activity is current. Always returns: every branch
    either gets an answer or takes a default, because a failed setup should still
    leave the caller with a working conversation.
    """
    state.consent = await _run_step(ConsentTask(), default=False, label="consent")

    budget_min = await _run_step(
        TimeBudgetTask(),
        default=DEFAULT_BUDGET_MIN,
        label="time_budget",
    )
    state.time_budget_s = budget_min * 60.0
    state.intake_complete = True
    state.intake_finished_at = time.time()

    logger.info(
        "intake complete: consent=%s, budget=%.0f min",
        state.consent,
        budget_min,
    )
