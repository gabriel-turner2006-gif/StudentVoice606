"""Keeps the student's stated time budget alive in the model's context.

The budget is worthless as a one-time statement: it has to be re-asserted as the
call runs, or the model forgets it and talks past the end. Each update is
appended to the chat context as a short line (see inject), which on the cascade
is rebuilt into the prompt on the next turn.

Updates are rationed rather than sent every turn. Nothing is said while the call
is young -- the opening note already carries the budget -- and the phrasing
coarsens with distance, so an unremarkable stretch of conversation does not
generate a push per turn. See _render for the granularity ladder.

Wrap-up is soft throughout. The last phase tells the model to land the
conversation; it never hangs up. The student ends the call.
"""

from __future__ import annotations

import asyncio
import logging
import time

from livekit.agents import Agent, AgentSession

logger = logging.getLogger("mba606.timekeeper")

# Marks a line as addressed to the model rather than spoken by anyone. transcript.py
# filters on this prefix, so it has to lead the string.
INJECTION_PREFIX = "[time check]"

# Wall-clock sweep, so a long silence still advances the phase even though no
# agent turn has completed to trigger a push.
TICK_INTERVAL_S = 20.0

# Below this fraction remaining, the clock starts being mentioned. Above it the
# budget stated at the start is enough and a push would be pure context cost.
QUIET_UNTIL_FRACTION = 0.5

_PHASE_GUIDANCE = {
    "midpoint": (
        "Past the halfway point. Start narrowing toward what matters most to them, "
        "and offer your read of what you have heard rather than only asking."
    ),
    "closing": (
        "Time is nearly complete. Start compressing. Ask the student what they want "
        "to remember, and draw out any commitment or open question."
    ),
    "over": (
        "The time the student asked for has run out. Close out now: confirm their "
        "takeaway, then let them go. Do not open a new thread. Leave hanging up to them."
    ),
}


class TimeKeeper:
    """Tracks the budget and re-injects it as the call runs.

    Inert until start() is called with a budget, so an intake that never got an
    answer simply means no time pressure rather than a crash.
    """

    def __init__(self, session: AgentSession, agent: Agent, base_instructions: str) -> None:
        self._session = session
        self._agent = agent
        # Kept only so the caller's constructor signature does not change; time
        # state no longer travels with the instructions.
        self._base = base_instructions

        self._budget_s: float | None = None
        self._started_at: float | None = None
        self._last_phase: str | None = None
        self._ticker: asyncio.Task[None] | None = None
        self._pushing: asyncio.Task[None] | None = None
        self._stopped = False

    # region lifecycle

    def start(self, budget_s: float) -> None:
        """Begin the clock. Call once intake has produced a budget."""
        self._budget_s = budget_s
        self._started_at = time.time()

        self._session.on("agent_state_changed", self._on_agent_state)
        self._ticker = asyncio.create_task(self._tick_loop(), name="timekeeper_tick")
        logger.info("timekeeper started with a %.0f minute budget", budget_s / 60.0)

    def revise(self, budget_s: float) -> None:
        """Replace the budget mid-call, restarting the clock from now.

        The student owns the time. When they say they have to go in five minutes,
        or that something freed up, that has to move the phase boundaries rather
        than just being acknowledged out loud. Clearing _last_phase lets the next
        tick re-evaluate from scratch, including backwards -- a student who buys
        themselves more time should stop being told to wrap up.
        """
        self._budget_s = budget_s
        self._started_at = time.time()
        self._last_phase = None
        logger.info("time budget revised to %.0f minutes", budget_s / 60.0)
        self._schedule_push()

    def stop(self) -> None:
        self._stopped = True
        for task in (self._ticker, self._pushing):
            if task is not None and not task.done():
                task.cancel()
        self._ticker = None

    # endregion

    # region state

    @property
    def remaining_s(self) -> float | None:
        if self._budget_s is None or self._started_at is None:
            return None
        return self._budget_s - (time.time() - self._started_at)

    def opening_note(self) -> str:
        """The one line that seeds the budget before the first mentor turn.

        Deliberately a chat-context line rather than an instructions edit: folding
        it into the prompt and calling update_instructions would re-send the whole
        mentor prompt, which is the exact cost this module exists to avoid.
        """
        if self._budget_s is None:
            return ""
        return (
            f"{INJECTION_PREFIX} The student has about {self._budget_s / 60.0:.0f} minutes. "
            f"Pace the conversation to fit that. You will be told as time runs down."
        )

    def _phase(self) -> str | None:
        """Which phase the call is in, or None while the clock is not worth mentioning."""
        remaining = self.remaining_s
        if remaining is None or not self._budget_s:
            return None

        fraction = remaining / self._budget_s
        if fraction <= 0:
            return "over"
        if fraction < 0.2:
            return "closing"
        if fraction <= QUIET_UNTIL_FRACTION:
            return "midpoint"
        return None

    def _render(self, phase: str) -> str:
        remaining = self.remaining_s or 0.0

        if phase == "over":
            return f"{INJECTION_PREFIX} {_PHASE_GUIDANCE[phase]}"

        # Avoid "About 1 minutes remain" -- under 90s the count is not the useful
        # part anyway, the instruction to start closing is.
        minutes = remaining / 60.0
        head = (
            "Less than two minutes remain."
            if minutes < 1.5
            else f"About {round(minutes):.0f} minutes remain."
        )
        return f"{INJECTION_PREFIX} {head} {_PHASE_GUIDANCE[phase]}"

    # endregion

    # region pushing

    def _on_agent_state(self, ev) -> None:
        # Injecting after the agent finishes speaking rather than before it
        # generates. One turn staler, but it cannot land in the middle of an
        # in-flight response.
        if ev.new_state == "listening":
            self._schedule_push()

    async def _tick_loop(self) -> None:
        try:
            while not self._stopped:
                await asyncio.sleep(TICK_INTERVAL_S)
                self._schedule_push()
        except asyncio.CancelledError:
            pass

    def _schedule_push(self) -> None:
        if self._stopped:
            return
        # One push in flight at a time; the next tick or turn will catch up.
        if self._pushing is not None and not self._pushing.done():
            return
        self._pushing = asyncio.create_task(self._push(), name="timekeeper_push")

    async def _push(self) -> None:
        phase = self._phase()
        if phase is None or phase == self._last_phase:
            return

        try:
            await self.inject(self._render(phase))
        except Exception:
            logger.exception("failed to inject time state")
            return

        self._last_phase = phase
        logger.info("entered %s phase", phase)

    async def inject(self, line: str) -> None:
        """Append one short line to the model's context without a spoken turn.

        Public because the opening budget note goes through the same path, before
        the clock is even running.
        """
        if not line:
            return
        chat_ctx = self._agent.chat_ctx.copy()
        chat_ctx.add_message(role="user", content=line)
        await self._agent.update_chat_ctx(chat_ctx)
        logger.debug("injected %s", line)

    # endregion
