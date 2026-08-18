"""Keeps the student's stated time budget alive in the model's context.

The cascade trick of rebuilding context every turn is not available on this
stack. With Gemini's server-side VAD owning turn-taking, the SDK returns out of
_user_turn_completed_task before ever calling on_user_turn_completed
(agent_activity.py), so that hook never fires here. Gemini Live's history is
append-only besides.

What does work is Agent.update_instructions(): on an active realtime session the
plugin sends the new instruction text as a LiveClientContent turn with
turn_complete=False -- a mid-session injection, no reconnect. The plugin's own
docstring points at this as the way to set system-level context.

The catch is that it re-sends the *whole* instruction string, mentor prompt
included, on every push. So pushes are rationed rather than made every turn:
nothing at all while the call is young, then escalating updates as the clock
matters more. See _render() for the granularity ladder.
"""

from __future__ import annotations

import asyncio
import logging
import time

from livekit.agents import Agent, AgentSession, llm

logger = logging.getLogger("mba606.timekeeper")

# Wall-clock sweep, so a long silence still advances the phase even though no
# agent turn has completed to trigger a push.
TICK_INTERVAL_S = 20.0

# Below this fraction remaining, the clock starts being mentioned. Above it the
# budget stated in the opening instructions is enough and every push would be
# pure context cost.
QUIET_UNTIL_FRACTION = 0.5

_PHASE_GUIDANCE = {
    "midpoint": (
        "Past the halfway point. Begin narrowing toward what matters most to them."
    ),
    "closing": (
        "Time is nearly complete. Start compressing. Ask the student what they want "
        "to remember, and draw out any commitment or open question."
    ),
    "over": (
        "The time the student asked for has run out. Close out now: confirm their "
        "takeaway, then let them go. Do not open a new thread and do not start a new "
        "line of questioning. Do not end the call yourself -- leave that to them."
    ),
}


def _format_remaining(seconds: float) -> str:
    """Phrase the remaining time at a granularity that suits how much is left.

    Coarse far out and fine at the end, so an ordinary stretch of conversation
    does not generate a push per turn. Each rung produces a distinct string, which
    is what the dedupe in _push() keys on.
    """
    minutes = seconds / 60.0

    if minutes > 10:
        return f"About {round(minutes / 5) * 5:.0f} minutes remain."
    if minutes >= 2:
        return f"About {round(minutes):.0f} minutes remain."
    if minutes >= 1.25:
        return "About a minute and a half remains."
    if minutes >= 1:
        return "About a minute remains."
    return "Less than a minute remains."


class TimeKeeper:
    """Tracks the budget and re-injects it as the call runs.

    Inert until start() is called with a budget, so an intake that never got an
    answer simply means no time pressure rather than a crash.
    """

    def __init__(self, session: AgentSession, agent: Agent, base_instructions: str) -> None:
        self._session = session
        self._agent = agent
        self._base = base_instructions

        self._budget_s: float | None = None
        self._started_at: float | None = None
        self._last_line: str | None = None
        self._ticker: asyncio.Task[None] | None = None
        self._pushing: asyncio.Task[None] | None = None
        self._stopped = False

    # region lifecycle

    def start(self, budget_s: float) -> None:
        """Begin the clock. Call once intake has produced a budget."""
        self._budget_s = budget_s
        self._started_at = time.time()

        if not self._supports_injection():
            logger.warning(
                "this model cannot take mid-session instruction updates -- the time "
                "budget will be stated once and never refreshed. See "
                "RealtimeModel.capabilities.mutable_instructions."
            )
            return

        self._session.on("agent_state_changed", self._on_agent_state)
        self._ticker = asyncio.create_task(self._tick_loop(), name="timekeeper_tick")
        logger.info("timekeeper started with a %.0f minute budget", budget_s / 60.0)

    def stop(self) -> None:
        self._stopped = True
        for task in (self._ticker, self._pushing):
            if task is not None and not task.done():
                task.cancel()
        self._ticker = None

    def _supports_injection(self) -> bool:
        """Whether update_instructions actually reaches the model.

        The Gemini plugin gates mid-session updates on mutable_instructions, which
        it computes as `"3.1" not in model`. On a 3.1 native-audio model the call
        returns cleanly and does nothing, so this has to be checked rather than
        assumed -- a silent no-op is exactly the failure that would go unnoticed.
        """
        model = self._session.llm
        if not isinstance(model, llm.RealtimeModel):
            # Cascade path: instructions are re-rendered per turn anyway.
            return True
        return bool(model.capabilities.mutable_instructions)

    # endregion

    # region state

    @property
    def remaining_s(self) -> float | None:
        if self._budget_s is None or self._started_at is None:
            return None
        return self._budget_s - (time.time() - self._started_at)

    def opening_note(self) -> str:
        """The budget line folded into the mentor instructions before turn one."""
        if self._budget_s is None:
            return ""
        return (
            f"\n\nThe student said they have about {self._budget_s / 60.0:.0f} minutes. "
            f"Pace the conversation to fit that. You will be told as time runs down."
        )

    def _phase(self, fraction: float) -> str | None:
        if fraction <= 0:
            return "over"
        if fraction < 0.2:
            return "closing"
        if fraction <= QUIET_UNTIL_FRACTION:
            return "midpoint"
        return None

    def _render(self) -> str | None:
        """The line to inject, or None while the clock is not worth mentioning.

        Granularity coarsens with distance so that an unremarkable stretch of the
        call does not generate a push per turn: five-minute steps far out,
        whole minutes in the middle, half minutes at the end.
        """
        remaining = self.remaining_s
        if remaining is None or self._budget_s is None:
            return None

        phase = self._phase(remaining / self._budget_s)
        if phase is None:
            return None

        if phase == "over":
            head = ""
        else:
            head = (
                f"The student asked for {self._budget_s / 60.0:.0f} minutes. "
                f"{_format_remaining(remaining)} "
            )

        return f"[time check] {head}{_PHASE_GUIDANCE[phase]}"

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
        line = self._render()
        if line is None or line == self._last_line:
            return

        try:
            await self._agent.update_instructions(f"{self._base}\n\n{line}")
        except Exception:
            logger.exception("failed to inject time state")
            return

        self._last_line = line
        logger.debug("injected %s", line)

    # endregion
