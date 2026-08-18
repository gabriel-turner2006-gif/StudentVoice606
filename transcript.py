"""Renders a finished session's history into the transcript string the backend wants.

The webhook takes `transcript` as a single string, and we want timestamps in it,
so the ChatContext is walked directly rather than round-tripped through
to_dict(). Every ChatMessage carries created_at, which is turned into an offset
from the moment the call was answered.
"""

from __future__ import annotations

import logging

from livekit.agents import llm

logger = logging.getLogger("mba606.transcript")

_SPEAKERS = {"user": "student", "assistant": "agent"}

# Time-state injections are addressed to the model, not spoken by anyone. They
# should never appear in a student's record even if they surface in history.
_INJECTION_PREFIX = "[time check]"


def render(history: llm.ChatContext, answered_at: float) -> tuple[str, int]:
    """Return (transcript text, number of substantive turns).

    The turn count is what decides whether a call is worth submitting at all: a
    consented call that dropped during intake is noise, not data.
    """
    lines: list[str] = []
    turns = 0

    for item in history.items:
        if not isinstance(item, llm.ChatMessage):
            continue  # tool calls and their results are machinery, not conversation

        speaker = _SPEAKERS.get(item.role)
        if speaker is None:
            continue

        text = (item.text_content or "").strip()
        if not text or text.startswith(_INJECTION_PREFIX):
            continue

        offset = max(0.0, item.created_at - answered_at)
        stamp = f"{int(offset) // 60:02d}:{int(offset) % 60:02d}"
        suffix = " [interrupted]" if item.interrupted else ""
        lines.append(f"[{stamp}] {speaker}: {text}{suffix}")
        turns += 1

    return "\n".join(lines), turns


def count_after(history: llm.ChatContext, since: float | None) -> int:
    """Count spoken turns that landed after `since`.

    Used to tell a real conversation from a call that hung up moments after
    consent. Intake itself always produces a handful of turns, so the raw total
    would say "substantive" about a call that contained nothing.
    """
    if since is None:
        return 0

    return sum(
        1
        for item in history.items
        if isinstance(item, llm.ChatMessage)
        and item.role in _SPEAKERS
        and item.created_at > since
        and (item.text_content or "").strip()
        and not (item.text_content or "").strip().startswith(_INJECTION_PREFIX)
    )
