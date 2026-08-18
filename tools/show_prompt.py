"""Print the instructions the mentor actually starts a call with.

    uv run tools/show_prompt.py
    uv run tools/show_prompt.py --date 2026-09-08
    uv run tools/show_prompt.py --quiet          # sizes and the week table only

The point is the size line. The course spine is static and rides in every request,
so it is the one part of the prompt whose cost is paid on every turn of every call
-- and the one part that is easy to let creep. Check it here before checking it on
a phone call.

Token counts are estimated at 4 chars/token, which is close enough to tell 900 from
2000 and not close enough to argue about.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import course  # noqa: E402
from agent import INSTRUCTIONS, mentor_instructions  # noqa: E402

# Rough ceiling for the spine. Not enforced anywhere -- this is a number to notice
# blowing past, not a build gate.
SPINE_BUDGET_TOKENS = 1000


def tokens(text: str) -> int:
    return round(len(text) / 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="ISO date to resolve the week against")
    parser.add_argument("--quiet", action="store_true", help="skip the prompt body")
    args = parser.parse_args()

    today = date.fromisoformat(args.date) if args.date else date.today()

    spine = course.spine(today)
    full = f"{INSTRUCTIONS}\n\n{spine}"

    if not args.quiet:
        print(full)
        print()
        print("=" * 72)

    print()
    print(f"date          {today}  ->  week {course.current_week(today)}")
    print(f"persona       {len(INSTRUCTIONS):>6}c  ~{tokens(INSTRUCTIONS):>4} tokens")
    over = "  OVER BUDGET" if tokens(spine) > SPINE_BUDGET_TOKENS else ""
    print(f"course spine  {len(spine):>6}c  ~{tokens(spine):>4} tokens{over}")
    print(f"total         {len(full):>6}c  ~{tokens(full):>4} tokens")

    # mentor_instructions() is what the agent really builds. If these diverge, the
    # thing being measured above is not the thing being sent.
    if mentor_instructions() != f"{INSTRUCTIONS}\n\n{course.spine()}":
        print("\nWARNING: mentor_instructions() does not match what was measured")

    print()
    print("week resolution")
    probes = [
        (course.MEETINGS[0] - timedelta(days=1), "day before the term"),
        *((meeting, f"meeting {i}") for i, meeting in enumerate(course.MEETINGS, 1)),
        (course.MEETINGS[-1] + timedelta(days=3), "three days after the last class"),
        (course.MEETINGS[-1] + timedelta(days=30), "a month after"),
    ]
    for day, label in probes:
        print(f"  {day}  week {str(course.current_week(day)):<4}  {label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
