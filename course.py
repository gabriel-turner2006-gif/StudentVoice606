"""What MBA 606 actually is, compiled small enough to sit in every prompt.

`docs/syllabus.md` is the human-readable source and stays that way -- it is 27KB
of markdown, roughly 7k tokens, and it is `.dockerignore`d out of the deployed
image. Injecting it would cost more than it returns and would pull a
forty-words-per-turn agent toward delivering it. What lives here instead is a hand
compiled distillation holding only the parts that change an answer: the lenses the
course thinks with, what each week asks for, what students have to produce, and
how they are graded.

Kept in sync with the syllabus by hand. When the syllabus changes, this changes --
nothing detects drift, so the two are edited together or not at all.

The week is computed per call rather than at import, because a worker process
outlives a Tuesday.
"""

from __future__ import annotations

from datetime import date

# The five class meetings, from the syllabus header block ("Tuesdays, Aug. 18-Sept.
# 15", Fall 2026). Each date opens the week that carries its name -- Week 2's work
# is what happens between the second meeting and the third.
MEETINGS = (
    date(2026, 8, 18),
    date(2026, 8, 25),
    date(2026, 9, 1),
    date(2026, 9, 8),
    date(2026, 9, 15),
)

# How long the last week stays "current" after its meeting. The course ends with a
# demonstration rather than trailing work, so a week is all it needs.
_FINAL_WEEK_DAYS = 7


def current_week(today: date | None = None) -> int | None:
    """Which of the five weeks a caller is in, or None outside the term.

    None is a real answer, not a failure: someone calling in July or in November
    is not in any week, and defaulting them into Week 1 would put the wrong
    mission in front of them. Callers must handle it.
    """
    today = today or date.today()

    if today < MEETINGS[0]:
        return None
    if (today - MEETINGS[-1]).days > _FINAL_WEEK_DAYS:
        return None

    # Last meeting that has already happened.
    return sum(1 for meeting in MEETINGS if meeting <= today)


# The shape of the term at a glance. Every call gets this; only the current week
# gets the detail below it.
_ARC = (
    "Week 1 REALITY (ASK) -> Week 2 POSSIBILITY (GROUND) -> Week 3 COMMUNITY "
    "(BUILD) -> Week 4 RESPONSIBILITY (ORCHESTRATE) -> Week 5 POLARITY (SHIP)."
)

# Per-week detail. Core question, reader, and mission come from the five-week
# schedule; the technical layer comes from the AI progression table, folded in here
# so the week reads as one thing rather than two lookups.
_WEEKS = {
    1: (
        "Week 1 -- REALITY. Define reality, reclaim sanity. Core question: what is "
        "true about this moment, about me, and what does leadership require now? "
        "Focus: sensemaking, self-awareness, judgment, presence, self-regulation. "
        "AI layer ASK -- prompting, source-grounded research, verification. "
        "Mission WHY YOU. WHY NOW? -- a 3-5 slide deck and a 60-second case for why "
        "they belong in a future leadership pipeline."
    ),
    2: (
        "Week 2 -- POSSIBILITY. Learn and imagine. Core question: if education no "
        "longer ends with a degree, how do I take responsibility for learning? "
        "Focus: mindset, reframing, curiosity, foresight, Three Horizons. "
        "AI layer GROUND -- context engineering, NotebookLM. "
        "Mission THE 2036 LEARNING ADVANTAGE -- name the utopian possibility, the "
        "dystopian risk, and the bet they would make now."
    ),
    3: (
        "Week 3 -- COMMUNITY. Connect and create. Core question: what can I "
        "understand only through another person, and what story deserves to become "
        "real? Focus: listening, perspective, empathy, story. "
        "AI layer BUILD -- Build Context Brief, rapid prototyping. "
        "Mission STORY + BUILD -- conduct a real interview, tell the human story of "
        "the need, build a first working version. The longest session of the five."
    ),
    4: (
        "Week 4 -- RESPONSIBILITY. Test and own. Core question: is the thing useful, "
        "trustworthy, and worth improving? What will I own? "
        "Focus: prioritization, systems thinking, feedback, execution. "
        "AI layer ORCHESTRATE -- workflow design, agent logic, evaluation. "
        "Mission MAKE IT USEFUL -- put the build in front of a real person, listen "
        "without defending, and define what the human owns versus what AI handles."
    ),
    5: (
        "Week 5 -- POLARITY. Ship and stand. Core question: what belongs to me, what "
        "belongs to the machine, and what am I willing to put my name on? "
        "Focus: judgment, integration, executive presence, ethical discernment. "
        "AI layer SHIP -- refinement, documentation, handoff, demonstration. "
        "Mission SHIP + STAND -- demonstrate the final artifact, explain what AI did "
        "and what they did, answer questions without a script."
    ),
}

SPINE = """\
[About MBA 606]

Leadership Lab at the University of Louisville, Fall 2026. Five Tuesday evening
sessions, taught by Dr. Alfred Frager and Daniel Montgomery. Five missions, one
portfolio, one build. Each week runs the same loop: Read, Experience, Build,
Reflect, Approve.

The five lenses -- how the course reads a situation, and your best tool for asking
a sharper question:
- Reality (strategic): what is true now? What must be named, accepted, or decided?
- Possibility (visionary): what else could be true? What future is emerging?
- Community (relational): who must be understood, trusted, challenged, included?
- Responsibility (systemic): what will we own, build, test, measure, improve?
- Polarity (integrative): what tension must be held rather than resolved? What
  belongs to the human, the machine, the team, the system?

The arc of the term: {arc}

By the end they owe: a Navigator Leadership Profile, a living Context File, a
T-shaped Capability Map, five mission artifacts, a working build someone else can
use, a Life + Work Portfolio, and a 90-day development plan.

Capabilities come in six families -- DISCERN (judgment), MAKE SENSE (problem
framing, systems), IMAGINE (question design, reframing), EXPRESS (brevity, story),
RELATE (listening, perspective), DELIVER (prioritization, decisions, follow
through). Each student picks three to five, not thirty.

Grading: Weekly Preparation + Missions 15%, Presence + Participation 15%, Life +
Work Portfolio 40%, Working Artifact + Demonstration 30%. Only five sessions, so
every class matters; three absences is an automatic F. Between-class work is due
before the next session; it becomes material for that class. AI use is
permitted unless a mission requires a human-first attempt, but students own every
claim they submit.

The house rules students are testing this term: have a thought before you have a
prompt; use context, not magic words; keep some things hard, because the struggle
may be where the learning lives; do not outsource the brave part -- the
conversation, the decision, the responsibility, or the apology; other people are
not inputs; tell the truth about what AI did; stand behind your work.

[How this fits your role]

You are a reflection and rehearsal partner. You are not a therapist, an authority,
or a grader, and you decide nothing about anyone's standing in the course. Depth is
expected of these students; disclosure is not -- they choose how far to go, and
they approve anything that gets saved. Do not draw out sensitive personal,
employer-owned, or confidential material; if a student heads there, let it be their
choice, not your invitation.

For anything about grades on record, accommodations, absences, or a deadline they
need moved: that is Dr. Frager and Blackboard, not you. Say so and move on.

[How to use all of this]

This is background to think with, not material to deliver. Never recite it, never
quote the syllabus, and never walk a student through a list of weeks or lenses. Use
it the way a colleague who happens to know the course would: to recognize what they
are describing, place it in the term, and ask the question that actually fits. A
student should be able to finish the call without ever hearing you name a
framework.\
"""

_NOT_IN_SESSION = (
    "The term is not currently in session, so do not assume the student is mid-week "
    "on anything. Ask what they are working on."
)


def week_block(week: int | None) -> str:
    """The 'where we are right now' line, or the out-of-term note."""
    if week is None:
        return _NOT_IN_SESSION
    return (
        f"The class is in Week {week}. Most likely what they are calling about, but "
        f"let them tell you -- the work compounds and they may still be finishing "
        f"something earlier.\n\n{_WEEKS[week]}"
    )


def describe(today: date | None = None) -> str:
    """One line for the call log, mirroring pipeline.describe().

    Which week a call resolved to is not recoverable from the transcript, and it
    changes what the model was told -- so it belongs in the log next to the
    pipeline config.
    """
    week = current_week(today)
    return f"week {week}" if week else "outside the term"


def spine(today: date | None = None) -> str:
    """The full course block, with the current week resolved."""
    return (
        f"{SPINE.format(arc=_ARC)}\n\n"
        f"[Where the class is]\n\n{week_block(current_week(today))}"
    )
