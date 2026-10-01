"""Build multiple-choice prompts whose answer is a single letter token.

One block looks like::

    {state}

    Question: {instructions}
    A. {option}
    B. {option}
    Answer:

A demonstration is the same block followed by " {gold letter}"; blocks are
separated by a blank line and the live block ends with "Answer:".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from davout.backends.base import LETTERS, Prompt
from davout.schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, to_text

_YES_NO = ("Yes", "No")


@dataclass(frozen=True)
class Demo:
    """A solved example. `answer` is the index of the gold option.

    For noul questions 0 means true (Yes) and 1 means false (No). When
    `candidate` names an option of a choice question, the demo is rendered as
    a candidate block ("Is the proposed answer correct?") and `answer` is
    0 (Yes) or 1 (No).
    """

    state: Any
    question: Question
    answer: int
    candidate: str | None = None


def _inline(value: Any) -> str:
    """Render an option value on a single line (compact JSON for objects/arrays)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _named(name: str, description: Any) -> str:
    desc = _inline(description)
    name = _inline(name)
    return f"{name}: {desc}" if desc else name


def _yes_no(true_text: Any = None, false_text: Any = None) -> list[str]:
    return [_named("Yes", true_text), _named("No", false_text)]


def _question_lines(question: Question) -> tuple[list[str], list[str]]:
    """Return (header lines, option texts) for a question."""
    ins = to_text(question.instructions)
    if isinstance(question, ChoiceQuestion):
        return [f"Question: {ins}"], [_named(n, d) for n, d in question.criteria.items()]
    if isinstance(question, ScoreQuestion):
        return [f"Question: {ins}"], [_inline(lvl) for lvl in question.criteria]
    if isinstance(question, NoulQuestion):
        crit = question.criteria or {}
        options = _yes_no(crit.get("true"), crit.get("false"))
        if ins.endswith("?"):
            return [f"Question: {ins}"], options
        return [f"Statement: {ins}", "Question: Is the statement true?"], options
    raise TypeError(f"unsupported question type {type(question).__name__}")


def _candidate_lines(question: ChoiceQuestion, option_name: str) -> tuple[list[str], list[str]]:
    if option_name not in question.criteria:
        raise ValueError(f"unknown option {option_name!r}")
    header = [
        f"Question: {to_text(question.instructions)}",
        f"Proposed answer: {_named(option_name, question.criteria[option_name])}",
        "Question: Is the proposed answer correct?",
    ]
    return header, _yes_no()


def _tail(header: list[str], options: list[str]) -> str:
    """Everything after the state text, ending in "Answer:"."""
    if not 2 <= len(options) <= len(LETTERS):
        raise ValueError(f"need 2..{len(LETTERS)} options, got {len(options)}")
    lines = header + [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    return "\n\n" + "\n".join(lines) + "\nAnswer:"


def _demo_text(demo: Demo) -> str:
    if demo.candidate is not None:
        if not isinstance(demo.question, ChoiceQuestion):
            raise ValueError("candidate demos need a choice question")
        header, options = _candidate_lines(demo.question, demo.candidate)
    else:
        header, options = _question_lines(demo.question)
    if not 0 <= demo.answer < len(options):
        raise ValueError(f"demo answer {demo.answer} out of range for {len(options)} options")
    return f"{to_text(demo.state)}{_tail(header, options)} {LETTERS[demo.answer]}\n\n"


def _prompt(state: Any, header: list[str], options: list[str], demos: Sequence[Demo]) -> Prompt:
    return Prompt(
        prefix="".join(_demo_text(d) for d in demos),
        state=to_text(state),
        suffix=_tail(header, options),
        n_labels=len(options),
    )


def build_prompt(state: Any, question: Question, demos: Sequence[Demo] = ()) -> Prompt:
    """Prompt for one question: demos in `prefix`, the live state in `state`."""
    header, options = _question_lines(question)
    return _prompt(state, header, options, demos)


def build_candidate_prompt(
    state: Any, question: ChoiceQuestion, option_name: str, demos: Sequence[Demo] = ()
) -> Prompt:
    """Yes/No prompt asking whether one option of a choice question is correct."""
    header, options = _candidate_lines(question, option_name)
    return _prompt(state, header, options, demos)


GENERIC_DEMOS: dict[str, list[Demo]] = {
    "choice": [
        Demo(
            "Ticket: I was charged twice for my subscription this month and need one payment refunded.",
            ChoiceQuestion(
                "Which team should handle this ticket?",
                {
                    "technical": "bugs, outages and error messages",
                    "billing": "payments, refunds and invoices",
                    "sales": "upgrades and new purchases",
                },
            ),
            1,
        ),
        Demo(
            "Log: ERROR db-pool: connection timed out; retry 3 of 3 failed.",
            ChoiceQuestion(
                "What does this log entry report?",
                {
                    "service startup": None,
                    "user login": None,
                    "configuration change": None,
                    "database failure": None,
                },
            ),
            3,
        ),
        Demo(
            "Review: The headphones arrived quickly, sound fantastic and the battery lasts all week.",
            ChoiceQuestion(
                "What is the sentiment of the review?",
                {
                    "positive": "the reviewer is happy with the product",
                    "negative": "the reviewer is unhappy with the product",
                },
            ),
            0,
        ),
        Demo(
            "Email: Hi team, I am away on Thursday. Can we move the planning meeting to Friday?",
            ChoiceQuestion(
                "What is the main purpose of this email?",
                {
                    "complaint": "reports a problem",
                    "payment reminder": "asks for an invoice to be paid",
                    "rescheduling request": "asks to move a meeting",
                    "job application": "applies for a position",
                    "newsletter": "shares news or promotions",
                },
            ),
            2,
        ),
    ],
    "score": [
        Demo(
            "Review: Absolutely love this blender. It is quiet, powerful and easy to clean.",
            ScoreQuestion(
                "How satisfied is the customer?",
                ["dissatisfied", "neither satisfied nor dissatisfied", "satisfied"],
            ),
            2,
        ),
        Demo(
            "Status update: one of the four modules is finished; the other three have not been started.",
            ScoreQuestion(
                "How much of the project is complete?",
                ["none of it", "about a quarter", "about half", "about three quarters", "all of it"],
            ),
            1,
        ),
        Demo(
            "Log: 02:14 nightly backup finished with 0 errors and 0 warnings.",
            ScoreQuestion(
                "How serious is the situation described in the log?",
                ["nothing is wrong", "something is seriously wrong"],
            ),
            0,
        ),
        Demo(
            "Teacher's note: Outstanding essay. The thesis is clear, the evidence is strong and the writing is flawless.",
            ScoreQuestion("How good is the essay according to the note?", ["poor", "fair", "good", "excellent"]),
            3,
        ),
    ],
    "noul": [
        Demo(
            "Tracking: order 5521 was delivered on Monday and signed for by the customer.",
            NoulQuestion("Has the order been delivered?", None),
            0,
        ),
        Demo(
            "Email: Thanks for the offer, but I have decided to stay with my current provider.",
            NoulQuestion(
                "The customer accepted the offer.",
                {
                    "true": "the customer agreed to the offer",
                    "false": "the customer declined or has not decided",
                },
            ),
            1,
        ),
        Demo(
            "Log: 09:15 user alice logged in from a known device; no alerts were raised.",
            NoulQuestion("Does this log show a security incident?", None),
            1,
        ),
        Demo(
            "Review: The soup was cold and the waiter was rude to us.",
            NoulQuestion("The review is negative.", None),
            0,
        ),
    ],
    "candidate": [
        Demo(
            "Ticket: The app crashes every time I open the camera screen.",
            ChoiceQuestion(
                "Which category fits this ticket?",
                {
                    "bug report": "something is broken or behaves incorrectly",
                    "feature request": "asks for new functionality",
                    "billing": "payments, refunds and invoices",
                },
            ),
            0,
            candidate="bug report",
        ),
        Demo(
            "Review: Shipping took three weeks and the box arrived crushed.",
            ChoiceQuestion("What is the sentiment of the review?", {"positive": None, "negative": None}),
            1,
            candidate="positive",
        ),
        Demo(
            "Email: Please find attached the invoice for March. Payment is due within 30 days.",
            ChoiceQuestion(
                "What type of email is this?",
                {
                    "invoice": "requests payment for goods or services",
                    "meeting invite": "proposes a time to meet",
                    "complaint": "reports a problem or dissatisfaction",
                    "newsletter": "shares general news or promotions",
                },
            ),
            1,
            candidate="meeting invite",
        ),
        Demo(
            "Log: 18:42 disk /dev/sda1 is 98% full; writes may start failing.",
            ChoiceQuestion(
                "Which alert type is this?",
                {
                    "storage": "disk space and volumes",
                    "network": "connectivity and latency",
                    "authentication": "logins and permissions",
                },
            ),
            0,
            candidate="storage",
        ),
    ],
}


def generic_demos(kind: str, k: int) -> list[Demo]:
    """First `k` built-in demonstrations for "choice", "score", "noul" or "candidate"."""
    if kind not in GENERIC_DEMOS:
        raise ValueError(f"unknown demo kind {kind!r}")
    return GENERIC_DEMOS[kind][: max(0, k)]
