import pytest

from davout.backends.base import LETTERS
from davout.prompts import (
    GENERIC_DEMOS,
    Demo,
    build_candidate_prompt,
    build_prompt,
    generic_demos,
)
from davout.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

CHOICE = ChoiceQuestion("Which team should handle this?", {"billing": "payments and refunds", "tech": None})
SCORE = ScoreQuestion("How urgent is it?", ["not urgent", "somewhat urgent", "very urgent"])
NOUL = NoulQuestion("Is the customer angry?", None)

DEMOS = [
    Demo("The printer is on fire.", ChoiceQuestion("Pick a team.", {"it": "hardware", "hr": None}), 0),
    Demo("All good here.", NoulQuestion("There is a problem.", None), 1),
]
DEMO_PREFIX = (
    "The printer is on fire.\n"
    "\n"
    "Question: Pick a team.\n"
    "A. it: hardware\n"
    "B. hr\n"
    "Answer: A\n"
    "\n"
    "All good here.\n"
    "\n"
    "Statement: There is a problem.\n"
    "Question: Is the statement true?\n"
    "A. Yes\n"
    "B. No\n"
    "Answer: B\n"
    "\n"
)


def text(p):
    return p.prefix + p.state + p.suffix


def test_choice_zero_shot_snapshot():
    p = build_prompt("I was charged twice.", CHOICE)
    assert p.prefix == ""
    assert p.state == "I was charged twice."
    assert p.n_labels == 2
    assert text(p) == (
        "I was charged twice.\n"
        "\n"
        "Question: Which team should handle this?\n"
        "A. billing: payments and refunds\n"
        "B. tech\n"
        "Answer:"
    )


def test_score_zero_shot_snapshot():
    p = build_prompt("Server is down!", SCORE)
    assert p.n_labels == 3
    assert text(p) == (
        "Server is down!\n"
        "\n"
        "Question: How urgent is it?\n"
        "A. not urgent\n"
        "B. somewhat urgent\n"
        "C. very urgent\n"
        "Answer:"
    )


def test_noul_zero_shot_snapshot():
    p = build_prompt("This is unacceptable.", NOUL)
    assert p.n_labels == 2
    assert text(p) == (
        "This is unacceptable.\n"
        "\n"
        "Question: Is the customer angry?\n"
        "A. Yes\n"
        "B. No\n"
        "Answer:"
    )


@pytest.mark.parametrize("question", [CHOICE, SCORE, NOUL])
def test_two_shot_snapshot(question):
    zero = build_prompt("Live state.", question)
    p = build_prompt("Live state.", question, DEMOS)
    assert p.prefix == DEMO_PREFIX
    assert p.state == "Live state."
    assert p.suffix == zero.suffix
    assert text(p) == DEMO_PREFIX + text(zero)
    assert text(p).endswith("Answer:")


def test_two_shot_full_text():
    p = build_prompt("Live state.", SCORE, DEMOS[:1])
    assert text(p) == (
        "The printer is on fire.\n"
        "\n"
        "Question: Pick a team.\n"
        "A. it: hardware\n"
        "B. hr\n"
        "Answer: A\n"
        "\n"
        "Live state.\n"
        "\n"
        "Question: How urgent is it?\n"
        "A. not urgent\n"
        "B. somewhat urgent\n"
        "C. very urgent\n"
        "Answer:"
    )


def test_null_and_structured_descriptions():
    q = ChoiceQuestion(
        "Pick",
        {"a": None, "b": "", "c": {"k": [1, 2], "s": "é"}, "d": ["x", "y"], "e": "two\nlines  here"},
    )
    p = build_prompt("s", q)
    assert p.suffix == (
        "\n\nQuestion: Pick\n"
        "A. a\n"
        "B. b\n"
        'C. c: {"k":[1,2],"s":"é"}\n'
        'D. d: ["x","y"]\n'
        "E. e: two lines here\n"
        "Answer:"
    )
    assert p.n_labels == 5


def test_structured_state_and_instructions():
    p = build_prompt({"user": "bob", "n": 2}, ChoiceQuestion(["step one", "step two"], {"x": None, "y": None}))
    assert p.state == '{\n  "user": "bob",\n  "n": 2\n}'
    assert p.suffix == (
        '\n\nQuestion: [\n  "step one",\n  "step two"\n]\n'
        "A. x\n"
        "B. y\n"
        "Answer:"
    )


def test_structured_score_levels():
    p = build_prompt("s", ScoreQuestion("Rate", [{"label": "low"}, "high"]))
    assert p.suffix == '\n\nQuestion: Rate\nA. {"label":"low"}\nB. high\nAnswer:'


def test_state_is_stripped():
    assert build_prompt("  padded \n", NOUL).state == "padded"


def test_noul_statement_form():
    p = build_prompt("s", NoulQuestion("The customer is angry.", None))
    assert p.suffix == (
        "\n\nStatement: The customer is angry.\n"
        "Question: Is the statement true?\n"
        "A. Yes\n"
        "B. No\n"
        "Answer:"
    )


def test_noul_question_form_with_criteria():
    q = NoulQuestion("Is it urgent? ", {"true": "needs action today", "false": "can wait"})
    assert build_prompt("s", q).suffix == (
        "\n\nQuestion: Is it urgent?\n"
        "A. Yes: needs action today\n"
        "B. No: can wait\n"
        "Answer:"
    )


def test_noul_partial_criteria():
    q = NoulQuestion("It is urgent", {"false": "can wait"})
    assert build_prompt("s", q).suffix == (
        "\n\nStatement: It is urgent\n"
        "Question: Is the statement true?\n"
        "A. Yes\n"
        "B. No: can wait\n"
        "Answer:"
    )


def test_candidate_prompt_snapshot():
    p = build_candidate_prompt("I was charged twice.", CHOICE, "billing")
    assert p.n_labels == 2
    assert text(p) == (
        "I was charged twice.\n"
        "\n"
        "Question: Which team should handle this?\n"
        "Proposed answer: billing: payments and refunds\n"
        "Question: Is the proposed answer correct?\n"
        "A. Yes\n"
        "B. No\n"
        "Answer:"
    )
    assert "Proposed answer: tech\n" in build_candidate_prompt("s", CHOICE, "tech").suffix


def test_candidate_prompt_with_demos():
    demo = Demo("It crashed.", ChoiceQuestion("Category?", {"bug": "broken", "idea": None}), 1, candidate="idea")
    p = build_candidate_prompt("s", CHOICE, "tech", [demo])
    assert p.prefix == (
        "It crashed.\n"
        "\n"
        "Question: Category?\n"
        "Proposed answer: idea\n"
        "Question: Is the proposed answer correct?\n"
        "A. Yes\n"
        "B. No\n"
        "Answer: B\n"
        "\n"
    )
    assert text(p).endswith("Answer:")


def test_errors():
    with pytest.raises(ValueError):
        build_candidate_prompt("s", CHOICE, "nope")
    with pytest.raises(ValueError):
        build_prompt("s", ChoiceQuestion("q", {f"o{i}": None for i in range(27)}))
    with pytest.raises(ValueError):
        build_prompt("s", CHOICE, [Demo("d", CHOICE, 2)])
    with pytest.raises(ValueError):
        build_prompt("s", CHOICE, [Demo("d", NOUL, 0, candidate="x")])


def test_generic_demos_shape():
    assert set(GENERIC_DEMOS) == {"choice", "score", "noul", "candidate"}
    for kind, demos in GENERIC_DEMOS.items():
        assert len(demos) == 4
        assert generic_demos(kind, 2) == demos[:2]
        assert generic_demos(kind, 0) == []
        assert generic_demos(kind, 9) == demos
        for d in demos:
            assert isinstance(d.state, str) and d.state
            assert (d.candidate is not None) == (kind == "candidate")
    with pytest.raises(ValueError):
        generic_demos("other", 1)


def test_generic_demos_render_and_balance():
    for kind, demos in GENERIC_DEMOS.items():
        golds, counts = [], []
        for d in demos:
            p = build_prompt("live", NOUL, [d])
            gold = LETTERS[d.answer]
            assert p.prefix.endswith(f"\nAnswer: {gold}\n\n")
            options = [ln for ln in p.prefix.splitlines() if ln[:3] in {f"{c}. " for c in LETTERS}]
            assert 2 <= len(options) <= 5
            assert d.answer < len(options)
            golds.append(gold)
            counts.append(len(options))
        if kind in ("choice", "score"):
            assert sorted(golds) == ["A", "B", "C", "D"]
            assert sorted(counts) == [2, 3, 4, 5]
        else:
            assert sorted(golds) == ["A", "A", "B", "B"]
            assert golds[:2] == ["A", "B"]
