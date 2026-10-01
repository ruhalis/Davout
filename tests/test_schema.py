import copy

import pytest

from davout.schema import (
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_LEVELS,
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    ValidationError,
    parse_request,
    to_text,
)

QUICKSTART = {
    "state": "Customer: my card was charged twice!",
    "model": "davout-1",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which department?",
            "criteria": {"billing": "Payment issues", "technical": None},
        },
        "mood": {
            "type": "score",
            "instructions": "How angry?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Wants a refund?",
            "criteria": {"true": "yes", "false": "no"},
        },
    },
}


def _with(**q):
    body = copy.deepcopy(QUICKSTART)
    body["questions"] = {"q": q}
    return body


def test_quickstart_parses():
    r = parse_request(QUICKSTART)
    assert r.model == "davout-1"
    d, m, n = r.questions["department"], r.questions["mood"], r.questions["refund"]
    assert isinstance(d, ChoiceQuestion) and d.type == "choice"
    assert list(d.criteria) == ["billing", "technical"]
    assert isinstance(m, ScoreQuestion) and m.type == "score" and len(m.criteria) == 3
    assert isinstance(n, NoulQuestion) and n.type == "noul"


def test_noul_criteria_optional():
    r = parse_request(_with(type="noul", instructions="x"))
    assert r.questions["q"].criteria is None
    r = parse_request(_with(type="noul", instructions="x", criteria=None))
    assert r.questions["q"].criteria is None


def test_structured_state_and_instructions():
    body = _with(type="noul", instructions={"a": 1})
    body["state"] = [1, 2]
    parse_request(body)


def _mut(f):
    body = copy.deepcopy(QUICKSTART)
    f(body)
    return body


REJECTIONS = [
    ("body", [1], ""),
    ("no_state", _mut(lambda b: b.pop("state")), "state"),
    ("null_state", _mut(lambda b: b.update(state=None)), "state"),
    ("number_state", _mut(lambda b: b.update(state=3)), "state"),
    ("bool_state", _mut(lambda b: b.update(state=True)), "state"),
    ("empty_model", _mut(lambda b: b.update(model="")), "model"),
    ("empty_questions", _mut(lambda b: b.update(questions={})), "questions"),
    ("extra_body_key", _mut(lambda b: b.update(extra=1)), "extra"),
    ("bad_type", _with(type="rank", instructions="x", criteria=[1]), "questions.q.type"),
    ("no_instructions", _with(type="noul"), "questions.q.instructions"),
    ("empty_instructions", _with(type="noul", instructions="  "), "questions.q.instructions"),
    ("extra_question_key", _with(type="noul", instructions="x", foo=1), "questions.q.foo"),
    ("choice_list", _with(type="choice", instructions="x", criteria=["a", "b"]), "questions.q.criteria"),
    ("choice_one", _with(type="choice", instructions="x", criteria={"a": None}), "questions.q.criteria"),
    ("choice_empty_name", _with(type="choice", instructions="x", criteria={"": 1, "b": None}), "questions.q.criteria"),
    ("choice_bad_value", _with(type="choice", instructions="x", criteria={"a": 1, "b": None}), "questions.q.criteria.a"),
    ("score_dict", _with(type="score", instructions="x", criteria={"a": "b"}), "questions.q.criteria"),
    ("score_one", _with(type="score", instructions="x", criteria=["a"]), "questions.q.criteria"),
    ("score_bad_level", _with(type="score", instructions="x", criteria=["a", None]), "questions.q.criteria[1]"),
    ("noul_bad_key", _with(type="noul", instructions="x", criteria={"maybe": "x"}), "questions.q.criteria.maybe"),
    ("noul_list", _with(type="noul", instructions="x", criteria=["true"]), "questions.q.criteria"),
]


@pytest.mark.parametrize("name,body,path", REJECTIONS, ids=[r[0] for r in REJECTIONS])
def test_rejections(name, body, path):
    with pytest.raises(ValidationError) as e:
        parse_request(body)
    assert e.value.path == path
    assert e.value.message


def test_choice_limits():
    ok = {f"o{i}": None for i in range(MAX_CHOICE_OPTIONS)}
    parse_request(_with(type="choice", instructions="x", criteria=ok))
    bad = {f"o{i}": None for i in range(MAX_CHOICE_OPTIONS + 1)}
    with pytest.raises(ValidationError):
        parse_request(_with(type="choice", instructions="x", criteria=bad))


def test_score_limits():
    parse_request(_with(type="score", instructions="x", criteria=[str(i) for i in range(MAX_SCORE_LEVELS)]))
    with pytest.raises(ValidationError):
        parse_request(_with(type="score", instructions="x", criteria=[str(i) for i in range(MAX_SCORE_LEVELS + 1)]))


def test_to_text():
    assert to_text("  hi \n") == "hi"
    assert to_text(None) == ""
    assert to_text({"a": "é"}) == '{\n  "a": "é"\n}'
    assert to_text([1]) == "[\n  1\n]"


def _noul(**kw):
    q = {"type": "noul", "instructions": "q"}
    q.update(kw)
    return {"state": "s", "model": "m", "questions": {"n": q}}


def test_noul_criteria_value_type_checked():
    for key in ("true", "false"):
        with pytest.raises(ValidationError) as e:
            parse_request(_noul(criteria={key: 5}))
        assert e.value.path == f"questions.n.criteria.{key}"
    ok = parse_request(_noul(criteria={"true": ["a"], "false": {"x": 1}}))
    assert ok.questions["n"].criteria == {"true": ["a"], "false": {"x": 1}}


def test_noul_empty_criteria_is_none():
    assert parse_request(_noul(criteria={})).questions["n"].criteria is None


@pytest.mark.parametrize("ins", [[], {}])
def test_empty_container_instructions_rejected(ins):
    with pytest.raises(ValidationError) as e:
        parse_request(_noul(instructions=ins))
    assert e.value.path == "questions.n.instructions"
