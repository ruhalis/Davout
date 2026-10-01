import math

import pytest

from davout.answers import choice_answer, confidence, noul_answer, score_answer


@pytest.mark.parametrize(
    "probs,expected",
    [
        ([0.61, 0.20, 0.19], 0.42),
        ([0.74, 0.065, 0.065, 0.065, 0.065], 0.67),
        ([0.40, 0.20, 0.20, 0.20], 0.20),
        ([0.57, 0.43, 0.0], 0.35),
        ([0.85, 0.0, 0.15], 0.78),
    ],
)
def test_confidence_reference(probs, expected):
    assert confidence(probs) == pytest.approx(expected, abs=0.01)


def test_confidence_bounds():
    assert confidence([1.0, 0.0]) == 1.0
    assert confidence([0.5, 0.5]) == 0.0
    with pytest.raises(ValueError):
        confidence([1.0])


def test_choice_answer():
    a = choice_answer(["x", "y", "z"], [0.2, 0.5, 0.3])
    assert a["type"] == "choice" and a["choice"] == "y"
    assert list(a["probabilities"]) == ["x", "y", "z"]
    assert a["probabilities"]["y"] == 0.5


def test_choice_tie_first():
    assert choice_answer(["a", "b"], [0.5, 0.5])["choice"] == "a"


def test_score_answer():
    a = score_answer(["lo", {"k": 1}, ["hi"]], [0.0, 0.57, 0.43])
    assert a["type"] == "score"
    assert a["score"] == pytest.approx(1.43)
    assert a["legend"] == {"0": "lo", "1": {"k": 1}, "2": ["hi"]}
    assert a["probabilities"] == {"0": 0.0, "1": 0.57, "2": 0.43}


def test_noul_answer():
    assert noul_answer(0.1234567891) == {"type": "noul", "noul": 0.123457}


def test_rounding():
    a = choice_answer(["a", "b"], [1 / 3 + 1 / 3, 1 / 3])
    assert a["probabilities"]["a"] == round(2 / 3, 6)


@pytest.mark.parametrize(
    "fn,args",
    [
        (choice_answer, (["a", "b"], [1.0])),
        (choice_answer, (["a", "b"], [0.5, 0.6])),
        (choice_answer, (["a", "b"], [math.nan, 1.0])),
        (choice_answer, (["a", "b"], [1.5, -0.5])),
        (score_answer, (["a", "b"], [0.2, 0.2])),
        (noul_answer, (1.2,)),
        (noul_answer, (math.inf,)),
        (noul_answer, (math.nan,)),
    ],
)
def test_validation(fn, args):
    with pytest.raises(ValueError):
        fn(*args)
