import json
import math

import pytest

from davout import engine as engine_mod
from davout.backends.base import Readout
from davout.calibrate import Calibrator
from davout.engine import Engine, build_engine
from davout.schema import parse_request
from davout.scorer import LetterScorer, RawScore

REQUEST = parse_request(
    {
        "state": "My order arrived broken and I want my money back.",
        "model": "anything",
        "questions": {
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "tech": None, "sales": None}},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
            "refund": {"type": "noul", "instructions": "Is a refund requested?", "criteria": None},
        },
    }
)


def logs(*probs: float) -> list[float]:
    return [math.log(p) for p in probs]


class FakeBackend:
    name = "fake"
    max_labels = 26

    def info(self) -> dict:
        return {"name": "fake", "device": "none"}

    def read(self, prompts):
        return [Readout([0.0] * p.n_labels, 7) for p in prompts]


class FakeScorer:
    shots = 2
    shortlist = 5

    def __init__(self, raws: list[RawScore]) -> None:
        self.backend = FakeBackend()
        self.raws = raws
        self.seen = None

    def score(self, state, questions):
        self.seen = (state, list(questions))
        return self.raws


def raws(truncated=(False, False, False)) -> list[RawScore]:
    return [
        RawScore(logs(0.7, 0.2, 0.1), 40, truncated[0], None, "letter"),
        RawScore(logs(0.2, 0.3, 0.5), 30, truncated[1], None, "letter"),
        RawScore([2.0, 0.0], 25, truncated[2], None, "letter"),
    ]


def test_response_shape():
    scorer = FakeScorer(raws())
    out = Engine(scorer).answer(REQUEST)
    assert list(out) == ["model", "answers", "usage", "timing"]
    assert out["model"] == "davout-hrm-text-1b"
    assert list(out["answers"]) == ["team", "urgency", "refund"]
    assert scorer.seen == (REQUEST.state, list(REQUEST.questions.values()))
    assert out["usage"] == {"input_tokens": 95, "output_tokens": 0}
    assert isinstance(out["timing"]["total_ms"], float) and out["timing"]["total_ms"] >= 0.0
    json.dumps(out, allow_nan=False)

    team, urgency, refund = out["answers"].values()
    assert team["type"] == "choice" and team["choice"] == "billing"
    assert team["probabilities"] == pytest.approx({"billing": 0.7, "tech": 0.2, "sales": 0.1})
    assert team["confidence"] == pytest.approx(0.55)
    assert urgency["type"] == "score"
    assert urgency["score"] == pytest.approx(1.3)
    assert urgency["legend"] == {"0": "low", "1": "medium", "2": "high"}
    assert urgency["probabilities"] == pytest.approx({"0": 0.2, "1": 0.3, "2": 0.5})
    assert refund == {"type": "noul", "noul": pytest.approx(1 / (1 + math.exp(-2.0)), abs=1e-6)}
    assert all("truncated" not in a for a in out["answers"].values())


def test_calibrator_changes_probabilities():
    identity = Engine(FakeScorer(raws())).answer(REQUEST)["answers"]
    cal = Calibrator(choice_T=2.0, score_T=0.5, noul_a=0.5, noul_b=-1.0)
    scaled = Engine(FakeScorer(raws()), cal).answer(REQUEST)["answers"]

    assert scaled["team"]["choice"] == "billing"
    assert scaled["team"]["probabilities"]["billing"] < identity["team"]["probabilities"]["billing"]
    z = [math.sqrt(p) for p in (0.7, 0.2, 0.1)]
    assert scaled["team"]["probabilities"]["billing"] == pytest.approx(z[0] / sum(z), abs=1e-6)
    sq = [p * p for p in (0.2, 0.3, 0.5)]
    assert scaled["urgency"]["probabilities"]["2"] == pytest.approx(sq[2] / sum(sq), abs=1e-6)
    assert scaled["urgency"]["score"] > identity["urgency"]["score"]
    assert scaled["refund"]["noul"] == pytest.approx(0.5)
    assert identity["refund"]["noul"] > 0.5


def test_truncated_flag_only_on_truncated_questions():
    out = Engine(FakeScorer(raws(truncated=(False, True, True)))).answer(REQUEST)["answers"]
    assert "truncated" not in out["team"]
    assert out["urgency"]["truncated"] is True and out["refund"]["truncated"] is True


def test_model_name_and_info():
    cal = Calibrator(choice_T=1.5)
    info = Engine(FakeScorer(raws()), cal, model_name="custom").info()
    assert info == {
        "model": "custom",
        "backend": {"name": "fake", "device": "none"},
        "shots": 2,
        "shortlist": 5,
        "calibration": cal.to_json(),
    }
    json.dumps(info, allow_nan=False)


def test_scorer_length_mismatch():
    with pytest.raises(RuntimeError):
        Engine(FakeScorer(raws()[:2])).answer(REQUEST)


def test_build_engine_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        build_engine(backend="nope")


def test_build_engine_from_registry(monkeypatch, tmp_path):
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return FakeBackend()

    monkeypatch.setitem(engine_mod.BACKENDS, "fake", factory)
    eng = build_engine(backend="fake", device="cpu", max_tokens=512, batch_size=2)
    assert seen == {"device": "cpu", "max_tokens": 512, "batch_size": 2}
    assert isinstance(eng.scorer, LetterScorer) and eng.scorer.shots == 3
    assert eng.model_name == "davout-fake"
    assert eng.calibrator == Calibrator()

    out = eng.answer(REQUEST)
    assert out["usage"] == {"input_tokens": 21, "output_tokens": 0}
    assert out["answers"]["team"]["probabilities"] == pytest.approx({"billing": 1 / 3, "tech": 1 / 3, "sales": 1 / 3})
    assert out["answers"]["refund"]["noul"] == 0.5

    path = tmp_path / "cal.json"
    Calibrator(noul_b=3.0).save(path)
    eng = build_engine(backend="fake", shots=0, calibration=str(path))
    assert eng.scorer.shots == 0
    assert eng.calibrator.noul_b == 3.0
    assert eng.info()["calibration"]["noul_b"] == 3.0
    assert eng.answer(REQUEST)["answers"]["refund"]["noul"] > 0.9


def test_two_stage_score_uses_two_stage_temperature():
    cal = Calibrator(choice_T=1.0, choice_two_stage_T=2.0)
    letter = RawScore(logs(0.7, 0.2, 0.1), 40, False, None, "letter")
    two_stage = RawScore(logs(0.7, 0.2, 0.1), 40, False, None, "two_stage")
    rest = raws()[1:]
    out_letter = Engine(FakeScorer([letter, *rest]), cal).answer(REQUEST)["answers"]["team"]
    out_two = Engine(FakeScorer([two_stage, *rest]), cal).answer(REQUEST)["answers"]["team"]
    assert out_letter["probabilities"] == pytest.approx({"billing": 0.7, "tech": 0.2, "sales": 0.1})
    z = [math.sqrt(p) for p in (0.7, 0.2, 0.1)]
    assert out_two["probabilities"]["billing"] == pytest.approx(z[0] / sum(z), abs=1e-6)
