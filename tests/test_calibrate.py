import numpy as np
import pytest

from davout.calibrate import Calibrator, fit_platt, fit_temperature, softmax
from davout.metrics import nll


def test_softmax():
    assert softmax([1000.0, 1000.0]).tolist() == [0.5, 0.5]
    s = softmax(np.array([[0.0, 1.0, 2.0], [3.0, 3.0, 3.0]]))
    assert s.shape == (2, 3) and np.allclose(s.sum(axis=1), 1)
    assert np.allclose(softmax([2.0, 0.0], T=2.0), softmax([1.0, 0.0]))


def test_fit_temperature():
    rng = np.random.default_rng(0)
    n, k = 4000, 5
    z = rng.normal(0, 1.5, size=(n, k))
    p = softmax(z)
    y = np.array([rng.choice(k, p=row) for row in p])
    logits = 3.0 * z
    T = fit_temperature(list(logits), y)
    assert abs(T - 3.0) / 3.0 < 0.15
    before = nll(softmax(logits), y)
    after = nll(softmax(logits, T), y)
    assert after < before


def test_fit_temperature_ragged():
    rng = np.random.default_rng(1)
    rows = [rng.normal(size=int(rng.integers(2, 6))) for _ in range(50)]
    labels = [int(np.argmax(r)) for r in rows]
    assert fit_temperature(rows, labels) > 0


def test_fit_platt():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 3, size=20000)
    p = 1 / (1 + np.exp(-(0.5 * m - 1.0)))
    y = (rng.random(20000) < p).astype(int)
    a, b = fit_platt(m, y)
    assert a == pytest.approx(0.5, abs=0.1)
    assert b == pytest.approx(-1.0, abs=0.1)


def test_identity_calibrator():
    c = Calibrator()
    logits = [0.3, -1.0, 2.0]
    assert np.allclose(c.choice_probs(logits), softmax(logits))
    assert np.allclose(c.score_probs(logits), softmax(logits))
    assert c.noul_prob(0.7) == pytest.approx(1 / (1 + np.exp(-0.7)))


def test_calibrator_roundtrip(tmp_path):
    c = Calibrator(choice_T=2.5, score_T=1.2, noul_a=0.5, noul_b=-1.0, meta={"n": 10})
    assert Calibrator.from_json(c.to_json()) == c
    path = tmp_path / "cal.json"
    c.save(path)
    assert Calibrator.load(path) == c
    assert c.choice_probs([2.5, 0.0])[0] == pytest.approx(float(softmax([1.0, 0.0])[0]))


@pytest.mark.parametrize(
    "kw",
    [
        {"choice_T": 0.0},
        {"score_T": -1.0},
        {"choice_T": float("nan")},
        {"score_T": float("inf")},
        {"noul_a": float("nan")},
        {"noul_b": float("inf")},
    ],
)
def test_calibrator_rejects_bad_params(kw):
    with pytest.raises(ValueError):
        Calibrator(**kw)


def test_from_json_unknown_keys():
    with pytest.raises(ValueError, match="bogus"):
        Calibrator.from_json({"choice_T": 1.0, "bogus": 1})


@pytest.mark.parametrize("T", [0.0, -1.0, float("nan"), float("inf")])
def test_softmax_bad_temperature(T):
    with pytest.raises(ValueError):
        softmax([1.0, 2.0], T)


def test_two_stage_temperature():
    c = Calibrator(choice_T=2.0, choice_two_stage_T=4.0)
    logits = [2.0, 0.0, -1.0]
    assert np.allclose(c.choice_probs(logits), softmax(logits, 2.0))
    assert np.allclose(c.choice_probs(logits, "letter"), softmax(logits, 2.0))
    assert np.allclose(c.choice_probs(logits, "two_stage"), softmax(logits, 4.0))
    assert np.allclose(c.choice_probs(logits, "nli"), softmax(logits, 2.0))


def test_two_stage_temperature_roundtrip_and_old_json():
    c = Calibrator(choice_T=2.5, choice_two_stage_T=0.7)
    assert Calibrator.from_json(c.to_json()) == c
    assert c.to_json()["choice_two_stage_T"] == 0.7
    old = Calibrator.from_json({"choice_T": 2.5, "score_T": 1.2, "noul_a": 0.5, "noul_b": -1.0, "meta": {}})
    assert old.choice_two_stage_T == 1.0 and old.choice_T == 2.5
    with pytest.raises(ValueError):
        Calibrator(choice_two_stage_T=0.0)
