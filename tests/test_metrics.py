import math

import numpy as np
import pytest

from davout.metrics import (
    accuracy,
    auroc,
    brier,
    ece,
    nll,
    ragged_summarize,
    reliability_bins,
    summarize,
)

P = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.9, 0.1]])
Y = np.array([0, 1, 1, 0])


def test_hand_computed():
    assert accuracy(P, Y) == 0.75
    assert nll(P, Y) == pytest.approx(-(math.log(0.8) + math.log(0.4) + math.log(0.7) + math.log(0.9)) / 4)
    assert brier(P, Y) == pytest.approx(0.25)
    assert ece(P, Y) == pytest.approx(0.3)


def test_ece_zero_when_calibrated():
    p = np.array([[0.75, 0.25]] * 4)
    y = np.array([0, 0, 0, 1])
    assert ece(p, y) == pytest.approx(0.0)


def test_reliability_bins():
    bins = reliability_bins(P, Y, n_bins=15)
    assert len(bins) == 15
    assert sum(b["count"] for b in bins) == 4
    empty = [b for b in bins if b["count"] == 0]
    assert empty and all(b["confidence"] is None and b["accuracy"] is None for b in empty)
    assert set(bins[0]) == {"lo", "hi", "count", "confidence", "accuracy"}


def test_auroc():
    y = np.array([0, 0, 1, 1])
    assert auroc([0.1, 0.2, 0.8, 0.9], y) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], y) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], y) == 0.5
    assert auroc([0.1, 0.5, 0.5, 0.9], y) == pytest.approx(0.875)
    assert math.isnan(auroc([0.1, 0.2], [1, 1]))


def test_summarize_binary_has_auroc():
    s = summarize(P, Y)
    assert s["n"] == 4 and s["auroc"] == pytest.approx(auroc(P[:, 1], Y))
    assert "auroc" not in summarize(np.full((3, 3), 1 / 3), [0, 1, 2])


def test_ragged_matches_summarize():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4), size=200)
    y = rng.integers(0, 4, size=200)
    a = summarize(p, y)
    b = ragged_summarize([r.tolist() for r in p], y.tolist())
    for k in ("accuracy", "nll", "brier", "ece"):
        assert b[k] == pytest.approx(a[k])
    assert b["n"] == 200


def test_ragged_differing_k():
    s = ragged_summarize([[0.7, 0.3], [0.2, 0.5, 0.3]], [0, 1])
    assert s["accuracy"] == 1.0 and s["n"] == 2


def test_errors():
    with pytest.raises(ValueError):
        accuracy(P, [0, 1])
    with pytest.raises(ValueError):
        nll(P, [0, 1, 1, 2])
    with pytest.raises(ValueError):
        brier(P, [0, 1, 1, -1])
    with pytest.raises(ValueError):
        ragged_summarize([[0.5, 0.5]], [2])
    with pytest.raises(ValueError):
        ragged_summarize([[0.5, 0.5]], [0, 1])


def test_auroc_label_validation():
    with pytest.raises(ValueError):
        auroc([0.1, 0.9], [0.5, 1])
    with pytest.raises(ValueError):
        auroc([0.1, 0.9], [0, 2])
    with pytest.raises(ValueError):
        auroc([], [])
    with pytest.raises(ValueError):
        auroc([0.1, 0.9, 0.5], [0, 1])
    assert auroc([0.1, 0.9], [0.0, 1.0]) == 1.0


def test_probability_validation():
    bad = [[0.5, float("nan")], [0.5, 0.5]]
    over = [[1.5, -0.5], [0.5, 0.5]]
    for fn in (accuracy, nll, brier, ece, reliability_bins, summarize):
        for b in (bad, over):
            with pytest.raises(ValueError):
                fn(b, [0, 1])
    with pytest.raises(ValueError):
        ragged_summarize(bad, [0, 1])
    with pytest.raises(ValueError):
        ragged_summarize(over, [0, 1])
    with pytest.raises(ValueError):
        auroc([float("inf"), 0.2], [0, 1])
    assert auroc([5.0, -2.0], [1, 0]) == 1.0
    accuracy([[1.0000005, 0.0], [0.5, 0.5]], [0, 1])


@pytest.mark.parametrize("n_bins", [0, -1])
def test_n_bins_must_be_positive(n_bins):
    with pytest.raises(ValueError):
        ece(P, Y, n_bins)
    with pytest.raises(ValueError):
        reliability_bins(P, Y, n_bins)


def test_empty_input_message():
    with pytest.raises(ValueError, match="empty input"):
        accuracy([], [])
    with pytest.raises(ValueError, match="empty input"):
        ece([], [])
