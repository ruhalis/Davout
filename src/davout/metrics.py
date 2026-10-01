"""Evaluation metrics for predicted class probabilities."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

_EPS = 1e-12
_TOL = 1e-6


def _check_probs(p: np.ndarray) -> None:
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite")
    if p.size and (p.min() < -_TOL or p.max() > 1 + _TOL):
        raise ValueError("probabilities must be in [0, 1]")


def _prep(probs: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels)
    if p.size == 0 or y.size == 0:
        raise ValueError("empty input")
    if p.ndim != 2 or y.ndim != 1 or p.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch: probs {p.shape}, labels {y.shape}")
    _check_probs(p)
    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(np.equal(np.mod(y, 1), 0)):
            raise ValueError("labels must be integers")
    y = y.astype(int)
    if y.min() < 0 or y.max() >= p.shape[1]:
        raise ValueError("labels out of range")
    return p, y


def accuracy(probs: Any, labels: Any) -> float:
    p, y = _prep(probs, labels)
    return float(np.mean(np.argmax(p, axis=1) == y))


def nll(probs: Any, labels: Any) -> float:
    p, y = _prep(probs, labels)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], _EPS, None))))


def brier(probs: Any, labels: Any) -> float:
    p, y = _prep(probs, labels)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def reliability_bins(probs: Any, labels: Any, n_bins: int = 15) -> list[dict[str, Any]]:
    """Per-bin count, mean confidence and accuracy of the top label."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p, y = _prep(probs, labels)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out: list[dict[str, Any]] = []
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        out.append(
            {
                "lo": b / n_bins,
                "hi": (b + 1) / n_bins,
                "count": c,
                "confidence": float(conf[m].mean()) if c else None,
                "accuracy": float(correct[m].mean()) if c else None,
            }
        )
    return out


def ece(probs: Any, labels: Any, n_bins: int = 15) -> float:
    """Top-label expected calibration error."""
    bins = reliability_bins(probs, labels, n_bins)
    n = sum(b["count"] for b in bins)
    return float(
        sum(b["count"] * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"]) / n
    )


def auroc(p_true: Any, labels: Any) -> float:
    """Rank-statistic AUROC with average ranks for ties; nan if one class only."""
    s = np.asarray(p_true, dtype=float)
    y_raw = np.asarray(labels)
    if s.size == 0 or y_raw.size == 0:
        raise ValueError("empty input")
    if s.ndim != 1 or s.shape != y_raw.shape:
        raise ValueError(f"shape mismatch: scores {s.shape}, labels {y_raw.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    try:
        y_f = y_raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError("labels must be integers in {0, 1}") from None
    if not np.all(np.isfinite(y_f)) or not np.all(np.isin(y_f, (0.0, 1.0))):
        raise ValueError("labels must be integers in {0, 1}")
    y = y_f.astype(int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def summarize(probs: Any, labels: Any) -> dict[str, Any]:
    """Accuracy, NLL, Brier, ECE, n (and AUROC when K == 2)."""
    p, y = _prep(probs, labels)
    out: dict[str, Any] = {
        "accuracy": accuracy(p, y),
        "nll": nll(p, y),
        "brier": brier(p, y),
        "ece": ece(p, y),
        "n": int(len(y)),
    }
    if p.shape[1] == 2:
        out["auroc"] = auroc(p[:, 1], y)
    return out


def ragged_summarize(prob_rows: list[Sequence[float]], labels: Sequence[int]) -> dict[str, Any]:
    """Like `summarize` for rows with differing numbers of classes."""
    if len(prob_rows) != len(labels):
        raise ValueError("prob_rows and labels differ in length")
    if not prob_rows:
        raise ValueError("empty input")
    nll_v, brier_v, conf, correct = [], [], [], []
    for row, lab in zip(prob_rows, labels):
        r = np.asarray(row, dtype=float)
        if r.ndim != 1 or not 0 <= int(lab) < len(r):
            raise ValueError("labels out of range or bad row shape")
        _check_probs(r)
        lab = int(lab)
        nll_v.append(-np.log(max(r[lab], _EPS)))
        oh = np.zeros_like(r)
        oh[lab] = 1.0
        brier_v.append(np.sum((r - oh) ** 2))
        conf.append(r.max())
        correct.append(float(r.argmax() == lab))
    conf_a, corr_a = np.array(conf), np.array(correct)
    n_bins = 15
    idx = np.minimum((conf_a * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.sum() * abs(corr_a[m].mean() - conf_a[m].mean())
    n = len(labels)
    return {
        "accuracy": float(corr_a.mean()),
        "nll": float(np.mean(nll_v)),
        "brier": float(np.mean(brier_v)),
        "ece": float(e / n),
        "n": n,
    }
