"""Build Jev-shaped typed answers from probabilities."""
from __future__ import annotations

import math
from typing import Any, Sequence

_DECIMALS = 6


def _check_probs(n_expected: int, probs: Sequence[float], normalized: bool) -> None:
    if len(probs) != n_expected:
        raise ValueError(f"expected {n_expected} probabilities, got {len(probs)}")
    for p in probs:
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"invalid probability {p!r}")
    if normalized and abs(sum(probs) - 1.0) > 1e-4:
        raise ValueError(f"probabilities sum to {sum(probs)}, expected 1")


def _r(x: float) -> float:
    return round(float(x), _DECIMALS)


def confidence(probs: Sequence[float]) -> float:
    """Peak-based confidence in [0, 1]: (n*max - 1) / (n - 1)."""
    n = len(probs)
    if n < 2:
        raise ValueError("confidence needs at least 2 probabilities")
    return min(1.0, max(0.0, (n * max(probs) - 1.0) / (n - 1)))


def choice_answer(names: Sequence[str], probs: Sequence[float]) -> dict[str, Any]:
    """Answer for a choice question; first option wins ties."""
    _check_probs(len(names), probs, True)
    best = max(range(len(probs)), key=lambda i: (probs[i], -i))
    return {
        "type": "choice",
        "choice": names[best],
        "confidence": _r(confidence(probs)),
        "probabilities": {n: _r(p) for n, p in zip(names, probs)},
    }


def score_answer(levels: Sequence[Any], probs: Sequence[float]) -> dict[str, Any]:
    """Answer for a score question: expected level index plus the legend."""
    _check_probs(len(levels), probs, True)
    return {
        "type": "score",
        "score": _r(sum(i * p for i, p in enumerate(probs))),
        "confidence": _r(confidence(probs)),
        "legend": {str(i): lvl for i, lvl in enumerate(levels)},
        "probabilities": {str(i): _r(p) for i, p in enumerate(probs)},
    }


def noul_answer(p_true: float) -> dict[str, Any]:
    """Answer for a noul question: probability the statement is true."""
    _check_probs(1, [p_true], False)
    return {"type": "noul", "noul": _r(p_true)}
