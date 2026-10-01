"""Post-hoc calibration (temperature scaling, Platt scaling)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def softmax(logits: Any, T: float = 1.0) -> np.ndarray:
    """Numerically stable softmax over the last axis, with temperature T."""
    if not math.isfinite(T) or T <= 0:
        raise ValueError(f"T must be finite and > 0, got {T}")
    z = np.asarray(logits, dtype=float) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x: Any) -> Any:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def _mean_nll(rows: list[np.ndarray], labels: Sequence[int], T: float) -> float:
    total = 0.0
    for r, y in zip(rows, labels):
        z = r / T
        m = z.max()
        total += math.log(np.exp(z - m).sum()) + m - z[y]
    return total / len(rows)


def fit_temperature(logit_rows: list[Sequence[float]], labels: Sequence[int]) -> float:
    """Fit a scalar T > 0 minimising mean NLL by golden-section search on log T."""
    if len(logit_rows) != len(labels) or not logit_rows:
        raise ValueError("logit_rows and labels must be non-empty and equal length")
    rows = [np.asarray(r, dtype=float) for r in logit_rows]
    labs = [int(y) for y in labels]
    for r, y in zip(rows, labs):
        if r.ndim != 1 or not 0 <= y < len(r):
            raise ValueError("labels out of range or bad row shape")

    def f(lt: float) -> float:
        return _mean_nll(rows, labs, math.exp(lt))

    lo, hi = math.log(0.05), math.log(50.0)
    phi = (math.sqrt(5) - 1) / 2
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(60):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = f(d)
    return float(math.exp((lo + hi) / 2))


def fit_platt(margins: Sequence[float], labels: Sequence[int]) -> tuple[float, float]:
    """Fit p = sigmoid(a*m + b) by damped Newton on ridge-regularised log loss."""
    m = np.asarray(margins, dtype=float)
    y = np.asarray(labels, dtype=float)
    if m.shape != y.shape or m.ndim != 1 or m.size == 0:
        raise ValueError("margins and labels must be equal-length 1-D sequences")
    ridge = 1e-6
    X = np.stack([m, np.ones_like(m)], axis=1)

    def loss(w: np.ndarray) -> float:
        z = X @ w
        ll = np.logaddexp(0.0, z) - y * z
        return float(ll.mean() + 0.5 * ridge * w @ w)

    w = np.array([1.0, 0.0])
    cur = loss(w)
    for _ in range(100):
        p = _sigmoid(X @ w)
        grad = X.T @ (p - y) / len(m) + ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X / len(m) + ridge * np.eye(2)
        step = np.linalg.solve(H, grad)
        t = 1.0
        while t > 1e-10:
            new = loss(w - t * step)
            if new <= cur:
                break
            t /= 2
        else:
            break
        w = w - t * step
        done = cur - new < 1e-14
        cur = new
        if done:
            break
    return float(w[0]), float(w[1])


@dataclass
class Calibrator:
    """Holds fitted calibration parameters for each question type."""

    choice_T: float = 1.0
    choice_two_stage_T: float = 1.0
    score_T: float = 1.0
    noul_a: float = 1.0
    noul_b: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("choice_T", "choice_two_stage_T", "score_T"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and > 0, got {v}")
        for name in ("noul_a", "noul_b"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")

    def choice_probs(self, logits: Sequence[float], stage: str = "letter") -> list[float]:
        """Choice probabilities; two-stage log-probabilities use their own temperature."""
        T = self.choice_two_stage_T if stage == "two_stage" else self.choice_T
        return softmax(logits, T).tolist()

    def score_probs(self, logits: Sequence[float]) -> list[float]:
        return softmax(logits, self.score_T).tolist()

    def noul_prob(self, margin: float) -> float:
        return float(_sigmoid(self.noul_a * margin + self.noul_b))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Calibrator:
        known = {f.name for f in fields(cls)}
        unknown = sorted(str(k) for k in d if k not in known)
        if unknown:
            raise ValueError(f"unknown Calibrator key(s): {', '.join(unknown)}")
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Calibrator:
        return cls.from_json(json.loads(Path(path).read_text()))
