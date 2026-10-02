"""Answer typed-decision requests: score, calibrate, shape the response."""
from __future__ import annotations

import time
from typing import Any, Callable, Protocol, Sequence

from davout.answers import choice_answer, noul_answer, score_answer
from davout.backends.base import LabelBackend
from davout.calibrate import Calibrator
from davout.schema import ChoiceQuestion, Question, Request, ScoreQuestion
from davout.scorer import LetterScorer, RawScore

DEFAULT_SHOTS = 0  # zero-shot is as accurate as few-shot and 2-3x faster


class Scorer(Protocol):
    def score(self, state: Any, questions: Sequence[Question]) -> list[RawScore]: ...


class Engine:
    """Runs a scorer over a request and returns the Jev response shape."""

    def __init__(
        self,
        scorer: Scorer,
        calibrator: Calibrator | None = None,
        model_name: str = "davout-hrm-text-1b",
    ) -> None:
        self.scorer = scorer
        self.calibrator = calibrator if calibrator is not None else Calibrator()
        self.model_name = model_name

    def _answer(self, question: Question, raw: RawScore) -> dict[str, Any]:
        cal = self.calibrator
        if isinstance(question, ChoiceQuestion):
            ans = choice_answer(list(question.criteria), cal.choice_probs(raw.logits, raw.stage))
        elif isinstance(question, ScoreQuestion):
            ans = score_answer(question.criteria, cal.score_probs(raw.logits))
        else:
            ans = noul_answer(cal.noul_prob(raw.logits[0] - raw.logits[1]))
        if raw.truncated:
            ans["truncated"] = True
        return ans

    def answer(self, request: Request) -> dict[str, Any]:
        """Answer every question of `request`, in the request's question order."""
        t0 = time.perf_counter()
        ids = list(request.questions)
        raws = self.scorer.score(request.state, [request.questions[i] for i in ids])
        if len(raws) != len(ids):
            raise RuntimeError(f"scorer returned {len(raws)} results for {len(ids)} questions")
        answers = {i: self._answer(request.questions[i], raw) for i, raw in zip(ids, raws)}
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": sum(r.prompt_tokens for r in raws), "output_tokens": 0},
            "timing": {"total_ms": (time.perf_counter() - t0) * 1000.0},
        }

    def info(self) -> dict[str, Any]:
        """Model name, backend details, prompting settings and calibration parameters."""
        out: dict[str, Any] = {"model": self.model_name}
        backend = getattr(self.scorer, "backend", None)
        if backend is not None:
            out["backend"] = backend.info()
        for key in ("shots", "shortlist"):
            if hasattr(self.scorer, key):
                out[key] = getattr(self.scorer, key)
        out["calibration"] = self.calibrator.to_json()
        return out


def _hrm_backend(*, device: str, max_tokens: int, batch_size: int, model: str | None = None) -> LabelBackend:
    from davout.backends.hrm import HrmBackend

    which = {} if model is None else {"model_id_or_path": model}
    return HrmBackend(device=device, max_tokens=max_tokens, batch_size=batch_size, **which)


# name -> factory(device=, max_tokens=, batch_size=[, model=]); add new backends here.
BACKENDS: dict[str, Callable[..., LabelBackend]] = {"hrm": _hrm_backend}


def build_engine(
    *,
    backend: str = "hrm",
    device: str = "auto",
    shots: int | None = None,
    calibration: str | None = None,
    max_tokens: int = 4096,
    batch_size: int = 8,
    model: str | None = None,
) -> Engine:
    """Build an `Engine` for a registered backend; `calibration` is a Calibrator JSON path.

    `model` is a model id or checkpoint path for the backend (None = its stock weights);
    a custom model shows in the reported name as `davout-<backend>@<model>`.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; available: {', '.join(sorted(BACKENDS))}")
    calibrator = Calibrator.load(calibration) if calibration else None
    which = {} if model is None else {"model": model}
    label_backend = BACKENDS[backend](device=device, max_tokens=max_tokens, batch_size=batch_size, **which)
    scorer = LetterScorer(label_backend, shots=DEFAULT_SHOTS if shots is None else shots)
    name = f"davout-{label_backend.name}" + ("" if model is None else f"@{model}")
    return Engine(scorer, calibrator, model_name=name)
