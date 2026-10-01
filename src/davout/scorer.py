"""Turn typed questions into letter-token prompts and raw logits."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from davout.backends.base import LabelBackend, Prompt, Readout
from davout.prompts import Demo, build_candidate_prompt, build_prompt, generic_demos
from davout.schema import ChoiceQuestion, Question


@dataclass
class RawScore:
    """Uncalibrated result for one question."""

    # choice: one per option, in criteria order; score: one per level; noul: [true, false]
    logits: list[float]
    prompt_tokens: int  # total over all prompts used for this question
    truncated: bool
    # same shape per H cycle when the backend provides it (None for two-stage choice)
    cycle_logits: list[list[float]] | None
    stage: str  # "letter" or "two_stage"


def _logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _log_softmax(xs: Sequence[float]) -> list[float]:
    lse = _logsumexp(xs)
    return [x - lse for x in xs]


def _fit_demo(demo: Demo, j: int, n: int) -> Demo:
    """Shrink a choice demo to at most `n` options, keeping the gold one.

    Keeps a window of consecutive options; the gold position inside the window
    varies with the demo index `j` so the gold letters stay spread out.
    """
    q = demo.question
    if demo.candidate is not None or not isinstance(q, ChoiceQuestion) or len(q.criteria) <= n:
        return demo
    names = list(q.criteria)
    start = min(max(demo.answer - j % n, 0), len(names) - n)
    kept = names[start : start + n]
    return replace(
        demo,
        question=ChoiceQuestion(q.instructions, {k: q.criteria[k] for k in kept}),
        answer=demo.answer - start,
    )


def _as_candidate(demo: Demo, j: int) -> Demo:
    """Turn a choice demo into a Yes/No candidate demo (gold on even `j`, a wrong option on odd)."""
    q = demo.question
    if demo.candidate is not None or not isinstance(q, ChoiceQuestion):
        return demo
    names = list(q.criteria)
    if j % 2 == 0:
        return replace(demo, candidate=names[demo.answer], answer=0)
    return replace(demo, candidate=names[(demo.answer + 1) % len(names)], answer=1)


class LetterScorer:
    """Scores questions by reading answer-letter logits from a `LabelBackend`.

    Choice questions with more options than the backend has letters are scored
    in two stages: every option is judged independently (Yes/No), then the
    best `shortlist` options are compared in one explicit choice.
    """

    def __init__(self, backend: LabelBackend, shots: int = 3, shortlist: int = 10) -> None:
        if shots < 0:
            raise ValueError("shots must be >= 0")
        if shortlist < 2:
            raise ValueError("shortlist must be >= 2")
        self.backend = backend
        self.shots = shots
        self.shortlist = shortlist

    def info(self) -> dict[str, Any]:
        """Backend details plus this scorer's prompting settings."""
        return {**self.backend.info(), "scorer": "letter", "shots": self.shots, "shortlist": self.shortlist}

    def _demos(self, kind: str, override: Sequence[Demo] | None, n: int) -> list[Demo]:
        if override is None:
            return generic_demos(kind, self.shots)
        if kind == "candidate":
            return [_as_candidate(d, j) for j, d in enumerate(override)]
        return [_fit_demo(d, j, n) for j, d in enumerate(override)]

    def score(
        self,
        state: Any,
        questions: Sequence[Question],
        demos: Mapping[int, Sequence[Demo]] | None = None,
    ) -> list[RawScore]:
        """Score every question against `state`; `demos[i]` replaces the built-in demos of question i."""
        demos = demos or {}
        max_labels = self.backend.max_labels
        two_stage = [isinstance(q, ChoiceQuestion) and len(q.criteria) > max_labels for q in questions]

        # Pass 1: one letter prompt per question, or one Yes/No prompt per option (two-stage).
        prompts: list[Prompt] = []
        spans: list[tuple[int, int]] = []  # per question: its slice of `prompts`
        for i, q in enumerate(questions):
            start = len(prompts)
            if two_stage[i]:
                shots = self._demos("candidate", demos.get(i), 2)
                prompts.extend(build_candidate_prompt(state, q, name, shots) for name in q.criteria)
            else:
                prompts.append(build_prompt(state, q, self._demos(q.type, demos.get(i), max_labels)))
            spans.append((start, len(prompts)))
        first = self._read(prompts)

        results: list[RawScore | None] = [None] * len(questions)
        pending: list[tuple[int, list[int], list[float], list[Readout]]] = []
        stage2: list[Prompt] = []
        for i, (q, (a, b)) in enumerate(zip(questions, spans)):
            outs = first[a:b]
            if not two_stage[i]:
                r = outs[0]
                results[i] = RawScore(list(r.logits), r.prompt_tokens, r.truncated, r.cycle_logits, "letter")
                continue
            names = list(q.criteria)
            log_p1 = _log_softmax([r.logits[0] - r.logits[1] for r in outs])
            k = min(self.shortlist, max_labels, len(names))
            keep = sorted(sorted(range(len(names)), key=lambda j: (-log_p1[j], j))[:k])
            short = ChoiceQuestion(q.instructions, {names[j]: q.criteria[names[j]] for j in keep})
            stage2.append(build_prompt(state, short, self._demos("choice", demos.get(i), k)))
            pending.append((i, keep, log_p1, outs))

        # Pass 2: an explicit choice among each shortlist.
        for (i, keep, log_p1, outs), r2 in zip(pending, self._read(stage2)):
            log_s = _logsumexp([log_p1[j] for j in keep])
            logits = list(log_p1)
            for j, lp in zip(keep, _log_softmax(r2.logits)):
                logits[j] = log_s + lp
            results[i] = RawScore(
                logits,
                sum(r.prompt_tokens for r in outs) + r2.prompt_tokens,
                any(r.truncated for r in outs) or r2.truncated,
                None,
                "two_stage",
            )
        return results  # type: ignore[return-value]

    def _read(self, prompts: Sequence[Prompt]) -> list[Readout]:
        if not prompts:
            return []
        outs = self.backend.read(prompts)
        if len(outs) != len(prompts):
            raise RuntimeError(f"backend returned {len(outs)} readouts for {len(prompts)} prompts")
        for p, r in zip(prompts, outs):
            if len(r.logits) != p.n_labels:
                raise RuntimeError(f"backend returned {len(r.logits)} logits for {p.n_labels} labels")
        return outs
