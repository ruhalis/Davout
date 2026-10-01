"""Baseline scorer: an NLI classifier (OpenJev) behind a local HTTP sidecar.

Every question becomes one or more `Premise: ...\\nHypothesis: ...` texts for
`POST {url}/classify`, which returns raw logits in the order contradiction,
entailment, neutral. The premise is the state; the hypotheses are built from
the question, so this arm consumes the same (state, question) inputs as the
letter scorer.
"""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence

from davout.schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, to_text
from davout.scorer import RawScore

DEFAULT_URL = "http://127.0.0.1:8765"
MARKER = "\nHypothesis: "
SAFE_MARKER = "\nHypothesis - "  # what a marker quoted inside the premise is rewritten to
MAX_TEXTS = 64  # texts per /classify request
MAX_TEXT_CHARS = 32000  # characters per text
MAX_BODY_BYTES = 900_000  # the sidecar refuses request bodies over 1 MiB

_CONTRADICTION, _ENTAILMENT, _NEUTRAL = 0, 1, 2
_ELLIPSIS = "… "


def _inline(value: Any) -> str:
    """Render content on one line (compact JSON for objects and arrays)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def hypotheses(question: Question) -> list[str]:
    """The NLI hypotheses for a question: one for noul, one per option or level otherwise."""
    ins = _inline(question.instructions)
    if isinstance(question, NoulQuestion):
        if ins.endswith("?"):
            hyp = f'The answer to the question "{ins}" is yes.'
        else:
            hyp = ins if ins.endswith((".", "!")) else ins + "."
        true_text = _inline((question.criteria or {}).get("true"))
        return [f"{hyp} ({true_text})" if true_text else hyp]
    if isinstance(question, ChoiceQuestion):
        out = []
        for name, desc in question.criteria.items():
            hyp = f'Regarding "{ins}", the correct answer is "{_inline(name)}"'
            out.append(hyp + (f": {_inline(desc)}" if desc is not None else "") + ".")
        return out
    if isinstance(question, ScoreQuestion):
        return [f'Regarding "{ins}", the best description is: {_inline(lvl)}' for lvl in question.criteria]
    raise TypeError(f"unsupported question type {type(question).__name__}")


def render(premise: str, hypothesis: str) -> tuple[str, bool]:
    """Build one /classify text; returns (text, premise_was_cut).

    A marker quoted in the premise is rewritten so the server's split at the
    last marker stays correct. A premise too long for the per-text character
    limit is cut from the left, like the letter backend cuts the state.
    """
    premise = premise.replace(MARKER, SAFE_MARKER)
    head = "Premise: "
    budget = MAX_TEXT_CHARS - len(head) - len(MARKER) - len(hypothesis)
    cut = len(premise) > budget
    if cut:
        keep = budget - len(_ELLIPSIS)
        if keep < 0:
            raise ValueError(f"hypothesis does not fit in {MAX_TEXT_CHARS} characters")
        premise = _ELLIPSIS + premise[len(premise) - keep :] if keep else ""
    return f"{head}{premise}{MARKER}{hypothesis}", cut


class NliScorer:
    """Scores questions with the OpenJev NLI sidecar; same interface as `LetterScorer`."""

    def __init__(self, url: str = DEFAULT_URL, api_key: str | None = None, timeout: float = 300.0) -> None:
        """`api_key` defaults to $OPENJEV_API_KEY; `timeout` is per HTTP request, in seconds."""
        self.url = url.rstrip("/")
        self.api_key = api_key if api_key is not None else (os.environ.get("OPENJEV_API_KEY") or None)
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # -- HTTP ---------------------------------------------------------------------------

    def _request(self, path: str, body: Any = None) -> Any:
        """One JSON request; raises RuntimeError with the server's error text on any failure."""
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url + path, data=data, headers=headers)
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(detail)
                if isinstance(parsed, dict) and "error" in parsed:
                    detail = str(parsed["error"])
            except ValueError:
                pass
            raise RuntimeError(f"openjev {path} failed (HTTP {e.code}): {detail.strip() or e.reason}") from None
        except OSError as e:  # URLError, timeouts, refused connections
            reason = getattr(e, "reason", None) or e
            raise RuntimeError(
                f"cannot reach openjev at {self.url}{path}: {reason} (is the sidecar running?)"
            ) from None
        except ValueError as e:
            raise RuntimeError(f"openjev {path} returned invalid JSON: {e}") from None

    def _classify(self, texts: Sequence[str]) -> list[tuple[list[float], int]]:
        """(3 raw logits, prompt_tokens) per text, chunked by text count and body size."""
        out: list[tuple[list[float], int]] = []
        chunk: list[str] = []
        size = 0
        chunks: list[list[str]] = []
        for text in texts:
            n = len(json.dumps(text, ensure_ascii=False).encode("utf-8")) + 2
            if chunk and (len(chunk) >= MAX_TEXTS or size + n > MAX_BODY_BYTES):
                chunks.append(chunk)
                chunk, size = [], 0
            chunk.append(text)
            size += n
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            reply = self._request("/classify", {"text": chunk})
            if not isinstance(reply, list) or len(reply) != len(chunk):
                raise RuntimeError(f"openjev /classify returned an unexpected reply for {len(chunk)} texts")
            for item in reply:
                try:
                    logits = [float(x) for x in item["embedding"]]
                    tokens = int(item.get("meta_info", {}).get("prompt_tokens", 0))
                except (TypeError, KeyError, ValueError, AttributeError):
                    raise RuntimeError(f"openjev /classify returned a malformed item: {item!r}") from None
                if len(logits) != 3 or not all(math.isfinite(x) for x in logits):
                    raise RuntimeError(f"openjev /classify returned bad logits: {logits!r}")
                out.append((logits, tokens))
        return out

    # -- scorer interface ---------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Scorer name and URL, plus the sidecar's /health fields when it is reachable."""
        out: dict[str, Any] = {}
        try:
            health = self._request("/health")
            if isinstance(health, dict):
                out.update(health)
        except RuntimeError as e:
            out["health_error"] = str(e)
        out.update({"name": "openjev-nli", "url": self.url})
        return out

    def score(
        self,
        state: Any,
        questions: Sequence[Question],
        demos: Mapping[int, Sequence[Any]] | None = None,
    ) -> list[RawScore]:
        """Score every question against `state`. `demos` is accepted and ignored."""
        premise = to_text(state)
        texts: list[str] = []
        spans: list[tuple[int, int]] = []
        cut: list[bool] = []
        for q in questions:
            start = len(texts)
            was_cut = False
            for hyp in hypotheses(q):
                text, c = render(premise, hyp)
                texts.append(text)
                was_cut = was_cut or c
            spans.append((start, len(texts)))
            cut.append(was_cut)
        replies = self._classify(texts) if texts else []

        results: list[RawScore] = []
        for q, (a, b), was_cut in zip(questions, spans, cut):
            part = replies[a:b]
            if isinstance(q, NoulQuestion):
                nli = part[0][0]
                logits = [nli[_ENTAILMENT], _logsumexp([nli[_CONTRADICTION], nli[_NEUTRAL]])]
            else:  # log P(entailment) of each option's or level's hypothesis
                logits = [nli[_ENTAILMENT] - _logsumexp(nli) for nli, _ in part]
            results.append(RawScore(logits, sum(t for _, t in part), was_cut, None, "nli"))
        return results
