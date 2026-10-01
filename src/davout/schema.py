"""Request schema and validation for typed-decision requests."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar

MAX_CHOICE_OPTIONS = 255
MIN_OPTIONS = 2
MAX_SCORE_LEVELS = 10


class ValidationError(ValueError):
    """Raised when a request body is invalid; `path` locates the problem."""

    def __init__(self, message: str, path: str = "") -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.message = message
        self.path = path


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: Any
    criteria: dict[str, Any]
    type: ClassVar[str] = "choice"


@dataclass(frozen=True)
class ScoreQuestion:
    instructions: Any
    criteria: list[Any]
    type: ClassVar[str] = "score"


@dataclass(frozen=True)
class NoulQuestion:
    instructions: Any
    criteria: dict[str, Any] | None
    type: ClassVar[str] = "noul"


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


@dataclass(frozen=True)
class Request:
    state: Any
    model: str
    questions: dict[str, Question]


def _is_content(v: Any) -> bool:
    return isinstance(v, (str, dict, list))


def _check_keys(obj: dict[str, Any], allowed: set[str], path: str) -> None:
    for k in obj:
        if k not in allowed:
            raise ValidationError(f"unknown key {k!r}", f"{path}.{k}" if path else k)


def _parse_question(qid: str, q: Any) -> Question:
    base = f"questions.{qid}"
    if not isinstance(q, dict):
        raise ValidationError("question must be an object", base)
    _check_keys(q, {"type", "instructions", "criteria"}, base)
    qtype = q.get("type")
    if qtype not in ("choice", "score", "noul"):
        raise ValidationError("type must be one of 'choice', 'score', 'noul'", f"{base}.type")
    if "instructions" not in q:
        raise ValidationError("instructions is required", f"{base}.instructions")
    ins = q["instructions"]
    if not _is_content(ins):
        raise ValidationError("instructions must be a string, object or array", f"{base}.instructions")
    if (isinstance(ins, str) and not ins.strip()) or (isinstance(ins, (list, dict)) and not ins):
        raise ValidationError("instructions must not be empty", f"{base}.instructions")

    cpath = f"{base}.criteria"
    crit = q.get("criteria")
    if qtype == "choice":
        if not isinstance(crit, dict):
            raise ValidationError("criteria must be an object", cpath)
        if not MIN_OPTIONS <= len(crit) <= MAX_CHOICE_OPTIONS:
            raise ValidationError(
                f"criteria must have {MIN_OPTIONS}..{MAX_CHOICE_OPTIONS} options", cpath
            )
        for name, desc in crit.items():
            if not isinstance(name, str) or not name.strip():
                raise ValidationError("option names must be non-empty strings", cpath)
            if desc is not None and not _is_content(desc):
                raise ValidationError(
                    "option description must be a string, object, array or null", f"{cpath}.{name}"
                )
        return ChoiceQuestion(ins, dict(crit))
    if qtype == "score":
        if not isinstance(crit, list):
            raise ValidationError("criteria must be an array", cpath)
        if not MIN_OPTIONS <= len(crit) <= MAX_SCORE_LEVELS:
            raise ValidationError(
                f"criteria must have {MIN_OPTIONS}..{MAX_SCORE_LEVELS} levels", cpath
            )
        for i, lvl in enumerate(crit):
            if not _is_content(lvl):
                raise ValidationError("level must be a string, object or array", f"{cpath}[{i}]")
        return ScoreQuestion(ins, list(crit))
    if crit is None:
        return NoulQuestion(ins, None)
    if not isinstance(crit, dict):
        raise ValidationError("criteria must be an object or null", cpath)
    for k, v in crit.items():
        if k not in ("true", "false"):
            raise ValidationError("criteria keys must be 'true' and/or 'false'", f"{cpath}.{k}")
        if not _is_content(v):
            raise ValidationError(
                "criteria value must be a string, object or array", f"{cpath}.{k}"
            )
    if not crit:
        return NoulQuestion(ins, None)
    return NoulQuestion(ins, dict(crit))


def parse_question(qid: str, obj: Any) -> Question:
    """Validate one question object; errors are located at `questions.{qid}`."""
    return _parse_question(qid, obj)


def parse_request(body: Any) -> Request:
    """Validate a decoded JSON body and return a `Request`."""
    if not isinstance(body, dict):
        raise ValidationError("body must be an object", "")
    _check_keys(body, {"state", "model", "questions"}, "")
    if "state" not in body:
        raise ValidationError("state is required", "state")
    if not _is_content(body["state"]):
        raise ValidationError("state must be a string, object or array", "state")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValidationError("model must be a non-empty string", "model")
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise ValidationError("questions must be a non-empty object", "questions")
    questions: dict[str, Question] = {}
    for qid, q in qs.items():
        if not isinstance(qid, str) or not qid:
            raise ValidationError("question ids must be non-empty strings", "questions")
        questions[qid] = _parse_question(qid, q)
    return Request(body["state"], model, questions)


def to_text(value: Any) -> str:
    """Render a string/object/array/None as prompt text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)
