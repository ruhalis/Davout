"""Training rows: read the JSONL contract, render prompts as serving does, pack batches.

A data directory holds `train.jsonl`, `dev_in.jsonl` and `dev_xfer.jsonl`. Each line is::

    {"id": str, "source": str, "family": "nli"|"reading"|"judgement"|"ovr"|"candidate"|"choice"|"score",
     "state": <string | object>, "question": {"type", "instructions", "criteria"}, "label": int}

plus `"candidate": "<option name>"` on candidate rows. Labels: noul 1 = true, 0 = false;
choice the gold option index; score the level index; candidate rows are choice questions
with label 1 = the proposed option is correct.

The target is a letter index: Yes/No prompts list `A. Yes`, `B. No`, so a true noul (or a
correct candidate) is index 0 and a false one index 1; choice and score use the label as is.
Nothing here imports torch until `collate` is called.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from davout.backends.base import LETTERS, Prompt, PromptTooLongError
from davout.prompts import build_candidate_prompt, build_prompt
from davout.schema import ChoiceQuestion, NoulQuestion, Question, ValidationError, parse_question

NOUL_FAMILIES = ("nli", "reading", "judgement", "ovr")
BINARY_FAMILIES = NOUL_FAMILIES + ("candidate",)  # Yes/No prompts: AUROC applies
FAMILIES = BINARY_FAMILIES + ("choice", "score")
_FAMILY_TYPE = {**{f: "noul" for f in NOUL_FAMILIES}, "candidate": "choice", "choice": "choice", "score": "score"}
_KEYS = {"id", "source", "family", "state", "question", "label", "candidate"}
_ENCODE_CHUNK = 512


@dataclass(frozen=True)
class Row:
    """One labelled training decision, as written by the data builder."""

    id: str
    source: str
    family: str
    state: Any
    question: Question
    label: int
    candidate: str | None = None


@dataclass(frozen=True)
class Item:
    """A tokenised row: prompt ids (ending in `<|im_end|>`), option count and target letter index."""

    row: Row
    ids: np.ndarray  # int32
    n_opt: int
    y: int

    def __len__(self) -> int:
        return len(self.ids)


def _parse_row(obj: Any) -> Row:
    if not isinstance(obj, dict):
        raise ValueError("each line must be a JSON object")
    unknown = sorted(k for k in obj if k not in _KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown}")
    for key in ("id", "source", "family", "state", "question", "label"):
        if key not in obj:
            raise ValueError(f"{key} is required")
    for key in ("id", "source"):
        if not isinstance(obj[key], str) or not obj[key]:
            raise ValueError(f"{key} must be a non-empty string")
    family = obj["family"]
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if not isinstance(obj["state"], (str, dict, list)):
        raise ValueError("state must be a string, object or array")
    try:
        question = parse_question("question", obj["question"])
    except ValidationError as e:
        raise ValueError(f"{e.path.replace('questions.question', 'question', 1)}: {e.message}") from None
    if question.type != _FAMILY_TYPE[family]:
        raise ValueError(f"family {family!r} needs a {_FAMILY_TYPE[family]} question, got {question.type!r}")
    label = obj["label"]
    if isinstance(label, bool) or not isinstance(label, int):
        raise ValueError(f"label must be an integer, got {label!r}")
    candidate = obj.get("candidate")
    if family == "candidate":
        if not isinstance(candidate, str) or candidate not in question.criteria:
            raise ValueError(f"candidate must name an option of the question, got {candidate!r}")
    elif "candidate" in obj:
        raise ValueError("candidate only applies to rows of family 'candidate'")
    row = Row(obj["id"], obj["source"], family, obj["state"], question, label, candidate)
    target(row)  # validates the label range
    return row


def read_rows(path: str | Path) -> list[Row]:
    """Rows of a training JSONL file, in file order; errors name the line."""
    path = Path(path)
    rows: list[Row] = []
    seen: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                obj = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{where}: invalid JSON ({e})") from None
            try:
                row = _parse_row(obj)
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None
            if row.id in seen:
                raise ValueError(f"{where}: duplicate id {row.id!r} (first used on line {seen[row.id]})")
            seen[row.id] = lineno
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def target(row: Row) -> tuple[int, int]:
    """(target letter index, number of options) for a row."""
    q = row.question
    if isinstance(q, NoulQuestion) or row.candidate is not None:
        if row.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1 for a Yes/No row, got {row.label}")
        return (0 if row.label == 1 else 1), 2
    n = len(q.criteria)
    if n > len(LETTERS):
        raise ValueError(f"{n} options do not fit the {len(LETTERS)} answer letters")
    if not 0 <= row.label < n:
        raise ValueError(f"label {row.label} out of range for {n} options")
    return row.label, n


def render(row: Row) -> Prompt:
    """The zero-shot prompt serving would build for this row."""
    if row.candidate is not None:
        assert isinstance(row.question, ChoiceQuestion)
        return build_candidate_prompt(row.state, row.question, row.candidate)
    return build_prompt(row.state, row.question)


def tokenize(rows: Sequence[Row], backend: Any) -> tuple[list[Item], list[str]]:
    """Tokenise rows exactly as serving does (`HrmBackend._encode`), in row order.

    Returns (items, ids of dropped rows). A row is dropped when its prompt exceeds the
    backend's `max_tokens`: training on a cut state would teach from partial evidence.
    """
    items: list[Item] = []
    dropped: list[str] = []
    for start in range(0, len(rows), _ENCODE_CHUNK):
        chunk = rows[start : start + _ENCODE_CHUNK]
        prompts = [render(r) for r in chunk]
        try:
            encoded = backend._encode(prompts)
        except PromptTooLongError:  # one prompt cannot fit even without its state
            encoded = []
            for p in prompts:
                try:
                    encoded.extend(backend._encode([p]))
                except PromptTooLongError:
                    encoded.append(([], True))
        for row, prompt, (ids, truncated) in zip(chunk, prompts, encoded):
            if truncated:
                dropped.append(row.id)
                continue
            y, n_opt = target(row)
            if n_opt != prompt.n_labels or not 0 <= y < n_opt:
                raise ValueError(f"row {row.id!r}: target {y} does not fit {prompt.n_labels} options")
            items.append(Item(row, np.asarray(ids, dtype=np.int32), n_opt, y))
    return items, dropped


def chunks(items: Sequence[Item], k: int) -> Iterator[Sequence[Item]]:
    """Consecutive groups of exactly `k` items, in order; a short tail is left out."""
    for i in range(0, len(items) - len(items) % k, k):
        yield items[i : i + k]


def pack(items_sorted: Sequence[Item], max_batch_tokens: int) -> list[list[Item]]:
    """Split items sorted by ascending length into micro-batches of rows x longest <= `max_batch_tokens`."""
    out: list[list[Item]] = []
    cur: list[Item] = []
    for item in items_sorted:
        if cur and (len(cur) + 1) * len(item) > max_batch_tokens:  # `item` is the longest so far
            out.append(cur)
            cur = []
        cur.append(item)
    if cur:
        out.append(cur)
    return out


def micro_batches(items: Sequence[Item], max_batch_tokens: int) -> list[list[Item]]:
    """Length-sorted micro-batches of one optimizer step's examples."""
    return pack(sorted(items, key=len), max_batch_tokens)


def collate(items: Sequence[Item], pad_id: int, device: Any = None) -> tuple[Any, Any, Any, Any]:
    """Left-padded `ids`, `mask` (1 = real token), `n_opt` and `y` tensors.

    Left padding puts the answer position last in every row. The mask doubles as
    `token_type_ids`: without it the model silently runs causal attention.
    """
    import torch

    width = max(len(it) for it in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for r, it in enumerate(items):
        ids[r, width - len(it) :] = torch.from_numpy(np.asarray(it.ids, dtype=np.int64))
        mask[r, width - len(it) :] = 1
    n_opt = torch.tensor([it.n_opt for it in items], dtype=torch.long)
    y = torch.tensor([it.y for it in items], dtype=torch.long)
    if device is not None:
        ids, mask, n_opt, y = (t.to(device) for t in (ids, mask, n_opt, y))
    return ids, mask, n_opt, y
