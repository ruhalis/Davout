"""Teacher scorer: a large chat model read the way Davout reads HRM, for evaluation and soft labels.

One prompt per decision: the state, the question and its lettered options go into a single user
turn of the model's chat template (thinking off); nothing is generated. The next-token logits
over the option letters are soft-maxed. Every decision is scored under two option orders, the
given one and a seeded shuffle; the two distributions are mapped back to the given order and
averaged, and `agree` records whether both orders picked the same option.

Works for choice questions of up to 26 options, score questions (levels are the options) and
noul questions (Yes / No). The model needs the optional `teacher` dependencies
(`uv sync --extra teacher`); the prompt and order helpers need nothing.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from davout.backends.base import LETTERS
from davout.prompts import build_prompt
from davout.schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, parse_question

MODEL_ID = "Qwen/Qwen3.6-27B"
SYSTEM = "You are a careful annotator. Read the text and the question, then answer with the letter of the one correct option."
REPLY = "Reply with the letter only."
_ANSWER = "\nAnswer:"


def n_options(question: Question) -> int:
    return 2 if isinstance(question, NoulQuestion) else len(question.criteria)


def reorder(question: Question, perm: Sequence[int]) -> Question:
    """The question with its options in the order `perm`: new option j is given option perm[j]."""
    if sorted(perm) != list(range(n_options(question))):
        raise ValueError(f"{list(perm)} is not a permutation of the {n_options(question)} options")
    if isinstance(question, ChoiceQuestion):
        items = list(question.criteria.items())
        return replace(question, criteria=dict(items[i] for i in perm))
    if isinstance(question, ScoreQuestion):
        return replace(question, criteria=[question.criteria[i] for i in perm])
    raise TypeError("only choice and score questions can be reordered")


def orders(question: Question, key: str, seed: int = 0) -> list[list[int]]:
    """The option orders a decision is scored under: the given one and a seeded shuffle that differs from it.

    Noul questions (Yes / No is part of the prompt layout) get the given order only.
    """
    n = n_options(question)
    given = list(range(n))
    if isinstance(question, NoulQuestion):
        return [given]
    rng = random.Random(f"{seed}:{key}")
    perm = given[:]
    while perm == given:
        rng.shuffle(perm)
    return [given, perm]


def to_given_order(probs: Sequence[float], perm: Sequence[int]) -> list[float]:
    """Probabilities read under `perm` (position j held given option perm[j]) back in the given order."""
    out = [0.0] * len(perm)
    for j, i in enumerate(perm):
        out[i] = float(probs[j])
    return out


def combine(per_order: Sequence[Sequence[float]]) -> tuple[list[float], bool]:
    """(mean distribution, whether every order has the same top option); inputs are in the given order."""
    n = len(per_order[0])
    mean = [sum(p[i] for p in per_order) / len(per_order) for i in range(n)]
    tops = {max(range(n), key=lambda i: (p[i], -i)) for p in per_order}
    return mean, len(tops) == 1


def softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    z = [math.exp(x - m) for x in logits]
    s = sum(z)
    return [x / s for x in z]


def messages(state: Any, question: Question) -> list[dict[str, str]]:
    """Chat messages for one decision: Davout's question block, minus its "Answer:" cue, in one user turn."""
    text = build_prompt(state, question).text
    if not text.endswith(_ANSWER):
        raise ValueError("unexpected prompt layout")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text[: -len(_ANSWER)] + "\n\n" + REPLY}]


class TeacherBackend:
    """A causal chat model loaded in 4-bit; `read` returns next-token logits over the option letters."""

    def __init__(self, model_id_or_path: str = MODEL_ID, batch_size: int = 16, max_batch_tokens: int = 8192,
                 four_bit: bool = True, model: Any = None, tokenizer: Any = None) -> None:  # fmt: skip
        import torch

        self._torch = torch
        self.model_id, self.batch_size, self.max_batch_tokens = model_id_or_path, batch_size, max_batch_tokens
        if model is None:
            from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

            quant = None
            if four_bit:
                quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                           bnb_4bit_compute_dtype=torch.bfloat16)  # fmt: skip
            tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id_or_path, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0}
            )
        self._model, self._tok = model.eval(), tokenizer
        self.letter_ids = []
        for letter in LETTERS:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"letter {letter!r} is not a single token")
            self.letter_ids.append(ids[0])
        self._pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def encode(self, msgs: list[dict[str, str]]) -> list[int]:
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self._tok.encode(text, add_special_tokens=False)

    def read(self, prompts: Sequence[list[dict[str, str]]], n_labels: Sequence[int]) -> list[list[float]]:
        """Letter logits (the first `n_labels[i]` letters) at the first answer position of each prompt."""
        torch = self._torch
        rows = [self.encode(m) for m in prompts]
        order = sorted(range(len(rows)), key=lambda i: -len(rows[i]))
        out: list[list[float] | None] = [None] * len(rows)
        device = next(self._model.parameters()).device
        letters = torch.tensor(self.letter_ids, device=device)
        start = 0
        while start < len(order):
            width = len(rows[order[start]])
            size = max(1, min(self.batch_size, self.max_batch_tokens // width, len(order) - start))
            batch = order[start : start + size]
            start += size
            ids = torch.full((len(batch), width), self._pad, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for r, i in enumerate(batch):  # left padding: the answer position is last in every row
                ids[r, width - len(rows[i]) :] = torch.tensor(rows[i], dtype=torch.long)
                mask[r, width - len(rows[i]) :] = 1
            with torch.inference_mode():
                # the body only, then the head on the last position: full logits would be rows x tokens x vocabulary
                hidden = self._model.model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state[:, -1]
                picked = self._model.lm_head(hidden).float()[:, letters].cpu()
            if not torch.isfinite(picked).all():
                raise RuntimeError("model produced non-finite logits")
            for r, i in enumerate(batch):
                out[i] = picked[r, : n_labels[i]].tolist()
        return out  # type: ignore[return-value]


    def generate(self, prompts: Sequence[list[dict[str, str]]], max_new_tokens: int, temperature: float = 0.9, top_p: float = 0.95) -> list[str]:
        """Sampled continuations (thinking off), one text per prompt; used to write synthetic data."""
        torch = self._torch
        rows = [self.encode(m) for m in prompts]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self._pad, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for r, row in enumerate(rows):
            ids[r, width - len(row) :] = torch.tensor(row, dtype=torch.long)
            mask[r, width - len(row) :] = 1
        device = next(self._model.parameters()).device
        with torch.inference_mode():
            out = self._model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=max_new_tokens,
                                       do_sample=True, temperature=temperature, top_p=top_p, pad_token_id=self._pad)  # fmt: skip
        return self._tok.batch_decode(out[:, width:], skip_special_tokens=True)


def score(backend: Any, decisions: Sequence[tuple[str, Any, Question]], seed: int = 0) -> list[dict[str, Any]]:
    """Score (id, state, question) decisions under both option orders; one result per decision.

    Each result has `probs` (mean over the orders, given order), `agree`, and `per_order`
    (the distribution of each order, mapped back to the given order).
    """
    prompts: list[list[dict[str, str]]] = []
    n_labels: list[int] = []
    plan: list[list[list[int]]] = []
    for key, state, question in decisions:
        n = n_options(question)
        if not 2 <= n <= len(LETTERS):
            raise ValueError(f"{key}: {n} options do not fit the {len(LETTERS)} answer letters")
        perms = orders(question, key, seed)
        plan.append(perms)
        for perm in perms:
            q = question if perm == list(range(n)) else reorder(question, perm)
            prompts.append(messages(state, q))
            n_labels.append(n)
    logits = backend.read(prompts, n_labels)
    results: list[dict[str, Any]] = []
    pos = 0
    for perms in plan:
        per_order = [to_given_order(softmax(logits[pos + k]), perm) for k, perm in enumerate(perms)]
        pos += len(perms)
        probs, agree = combine(per_order)
        results.append({"probs": probs, "agree": agree, "per_order": per_order})
    return results


def score_file(backend: Any, in_path: str | Path, out_path: str | Path, seed: int = 0, chunk: int = 256) -> dict[str, Any]:
    """Score a JSONL of decisions (`id`, `state`, `question`, optional `label`); writes one JSON line per decision.

    Rows already in `out_path` are skipped, so an interrupted run resumes. Returns counts and timing.
    """
    rows = [json.loads(line) for line in Path(in_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [r for r in rows if r["id"] not in done]
    t0 = time.perf_counter()
    passes = 0
    with out_path.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), chunk):
            part = todo[start : start + chunk]
            results = score(backend, [(r["id"], r["state"], parse_question(r["id"], r["question"])) for r in part], seed)
            for r, res in zip(part, results):
                passes += len(res["per_order"])
                f.write(json.dumps({"id": r["id"], "label": r.get("label"), **res}) + "\n")
            f.flush()
            rate = (start + len(part)) / (time.perf_counter() - t0)
            print(f"[teacher] {start + len(part)}/{len(todo)} decisions, {rate:.2f} decisions/s", file=sys.stderr, flush=True)
    seconds = time.perf_counter() - t0
    return {"decisions": len(todo), "skipped": len(done), "forward_passes": passes, "seconds": round(seconds, 1),
            "decisions_per_s": round(len(todo) / seconds, 3) if todo else None}  # fmt: skip
