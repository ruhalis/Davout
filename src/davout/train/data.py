"""Build the fine-tuning data: typed decisions converted from public datasets.

`build(out_dir)` writes `train.jsonl`, `dev_in.jsonl`, `dev_xfer.jsonl` and `manifest.json`.
Each JSONL line is::

    {"id": str, "source": str, "family": "nli"|"reading"|"judgement"|"ovr"|"candidate"|"choice"|"score",
     "state": <string | object>, "question": {"type", "instructions", "criteria"}, "label": int}

plus `"candidate": "<option name>"` on candidate rows. Labels: noul 1 = true, 0 = false;
choice the index of the gold option in `criteria` order; score the level index; candidate
rows hold a choice question and label 1 when the proposed option is correct. The trainer
renders a row with `build_prompt(state, question)` (or `build_candidate_prompt`), so every
variation (state wrappers, wording, option order) is already in the row.

Sampling is deterministic for a given (seed, scale, sources). `datasets` is imported only
when a split is loaded.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from davout.backends.base import Prompt
from davout.prompts import build_candidate_prompt, build_prompt
from davout.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, parse_question
from davout.train import sources as S
from davout.train.sources import Label, LabelSet, Source, TextStyle

MAX_TOKENS = 512  # prompts longer than this are dropped
MAX_OPTIONS = 26  # one answer letter per option
FAMILIES = ("nli", "reading", "judgement", "ovr", "candidate", "choice", "score")
NOUL_FAMILIES = ("nli", "reading", "judgement", "ovr")
DEV_IN_TOTAL = 2000

# Label balance
NLI_TRUE_RATE = 0.475  # neutral counts as false
READING_TRUE_RATE = 0.5
OVR_TRUE_RATE = 0.30
CANDIDATE_TRUE_RATE = 0.25
XFER_CANDIDATE_TRUE_RATE = 1 / 3
JUDGEMENT_TRUE_RATE = {"paws": 0.45, "tweet_offensive": 0.35, "tweet_hate": 0.42, "tweet_irony": 0.50}
SIBLING_NEGATIVE_RATE = 0.5  # share of negatives drawn from the gold label's parent group

# Augmentation
P_STATEMENT = 0.55  # noul: bare statement; otherwise a question
P_CRITERIA = (("none", 0.55), ("both", 0.25), ("true", 0.15), ("false", 0.05))
P_NEGATED = 0.075  # one-vs-rest statements negated, label flipped
P_NLI_FRAMED = 0.2  # statement-form NLI that names the text ("According to the text, ...")
P_WRAPPER = (("bare", 0.50), ("labelled", 0.30), ("json", 0.15), ("quoted", 0.05))
P_PAIR_JSON = 0.4  # two-part states: JSON object, otherwise labelled lines
P_STRUCTURED_READING = 0.45  # question kept in the state; otherwise it moves into the instruction
P_K = (((2, 5), 0.55), ((6, 10), 0.25), ((11, 20), 0.12), ((MAX_OPTIONS, MAX_OPTIONS), 0.08))
P_DESCRIPTIONS = (("all", 0.40), ("none", 0.50), ("mixed", 0.10))
NAME_STYLES = ("raw", "spaced", "title")
P_NOTA = 0.08  # gold option removed, "other" is correct
P_OTHER_DISTRACTOR = 0.12  # "other" present but wrong, so its presence does not give the answer away
P_SCORE_LEVELS = (("native", 0.70), ("three", 0.20), ("two", 0.10))
P_TREC_FINE = 0.4  # share of TREC examples asked over the 50 fine labels
P_PUBMED_PREFIX = 0.3
NEWSGROUPS_MAX_CHARS = 1200
APP_REVIEW_MIN_CHARS = 40
MULTIRC_PER_PARAGRAPH = 2
XFER_CHOICE_OPTIONS = 10
SHORTLIST_OPTIONS = 10  # shortlist rows: as many options as the second stage of a large Choice compares
SHORTLIST_NEAR = 6  # of the nine wrong options, how many come from the gold label's nearest names
NAME_NEIGHBOURS = 15  # how many of the most similar label names count as hard negatives

# Characters per token assumed when the HRM tokenizer is not available. Measured on the
# 96,000 real train prompts: mean 3.44, 1st percentile 2.58, minimum 2.27 (manifest
# "length.chars_per_token"), so 2.2 over-counts every one of them.
PROXY_CHARS_PER_TOKEN = 2.2

_LINE_BREAKS = dict.fromkeys(map(ord, "  \x85\x00"), " ")
_WS = re.compile(r"\s+")
_U_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
_SAFE_LOWER = frozenset(  # sentence openers that are lower-case inside a sentence
    "the a an there it this that these those he she they we you some many most all no people nobody everyone "
    "someone somebody one two three four five in on at if when his her their my our your its every each both "
    "several few not none nothing something as for after before during with without to by from more less "
    "fewer over under about only since until while although because however also then now here today another "
    "other any such between among within around through into even just still never always often sometimes "
    "usually almost nearly everything anything everybody anyone men women children kids boys girls man woman "
    "humans scientists researchers students animals plants water is are was were has have had".split()
)
_AUX = frozenset("is are was were does do did can could will would should has have had may might".split())


class DataGuardError(AssertionError):
    """A build-time guard failed: a label mapping, a leak check or a format rule."""


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise DataGuardError(message)


# -- small helpers ----------------------------------------------------------------------


def _clean(text: Any) -> str:
    """Text as published, minus characters that would break a JSONL line, stripped."""
    return str(text).translate(_LINE_BREAKS).strip()


def _unescape(text: str) -> str:
    """Turn literal `\\uXXXX` sequences (tweet_eval publishes "\\u2019" for an apostrophe) into characters."""
    if "\\u" not in text:
        return text
    out = _U_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
    return out.encode("utf-16", "surrogatepass").decode("utf-16", "replace")  # joins surrogate pairs


def _key(*parts: str) -> str:
    """Content key: case and whitespace do not make two texts different."""
    return "\x1f".join(_WS.sub(" ", p).strip().lower() for p in parts)


def _pick(rng: random.Random, weighted: Sequence[tuple[Any, float]]) -> Any:
    r = rng.random()
    acc = 0.0
    for value, p in weighted:
        acc += p
        if r < acc:
            return value
    return weighted[-1][0]


def _title(name: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in name.split(" "))


def styled(label: Label, style: str) -> str:
    """Option name of a label in one of NAME_STYLES."""
    if style == "raw":
        return label.raw
    if style == "title":
        return _title(label.name)
    return label.name


def _lower_first(text: str) -> str:
    first = text.split(" ", 1)[0]
    if first.lower() in _SAFE_LOWER and first == first.capitalize():
        return text[:1].lower() + text[1:]
    return text


def _clause(sentence: str) -> str:
    """A sentence as a subordinate clause: no final punctuation, lower-case start where safe."""
    return _lower_first(sentence.strip().rstrip(".!… ").strip())


def _scaled(n: int, scale: float) -> int:
    return max(1, int(round(n * scale))) if n > 0 else 0


def _split_proportional(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Integers that sum to `total`, proportional to `weights` (largest remainder)."""
    s = sum(weights.values())
    if s <= 0 or total <= 0:
        return {k: 0 for k in weights}
    exact = {k: total * w / s for k, w in weights.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k in sorted(weights, key=lambda k: (-(exact[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


# -- state wrappers ---------------------------------------------------------------------


def wrap_text(text: str, style: TextStyle, rng: random.Random, extra: Mapping[str, str] | None = None) -> tuple[Any, str, str]:
    """Wrap a single text: (state, wrapper kind, noun that names the text in templates)."""
    kind = _pick(rng, P_WRAPPER)
    if kind == "labelled":
        label = rng.choice(style.labels)
        noun = label.lower() if " " not in label else style.noun
        return f"{label}: {text}", kind, noun
    if kind == "json":
        key = rng.choice(style.keys)
        return {**(extra or {}), key: text}, kind, key
    if kind == "quoted":
        return f'"{text}"', kind, style.noun
    return text, kind, style.noun


def wrap_pair(first: str, second: str, rng: random.Random, labels: Sequence[tuple[str, str]], keys: Sequence[tuple[str, str]]) -> tuple[Any, str]:
    """Wrap a two-part state as a JSON object or as two labelled lines."""
    if rng.random() < P_PAIR_JSON:
        k1, k2 = rng.choice(list(keys))
        return {k1: first, k2: second}, "pair_json"
    l1, l2 = rng.choice(list(labels))
    return f"{l1}: {first}\n{l2}: {second}", "pair_labelled"


def wrap_fields(values: Sequence[str], rng: random.Random, keys: Sequence[str], labels: Sequence[str]) -> tuple[Any, str]:
    """Wrap a state made of named parts as a JSON object or as labelled lines."""
    if rng.random() < P_PAIR_JSON:
        return dict(zip(keys, values)), "fields_json"
    return "\n".join(f"{label}: {value}" for label, value in zip(labels, values)), "fields_labelled"


def noul_criteria(rng: random.Random, true_texts: Sequence[str], false_texts: Sequence[str]) -> tuple[dict[str, str] | None, str]:
    """Criteria for a noul question: none, both, true-only or false-only."""
    mode = _pick(rng, P_CRITERIA)
    if mode == "none" or (not true_texts and not false_texts):
        return None, "none"
    if not true_texts:
        mode = "false"
    elif not false_texts:
        mode = "true"
    crit: dict[str, str] = {}
    if mode in ("both", "true"):
        crit["true"] = rng.choice(list(true_texts))
    if mode in ("both", "false"):
        crit["false"] = rng.choice(list(false_texts))
    return crit, mode


def _noul(instructions: str, criteria: dict[str, str] | None) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions, "criteria": criteria}


def _form(rng: random.Random) -> str:
    return "statement" if rng.random() < P_STATEMENT else "question"


# -- example makers ---------------------------------------------------------------------
#
# Each maker returns {"state", "question", "label", "meta"} (plus "candidate"), or None
# when the row cannot be phrased (the caller counts it as dropped).


def nli_example(premise: str, hypothesis: str, entailed: bool, style: TextStyle, rng: random.Random) -> dict[str, Any] | None:
    """State = premise; the hypothesis is the statement (or is turned into a question)."""
    if not premise or not hypothesis or hypothesis.endswith("?"):
        return None
    state, wrapper, noun = wrap_text(premise, style, rng)
    form = _form(rng)
    if form == "statement":
        if rng.random() < P_NLI_FRAMED:
            ins = rng.choice(S.NLI_FRAMED_STATEMENTS).format(noun=noun, h=_clause(hypothesis))
        else:
            ins = hypothesis
    else:
        ins = rng.choice(S.NLI_QUESTIONS).format(noun=noun, h=_clause(hypothesis), H=hypothesis)
    crit, mode = noul_criteria(
        rng, [t.format(noun=noun) for t in S.NLI_TRUE], [t.format(noun=noun) for t in S.NLI_FALSE]
    )
    meta = {"form": form, "criteria": mode, "wrapper": wrapper}
    return {"state": state, "question": _noul(ins, crit), "label": int(entailed), "meta": meta}


def reading_example(passage: str, question: str, answerable: bool, kind: str, style: TextStyle, rng: random.Random) -> dict[str, Any] | None:
    """Can `question` be answered from `passage`? `kind` is "squad_v2" (passage) or "qnli" (sentence)."""
    if not passage or not question:
        return None
    squad = kind == "squad_v2"
    form = _form(rng)
    if rng.random() < P_STRUCTURED_READING:
        part = "passage" if squad else "sentence"
        if rng.random() < P_PAIR_JSON:
            state: Any = {part: passage, "question": question}
            wrapper = "pair_json"
        else:
            state = f"{part.capitalize()}: {passage}\nQuestion: {question}"
            wrapper = "pair_labelled"
        if form == "statement":
            ins = rng.choice(S.SQUAD_STRUCT_STATEMENTS if squad else S.QNLI_STRUCT_STATEMENTS)
        else:
            ins = rng.choice(S.SQUAD_STRUCT_QUESTIONS if squad else S.QNLI_STRUCT_QUESTIONS)
    else:
        state, wrapper, noun = wrap_text(passage, style, rng)
        if form == "statement":
            templates = S.SQUAD_EMBED_STATEMENTS if squad else S.QNLI_EMBED_STATEMENTS
        else:
            templates = S.SQUAD_EMBED_QUESTIONS if squad else S.QNLI_EMBED_QUESTIONS
            if not question.endswith("?"):  # the "...this question: {q}" template must end in "?"
                templates = tuple(t for t in templates if not t.endswith("{q}"))
        ins = rng.choice(templates).format(noun=noun, q=question)
    crit, mode = noul_criteria(
        rng, S.SQUAD_TRUE if squad else S.QNLI_TRUE, S.SQUAD_FALSE if squad else S.QNLI_FALSE
    )
    meta = {"form": form, "criteria": mode, "wrapper": wrapper}
    return {"state": state, "question": _noul(ins, crit), "label": int(answerable), "meta": meta}


def pubmed_example(abstract: str, question: str, yes: bool, style: TextStyle, rng: random.Random) -> dict[str, Any] | None:
    """The dataset's yes/no question is the instruction; the abstract is the state."""
    if not abstract or not question.endswith("?"):
        return None
    state, wrapper, noun = wrap_text(abstract, style, rng)
    ins = question
    first = question.split(" ", 1)[0]
    if rng.random() < P_PUBMED_PREFIX and first.lower() in _AUX:
        ins = rng.choice(S.PUBMED_PREFIXES).format(noun=noun) + question[:1].lower() + question[1:]
    crit, mode = noul_criteria(rng, S.PUBMED_TRUE, S.PUBMED_FALSE)
    meta = {"form": "question", "criteria": mode, "wrapper": wrapper}
    return {"state": state, "question": _noul(ins, crit), "label": int(yes), "meta": meta}


def judgement_example(state: Any, wrapper: str, positive: bool, judgement: S.Judgement, rng: random.Random) -> dict[str, Any]:
    """A yes/no judgement about an already wrapped state (paraphrase, offensive, hateful, ironic)."""
    form = _form(rng)
    ins = rng.choice(judgement.statements if form == "statement" else judgement.questions)
    crit, mode = noul_criteria(rng, judgement.true, judgement.false)
    meta = {"form": form, "criteria": mode, "wrapper": wrapper}
    return {"state": state, "question": _noul(ins, crit), "label": int(positive), "meta": meta}


def draw_k(rng: random.Random, n_available: int) -> int:
    """Number of options: 2-5 (55%), 6-10 (25%), 11-20 (12%), 26 or all (8%)."""
    lo, hi = _pick(rng, P_K)
    return max(2, min(rng.randint(lo, hi), n_available, MAX_OPTIONS))


def _other_name(style: str, rng: random.Random) -> str:
    name = rng.choice(S.OTHER_NAMES)
    if style == "raw":
        return "other"
    return name[:1].upper() + name[1:] if style == "title" else name


def _criteria(real: Sequence[Label], style: str, has_desc: bool, other: str | None, rng: random.Random) -> tuple[dict[str, Any], str]:
    """Option map for `real` labels (already ordered) plus an optional "other" option."""
    mode = _pick(rng, P_DESCRIPTIONS) if has_desc else "none"

    def describe(desc: str | None) -> str | None:
        if mode == "all" or (mode == "mixed" and rng.random() < 0.5):
            return desc
        return None

    items: list[tuple[str, Any]] = [(styled(label, style), describe(label.desc)) for label in real]
    if other is not None:
        entry = (other, describe(rng.choice(S.OTHER_DESCS)))
        if "above" in other.lower() or rng.random() < 0.5:
            items.append(entry)
        else:
            items.insert(rng.randrange(len(items) + 1), entry)
    criteria = dict(items)
    _require(len(criteria) == len(items), f"option names collide: {[n for n, _ in items]}")
    return criteria, mode


def choice_example(state: Any, wrapper: str, gold: Label | None, labels: LabelSet, rng: random.Random) -> dict[str, Any]:
    """A choice over shuffled options; `gold=None` means the right answer is "other"."""
    style = rng.choice(NAME_STYLES)
    k = draw_k(rng, len(labels.labels))
    others = [label for label in labels.labels if label is not gold]
    nota = gold is None or (len(others) >= 1 and rng.random() < P_NOTA)
    if nota:
        real = rng.sample(others, min(k - 1, len(others)))
        other: str | None = _other_name(style, rng)
    else:
        real = [gold] + rng.sample(others, min(k - 1, len(others)))
        other = None
        if len(real) >= 3 and rng.random() < P_OTHER_DISTRACTOR:
            real.pop()  # "other" takes the place of one distractor
            other = _other_name(style, rng)
    rng.shuffle(real)
    criteria, desc_mode = _criteria(real, style, any(label.desc for label in labels.labels), other, rng)
    names = list(criteria)
    label = names.index(other) if nota else names.index(styled(gold, style))
    question = {"type": "choice", "instructions": rng.choice(labels.choice_instructions), "criteria": criteria}
    meta = {"k": len(names), "descriptions": desc_mode, "names": style, "nota": nota,
            "other": other is not None, "wrapper": wrapper}  # fmt: skip
    return {"state": state, "question": question, "label": label, "meta": meta}


def _trigrams(name: str) -> frozenset[str]:
    padded = f"  {name.lower()} "
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


@functools.lru_cache(maxsize=None)
def name_neighbours(labels: LabelSet) -> dict[str, tuple[Label, ...]]:
    """For each label (by `raw`), the NAME_NEIGHBOURS other labels with the most similar names.

    Similarity is the Jaccard overlap of character trigrams; ties keep the label order.
    """
    grams = [_trigrams(label.name) for label in labels.labels]
    out: dict[str, tuple[Label, ...]] = {}
    for i, label in enumerate(labels.labels):
        sim = [len(grams[i] & g) / len(grams[i] | g) for g in grams]
        order = sorted((j for j in range(len(grams)) if j != i), key=lambda j: (-sim[j], j))
        out[label.raw] = tuple(labels.labels[j] for j in order[:NAME_NEIGHBOURS])
    return out


def _wrong_label(golds: Sequence[Label], labels: LabelSet, rng: random.Random) -> tuple[Label, bool]:
    """A label that is not gold; half the time a hard one: from the gold label's parent group if
    there is one, or (label sets marked `hard`) one of the labels with the most similar names."""
    pool = [label for label in labels.labels if label not in golds]
    groups = {g.group for g in golds if g.group}
    siblings = [label for label in pool if label.group in groups]
    if labels.hard:
        near = {n.raw for g in golds for n in name_neighbours(labels)[g.raw]}
        siblings = [label for label in pool if label.raw in near]
    if siblings and rng.random() < SIBLING_NEGATIVE_RATE:
        return rng.choice(siblings), True
    return rng.choice(pool), False


def shortlist_example(state: Any, wrapper: str, gold: Label, labels: LabelSet, rng: random.Random,
                      mined: Sequence[str] | None = None) -> dict[str, Any]:  # fmt: skip
    """A choice shaped like the second stage of a large Choice: the gold option among nine wrong
    ones. `mined` lists wrong labels (`raw`) as a model's candidate stage ranked them, best first,
    and the first nine are used; without it SHORTLIST_NEAR come from the most similar names."""
    style = rng.choice(NAME_STYLES)
    if mined is not None:
        by_raw = labels.by_raw()
        wrong = [by_raw[raw] for raw in mined if raw != gold.raw][: SHORTLIST_OPTIONS - 1]
        _require(len(wrong) == SHORTLIST_OPTIONS - 1, f"{gold.raw}: fewer than {SHORTLIST_OPTIONS - 1} mined negatives")
        real = [gold] + wrong
    else:
        near = rng.sample(name_neighbours(labels)[gold.raw], SHORTLIST_NEAR)
        rest = [label for label in labels.labels if label is not gold and label not in near]
        real = [gold] + near + rng.sample(rest, SHORTLIST_OPTIONS - 1 - len(near))
    rng.shuffle(real)
    criteria, desc_mode = _criteria(real, style, any(label.desc for label in labels.labels), None, rng)
    question = {"type": "choice", "instructions": rng.choice(labels.choice_instructions), "criteria": criteria}
    meta = {"k": len(criteria), "descriptions": desc_mode, "names": style, "nota": False, "other": False,
            "shortlist": "mined" if mined is not None else "names", "wrapper": wrapper}  # fmt: skip
    return {"state": state, "question": question, "label": list(criteria).index(styled(gold, style)), "meta": meta}


def candidate_example(state: Any, wrapper: str, gold: Label | None, labels: LabelSet, rng: random.Random, want_true: bool, k: int | None = None) -> dict[str, Any]:
    """A choice question plus one proposed option; label 1 when the proposal is the gold option."""
    style = rng.choice(NAME_STYLES)
    k = k or draw_k(rng, len(labels.labels))
    sibling = False
    if want_true:
        cand = gold
    else:
        cand, sibling = _wrong_label([gold] if gold else [], labels, rng)
    chosen = [label for label in (gold, cand) if label is not None]
    chosen = list(dict.fromkeys(chosen))
    rest = [label for label in labels.labels if label not in chosen]
    other = _other_name(style, rng) if gold is None else None
    n_more = max(0, min(k - len(chosen) - (other is not None), len(rest)))
    real = chosen + rng.sample(rest, n_more)
    rng.shuffle(real)
    criteria, desc_mode = _criteria(real, style, any(label.desc for label in labels.labels), other, rng)
    candidate = other if cand is None else styled(cand, style)
    question = {"type": "choice", "instructions": rng.choice(labels.choice_instructions), "criteria": criteria}
    meta = {"k": len(criteria), "descriptions": desc_mode, "names": style, "sibling_negative": sibling,
            "wrapper": wrapper}  # fmt: skip
    return {"state": state, "question": question, "label": int(want_true), "candidate": candidate, "meta": meta}


def ovr_example(state: Any, wrapper: str, golds: Sequence[Label], labels: LabelSet, rng: random.Random, want_true: bool) -> dict[str, Any]:
    """One-vs-rest noul: is the text about one given label? `want_true` is the final label."""
    negated = bool(golds) and rng.random() < P_NEGATED
    match = want_true != negated
    _require(bool(golds) or not match, "a row without gold labels cannot match a label")
    sibling = False
    if match:
        asked = rng.choice(list(golds))
    else:
        asked, sibling = _wrong_label(golds, labels, rng)
    p = styled(asked, rng.choice(NAME_STYLES)) if labels.quote_names else (asked.phrase or asked.name)
    if negated:
        form = "statement"
        ins = rng.choice(asked.negations or labels.negations).format(p=p)
    else:
        form = _form(rng)
        if form == "statement":
            ins = rng.choice(asked.statements or labels.statements).format(p=p)
        else:
            ins = rng.choice(asked.questions or labels.questions).format(p=p)
    true_texts = [asked.desc] if asked.desc else [t.format(p=p) for t in labels.true_criteria]
    false_texts = list(labels.false_criteria)
    if negated:  # "not about X" is true when the text is about something else
        true_texts, false_texts = false_texts, true_texts
    crit, mode = noul_criteria(rng, true_texts, false_texts)
    meta = {"form": form, "criteria": mode, "negated": negated, "sibling_negative": sibling, "wrapper": wrapper}
    return {"state": state, "question": _noul(ins, crit), "label": int(want_true), "meta": meta}


def score_example(state: Any, wrapper: str, level: int, source: str, rng: random.Random) -> dict[str, Any]:
    """An ascending score question: native levels, or collapsed to three or two."""
    scale = rng.choice(S.SCALES[source])
    native = S.SCORE_NATIVE[source]
    _require(0 <= level < native, f"{source}: level {level} outside 0..{native - 1}")
    n = {"native": native, "three": 3, "two": 2}[_pick(rng, P_SCORE_LEVELS)]
    label = level
    if n != native:
        mapping = S.SCORE_COLLAPSE[source].get(n)
        if mapping is None or mapping[level] is None or n not in scale.levels:
            n = native  # the level has no place on the collapsed scale
        else:
            label = mapping[level]
    text = "terse" if rng.random() < 0.5 else "descriptive"
    question = {"type": "score", "instructions": rng.choice(scale.instructions), "criteria": list(scale.levels[n][text])}
    meta = {"levels": n, "collapsed": n != native, "text": text, "wrapper": wrapper}
    return {"state": state, "question": question, "label": label, "meta": meta}


# -- raw rows -> items ------------------------------------------------------------------
#
# `normalise` turns one dataset row into an item (a dict with "keys", the content keys used
# to keep splits disjoint) or returns a string naming why the row is not used.

_VITAMINC = {"SUPPORTS": "entailment", "REFUTES": "contradiction", "NOT ENOUGH INFO": "neutral"}
_SCITAIL = {"entails": "entailment", "neutral": "neutral"}


def stsb_level(score: float) -> int:
    """round(5 * score) with halves rounded up; scores are float32 multiples of 0.01-0.2."""
    return int(math.floor(round(5 * float(score), 4) + 0.5))


def civil_level(toxicity: float) -> int:
    """Bins 0, <= 0.3, <= 0.7, above (toxicity is a float32 share of annotators)."""
    t = float(toxicity)
    if t <= 0.0:
        return 0
    return 1 if t <= S.CIVIL_BINS[1] + 1e-6 else 2 if t <= S.CIVIL_BINS[2] + 1e-6 else 3


def _nli_item(premise: Any, hypothesis: Any, cls: str | None) -> dict[str, Any] | str:
    premise, hypothesis = _clean(premise), _clean(hypothesis)
    if cls is None:
        return "unlabelled"
    if not premise or not hypothesis:
        return "empty_text"
    return {"premise": premise, "hypothesis": hypothesis, "cls": cls, "keys": (_key(premise, hypothesis),)}


def _text_item(text: Any, **fields: Any) -> dict[str, Any] | str:
    text = _clean(text)
    if not text:
        return "empty_text"
    return {"text": text, "keys": (_key(text),), **fields}


def _class_name(value: Any, names: Sequence[str]) -> str | None:
    return names[value] if isinstance(value, int) and 0 <= value < len(names) else None


def normalise(source: str, row: Mapping[str, Any], rng: random.Random) -> dict[str, Any] | str:
    """One dataset row as an item, or the reason it is skipped."""
    if source in S.MINED_SOURCES:  # the rows of another source; "cls" (the intent) is what gets balanced
        item = normalise(S.MINED_SOURCES[source], row, rng)
        if isinstance(item, str):
            return item
        golds = next(iter(item["golds"].values()))
        return {**item, "cls": golds[0].raw} if len(golds) == 1 else "out_of_scope"
    if source in ("mnli", "anli"):
        return _nli_item(row["premise"], row["hypothesis"], _class_name(row["label"], S.NLI_NAMES))
    if source == "wanli":
        return _nli_item(row["premise"], row["hypothesis"], row["gold"] if row["gold"] in S.NLI_NAMES else None)
    if source == "vitaminc":
        return _nli_item(row["evidence"], row["claim"], _VITAMINC.get(row["label"]))
    if source == "scitail":
        return _nli_item(row["premise"], row["hypothesis"], _SCITAIL.get(row["label"]))
    if source == "rte":
        cls = {0: "entailment", 1: "not_entailment"}.get(row["label"])
        return _nli_item(row["premise"], row["hypothesis"], cls)
    if source == "squad_v2":
        passage, question = _clean(row["context"]), _clean(row["question"])
        if not passage or not question:
            return "empty_text"
        answerable = len(row["answers"]["text"]) > 0
        return {"passage": passage, "question": question, "cls": answerable, "keys": (_key(passage, question),)}
    if source == "pubmedqa":
        passage = " ".join(_clean(c) for c in row["context"]["contexts"])
        question = _clean(row["question"])
        if row["final_decision"] not in ("yes", "no"):
            return "unlabelled"
        if not passage or not question:
            return "empty_text"
        letters = [c for c in question if c.isalpha()]
        if sum(c.isupper() for c in letters) > 0.5 * len(letters):
            return "malformed_question"  # pqa_artificial turns some titles into "Do aNALYSIS OF ...?"
        return {"passage": passage, "question": question, "cls": row["final_decision"] == "yes",
                "keys": (_key(passage, question),)}  # fmt: skip
    if source == "qnli":
        passage, question = _clean(row["sentence"]), _clean(row["question"])
        if row["label"] not in (0, 1):
            return "unlabelled"
        if not passage or not question:
            return "empty_text"
        return {"passage": passage, "question": question, "cls": row["label"] == 0, "keys": (_key(passage, question),)}
    if source == "multirc":
        passage, question, answer = _clean(row["paragraph"]), _clean(row["question"]), _clean(row["answer"])
        if row["label"] not in (0, 1):
            return "unlabelled"
        if not passage or not question or not answer:
            return "empty_text"
        return {"passage": passage, "question": question, "answer": answer, "cls": row["label"] == 1,
                "limit": _key(passage), "keys": (_key(passage, question, answer),)}  # fmt: skip
    if source == "paws":
        a, b = _clean(row["sentence1"]), _clean(row["sentence2"])
        if row["label"] not in (0, 1):
            return "unlabelled"
        if not a or not b:
            return "empty_text"
        return {"a": a, "b": b, "cls": row["label"] == 1, "keys": (_key(a, b),)}
    if source in ("tweet_offensive", "tweet_hate", "tweet_irony"):
        if row["label"] not in (0, 1):
            return "unlabelled"
        return _text_item(_unescape(str(row["text"])), cls=row["label"] == 1)
    if source == "yahoo":
        title = _clean(str(row["question_title"]).replace("\\n", " "))
        content = _clean(str(row["question_content"]).replace("\\n", " "))
        gold = _class_name(row["topic"], [label.raw for label in S.YAHOO.labels])
        if gold is None:
            return "unlabelled"
        return _text_item(f"{title}\n{content}" if content else title, golds={"topic": [S.YAHOO.by_raw()[gold]]})
    if source == "newsgroups":
        gold = S.NEWSGROUPS.by_raw().get(row["label_text"])
        text = _clean(row["text"])[:NEWSGROUPS_MAX_CHARS].rstrip()
        if gold is None:
            return "unlabelled"
        if len(text) < 20:
            return "empty_text"
        return _text_item(text, golds={"group": [gold]})
    if source == "dbpedia":
        gold = _class_name(row["label"], [label.raw for label in S.DBPEDIA.labels])
        if gold is None:
            return "unlabelled"
        return _text_item(row["content"], golds={"type": [S.DBPEDIA.by_raw()[gold]]}, extra={"title": _clean(row["title"])})
    if source == "clinc":
        name = _class_name(row["intent"], S.CLINC_INTENTS)
        if name is None:
            return "unlabelled"
        if name in S.CLINC_EXCLUDED:
            return "excluded_intent"
        golds = [] if name == S.CLINC_OOS else [S.CLINC.by_raw()[name]]
        return _text_item(row["text"], golds={"intent": golds})
    if source in ("massive_intent", "massive_scenario"):
        setname, labels = ("intent", S.MASSIVE_INTENT) if source == "massive_intent" else ("scenario", S.MASSIVE_SCENARIO)
        gold = labels.by_raw().get(row["label_text"])
        if gold is None:
            return "unlabelled"
        item = _text_item(row["text"], golds={setname: [gold]}, uid=str(row["id"]))
        if isinstance(item, dict):
            item["keys"] = (f"massive-id:{row['id']}", *item["keys"])
        return item
    if source == "trec":
        coarse = S.TREC_COARSE.by_raw().get(row["label_coarse_text"])
        fine = S.TREC_FINE.by_raw().get(row["label_text"])
        if coarse is None or fine is None:
            return "unlabelled"
        return _text_item(row["text"], golds={"coarse": [coarse], "fine": [fine]})
    if source == "go_emotions":
        names = [label.raw for label in S.GO_EMOTIONS.labels]
        golds = [_class_name(i, names) for i in row["labels"]]
        if not golds or None in golds:
            return "unlabelled"
        return _text_item(row["text"], golds={"emotion": [S.GO_EMOTIONS.by_raw()[g] for g in golds]})
    if source == "emotion":
        gold = _class_name(row["label"], [label.raw for label in S.EMOTION.labels])
        if gold is None:
            return "unlabelled"
        return _text_item(row["text"], golds={"emotion": [S.EMOTION.by_raw()[gold]]})
    if source == "dailydialog":
        utterances, acts = row["utterances"], row["acts"]
        if not utterances or len(utterances) != len(acts):
            return "empty_text"
        j = rng.randrange(len(utterances))  # one utterance per dialogue
        gold = S.DAILYDIALOG.by_raw().get(str(acts[j]))
        if gold is None:
            return "unlabelled"
        text = _clean(utterances[j])
        if len(text) < 2:
            return "empty_text"
        return _text_item(text, golds={"act": [gold]}, sub=str(j))
    if source == "hwu64":
        gold = S.HWU64.by_raw().get(row["label"])
        if gold is None:
            return "unlabelled"
        return _text_item(row["text"], golds={"intent": [gold]})
    if source == "amazon_reviews":
        if row["label"] not in range(5):
            return "unlabelled"
        return _text_item(row["text"], level=int(row["label"]))
    if source == "tweet_sentiment":
        if row["label"] not in range(3):
            return "unlabelled"
        return _text_item(_unescape(str(row["text"])), level=int(row["label"]))
    if source == "civil_comments":
        t = row["toxicity"]
        if t is None or not 0.0 <= float(t) <= 1.0:
            return "unlabelled"
        return _text_item(row["text"], level=civil_level(t))
    if source == "stsb":
        a, b = _clean(row["sentence1"]), _clean(row["sentence2"])
        if row["score"] is None or not 0.0 <= float(row["score"]) <= 1.0:
            return "unlabelled"
        if not a or not b:
            return "empty_text"
        return {"a": a, "b": b, "level": stsb_level(row["score"]), "keys": (_key(a, b),)}
    if source == "app_reviews":
        if row["star"] not in range(1, 6):
            return "unlabelled"
        text = _clean(row["review"])
        if len(text) < APP_REVIEW_MIN_CHARS:
            return "too_short"
        return _text_item(text, level=int(row["star"]) - 1)
    if source == "ledgar":
        gold = _class_name(row["label"], S.LEDGAR_NAMES)
        if gold is None:
            return "unlabelled"
        return _text_item(row["text"], golds={"provision": [S.LEDGAR.by_raw()[gold]]})
    if source == "fewrel":
        gold = S.FEWREL_BY_PROPERTY.get(row["relation"])
        tokens = [str(t) for t in row["tokens"]]

        def mention(entity: Mapping[str, Any]) -> str:  # the first mention, as written in the sentence
            spans = entity["indices"]
            return _clean(" ".join(tokens[i] for i in spans[0] if 0 <= i < len(tokens))) if spans else ""

        sentence, head, tail = _clean(" ".join(tokens)), mention(row["head"]), mention(row["tail"])
        if gold is None:
            return "unlabelled"
        if not sentence or not head or not tail:
            return "empty_text"
        return {"text": sentence, "fields": (sentence, head, tail), "golds": {"relation": [gold]},
                "keys": (_key(sentence, head, tail),)}  # fmt: skip
    raise ValueError(f"unknown source {source!r}")


# -- items -> examples ------------------------------------------------------------------

_STYLE_KEY = {
    "tweet_offensive": "tweet", "tweet_hate": "tweet", "tweet_irony": "tweet", "tweet_sentiment": "tweet",
    "massive_intent": "massive", "massive_scenario": "massive", "massive_hn": "massive", "clinc_hn": "clinc",
}  # fmt: skip
_FALSE_CLASSES = {"scitail": ("neutral",)}


def _style(source: str) -> TextStyle:
    return S.STYLES[_STYLE_KEY.get(source, source)]


def class_quotas(source: str, family: str, n: int) -> dict[Any, int] | None:
    """How many examples of each row class a family takes; None = as they come."""
    if family == "nli" and source != "rte":
        false = _FALSE_CLASSES.get(source, ("neutral", "contradiction"))
        n_true = int(round(n * NLI_TRUE_RATE))
        return {"entailment": n_true, **_split_proportional(n - n_true, dict.fromkeys(false, 1.0))}
    if family == "reading" and source != "multirc":
        n_true = int(round(n * READING_TRUE_RATE))
        return {True: n_true, False: n - n_true}
    if family == "judgement":
        n_true = int(round(n * JUDGEMENT_TRUE_RATE[source]))
        return {True: n_true, False: n - n_true}
    if family == "score" and source in ("civil_comments", "app_reviews"):
        levels = range(4 if source == "civil_comments" else 5)
        return _split_proportional(n, dict.fromkeys(levels, 1.0))  # type: ignore[arg-type]
    return None


def positive_rate(source: str, family: str) -> float | None:
    """Share of true labels for families whose label is chosen when the example is made."""
    if family == "ovr":
        return OVR_TRUE_RATE
    if family == "candidate":
        return XFER_CANDIDATE_TRUE_RATE if source in S.XFER_SOURCES else CANDIDATE_TRUE_RATE
    return None


def item_class(family: str, item: Mapping[str, Any]) -> Any:
    return item["level"] if family == "score" else item.get("cls")


def _labelset(source: str, family: str, item: Mapping[str, Any], rng: random.Random) -> tuple[str, LabelSet]:
    sets = S.label_sets(source)
    if source == "trec":
        name = "fine" if rng.random() < P_TREC_FINE else "coarse"
        return name, sets[name]
    return next(iter(sets.items()))


def usable(source: str, family: str, item: Mapping[str, Any]) -> bool:
    """Whether a family can use this item at all (it stays available to other families if not)."""
    if family == "choice" and "golds" in item:
        return all(len(g) <= 1 for g in item["golds"].values())  # single-label rows only
    return True


def can_be_true(source: str, family: str, item: Mapping[str, Any]) -> bool:
    if family == "ovr":
        return all(len(g) >= 1 for g in item["golds"].values())  # clinc "oos" matches no intent
    return True


def make_example(source: str, family: str, item: Mapping[str, Any], rng: random.Random, want_true: bool | None = None) -> dict[str, Any] | None:
    """Turn an item into one augmented training example of the given family."""
    if family == "nli":
        return nli_example(item["premise"], item["hypothesis"], item["cls"] == "entailment", _style(source), rng)
    if family == "reading":
        if source == "pubmedqa":
            return pubmed_example(item["passage"], item["question"], item["cls"], _style(source), rng)
        return reading_example(item["passage"], item["question"], item["cls"], source, _style(source), rng)
    if family == "judgement":
        if source == "paws":
            state, wrapper = wrap_pair(item["a"], item["b"], rng, S.PAIR_LABELS, S.PAIR_KEYS)
        else:
            state, wrapper, _ = wrap_text(item["text"], _style(source), rng)
        return judgement_example(state, wrapper, item["cls"], S.JUDGEMENTS[source], rng)
    if family == "score":
        if source == "stsb":
            state, wrapper = wrap_pair(item["a"], item["b"], rng, S.PAIR_LABELS, S.PAIR_KEYS)
        else:
            state, wrapper, _ = wrap_text(item["text"], _style(source), rng)
        return score_example(state, wrapper, item["level"], source, rng)
    setname, labels = _labelset(source, family, item, rng)
    golds: list[Label] = item["golds"][setname]
    if "fields" in item:
        state, wrapper = wrap_fields(item["fields"], rng, *S.RELATION_FIELDS)
    else:
        state, wrapper, _ = wrap_text(item["text"], _style(source), rng, item.get("extra"))
    if family == "choice" and (labels.hard or "mined" in item):
        ex = shortlist_example(state, wrapper, golds[0], labels, rng, item.get("mined"))
    elif family == "choice":
        ex = choice_example(state, wrapper, golds[0] if golds else None, labels, rng)
    elif family == "candidate":
        ex = candidate_example(state, wrapper, golds[0] if golds else None, labels, rng, bool(want_true))
    elif family == "ovr":
        ex = ovr_example(state, wrapper, golds, labels, rng, bool(want_true))
    else:
        raise ValueError(f"unknown family {family!r}")
    ex["meta"]["labelset"] = setname
    return ex


def make_xfer_example(source: str, family: str, item: Mapping[str, Any], rng: random.Random, want_true: bool | None = None) -> dict[str, Any] | None:
    """dev_xfer examples: one canonical format per source, no augmentation."""
    if source == "rte":
        if item["hypothesis"].endswith("?"):
            return None
        return {"state": item["premise"], "question": _noul(item["hypothesis"], None),
                "label": int(item["cls"] == "entailment"), "meta": {}}  # fmt: skip
    if source == "multirc":
        state = {"passage": item["passage"], "question": item["question"], "answer": item["answer"]}
        return {"state": state, "question": _noul(S.XFER_MULTIRC_INSTRUCTIONS, None), "label": int(item["cls"]), "meta": {}}
    if source == "app_reviews":
        question = {"type": "score", "instructions": S.XFER_APP_REVIEWS_INSTRUCTIONS,
                    "criteria": list(S.XFER_APP_REVIEWS_LEVELS)}  # fmt: skip
        return {"state": item["text"], "question": question, "label": item["level"], "meta": {}}
    if source == "hwu64":
        labels = S.HWU64
        gold: Label = item["golds"]["intent"][0]
        if family == "candidate":
            return _plain_candidate(item["text"], gold, labels, rng, bool(want_true))
        others = [label for label in labels.labels if label is not gold]
        real = [gold] + rng.sample(others, XFER_CHOICE_OPTIONS - 1)
        rng.shuffle(real)
        question = {"type": "choice", "instructions": labels.choice_instructions[0],
                    "criteria": {label.name: None for label in real}}  # fmt: skip
        return {"state": item["text"], "question": question, "label": real.index(gold), "meta": {"k": len(real)}}
    raise ValueError(f"unknown dev_xfer source {source!r}")


def _plain_candidate(text: str, gold: Label, labels: LabelSet, rng: random.Random, want_true: bool) -> dict[str, Any]:
    """Canonical candidate row: ten spaced option names, no descriptions."""
    cand, sibling = (gold, False) if want_true else _wrong_label([gold], labels, rng)
    chosen = list(dict.fromkeys([gold, cand]))
    rest = [label for label in labels.labels if label not in chosen]
    real = chosen + rng.sample(rest, XFER_CHOICE_OPTIONS - len(chosen))
    rng.shuffle(real)
    question = {"type": "choice", "instructions": labels.choice_instructions[0],
                "criteria": {label.name: None for label in real}}  # fmt: skip
    return {"state": text, "question": question, "label": int(want_true), "candidate": cand.name,
            "meta": {"k": len(real), "sibling_negative": sibling}}  # fmt: skip


# -- prompts and token counts -----------------------------------------------------------


def render(row: Mapping[str, Any]) -> Prompt:
    """The zero-shot prompt the trainer builds for a row."""
    question = parse_question("row", row["question"])
    if row.get("candidate") is not None:
        _require(isinstance(question, ChoiceQuestion), "candidate rows need a choice question")
        return build_candidate_prompt(row["state"], question, row["candidate"])
    return build_prompt(row["state"], question)


class HrmTokenCounter:
    """Token counts identical to `HrmBackend._encode` for text without control-token strings."""

    name = "hrm-tokenizer"

    def __init__(self, model_id: str | None = None) -> None:
        from transformers import AutoTokenizer

        from davout.backends import hrm

        self.model_id = model_id or hrm.MODEL_ID
        self._tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
        self._head, self._tail = hrm.IM_START + hrm.DIRECT, hrm.IM_END
        specials = set(self._tok.get_added_vocab()) | set(self._tok.all_special_tokens)
        self._special_re = re.compile("|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)))

    def has_control_text(self, text: str) -> bool:
        return self._special_re.search(text) is not None

    def __call__(self, text: str) -> int:
        return len(self._tok.encode(self._head + text + self._tail, add_special_tokens=False))


class CharProxyCounter:
    """Conservative stand-in for the tokenizer: characters / PROXY_CHARS_PER_TOKEN, rounded up."""

    name = "char-proxy"

    def __init__(self, chars_per_token: float = PROXY_CHARS_PER_TOKEN) -> None:
        self.chars_per_token = chars_per_token

    def has_control_text(self, text: str) -> bool:
        return "<|" in text

    def __call__(self, text: str) -> int:
        return int(math.ceil(len(text) / self.chars_per_token)) + 3


class _FnCounter:
    def __init__(self, fn: Callable[[str], int]) -> None:
        self._fn = fn
        self.name = getattr(fn, "name", None) or getattr(fn, "__name__", "custom")

    def has_control_text(self, text: str) -> bool:
        return "<|" in text

    def __call__(self, text: str) -> int:
        return self._fn(text)


def _default_counter() -> Any:
    try:
        return HrmTokenCounter()
    except Exception:  # tokenizer not downloaded, transformers missing, ...
        return CharProxyCounter()


# -- guards -----------------------------------------------------------------------------


def _norm_text(text: Any) -> str:
    return _WS.sub(" ", str(text)).strip().lower().rstrip(".?! ")


def benchmark_texts() -> tuple[set[str], set[str], set[str]]:
    """(instructions of the held-out tasks, Yelp level strings, benchmark dataset ids)."""
    from davout.bench.tasks import TASKS

    instructions = {_norm_text(t.question.instructions) for t in TASKS.values()}
    yelp = {_norm_text(level) for level in TASKS["yelp"].question.criteria}
    return instructions, yelp, {t.dataset for t in TASKS.values()}


def check_registry() -> None:
    """Static guards: excluded datasets, the CLINC exclusion list, label-set sanity, wording."""
    instructions, yelp, bench_ids = benchmark_texts()
    excluded = set(S.EXCLUDED_DATASETS) | bench_ids
    recipes = list({src.name: src for registry in S.RECIPES.values() for src in registry.values()}.values())
    for src in recipes + list(S.XFER_SOURCES.values()):
        _require(src.dataset not in excluded, f"{src.name}: dataset {src.dataset} is excluded")
        _require((src.dataset, src.config) not in S.EXCLUDED_CONFIGS, f"{src.name}: config {src.config} is excluded")
    _require(len(S.CLINC_EXCLUDED) == 32, "the CLINC exclusion list must hold 32 intents")
    _require(S.CLINC_EXCLUDED <= set(S.CLINC_INTENTS), "CLINC exclusions must be CLINC intent names")
    _require(len(S.CLINC.labels) == 151 - 32 - 1, "CLINC should keep 118 intents")
    _require(sum(sum(s.counts.values()) for s in S.SOURCES.values()) == 96_000, "the mix must total 96,000")
    _require(sum(s.dev_n for s in S.SOURCES.values()) == DEV_IN_TOTAL, "dev_in must total 2,000")
    others = {_norm_text(n) for n in S.OTHER_NAMES}
    wording: list[str] = [S.XFER_MULTIRC_INSTRUCTIONS, S.XFER_APP_REVIEWS_INSTRUCTIONS]
    for src in [s.name for s in recipes] + list(S.XFER_SOURCES):
        for name, labels in S.label_sets(src).items():
            _require(not labels.hard or len(labels.labels) > max(NAME_NEIGHBOURS, SHORTLIST_OPTIONS),
                     f"{src}/{name}: too few labels for hard negatives")  # fmt: skip
            for style in NAME_STYLES:
                names = [styled(label, style) for label in labels.labels]
                _require(len(set(names)) == len(names), f"{src}/{name}: {style} option names collide")
                _require(not others & {_norm_text(n) for n in names}, f"{src}/{name}: a label is named like 'other'")
            wording += labels.choice_instructions
            for label in labels.labels:
                for t in (*labels.statements, *labels.questions, *labels.negations,
                          *label.statements, *label.questions, *label.negations):  # fmt: skip
                    wording.append(t.format(p=label.phrase or label.name))
            _require(all(t.endswith("?") for t in labels.questions), f"{src}/{name}: questions must end in '?'")
            _require(not any(t.endswith("?") for t in (*labels.statements, *labels.negations)),
                     f"{src}/{name}: statements must not end in '?'")  # fmt: skip
    for j in S.JUDGEMENTS.values():
        wording += [*j.statements, *j.questions]
    for scales in S.SCALES.values():
        for scale in scales:
            wording += scale.instructions
            for by_text in scale.levels.values():
                for levels in by_text.values():
                    _require(not yelp & {_norm_text(x) for x in levels}, "a score level reuses a Yelp level string")
    _require(not yelp & {_norm_text(x) for x in S.XFER_APP_REVIEWS_LEVELS}, "dev_xfer reuses a Yelp level string")
    wording += [*S.SQUAD_STRUCT_STATEMENTS, *S.SQUAD_STRUCT_QUESTIONS, *S.QNLI_STRUCT_STATEMENTS, *S.QNLI_STRUCT_QUESTIONS]
    clash = instructions & {_norm_text(w) for w in wording}
    _require(not clash, f"training wording equals a benchmark instruction: {sorted(clash)}")


def check_example(row: Mapping[str, Any], prompt_text: str, instructions: set[str], yelp: set[str]) -> None:
    """Per-row guards on what the model will actually see."""
    q = row["question"]
    _require(_norm_text(q["instructions"]) not in instructions,
             f"{row['id']}: instruction equals a benchmark instruction: {q['instructions']!r}")  # fmt: skip
    _require(S.FORBIDDEN_WRAPPER not in prompt_text, f"{row['id']}: forbidden wrapper in the prompt")
    parsed = parse_question(row["id"], q)
    if isinstance(parsed, ChoiceQuestion):
        n = len(parsed.criteria)
        _require(2 <= n <= MAX_OPTIONS, f"{row['id']}: {n} options")
        if row["family"] == "candidate":
            _require(row.get("candidate") in parsed.criteria, f"{row['id']}: candidate is not an option")
            _require(row["label"] in (0, 1), f"{row['id']}: candidate label must be 0 or 1")
        else:
            _require(0 <= row["label"] < n, f"{row['id']}: label {row['label']} out of range")
    elif isinstance(parsed, ScoreQuestion):
        _require(0 <= row["label"] < len(parsed.criteria), f"{row['id']}: level {row['label']} out of range")
        _require(not yelp & {_norm_text(x) for x in parsed.criteria}, f"{row['id']}: Yelp level string reused")
    else:
        _require(isinstance(parsed, NoulQuestion) and row["label"] in (0, 1), f"{row['id']}: noul label must be 0 or 1")
    want = {"choice": "choice", "candidate": "choice", "score": "score"}.get(row["family"], "noul")
    _require(q["type"] == want, f"{row['id']}: family {row['family']} needs a {want} question")


# -- label-mapping checks against the loaded data ---------------------------------------


def _column(rows: Any, name: str) -> list[Any]:
    if hasattr(rows, "column_names"):
        return list(rows[name])
    return [r[name] for r in rows]


def _class_names(rows: Any, column: str) -> list[str] | None:
    feature = (getattr(rows, "features", None) or {}).get(column)
    feature = getattr(feature, "feature", feature)  # List(ClassLabel) -> ClassLabel
    names = getattr(feature, "names", None)
    return list(names) if names is not None else None


def _expect_names(rows: Any, column: str, expected: Sequence[str], where: str) -> str:
    names = _class_names(rows, column)
    if names is None:
        return "no ClassLabel metadata (not checked)"
    _require(names == list(expected), f"{where}: {column} names are {names}, expected {list(expected)}")
    return f"ClassLabel names match ({len(names)})"


def _expect_values(rows: Any, column: str, expected: Sequence[Any], where: str, exact: bool) -> dict[str, int]:
    counts = Counter(_column(rows, column))
    extra = set(counts) - set(expected)
    _require(not extra, f"{where}: unexpected {column} values {sorted(map(str, extra))[:10]}")
    if exact:
        missing = set(expected) - set(counts)
        _require(not missing, f"{where}: {column} values never occur: {sorted(map(str, missing))[:10]}")
    return {str(k): v for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))}


def dailydialog_acts(rows: Any) -> dict[str, Any]:
    """Share of utterances ending in "?" per act id; `ok` when the registry's question id is the one."""
    total: Counter[int] = Counter()
    asks: Counter[int] = Counter()
    for utterances, acts in zip(_column(rows, "utterances"), _column(rows, "acts")):
        for u, a in zip(utterances, acts):
            total[int(a)] += 1
            asks[int(a)] += str(u).strip().endswith("?")
    rates = {a: asks[a] / total[a] for a in sorted(total)}
    q = S.DAILYDIALOG_QUESTION_ACT
    known = {int(label.raw) for label in S.DAILYDIALOG.labels}
    ok = set(total) <= known and rates.get(q, 0.0) >= 0.8 and all(r < 0.5 for a, r in rates.items() if a != q)
    return {"ok": ok, "question_act": q, "ends_with_question_mark": {str(a): round(r, 3) for a, r in rates.items()},
            "utterances": {str(a): total[a] for a in sorted(total)}}  # fmt: skip


def verify_source(src: Source, split: str, rows: Any) -> dict[str, Any]:
    """Assert the label mapping of one loaded split; returns what was found (for the manifest)."""
    name, where = S.MINED_SOURCES.get(src.name, src.name), f"{src.dataset} [{split}]"
    exact = hasattr(rows, "features") and split in src.train_splits  # real data, full split
    columns = getattr(rows, "column_names", None)
    if columns is not None:
        missing = [c for c in src.columns if c not in columns]
        _require(not missing, f"{where}: missing column(s) {missing}; found {columns}")
    _require(len(rows) > 0, f"{where}: no rows")
    out: dict[str, Any] = {"rows": len(rows)}
    if name in ("mnli", "anli"):
        out["label"] = _expect_names(rows, "label", S.NLI_NAMES, where)
    elif name == "wanli":
        out["gold"] = _expect_values(rows, "gold", S.NLI_NAMES, where, exact)
    elif name == "vitaminc":
        out["label"] = _expect_values(rows, "label", list(_VITAMINC), where, exact)
    elif name == "scitail":
        out["label"] = _expect_values(rows, "label", list(_SCITAIL), where, exact)
    elif name == "squad_v2":
        answers = _column(rows, "answers")
        answerable = [i for i, a in enumerate(answers) if len(a["text"]) > 0]
        _require(0 < len(answerable) < len(rows), f"{where}: need both answerable and unanswerable questions")
        probe = answerable[:: max(1, len(answerable) // 300)][:300]
        found = sum(answers[i]["text"][0] in rows[i]["context"] for i in probe)
        _require(found >= 0.98 * len(probe), f"{where}: answers are not spans of the context ({found}/{len(probe)})")
        out.update(answerable=len(answerable), unanswerable=len(rows) - len(answerable),
                   answer_is_span_of_context=f"{found}/{len(probe)} sampled")  # fmt: skip
    elif name == "pubmedqa":
        out["final_decision"] = _expect_values(rows, "final_decision", ("yes", "no"), where, exact)
    elif name in ("qnli", "rte"):
        out["label"] = _expect_names(rows, "label", ("entailment", "not_entailment"), where)
    elif name == "paws":
        out["label"] = _expect_names(rows, "label", ("0", "1"), where)
        out["values"] = _expect_values(rows, "label", (0, 1), where, exact)
    elif name == "tweet_offensive":
        out["label"] = _expect_names(rows, "label", ("non-offensive", "offensive"), where)
    elif name == "tweet_hate":
        out["label"] = _expect_names(rows, "label", ("non-hate", "hate"), where)
    elif name == "tweet_irony":
        out["label"] = _expect_names(rows, "label", ("non_irony", "irony"), where)
    elif name == "tweet_sentiment":
        out["label"] = _expect_names(rows, "label", ("negative", "neutral", "positive"), where)
    elif name == "yahoo":
        out["topic"] = _expect_names(rows, "topic", [label.raw for label in S.YAHOO.labels], where)
    elif name == "newsgroups":
        out["label_text"] = _expect_values(rows, "label_text", list(S.NEWSGROUPS.by_raw()), where, exact)
    elif name == "dbpedia":
        out["label"] = _expect_names(rows, "label", [label.raw for label in S.DBPEDIA.labels], where)
    elif name == "clinc":
        out["intent"] = _expect_names(rows, "intent", S.CLINC_INTENTS, where)
        counts = Counter(_column(rows, "intent"))
        out["oos_rows"] = counts.get(S.CLINC_INTENTS.index(S.CLINC_OOS), 0)
        out["excluded_rows"] = sum(counts.get(S.CLINC_INTENTS.index(n), 0) for n in S.CLINC_EXCLUDED)
    elif name == "massive_intent":
        counts = _expect_values(rows, "label_text", S.MASSIVE_INTENT_NAMES, where, exact)
        out["label_text"] = f"{len(counts)} of the 60 registry intents occur"
    elif name == "massive_scenario":
        counts = _expect_values(rows, "label_text", list(S.MASSIVE_SCENARIO.by_raw()), where, exact)
        out["label_text"] = f"{len(counts)} of the 18 registry scenarios occur"
    elif name == "trec":
        pairs = set(zip(_column(rows, "label_text"), _column(rows, "label_coarse_text")))
        expected = {(fine, coarse) for fine, _orig, coarse in S.TREC_FINE_NAMES}
        _require(pairs <= expected, f"{where}: unexpected (fine, coarse) labels {sorted(pairs - expected)[:5]}")
        _require(not exact or pairs == expected, f"{where}: labels never seen: {sorted(expected - pairs)[:5]}")
        out["labels"] = f"{len(pairs)} of the 50 fine labels occur, each under its registry coarse label"
    elif name == "go_emotions":
        out["labels"] = _expect_names(rows, "labels", [label.raw for label in S.GO_EMOTIONS.labels], where)
    elif name == "emotion":
        out["label"] = _expect_names(rows, "label", [label.raw for label in S.EMOTION.labels], where)
    elif name == "dailydialog":
        out.update(dailydialog_acts(rows))
    elif name == "amazon_reviews":
        out["label"] = _expect_values(rows, "label", range(5), where, exact)
    elif name == "stsb":
        scores = [float(s) for s in _column(rows, "score")]
        _require(0.0 <= min(scores) and max(scores) <= 1.0, f"{where}: score outside [0, 1]")
        levels = Counter(stsb_level(s) for s in scores)
        out.update(score_range=[min(scores), max(scores)], levels={str(k): levels[k] for k in sorted(levels)})
    elif name == "civil_comments":
        tox = [float(t) for t in _column(rows, "toxicity")]
        _require(0.0 <= min(tox) and max(tox) <= 1.0, f"{where}: toxicity outside [0, 1]")
        bins = Counter(civil_level(t) for t in tox)
        _require(not exact or all(bins[b] > 0 for b in range(4)), f"{where}: an empty toxicity bin: {dict(bins)}")
        out.update(toxicity_range=[min(tox), max(tox)], bins={str(b): bins[b] for b in range(4)})
    elif name == "hwu64":
        counts = _expect_values(rows, "label", S.HWU64_NAMES, where, False)
        out["label"] = f"{len(counts)} of the 64 registry intents occur"
    elif name == "multirc":
        out["label"] = _expect_names(rows, "label", ("False", "True"), where)
        out["paragraphs"] = len(set(_column(rows, "paragraph")))
    elif name == "app_reviews":
        out["star"] = _expect_values(rows, "star", range(1, 6), where, exact)
    elif name == "ledgar":
        out["label"] = _expect_names(rows, "label", S.LEDGAR_NAMES, where)
    elif name == "fewrel":
        counts = _expect_values(rows, "relation", list(S.FEWREL_BY_PROPERTY), where, exact)
        named = {(r, n[0]) for r, n in zip(_column(rows, "relation"), _column(rows, "names"))}
        expected = {(prop, relation) for prop, relation, _desc in S.FEWREL_RELATIONS}
        _require(named <= expected, f"{where}: unexpected relation names {sorted(named - expected)[:5]}")
        out["relation"] = f"{len(counts)} of the 64 registry relations occur, each under its registry name"
    return out


def verify_massive_pair(intent_rows: Any, scenario_rows: Any) -> dict[str, Any]:
    """The two MASSIVE datasets hold the same utterances: same ids, same text, scenario = intent prefix."""
    a = dict(zip(map(str, _column(intent_rows, "id")), zip(_column(intent_rows, "text"), _column(intent_rows, "label_text"))))
    b = dict(zip(map(str, _column(scenario_rows, "id")), zip(_column(scenario_rows, "text"), _column(scenario_rows, "label_text"))))
    shared = set(a) & set(b)
    _require(all(a[i][0] == b[i][0] for i in shared), "MASSIVE: an id has different text in the two datasets")
    _require(all(a[i][1].split("_")[0] == b[i][1] for i in shared), "MASSIVE: scenario is not the intent prefix")
    return {"shared_ids": len(shared), "intent_ids": len(a), "scenario_ids": len(b),
            "same_text_and_scenario_is_intent_prefix": True}  # fmt: skip


# -- loading ----------------------------------------------------------------------------


def hf_loader(src: Source, split: str) -> Any:
    """Load one split with `datasets` (imported here so the module works without it)."""
    try:
        import datasets
    except ImportError:
        raise RuntimeError("building the training data needs the `datasets` library; install it with `uv sync --extra eval`") from None
    if src.data_files is not None:
        # Plain files of a repo whose default loader is a script: fetch them (or find them in
        # the cache when offline) and read the parquet directly.
        from huggingface_hub import snapshot_download

        pattern = src.data_files.format(split=split)
        root = Path(snapshot_download(src.dataset, repo_type="dataset", revision=src.revision, allow_patterns=pattern))
        files = sorted(str(f) for f in root.glob(pattern))
        if not files:
            raise RuntimeError(f"{src.dataset}@{src.revision}: no files match {pattern}")
        rows = datasets.load_dataset("parquet", data_files=files, split="train")
        rows._davout_revision = root.name  # snapshot directories are named after the commit
        return rows
    return datasets.load_dataset(src.dataset, src.config, split=split, revision=src.revision)


def _revision(rows: Any) -> str | None:
    """Commit hash of a loaded Hub dataset, read from its cache path."""
    if getattr(rows, "_davout_revision", None):
        return rows._davout_revision
    for f in getattr(rows, "cache_files", None) or []:
        m = re.search(r"/([0-9a-f]{40})/", str(f.get("filename", "")))
        if m:
            return m.group(1)
    return None


# -- mined negatives and held-out texts --------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: Any) -> list[str]:
    return _WORD.findall(str(text).lower())


class TextGuard:
    """Texts of held-out sets: `hit(text)` when a text repeats one or shares `n` consecutive words with one."""

    def __init__(self, texts: Sequence[Any], n: int = S.GUARD_NGRAM) -> None:
        self.n = n
        self.exact: set[str] = set()
        self.grams: set[tuple[str, ...]] = set()
        for text in texts:
            words = _words(text)
            self.exact.add(" ".join(words))
            self.grams.update(tuple(words[i : i + n]) for i in range(len(words) - n + 1))

    def hit(self, text: Any) -> bool:
        words = _words(text)
        if " ".join(words) in self.exact:
            return True
        return any(tuple(words[i : i + self.n]) in self.grams for i in range(len(words) - self.n + 1))


def read_mined(directory: str | Path, names: Sequence[str]) -> dict[str, dict[str, list[str]]]:
    """`<directory>/<source>.jsonl` as written by `davout train mine-negatives`: row id -> wrong labels, best first."""
    out: dict[str, dict[str, list[str]]] = {}
    for name in names:
        path = Path(directory) / f"{name}.jsonl"
        if not path.is_file():
            raise ValueError(f"{path} not found; run `davout train mine-negatives` first")
        with path.open(encoding="utf-8") as f:
            out[name] = {r["id"]: r["wrong"] for r in map(json.loads, f)}
    return out


def balanced_quotas(available: Mapping[Any, int], n: int) -> dict[Any, int]:
    """`n` examples spread evenly over the classes, none above what is available (water-filling)."""
    quotas = dict.fromkeys(available, 0)
    left = min(n, sum(available.values()))
    while left > 0:
        open_ = sorted((c for c in available if quotas[c] < available[c]), key=str)
        share = max(1, left // len(open_))
        for c in open_:
            add = min(share, available[c] - quotas[c], left)
            quotas[c] += add
            left -= add
            if left == 0:
                break
    return quotas


# -- build ------------------------------------------------------------------------------


class _Pool:
    """Rows of one split in a seeded order; every row is used at most once."""

    def __init__(self, src: Source, split: str, rows: Any, seed: int) -> None:
        self.src, self.split, self.rows, self.seed = src, split, rows, seed
        self.order = list(range(len(rows)))
        random.Random(f"{seed}:order:{src.name}:{split}").shuffle(self.order)
        self.taken: set[int] = set()
        self.dead: dict[int, str] = {}  # rows that can never be used, with the reason
        self._items: dict[int, Any] = {}

    def item(self, idx: int) -> dict[str, Any] | str:
        if idx not in self._items:
            rng = random.Random(f"{self.seed}:row:{self.src.name}:{self.split}:{idx}")
            self._items[idx] = normalise(self.src.name, self.rows[idx], rng)
        return self._items[idx]


class _Build:
    def __init__(self, seed: int, scale: float, loader: Callable[[Source, str], Any], counter: Any) -> None:
        self.seed, self.scale, self.loader, self.counter = seed, scale, loader, counter
        self.pools: dict[tuple[str, str], _Pool] = {}
        self.reserved: set[str] = set()  # content keys of dev rows
        self.used: dict[str, set[str]] = defaultdict(set)  # content keys of train rows per source group
        self.dropped: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        self.shortfall: dict[str, dict[str, int]] = defaultdict(dict)
        self.checks: dict[str, dict[str, Any]] = defaultdict(dict)
        self.revisions: dict[str, str | None] = {}
        self.lengths: dict[str, list[tuple[str, int, int]]] = defaultdict(list)  # role -> (family, chars, tokens)
        self.meta: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        self.uids: dict[str, set[str]] = defaultdict(set)
        self.excluded_texts: dict[str, set[str]] = {}
        self.mined: dict[str, dict[str, list[str]]] = {}  # source -> row id -> wrong labels, best first
        self.guard: TextGuard | None = None  # held-out texts the mined sources must not repeat
        self.instructions, self.yelp, _ = benchmark_texts()

    def pool(self, src: Source, split: str) -> _Pool:
        key = (src.name, split)
        if key not in self.pools:
            rows = self.loader(src, split)
            self.checks[src.name][split] = verify_source(src, split, rows)
            self.revisions.setdefault(src.name, _revision(rows))
            self.pools[key] = _Pool(src, split, rows, self.seed)
        return self.pools[key]

    def collect(self, role: str, src: Source, family: str, split: str, n: int, out: list[dict[str, Any]],
                limits: Counter[str] | None = None) -> None:  # fmt: skip
        """Append up to `n` examples of one family from one split to `out`."""
        if n <= 0:
            return
        pool = self.pool(src, split)
        name = src.name
        rng = random.Random(f"{self.seed}:{role}:{name}:{family}:{split}")
        xfer = role == "dev_xfer"
        quotas = class_quotas(name, family, n)
        if name in self.mined and role == "train":  # as even over the intents as the rows left allow
            free: Counter[str] = Counter()
            for i in pool.order:
                it = pool.item(i)
                if i in pool.taken or i in pool.dead or isinstance(it, str) or f"{name}/{split}/{i}" not in self.mined[name]:
                    continue
                if self.guard is None or not self.guard.hit(it["text"]):
                    free[it["cls"]] += 1
            quotas = balanced_quotas(free, n)
        rate = positive_rate(name, family)
        need_pos = int(round(n * rate)) if rate is not None else 0
        need_neg = n - need_pos
        group = src.group or name
        drops = self.dropped[role][name]
        start = len(out)
        for idx in pool.order:
            if len(out) - start >= n:
                break
            if idx in pool.taken or idx in pool.dead:
                continue
            item = pool.item(idx)
            if isinstance(item, str):
                pool.dead[idx] = item
                drops[item] += 1
                continue
            keys = item["keys"]
            if name in self.mined:
                if self.guard is not None and self.guard.hit(item["text"]):
                    pool.dead[idx] = "held_out_text_overlap"
                    drops["held_out_text_overlap"] += 1
                    continue
                wrong = self.mined[name].get(f"{name}/{split}/{idx}")
                if wrong is None:
                    pool.dead[idx] = "not_mined"
                    drops["not_mined"] += 1
                    continue
                item = {**item, "mined": wrong}
            if xfer and any(k in self.excluded_texts.get(name, ()) for k in keys):
                pool.dead[idx] = "text_in_training_corpus"
                drops["text_in_training_corpus"] += 1
                continue
            if any(k in self.reserved for k in keys):
                if role == "train":
                    drops["held_out_for_dev"] += 1
                else:
                    drops["duplicate_text"] += 1
                pool.dead[idx] = "reserved"
                continue
            if role == "train" and any(k in self.used[group] for k in keys):
                # for the second MASSIVE source this is a row its sibling already used
                reason = "used_by_sibling_source" if keys[0] in self.used[group] and group != name else "duplicate_text"
                pool.dead[idx] = reason
                drops[reason] += 1
                continue
            if not usable(name, family, item):
                continue
            cls = item_class(family, item)
            if quotas is not None and quotas.get(cls, 0) <= 0:
                continue
            if limits is not None and limits[item["limit"]] >= MULTIRC_PER_PARAGRAPH:
                continue
            want: bool | None = None
            if rate is not None:
                if not can_be_true(name, family, item):
                    if need_neg <= 0:
                        continue
                    want = False
                else:
                    want = rng.random() < need_pos / max(1, need_pos + need_neg)
            ex = (make_xfer_example if xfer else make_example)(name, family, item, rng, want)
            if ex is None:
                pool.dead[idx] = "cannot_phrase"
                drops["cannot_phrase"] += 1
                continue
            row = {"id": f"{name}/{split}/{idx}" + (f".{item['sub']}" if "sub" in item else ""),
                   "source": name, "family": family, "state": ex["state"], "question": ex["question"],
                   "label": ex["label"]}  # fmt: skip
            if "candidate" in ex:
                row["candidate"] = ex["candidate"]
            text = render(row).text
            check_example(row, text, self.instructions, self.yelp)
            if self.counter.has_control_text(text):
                pool.dead[idx] = "control_token_text"
                drops["control_token_text"] += 1
                continue
            tokens = self.counter(text)
            if tokens > MAX_TOKENS:
                drops["over_512_tokens"] += 1
                if family in ("nli", "reading", "judgement", "score"):
                    pool.dead[idx] = "over_512_tokens"  # the state alone is too long; no family can use it
                continue
            pool.taken.add(idx)
            (self.reserved if role != "train" else self.used[group]).update(keys)
            if "uid" in item:
                self.uids[f"{role}:{name}"].add(item["uid"])
            if quotas is not None:
                quotas[cls] -= 1
            if limits is not None:
                limits[item["limit"]] += 1
            if want is True:
                need_pos -= 1
            elif want is False:
                need_neg -= 1
            self.lengths[role].append((family, len(text), tokens))
            for k, v in ex["meta"].items():
                self.meta[role][f"{family}.{k}"][str(v)] += 1
            out.append(row)
        if len(out) - start < n:
            self.shortfall[role][f"{name}/{family}/{split}"] = n - (len(out) - start)


def _dev_alloc(counts: Mapping[str, Mapping[str, int]], dev_n: Mapping[str, int], scale: float) -> dict[str, dict[str, int]]:
    return {name: _split_proportional(_scaled(dev_n[name], scale), counts[name]) for name in counts}


def _stats(tokens: Sequence[int]) -> dict[str, Any]:
    if not tokens:
        return {"n": 0}
    s = sorted(tokens)

    def pct(p: float) -> int:
        return s[min(len(s) - 1, int(p * len(s)))]

    return {"n": len(s), "mean": round(sum(s) / len(s), 1), "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
            "max": s[-1], "total": sum(s)}  # fmt: skip


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_family: Counter[str] = Counter()
    pos: dict[str, list[int]] = defaultdict(list)
    pos_source: dict[str, list[int]] = defaultdict(list)
    options: dict[str, Counter[int]] = {"choice": Counter(), "candidate": Counter()}
    gold_letter: Counter[int] = Counter()
    for r in rows:
        fam = r["family"]
        by_source[r["source"]][fam] += 1
        by_family[fam] += 1
        if fam in NOUL_FAMILIES or fam == "candidate":
            pos[fam].append(r["label"])
            pos_source[f"{r['source']}/{fam}"].append(r["label"])
        if fam in options:
            options[fam][len(r["question"]["criteria"])] += 1
        if fam == "choice":
            gold_letter[r["label"]] += 1
    return {
        "rows": len(rows),
        "per_source": {s: {**dict(c), "total": sum(c.values())} for s, c in by_source.items()},
        "per_family": dict(by_family),
        "positive_rate": {f: round(sum(v) / len(v), 4) for f, v in pos.items()},
        "positive_rate_per_source": {f: round(sum(v) / len(v), 4) for f, v in sorted(pos_source.items())},
        "choice_option_counts": {str(k): options["choice"][k] for k in sorted(options["choice"])},
        "candidate_option_counts": {str(k): options["candidate"][k] for k in sorted(options["candidate"])},
        "choice_gold_position": {str(k): gold_letter[k] for k in sorted(gold_letter)},
    }


def _content(row: Mapping[str, Any]) -> str:
    return json.dumps([row["state"], row["question"], row.get("candidate")], ensure_ascii=False, sort_keys=True)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(
    out_dir: str | Path,
    seed: int = 0,
    scale: float = 1.0,
    sources: Sequence[str] | None = None,
    *,
    recipe: str = "v1",
    negatives: str | Path | None = None,
    extra: str | Path | None = None,
    loader: Callable[[Source, str], Any] | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> dict:
    """Write train.jsonl, dev_in.jsonl, dev_xfer.jsonl and manifest.json; return the manifest.

    `scale` multiplies every per-source count, the dev sets included (1.0 = 96,000 train
    rows, 2,000 dev_in, about 1,080 dev_xfer). `sources` restricts the build to the named
    registry entries (training and dev_xfer sources alike). `recipe` names the training mix
    (`S.RECIPES`): "v1" is the 96,000-row spec, "cand_v1" the 20,000-row candidate-stage
    experiment and "cand_mix_v1" those rows plus 20,000 of the "v1" mix; the dev_xfer of
    every recipe is the one of "v1". "intent_hn_v1" needs `negatives`, the directory written
    by `davout train mine-negatives`. "teacher_intent_v1" needs `extra`, a directory with the
    finished rows `train.jsonl` and `dev_in.jsonl` of `davout.train.synth.soft_rows`; they pass
    the per-row guards and the length limit and are then mixed into the train and dev_in sets. `loader(source, split)` replaces
    the Hugging Face loader and `token_counter(prompt_text)` the HRM tokenizer; both exist
    for tests.
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if recipe not in S.RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; available: {', '.join(S.RECIPES)}")
    registry = S.RECIPES[recipe]
    known = {**registry, **S.XFER_SOURCES}
    selected = list(known) if sources is None else list(dict.fromkeys(sources))
    unknown = [name for name in selected if name not in known]
    if unknown:
        raise ValueError(f"unknown source(s) {unknown}; available: {', '.join(known)}")
    check_registry()
    counter = _FnCounter(token_counter) if token_counter is not None else _default_counter()
    b = _Build(seed, scale, loader or hf_loader, counter)
    if (recipe in S.EXTRA_ROW_RECIPES) != (extra is not None):
        raise ValueError(f"recipe {recipe!r} {'needs' if extra is None else 'does not take'} `extra` rows")
    mined_names = [n for n in registry if n in S.MINED_SOURCES and n in selected]
    if mined_names:
        if negatives is None:
            raise ValueError(f"recipe {recipe!r} needs `negatives`: the directory written by `davout train mine-negatives`")
        b.mined = read_mined(negatives, mined_names)
        b.guard = TextGuard([t for src in S.GUARD_TEXTS for split in src.train_splits for t in _column(b.loader(src, split), "text")])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deviations: list[str] = []

    train_names = [n for n in registry if n in selected]
    xfer_names = [n for n in S.XFER_SOURCES if n in selected]
    counts = {n: dict(registry[n].counts) for n in train_names}
    dev_n = {n: registry[n].dev_n for n in train_names}

    # The spec's fallback: without a confirmed "question" act, dailydialog's rows go to yahoo.
    if "dailydialog" in counts:
        src = S.SOURCES["dailydialog"]
        acts = b.pool(src, src.train_splits[0]).rows
        if not dailydialog_acts(acts)["ok"]:
            fb_name, fb_family = S.DAILYDIALOG_FALLBACK
            moved = counts.pop("dailydialog")["ovr"]
            moved_dev = dev_n.pop("dailydialog")
            train_names.remove("dailydialog")
            if fb_name in counts:
                counts[fb_name][fb_family] = counts[fb_name].get(fb_family, 0) + moved
                dev_n[fb_name] += moved_dev
            deviations.append(
                f"dailydialog dropped: act {S.DAILYDIALOG_QUESTION_ACT} is not the act whose utterances end in '?'; "
                f"its {moved} examples moved to {fb_name}/{fb_family}"
            )
    if "massive_intent" in counts and "massive_scenario" in counts:
        b.checks["massive_pair"]["train"] = verify_massive_pair(
            b.pool(S.SOURCES["massive_intent"], "train").rows, b.pool(S.SOURCES["massive_scenario"], "train").rows
        )

    # dev_xfer first, then dev_in, then train: later sets skip any content already held out.
    xfer: list[dict[str, Any]] = []
    if "hwu64" in xfer_names:
        texts: set[str] = set()
        for name in ("massive_intent", "clinc"):
            # another recipe continues from a model trained on "v1", so its dev_xfer stays that of "v1"
            if name in counts or recipe != "v1":
                src = S.SOURCES[name]
                for split in src.train_splits:
                    texts.update(_key(_clean(t)) for t in _column(b.pool(src, split).rows, "text"))
        b.excluded_texts["hwu64"] = texts
    for name in xfer_names:
        src = S.XFER_SOURCES[name]
        for family, n in src.counts.items():
            target = _scaled(n, scale)
            if name == "multirc":  # validation first, at most two per paragraph, then the train split
                rows: list[dict[str, Any]] = []
                limits: Counter[str] = Counter()
                for split in src.dev_splits:
                    b.collect("dev_xfer", src, family, split, target - len(rows), rows, limits)
                    b.shortfall["dev_xfer"].pop(f"{name}/{family}/{split}", None)
                if len(rows) < target:
                    b.shortfall["dev_xfer"][f"{name}/{family}"] = target - len(rows)
                xfer += rows
            else:
                b.collect("dev_xfer", src, family, src.dev_splits[0], target, xfer)

    dev: list[dict[str, Any]] = []
    for name, families in _dev_alloc(counts, dev_n, scale).items():
        src = registry[name]
        splits = src.dev_splits or src.train_splits
        for family, n in families.items():
            for split, share in _split_proportional(n, dict.fromkeys(splits, 1.0)).items():
                b.collect("dev_in", src, family, split, share, dev)

    train: list[dict[str, Any]] = []
    for name in train_names:
        src = registry[name]
        for family, n in counts[name].items():
            for split, share in _split_proportional(_scaled(n, scale), dict.fromkeys(src.train_splits, 1.0)).items():
                b.collect("train", src, family, split, share, train)
    extra_stats: dict[str, Any] = {}
    if extra is not None:
        for name, rows_ in (("train", train), ("dev_in", dev)):
            kept: Counter[str] = Counter()
            for line in (Path(extra) / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                _require(row["source"] not in registry and row["family"] == "choice", f"{row['id']}: not an extra choice row")
                text = render(row).text
                check_example(row, text, b.instructions, b.yelp)
                tokens = b.counter(text)
                if b.counter.has_control_text(text) or tokens > MAX_TOKENS:
                    b.dropped[name][row["source"]]["over_512_tokens_or_control_text"] += 1
                    continue
                b.lengths[name].append(("choice", len(text), tokens))
                kept[row["source"]] += 1
                rows_.append(row)
            extra_stats[name] = dict(kept)
    random.Random(f"{seed}:train-order").shuffle(train)

    # Final guards over the finished sets.
    for rows_ in (train, dev, xfer):
        ids = [r["id"] for r in rows_]
        _require(len(set(ids)) == len(ids), "duplicate row ids")
    held_out = {_content(r) for r in dev} | {_content(r) for r in xfer}
    _require(not any(_content(r) in held_out for r in train), "a train row repeats a dev row")
    _require(not any(b.used[g] & b.reserved for g in b.used), "train and dev share content")
    _require(not {_content(r) for r in dev} & {_content(r) for r in xfer}, "dev_in and dev_xfer overlap")
    _require(not b.uids["train:massive_intent"] & b.uids["train:massive_scenario"],
             "MASSIVE intent and scenario examples share utterance ids")  # fmt: skip

    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "dev_in.jsonl", dev)
    _write_jsonl(out_dir / "dev_xfer.jsonl", xfer)

    def lengths(role: str) -> dict[str, Any]:
        rows_ = b.lengths[role]
        per_family = {f: _stats([t for fam, _c, t in rows_ if fam == f]) for f in FAMILIES}
        return {"all": _stats([t for _f, _c, t in rows_]), "per_family": {f: s for f, s in per_family.items() if s["n"]}}

    sample = b.lengths["train"][:: max(1, len(b.lengths["train"]) // 5000)] or [("", 4, 1)]
    ratios = sorted(c / t for _f, c, t in sample)
    manifest = {
        "seed": seed,
        "scale": scale,
        "sources": selected,
        "files": {
            f"{name}.jsonl": {"rows": len(rows_), "sha256": hashlib.sha256((out_dir / f"{name}.jsonl").read_bytes()).hexdigest()}
            for name, rows_ in (("train", train), ("dev_in", dev), ("dev_xfer", xfer))
        },
        "spec_counts": {n: dict(registry[n].counts) for n in registry},
        "train": _summary(train),
        "dev_in": _summary(dev),
        "dev_xfer": _summary(xfer),
        "augmentation": {role: {k: dict(v) for k, v in sorted(m.items())} for role, m in b.meta.items()},
        "dropped": {role: {s: dict(c) for s, c in d.items() if c} for role, d in b.dropped.items()},
        "shortfall": {role: dict(d) for role, d in b.shortfall.items() if d},
        "deviations": deviations,
        "label_checks": {k: dict(v) for k, v in b.checks.items()},
        "length": {
            "max_tokens": MAX_TOKENS,
            "token_counter": counter.name,
            "wrapping": "<|im_start|><|object_ref_start|>{prompt}<|im_end|>, as HrmBackend._encode",
            "train": lengths("train"),
            "dev_in": lengths("dev_in"),
            "dev_xfer": lengths("dev_xfer"),
            "chars_per_token": {
                "note": "measured with the counter above on a sample of train prompts; the character proxy "
                f"(used only when the HRM tokenizer is missing) assumes {PROXY_CHARS_PER_TOKEN}",
                "sampled": len(sample),
                "mean": round(sum(ratios) / len(ratios), 2),
                "p01": round(ratios[int(0.01 * len(ratios))], 2),
                "min": round(ratios[0], 2),
                "proxy_undercounts": sum(math.ceil(c / PROXY_CHARS_PER_TOKEN) + 3 < t for _f, c, t in sample),
            },
        },
        "guards": {
            "benchmark_instructions_never_used": sorted(b.instructions),
            "yelp_levels_never_used": True,
            "forbidden_wrapper": S.FORBIDDEN_WRAPPER,
            "max_options": MAX_OPTIONS,
            "excluded_datasets": sorted(S.EXCLUDED_DATASETS),
            "clinc_excluded_intents": sorted(S.CLINC_EXCLUDED),
            "massive_intent_scenario_ids_disjoint": True,
            "train_dev_content_disjoint": True,
        },
        "near_domain": S.NEAR_DOMAIN,
        "dataset_revisions": dict(b.revisions),
        "datasets": {n: {"dataset": s.dataset, "config": s.config, "train_splits": list(s.train_splits),
                         "dev_splits": list(s.dev_splits), "table_row": s.number} for n, s in known.items() if n in selected},  # fmt: skip
    }
    if recipe != "v1":
        manifest = {"recipe": recipe, **manifest}
    if extra is not None:
        manifest["extra_rows"] = extra_stats
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
