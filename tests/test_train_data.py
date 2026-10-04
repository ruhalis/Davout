"""Tests for the fine-tuning data builder.

No network and no `datasets` library: the builder is fed hand-written rows shaped like the
schemas read from the real datasets, through its `loader` hook, and a character-based
token counter instead of the HRM tokenizer.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from davout.bench.tasks import TASKS
from davout.prompts import build_candidate_prompt, build_prompt
from davout.schema import ChoiceQuestion, parse_question
from davout.train import data
from davout.train import sources as S
from davout.train.data import DataGuardError

FILES = ("train.jsonl", "dev_in.jsonl", "dev_xfer.jsonl")
ROW_KEYS = {"id", "source", "family", "state", "question", "label", "candidate"}


# -- fake datasets ----------------------------------------------------------------------


def _fake_row(name: str, split: str, i: int) -> dict[str, Any]:
    tag = f"{name} {split} {i}"
    if name in ("mnli", "anli", "rte"):
        return {"premise": f"The report {tag} was filed on Monday.", "hypothesis": f"A report was filed, case {i}.",
                "label": i % 3 if name != "rte" else i % 2}  # fmt: skip
    if name == "wanli":
        return {"premise": f"Premise {tag}.", "hypothesis": f"Hypothesis number {i}.", "gold": S.NLI_NAMES[i % 3]}
    if name == "vitaminc":
        label = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")[i % 3]
        return {"evidence": f"Evidence {tag} says 12 units .", "claim": f"There were over 10 units in case {i} .", "label": label}
    if name == "scitail":
        return {"premise": f"Premise {tag}.", "hypothesis": f"Science fact {i}.", "label": ("entails", "neutral")[i % 2]}
    if name == "squad_v2":
        context = f"Context {tag}: the river is 40 km long."
        answers = {"text": ["40 km"], "answer_start": [0]} if i % 2 else {"text": [], "answer_start": []}
        return {"context": context, "question": f"How long is river {i}?", "answers": answers}
    if name == "pubmedqa":
        return {"question": f"Does drug {i} reduce pain in {split} patients?",
                "context": {"contexts": [f"Background {tag}.", "Results were significant."], "labels": [], "meshes": []},
                "final_decision": "no" if i % 5 == 0 else "yes"}  # fmt: skip
    if name == "qnli":
        return {"question": f"When was item {i} made?", "sentence": f"Sentence {tag} was made in 1990.", "label": i % 2}
    if name == "paws":
        return {"sentence1": f"A {tag} went north .", "sentence2": f"North went a {tag} .", "label": i % 2}
    if name.startswith("tweet_"):
        return {"text": f"@user tweet {tag} #tag", "label": i % (3 if name == "tweet_sentiment" else 2)}
    if name == "yahoo":
        return {"question_title": f"Why is {tag}?", "question_content": "more\\ndetail" if i % 2 else "", "topic": i % 10}
    if name == "newsgroups":
        long = " lorem ipsum dolor" * (200 if i % 7 == 0 else 2)
        return {"text": f"Post {tag} about things.{long}", "label_text": S.NEWSGROUPS.labels[i % 20].raw}
    if name == "dbpedia":
        return {"title": f"Thing {i}", "content": f" Thing {tag} is an entity.", "label": i % 14}
    if name == "clinc":
        return {"text": f"please do {tag}", "intent": i % 151}
    if name in ("massive_intent", "massive_scenario"):
        intent = S.MASSIVE_INTENT_NAMES[i % 60]
        label = intent if name == "massive_intent" else intent.split("_")[0]
        return {"id": f"{split[0]}{i}", "text": f"olly do massive {split} {i}", "label_text": label}
    if name == "trec":
        fine, _orig, coarse = S.TREC_FINE_NAMES[i % 50]
        return {"text": f"What is {tag} ?", "label_text": fine, "label_coarse_text": coarse}
    if name == "go_emotions":
        return {"text": f"wow {tag}", "labels": [i % 28] if i % 5 else [i % 28, (i + 3) % 28]}
    if name == "emotion":
        return {"text": f"i feel {tag}", "label": i % 6}
    if name == "dailydialog":
        return {"utterances": [f"I went there {tag} .", f"Did you go {tag} ?", f"Please come {tag} .", f"Sure , I will {tag} ."],
                "acts": [1, 2, 3, 4]}  # fmt: skip
    if name == "amazon_reviews":
        return {"text": f"Review {tag}: it works.", "label": i % 5}
    if name == "stsb":
        return {"sentence1": f"A man {tag} runs.", "sentence2": f"A person {i} runs.", "score": (i % 6) / 5}
    if name == "civil_comments":
        return {"text": f"Comment {tag}.", "toxicity": (0.0, 0.2, 0.5, 0.9)[i % 4]}
    if name == "hwu64":
        return {"text": f"hey olly {tag}", "label": S.HWU64_NAMES[i % 64]}
    if name == "multirc":
        per = 6 if split == "validation" else 2
        return {"paragraph": f"Paragraph {split} {i // per} tells a story.", "question": f"Who did {i}?",
                "answer": f"Person {i}", "label": i % 2}  # fmt: skip
    if name == "app_reviews":
        return {"review": f"App review {tag}: " + ("works fine on my phone, thanks a lot" if i % 3 else "ok"), "star": 1 + i % 5}
    if name == "ledgar":
        return {"text": f"The parties agree that clause {tag} shall survive termination.", "label": i % 100}
    if name == "fewrel":
        raw, relation, _desc = S.FEWREL_RELATIONS[i % 64]
        return {"relation": raw, "tokens": ["Alpha", str(i), "of", split, "is", "linked", "to", "Beta", "City", "."],
                "head": {"text": "alpha", "type": "Q1", "indices": [[0, 1]]},
                "tail": {"text": "beta city", "type": "Q2", "indices": [[7, 8]]}, "names": [relation, "described"]}  # fmt: skip
    if name in ("clinc_hn", "massive_hn"):
        row = _fake_row(S.MINED_SOURCES[name], split, i)
        if name == "clinc_hn" and i % 50 == 7:
            row["text"] = "i need to know how i can activate the card that arrived today"  # a held-out text
        return row
    if name == "banking77":
        return {"text": "I need to know how I can activate the card that arrived today!", "label": i % 77}
    raise AssertionError(name)


def fake_loader(src: S.Source, split: str) -> list[dict[str, Any]]:
    n = 30 if (src.name, split) == ("multirc", "validation") else 600
    return [_fake_row(src.name, split, i) for i in range(n)]


def count_tokens(text: str) -> int:
    return len(text) // 4


def run_build(out: Path, **kwargs: Any) -> dict:
    kwargs.setdefault("scale", 0.02)
    kwargs.setdefault("loader", fake_loader)
    kwargs.setdefault("token_counter", count_tokens)
    return data.build(out, **kwargs)


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("train_v0")
    return out, run_build(out)


def rows_of(built: tuple[Path, dict], name: str) -> list[dict[str, Any]]:
    return read(built[0] / name)


# -- registry ---------------------------------------------------------------------------


def test_registry_matches_the_spec_totals() -> None:
    data.check_registry()
    per_family: Counter[str] = Counter()
    for src in S.SOURCES.values():
        per_family.update(src.counts)
    assert per_family == {"nli": 24_000, "ovr": 14_000, "candidate": 6_000, "reading": 8_000,
                          "judgement": 4_000, "choice": 28_000, "score": 12_000}  # fmt: skip
    assert sum(per_family.values()) == 96_000
    assert sum(src.dev_n for src in S.SOURCES.values()) == 2000
    assert sorted({src.number for src in S.SOURCES.values()}) == list(range(1, 25))
    assert sum(sum(src.counts.values()) for src in S.XFER_SOURCES.values()) == 1077


def test_registry_excludes_benchmark_datasets_and_banking_intents() -> None:
    used = {src.dataset for src in list(S.SOURCES.values()) + list(S.XFER_SOURCES.values())}
    assert not used & {t.dataset for t in TASKS.values()}
    assert not used & S.EXCLUDED_DATASETS
    assert len(S.CLINC_EXCLUDED) == 32 and S.CLINC_EXCLUDED <= set(S.CLINC_INTENTS)
    kept = {label.raw for label in S.CLINC.labels}
    assert not kept & S.CLINC_EXCLUDED and S.CLINC_OOS not in kept and len(kept) == 118
    assert S.CLINC_INTENTS.index("oos") == 42


def test_label_descriptions_are_written_for_the_listed_sets() -> None:
    described = {name: sum(1 for ls in S.label_sets(name).values() for label in ls.labels if label.desc) for name in S.SOURCES}
    assert {k: v for k, v in described.items() if v} == {
        "yahoo": 10, "newsgroups": 20, "dbpedia": 14, "massive_scenario": 18, "trec": 6, "go_emotions": 28,
        "emotion": 6, "dailydialog": 4,
    }  # fmt: skip


def test_no_training_wording_equals_a_benchmark_instruction() -> None:
    bench = TASKS["ag_news"].question.instructions
    original = S.YAHOO.choice_instructions
    try:  # the label set is frozen; put a benchmark instruction among its paraphrases
        object.__setattr__(S.YAHOO, "choice_instructions", (*original, bench.upper()))
        with pytest.raises(DataGuardError, match="benchmark instruction"):
            data.check_registry()
    finally:
        object.__setattr__(S.YAHOO, "choice_instructions", original)
    data.check_registry()


# -- the module imports without `datasets` ----------------------------------------------


def test_datasets_is_only_needed_to_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasets", None)  # makes `import datasets` fail
    with pytest.raises(RuntimeError, match="datasets"):
        data.hf_loader(S.SOURCES["mnli"], "train")


# -- row conversions --------------------------------------------------------------------


def norm(name: str, row: dict[str, Any]) -> Any:
    return data.normalise(name, row, random.Random(0))


def test_nli_mappings_count_neutral_as_false() -> None:
    row = {"premise": " A dog runs. ", "hypothesis": "An animal moves.", "label": 0}
    assert norm("mnli", row)["cls"] == "entailment"
    assert norm("anli", {**row, "label": 1})["cls"] == "neutral"
    assert norm("mnli", {**row, "label": -1}) == "unlabelled"
    assert norm("wanli", {"premise": "p", "hypothesis": "h", "gold": "contradiction"})["cls"] == "contradiction"
    vit = {"evidence": "e", "claim": "c", "label": "NOT ENOUGH INFO"}
    assert norm("vitaminc", vit)["cls"] == "neutral" and norm("vitaminc", vit)["premise"] == "e"
    assert norm("vitaminc", {**vit, "label": "SUPPORTS"})["cls"] == "entailment"
    assert norm("vitaminc", {**vit, "label": "REFUTES"})["cls"] == "contradiction"
    assert norm("scitail", {"premise": "p", "hypothesis": "h", "label": "entails"})["cls"] == "entailment"
    assert norm("scitail", {"premise": "p", "hypothesis": "h", "label": "neutral"})["cls"] == "neutral"
    for cls, want in (("entailment", 1), ("neutral", 0), ("contradiction", 0)):
        item = {"premise": "A dog runs.", "hypothesis": "An animal moves.", "cls": cls}
        for seed in range(20):
            ex = data.make_example("mnli", "nli", item, random.Random(seed))
            assert ex["label"] == want


def test_reading_and_judgement_mappings() -> None:
    squad = {"context": "The river is 40 km long.", "question": "How long is it?", "answers": {"text": ["40 km"], "answer_start": [13]}}
    assert norm("squad_v2", squad)["cls"] is True
    assert norm("squad_v2", {**squad, "answers": {"text": [], "answer_start": []}})["cls"] is False
    pub = {"question": "Does it work?", "context": {"contexts": ["Aim.", "Result."]}, "final_decision": "yes"}
    assert norm("pubmedqa", pub)["cls"] is True and norm("pubmedqa", pub)["passage"] == "Aim. Result."
    assert norm("pubmedqa", {**pub, "final_decision": "no"})["cls"] is False
    assert norm("pubmedqa", {**pub, "final_decision": "maybe"}) == "unlabelled"
    assert norm("pubmedqa", {**pub, "question": "Do aNALYSIS OF PARS PLANA INCISIONS?"}) == "malformed_question"
    qnli = {"question": "When?", "sentence": "In 1990.", "label": 0}  # 0 = entailment
    assert norm("qnli", qnli)["cls"] is True and norm("qnli", {**qnli, "label": 1})["cls"] is False
    assert norm("qnli", {**qnli, "label": -1}) == "unlabelled"
    paws = {"sentence1": "a b", "sentence2": "b a", "label": 1}
    assert norm("paws", paws)["cls"] is True and norm("paws", {**paws, "label": 0})["cls"] is False
    assert norm("tweet_hate", {"text": "t", "label": 1})["cls"] is True
    # tweet_eval publishes some characters as literal \\uXXXX sequences
    tweet = norm("tweet_sentiment", {"text": "that\\u2019s fun\\u002c really \\ud83d\\ude02 \\ud83d", "label": 2})
    assert tweet["text"] == "that\u2019s fun, really \U0001f602 \ufffd" and tweet["level"] == 2
    json.dumps(tweet["text"], ensure_ascii=False).encode("utf-8")  # no lone surrogate survives
    assert norm("rte", {"premise": "p", "hypothesis": "h", "label": 0})["cls"] == "entailment"
    assert norm("multirc", {"paragraph": "p", "question": "q", "answer": "a", "label": 1})["cls"] is True


def test_classification_rows() -> None:
    assert norm("clinc", {"text": "send money", "intent": S.CLINC_INTENTS.index("transfer")}) == "excluded_intent"
    oos = norm("clinc", {"text": "what is the meaning of xyzzy", "intent": 42})
    assert oos["golds"] == {"intent": []}
    kept = norm("clinc", {"text": "set a timer", "intent": S.CLINC_INTENTS.index("timer")})
    assert kept["golds"]["intent"][0].raw == "timer"
    m = norm("massive_intent", {"id": "17", "text": "wake me up", "label_text": "alarm_set"})
    assert m["golds"]["intent"][0].group == "alarm" and m["uid"] == "17" and m["keys"][0] == "massive-id:17"
    s = norm("massive_scenario", {"id": "17", "text": "wake me up", "label_text": "alarm"})
    assert s["keys"] == m["keys"]  # the same utterance cannot serve both MASSIVE sources
    t = norm("trec", {"text": "Who wrote it ?", "label_text": "an individual", "label_coarse_text": "human beings"})
    assert t["golds"]["fine"][0].group == "human beings" and t["golds"]["coarse"][0].name == "person"
    assert norm("trec", {"text": "x", "label_text": "nonsense", "label_coarse_text": "entities"}) == "unlabelled"
    g = norm("go_emotions", {"text": "thanks!", "labels": [15, 17]})
    assert [label.raw for label in g["golds"]["emotion"]] == ["gratitude", "joy"]
    assert not data.usable("go_emotions", "choice", g) and data.usable("go_emotions", "ovr", g)
    y = norm("yahoo", {"question_title": "Why?", "question_content": "line\\nbreak", "topic": 5})
    assert y["text"] == "Why?\nline break" and y["golds"]["topic"][0].raw == "Sports"
    d = norm("dbpedia", {"title": "ACME", "content": " ACME is a company.", "label": 0})
    assert d["text"] == "ACME is a company." and d["extra"] == {"title": "ACME"}
    n = norm("newsgroups", {"text": "x" * 5000, "label_text": "sci.space"})
    assert len(n["text"]) == data.NEWSGROUPS_MAX_CHARS
    dd = norm("dailydialog", {"utterances": ["Is it far ?"], "acts": [2]})
    assert dd["golds"]["act"][0].name == "question" and dd["sub"] == "0"


def test_score_levels() -> None:
    assert [data.civil_level(t) for t in (0.0, 0.1, 0.30000001192, 0.5, 0.69999998808, 0.8, 1.0)] == [0, 1, 1, 2, 2, 3, 3]
    assert [data.stsb_level(s) for s in (0.0, 0.1, 0.30000001192, 0.5, 0.76, 1.0)] == [0, 1, 2, 3, 4, 5]
    assert norm("amazon_reviews", {"text": "ok", "label": 4})["level"] == 4
    assert norm("app_reviews", {"review": "short", "star": 5}) == "too_short"
    assert norm("app_reviews", {"review": "x" * 40, "star": 1})["level"] == 0
    assert norm("civil_comments", {"text": "hm", "toxicity": 1.5}) == "unlabelled"
    for level, three, two in ((0, 0, 0), (1, 0, 0), (2, 1, None), (3, 2, 1), (4, 2, 1)):
        seen = set()
        for seed in range(200):
            ex = data.score_example("r", "bare", level, "amazon_reviews", random.Random(seed))
            n = len(ex["question"]["criteria"])
            seen.add(n)
            assert ex["label"] == {5: level, 3: three, 2: two}[n]
            assert parse_question("q", ex["question"]).type == "score"
        assert seen == ({5, 3, 2} if two is not None else {5, 3})


def test_score_levels_never_reuse_yelp_strings() -> None:
    yelp = set(TASKS["yelp"].question.criteria)
    for scales in S.SCALES.values():
        for scale in scales:
            for by_text in scale.levels.values():
                for levels in by_text.values():
                    assert not yelp & set(levels)
            assert TASKS["yelp"].question.instructions not in scale.instructions


# -- augmentation -----------------------------------------------------------------------


def test_choice_examples_keep_the_gold_index_and_26_options() -> None:
    labels = S.CLINC
    gold = labels.by_raw()["timer"]
    ks: Counter[int] = Counter()
    notas = others = 0
    for seed in range(4000):
        ex = data.choice_example("set a timer", "bare", gold, labels, random.Random(seed))
        names = list(ex["question"]["criteria"])
        assert 2 <= len(names) <= 26
        picked = names[ex["label"]]
        if ex["meta"]["nota"]:
            notas += 1
            assert picked.lower() in S.OTHER_NAMES
            assert not {data.styled(gold, s) for s in data.NAME_STYLES} & set(names)
        else:
            assert picked == data.styled(gold, ex["meta"]["names"])
            others += ex["meta"]["other"]
        ks[len(names)] += 1
        assert parse_question("q", ex["question"]).type == "choice"
    share = lambda lo, hi: sum(v for k, v in ks.items() if lo <= k <= hi) / 4000  # noqa: E731
    assert 0.50 < share(2, 5) < 0.60 and 0.20 < share(6, 10) < 0.30
    assert 0.09 < share(11, 20) < 0.15 and 0.05 < share(26, 26) < 0.11
    assert 0.06 < notas / 4000 < 0.10
    assert others > notas  # "other" is more often wrong than right, so it gives nothing away


def test_choice_options_are_shuffled_and_described() -> None:
    gold = S.YAHOO.by_raw()["Health"]
    positions: Counter[int] = Counter()
    modes: Counter[str] = Counter()
    styles: Counter[str] = Counter()
    for seed in range(3000):
        ex = data.choice_example("q", "bare", gold, S.YAHOO, random.Random(seed))
        crit = ex["question"]["criteria"]
        modes[ex["meta"]["descriptions"]] += 1
        styles[ex["meta"]["names"]] += 1
        if len(crit) == 4 and not ex["meta"]["nota"]:
            positions[ex["label"]] += 1
        described = [v is not None for v in crit.values()]
        assert all(described) if ex["meta"]["descriptions"] == "all" else not any(described) or ex["meta"]["descriptions"] == "mixed"
    assert len(positions) == 4 and min(positions.values()) > 0.15 * sum(positions.values())
    assert 0.36 < modes["all"] / 3000 < 0.44 and 0.46 < modes["none"] / 3000 < 0.54 and 0.07 < modes["mixed"] / 3000 < 0.13
    assert set(styles) == {"raw", "spaced", "title"}
    oos = data.choice_example("gibberish", "bare", None, S.CLINC, random.Random(1))
    assert list(oos["question"]["criteria"])[oos["label"]].lower() in S.OTHER_NAMES


def test_candidate_examples() -> None:
    labels = S.MASSIVE_INTENT
    gold = labels.by_raw()["alarm_set"]
    siblings = 0
    for seed in range(600):
        want = seed % 4 == 0
        ex = data.candidate_example("wake me up", "bare", gold, labels, random.Random(seed), want)
        q = parse_question("q", ex["question"])
        assert isinstance(q, ChoiceQuestion) and len(q.criteria) <= 26
        assert ex["candidate"] in q.criteria and ex["label"] == int(want)
        gold_name = data.styled(gold, ex["meta"]["names"])
        assert gold_name in q.criteria
        assert (ex["candidate"] == gold_name) == want
        siblings += ex["meta"]["sibling_negative"]
        prompt = build_candidate_prompt(ex["state"], q, ex["candidate"])
        assert prompt.n_labels == 2 and "Proposed answer: " + ex["candidate"] in prompt.text
    assert 0.4 * 450 < siblings < 0.6 * 450  # half of the negatives share the gold intent's scenario
    oos = data.candidate_example("gibberish", "bare", None, S.CLINC, random.Random(3), True)
    assert oos["candidate"].lower() in S.OTHER_NAMES and oos["label"] == 1


def test_one_vs_rest_examples() -> None:
    labels = S.NEWSGROUPS
    gold = labels.by_raw()["rec.autos"]
    forms: Counter[str] = Counter()
    crits: Counter[str] = Counter()
    negated = siblings = negatives = 0
    for seed in range(4000):
        want = seed % 10 < 3
        ex = data.ovr_example("my car broke", "bare", [gold], labels, random.Random(seed), want)
        ins, meta = ex["question"]["instructions"], ex["meta"]
        about_gold = gold.phrase in ins.split(" ")[-1].rstrip(".?") or f" {gold.phrase}" in ins
        assert ex["label"] == int(want) == int(about_gold != meta["negated"])
        assert ins.endswith("?") == (meta["form"] == "question")
        assert not (meta["negated"] and meta["form"] == "question")
        forms[meta["form"]] += 1
        crits[meta["criteria"]] += 1
        negated += meta["negated"]
        if not about_gold:
            negatives += 1
            siblings += meta["sibling_negative"]
        assert parse_question("q", ex["question"]).type == "noul"
    assert 0.05 <= negated / 4000 <= 0.10
    assert 0.52 < crits["none"] / 4000 < 0.58 and 0.22 < crits["both"] / 4000 < 0.28
    assert 0.12 < crits["true"] / 4000 < 0.18 and 0.03 < crits["false"] / 4000 < 0.07
    assert 0.40 < forms["question"] / 4000 < 0.47  # 45% of the rows that are not negated
    assert 0.44 < siblings / negatives < 0.56
    # a row with no gold label (CLINC "oos") is always a plain negative
    ex = data.ovr_example("gibberish", "bare", [], S.CLINC, random.Random(0), False)
    assert ex["label"] == 0 and not ex["meta"]["negated"]
    # negation swaps the criteria: "not about X" is true when the text is about something else
    for seed in range(400):
        ex = data.ovr_example("my car broke", "bare", [gold], labels, random.Random(seed), False)
        crit = ex["question"]["criteria"]
        if ex["meta"]["negated"] and crit and "false" in crit:
            assert crit["false"] == gold.desc
            break
    else:
        raise AssertionError("no negated example with criteria")


def test_noul_forms_and_state_wrappers() -> None:
    forms: Counter[str] = Counter()
    wrappers: Counter[str] = Counter()
    for seed in range(4000):
        ex = data.nli_example("The Cat sat.", "The cat is sitting.", True, S.STYLES["mnli"], random.Random(seed))
        ins = ex["question"]["instructions"]
        forms[ex["meta"]["form"]] += 1
        wrappers[ex["meta"]["wrapper"]] += 1
        assert ins.endswith("?") == (ex["meta"]["form"] == "question")
        prompt = build_prompt(ex["state"], parse_question("q", ex["question"]))
        assert ("Statement: " in prompt.suffix) == (ex["meta"]["form"] == "statement")
        assert S.FORBIDDEN_WRAPPER not in prompt.text
        if ex["meta"]["wrapper"] == "json":
            assert isinstance(ex["state"], dict) and list(ex["state"].values()) == ["The Cat sat."]
        elif ex["meta"]["wrapper"] == "bare":
            assert ex["state"] == "The Cat sat."
        elif ex["meta"]["wrapper"] == "quoted":
            assert ex["state"] == '"The Cat sat."'
        else:
            assert ex["state"].endswith(": The Cat sat.")
    assert 0.52 < forms["statement"] / 4000 < 0.58
    assert 0.47 < wrappers["bare"] / 4000 < 0.53 and 0.27 < wrappers["labelled"] / 4000 < 0.33
    assert 0.12 < wrappers["json"] / 4000 < 0.18 and 0.03 < wrappers["quoted"] / 4000 < 0.07
    assert data.nli_example("p", "Is it so?", True, S.STYLES["mnli"], random.Random(0)) is None
    assert data._clause("The cat sat.") == "the cat sat" and data._clause("Jon left!") == "Jon left"


def test_reading_examples() -> None:
    for seed in range(300):
        ex = data.reading_example("The river is long.", "How long is it?", True, "squad_v2", S.STYLES["squad_v2"], random.Random(seed))
        ins = ex["question"]["instructions"]
        assert ex["label"] == 1 and ins.endswith("?") == (ex["meta"]["form"] == "question")
        if ex["meta"]["wrapper"].startswith("pair"):
            assert "How long is it?" in json.dumps(ex["state"])
        else:
            assert "How long is it?" in ins
        pub = data.pubmed_example("Results were good.", "Does it work?", False, S.STYLES["pubmedqa"], random.Random(seed))
        assert pub["label"] == 0 and pub["question"]["instructions"].lower().endswith("does it work?")
    assert data.pubmed_example("x", "Not a question", True, S.STYLES["pubmedqa"], random.Random(0)) is None


# -- label-mapping checks ---------------------------------------------------------------


def test_verify_rejects_unexpected_label_values() -> None:
    def rows(name: str, n: int = 12, **change: Any) -> list[dict[str, Any]]:
        return [{**_fake_row(name, "train", i), **change} for i in range(n)]

    for name in list(S.SOURCES) + list(S.XFER_SOURCES):
        src = {**S.SOURCES, **S.XFER_SOURCES}[name]
        data.verify_source(src, "train", fake_loader(src, "train"))  # the fake data passes
    with pytest.raises(DataGuardError, match="final_decision"):
        data.verify_source(S.SOURCES["pubmedqa"], "train", rows("pubmedqa", final_decision="maybe"))
    with pytest.raises(DataGuardError, match="toxicity"):
        data.verify_source(S.SOURCES["civil_comments"], "train", rows("civil_comments", toxicity=1.5))
    with pytest.raises(DataGuardError, match="answerable"):
        data.verify_source(S.SOURCES["squad_v2"], "train", rows("squad_v2", answers={"text": ["40 km"], "answer_start": [0]}))
    with pytest.raises(DataGuardError, match="spans"):
        data.verify_source(S.SOURCES["squad_v2"], "train", rows("squad_v2", context="unrelated"))
    with pytest.raises(DataGuardError, match="label_text"):
        data.verify_source(S.SOURCES["massive_intent"], "train", rows("massive_intent", label_text="pay_bill"))
    with pytest.raises(DataGuardError, match="fine, coarse"):
        data.verify_source(S.SOURCES["trec"], "train", rows("trec", label_coarse_text="entities"))
    with pytest.raises(DataGuardError, match="gold"):
        data.verify_source(S.SOURCES["wanli"], "train", rows("wanli", gold="maybe"))
    with pytest.raises(DataGuardError, match="score"):
        data.verify_source(S.SOURCES["stsb"], "train", rows("stsb", score=3.2))
    with pytest.raises(DataGuardError, match="scenario is not the intent prefix"):
        data.verify_massive_pair(rows("massive_intent"), rows("massive_scenario", label_text="weather"))

    class Typed(list):  # a split that carries ClassLabel metadata, like a `datasets.Dataset`
        column_names = ["text", "label"]

        def __init__(self, items: list, names: list[str]) -> None:
            super().__init__(items)
            self.features = {"label": type("ClassLabel", (), {"names": names})()}

        def __getitem__(self, key: Any) -> Any:
            return [r[key] for r in self] if isinstance(key, str) else super().__getitem__(key)

    ok = Typed(rows("tweet_irony"), ["non_irony", "irony"])
    assert "match" in data.verify_source(S.SOURCES["tweet_irony"], "train", ok)["label"]
    with pytest.raises(DataGuardError, match="names are"):
        data.verify_source(S.SOURCES["tweet_irony"], "train", Typed(rows("tweet_irony"), ["irony", "non_irony"]))


def test_dailydialog_question_act_check_and_fallback(tmp_path: Path) -> None:
    good = fake_loader(S.SOURCES["dailydialog"], "train")
    found = data.dailydialog_acts(good)
    assert found["ok"] and found["ends_with_question_mark"] == {"1": 0.0, "2": 1.0, "3": 0.0, "4": 0.0}

    def shifted(src: S.Source, split: str) -> list[dict[str, Any]]:
        rows = fake_loader(src, split)
        if src.name == "dailydialog":  # act ids mean something else: 2 is no longer the question act
            rows = [{**r, "acts": [2, 1, 3, 4]} for r in rows]
        return rows

    assert not data.dailydialog_acts(shifted(S.SOURCES["dailydialog"], "train"))["ok"]
    names = ["yahoo", "dailydialog"]
    base = run_build(tmp_path / "a", sources=names)
    moved = run_build(tmp_path / "b", sources=names, loader=shifted)
    assert base["train"]["per_source"]["dailydialog"]["ovr"] == 40 and not base["deviations"]
    assert "dailydialog" not in moved["train"]["per_source"] and "dailydialog dropped" in moved["deviations"][0]
    assert moved["train"]["per_source"]["yahoo"]["ovr"] == base["train"]["per_source"]["yahoo"]["ovr"] + 40
    assert moved["train"]["rows"] == base["train"]["rows"]


# -- per-row guards ---------------------------------------------------------------------


def test_check_example_guards() -> None:
    instructions, yelp, _ = data.benchmark_texts()

    def check(row: dict[str, Any]) -> None:
        data.check_example(row, data.render(row).text, instructions, yelp)

    ok = {"id": "x", "source": "s", "family": "nli", "state": "A text.", "label": 1,
          "question": {"type": "noul", "instructions": "It is a text.", "criteria": None}}  # fmt: skip
    check(ok)
    for task in TASKS.values():
        if task.kind == "noul":
            with pytest.raises(DataGuardError, match="benchmark instruction"):
                check({**ok, "question": {"type": "noul", "instructions": task.question.instructions, "criteria": None}})
    with pytest.raises(DataGuardError, match="forbidden wrapper"):
        check({**ok, "state": "The user wrote this message: hi"})
    with pytest.raises(DataGuardError, match="noul label"):
        check({**ok, "label": 2})
    many = {"type": "choice", "instructions": "Pick one.", "criteria": {f"option {i}": None for i in range(27)}}
    with pytest.raises(DataGuardError, match="27 options"):
        check({**ok, "family": "candidate", "question": many, "candidate": "option 3", "label": 1})
    few = {"type": "choice", "instructions": "Pick one.", "criteria": {"a": None, "b": None}}
    with pytest.raises(DataGuardError, match="out of range"):
        check({**ok, "family": "choice", "question": few, "label": 2})
    with pytest.raises(DataGuardError, match="needs a noul question"):
        check({**ok, "question": few, "label": 1})
    levels = {"type": "score", "instructions": "How good?", "criteria": list(TASKS["yelp"].question.criteria)}
    with pytest.raises(DataGuardError, match="Yelp"):
        check({**ok, "family": "score", "question": levels, "label": 0})
    with pytest.raises(DataGuardError, match="not an option"):
        data.check_example({**ok, "family": "candidate", "question": few, "candidate": "c"}, "", instructions, yelp)


# -- the built files --------------------------------------------------------------------


def test_build_writes_the_contract(built: tuple[Path, dict]) -> None:
    out, manifest = built
    assert sorted(p.name for p in out.iterdir()) == ["dev_in.jsonl", "dev_xfer.jsonl", "manifest.json", "train.jsonl"]
    assert json.loads((out / "manifest.json").read_text()) == manifest
    for name in FILES:
        entry = manifest["files"][name]
        assert entry["rows"] == len(read(out / name))
        assert entry["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    bench = {t.question.instructions for t in TASKS.values()}
    seen_families: set[str] = set()
    for name in FILES:
        rows = rows_of(built, name)
        assert rows and len({r["id"] for r in rows}) == len(rows)
        for r in rows:
            assert set(r) <= ROW_KEYS and set(r) >= ROW_KEYS - {"candidate"}
            assert r["family"] in data.FAMILIES and isinstance(r["state"], (str, dict))
            assert set(r["question"]) == {"type", "instructions", "criteria"}
            q = parse_question("q", r["question"])
            assert r["question"]["instructions"] not in bench
            assert isinstance(r["label"], int) and not isinstance(r["label"], bool)
            if r["family"] == "candidate":
                assert q.type == "choice" and r["candidate"] in q.criteria and r["label"] in (0, 1)
                prompt = build_candidate_prompt(r["state"], q, r["candidate"])
            else:
                assert "candidate" not in r
                prompt = build_prompt(r["state"], q)
                if r["family"] in ("choice", "score"):
                    assert q.type == r["family"] and 0 <= r["label"] < len(q.criteria)
                else:
                    assert q.type == "noul" and r["label"] in (0, 1)
            if q.type == "choice":
                assert 2 <= len(q.criteria) <= 26
            assert prompt.text == data.render(r).text and count_tokens(prompt.text) <= data.MAX_TOKENS
            assert S.FORBIDDEN_WRAPPER not in prompt.text
            seen_families.add(r["family"])
    assert seen_families == set(data.FAMILIES)


def test_build_counts_follow_the_registry(built: tuple[Path, dict]) -> None:
    _, manifest = built
    assert not manifest["shortfall"] and not manifest["deviations"]
    for name, src in S.SOURCES.items():
        got = manifest["train"]["per_source"][name]
        for family, n in src.counts.items():
            assert got[family] == max(1, round(n * 0.02)), (name, family)
    assert manifest["train"]["rows"] == sum(max(1, round(n * 0.02)) for s in S.SOURCES.values() for n in s.counts.values())
    assert set(manifest["dev_in"]["per_source"]) == set(S.SOURCES)
    assert manifest["dev_in"]["rows"] == sum(max(1, round(s.dev_n * 0.02)) for s in S.SOURCES.values())
    assert manifest["dev_xfer"]["per_source"] == {
        "hwu64": {"choice": 3, "candidate": 3, "total": 6}, "rte": {"nli": 6, "total": 6},
        "multirc": {"reading": 5, "total": 5}, "app_reviews": {"score": 5, "total": 5},
    }  # fmt: skip
    rates = manifest["train"]["positive_rate"]
    assert abs(rates["nli"] - 0.475) < 0.02 and abs(rates["reading"] - 0.5) < 0.02
    assert abs(rates["ovr"] - 0.30) < 0.03 and abs(rates["candidate"] - 0.25) < 0.03
    for key in ("seed", "scale", "train", "dev_in", "dev_xfer", "dropped", "near_domain", "dataset_revisions",
                "label_checks", "length", "guards", "augmentation"):  # fmt: skip
        assert key in manifest
    assert manifest["length"]["train"]["all"]["max"] <= 512
    assert set(manifest["near_domain"]) >= {"ag_news", "yelp", "sst2", "boolq", "banking77", "sms_spam"}
    assert manifest["label_checks"]["dailydialog"]["train"]["ok"]
    assert manifest["label_checks"]["pubmedqa"]["train"]["final_decision"] == {"no": 120, "yes": 480}


def test_long_prompts_and_excluded_rows_are_dropped(built: tuple[Path, dict]) -> None:
    _, manifest = built
    dropped = manifest["dropped"]["train"]
    assert dropped["newsgroups"]["over_512_tokens"] > 0  # every seventh fake post is far too long
    assert dropped["clinc"]["excluded_intent"] > 0
    assert dropped["massive_scenario"]["used_by_sibling_source"] > 0
    banned = {name.replace("_", " ") for name in S.CLINC_EXCLUDED} | {"oos"}
    for r in rows_of(built, "train.jsonl"):
        if r["source"] == "clinc" and r["family"] != "ovr":
            assert not banned & {n.lower().replace("_", " ") for n in r["question"]["criteria"]}


def test_train_and_dev_sets_are_disjoint(built: tuple[Path, dict]) -> None:
    train, dev, xfer = (rows_of(built, name) for name in FILES)

    def content(r: dict[str, Any]) -> str:
        return json.dumps([r["state"], r["question"]], sort_keys=True)

    assert not {content(r) for r in train} & ({content(r) for r in dev} | {content(r) for r in xfer})
    assert not {r["id"] for r in train} & ({r["id"] for r in dev} | {r["id"] for r in xfer})
    assert {r["source"] for r in xfer} == set(S.XFER_SOURCES) and not {r["source"] for r in train} & set(S.XFER_SOURCES)
    # MASSIVE: no utterance id serves both the intent and the scenario source
    ids = {s: {r["id"].split("/")[-1] for r in train if r["source"] == s} for s in ("massive_intent", "massive_scenario")}
    assert ids["massive_intent"] and ids["massive_scenario"] and not ids["massive_intent"] & ids["massive_scenario"]
    # a text is used once per source, whatever the family (the fake texts carry "<source> <split> <i>")
    for source in ("clinc", "yahoo", "trec", "go_emotions"):
        rows = [r for r in train + dev if r["source"] == source]
        tags = [re.search(rf"{source} \w+ \d+", json.dumps(r["state"])).group(0) for r in rows]
        assert len({r["family"] for r in rows}) >= 2 and len(set(tags)) == len(tags)


def test_dev_xfer_uses_one_canonical_format_per_source(built: tuple[Path, dict], tmp_path: Path) -> None:
    out = tmp_path / "xfer"
    run_build(out, scale=0.2, sources=list(S.XFER_SOURCES))
    rows = read(out / "dev_xfer.jsonl")
    assert not (out / "train.jsonl").read_text() and not (out / "dev_in.jsonl").read_text()
    by = {name: [r for r in rows if r["source"] == name] for name in S.XFER_SOURCES}
    assert [len(by[n]) for n in ("hwu64", "rte", "multirc", "app_reviews")] == [60, 55, 50, 50]
    for r in by["hwu64"]:
        assert isinstance(r["state"], str) and len(r["question"]["criteria"]) == 10
        assert r["question"]["instructions"] == S.HWU64.choice_instructions[0]
        assert set(r["question"]["criteria"].values()) == {None}
    cands = [r for r in by["hwu64"] if r["family"] == "candidate"]
    assert len(cands) == 30 and sum(r["label"] for r in cands) == 10
    for r in by["rte"]:
        assert r["family"] == "nli" and r["question"]["criteria"] is None and isinstance(r["state"], str)
    for r in by["multirc"]:
        assert set(r["state"]) == {"passage", "question", "answer"} and r["family"] == "reading"
        assert r["question"]["instructions"] == S.XFER_MULTIRC_INSTRUCTIONS
    per_paragraph = Counter(r["state"]["passage"] for r in by["multirc"])
    assert max(per_paragraph.values()) <= 2
    assert sum(r["id"].startswith("multirc/validation/") for r in by["multirc"]) == 10  # 5 paragraphs x 2, then train
    assert Counter(r["label"] for r in by["app_reviews"]) == {0: 10, 1: 10, 2: 10, 3: 10, 4: 10}
    assert all(len(r["state"]) >= 40 and r["question"]["criteria"] == list(S.XFER_APP_REVIEWS_LEVELS) for r in by["app_reviews"])


def test_hwu_rows_seen_in_training_corpora_are_left_out(tmp_path: Path) -> None:
    def loader(src: S.Source, split: str) -> list[dict[str, Any]]:
        rows = fake_loader(src, split)
        if src.name == "hwu64":  # the first half of HWU repeats MASSIVE train utterances
            rows = [{**r, "text": f"OLLY do  massive train {i}"} if i < 300 else r for i, r in enumerate(rows)]
        return rows

    manifest = run_build(tmp_path, scale=0.2, sources=["massive_intent", "hwu64"], loader=loader)
    assert manifest["dropped"]["dev_xfer"]["hwu64"]["text_in_training_corpus"] > 0
    assert all("massive train" not in r["state"] for r in read(tmp_path / "dev_xfer.jsonl"))


def test_build_is_deterministic(tmp_path: Path) -> None:
    names = ["mnli", "squad_v2", "yahoo", "clinc", "massive_intent", "massive_scenario", "stsb", "hwu64", "multirc"]
    a = run_build(tmp_path / "a", sources=names)
    b = run_build(tmp_path / "b", sources=names)
    c = run_build(tmp_path / "c", sources=names, seed=1)
    assert a == b
    for name in FILES + ("manifest.json",):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert (tmp_path / "a" / "train.jsonl").read_bytes() != (tmp_path / "c" / "train.jsonl").read_bytes()
    rows = read(tmp_path / "a" / "train.jsonl")
    assert set(a["train"]["per_source"]) == set(names) - {"hwu64", "multirc"}
    # the file order is a shuffle across sources, not source by source
    first = [r["source"] for r in rows[:60]]
    assert len(set(first)) >= 4


def test_trainer_reader_accepts_the_files(built: tuple[Path, dict]) -> None:
    fmt = pytest.importorskip("davout.train.format")
    for name in FILES:
        rows = fmt.read_rows(built[0] / name)
        assert len(rows) == len(rows_of(built, name))


def test_hard_negatives_and_shortlists_use_similar_names() -> None:
    labels = S.LEDGAR
    gold = labels.by_raw()["Waivers"]
    near = data.name_neighbours(labels)["Waivers"]
    near_names = {label.name for label in near}
    assert len(near) == data.NAME_NEIGHBOURS and gold not in near
    assert {"No Waivers", "Waiver Of Jury Trials"} <= {label.raw for label in near[:3]}
    hard = 0
    for seed in range(400):
        ex = data.candidate_example("clause", "bare", gold, labels, random.Random(seed), False)
        assert ex["label"] == 0 and ex["candidate"].lower() != "waivers"
        hard += ex["meta"]["sibling_negative"]
        assert not ex["meta"]["sibling_negative"] or ex["candidate"].lower() in near_names
    assert 0.4 * 400 < hard < 0.6 * 400  # half of the negatives have a name close to the gold one
    positions: Counter[int] = Counter()
    for seed in range(200):
        ex = data.shortlist_example("clause", "bare", gold, labels, random.Random(seed))
        names = [n.lower() for n in ex["question"]["criteria"]]
        assert len(names) == data.SHORTLIST_OPTIONS and names[ex["label"]] == "waivers"
        assert sum(n in near_names for n in names) >= data.SHORTLIST_NEAR
        assert not set(names) & {n.lower() for n in S.OTHER_NAMES}
        positions[ex["label"]] += 1
    assert len(positions) == data.SHORTLIST_OPTIONS  # the gold option moves around


def test_candidate_recipe_builds_candidate_rows_and_shortlists(tmp_path: Path, built: tuple[Path, dict]) -> None:
    manifest = run_build(tmp_path, scale=0.02, recipe="cand_v1")
    assert manifest["recipe"] == "cand_v1" and "recipe" not in built[1]
    rows = read(tmp_path / "train.jsonl")
    assert Counter((r["source"], r["family"]) for r in rows) == {
        ("ledgar", "candidate"): 140, ("ledgar", "choice"): 60, ("fewrel", "candidate"): 140, ("fewrel", "choice"): 60,
    }  # fmt: skip
    assert all(len(r["question"]["criteria"]) == 10 for r in rows if r["family"] == "choice")
    assert manifest["train"]["positive_rate"]["candidate"] == pytest.approx(0.25)
    relation = next(r for r in rows if r["source"] == "fewrel")
    text = data.render(relation).text.lower()
    assert "subject" in text and "object" in text and "beta city" in text
    assert {r["source"] for r in read(tmp_path / "dev_in.jsonl")} == {"ledgar", "fewrel"}
    # dev_xfer is the one of the default recipe, and the default mix never sees the new sources
    assert (tmp_path / "dev_xfer.jsonl").read_bytes() == (built[0] / "dev_xfer.jsonl").read_bytes()
    assert not {"ledgar", "fewrel"} & (set(S.SOURCES) | set(built[1]["train"]["per_source"]))
    assert not {r["id"] for r in rows} & {r["id"] for r in read(tmp_path / "dev_in.jsonl")}
    fmt = pytest.importorskip("davout.train.format")
    assert len(fmt.read_rows(tmp_path / "train.jsonl")) == len(rows)


def test_candidate_mix_recipe_adds_default_rows_in_proportion(tmp_path: Path) -> None:
    mix = S.CAND_MIX_SOURCES
    v1_part = {n: s for n, s in mix.items() if n in S.SOURCES}
    assert sum(sum(s.counts.values()) for s in v1_part.values()) == 20_000 and set(v1_part) == set(S.SOURCES)
    per_family: Counter[str] = Counter()
    for src in v1_part.values():
        per_family.update(src.counts)
        assert src.dev_n == 0 and set(src.counts) == set(S.SOURCES[src.name].counts)
    full: Counter[str] = Counter()
    for src in S.SOURCES.values():
        full.update(src.counts)
    assert all(abs(per_family[f] - full[f] * 20_000 / 96_000) <= 3 for f in full)
    assert S.SOURCES["mnli"].counts == {"nli": 10_000} and S.SOURCES["mnli"].dev_n == 84  # the default mix is untouched

    run_build(tmp_path / "cand", scale=0.05, recipe="cand_v1")
    manifest = run_build(tmp_path / "mix", scale=0.05, recipe="cand_mix_v1")
    rows = read(tmp_path / "mix" / "train.jsonl")
    assert abs(len(rows) - 2 * 1000) <= 5 and manifest["shortfall"] == {}  # each cell is rounded at this scale
    new = sorted(json.dumps(r, sort_keys=True) for r in rows if r["source"] in S.CAND_SOURCES)
    assert new == sorted(json.dumps(r, sort_keys=True) for r in read(tmp_path / "cand" / "train.jsonl"))
    assert len({r["source"] for r in rows[:100]}) >= 6  # shuffled across both parts
    for name in ("dev_in.jsonl", "dev_xfer.jsonl"):
        assert (tmp_path / "mix" / name).read_bytes() == (tmp_path / "cand" / name).read_bytes()


class _RankBackend:
    """Candidate stage that prefers labels in label order: logit of Yes falls with the prompt's position."""

    max_labels = 26

    def __init__(self) -> None:
        self.prompts: list[Any] = []

    def read(self, prompts: Any) -> list[Any]:
        from davout.backends.base import Readout

        self.prompts += prompts
        return [Readout([-float(len(p.suffix)), 0.0], 5) for p in prompts]


def test_mined_recipe_builds_balanced_shortlists_from_the_mined_ranking(tmp_path: Path, built: tuple[Path, dict]) -> None:
    from davout.train.mine import KEEP, mine

    backend = _RankBackend()
    stats = mine(tmp_path / "neg", backend, loader=fake_loader)
    assert set(stats) == {"clinc_hn", "massive_hn"} and stats["massive_hn"]["rows"] == 600
    assert stats["clinc_hn"]["rows"] < 600  # banking intents and out-of-scope rows are never mined
    assert "Proposed answer: " in backend.prompts[0].suffix and backend.prompts[0].n_labels == 2
    mined = data.read_mined(tmp_path / "neg", ["clinc_hn", "massive_hn"])
    assert all(len(w) == KEEP for w in mined["massive_hn"].values())

    with pytest.raises(ValueError, match="negatives"):
        run_build(tmp_path / "none", recipe="intent_hn_v1")
    manifest = run_build(tmp_path / "hn", scale=0.02, recipe="intent_hn_v1", negatives=tmp_path / "neg")
    rows = read(tmp_path / "hn" / "train.jsonl")
    new = [r for r in rows if r["source"] in S.MINED_SOURCES]
    assert Counter(r["source"] for r in new) == {"clinc_hn": 201, "massive_hn": 201} and manifest["shortfall"] == {}
    assert abs(len(rows) - len(new) - 400) <= 10  # each default-mix cell is rounded at this scale
    by_raw = {"clinc_hn": S.CLINC.by_raw(), "massive_hn": S.MASSIVE_INTENT.by_raw()}
    for r in new:
        names = [n.lower().replace("_", " ") for n in r["question"]["criteria"]]
        assert r["family"] == "choice" and len(names) == 10
        wrong = [by_raw[r["source"]][raw].name for raw in mined[r["source"]][r["id"]][:9]]
        assert sorted(names) == sorted(wrong + [names[r["label"]]]) and names[r["label"]] not in wrong
    assert len({r["label"] for r in new}) == 10  # the gold option moves around
    per_intent = Counter(r["question"]["criteria"] and list(r["question"]["criteria"])[r["label"]].lower().replace("_", " ")
                         for r in new if r["source"] == "massive_hn")  # fmt: skip
    assert len(per_intent) == 60 and max(per_intent.values()) - min(per_intent.values()) <= 1
    # held-out texts are dropped, and dev_xfer stays the default one
    assert manifest["dropped"]["train"]["clinc_hn"]["held_out_text_overlap"] > 0
    assert not any("activate the card" in json.dumps(r["state"]) for r in rows + read(tmp_path / "hn" / "dev_in.jsonl"))
    assert {r["source"] for r in read(tmp_path / "hn" / "dev_in.jsonl")} == set(S.MINED_SOURCES)
    assert (tmp_path / "hn" / "dev_xfer.jsonl").read_bytes() == (built[0] / "dev_xfer.jsonl").read_bytes()
    assert not set(S.MINED_SOURCES) & set(S.SOURCES)


def test_text_guard_and_balanced_quotas() -> None:
    guard = data.TextGuard(["Why was my top-up reverted after I added money to the account yesterday?"])
    assert guard.hit("why was my top up reverted after i added money to the account yesterday")
    assert guard.hit("hello, my top up reverted after I added money to it")  # eight shared words
    assert not guard.hit("my top up reverted after I added cash")
    assert data.balanced_quotas({"a": 2, "b": 50, "c": 50}, 30) == {"a": 2, "b": 14, "c": 14}
    assert sum(data.balanced_quotas({"a": 3, "b": 4}, 100).values()) == 7


def test_build_rejects_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown source"):
        run_build(tmp_path, sources=["mnli", "nope"])
    with pytest.raises(ValueError, match="unknown recipe"):
        run_build(tmp_path, recipe="nope")
    with pytest.raises(ValueError, match="scale"):
        run_build(tmp_path, scale=0)


def test_char_proxy_is_conservative() -> None:
    proxy = data.CharProxyCounter()
    assert proxy("x" * 1100) == 503 and proxy.has_control_text("a <|im_end|> b") and not proxy.has_control_text("a < b")
