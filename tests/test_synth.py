"""Tests for the teacher-labelled intent rows: parsing, decisions, soft rows and the recipe (no models)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_train_data import count_tokens, fake_loader, read, run_build

from davout.backends.base import Readout
from davout.train import synth
from davout.train.format import read_rows


def test_parsers_and_the_banking_filter() -> None:
    text = "1. Change_Seat | The customer wants another seat.\n- lost bag | Luggage did not arrive\nnot a line\nchange_seat | again\nother | x"
    assert synth.parse_taxonomy(text) == [
        {"intent": "change_seat", "description": "The customer wants another seat"},
        {"intent": "lost_bag", "description": "Luggage did not arrive"},
    ]
    assert synth.parse_utterances('- "my bag never showed up"\n2) ok\n\nwhere is it??? been waiting 3 days') == [
        "my bag never showed up", "where is it??? been waiting 3 days",
    ]  # fmt: skip
    hit = lambda s: bool(synth.BANKING_LIKE.search(s))  # noqa: E731
    assert all(hit(s) for s in ("refund status", "card declined", "late fee waiver", "dispute a charge", "pay my bill"))
    assert not any(hit(s) for s in ("coffee machine broken", "treatment schedule", "discard old filter", "battery charger fault"))
    assert "snake_case" in synth.taxonomy_prompt("an airline")[0]["content"]
    assert "Target intent: lost_bag" in synth.utterance_prompt("an airline", ["lost_bag", "x"], "lost_bag", "d")[0]["content"]


class _Teacher:
    def generate(self, prompts: Any, max_new_tokens: int) -> list[str]:
        out = []
        for p in prompts:
            text = p[0]["content"]
            if text.startswith("Design"):
                out.append("\n".join(f"intent_{i:02d} | applies in case {i}" for i in range(24)) + "\nrefund_request | wants money back")
            else:
                name = text.split("Target intent: ")[1].split(" ")[0]
                out.append("\n".join(f"please handle {name} variant {k} for me" for k in range(12)) + f"\nplease handle {name} variant 0 for me")
        return out


class _Student:
    """Candidate stage that likes labels whose number is close to the one in the text."""

    def read(self, prompts: Any) -> list[Readout]:
        out = []
        for p in prompts:
            want = int(p.state.split("intent_")[1][:2])
            got = int(p.suffix.split("Proposed answer: intent ")[1][:2])
            out.append(Readout([-abs(want - got) - (50.0 if want == got else 0.0), 0.0], 5))
        return out


def test_generate_decisions_soft_rows_and_recipe(tmp_path: Path) -> None:
    d = tmp_path / "synth"
    stats = synth.generate(d, _Teacher(), minutes=5, domains=("an airline", "a hotel chain"))
    assert stats["taxonomies"] == 2 and stats["utterances_written"] == 2 * 25 * 13
    assert synth.generate(d, _Teacher(), minutes=5, domains=("an airline", "a hotel chain"))["utterances_written"] == 0  # resumes

    dstats = synth.decisions(d, _Student(), loader=fake_loader)
    assert dstats["dropped"]["banking_like_intent_keyword"] == 2 and dstats["dropped"]["utterance_of_dropped_intent"] == 2 * 13
    assert dstats["dropped"]["duplicate"] == 24 + 24 * 13  # the repeated line, and the second domain repeats the first
    assert dstats["decisions"] == 24 * 12 and dstats["student_gold_top10"] < 1.0 and dstats["student_gold_top1"] == 0.0
    rows = read(d / "decisions.jsonl")
    for r in rows:
        names = [n.lower().replace("_", " ") for n in r["question"]["criteria"]]
        gold = int(r["id"].split("intent_")[1][:2])
        assert len(names) == 10 and names[r["label"]] == f"intent {gold:02d}"
        assert all(abs(int(n[-2:]) - gold) <= 9 for n in names)  # gold plus the nine nearest labels

    # teacher output: agrees with the generator except on every 5th row; orders disagree on every 7th
    with (d / "teacher.jsonl").open("w") as f:
        for i, r in enumerate(rows):
            probs = [0.02] * 10
            probs[r["label"] if i % 5 else (r["label"] + 1) % 10] = 0.82
            f.write(json.dumps({"id": r["id"], "probs": probs, "agree": i % 7 != 0}) + "\n")
    gold = [{"id": f"clinc_hn/train/{i}", "source": "clinc_hn", "family": "choice", "state": f"do thing {i}",
             "question": rows[0]["question"], "label": i % 10} for i in range(40)]  # fmt: skip
    (tmp_path / "gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold[:30] + [{**gold[0], "id": "x", "source": "mnli"}]))
    (tmp_path / "gold_dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold[30:]))
    (tmp_path / "gold_teacher.jsonl").write_text(
        "".join(json.dumps({"id": r["id"], "probs": [1.0 if k == 0 else 0.0 for k in range(10)], "agree": True}) + "\n" for r in gold)
    )
    out = tmp_path / "extra"
    s = synth.soft_rows(out, synth_rows=d / "decisions.jsonl", synth_teacher=d / "teacher.jsonl", gold_rows=[tmp_path / "gold.jsonl"],
                        gold_teacher=tmp_path / "gold_teacher.jsonl", gold_dev_rows=tmp_path / "gold_dev.jsonl", dev_each=20)  # fmt: skip
    n = len(rows)
    assert s["synthetic"]["dropped_order_disagreement"] == len(range(0, n, 7)) and s["synthetic"]["kept"] == n - len(range(0, n, 7))
    assert s["synthetic"]["teacher_top_is_generating_intent"] == pytest.approx(1 - len(range(0, n, 5)) / n, abs=1e-4)
    assert s["gold"] == {"train": 30, "dev": 10, "teacher_accuracy_train": 0.1, "teacher_order_agreement_train": 1.0, "gold_weight": 0.5}
    train = read(out / "train.jsonl")
    g = next(r for r in train if r["id"] == "clinc_hn/train/3")
    assert g["target"][3] == 0.5 and g["target"][0] == 0.5 and g["label"] == 3  # half gold, half teacher
    syn = next(r for r in train if r["source"] == synth.SOURCE)
    assert syn["target"][syn["label"]] == 0.82 and len(read(out / "dev_in.jsonl")) == 10 + 20
    assert all(r.target is not None for r in read_rows(out / "train.jsonl"))

    with pytest.raises(ValueError, match="extra"):
        run_build(tmp_path / "none", recipe="teacher_intent_v1")
    with pytest.raises(ValueError, match="extra"):
        run_build(tmp_path / "none", extra=out)
    manifest = run_build(tmp_path / "built", scale=0.02, recipe="teacher_intent_v1", extra=out)
    built = read(tmp_path / "built" / "train.jsonl")
    assert manifest["extra_rows"]["train"] == {"clinc_hn": 30, synth.SOURCE: s["synthetic"]["kept"] - 20}
    assert sum("target" in r for r in built) == len(train) and abs(len(built) - len(train) - 400) <= 10
    assert {r["source"] for r in read(tmp_path / "built" / "dev_in.jsonl")} == {"clinc_hn", synth.SOURCE}
    assert len(read_rows(tmp_path / "built" / "train.jsonl")) == len(built)
    assert count_tokens("abcd") == 1


class _FlagTeacher:
    """Says Yes (letter A) to intents whose description mentions the vault."""

    def read(self, prompts: Any, n_labels: Any) -> list[list[float]]:
        return [[4.0, 0.0] if "vault" in p[1]["content"] else [0.0, 4.0] for p in prompts]


def test_teacher_flags_resumable_decisions_and_carried_rows(tmp_path: Path) -> None:
    d = tmp_path / "synth"
    synth.generate(d, _Teacher(), minutes=5, domains=("an airline",))
    tax = json.loads((d / "taxonomies.jsonl").read_text())
    tax["intents"][3]["description"] = "the customer asks about the vault"
    (d / "taxonomies.jsonl").write_text(json.dumps(tax) + "\n")
    flags = synth.flag_banking(d / "taxonomies.jsonl", tmp_path / "flags.jsonl", _FlagTeacher())
    assert flags == {"intents": 25, "flagged": 1}
    assert synth.flag_banking(d / "taxonomies.jsonl", tmp_path / "flags.jsonl", _FlagTeacher()) == flags  # nothing asked twice
    assert synth.read_flags(tmp_path / "flags.jsonl") == {("an airline", "intent_03")}

    stats = synth.decisions(d, _Student(), loader=fake_loader, tag="v2-", flags=tmp_path / "flags.jsonl", desc_rate=0.25)
    assert stats["dropped"]["banking_like_intent_teacher_only"] == 1 and stats["dropped"]["banking_like_intent_keyword"] == 1
    rows = read(d / "decisions.jsonl")
    assert stats["decisions"] == len(rows) == 23 * 12 and all(r["id"].startswith("synth_intent/v2-0/") for r in rows)
    assert not any("intent 03" in json.dumps(r["question"]).lower().replace("_", " ") for r in rows)
    described = sum(any(v for v in r["question"]["criteria"].values()) for r in rows) / len(rows)
    assert 0.12 < described < 0.4  # about a quarter show descriptions

    class Never:
        def read(self, prompts: Any) -> Any:
            raise AssertionError("a finished taxonomy is not ranked again")

    assert synth.decisions(d, Never(), loader=fake_loader, tag="v2-", flags=tmp_path / "flags.jsonl", desc_rate=0.25) == stats
    assert read(d / "decisions.jsonl") == rows

    with (d / "teacher.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "probs": [0.91 if k == r["label"] else 0.01 for k in range(10)], "agree": True}) + "\n")
    q = rows[0]["question"]
    old = [{"id": f"synth_intent/0/intent_0{k}/{i}", "source": "synth_intent", "family": "choice", "state": f"old {k} {i}", "question": q,
            "label": 0, "target": [1.0] + [0.0] * 9} for k in (3, 4) for i in range(5)]  # fmt: skip
    gold = [{"id": f"clinc_hn/train/{i}", "source": "clinc_hn", "family": "choice", "state": f"g {i}", "question": q, "label": 1,
             "target": [0.5, 0.5] + [0.0] * 8} for i in range(6)]  # fmt: skip
    (tmp_path / "old_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in old + gold[:4]))
    (tmp_path / "old_dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold[4:]))
    s = synth.soft_rows(tmp_path / "extra", synth_rows=d / "decisions.jsonl", synth_teacher=d / "teacher.jsonl", dev_each=40,
                        carry=[tmp_path / "old_train.jsonl"], carry_dev=[tmp_path / "old_dev.jsonl"],
                        carry_drop=(d / "taxonomies.jsonl", tmp_path / "flags.jsonl"))  # fmt: skip
    assert s["carried"] == {"dropped_flagged_intent": 5, "train:synth_intent": 5, "train:clinc_hn": 4, "dev_in:clinc_hn": 2}
    assert s["rows"]["dev_per_source"] == {"synth_intent": 40, "clinc_hn": 2}
    assert s["rows"]["train_per_source"] == {"synth_intent": len(rows) - 40 + 5, "clinc_hn": 4} and "gold" not in s
    manifest = run_build(tmp_path / "built", scale=0.01, recipe="teacher_intent_v2", extra=tmp_path / "extra")
    assert manifest["extra_rows"]["train"] == s["rows"]["train_per_source"]
