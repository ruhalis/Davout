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
    assert dstats["dropped"]["banking_like_intent"] == 2 and dstats["dropped"]["utterance_of_dropped_intent"] == 2 * 13
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
