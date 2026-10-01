"""Tests for benchmark tasks: JSONL loading, sampling and the registry's row mappings.

No network and no `datasets` library: registry specs are fed hand-written rows
shaped like the schema verified from the dataset metadata.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from davout import cli
from davout.bench.tasks import TASKS, Example, load_jsonl_task, load_task, sample_task, to_demo
from davout.prompts import build_prompt
from davout.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, parse_question

CHOICE_Q = {"type": "choice", "instructions": "Pick a team.", "criteria": {"billing": "money", "tech": None}}
SCORE_Q = {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}
NOUL_Q = {"type": "noul", "instructions": "Is it spam?"}


GOOD = {
    "choice": {"id": "t0", "state": "ok", "question": CHOICE_Q, "label": "tech"},
    "score": {"id": "t0", "state": "ok", "question": SCORE_Q, "label": 0},
    "noul": {"id": "t0", "state": "ok", "question": NOUL_Q, "label": True},
}


def write_jsonl(path: Path, rows: list[Any]) -> Path:
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


def choice_rows(n: int) -> list[dict]:
    return [
        {"id": f"t{i}", "state": f"ticket {i}", "question": CHOICE_Q, "label": ("billing", "tech")[i % 2]}
        for i in range(n)
    ]


def ids(examples: list[Example]) -> list[str]:
    return [e.id for e in examples]


# -- JSONL ------------------------------------------------------------------------------


def test_jsonl_label_mapping_per_type(tmp_path: Path) -> None:
    choice = load_jsonl_task(write_jsonl(tmp_path / "c.jsonl", choice_rows(40)), n_calib=10, n_test=10, n_shots=4)
    assert choice.name == "c" and choice.kind == "choice"
    for ex in choice.shots + choice.calib + choice.test:
        assert isinstance(ex.question, ChoiceQuestion)
        assert ex.label == int(ex.id[1:]) % 2  # "billing" -> 0, "tech" -> 1

    rows = [{"state": {"n": i}, "question": SCORE_Q, "label": i % 3} for i in range(30)]
    score = load_jsonl_task(write_jsonl(tmp_path / "s.jsonl", rows), name="urgency", n_calib=5, n_test=5, n_shots=3)
    assert score.name == "urgency" and score.kind == "score"
    for ex in score.calib + score.test:
        assert isinstance(ex.question, ScoreQuestion) and ex.label == ex.state["n"] % 3
        assert ex.id == f"urgency/{ex.state['n'] + 1}"  # default id: name/line number

    rows = [{"state": f"m{i}", "question": NOUL_Q, "label": i % 3 == 0} for i in range(30)]
    noul = load_jsonl_task(write_jsonl(tmp_path / "n.jsonl", rows), n_calib=5, n_test=5, n_shots=2)
    assert noul.kind == "noul"
    for ex in noul.calib + noul.test:
        assert isinstance(ex.question, NoulQuestion)
        assert ex.label == int(int(ex.state[1:]) % 3 == 0)  # true -> 1, false -> 0


def test_jsonl_splits_are_deterministic_and_disjoint(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "c.jsonl", choice_rows(100))
    a = load_jsonl_task(path, n_calib=30, n_test=40, n_shots=6, seed=3)
    b = load_jsonl_task(path, n_calib=30, n_test=40, n_shots=6, seed=3)
    assert (ids(a.shots), ids(a.calib), ids(a.test)) == (ids(b.shots), ids(b.calib), ids(b.test))
    assert (len(a.shots), len(a.calib), len(a.test)) == (6, 30, 40)
    everything = ids(a.shots) + ids(a.calib) + ids(a.test)
    assert len(set(everything)) == len(everything)

    other = load_jsonl_task(path, n_calib=30, n_test=40, n_shots=6, seed=4)
    assert ids(other.test) != ids(a.test)
    # the test set does not depend on how many calibration rows or shots are asked for
    fewer = load_jsonl_task(path, n_calib=10, n_test=40, n_shots=2, seed=3)
    assert ids(fewer.test) == ids(a.test) and ids(fewer.calib) == ids(a.calib)[:10]
    assert ids(fewer.shots) == ids(a.shots)[:2]


def test_shots_cover_the_labels(tmp_path: Path) -> None:
    rows = [{"state": f"m{i}", "question": NOUL_Q, "label": i % 10 == 0} for i in range(200)]
    task = load_jsonl_task(write_jsonl(tmp_path / "n.jsonl", rows), n_calib=20, n_test=20, n_shots=4)
    assert [e.label for e in task.shots].count(1) == 2  # 10% positives overall, half of the shots


def test_small_file_is_split_proportionally(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    task = load_jsonl_task(write_jsonl(tmp_path / "c.jsonl", choice_rows(31)), n_calib=300, n_test=300, n_shots=20)
    assert (len(task.shots), len(task.calib), len(task.test)) == (1, 15, 15)
    assert len(set(ids(task.shots) + ids(task.calib) + ids(task.test))) == 31
    assert "only 31 rows" in capsys.readouterr().err
    with pytest.raises(ValueError, match="too few"):
        load_jsonl_task(write_jsonl(tmp_path / "tiny.jsonl", choice_rows(1)), n_calib=5, n_test=5, n_shots=0)


@pytest.mark.parametrize(
    "bad, line, message",
    [
        ("{not json", 2, "invalid JSON"),
        ([1, 2], 2, "must be a JSON object"),
        ({"state": "x", "question": CHOICE_Q}, 2, "label is required"),
        ({"state": "x", "question": CHOICE_Q, "label": "sales"}, 2, "option names"),
        ({"state": "x", "question": CHOICE_Q, "label": 0}, 2, "option names"),
        ({"state": "x", "question": SCORE_Q, "label": 3}, 2, "level index 0..2"),
        ({"state": "x", "question": SCORE_Q, "label": True}, 2, "level index"),
        ({"state": "x", "question": NOUL_Q, "label": "yes"}, 2, "true or false"),
        ({"state": 5, "question": CHOICE_Q, "label": "tech"}, 2, "state must be"),
        ({"state": "x", "question": {"type": "choice", "instructions": "q", "criteria": {"a": None}},
          "label": "a"}, 2, "question.criteria: criteria must have 2..255 options"),
        ({"state": "x", "question": CHOICE_Q, "label": "tech", "extra": 1}, 2, "unknown key"),
        ({"id": "t0", "state": "x", "question": CHOICE_Q, "label": "tech"}, 2, "duplicate id 't0'"),
    ],
)
def test_jsonl_bad_rows_name_the_line(tmp_path: Path, bad: Any, line: int, message: str) -> None:
    # line 1 is a valid row of the same question type, so the error must point at line 2
    question = bad.get("question") if isinstance(bad, dict) else None
    qtype = question.get("type", "choice") if isinstance(question, dict) else "choice"
    path = write_jsonl(tmp_path / "bad.jsonl", [GOOD[qtype], bad])
    with pytest.raises(ValueError) as err:
        load_jsonl_task(path)
    assert str(err.value).startswith(f"{path}:{line}: ") and message in str(err.value)


def test_jsonl_mixed_types_and_empty(tmp_path: Path) -> None:
    rows = [choice_rows(1)[0], {"state": "x", "question": NOUL_Q, "label": True}]
    path = write_jsonl(tmp_path / "mixed.jsonl", rows)
    with pytest.raises(ValueError, match=rf"{path}:2: question type 'noul' differs"):
        load_jsonl_task(path)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    with pytest.raises(ValueError, match="no rows"):
        load_jsonl_task(empty)


# -- demos ------------------------------------------------------------------------------


def test_to_demo_maps_labels_to_prompt_answers() -> None:
    noul = NoulQuestion("Is it spam?", None)
    assert to_demo(Example("a", "win cash", noul, 1)).answer == 0  # true -> "Yes" (option A)
    assert to_demo(Example("b", "see you at 5", noul, 0)).answer == 1  # false -> "No" (option B)
    choice = ChoiceQuestion("Pick.", {"x": None, "y": None, "z": None})
    demo = to_demo(Example("c", "state", choice, 2))
    assert (demo.state, demo.question, demo.answer) == ("state", choice, 2)
    prompt = build_prompt("live", noul, [to_demo(Example("a", "win cash", noul, 1))])
    assert prompt.prefix.endswith("A. Yes\nB. No\nAnswer: A\n\n")


# -- registry ---------------------------------------------------------------------------

# Rows shaped like the dataset schemas verified from the Hub metadata.
SAMPLE_ROWS: dict[str, list[tuple[dict, Any, int]]] = {
    "boolq": [
        ({"question": "is the sky blue", "answer": True, "passage": "The sky is blue."},
         {"passage": "The sky is blue.", "question": "is the sky blue"}, 1),
        ({"question": "do fish fly", "answer": False, "passage": "Fish swim."},
         {"passage": "Fish swim.", "question": "do fish fly"}, 0),
    ],
    "sms_spam": [
        ({"sms": "WINNER!! Claim your prize now\n", "label": 1}, "WINNER!! Claim your prize now", 1),
        ({"sms": "See you at five\n", "label": 0}, "See you at five", 0),
    ],
    "sst2": [
        ({"idx": 0, "sentence": "a charming and often affecting journey ", "label": 1},
         "a charming and often affecting journey", 1),
        ({"idx": 1, "sentence": "unflinchingly bleak and desperate ", "label": 0},
         "unflinchingly bleak and desperate", 0),
    ],
    "ag_news": [
        ({"text": "Stocks rallied on Wall Street.", "label": 2}, "Stocks rallied on Wall Street.", 2),
        ({"text": "New chip doubles laptop battery life.", "label": 3}, "New chip doubles laptop battery life.", 3),
    ],
    "banking77": [
        ({"text": "I am still waiting on my card?", "label": 11}, "I am still waiting on my card?", 11),
        ({"text": "How do I change my PIN?", "label": 21}, "How do I change my PIN?", 21),
    ],
    "yelp": [
        ({"label": 4, "text": "Best tacos in town. " * 200}, ("Best tacos in town. " * 200).strip()[:2000], 4),
        ({"label": 0, "text": "Never again."}, "Never again.", 0),
    ],
}  # fmt: skip


def test_registry_has_the_target_tasks() -> None:
    assert list(TASKS) == ["boolq", "sms_spam", "sst2", "ag_news", "banking77", "yelp"]
    assert {name: spec.kind for name, spec in TASKS.items()} == {
        "boolq": "noul", "sms_spam": "noul", "sst2": "noul",
        "ag_news": "choice", "banking77": "choice", "yelp": "score",
    }  # fmt: skip
    assert TASKS["banking77"].dataset == "legacy-datasets/banking77"  # PolyAI/banking77 is script-only


@pytest.mark.parametrize("name", list(TASKS))
def test_registry_row_mapping(name: str) -> None:
    spec = TASKS[name]
    for row, state, label in SAMPLE_ROWS[name]:
        assert set(spec.columns) <= set(row)
        assert spec.to_example(row) == (state, label)
    # the shared question is a valid schema question
    q = spec.question
    assert parse_question(name, {"type": q.type, "instructions": q.instructions, "criteria": q.criteria}) == q


def test_registry_questions() -> None:
    assert TASKS["boolq"].question.instructions == "According to the passage, is the answer to `question` yes?"
    assert TASKS["sms_spam"].question.instructions == "The message is spam."
    assert TASKS["sst2"].question.instructions == "The review expresses a positive sentiment."
    ag = TASKS["ag_news"].question
    assert list(ag.criteria) == ["world", "sports", "business", "sci_tech"] and all(ag.criteria.values())
    assert len(TASKS["ag_news"].label_names) == 4
    bank = TASKS["banking77"].question
    assert len(bank.criteria) == 77 == len(set(bank.criteria)) and set(bank.criteria.values()) == {None}
    assert list(bank.criteria)[11] == "card arrival" and not any("_" in n for n in bank.criteria)
    yelp = TASKS["yelp"].question
    assert len(yelp.criteria) == 5
    assert yelp.criteria[0] == "Terrible experience; would warn others away"
    assert yelp.criteria[4] == "Excellent; enthusiastic recommendation"
    assert len(TASKS["yelp"].to_example({"label": 2, "text": "z" * 5000})[0]) == 2000


def fake_splits(name: str, n: int) -> dict[str, list[dict]]:
    """Every split the spec uses, filled with `n` rows cycling through the sample rows and labels."""
    spec = TASKS[name]
    n_labels = 2 if spec.kind == "noul" else len(spec.question.criteria)
    out = {}
    for split in {spec.shots_split, spec.calib_split, spec.test_split}:
        rows = []
        for i in range(n):
            row = dict(SAMPLE_ROWS[name][0][0])
            row[spec.label_column] = bool(i % 2) if name == "boolq" else i % n_labels
            rows.append(row)
        out[split] = rows
    return out


@pytest.mark.parametrize("name", list(TASKS))
def test_sample_task_uses_the_right_splits(name: str) -> None:
    spec = TASKS[name]
    splits = fake_splits(name, 400)
    task = sample_task(spec, splits, n_calib=50, n_test=60, n_shots=8, seed=1)
    again = sample_task(spec, splits, n_calib=50, n_test=60, n_shots=8, seed=1)
    assert (ids(task.shots), ids(task.calib), ids(task.test)) == (ids(again.shots), ids(again.calib), ids(again.test))
    assert (task.name, task.kind) == (name, spec.kind)
    assert (len(task.shots), len(task.calib), len(task.test)) == (8, 50, 60)
    for part, split in ((task.shots, spec.shots_split), (task.calib, spec.calib_split), (task.test, spec.test_split)):
        assert all(e.id.startswith(f"{name}/{split}/") and e.question is spec.question for e in part)
    everything = ids(task.shots) + ids(task.calib) + ids(task.test)
    assert len(set(everything)) == len(everything)  # disjoint even when roles share a split
    n_labels = 2 if spec.kind == "noul" else len(spec.question.criteria)
    assert len({e.label for e in task.shots}) == min(8, n_labels)


def test_load_task_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="unknown task 'nope'; available: boolq"):
        load_task("nope")
    monkeypatch.setitem(sys.modules, "datasets", None)  # makes `import datasets` fail
    with pytest.raises(RuntimeError, match="uv sync --extra eval"):
        load_task("boolq")


def test_load_task_checks_the_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    class Feature:
        names = ["spam", "ham"]  # swapped on purpose

    class FakeDataset(list):
        column_names = ["sms", "label"]
        features = {"label": Feature()}

    class FakeDatasets:
        @staticmethod
        def load_dataset(dataset: str, config: str, split: str) -> FakeDataset:
            assert (dataset, config, split) == ("ucirvine/sms_spam", "plain_text", "train")
            return FakeDataset({"sms": f"message {i}", "label": i % 2} for i in range(100))

    monkeypatch.setitem(sys.modules, "datasets", FakeDatasets)
    with pytest.raises(RuntimeError, match="label names changed"):
        load_task("sms_spam", n_calib=10, n_test=10)
    Feature.names = ["ham", "spam"]
    task = load_task("sms_spam", n_calib=10, n_test=10, n_shots=4)
    assert (len(task.shots), len(task.calib), len(task.test)) == (4, 10, 10)
    assert all(e.label == int(e.state.split()[1]) % 2 for e in task.test)
    FakeDataset.column_names = ["text", "label"]
    with pytest.raises(RuntimeError, match="no column"):
        load_task("sms_spam")


def test_cli_bench_tasks(capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["bench", "tasks"]) == 0
    out = capsys.readouterr().out
    for name, spec in TASKS.items():
        assert name in out and spec.dataset in out
