"""Tests for the teacher scorer's prompt builder and option-order maths (no model)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from davout import teacher
from davout.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

CHOICE = ChoiceQuestion("Which team should handle this?", {"billing": "payments", "technical": None, "sales": None})
SCORE = ScoreQuestion("How happy is the customer?", ["unhappy", "neutral", "happy"])


def test_messages_hold_the_question_block_without_the_answer_cue() -> None:
    msgs = teacher.messages("I was charged twice.", CHOICE)
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert user.startswith("I was charged twice.\n\nQuestion: Which team should handle this?\nA. billing: payments\nB. technical\nC. sales")
    assert "Answer:" not in user and user.endswith(teacher.REPLY)
    assert "A. unhappy\nB. neutral\nC. happy" in teacher.messages("ok", SCORE)[1]["content"]
    assert "A. Yes\nB. No" in teacher.messages("ok", NoulQuestion("Is it urgent?", None))[1]["content"]


def test_reorder_and_mapping_back_are_inverse() -> None:
    perm = [2, 0, 1]
    q = teacher.reorder(CHOICE, perm)
    assert list(q.criteria) == ["sales", "billing", "technical"] and q.criteria["billing"] == "payments"
    assert teacher.reorder(SCORE, perm).criteria == ["happy", "unhappy", "neutral"]
    # the model puts 0.7 on letter A of the shuffled prompt, which is "sales" (given index 2)
    assert teacher.to_given_order([0.7, 0.2, 0.1], perm) == [0.2, 0.1, 0.7]
    with pytest.raises(ValueError):
        teacher.reorder(CHOICE, [0, 0, 1])


def test_orders_are_seeded_and_differ_from_the_given_order() -> None:
    a, b = teacher.orders(CHOICE, "row-1"), teacher.orders(CHOICE, "row-1")
    assert a == b and a[0] == [0, 1, 2] and a[1] != [0, 1, 2] and sorted(a[1]) == [0, 1, 2]
    assert teacher.orders(NoulQuestion("x?", None), "row-1") == [[0, 1]]
    assert len({tuple(teacher.orders(CHOICE, f"row-{i}")[1]) for i in range(40)}) > 1


def test_combine_averages_and_reports_agreement() -> None:
    mean, agree = teacher.combine([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    assert mean == pytest.approx([0.4, 0.5, 0.1]) and agree is False
    assert teacher.combine([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]])[1] is True
    assert teacher.softmax([0.0, 0.0]) == pytest.approx([0.5, 0.5])


class _PositionBackend:
    """Always prefers letter A, whatever it holds: a model with pure position bias."""

    def read(self, prompts, n_labels):
        self.prompts = prompts
        return [[2.0] + [0.0] * (n - 1) for n in n_labels]


class _ContentBackend:
    """Prefers the option named "technical", wherever it stands."""

    def read(self, prompts, n_labels):
        out = []
        for p, n in zip(prompts, n_labels):
            lines = [line for line in p[1]["content"].splitlines() if line[1:3] == ". "]
            out.append([3.0 if "technical" in line else 0.0 for line in lines[:n]])
        return out


def test_score_cancels_position_bias_and_keeps_content(tmp_path: Path) -> None:
    decisions = [(f"d{i}", "state", CHOICE) for i in range(30)]
    biased = teacher.score(_PositionBackend(), decisions)
    assert not all(r["agree"] for r in biased)  # letter A is a different option in the shuffled order
    assert all(sum(r["probs"]) == pytest.approx(1.0) and len(r["per_order"]) == 2 for r in biased)
    content = teacher.score(_ContentBackend(), decisions)
    assert all(r["agree"] and max(range(3), key=r["probs"].__getitem__) == 1 for r in content)

    rows = [{"id": f"d{i}", "state": "s", "question": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "technical": None}}, "label": 1}
            for i in range(5)]  # fmt: skip
    (tmp_path / "in.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = teacher.score_file(_ContentBackend(), tmp_path / "in.jsonl", tmp_path / "out.jsonl")
    assert stats["decisions"] == 5 and stats["forward_passes"] == 10
    out = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [r["id"] for r in out] == [f"d{i}" for i in range(5)] and all(r["label"] == 1 and r["probs"][1] > 0.9 for r in out)
    assert teacher.score_file(_ContentBackend(), tmp_path / "in.jsonl", tmp_path / "out.jsonl")["decisions"] == 0  # resumes
