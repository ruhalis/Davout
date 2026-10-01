import math

import numpy as np
import pytest

from davout.backends.base import Prompt, Readout
from davout.calibrate import softmax
from davout.prompts import GENERIC_DEMOS, Demo, build_prompt
from davout.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from davout.scorer import LetterScorer, RawScore, _as_candidate, _fit_demo


class FakeBackend:
    """Scores by keyword: an option's logit is how often its first word occurs in the state."""

    name = "fake"

    def __init__(self, max_labels: int = 26) -> None:
        self.max_labels = max_labels
        self.calls: list[list[Prompt]] = []

    def info(self) -> dict:
        return {"name": self.name}

    @staticmethod
    def _key(option_text: str) -> str:
        return option_text.split(":")[0].split()[0].lower()

    def _one(self, p: Prompt) -> Readout:
        words = p.state.lower().replace(".", " ").split()
        lines = p.suffix.splitlines()
        proposed = [ln for ln in lines if ln.startswith("Proposed answer: ")]
        if proposed:
            logits = [float(words.count(self._key(proposed[0].removeprefix("Proposed answer: ")))), 0.0]
        else:
            options = [ln[3:] for ln in lines if len(ln) > 2 and ln[1:3] == ". "]
            logits = [float(words.count(self._key(o))) for o in options]
        assert len(logits) == p.n_labels
        return Readout(
            logits=logits,
            prompt_tokens=len((p.prefix + p.state + p.suffix).split()),
            truncated="TRUNC" in p.state,
            cycle_logits=[[0.0] * len(logits), logits],
        )

    def read(self, prompts):
        self.calls.append(list(prompts))
        return [self._one(p) for p in prompts]


NAMES = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
BIG = ChoiceQuestion("Which word?", {n: f"the word {n}" for n in NAMES})
QUESTIONS = [
    ChoiceQuestion("Which team?", {"billing": "payments", "tech": None, "sales": "upgrades"}),
    ScoreQuestion("How urgent?", ["low", "medium", "high", "critical"]),
    NoulQuestion("Is a refund requested?", {"true": "yes it is", "false": "no it is not"}),
    ChoiceQuestion("Sentiment?", {"positive": None, "negative": None}),
]


def test_mixed_types_single_read_in_order():
    backend = FakeBackend()
    out = LetterScorer(backend, shots=0).score("Tech tech issue, high priority. Negative.", QUESTIONS)
    assert len(backend.calls) == 1
    prompts = backend.calls[0]
    assert [p.n_labels for p in prompts] == [3, 4, 2, 2]
    assert ["Which team?" in prompts[0].suffix, "How urgent?" in prompts[1].suffix] == [True, True]
    assert "Is a refund requested?" in prompts[2].suffix and "Sentiment?" in prompts[3].suffix
    assert all(isinstance(r, RawScore) and r.stage == "letter" and not r.truncated for r in out)
    assert out[0].logits == [0.0, 2.0, 0.0]
    assert out[1].logits == [0.0, 0.0, 1.0, 0.0]
    assert out[3].logits == [0.0, 1.0]
    assert out[1].cycle_logits == [[0.0] * 4, [0.0, 0.0, 1.0, 0.0]]
    assert [r.prompt_tokens for r in out] == [len((p.prefix + p.state + p.suffix).split()) for p in prompts]


def test_noul_logits_are_true_then_false():
    backend = FakeBackend()
    q = NoulQuestion("Is a refund requested?", None)
    (yes,) = LetterScorer(backend, shots=0).score("yes yes no", [q])
    (no,) = LetterScorer(backend, shots=0).score("no", [q])
    assert yes.logits == [2.0, 1.0]
    assert no.logits == [0.0, 1.0]
    assert "A. Yes\nB. No" in backend.calls[0][0].suffix


def test_shots_zero_gives_empty_prefix():
    backend = FakeBackend(max_labels=4)
    LetterScorer(backend, shots=0).score("alpha", QUESTIONS[:3] + [BIG])
    assert all(p.prefix == "" for call in backend.calls for p in call)


def test_default_is_zero_shot():
    backend = FakeBackend()
    scorer = LetterScorer(backend)
    assert scorer.info()["shots"] == 0 and scorer.shortlist == 10
    scorer.score("state", QUESTIONS[:3])
    assert all(p.prefix == "" for p in backend.calls[0])


def test_shots_use_generic_demos_per_type():
    backend = FakeBackend()
    scorer = LetterScorer(backend, shots=3)
    assert scorer.shots == 3 and scorer.shortlist == 10
    scorer.score("state", QUESTIONS[:3])
    for p, kind in zip(backend.calls[0], ["choice", "score", "noul"]):
        assert p.prefix == build_prompt("x", QUESTIONS[0], GENERIC_DEMOS[kind][:3]).prefix
        assert p.prefix.count("\nAnswer: ") == 3
        assert p.state == "state"


def test_demos_override_replaces_generic():
    backend = FakeBackend()
    mine = [Demo("my demo state", QUESTIONS[1], 2)]
    LetterScorer(backend, shots=3).score("state", QUESTIONS[:2], demos={1: mine})
    generic, overridden = backend.calls[0]
    assert generic.prefix == build_prompt("x", QUESTIONS[0], GENERIC_DEMOS["choice"][:3]).prefix
    assert overridden.prefix == build_prompt("x", QUESTIONS[1], mine).prefix
    assert overridden.prefix.startswith("my demo state\n\nQuestion: How urgent?")
    assert overridden.prefix.count("\nAnswer: ") == 1


def test_two_stage_probabilities():
    backend = FakeBackend(max_labels=4)
    state = "charlie charlie charlie foxtrot foxtrot alpha"
    (r,) = LetterScorer(backend, shots=0, shortlist=3).score(state, [BIG])
    stage1, stage2 = backend.calls
    assert len(stage1) == len(NAMES) and all(p.n_labels == 2 for p in stage1)
    assert [f"Proposed answer: {n}: the word {n}\n" in p.suffix for p, n in zip(stage1, NAMES)] == [True] * 8
    assert all("Question: Which word?\n" in p.suffix for p in stage1)
    assert len(stage2) == 1 and stage2[0].n_labels == 3
    assert stage2[0].suffix == (
        "\n\nQuestion: Which word?\n"
        "A. alpha: the word alpha\n"
        "B. charlie: the word charlie\n"
        "C. foxtrot: the word foxtrot\n"
        "Answer:"
    )

    margins = np.array([1.0, 0, 3, 0, 0, 2, 0, 0])
    p1 = softmax(margins)
    keep = [0, 2, 5]
    expected = p1.copy()
    expected[keep] = p1[keep].sum() * softmax([1.0, 3.0, 2.0])
    probs = softmax(r.logits)
    assert probs.sum() == pytest.approx(1.0)
    assert np.exp(r.logits).sum() == pytest.approx(1.0)  # logits are log-probabilities
    assert np.allclose(probs, expected, atol=1e-12)
    assert int(np.argmax(probs)) == 2
    assert r.stage == "two_stage" and r.cycle_logits is None and not r.truncated
    assert r.prompt_tokens == sum(
        len((p.prefix + p.state + p.suffix).split()) for p in stage1 + stage2
    )


def test_two_stage_shortlist_ties_keep_earliest_and_cap():
    backend = FakeBackend(max_labels=4)
    LetterScorer(backend, shots=0, shortlist=10).score("delta", [BIG])
    stage2 = backend.calls[1][0]
    # shortlist is capped by max_labels; ties go to the earliest options, kept in criteria order
    assert stage2.n_labels == 4
    assert [ln[3:].split(":")[0] for ln in stage2.suffix.splitlines() if ln[1:3] == ". "] == [
        "alpha",
        "bravo",
        "charlie",
        "delta",
    ]


def test_two_stage_mixed_with_single_pass_uses_two_reads():
    backend = FakeBackend(max_labels=4)
    out = LetterScorer(backend, shots=0, shortlist=2).score(
        "golf golf tech TRUNC", [QUESTIONS[0], BIG, QUESTIONS[3], BIG]
    )
    assert len(backend.calls) == 2
    assert [p.n_labels for p in backend.calls[0]] == [3] + [2] * 8 + [2] + [2] * 8
    assert [p.n_labels for p in backend.calls[1]] == [2, 2]
    assert [r.stage for r in out] == ["letter", "two_stage", "letter", "two_stage"]
    assert all(r.truncated for r in out)
    assert out[1].logits == out[3].logits and len(out[1].logits) == 8
    assert max(range(8), key=lambda i: out[1].logits[i]) == NAMES.index("golf")
    assert all(math.isfinite(x) for x in out[1].logits)


def test_two_stage_uses_candidate_then_choice_demos():
    backend = FakeBackend(max_labels=4)
    LetterScorer(backend, shots=2, shortlist=3).score("alpha", [BIG])
    stage1, stage2 = backend.calls
    assert stage1[0].prefix == build_prompt("x", QUESTIONS[0], GENERIC_DEMOS["candidate"][:2]).prefix
    assert stage1[0].prefix.count("Proposed answer: ") == 2
    assert stage2[0].prefix == build_prompt("x", QUESTIONS[0], GENERIC_DEMOS["choice"][:2]).prefix


def test_two_stage_demo_override_is_adapted():
    backend = FakeBackend(max_labels=4)
    mine = [Demo("demo golf", BIG, 6), Demo("demo bravo", BIG, 1)]
    LetterScorer(backend, shots=3, shortlist=3).score("alpha", [BIG], demos={0: mine})
    stage1, stage2 = backend.calls
    # stage 1: gold proposed for the first demo (Yes), a wrong option for the second (No)
    assert "Proposed answer: golf: the word golf\n" in stage1[0].prefix
    assert "Proposed answer: charlie: the word charlie\n" in stage1[0].prefix
    assert [ln for ln in stage1[0].prefix.splitlines() if ln.startswith("Answer:")] == ["Answer: A", "Answer: B"]
    # stage 2: demos are cut down to the shortlist size and keep their gold option
    blocks = stage2[0].prefix.strip().split("\n\n")
    assert blocks[0] == "demo golf" and blocks[2] == "demo bravo"
    assert blocks[1].splitlines() == [
        "Question: Which word?",
        "A. foxtrot: the word foxtrot",
        "B. golf: the word golf",
        "C. hotel: the word hotel",
        "Answer: B",
    ]
    for block, gold in ((blocks[1], "golf"), (blocks[3], "bravo")):
        lines = block.splitlines()
        options = [ln for ln in lines if ln[1:3] == ". "]
        assert len(options) == 3
        letter = lines[-1].removeprefix("Answer: ")
        assert [o for o in options if o.startswith(f"{letter}. ")] == [f"{letter}. {gold}: the word {gold}"]


def test_fit_demo_window():
    q = ChoiceQuestion("q", {f"o{i}": None for i in range(10)})
    for gold in (0, 4, 9):
        positions = set()
        for j in range(4):
            d = _fit_demo(Demo("s", q, gold), j, 4)
            names = list(d.question.criteria)
            assert len(names) == 4 and names[d.answer] == f"o{gold}"
            assert names == [f"o{i}" for i in range(int(names[0][1:]), int(names[0][1:]) + 4)]
            positions.add(d.answer)
        assert len(positions) > 1 or gold in (0, 9)
    small = Demo("s", QUESTIONS[0], 1)
    assert _fit_demo(small, 0, 4) is small
    score = Demo("s", QUESTIONS[1], 1)
    assert _fit_demo(score, 0, 2) is score


def test_as_candidate():
    d = Demo("s", QUESTIONS[0], 2)
    yes, no = _as_candidate(d, 0), _as_candidate(d, 1)
    assert (yes.candidate, yes.answer) == ("sales", 0)
    assert (no.candidate, no.answer) == ("billing", 1)
    ready = Demo("s", QUESTIONS[0], 1, candidate="tech")
    assert _as_candidate(ready, 0) is ready


def test_bad_arguments_and_backend_mismatch():
    with pytest.raises(ValueError):
        LetterScorer(FakeBackend(), shots=-1)
    with pytest.raises(ValueError):
        LetterScorer(FakeBackend(), shortlist=1)

    class Short(FakeBackend):
        def read(self, prompts):
            return super().read(prompts)[:-1]

    with pytest.raises(RuntimeError):
        LetterScorer(Short(), shots=0).score("s", QUESTIONS)
    assert LetterScorer(FakeBackend()).score("s", []) == []


def test_info():
    scorer = LetterScorer(FakeBackend(), shots=2, shortlist=7)
    assert scorer.info() == {"name": "fake", "scorer": "letter", "shots": 2, "shortlist": 7}
